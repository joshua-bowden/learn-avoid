"""Batch obstacle-aware augmentation over all original demos using mjpl.

For each demo: place a static pillar obstacle (fixed for that episode, swept
left-right across episodes), plan with mjpl CBiRRT between the demo's exact
pick/place joint configs (obstacle-only clearance), execute via OSC, drop the
bowl onto the plate, and record. Episodes are capped at --max_steps (auto-fail
on timeout). Saves an HDF5 in the original LIBERO demo schema, per-episode videos
with a step counter overlay, and a CSV that updates as each episode finishes.
"""
from __future__ import annotations

import csv
import dataclasses
import logging
import pathlib
import sys
import time

import h5py
import mujoco
import numpy as np
import tyro

_LIBERO_EXAMPLES = pathlib.Path(__file__).resolve().parents[1]
if str(_LIBERO_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_LIBERO_EXAMPLES))

import mjpl  # noqa: E402

from common import (  # noqa: E402
    GRIPPER_CLOSED_ACTION, GRIPPER_OPEN_ACTION, LIBERO_DUMMY_ACTION,
    DualCameraRecorder, check_obstacle_collision, check_task_success,
    demo_eef_goal, demo_hdf5_path, get_obstacle_geom_ids, get_robot_geom_ids,
    load_episode_from_hdf5, make_env, obstacle_bddl_path, quat2axisangle,
    reset_env_to_demo, set_obstacle_pose, sorted_demo_keys,
    sync_exec_env_from_demo, task_bddl_path, TASK_NAME, TASK_SUITE,
)
from keyframes import extract_keyframes  # noqa: E402
from mjpl_execute import densify_joint_path, fk_pose, osc_track_poses  # noqa: E402
from mjpl_plan import ARM_IDX, ARM_JOINTS, ObstacleClearanceConstraint  # noqa: E402
from planner import _collision_geoms  # noqa: E402

# Obstacle sweep: fixed X/Z (keeps keyframe configs reachable), Y swept across episodes.
OBSTACLE_SWEEP_X = -0.36
OBSTACLE_SWEEP_Z = 0.98
OBSTACLE_Y_LO = 0.08
OBSTACLE_Y_HI = 0.20


@dataclasses.dataclass
class Args:
    demo_file: str = ""
    output_file: str = f"data/research/mjpl/{TASK_NAME}_obstacle_demo.hdf5"
    video_out_path: str = "data/research/mjpl/videos"
    csv_path: str = "data/research/mjpl/results.csv"
    num_episodes: int = 50
    start_episode: int = 0
    max_episodes: int = 0
    resolution: int = 256
    seed: int = 0
    num_steps_wait: int = 8
    clearance: float = 0.04
    drop_height: float = 0.10
    grasp_steps: int = 12
    release_steps: int = 10
    settle_steps: int = 8
    max_steps: int = 300


def build_planner(exec_env, clearance, seed):
    """Build the obstacle-only constraint + RRT for the env's CURRENT model.

    Must be called after every env.reset(), which creates a fresh model object.
    """
    m = exec_env.sim.model._model
    robot_geoms = _collision_geoms(m, get_robot_geom_ids(exec_env))
    obstacle_geoms = _collision_geoms(m, get_obstacle_geom_ids(exec_env))
    lo, hi = m.jnt_range[ARM_IDX, 0], m.jnt_range[ARM_IDX, 1]
    constraint = ObstacleClearanceConstraint(m, robot_geoms, obstacle_geoms,
                                             clearance, ARM_IDX, lo, hi)
    planner = mjpl.RRT(m, ARM_JOINTS, [constraint], max_planning_time=15.0,
                       epsilon=0.05, seed=seed, goal_biasing_probability=0.1)
    return m, mujoco.MjData(m), constraint, planner


def overlay_step(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Draw a small red step counter in the upper-left of each frame."""
    from PIL import Image, ImageDraw
    out = []
    for i, fr in enumerate(frames):
        img = Image.fromarray(fr)
        d = ImageDraw.Draw(img)
        d.text((2, 1), f"step {i}", fill=(255, 0, 0))
        out.append(np.asarray(img))
    return out


def main(cfg: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    demo_path = demo_hdf5_path(cfg.demo_file or None)
    out_path = pathlib.Path(cfg.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_dir = pathlib.Path(cfg.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pathlib.Path(cfg.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    ref_env = make_env(task_bddl_path(), resolution=cfg.resolution, seed=cfg.seed)
    exec_env = make_env(obstacle_bddl_path(), resolution=cfg.resolution, seed=cfg.seed)
    env = exec_env

    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["episode", "obstacle_x", "obstacle_y", "success",
                     "steps", "wall_time_s", "collision", "status"])
    csv_f.flush()

    with h5py.File(demo_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        grp = f_out.create_group("data")
        for k, v in f_in["data"].attrs.items():
            grp.attrs[k] = v

        all_keys = sorted_demo_keys(f_in)[: cfg.num_episodes]
        start = cfg.start_episode
        end = start + cfg.max_episodes if cfg.max_episodes > 0 else len(all_keys)
        shard = list(enumerate(all_keys))[start:end]  # (global_idx, key)
        n_eps = len(shard)
        n_success = 0
        total_len = 0

        for ep_idx, ep_key in shard:
            t0 = time.perf_counter()
            # Deterministic per-episode obstacle Y (independent of sharding).
            oy = float(np.random.default_rng(cfg.seed + ep_idx).uniform(OBSTACLE_Y_LO, OBSTACLE_Y_HI))
            ox, oz = OBSTACLE_SWEEP_X, OBSTACLE_SWEEP_Z

            rec = {k: [] for k in (
                "states", "actions", "gripper", "joint", "ee", "agent", "wrist", "robot")}
            cameras = DualCameraRecorder()
            ep = {"collision": False, "steps": 0}
            status = "ok"

            try:
                model_xml, states, actions = load_episode_from_hdf5(f_in, ep_key)
                kf = extract_keyframes(actions)

                def demo_arm(st):
                    reset_env_to_demo(ref_env, model_xml, st)
                    return ref_env.sim.data.qpos[:7].copy()

                q_pick_arm = demo_arm(states[kf.pick])
                q_place_arm = demo_arm(states[kf.place])
                place_pos, _ = demo_eef_goal(ref_env, model_xml, states[kf.place])

                # Reset the exec env every episode so its timestep/controller/state
                # don't leak across episodes (was causing horizon-termination crashes
                # and initial-pose drift). reset() builds a fresh model -> rebuild planner.
                exec_env.reset()
                sync_exec_env_from_demo(ref_env, exec_env, model_xml, states[kf.initial],
                                        obstacle_x=ox, obstacle_y=oy, use_obstacle=True)
                set_obstacle_pose(exec_env, x=ox, y=oy, z=oz)
                m, fk_data, constraint, planner = build_planner(exec_env, cfg.clearance, cfg.seed)
                for _ in range(cfg.num_steps_wait):
                    env.step(LIBERO_DUMMY_ACTION)

                def record_step(obs, action):
                    pos = obs["robot0_eef_pos"]
                    ori = quat2axisangle(obs["robot0_eef_quat"])
                    rec["states"].append(env.sim.get_state().flatten())
                    rec["actions"].append(np.asarray(action))
                    rec["gripper"].append(obs["robot0_gripper_qpos"])
                    rec["joint"].append(obs["robot0_joint_pos"])
                    rec["ee"].append(np.hstack((pos, ori)))
                    rec["agent"].append(obs["agentview_image"])
                    rec["wrist"].append(obs["robot0_eye_in_hand_image"])
                    rec["robot"].append(env.env.get_robot_state_vector(obs))
                    cameras.append(obs)
                    ep["steps"] += 1
                    if check_obstacle_collision(env):
                        ep["collision"] = True

                def terminated():
                    return bool(getattr(env.env, "done", False))

                def over_budget():
                    return ep["steps"] >= cfg.max_steps or terminated()

                def step_gripper(val, n):
                    a = np.array([0.0] * 6 + [val])
                    for _ in range(n):
                        if over_budget():
                            return
                        try:
                            obs, _, _, _ = env.step(a.tolist())
                        except ValueError:
                            return
                        record_step(obs, a)

                def plan_arm(q_goal_arm):
                    q0 = env.sim.data.qpos.copy()
                    qg = q0.copy(); qg[:7] = q_goal_arm
                    try:
                        wps = planner.plan_to_config(q0, qg)
                    except ValueError:
                        return None  # goal in collision with obstacle
                    if not wps:
                        return None
                    wps = mjpl.smooth_path(wps, [constraint], eps=planner.epsilon, seed=cfg.seed)
                    return [fk_pose(m, fk_data, q) for q in densify_joint_path(wps, max_step=0.08)]

                poses1 = plan_arm(q_pick_arm)
                if poses1 is None:
                    status = "plan_fail_pick"
                else:
                    osc_track_poses(env, poses1, GRIPPER_OPEN_ACTION, record_step, stop_fn=over_budget,
                                    max_substeps=3, pos_tol=0.015, final_substeps=28, final_pos_tol=0.004)
                    step_gripper(GRIPPER_CLOSED_ACTION, cfg.grasp_steps)
                    if not over_budget():
                        poses2 = plan_arm(q_place_arm)
                        if poses2 is None:
                            status = "plan_fail_place"
                        else:
                            floor_z = float(place_pos[2]) + cfg.drop_height
                            poses2 = [(np.array([p[0], p[1], max(p[2], floor_z)]), q) for p, q in poses2]
                            osc_track_poses(env, poses2, GRIPPER_CLOSED_ACTION, record_step, stop_fn=over_budget,
                                            max_substeps=3, pos_tol=0.015, final_substeps=26, final_pos_tol=0.006)
                            step_gripper(GRIPPER_OPEN_ACTION, cfg.release_steps)
                            for _ in range(cfg.settle_steps):
                                if over_budget():
                                    break
                                try:
                                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                                except ValueError:
                                    break
                                record_step(obs, LIBERO_DUMMY_ACTION)
            except Exception as e:  # one bad episode must not kill the whole run
                status = f"error:{type(e).__name__}"
                logging.exception("ep %d (%s) raised", ep_idx, ep_key)

            timed_out = ep["steps"] >= cfg.max_steps
            try:
                task_ok = bool(check_task_success(env))
            except Exception:
                task_ok = False
            success = task_ok and status == "ok" and not timed_out
            if timed_out and status == "ok":
                status = "timeout"
            n_success += int(success)
            wall = time.perf_counter() - t0

            if rec["actions"]:
                _save_episode(grp, ep_idx, rec, success, env)
                total_len += len(rec["actions"])
                _save_video(video_dir, f"ep_{ep_idx:02d}",
                            overlay_step(cameras.agent_frames), overlay_step(cameras.wrist_frames))

            writer.writerow([ep_idx, f"{ox:.3f}", f"{oy:.3f}", int(success),
                             ep["steps"], f"{wall:.1f}", int(ep["collision"]), status])
            csv_f.flush()
            logging.info("ep %d/%d: success=%s steps=%d time=%.1fs collision=%s status=%s y=%.3f",
                         ep_idx, cfg.num_episodes - 1, success, ep["steps"], wall, ep["collision"], status, oy)

        grp.attrs["num_demos"] = n_eps
        grp.attrs["total"] = total_len

    csv_f.close()
    ref_env.close(); exec_env.close()
    logging.info("DONE: %d/%d success. HDF5=%s CSV=%s", n_success, n_eps, out_path, csv_path)


def _save_episode(grp, ep_idx, rec, success, env):
    ep_grp = grp.create_group(f"demo_{ep_idx}")
    obs_grp = ep_grp.create_group("obs")
    obs_grp.create_dataset("gripper_states", data=np.stack(rec["gripper"]))
    obs_grp.create_dataset("joint_states", data=np.stack(rec["joint"]))
    ee = np.stack(rec["ee"])
    obs_grp.create_dataset("ee_states", data=ee)
    obs_grp.create_dataset("ee_pos", data=ee[:, :3])
    obs_grp.create_dataset("ee_ori", data=ee[:, 3:])
    obs_grp.create_dataset("agentview_rgb", data=np.stack(rec["agent"]))
    obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(rec["wrist"]))
    ep_grp.create_dataset("actions", data=np.stack(rec["actions"]))
    ep_grp.create_dataset("states", data=np.stack(rec["states"]))
    ep_grp.create_dataset("robot_states", data=np.stack(rec["robot"]))
    n = len(rec["actions"])
    rewards = np.zeros(n, dtype=np.uint8)
    dones = np.zeros(n, dtype=np.uint8)
    if success:
        rewards[-1] = 1
        dones[-1] = 1
    ep_grp.create_dataset("rewards", data=rewards)
    ep_grp.create_dataset("dones", data=dones)
    ep_grp.attrs["num_samples"] = n
    ep_grp.attrs["model_file"] = env.sim.model.get_xml()
    ep_grp.attrs["success"] = int(success)


def _save_video(out_dir, prefix, frames_a, frames_w, fps=20):
    import imageio
    imageio.mimwrite(out_dir / f"{prefix}_agent.mp4", frames_a, fps=fps)
    imageio.mimwrite(out_dir / f"{prefix}_wrist.mp4", frames_w, fps=fps)


if __name__ == "__main__":
    main(tyro.cli(Args))
