"""Obstacle-aware demo augmentation: joint-space RRT + OSC execution.

Pipeline per demo (general, keyframe-driven):
  1. Sync exec env (with obstacle) to the demo's initial layout.
  2. IK demo pick/place EEF poses -> goal joint configs.
  3. RRT-Connect (fast MuJoCo collision oracle) for initial->pick and pick->place.
  4. Execute each planned joint path by OSC-tracking its FK EEF waypoints,
     stepping the real controller so recorded actions/observations are dynamics-
     consistent. Grasp/release are gripper-only steps.
  5. Record states/actions/images, check success + collision, save HDF5 + video.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys
import time

import h5py
import numpy as np
import tyro

_LIBERO_EXAMPLES = pathlib.Path(__file__).resolve().parents[1]
if str(_LIBERO_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_LIBERO_EXAMPLES))

from scipy.spatial.transform import Rotation  # noqa: E402

from common import (  # noqa: E402
    GRIPPER_CLOSED_ACTION,
    GRIPPER_OPEN_ACTION,
    LIBERO_DUMMY_ACTION,
    OBSTACLE_HALF_HEIGHT,
    TASK_ID,
    TASK_NAME,
    TASK_SUITE,
    DualCameraRecorder,
    EF_BODY_NAME,
    _obstacle_object,
    check_obstacle_collision,
    check_task_success,
    demo_eef_goal,
    demo_hdf5_path,
    load_episode_from_hdf5,
    make_env,
    obstacle_bddl_path,
    obstacle_pose_for_episode,
    quat2axisangle,
    sorted_demo_keys,
    sync_exec_env_from_demo,
    task_bddl_path,
)
from keyframes import extract_keyframes  # noqa: E402
from libero.libero import benchmark  # noqa: E402

# OSC_POSE controller output scale (m, rad) per LIBERO env config.
POS_SCALE = 0.05
ORI_SCALE = 0.5


@dataclasses.dataclass
class Args:
    demo_file: str = ""
    output_file: str = ""
    num_episodes: int = 50
    max_episodes: int = 1
    resolution: int = 256
    video_out_path: str = "data/research/planned"
    seed: int = 0
    num_steps_wait: int = 10
    clearance: float = 0.06
    grasp_steps: int = 12
    release_steps: int = 10


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def osc_action(cpos, cquat_xyzw, tpos, tquat_xyzw, gripper) -> np.ndarray:
    """OSC_POSE delta action driving EEF toward (tpos, tquat).

    Orientation delta is the true axis-angle rotation from current to target
    (world frame), not an axis-angle subtraction.
    """
    dpos = np.clip((tpos - cpos) / POS_SCALE, -1.0, 1.0)
    q_err = Rotation.from_quat(tquat_xyzw) * Rotation.from_quat(cquat_xyzw).inv()
    dori = np.clip(q_err.as_rotvec() / ORI_SCALE, -1.0, 1.0)
    return np.concatenate([dpos, dori, [gripper]])


def densify_positions(waypoints, max_step=0.015):
    """Linear EEF-position interpolation to <= max_step spacing (m)."""
    dense = [np.asarray(waypoints[0], dtype=float)]
    for i in range(len(waypoints) - 1):
        a = np.asarray(waypoints[i], dtype=float)
        b = np.asarray(waypoints[i + 1], dtype=float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / max_step)))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return dense


def transit_over_waypoints(start_pos, goal_pos, transit_z):
    """Up-and-over EEF path: lift to transit height, translate, descend."""
    p_up = np.array([start_pos[0], start_pos[1], transit_z])
    p_over = np.array([goal_pos[0], goal_pos[1], transit_z])
    return densify_positions([start_pos, p_up, p_over, np.asarray(goal_pos)])


def osc_track_positions(
    env, positions, goal_quat_xyzw, gripper_val, record_fn,
    pos_tol=0.012, max_substeps=4, final_pos_tol=0.008, final_substeps=20,
) -> bool:
    """Track EEF position waypoints holding a fixed target orientation.

    Returns False if the obstacle is contacted during execution.
    """
    obs = env.env._get_observations()
    for i, tpos in enumerate(positions):
        is_last = i == len(positions) - 1
        tol = final_pos_tol if is_last else pos_tol
        subs = final_substeps if is_last else max_substeps
        for _ in range(subs):
            cpos = np.asarray(obs["robot0_eef_pos"])
            cquat = np.asarray(obs["robot0_eef_quat"])  # xyzw (robosuite)
            if np.linalg.norm(cpos - tpos) < tol:
                break
            action = osc_action(cpos, cquat, tpos, goal_quat_xyzw, gripper_val)
            obs, _, _, _ = env.step(action.tolist())
            record_fn(obs, action)
            if check_obstacle_collision(env):
                return False
    return True


def obstacle_top_z(env) -> float:
    body = _obstacle_object(env).root_body
    return float(env.sim.data.get_body_xpos(body)[2]) + OBSTACLE_HALF_HEIGHT


def augment(cfg: Args) -> None:
    np.random.seed(cfg.seed)
    demo_path = demo_hdf5_path(cfg.demo_file or None)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file missing: {demo_path}")

    out_path = pathlib.Path(
        cfg.output_file or f"data/augmented/{TASK_SUITE}/{TASK_NAME}_demo.hdf5"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(cfg.video_out_path).mkdir(parents=True, exist_ok=True)

    benchmark.get_benchmark_dict()[TASK_SUITE]().get_task(TASK_ID)
    ref_env = make_env(task_bddl_path(), resolution=cfg.resolution, seed=cfg.seed)
    exec_env = make_env(obstacle_bddl_path(), resolution=cfg.resolution, seed=cfg.seed)
    env = exec_env

    stats = {"success": 0, "collision": 0, "plan_fail": 0}

    with h5py.File(demo_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        grp = f_out.create_group("data")
        for k, v in f_in["data"].attrs.items():
            grp.attrs[k] = v

        demo_keys = sorted_demo_keys(f_in)[: cfg.num_episodes]
        if cfg.max_episodes > 0:
            demo_keys = demo_keys[: cfg.max_episodes]

        total_len = 0
        for ep_idx, ep_key in enumerate(demo_keys):
            ep_t0 = time.perf_counter()
            model_xml, states, actions = load_episode_from_hdf5(f_in, ep_key)
            kf = extract_keyframes(actions)
            obstacle_x, obstacle_y = obstacle_pose_for_episode(ep_idx, len(demo_keys))
            logging.info("Ep %s keyframes=%s obstacle=(%.3f, %.3f)",
                         ep_key, kf.as_dict(), obstacle_x, obstacle_y)

            # Goal EEF poses from the original (obstacle-free) demo.
            pick_pos, pick_quat = demo_eef_goal(ref_env, model_xml, states[kf.pick])
            place_pos, place_quat = demo_eef_goal(ref_env, model_xml, states[kf.place])

            sync_exec_env_from_demo(
                ref_env, exec_env, model_xml, states[kf.initial],
                obstacle_x=obstacle_x, obstacle_y=obstacle_y, use_obstacle=True,
            )
            for _ in range(cfg.num_steps_wait):
                env.step(LIBERO_DUMMY_ACTION)

            # Recording buffers (LIBERO demo schema).
            rec = {k: [] for k in (
                "states", "actions", "gripper", "joint", "ee", "agent", "wrist", "robot")}
            cameras = DualCameraRecorder()
            episode_collision = False

            def record_step(obs, action):
                nonlocal episode_collision
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
                if check_obstacle_collision(env):
                    episode_collision = True

            def step_gripper(gripper_val, n_steps):
                action = np.array([0.0] * 6 + [gripper_val])
                for _ in range(n_steps):
                    obs, _, _, _ = env.step(action.tolist())
                    record_step(obs, action)

            transit_z = obstacle_top_z(env) + 0.07
            logging.info("transit_z=%.3f (obstacle top=%.3f)",
                         transit_z, obstacle_top_z(env))

            def transit_to(goal_pos, goal_quat_wxyz, gripper):
                """Lift over the obstacle, translate above goal, descend."""
                start_pos = env.sim.data.get_body_xpos(EF_BODY_NAME).copy()
                tz = max(transit_z, goal_pos[2] + 0.10, start_pos[2])
                positions = transit_over_waypoints(start_pos, goal_pos, tz)
                return osc_track_positions(
                    env, positions, _wxyz_to_xyzw(goal_quat_wxyz), gripper, record_step
                )

            # --- Segment 1: initial -> over -> pick (gripper open) ---
            if not transit_to(pick_pos, pick_quat, GRIPPER_OPEN_ACTION):
                episode_collision = True

            # --- Grasp ---
            step_gripper(GRIPPER_CLOSED_ACTION, cfg.grasp_steps)

            # --- Segment 2: pick -> over -> place (gripper closed, carrying) ---
            if not transit_to(place_pos, place_quat, GRIPPER_CLOSED_ACTION):
                episode_collision = True

            # --- Release ---
            step_gripper(GRIPPER_OPEN_ACTION, cfg.release_steps)

            task_success = check_task_success(env)
            stats["success"] += int(task_success)
            stats["collision"] += int(episode_collision)

            if not rec["actions"]:
                logging.warning("Ep %s: no steps recorded, skipping", ep_key)
                continue

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
            if task_success:
                rewards[-1] = 1
                dones[-1] = 1
            ep_grp.create_dataset("rewards", data=rewards)
            ep_grp.create_dataset("dones", data=dones)
            ep_grp.attrs["num_samples"] = n
            ep_grp.attrs["model_file"] = env.sim.model.get_xml()
            ep_grp.attrs["success"] = int(task_success)
            total_len += n

            cameras.save(pathlib.Path(cfg.video_out_path), f"planned_{ep_key}")
            logging.info(
                "Ep %d: steps=%d success=%s collision=%s total=%.1fs",
                ep_idx, n, task_success, episode_collision, time.perf_counter() - ep_t0,
            )

        grp.attrs["num_demos"] = len(demo_keys)
        grp.attrs["total"] = total_len

    ref_env.close()
    exec_env.close()
    logging.info("Wrote augmented dataset: %s", out_path)
    logging.info("Stats: %s", stats)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    augment(tyro.cli(Args))
