"""Execute an mjpl plan between exact demo keyframes, avoiding the obstacle.

Plan (CBiRRT, obstacle-only clearance) between the demo's exact pick/place JOINT
configs, then execute on the real env by OSC-tracking the forward-kinematics EEF
poses of the densified planned path. Grasp/release happen at the keyframes, so the
grip and place replicate the original demonstration. Records a dual-camera video.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

import mjpl

from augment_planned import _wxyz_to_xyzw, osc_action
from common import (
    GRIPPER_CLOSED_ACTION, GRIPPER_OPEN_ACTION, LIBERO_DUMMY_ACTION,
    DualCameraRecorder, check_obstacle_collision, check_task_success,
    demo_eef_goal, demo_hdf5_path, get_obstacle_geom_ids, get_robot_geom_ids,
    load_episode_from_hdf5, make_env, obstacle_bddl_path, reset_env_to_demo,
    set_obstacle_pose, sync_exec_env_from_demo, task_bddl_path,
)
from keyframes import extract_keyframes
from mjpl_plan import ARM_IDX, ARM_JOINTS, EE_SITE, ObstacleClearanceConstraint
from planner import _collision_geoms

import h5py
import pathlib


def fk_pose(model, data, q):
    """Forward kinematics: full qpos -> (eef_pos, eef_quat_xyzw)."""
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    pos = data.site(EE_SITE).xpos.copy()
    mat = data.site(EE_SITE).xmat.reshape(3, 3)
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(mat).as_quat()  # xyzw
    return pos, quat


def densify_joint_path(waypoints, max_step=0.04):
    """Interpolate joint waypoints to <= max_step (rad) spacing on the arm."""
    dense = [np.asarray(waypoints[0], dtype=float)]
    for i in range(len(waypoints) - 1):
        a = np.asarray(waypoints[i], dtype=float)
        b = np.asarray(waypoints[i + 1], dtype=float)
        n = max(1, int(np.ceil(np.linalg.norm(b[ARM_IDX] - a[ARM_IDX]) / max_step)))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return dense


def osc_track_poses(env, poses, gripper_val, record_fn,
                    pos_tol=0.012, ori_tol=0.08, max_substeps=4,
                    final_pos_tol=0.006, final_substeps=25, stop_fn=None):
    """OSC-track a list of (pos, quat_xyzw) poses. Returns False on obstacle contact.

    If stop_fn() returns True (e.g. step budget exhausted), tracking aborts early.
    """
    obs = env.env._get_observations()
    collided = False
    for i, (tpos, tquat) in enumerate(poses):
        is_last = i == len(poses) - 1
        tol = final_pos_tol if is_last else pos_tol
        subs = final_substeps if is_last else max_substeps
        for _ in range(subs):
            if stop_fn is not None and stop_fn():
                return not collided
            cpos = np.asarray(obs["robot0_eef_pos"])
            cquat = np.asarray(obs["robot0_eef_quat"])
            if np.linalg.norm(cpos - tpos) < tol and not is_last:
                break
            action = osc_action(cpos, cquat, tpos, tquat, gripper_val)
            try:
                obs, _, done, _ = env.step(action.tolist())
            except ValueError:  # robosuite: "executing action in terminated episode"
                return not collided
            record_fn(obs, action)
            if check_obstacle_collision(env):
                collided = True
            if done:  # task succeeded; env is now terminated
                return not collided
    return not collided


def main():
    clearance = 0.04
    with h5py.File(demo_hdf5_path(), "r") as f:
        model_xml, states, actions = load_episode_from_hdf5(f, "demo_0")
        kf = extract_keyframes(actions)

    ref_env = make_env(task_bddl_path(), resolution=256, seed=0)
    exec_env = make_env(obstacle_bddl_path(), resolution=256, seed=0)
    env = exec_env
    m = env.sim.model._model
    fk_data = mujoco.MjData(m)

    def demo_arm(st):
        reset_env_to_demo(ref_env, model_xml, st)
        return ref_env.sim.data.qpos[:7].copy()

    q_pick_arm = demo_arm(states[kf.pick])
    q_place_arm = demo_arm(states[kf.place])
    init_pos, _ = demo_eef_goal(ref_env, model_xml, states[kf.initial])
    pick_pos, pick_quat = demo_eef_goal(ref_env, model_xml, states[kf.pick])
    place_pos, place_quat = demo_eef_goal(ref_env, model_xml, states[kf.place])

    mid = init_pos + 0.5 * (pick_pos - init_pos)
    sync_exec_env_from_demo(ref_env, exec_env, model_xml, states[kf.initial],
                            obstacle_x=float(mid[0]), obstacle_y=float(mid[1]),
                            use_obstacle=True)
    for _ in range(10):
        env.step(LIBERO_DUMMY_ACTION)

    robot_geoms = _collision_geoms(m, get_robot_geom_ids(exec_env))
    obstacle_geoms = _collision_geoms(m, get_obstacle_geom_ids(exec_env))
    lo, hi = m.jnt_range[ARM_IDX, 0], m.jnt_range[ARM_IDX, 1]
    constraint = ObstacleClearanceConstraint(m, robot_geoms, obstacle_geoms,
                                             clearance, ARM_IDX, lo, hi)
    planner = mjpl.RRT(m, ARM_JOINTS, [constraint], max_planning_time=15.0,
                       epsilon=0.05, seed=0, goal_biasing_probability=0.1)

    def plan_arm(q_goal_arm):
        q0 = env.sim.data.qpos.copy()
        qg = q0.copy(); qg[:7] = q_goal_arm
        wps = planner.plan_to_config(q0, qg)
        if not wps:
            return None
        wps = mjpl.smooth_path(wps, [constraint], eps=planner.epsilon, seed=0)
        dense = densify_joint_path(wps)
        return [fk_pose(m, fk_data, q) for q in dense]

    rec = {k: [] for k in ("agent", "wrist")}
    cameras = DualCameraRecorder()
    collided = {"v": False}

    def record_step(obs, action):
        cameras.append(obs)
        if check_obstacle_collision(env):
            collided["v"] = True

    def step_gripper(val, n):
        a = np.array([0.0] * 6 + [val])
        for _ in range(n):
            obs, _, _, _ = env.step(a.tolist())
            record_step(obs, a)

    def bowl():
        return env.sim.data.get_body_xpos("akita_black_bowl_1_main").copy()
    plate = env.sim.data.get_body_xpos("plate_1_main").copy()
    eef_now = lambda: np.asarray(env.env._get_observations()["robot0_eef_pos"])

    t0 = time.time()
    print("plate xyz           :", np.round(plate, 3))
    print("bowl @settle        :", np.round(bowl(), 3))
    poses1 = plan_arm(q_pick_arm)
    if poses1 is None:
        print("initial->pick plan FAILED"); return
    if not osc_track_poses(env, poses1, GRIPPER_OPEN_ACTION, record_step):
        collided["v"] = True
    eef_pick = eef_now()
    print("bowl @pre-grasp     :", np.round(bowl(), 3), "(eef-bowl xy off:",
          np.round((eef_pick - bowl())[:2], 3), ")")

    step_gripper(GRIPPER_CLOSED_ACTION, 14)
    print("bowl @grasped       :", np.round(bowl(), 3))

    poses2 = plan_arm(q_place_arm)
    if poses2 is None:
        print("pick->place plan FAILED"); return
    # Place 10cm above the demo's (flat-on-plate) pose, then release so the bowl
    # drops onto the plate -- the success detector requires a drop, not a flat set.
    drop_height = 0.10
    floor_z = float(place_pos[2]) + drop_height
    poses2 = [(np.array([p[0], p[1], max(p[2], floor_z)]), q) for p, q in poses2]
    if not osc_track_poses(env, poses2, GRIPPER_CLOSED_ACTION, record_step):
        collided["v"] = True
    eef_place = eef_now()
    print("bowl @pre-release   :", np.round(bowl(), 3), "(eef-bowl xy off:",
          np.round((eef_place - bowl())[:2], 3), ")")

    step_gripper(GRIPPER_OPEN_ACTION, 12)
    for _ in range(20):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION); record_step(obs, LIBERO_DUMMY_ACTION)

    success = check_task_success(env)
    bowl = env.sim.data.get_body_xpos("akita_black_bowl_1_main").copy()
    print(f"\n=== demo_0 mjpl execution ({time.time()-t0:.1f}s) ===")
    print(f"  pick  EEF reached err: {np.linalg.norm(eef_pick - pick_pos):.4f}")
    raised_place = np.array([place_pos[0], place_pos[1], place_pos[2] + 0.10])
    print(f"  place EEF reached err: {np.linalg.norm(eef_place - raised_place):.4f} (vs +10cm target)")
    print(f"  bowl final xyz       : {np.round(bowl,3)}")
    print(f"  obstacle collision   : {collided['v']}")
    print(f"  task success         : {success}")

    out = pathlib.Path("data/research/mjpl"); out.mkdir(parents=True, exist_ok=True)
    cameras.save(out, "mjpl_demo_0")
    print(f"  video saved to       : {out}/mjpl_demo_0_*.mp4")
    ref_env.close(); exec_env.close()


if __name__ == "__main__":
    main()
