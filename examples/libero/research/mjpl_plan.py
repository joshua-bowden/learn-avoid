"""Plan between exact demo keyframe configs with mjpl (CBiRRT), obstacle-only collision.

Uses the raw LIBERO model (no stripping): the Panda arm joints are the first
joints in the model, so mjpl's joint sampler indexes them correctly even though
the scene contains free-joint objects. Collision is restricted to robot-vs-obstacle
via a custom mjpl Constraint, so we ignore all other objects (per the task spec).
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

import mjpl
from mjpl.constraint.constraint_interface import Constraint

from common import (
    EF_BODY_NAME, LIBERO_DUMMY_ACTION, demo_eef_goal, demo_hdf5_path,
    get_obstacle_geom_ids, get_robot_geom_ids, load_episode_from_hdf5, make_env,
    obstacle_bddl_path, obstacle_pose_for_episode, reset_env_to_demo,
    sync_exec_env_from_demo, task_bddl_path,
)
from keyframes import extract_keyframes
from planner import _collision_geoms

ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
ARM_IDX = list(range(7))
EE_SITE = "gripper0_grip_site"


class ObstacleClearanceConstraint(Constraint):
    """mjpl constraint: arm-joint limits + min clearance to obstacle geoms only."""

    def __init__(self, model, robot_geoms, obstacle_geoms, clearance, arm_idx, lo, hi):
        self.model = model
        self.data = mujoco.MjData(model)
        self.rg = list(robot_geoms)
        self.og = list(obstacle_geoms)
        self.clearance = clearance
        self.arm_idx = arm_idx
        self.lo = lo
        self.hi = hi
        self._fromto = np.zeros(6)

    def _clear(self, q: np.ndarray) -> bool:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        for r in self.rg:
            for o in self.og:
                if mujoco.mj_geomDistance(self.model, self.data, r, o, 5.0, self._fromto) < self.clearance:
                    return False
        return True

    def valid_config(self, q: np.ndarray) -> bool:
        a = q[self.arm_idx]
        if np.any(a < self.lo) or np.any(a > self.hi):
            return False
        return self._clear(q)

    def apply(self, q_old, q):
        """Validate q AND the segment q_old->q (dense edge collision check)."""
        if not self.valid_config(q):
            return None
        seg = q[self.arm_idx] - q_old[self.arm_idx]
        n = int(np.ceil(np.linalg.norm(seg) / 0.02))
        for k in range(1, n):
            qi = q_old.copy()
            qi[self.arm_idx] = q_old[self.arm_idx] + (k / n) * seg
            if not self._clear(qi):
                return None
        return q


def demo_arm_qpos(ref_env, model_xml, state) -> np.ndarray:
    reset_env_to_demo(ref_env, model_xml, state)
    return ref_env.sim.data.qpos[:7].copy()


def main():
    clearance = 0.04
    with h5py_open() as f:
        model_xml, states, actions = load_episode_from_hdf5(f, "demo_0")
        kf = extract_keyframes(actions)

    ref_env = make_env(task_bddl_path(), resolution=128, seed=0)
    exec_env = make_env(obstacle_bddl_path(), resolution=128, seed=0)
    m = exec_env.sim.model._model

    q_pick_arm = demo_arm_qpos(ref_env, model_xml, states[kf.pick])
    q_place_arm = demo_arm_qpos(ref_env, model_xml, states[kf.place])
    init_pos, _ = demo_eef_goal(ref_env, model_xml, states[kf.initial])
    pick_pos, _ = demo_eef_goal(ref_env, model_xml, states[kf.pick])
    place_pos, _ = demo_eef_goal(ref_env, model_xml, states[kf.place])

    # Place the pillar on the init->pick end-effector segment so it blocks the
    # straight-line reach to the bowl (forcing the planner to route around it).
    mid = init_pos + 0.5 * (pick_pos - init_pos)
    ox, oy = float(mid[0]), float(mid[1])
    sync_exec_env_from_demo(ref_env, exec_env, model_xml, states[kf.initial],
                            obstacle_x=ox, obstacle_y=oy, use_obstacle=True)
    for _ in range(10):
        exec_env.step(LIBERO_DUMMY_ACTION)

    robot_geoms = _collision_geoms(m, get_robot_geom_ids(exec_env))
    obstacle_geoms = _collision_geoms(m, get_obstacle_geom_ids(exec_env))
    lo, hi = m.jnt_range[ARM_IDX, 0], m.jnt_range[ARM_IDX, 1]

    constraint = ObstacleClearanceConstraint(m, robot_geoms, obstacle_geoms, clearance, ARM_IDX, lo, hi)
    planner = mjpl.RRT(m, ARM_JOINTS, [constraint],
                       max_planning_time=15.0, epsilon=0.05, seed=0,
                       goal_biasing_probability=0.1)

    q_init = exec_env.sim.data.qpos.copy()
    q_pick = q_init.copy(); q_pick[:7] = q_pick_arm
    q_place = q_init.copy(); q_place[:7] = q_place_arm

    print("clearance         :", clearance)
    print("init arm dist      :", _dist(constraint, q_init))
    print("pick arm dist      :", _dist(constraint, q_pick), "valid:", constraint.valid_config(q_pick))
    print("place arm dist     :", _dist(constraint, q_place), "valid:", constraint.valid_config(q_place))

    for label, q_goal in [("initial->pick", q_pick), ("pick->place", q_place)]:
        q_start = q_init if label == "initial->pick" else q_pick
        t0 = time.time()
        wps = planner.plan_to_config(q_start, q_goal)
        dt = time.time() - t0
        if not wps:
            print(f"{label}: PLAN FAILED ({dt:.2f}s)")
            continue
        sm = mjpl.smooth_path(wps, [constraint], eps=planner.epsilon, seed=0, sparse=True)
        # FK of final waypoint to verify it matches the demo target
        constraint.data.qpos[:] = wps[-1]
        mujoco.mj_kinematics(m, constraint.data)
        ee = constraint.data.site(EE_SITE).xpos.copy()
        target = pick_pos if label == "initial->pick" else place_pos
        print(f"{label}: OK {dt:.2f}s, {len(wps)} wps -> {len(sm)} after shortcut; "
              f"end-EE err={np.linalg.norm(ee - target):.4f}")

    ref_env.close(); exec_env.close()


def _dist(constraint, q):
    constraint.data.qpos[:] = q
    mujoco.mj_kinematics(constraint.model, constraint.data)
    md = float("inf")
    ft = np.zeros(6)
    for r in constraint.rg:
        for o in constraint.og:
            md = min(md, mujoco.mj_geomDistance(constraint.model, constraint.data, r, o, 5.0, ft))
    return round(md, 4)


def h5py_open():
    import h5py
    return h5py.File(demo_hdf5_path(), "r")


if __name__ == "__main__":
    main()
