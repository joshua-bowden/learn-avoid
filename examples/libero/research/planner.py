"""Fast joint-space motion planning for LIBERO scenes.

Two pieces, decoupled so they generalize beyond the red-box obstacle:

  * ``CollisionChecker`` - queries MuJoCo collision directly on the env's
    mjModel/mjData (``mj_kinematics`` + ``mj_geomDistance``) WITHOUT stepping the
    controller or rendering. ~0.05-1 ms/check vs ~80 ms for ``env.step``.
  * ``RRTConnect`` - bidirectional RRT in 7-DOF joint space that depends only on
    the checker interface, plus greedy shortcutting.

The checker mutates ``qpos[:7]`` of the live env data during queries, so callers
must snapshot/restore the env state around a planning episode (``save``/``restore``).
"""

from __future__ import annotations

import logging
from typing import Optional

import mujoco
import numpy as np

from common import EF_BODY_NAME, get_obstacle_geom_ids, get_robot_geom_ids, quat2axisangle

ARM_DOF = 7

# Franka Panda joint limits (rad).
JOINT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8971])


def _collision_geoms(model, geom_ids) -> list[int]:
    """Keep only geoms that participate in contacts (contype/conaffinity set)."""
    return [
        g for g in geom_ids
        if int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0
    ]


class CollisionChecker:
    """Robot-vs-obstacle distance queries on raw MuJoCo data (no stepping)."""

    def __init__(self, env, clearance: float = 0.05):
        self.env = env
        self.m = env.sim.model._model
        self.d = env.sim.data._data
        self.clearance = clearance
        self.robot_geoms = _collision_geoms(self.m, get_robot_geom_ids(env))
        self.obstacle_geoms = _collision_geoms(self.m, get_obstacle_geom_ids(env))
        self.ef_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, EF_BODY_NAME)
        self._fromto = np.zeros(6, dtype=np.float64)
        logging.info(
            "CollisionChecker: %d robot geoms x %d obstacle geoms, clearance=%.3f",
            len(self.robot_geoms), len(self.obstacle_geoms), clearance,
        )

    # --- env state management -------------------------------------------------
    def save(self) -> np.ndarray:
        return self.env.sim.get_state().flatten()

    def restore(self, state: np.ndarray) -> None:
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()

    # --- low-level kinematics --------------------------------------------------
    def _place_arm(self, q: np.ndarray) -> None:
        self.d.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self.m, self.d)

    def obstacle_distance(self, q: np.ndarray) -> float:
        """Min separation (m) between robot collision geoms and obstacle geoms."""
        if not self.obstacle_geoms:
            return float("inf")
        self._place_arm(q)
        min_d = float("inf")
        for rg in self.robot_geoms:
            for og in self.obstacle_geoms:
                d = mujoco.mj_geomDistance(self.m, self.d, rg, og, 5.0, self._fromto)
                if d < min_d:
                    min_d = d
        return min_d

    def in_collision(self, q: np.ndarray, clearance: Optional[float] = None) -> bool:
        margin = self.clearance if clearance is None else clearance
        return self.obstacle_distance(q) < margin

    def segment_free(
        self, q0: np.ndarray, q1: np.ndarray, clearance: Optional[float] = None,
        max_step: float = 0.04,
    ) -> bool:
        """Dense collision check along the straight joint-space segment."""
        margin = self.clearance if clearance is None else clearance
        n = max(2, int(np.ceil(np.linalg.norm(q1 - q0) / max_step)) + 1)
        for alpha in np.linspace(0.0, 1.0, n):
            if self.obstacle_distance((1.0 - alpha) * q0 + alpha * q1) < margin:
                return False
        return True

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics for arm config -> (eef_pos, eef_axisangle).

        Orientation is returned as axis-angle in the robosuite (xyzw) convention so
        it composes with ``obs['robot0_eef_quat']``-derived deltas during execution.
        """
        self._place_arm(q)
        pos = self.d.xpos[self.ef_body_id].copy()
        quat_wxyz = self.d.xquat[self.ef_body_id]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return pos, quat2axisangle(quat_xyzw)


class RRTConnect:
    """Bidirectional RRT in joint space against a CollisionChecker."""

    def __init__(
        self,
        checker: CollisionChecker,
        step_size: float = 0.04,
        max_iters: int = 8000,
        goal_threshold: float = 0.05,
        joint_low: np.ndarray = JOINT_LOW,
        joint_high: np.ndarray = JOINT_HIGH,
        seed: int = 0,
    ):
        self.checker = checker
        self.step_size = step_size
        self.max_iters = max_iters
        self.goal_threshold = goal_threshold
        self.lo = joint_low
        self.hi = joint_high
        self.rng = np.random.default_rng(seed)

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        delta = q_to - q_from
        dist = float(np.linalg.norm(delta))
        if dist <= self.step_size:
            return np.clip(q_to, self.lo, self.hi)
        return np.clip(q_from + self.step_size * delta / dist, self.lo, self.hi)

    @staticmethod
    def _nearest(tree: list[np.ndarray], q: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(np.asarray(tree) - q, axis=1)))

    def _extend(self, tree, parent, q_target):
        idx = self._nearest(tree, q_target)
        q_new = self._steer(tree[idx], q_target)
        if not self.checker.segment_free(tree[idx], q_new):
            return None
        tree.append(q_new)
        parent.append(idx)
        return len(tree) - 1

    def _connect(self, tree, parent, q_target):
        idx = None
        while True:
            new_idx = self._extend(tree, parent, q_target)
            if new_idx is None:
                return None
            idx = new_idx
            if np.linalg.norm(tree[idx] - q_target) < self.goal_threshold:
                return idx

    @staticmethod
    def _path_to_root(tree, parent, idx) -> list[np.ndarray]:
        path = []
        while idx != -1:
            path.append(tree[idx])
            idx = parent[idx]
        path.reverse()
        return path

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> Optional[list[np.ndarray]]:
        if self.checker.in_collision(q_start):
            logging.warning("RRT start in collision (dist=%.4f)", self.checker.obstacle_distance(q_start))
            return None
        if self.checker.in_collision(q_goal):
            logging.warning("RRT goal in collision (dist=%.4f)", self.checker.obstacle_distance(q_goal))
            return None
        if self.checker.segment_free(q_start, q_goal):
            return self.shortcut([q_start.copy(), q_goal.copy()])

        tree_a, parent_a = [q_start.copy()], [-1]
        tree_b, parent_b = [q_goal.copy()], [-1]
        start_is_a = True

        for it in range(self.max_iters):
            q_rand = np.clip(self.rng.uniform(self.lo, self.hi), self.lo, self.hi)
            a_idx = self._extend(tree_a, parent_a, q_rand)
            if a_idx is not None:
                b_idx = self._connect(tree_b, parent_b, tree_a[a_idx])
                if b_idx is not None:
                    path_a = self._path_to_root(tree_a, parent_a, a_idx)
                    path_b = self._path_to_root(tree_b, parent_b, b_idx)
                    if start_is_a:
                        full = path_a + path_b[::-1]
                    else:
                        full = path_b + path_a[::-1]
                    logging.info("RRT-Connect: %d iters, %d waypoints", it, len(full))
                    return self.shortcut(full)
            tree_a, tree_b = tree_b, tree_a
            parent_a, parent_b = parent_b, parent_a
            start_is_a = not start_is_a

        logging.warning("RRT-Connect failed after %d iters", self.max_iters)
        return None

    def shortcut(self, path: list[np.ndarray], n_iters: int = 150) -> list[np.ndarray]:
        if len(path) < 3:
            return [q.copy() for q in path]
        path = [q.copy() for q in path]
        for _ in range(n_iters):
            if len(path) < 3:
                break
            i = int(self.rng.integers(0, len(path) - 2))
            j = int(self.rng.integers(i + 2, len(path)))
            if self.checker.segment_free(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return path

    def plan_through(self, waypoints: list[np.ndarray]) -> Optional[list[np.ndarray]]:
        full: list[np.ndarray] = []
        for i in range(len(waypoints) - 1):
            seg = self.plan(waypoints[i], waypoints[i + 1])
            if seg is None:
                return None
            full.extend(seg if not full else seg[1:])
        return full


def densify_joint_path(path: list[np.ndarray], max_step: float = 0.03) -> list[np.ndarray]:
    """Interpolate a joint path so consecutive configs differ by <= max_step (rad)."""
    if len(path) < 2:
        return [q.copy() for q in path]
    dense = [path[0].copy()]
    for i in range(len(path) - 1):
        seg = path[i + 1] - path[i]
        n = max(1, int(np.ceil(np.linalg.norm(seg) / max_step)))
        for k in range(1, n + 1):
            dense.append((path[i] + (k / n) * seg).copy())
    return dense
