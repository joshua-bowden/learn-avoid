"""RRT-Connect motion planner in joint space with MuJoCo collision checking."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import mink
import mujoco
import numpy as np

from common import EF_BODY_NAME, get_obstacle_geom_ids, get_robot_geom_ids

ARM_DOF = 7
# Minimum allowed distance between any arm geom and obstacle (meters).
OBSTACLE_CLEARANCE = 0.09
EXECUTION_CLEARANCE_RATIO = 0.72  # stop execution earlier than planning clearance
DEFAULT_JOINT_LIMITS = (
    np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8971]),
)


class JointSpacePlanner:
    """RRT-Connect for the Panda arm with full-arm obstacle clearance."""

    def __init__(
        self,
        env,
        step_size: float = 0.025,
        max_iters: int = 12000,
        goal_threshold: float = 0.06,
        ignore_body_names: Optional[set[str]] = None,
        obstacle_clearance: float = OBSTACLE_CLEARANCE,
    ):
        self.env = env
        self.step_size = step_size
        self.max_iters = max_iters
        self.goal_threshold = goal_threshold
        self.ignore_body_names = ignore_body_names or set()
        self.obstacle_clearance = obstacle_clearance
        self.robot_geom_ids = get_robot_geom_ids(env)
        self.obstacle_geom_ids = get_obstacle_geom_ids(env)
        self.jlo, self.jhi = DEFAULT_JOINT_LIMITS
        self._m = env.sim.model._model
        self._d = env.sim.data._data

    def _set_arm_qpos(self, q: np.ndarray) -> None:
        data = self.env.sim.data
        data.qpos[:ARM_DOF] = q
        data.qvel[:ARM_DOF] = 0.0
        self.env.sim.forward()

    def _robot_obstacle_distance(self) -> float:
        """Minimum distance between any robot arm geom and obstacle geoms."""
        if not self.obstacle_geom_ids:
            return float("inf")
        min_dist = float("inf")
        fromto = np.zeros(6, dtype=np.float64)
        for rg in self.robot_geom_ids:
            for og in self.obstacle_geom_ids:
                dist = mujoco.mj_geomDistance(self._m, self._d, rg, og, 10.0, fromto)
                min_dist = min(min_dist, dist)
        return min_dist

    def _in_collision(self) -> bool:
        if self._robot_obstacle_distance() < self.obstacle_clearance:
            return True

        sim = self.env.sim
        data = sim.data
        ignore_ids = set()
        for name in self.ignore_body_names:
            try:
                bid = sim.model.body_name2id(name)
                for i in range(sim.model.ngeom):
                    if sim.model.geom_bodyid[i] == bid:
                        ignore_ids.add(i)
            except Exception:
                pass

        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in ignore_ids or g2 in ignore_ids:
                continue
            if g1 in self.robot_geom_ids or g2 in self.robot_geom_ids:
                other = g2 if g1 in self.robot_geom_ids else g1
                if other not in self.obstacle_geom_ids and other not in ignore_ids:
                    if c.dist < -0.002:
                        return True
        return False

    def config_obstacle_distance(self, q: np.ndarray, save_state_fn, restore_state_fn) -> float:
        base = save_state_fn()
        self._set_arm_qpos(q)
        dist = self._robot_obstacle_distance()
        restore_state_fn(base)
        return dist

    def config_in_collision(
        self, q: np.ndarray, save_state_fn, restore_state_fn, clearance: float | None = None
    ) -> bool:
        margin = clearance if clearance is not None else self.obstacle_clearance
        return self.config_obstacle_distance(q, save_state_fn, restore_state_fn) < margin

    def densify_for_execution(
        self,
        waypoints: list[np.ndarray],
        steps_per_segment: int = 20,
        save_state_fn: Callable[[], np.ndarray] | None = None,
        restore_state_fn: Callable[[np.ndarray], None] | None = None,
    ) -> list[np.ndarray]:
        """Densify with collision checks — joint interpolation can cut through obstacles."""
        if save_state_fn is None or restore_state_fn is None:
            return densify_path(waypoints, steps_per_segment=steps_per_segment)
        return collision_free_densify(
            self,
            waypoints,
            save_state_fn,
            restore_state_fn,
            steps_per_segment=steps_per_segment,
            clearance=self.obstacle_clearance,
        )

    def _segment_collision_free(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        save_state_fn: Callable[[], np.ndarray],
        restore_state_fn: Callable[[np.ndarray], None],
        n_checks: int = 128,
    ) -> bool:
        base = save_state_fn()
        for alpha in np.linspace(0.0, 1.0, n_checks):
            q = (1.0 - alpha) * q_from + alpha * q_to
            restore_state_fn(base)
            self._set_arm_qpos(q)
            if self._in_collision():
                restore_state_fn(base)
                return False
        restore_state_fn(base)
        return True

    def shortcut_path(
        self,
        waypoints: list[np.ndarray],
        save_state_fn: Callable[[], np.ndarray],
        restore_state_fn: Callable[[np.ndarray], None],
        n_iters: int = 80,
    ) -> list[np.ndarray]:
        """Greedy shortcutting to reduce unnecessary zig-zags."""
        if len(waypoints) < 3:
            return waypoints
        path = [q.copy() for q in waypoints]
        for _ in range(n_iters):
            if len(path) < 3:
                break
            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))
            if self._segment_collision_free(path[i], path[j], save_state_fn, restore_state_fn):
                path = path[: i + 1] + path[j:]
        return path

    def validate_path(
        self,
        waypoints: list[np.ndarray],
        save_state_fn: Callable[[], np.ndarray],
        restore_state_fn: Callable[[np.ndarray], None],
    ) -> bool:
        """Check waypoints and straight joint-space segments between them."""
        if not waypoints:
            return False
        base = save_state_fn()
        for q in waypoints:
            restore_state_fn(base)
            self._set_arm_qpos(q)
            if self._in_collision():
                restore_state_fn(base)
                return False
        for i in range(len(waypoints) - 1):
            if not self._segment_collision_free(
                waypoints[i], waypoints[i + 1], save_state_fn, restore_state_fn
            ):
                return False
        restore_state_fn(base)
        return True

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        delta = q_to - q_from
        dist = np.linalg.norm(delta)
        if dist <= self.step_size:
            return np.clip(q_to, self.jlo, self.jhi)
        q_new = q_from + self.step_size * delta / dist
        return np.clip(q_new, self.jlo, self.jhi)

    def _near(self, tree: list[np.ndarray], q: np.ndarray) -> int:
        dists = [np.linalg.norm(n - q) for n in tree]
        return int(np.argmin(dists))

    def plan_through(
        self,
        q_waypoints: list[np.ndarray],
        save_state_fn: Callable[[], np.ndarray],
        restore_state_fn: Callable[[np.ndarray], None],
    ) -> Optional[list[np.ndarray]]:
        """Chain RRT segments through intermediate joint-space goals."""
        if len(q_waypoints) < 2:
            return None
        full_path: list[np.ndarray] = []
        for i in range(len(q_waypoints) - 1):
            seg = self.plan(
                q_waypoints[i], q_waypoints[i + 1], save_state_fn, restore_state_fn
            )
            if seg is None:
                return None
            if full_path:
                full_path.extend(seg[1:])
            else:
                full_path.extend(seg)
        return full_path

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        save_state_fn: Callable[[], np.ndarray],
        restore_state_fn: Callable[[np.ndarray], None],
    ) -> Optional[list[np.ndarray]]:
        """RRT-Connect; returns shortcut waypoints or None."""
        restore_state_fn(save_state_fn())
        self._set_arm_qpos(q_start)
        if self._in_collision():
            logging.warning("Start configuration in collision (dist=%.4f)", self._robot_obstacle_distance())
            return None

        tree_a = [q_start.copy()]
        tree_b = [q_goal.copy()]
        parent_a = [-1]
        parent_b = [-1]

        for iteration in range(self.max_iters):
            if iteration % 3 == 0:
                q_rand = q_goal if np.random.rand() < 0.5 else q_start
            else:
                q_rand = np.random.uniform(self.jlo, self.jhi)

            idx = self._near(tree_a, q_rand)
            q_new = self._steer(tree_a[idx], q_rand)
            restore_state_fn(save_state_fn())
            if not self._segment_collision_free(
                tree_a[idx], q_new, save_state_fn, restore_state_fn, n_checks=32
            ):
                tree_a, tree_b = tree_b, tree_a
                parent_a, parent_b = parent_b, parent_a
                continue
            tree_a.append(q_new)
            parent_a.append(idx)

            path = self._connect(tree_b, parent_b, q_new, save_state_fn, restore_state_fn)
            if path is not None:
                full = self._merge_paths(tree_a, parent_a, path)
                full = self.shortcut_path(full, save_state_fn, restore_state_fn)
                if not self.validate_path(full, save_state_fn, restore_state_fn):
                    restore_state_fn(save_state_fn())
                    tree_a, tree_b = tree_b, tree_a
                    parent_a, parent_b = parent_b, parent_a
                    continue
                logging.info(
                    "RRT-Connect succeeded in %d iters, %d waypoints (after shortcut)",
                    iteration,
                    len(full),
                )
                restore_state_fn(save_state_fn())
                return full

            tree_a, tree_b = tree_b, tree_a
            parent_a, parent_b = parent_b, parent_a

        logging.warning("RRT-Connect failed after %d iterations", self.max_iters)
        restore_state_fn(save_state_fn())
        return None

    def _connect(self, tree, parent, q_target, save_state_fn, restore_state_fn):
        idx = self._near(tree, q_target)
        q = tree[idx]
        path = [q.copy()]
        for _ in range(80):
            if np.linalg.norm(q - q_target) < self.goal_threshold:
                path.reverse()
                return path
            q_next = self._steer(q, q_target)
            if not self._segment_collision_free(q, q_next, save_state_fn, restore_state_fn, n_checks=32):
                return None
            q = q_next
            tree.append(q)
            parent.append(idx)
            idx = len(tree) - 1
            path.append(q.copy())
        return None

    @staticmethod
    def _merge_paths(tree_a, parent_a, path_b) -> list[np.ndarray]:
        connect = tree_a[-1]
        idx = len(tree_a) - 1
        path_a = [connect]
        while idx >= 0:
            idx = parent_a[idx]
            if idx >= 0:
                path_a.append(tree_a[idx])
        path_a.reverse()
        return path_a + path_b


def densify_path(waypoints: list[np.ndarray], steps_per_segment: int = 8) -> list[np.ndarray]:
    if len(waypoints) < 2:
        return waypoints
    dense = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        for alpha in np.linspace(0, 1, steps_per_segment + 1)[1:]:
            q = (1 - alpha) * waypoints[i] + alpha * waypoints[i + 1]
            dense.append(q.copy())
    return dense


def collision_free_densify(
    planner: JointSpacePlanner,
    waypoints: list[np.ndarray],
    save_state_fn: Callable[[], np.ndarray],
    restore_state_fn: Callable[[np.ndarray], None],
    steps_per_segment: int = 20,
    clearance: float | None = None,
    max_depth: int = 7,
) -> list[np.ndarray]:
    """Adaptively subdivide joint segments until every step is collision-free."""
    del steps_per_segment  # kept for API compatibility
    if len(waypoints) < 2:
        return [q.copy() for q in waypoints]

    margin = clearance if clearance is not None else planner.obstacle_clearance
    dense: list[np.ndarray] = [waypoints[0].copy()]

    def _append_free(q_from: np.ndarray, q_to: np.ndarray, depth: int) -> bool:
        if np.linalg.norm(q_from - q_to) < 1e-6:
            return True
        if planner._segment_collision_free(q_from, q_to, save_state_fn, restore_state_fn, n_checks=64):
            dense.append(q_to.copy())
            return True
        if depth >= max_depth:
            return False
        q_mid = 0.5 * (q_from + q_to)
        if planner.config_in_collision(q_mid, save_state_fn, restore_state_fn, margin):
            return False
        if not _append_free(q_from, q_mid, depth + 1):
            return False
        return _append_free(q_mid, q_to, depth + 1)

    for q_to in waypoints[1:]:
        if not _append_free(dense[-1], q_to, 0):
            return []
    return dense


def solve_ik_to_pose(env, target_pos: np.ndarray, target_quat_wxyz: np.ndarray) -> np.ndarray:
    """Operational-space IK via mink; returns 7-DOF arm qpos."""
    model = env.sim.model._model
    data = env.sim.data._data
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)

    target_rot = mink.SO3(wxyz=target_quat_wxyz)
    end_effector_task = mink.FrameTask(
        frame_name=EF_BODY_NAME,
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]
    T_wt = mink.SE3.from_rotation_and_translation(rotation=target_rot, translation=target_pos)
    end_effector_task.set_target(T_wt)
    posture_task.set_target_from_configuration(configuration)

    for _ in range(40):
        vel = mink.solve_ik(configuration, tasks, 0.05, "quadprog", 1e-3)
        configuration.integrate_inplace(vel, 0.05)
        err = end_effector_task.compute_error(configuration)
        if np.linalg.norm(err[:3]) <= 1e-4 and np.linalg.norm(err[3:]) <= 1e-4:
            break
    return configuration.q[:ARM_DOF].copy()


def collision_free_ik_goal(
    planner: JointSpacePlanner,
    env,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    save_state_fn,
    restore_state_fn,
    n_samples: int = 200,
    z_offset: float = 0.0,
) -> Optional[np.ndarray]:
    """
    IK to demo EEF pose (+ optional Z lift for arc detours), sample until collision-free.
    """
    target_pos = target_pos.copy()
    target_pos[2] += z_offset
    base = save_state_fn()
    q = solve_ik_to_pose(env, target_pos, target_quat_wxyz)
    restore_state_fn(base)
    planner._set_arm_qpos(q)
    if not planner._in_collision():
        restore_state_fn(base)
        return q

    jlo, jhi = planner.jlo, planner.jhi
    scales = [0.04, 0.08, 0.12, 0.16]
    for scale in scales:
        for _ in range(n_samples // len(scales)):
            q_try = q + np.random.randn(ARM_DOF) * scale
            q_try = np.clip(q_try, jlo, jhi)
            restore_state_fn(base)
            planner._set_arm_qpos(q_try)
            if not planner._in_collision():
                restore_state_fn(base)
                return q_try
    restore_state_fn(base)
    return None
