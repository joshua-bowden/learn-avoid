"""Shared helpers for LIBERO obstacle-avoidance research scripts."""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
from typing import Optional

import h5py
import numpy as np

import obstacle  # noqa: F401  — registers RedObstacle

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
GRIPPER_OPEN_ACTION = -1.0
GRIPPER_CLOSED_ACTION = 1.0
OBSTACLE_NAME = "red_obstacle_1"
EF_BODY_NAME = "gripper0_eef"
SKIP_STEPS = 5  # match create_dataset.py cap_index

TASK_SUITE = "libero_spatial"
TASK_ID = 1
TASK_NAME = "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"


def demo_hdf5_path(custom: Optional[str] = None) -> pathlib.Path:
    if custom:
        return pathlib.Path(custom)
    from libero.libero import get_libero_path

    return pathlib.Path(get_libero_path("datasets")) / TASK_SUITE / f"{TASK_NAME}_demo.hdf5"


def obstacle_bddl_path() -> pathlib.Path:
    return pathlib.Path("examples/libero/obstacle/obstacle.bddl")


def task_bddl_path(task_suite: str = TASK_SUITE, task_id: int = TASK_ID) -> pathlib.Path:
    from libero.libero import benchmark, get_libero_path

    task = benchmark.get_benchmark_dict()[task_suite]().get_task(task_id)
    return pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file


def copy_sim_dynamics_from_ref(ref_env, exec_env) -> None:
    """Copy joint positions/velocities from reference env into execution env."""
    ref_data = ref_env.sim.data
    exec_data = exec_env.sim.data
    n = min(len(ref_data.qpos), len(exec_data.qpos))
    exec_data.qpos[:n] = ref_data.qpos[:n]
    exec_data.qvel[:n] = ref_data.qvel[:n]
    exec_env.sim.forward()


def sync_exec_env_from_demo(
    ref_env,
    exec_env,
    model_xml: str,
    demo_state: np.ndarray,
    obstacle_x: float = 0.0,
    obstacle_y: float = 0.0,
    use_obstacle: bool = True,
) -> None:
    """One-time layout sync: demo state → exec env; optionally place obstacle."""
    reset_env_to_demo(ref_env, model_xml, demo_state)
    copy_sim_dynamics_from_ref(ref_env, exec_env)
    if use_obstacle:
        set_obstacle_pose(exec_env, x=obstacle_x, y=obstacle_y)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def is_gripper_closed_action(action) -> bool:
    return float(np.asarray(action).reshape(-1)[-1]) > 0.0


def get_robot_geom_ids(env) -> set[int]:
    """All collision geoms on the mounted arm, gripper, and base."""
    model = env.sim.model
    robot_geom_ids: set[int] = set()
    for body_id in range(model.nbody):
        name = model.body_id2name(body_id)
        if not name:
            continue
        if not (
            name.startswith("robot0")
            or name.startswith("gripper0")
            or name.startswith("mount")
            or name.startswith("mobilebase")
        ):
            continue
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                robot_geom_ids.add(geom_id)
    return robot_geom_ids


def _obstacle_object(env):
    inner = env.env
    if OBSTACLE_NAME in inner.fixtures_dict:
        return inner.fixtures_dict[OBSTACLE_NAME]
    return inner.objects_dict[OBSTACLE_NAME]


def get_obstacle_geom_ids(env) -> set[int]:
    model = env.sim.model
    obstacle_body_name = _obstacle_object(env).root_body
    obstacle_body_id = model.body_name2id(obstacle_body_name)
    return {i for i in range(model.ngeom) if model.geom_bodyid[i] == obstacle_body_id}


def check_obstacle_collision(env, robot_geom_ids: Optional[set[int]] = None) -> bool:
    """True when robot geoms contact the red obstacle."""
    inner = env.env
    if OBSTACLE_NAME not in inner.fixtures_dict and OBSTACLE_NAME not in inner.objects_dict:
        return False
    if robot_geom_ids is None:
        robot_geom_ids = get_robot_geom_ids(env)
    obstacle_geom_ids = get_obstacle_geom_ids(env)
    data = env.sim.data
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if (g1 in robot_geom_ids and g2 in obstacle_geom_ids) or (
            g2 in robot_geom_ids and g1 in obstacle_geom_ids
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Obstacle geometry & placement
# ---------------------------------------------------------------------------
# MuJoCo box half-height in examples/libero/obstacle/assets/red_obstacle.xml (geom size Z).
# Full box height = 2 * OBSTACLE_HALF_HEIGHT.
OBSTACLE_HALF_HEIGHT = 0.12

# Table surface height in world frame (robosuite tabletop arena).
TABLE_SURFACE_Z = 0.91

# Fixed X on the plate → ramekin → obstacle line (beyond ramekin toward the robot).
OBSTACLE_X = -0.32

# Per-episode left/right slide toward (+Y) or away from (−Y) the stove (stove Y ≈ −0.14).
OBSTACLE_Y_MIN = 0.14
OBSTACLE_Y_MAX = 0.26


def obstacle_base_z() -> float:
    """World-frame Z of obstacle body origin so box bottom rests on the table."""
    return TABLE_SURFACE_Z + OBSTACLE_HALF_HEIGHT


def obstacle_y_for_episode(episode_idx: int, num_episodes: int = 50) -> float:
    """Sweep obstacle left/right (table Y) toward/away from stove across episodes."""
    if num_episodes <= 1:
        return (OBSTACLE_Y_MAX + OBSTACLE_Y_MIN) / 2
    t = episode_idx / (num_episodes - 1)
    return OBSTACLE_Y_MAX + t * (OBSTACLE_Y_MIN - OBSTACLE_Y_MAX)


def obstacle_pose_for_episode(
    episode_idx: int, num_episodes: int = 50
) -> tuple[float, float]:
    """Return (x, y) obstacle position for an episode."""
    return OBSTACLE_X, obstacle_y_for_episode(episode_idx, num_episodes)


def set_obstacle_pose(
    env,
    x: float,
    y: float | None = None,
    z: float | None = None,
) -> None:
    """
    Move the static red obstacle after env reset.

    Args:
        env: LIBERO OffScreenRenderEnv with obstacle.bddl loaded.
        x: Table X on the plate–ramekin line. Fixed at OBSTACLE_X; use y for per-episode sweep.
        y: Table Y slide toward/away from stove. Default from obstacle_y_for_episode().
        z: Body-origin height. Default places box bottom on TABLE_SURFACE_Z.
    """
    if y is None:
        y = (OBSTACLE_Y_MAX + OBSTACLE_Y_MIN) / 2
    if z is None:
        z = obstacle_base_z()
    body_name = _obstacle_object(env).root_body
    body_id = env.sim.model.body_name2id(body_name)
    # The obstacle is a static (welded) body, so MuJoCo recomputes data.body_xpos
    # from model.body_pos on every forward/step. Set model.body_pos so the move is
    # persistent and visible to any MjData built from this model (e.g. the planner).
    env.sim.model._model.body_pos[body_id] = np.array([x, y, z])
    env.sim.data.body_xpos[body_id] = np.array([x, y, z])
    env.sim.forward()


def set_obstacle_xy(env, x: float, y: float, z: float | None = None) -> None:
    """Backward-compatible alias; prefer set_obstacle_pose()."""
    set_obstacle_pose(env, x=x, y=y, z=z)


def check_task_success(env) -> bool:
    """LIBERO built-in BDDL goal check (e.g. bowl on plate). Same as replay/eval."""
    return bool(env.check_success())


def make_env(bddl_file: pathlib.Path, resolution: int = 256, seed: int = 0):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def load_episode_from_hdf5(f: h5py.File, ep_key: str):
    """Return model_xml, states, actions for one demo."""
    grp = f[f"data/{ep_key}"]
    model_xml = grp.attrs["model_file"]
    if isinstance(model_xml, bytes):
        model_xml = model_xml.decode("utf-8")
    states = grp["states"][()]
    actions = np.array(grp["actions"][()])
    return model_xml, states, actions


def postprocess_demo_model_xml(xml_str: str) -> str:
    """Fix robosuite + LIBERO asset paths embedded in demonstration model XML."""
    import xml.etree.ElementTree as ET

    import libero.libero.utils.utils as libero_utils
    import robosuite
    from libero.libero import get_libero_path

    xml_str = libero_utils.postprocess_model_xml(xml_str, {})

    assets_root = pathlib.Path(get_libero_path("assets"))
    tree = ET.fromstring(xml_str)
    asset = tree.find("asset")
    if asset is None:
        return xml_str

    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        old_path = elem.get("file")
        if not old_path:
            continue
        parts = pathlib.PurePosixPath(old_path).parts
        if "assets" in parts:
            idx = parts.index("assets")
            rel = pathlib.Path(*parts[idx + 1 :])
            candidate = assets_root / rel
            if candidate.exists():
                elem.set("file", str(candidate))
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def reset_env_to_demo(env, model_xml: str, init_state: np.ndarray) -> None:
    env.reset()
    model_xml = postprocess_demo_model_xml(model_xml)
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()


def sorted_demo_keys(f: h5py.File) -> list[str]:
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    return [demos[i] for i in inds]


def eef_pose_from_obs(obs) -> tuple[np.ndarray, np.ndarray]:
    pos = obs["robot0_eef_pos"].copy()
    ori = quat2axisangle(obs["robot0_eef_quat"])
    return pos, ori


def compute_action_from_eef_delta(
    prev_pos, prev_ori, curr_pos, curr_ori, gripper: float, pos_scale=0.05, ori_scale=0.5
) -> np.ndarray:
    """Approximate OSC delta action from consecutive EEF poses."""
    dpos = (curr_pos - prev_pos) / pos_scale
    dori = (curr_ori - prev_ori) / ori_scale
    action = np.concatenate([dpos, dori, [gripper]])
    return np.clip(action, -1.0, 1.0)


def eef_pose_for_arm_qpos(env, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FK for a candidate arm qpos; restores sim state before returning."""
    saved = env.sim.get_state().flatten()
    env.sim.data.qpos[:7] = q
    env.sim.data.qvel[:7] = 0.0
    env.sim.forward()
    pos = env.sim.data.get_body_xpos(EF_BODY_NAME).copy()
    quat = env.sim.data.get_body_xquat(EF_BODY_NAME).copy()
    ori = quat2axisangle(quat)
    env.sim.set_state_from_flattened(saved)
    env.sim.forward()
    return pos, ori


def validate_arm_clearance(
    env,
    eef_pos: np.ndarray,
    eef_quat_wxyz: np.ndarray,
    save_fn,
    restore_fn,
    min_clearance: float = 0.04,
) -> bool:
    """MuJoCo check: full arm vs red obstacle only (single config, fast)."""
    from motion_planner import solve_ik_to_pose

    base = save_fn()
    q = solve_ik_to_pose(env, eef_pos, eef_quat_wxyz)
    env.sim.data.qpos[:7] = q
    env.sim.data.qvel[:7] = 0.0
    env.sim.forward()
    ok = _robot_obstacle_dist(env) >= min_clearance
    restore_fn(base)
    return ok


def snap_eef_to_pose(
    env,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    save_fn,
    restore_fn,
    min_clearance: float = 0.04,
    n_ik_retries: int = 40,
) -> bool:
    """Set arm via IK to target EEF if arm–obstacle clearance OK (segment boundary snap)."""
    from motion_planner import solve_ik_to_pose

    base = save_fn()
    q = solve_ik_to_pose(env, target_pos, target_quat_wxyz)
    jlo = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    jhi = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8971])
    for attempt in range(n_ik_retries):
        q_try = q if attempt == 0 else np.clip(q + np.random.randn(7) * 0.06, jlo, jhi)
        env.sim.data.qpos[:7] = q_try
        env.sim.data.qvel[:7] = 0.0
        env.sim.forward()
        if _robot_obstacle_dist(env) >= min_clearance and not check_obstacle_collision(env):
            return True
    restore_fn(base)
    return False


def execute_cartesian_path(
    env,
    positions: list[np.ndarray],
    target_quat_wxyz: np.ndarray,
    gripper_val: float,
    record_step_fn,
    steps_per_segment: int = 8,
    pos_tol: float = 0.008,
    max_substeps: int = 40,
) -> bool:
    """Follow EEF position waypoints via OSC deltas (matches demo action format)."""
    if len(positions) < 2:
        dense = [p.copy() for p in positions]
    else:
        dense = [positions[0].copy()]
        for i in range(len(positions) - 1):
            for alpha in np.linspace(0, 1, steps_per_segment + 1)[1:]:
                dense.append(((1 - alpha) * positions[i] + alpha * positions[i + 1]).copy())

    target_ori = quat2axisangle(target_quat_wxyz)
    obs = env.env._get_observations()

    for target_pos in dense:
        for _ in range(max_substeps):
            curr_pos, curr_ori = eef_pose_from_obs(obs)
            if np.linalg.norm(curr_pos - target_pos) < pos_tol:
                break
            action = compute_action_from_eef_delta(
                curr_pos, curr_ori, target_pos, target_ori, gripper_val
            )
            obs, _, _, _ = env.step(action.tolist())
            record_step_fn(obs, action)
            if check_obstacle_collision(env):
                return False
    return True


def execute_planned_qpos_path(
    env,
    planner,
    qpos_path: list[np.ndarray],
    gripper_val: float,
    record_step_fn,
    save_state_fn,
    restore_state_fn,
    steps_per_segment: int = 20,
    joint_blend: float = 0.07,
    joint_tol: float = 0.015,
    max_substeps: int = 100,
    exec_clearance: float | None = None,
    use_direct_joint: bool | None = None,
) -> bool:
    """
    Track densified RRT waypoints. Open-gripper segments use direct joint qpos
    stepping (matches planned path); closed-gripper segments use OSC blends.
    """
    from motion_planner import EXECUTION_CLEARANCE_RATIO

    if exec_clearance is None:
        exec_clearance = planner.obstacle_clearance * EXECUTION_CLEARANCE_RATIO

    if use_direct_joint is None:
        use_direct_joint = gripper_val <= 0.0  # open gripper

    dense = planner.densify_for_execution(
        qpos_path,
        steps_per_segment=steps_per_segment,
        save_state_fn=save_state_fn,
        restore_state_fn=restore_state_fn,
    )
    if not dense:
        return False

    if use_direct_joint:
        for q_wp in dense:
            if planner.config_in_collision(q_wp, save_state_fn, restore_state_fn, exec_clearance):
                return False
            env.sim.data.qpos[:7] = q_wp
            env.sim.data.qvel[:7] = 0.0
            env.sim.forward()
            obs = env.env._get_observations()
            action = np.array([0.0] * 6 + [gripper_val])
            record_step_fn(obs, action)
            if check_obstacle_collision(env):
                return False
        return True

    for q_wp in dense:
        for _ in range(max_substeps):
            obs = env.env._get_observations()
            curr_q = np.asarray(obs["robot0_joint_pos"][:7])
            if np.linalg.norm(curr_q - q_wp) < joint_tol:
                break

            q_next = curr_q + joint_blend * (q_wp - curr_q)
            if planner.config_in_collision(q_next, save_state_fn, restore_state_fn, exec_clearance):
                q_next = curr_q + (joint_blend * 0.35) * (q_wp - curr_q)
                if planner.config_in_collision(q_next, save_state_fn, restore_state_fn, exec_clearance):
                    return False

            target_pos, target_ori = eef_pose_for_arm_qpos(env, q_next)
            curr_pos, curr_ori = eef_pose_from_obs(obs)
            action = compute_action_from_eef_delta(
                curr_pos, curr_ori, target_pos, target_ori, gripper_val
            )
            obs, _, _, _ = env.step(action.tolist())
            record_step_fn(obs, action)

            if check_obstacle_collision(env):
                return False
            if _robot_obstacle_dist(env) < exec_clearance * 0.85:
                return False
    return True


def _robot_obstacle_dist(env) -> float:
    import mujoco

    inner = env.env
    if OBSTACLE_NAME not in inner.fixtures_dict and OBSTACLE_NAME not in inner.objects_dict:
        return float("inf")
    robot_ids = get_robot_geom_ids(env)
    obs_ids = get_obstacle_geom_ids(env)
    m = env.sim.model._model
    d = env.sim.data._data
    fromto = np.zeros(6, dtype=np.float64)
    min_d = float("inf")
    for rg in robot_ids:
        for og in obs_ids:
            min_d = min(min_d, mujoco.mj_geomDistance(m, d, rg, og, 10.0, fromto))
    return min_d


def demo_eef_goal(
    ref_env,
    model_xml: str,
    demo_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Target EEF pose from demo state (reference env, no obstacle)."""
    reset_env_to_demo(ref_env, model_xml, demo_state)
    pos = ref_env.sim.data.get_body_xpos(EF_BODY_NAME).copy()
    quat = ref_env.sim.data.get_body_xquat(EF_BODY_NAME).copy()
    return pos, quat


def log_dataset_info(path: pathlib.Path) -> None:
    from libero.libero.utils.dataset_utils import get_dataset_info

    if not path.exists():
        logging.warning("Dataset not found: %s", path)
        return
    get_dataset_info(str(path), verbose=False)


def read_hdf5_env_meta(f: h5py.File) -> dict:
    return json.loads(f["data"].attrs.get("env_args", "{}"))


AGENT_CAMERA_KEY = "agentview_image"
WRIST_CAMERA_KEY = "robot0_eye_in_hand_image"


def camera_frame(obs, key: str = AGENT_CAMERA_KEY) -> np.ndarray:
    """LIBERO camera obs → uint8 HWC, flipped to match eval scripts."""
    img = np.ascontiguousarray(obs[key][::-1, ::-1])
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img


class DualCameraRecorder:
    """Accumulate agentview + wrist frames during sim rollouts."""

    def __init__(self) -> None:
        self.agent_frames: list[np.ndarray] = []
        self.wrist_frames: list[np.ndarray] = []

    def append(self, obs) -> None:
        self.agent_frames.append(camera_frame(obs, AGENT_CAMERA_KEY))
        self.wrist_frames.append(camera_frame(obs, WRIST_CAMERA_KEY))

    def truncate(self, n: int) -> None:
        """Drop frames after index n (for segment retry)."""
        self.agent_frames = self.agent_frames[:n]
        self.wrist_frames = self.wrist_frames[:n]

    def save(self, out_dir: pathlib.Path, prefix: str, fps: int = 20) -> tuple[pathlib.Path, pathlib.Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        agent_path = out_dir / f"{prefix}_agent.mp4"
        wrist_path = out_dir / f"{prefix}_wrist.mp4"
        if not self.agent_frames:
            raise ValueError(f"No frames recorded for {prefix}")
        import imageio

        imageio.mimwrite(agent_path, self.agent_frames, fps=fps)
        imageio.mimwrite(wrist_path, self.wrist_frames, fps=fps)
        logging.info("Saved agent video: %s", agent_path)
        logging.info("Saved wrist video: %s", wrist_path)
        return agent_path, wrist_path
