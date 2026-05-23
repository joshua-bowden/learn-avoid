import collections
import dataclasses
import logging
import math
import pathlib

import cv2
import imageio
import mink
import numpy as np
import tyro

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

import obstacle  # noqa: F401


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
GRIPPER_CLOSED_ACTION = [0.0] * 6 + [1.0]
LIBERO_ENV_RESOLUTION = 512
OBSTACLE_NAME = "red_obstacle_1"
EF_BODY_NAME = "gripper0_eef"

IK_SOLVER = "quadprog"
IK_POS_THRESHOLD = 1e-4
IK_ORI_THRESHOLD = 1e-4
IK_MAX_ITERS = 30
IK_DT = 0.05

# IK retreat after grasp: gripper closed this many consecutive steps, then move by delta.
GRIPPER_CLOSED_STEPS = 5
IK_MOTION_STEPS = 40
# World-frame offset (m): +y = left, +z = up
IK_DELTA_XYZ = np.array([0.0, 1.0, 1.0])


def check_obstacle_collision(env):
    """Check MuJoCo contacts directly for robot-vs-obstacle overlap."""
    sim = env.sim
    model = sim.model
    data = sim.data

    obstacle_body_name = env.env.objects_dict[OBSTACLE_NAME].root_body
    obstacle_body_id = model.body_name2id(obstacle_body_name)
    obstacle_geom_ids = set(
        i for i in range(model.ngeom) if model.geom_bodyid[i] == obstacle_body_id
    )

    robot = env.env.robots[0]
    robot_geom_ids = set()
    for body_name in robot.robot_model.bodies + (robot.gripper.bodies if robot.gripper else []):
        try:
            bid = model.body_name2id(body_name)
            for i in range(model.ngeom):
                if model.geom_bodyid[i] == bid:
                    robot_geom_ids.add(i)
        except Exception:
            continue

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if (g1 in robot_geom_ids and g2 in obstacle_geom_ids) or (
            g2 in robot_geom_ids and g1 in obstacle_geom_ids
        ):
            return True
    return False


def converge_ik(configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters):
    """Runs IK until position/orientation error is below threshold."""
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        configuration.integrate_inplace(vel, dt)

        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
        if pos_achieved and ori_achieved:
            return True
    return False


def solve_ik_to_position(env, target_pos: np.ndarray) -> np.ndarray:
    """Solve IK for target end-effector position, keeping current orientation."""
    model = env.sim.model._model
    data = env.sim.data._data

    configuration = mink.Configuration(model)
    configuration.update(data.qpos)

    ef_frame_body_id = model.body(EF_BODY_NAME).id
    initial_rotation = mink.SO3(wxyz=data.xquat[ef_frame_body_id])

    end_effector_task = mink.FrameTask(
        frame_name=EF_BODY_NAME,
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    T_wt = mink.SE3.from_rotation_and_translation(
        rotation=initial_rotation,
        translation=target_pos,
    )
    end_effector_task.set_target(T_wt)
    posture_task.set_target_from_configuration(configuration)

    converge_ik(
        configuration,
        tasks,
        IK_DT,
        IK_SOLVER,
        IK_POS_THRESHOLD,
        IK_ORI_THRESHOLD,
        IK_MAX_ITERS,
    )
    return configuration.q[:7].copy()


def is_gripper_closed_action(action) -> bool:
    """True when the policy command is closing the gripper."""
    return float(np.asarray(action).reshape(-1)[-1]) > 0.0


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 15
    num_trials: int = 1
    max_steps: int = 250
    video_out_path: str = "data/baseline_ik"
    seed: int = 10


def collect_rollout(args: Args) -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    task = task_suite.get_task(1)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    total_episodes = 0
    total_successes = 0
    total_collisions = 0

    for trial_id in range(args.num_trials):
        env.reset()

        for _ in range(args.num_steps_wait):
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

        logging.info(f"\nTrial {trial_id}: {task_description}")

        action_plan = collections.deque()
        t = 0
        replay_images = []
        replay_wrist_images = []
        episode_had_collision = False

        gripper_closed_streak = 0
        ik_intervention_done = False
        ik_executing = False
        ik_step = 0
        ik_start_qpos = None
        ik_target_qpos = None

        while t < args.max_steps:
            try:
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(img)
                wrist_img = image_tools.convert_to_uint8(wrist_img)

                state_vec = np.concatenate((
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ))

                new_img = img.copy()
                cv2.putText(new_img, f"{t}: {task_description}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                replay_images.append(new_img)

                new_wrist = wrist_img.copy()
                cv2.putText(new_wrist, f"{t}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                replay_wrist_images.append(new_wrist)

                if ik_executing:
                    alpha = (ik_step + 1) / IK_MOTION_STEPS
                    interp_qpos = (1.0 - alpha) * ik_start_qpos + alpha * ik_target_qpos
                    data = env.sim.data._data
                    data.qpos[:7] = interp_qpos
                    data.qvel[:7] = 0.0
                    env.sim.forward()
                    obs, reward, done, info = env.step(GRIPPER_CLOSED_ACTION)

                    ik_step += 1
                    if ik_step >= IK_MOTION_STEPS:
                        ik_executing = False
                        action_plan.clear()
                        logging.info(
                            "IK retreat complete (delta=%s); resuming VLA",
                            IK_DELTA_XYZ,
                        )
                else:
                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state_vec,
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    if (
                        not ik_intervention_done
                        and is_gripper_closed_action(action)
                    ):
                        gripper_closed_streak += 1
                    else:
                        gripper_closed_streak = 0

                    if (
                        not ik_intervention_done
                        and gripper_closed_streak >= GRIPPER_CLOSED_STEPS
                    ):
                        current_pos = obs["robot0_eef_pos"].copy()
                        target_pos = current_pos + IK_DELTA_XYZ
                        logging.info(
                            "Gripper closed for %d steps; IK retreat from %s to %s",
                            GRIPPER_CLOSED_STEPS,
                            current_pos,
                            target_pos,
                        )
                        ik_start_qpos = env.sim.data._data.qpos[:7].copy()
                        ik_target_qpos = solve_ik_to_position(env, target_pos)
                        ik_intervention_done = True
                        ik_executing = True
                        ik_step = 0
                        action_plan.clear()
                        continue

                    obs, reward, done, info = env.step(action.tolist())

                if not episode_had_collision and check_obstacle_collision(env):
                    episode_had_collision = True

                if done:
                    total_successes += 1
                    break
                t += 1

            except Exception as e:
                logging.error(f"Caught exception: {e}")
                break

        total_episodes += 1
        if episode_had_collision:
            total_collisions += 1

        suffix = "success" if done else "fail"
        agent_filename = f"agent_task_1_episode_{trial_id}_{suffix}.mp4"
        wrist_filename = f"wrist_task_1_episode_{trial_id}_{suffix}.mp4"
        logging.info(f"Saving videos: {args.video_out_path}/{{{agent_filename}, {wrist_filename}}}")
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / agent_filename,
            [np.asarray(x) for x in replay_images],
            fps=10,
        )
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / wrist_filename,
            [np.asarray(x) for x in replay_wrist_images],
            fps=10,
        )

        success_pct = total_successes / total_episodes * 100
        collision_pct = total_collisions / total_episodes * 100
        logging.info(
            f"Episode {total_episodes}: {'SUCCESS' if done else 'FAIL'} | "
            f"collision={'YES' if episode_had_collision else 'no'} | "
            f"Success rate: {total_successes}/{total_episodes} ({success_pct:.1f}%) | "
            f"Collision rate: {total_collisions}/{total_episodes} ({collision_pct:.1f}%)"
        )

    logging.info("\n=== Final Results ===")
    success_pct = total_successes / total_episodes * 100 if total_episodes else 0
    collision_pct = total_collisions / total_episodes * 100 if total_episodes else 0
    logging.info(f"Success rate: {total_successes}/{total_episodes} ({success_pct:.1f}%)")
    logging.info(f"Collision rate: {total_collisions}/{total_episodes} ({collision_pct:.1f}%)")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path("examples/libero/obstacle/obstacle.bddl")
    logging.info(f"BDDL: {task_bddl_file}")
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(collect_rollout)
