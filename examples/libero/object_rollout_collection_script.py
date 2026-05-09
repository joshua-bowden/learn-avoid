import collections
import dataclasses
import datetime
import logging
import math
import pathlib
import shutil
import time
import typing

import cv2
import imageio
import mujoco
import mink
import numpy as np
import tqdm
import tyro
import matplotlib.pyplot as plt
import h5py

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.robots.mounted_panda import MountedPanda
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from robosuite.utils import sim_utils


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 512  #256 #resolution used to render training data
DEFAULT_QPOS = np.array([0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4])



def converge_ik(
    configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters
):
    """
    Runs up to 'max_iters' of IK steps. Returns True if position and orientation
    are below thresholds, otherwise False.
    """
    for idx in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        configuration.integrate_inplace(vel, dt)

        # Only checking the first FrameTask here (end_effector_task).
        # If you want to check multiple tasks, sum or combine their errors.
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
        # print('idx # ', idx, 'pos error = ', err[:3], np.linalg.norm(err[:3]))

        if pos_achieved and ori_achieved:
            return True
    return False


def _get_arm_qpos_near_object(env, target_obj_name: str) -> np.ndarray:
    """Uses IK to find a qpos that places the arm near the target object."""

    # IK parameters
    ik_solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_ik_iters = 30
    ef_body_name = 'gripper0_eef'

    model = env.sim.model._model
    data = env.sim.data._data

    target_body_name = env.env.objects_dict[target_obj_name].root_body
    obj_id = model.body(target_body_name).id
    obj_pos = data.xpos[obj_id]

    offset = np.array([
        np.random.uniform(-0.3, 0.3),
        np.random.uniform(-0.3, 0.3),
        np.random.uniform(0.1, 0.3)
    ])
    target_pos = obj_pos + offset

    # IK to find qpos
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)

    ef_frame_body_id = model.body(ef_body_name).id

    # Define tasks
    end_effector_task = mink.FrameTask(
        frame_name=ef_body_name,
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    initial_rotation = mink.SO3(wxyz=data.xquat[ef_frame_body_id])

    T_wt = mink.SE3.from_rotation_and_translation(rotation=initial_rotation, translation=target_pos)
    end_effector_task.set_target(T_wt)
    posture_task.set_target_from_configuration(configuration)

    converge_ik(
        configuration,
        tasks,
        0.05,  # dt
        ik_solver,
        pos_threshold,
        ori_threshold,
        max_ik_iters,
    )

    q = configuration.q.copy()
    q[6] += np.random.uniform(-0.4 * np.pi, 0.4 * np.pi)
    return q[:7]
    # return configuration.q[:7].copy()  # First 7 are robot joints


def check_gripper_bowl_contact(env, bowl_name):
    robot = env.env.robots[0]
    gripper = robot.gripper
    if gripper is None:
        return False
    contacts = sim_utils.get_contacts(env.sim, gripper)
    for contact in contacts:
        bowl_model = env.env.objects_dict[bowl_name]
        if contact in bowl_model.contact_geoms:
            return True
    return False


def check_collisions(env, movable_objects, t):
    """Checks for collisions between the arm and objects, and between objects."""
    robot = env.env.robots[0]
    # Prepare list of models to check against objects: robot base/arm and gripper
    robot_parts = [robot.robot_model]
    if robot.gripper is not None:
        robot_parts.append(robot.gripper)

    current_collisions = set()

    # Helper to log and add collision
    def log_collision(entity1_name, entity2_name, t):
        collision_str = f"{entity1_name} collided with {entity2_name}"
        if collision_str not in current_collisions:
            logging.warning(f"Collision detected: {collision_str} at time {t}")
            current_collisions.add(collision_str)

    # Check robot parts vs objects
    for model in robot_parts:
        contacts = sim_utils.get_contacts(env.sim, model)
        for contact in contacts:
            for obj_name in movable_objects:
                obj_model = env.env.objects_dict[obj_name]
                if contact in obj_model.contact_geoms:
                    log_collision(f"robot({type(model).__name__})", obj_name, t)

    # Check object vs object
    for i, obj_name_1 in enumerate(movable_objects):
        obj_model_1 = env.env.objects_dict[obj_name_1]
        contacts_1 = sim_utils.get_contacts(env.sim, obj_model_1)
        for j, obj_name_2 in enumerate(movable_objects):
            if i >= j: # Avoid duplicate checks and self-checks
                continue
            obj_model_2 = env.env.objects_dict[obj_name_2]
            for contact in contacts_1:
                if contact in obj_model_2.contact_geoms:
                    log_collision(obj_name_1, obj_name_2, t)

    return list(current_collisions)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    #resize_size: int = 224
    replan_steps: int = 5
    repo_id: str = "local/libero_trajectories_in_lerobot"

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_object"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    #task_id: int = 1
    num_steps_wait: int = 15  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 500  # Number of rollouts per task; arm init cycles over DEFAULT + movable_objects
    data_collection_start_idx = 0
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = f"data/libero/videos/object/resampled_{num_trials_per_task}_dataset"  # Path to save videos
    data_saving_path: str = f"examples/libero/object_resampled_{num_trials_per_task}_dataset.hdf5"

    seed: int = 10  # Random Seed (for reproducibility) [seed 12 for train data, seed 10 for test data]


def collect_rollout(args: Args) -> None:
    # Set random seed
    # np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    # task_id = args.task_id

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 250  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port) # start policy client

    # Create collision log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    collision_log_path = pathlib.Path(args.video_out_path) / f"unexpected_collisions_{timestamp}.txt"

    total_episodes, total_successes = 0, 0
    stats_per_arm = collections.defaultdict(lambda: {"success": 0, "episodes": 0})

    # fixed_language_instruction = 'pick up the black bowl and place it on the plate'

    # Start episodes
    task_episodes, task_successes = 0, 0


    with h5py.File(args.data_saving_path, 'w') as hdf5_file:

        episode_counter = 0
        for task_id in [8]:

            task = task_suite.get_task(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed) # Initialize LIBERO environment and task description
            env.reset() # Perform explicit reset to load the scene and bodies to find movable objects

            # get the movable objects
            movable_objects = list(env.env.objects_dict.keys())
            # exclude basket_1
            movable_objects = [obj for obj in movable_objects if obj != 'basket_1']
            logging.info(f"Using movable objects: {movable_objects}")

            if not movable_objects:
                raise NotImplementedError('No movable objects found. Exit')
            
            arm_group_names_arr = ['DEFAULT'] + movable_objects  # around movable objects and the default position
            num_arm_groups = len(arm_group_names_arr)

            trial_id = args.data_collection_start_idx
            while trial_id < args.data_collection_start_idx + args.num_trials_per_task:
            # for trial_id in range(args.num_trials_per_task):
                # resample the init config of the arm for each trial
                env.reset() # Perform explicit reset to load the scene and bodies to find movable objects

                # wait for the env to fully initialize
                env_warmup_step_idx = 0
                # episode_collisions = set()
                while env_warmup_step_idx < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    env_warmup_step_idx += 1
                
                collisions = check_collisions(env, movable_objects, env_warmup_step_idx)
                if collisions:
                    logging.info('Collision, sample another env.')
                    continue
                    # episode_collisions.update(collisions)

                # solve for target arm init config
                arm_group_idx = trial_id % num_arm_groups
                arm_group_name = arm_group_names_arr[arm_group_idx]
                if arm_group_name != 'DEFAULT':
                    sol_q_pos = _get_arm_qpos_near_object(env, arm_group_name)
                else:
                    if trial_id // num_arm_groups == 0:
                        sol_q_pos = DEFAULT_QPOS + np.array([0, 0, 0, np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 0, np.random.uniform(-0.2 * np.pi, 0.2 * np.pi)])
                    else:
                        sol_q_pos = DEFAULT_QPOS + np.array([0, 0, 0, np.random.uniform(-0.25, 0.25), np.random.uniform(-0.25, 0.25), 0, np.random.uniform(-0.4 * np.pi, 0.4 * np.pi)])

                logging.info(f"\nTask: {task_description} | #{trial_id // num_arm_groups}th {arm_group_name} (arm group {arm_group_idx}) init config")
                
                action_plan = collections.deque()
                t = 0
                replay_images = []
                
                # Collect trajectory data
                trajectory_images = []
                trajectory_wrist_images = []
                trajectory_states = []
                trajectory_actions = []
                trajectory_joint_pos = []
                trajectory_joint_vel = []
                trajectory_gripper_pos = []
                trajectory_gripper_vel = []

                logging.info(f"Starting episode {task_episodes+1}...")
                data = env.sim.data._data
                data.qpos[:7] = sol_q_pos
                data.qvel[:7] = 0.0
                env.sim.forward()
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

                # Sustained contact resampling: for chocolate pudding task (task_id 8), goal object is chocolate_pudding_1 (from BDDL)
                goal_object_name = "chocolate_pudding_1"
                sustained_contact = False  # True only if we had contact every step in the last 20
                in_last_20_steps = False

                while t < max_steps:
                    try:
                        # Check for sustained contact in the last 20 steps
                        if t >= max_steps - 20:
                            sustained_contact = (sustained_contact if in_last_20_steps else True) and check_gripper_bowl_contact(env, goal_object_name)
                            in_last_20_steps = True

                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(img)
                        wrist_img = image_tools.convert_to_uint8(wrist_img)

                        # for data collection
                        state_vec = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )

                        # Save preprocessed image for replay video
                        new_img = img.copy()
                        cv2.putText(new_img, f"{t}: {task_description}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        replay_images.append(new_img)

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": state_vec,
                                "prompt": str(task_description),
                            }

                            # Query model to get action
                            action_chunk = client.infer(element)["actions"]
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Store trajectory data (actions and proprioception: joint angles and velocity)
                        trajectory_images.append(img)
                        trajectory_wrist_images.append(wrist_img)
                        trajectory_states.append(state_vec)
                        trajectory_actions.append(action)
                        trajectory_joint_pos.append(obs["robot0_joint_pos"].copy())
                        trajectory_joint_vel.append(data.qvel[:7].copy())
                        trajectory_gripper_pos.append(obs["robot0_gripper_qpos"].copy())
                        n_gripper = len(obs["robot0_gripper_qpos"]) # 2 for panda
                        trajectory_gripper_vel.append(data.qvel[7 : 7 + n_gripper].copy())

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.error(f"Caught exception: {e}")
                        break

                # Check for sustained contact failure case
                if not done and sustained_contact:
                    logging.warning(f"Episode failed but gripper maintained contact with {goal_object_name} for last 20 steps (indicates it was carrying and would succeed). Resampling...")
                    continue

                # Save trajectory to HDF5
                if len(trajectory_images) > 10:  # Only save if we collected data
                    ep_grp = hdf5_file.create_group(f"episode_{trial_id}")
                    
                    # Store observations
                    ep_grp.create_dataset(
                        "observations/image", 
                        data=np.array(trajectory_images), 
                        compression="gzip",
                        compression_opts=4
                    )
                    ep_grp.create_dataset(
                        "observations/wrist_image", 
                        data=np.array(trajectory_wrist_images), 
                        compression="gzip",
                        compression_opts=4
                    )
                    ep_grp.create_dataset(
                        "observations/state", 
                        data=np.array(trajectory_states)
                    )
                    ep_grp.create_dataset(
                        "observations/joint_pos",
                        data=np.array(trajectory_joint_pos),
                    )
                    ep_grp.create_dataset(
                        "observations/joint_vel",
                        data=np.array(trajectory_joint_vel),
                    )
                    ep_grp.create_dataset(
                        "observations/gripper_pos",
                        data=np.array(trajectory_gripper_pos),
                    )
                    ep_grp.create_dataset(
                        "observations/gripper_vel",
                        data=np.array(trajectory_gripper_vel),
                    )

                    # Store actions
                    ep_grp.create_dataset(
                        "actions",
                        data=np.array(trajectory_actions),
                    )
                    
                    # Store metadata as attributes
                    ep_grp.attrs["task_id"] = task_id
                    ep_grp.attrs["task_suite"] = args.task_suite_name
                    ep_grp.attrs["task_description"] = task_description
                    ep_grp.attrs["arm_config"] = arm_group_name
                    ep_grp.attrs["arm_idx"] = arm_group_idx
                    ep_grp.attrs["success"] = done
                    ep_grp.attrs["trajectory_length"] = len(trajectory_actions)
                    ep_grp.attrs["timestamp"] = datetime.datetime.now().isoformat()
                    
                    episode_counter += 1
                    logging.info(f"Saved episode {trial_id} to HDF5")
                else:
                    # Too few steps; retry same trial_id (no increment)
                    continue
                
                # Save a replay video of the episode
                suffix = "success" if done else "fail"
                filename = f"task_{task_id}_episode_{trial_id}_{suffix}.mp4"

                logging.info(f"saving video to {args.video_out_path}/{filename}")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / filename,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                trial_id += 1
                task_episodes += 1
                total_episodes += 1

                # if episode_collisions:
                #     logging.warning(f"Saving above collisions and video filename to {collision_log_path}")
                #     with open(collision_log_path, "a") as f:
                #         f.write(f"{filename} {list(episode_collisions)}\n")

                # Log current results
                logging.info(f"Success: {done}")
                
                # Update stats
                stats_per_arm[arm_group_name]["episodes"] += 1
                if done:
                    stats_per_arm[arm_group_name]["success"] += 1

                logging.info(f"# episodes completed so far: {total_episodes}")
                pct = (total_successes / total_episodes * 100) if total_episodes > 0 else 0.0
                logging.info(f"# successes: {total_successes} ({pct:.1f}%)")

        # Log final results
        task_rate = (float(task_successes) / float(task_episodes)) if task_episodes > 0 else 0.0
        total_rate = (float(total_successes) / float(total_episodes)) if total_episodes > 0 else 0.0
        logging.info(f"Current task success rate: {task_rate}")
        logging.info(f"Current total success rate: {total_rate}")

    total_rate = (float(total_successes) / float(total_episodes)) if total_episodes > 0 else 0.0
    logging.info(f"Total success rate: {total_rate}")
    logging.info(f"Total episodes: {total_episodes}")

    logging.info("\nSuccess Statistics per Arm State:")
    for arm_name, stats in stats_per_arm.items():
        rate = (stats["success"] / stats["episodes"] * 100) if stats["episodes"] > 0 else 0
        logging.info(f"  {arm_name}: {stats['success']}/{stats['episodes']} ({rate:.1f}%)")

    # Save success stats to a txt file
    stats_path = pathlib.Path(f"examples/libero/success_stats_{timestamp}.txt")
    overall_rate = (total_successes / total_episodes * 100) if total_episodes > 0 else 0.0
    with open(stats_path, "w") as f:
        f.write(f"Overall: {total_successes} / {total_episodes} ({overall_rate:.1f}%)\n")
        f.write("\nPer arm group (arm_group_idx, name):\n")
        for arm_group_idx, arm_name in enumerate(arm_group_names_arr):
            stats = stats_per_arm.get(arm_name, {"success": 0, "episodes": 0})
            rate = (stats["success"] / stats["episodes"] * 100) if stats["episodes"] > 0 else 0.0
            f.write(f"  arm_group_idx={arm_group_idx} {arm_name}: {stats['success']} / {stats['episodes']} ({rate:.1f}%)\n")
    logging.info(f"Wrote success stats to {stats_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    # assuming running from openpi folder
    # logging.info("ALTERED BDDLS FROM SPATIAL ONEBOWL")
    # task_bddl_file = f"./examples/libero/spatial_onebowl_bddls/{task.bddl_file}"
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    logging.info(f"task_bddl_file is {task_bddl_file}")
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    # env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(collect_rollout)
