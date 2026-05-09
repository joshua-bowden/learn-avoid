# from libero.libero import benchmark
import numpy as np
import sys, os
from main import _get_libero_env
from libero.libero import benchmark
import time


import mujoco
import mink # for ik computation


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# IK parameters
SOLVER = "quadprog"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 30
EF_BODY_NAME = 'gripper0_eef'


# def visualize_mj_model(mj_model, mj_data):
#     # mj_data = mujoco.MjData(mj_model)
#     mujoco.mj_forward(mj_model, mj_data)
#     with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
#         print("Press ESC to exit viewer.")
#         # Keep the viewer open
#         while viewer.is_running():
#             # Step the simulation forward
#             mujoco.mj_step(mj_model, mj_data)
#             viewer.sync()
#             time.sleep(0.01)  # Small delay to control frame rate

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
        print('idx # ', idx, 'pos error = ', err[:3], np.linalg.norm(err[:3]))

        if pos_achieved and ori_achieved:
            return True
    return False




def ik_example_main():
    task_suite_name = 'libero_spatial'
    task_id = 1

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    
    env, task_description = _get_libero_env(task, 224, 2)
    sim = env.sim
    model = sim.model._model # type mj model
    data = sim.data._data # type mjdata

    mujoco.mj_forward(model, data)
    
    # print("Bodies in model:")
    # for i in range(model.nbody):
    #     print(f"id={model.body(i).id}, name={model.body(i).name}, x pos = {data.xpos[model.body(i).id]}")

    base_link_pos = data.xpos[model.body(2).id]

    # mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    # configuration.update(data.qpos)
    # posture_task.set_target_from_configuration(configuration)

    ef_target_position = base_link_pos + np.array([0.5, 0.2, 0.5])
    mujoco.mj_forward(model, data)

    ef_frame_body_id = model.body(EF_BODY_NAME).id

    # print('current ef frame xpos', data.xpos[ef_frame_body_id])
    # exit()

    configuration = mink.Configuration(model)
    # Define tasks - using the hand body as end-effector
    end_effector_task = mink.FrameTask(
        frame_name=EF_BODY_NAME,
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    initial_rotation = mink.SO3(wxyz=data.xquat[ef_frame_body_id])

    T_wt = mink.SE3.from_rotation_and_translation(
                rotation=initial_rotation,
                translation=ef_target_position
            )
    end_effector_task.set_target(T_wt)
    posture_task.set_target_from_configuration(configuration)

    dt = 0.05
    converge_ik(
                configuration,
                tasks,
                dt,
                SOLVER,
                POS_THRESHOLD,
                ORI_THRESHOLD,
                MAX_ITERS,
            )
    
    sol = configuration.q
    sol_q_pos = sol[:7]
    print('solution q pos = ', sol_q_pos)

    data.qpos[:7] = sol_q_pos
    mujoco.mj_forward(model, data)

    ef_xpos = data.xpos[ef_frame_body_id]
    print("End-effector position:", ef_xpos)
    print("Target position:", ef_target_position)
    print("Position error:", np.linalg.norm(ef_xpos - ef_target_position))

    
    

    
    # print("Bodies in model:")
    # for i in range(model.nbody):
    #     print(f"id={model.body(i).id}, name={model.body(i).name}")
    # exit()

    # Create a Mink configuration
    # configuration = mink.Configuration(model)


    


    # visualize_mj_model(model, data)


if __name__ == "__main__":
    ik_example_main()