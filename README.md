# Auto-Augmenting Robot Manipulation Demonstrations to Learn to Avoid Collisions

**Joshua Bowden, jjosh**
**June 3, 2026**

---

## Background and Setup

Current state-of-the-art robot manipulation demos are very good at completing tasks given that they have an ideal setup for the task. Critical inspection of the robot environment reveals that there are no humans or obsatcles in the way; at most there are distracting objects on the table. As robots move from robot-oriented spaces, like a factory specially designed for them, to human spaces, they need to be aware of and able to avoid collisions with humans and the environment. This work is partly inspired by my day job in automation at a therapeutics company where we are trying to move robots from a dedicated workcell (instruments are lined up for an enclosed arm that memorizes the exact position to operate each one) to the wet lab where there is a dynamic environment, scientists, and $100k instruments.

<img width="500" height="300" alt="pi" src="https://github.com/user-attachments/assets/ef8ecbe9-d35e-4114-b23c-3dfa763a8cfa" />


When we step back from manipulation and look at robotics as a whole, we notice that there is a whole class of robots primarily focused on navigation and obstacle avoidance. Localization and mapping enables turtlebots, robot dogs, drones, and self-driving cars to move in their environment while building an explicit map and understanding what space they can occupy [1]. These robots do this both with traditional methods, such as LIDAR mapping, and with learned methods, like 3d reconstruction and Gaussian splatting for creating the environment and learned models for moving in the environment, like a self-driving car or simpler controllers.

<img width="400" height="180" alt="bot" src="https://github.com/user-attachments/assets/7ca53b7a-6d2a-4b38-a8de-cce428b64f05" />


Meanwhile, manipulators have a traditional form of obstacle avoidance known as motion planning, where the joints of the arm are checked against a 3D map of the world while trying to go from a given pose to a target pose. However, the field does not seem to have thought about representing this idea in learning-based models. Robotics foundation models are along a spectrum with fully latent understanding on one side, and on the other side are attempts to pull out explicit understanding of things like object recognition, object pose, or grasp pose [3]. But mapping of the environment and the position of the whole arm (joints and end effector) is missing from this spectrum.

<img width="206" height="156" alt="trad" src="https://github.com/user-attachments/assets/263d24fc-b386-46b7-821b-751052b0a96d" />

To formulate the problem of manipulation and avoidance, we define the goal as training a robot arm to avoid obstacles while completing a separate pick-and-place manipulation task. Our metrics are pick-and-place success rate and obstacle collision rate. The inputs are 1) expert pick-and-place demonstrations in simulation, which are a plentiful source of robotics data, and 2) a motion planner, which calculate a path from pose A to B while avoiding obstacles. THe outputs are 1) augmented demonstrations that complete the task and avoid an inserted obstacle, and 2) a trained robot policy that does well on the metrics.

See below for two examples of pi0.5 VLA [4] failing.

<img width="400" height="400" alt="agent_task_1_episode_0_fail" src="https://github.com/user-attachments/assets/24bae077-48a0-4244-abfe-5280eeed6582" />
<img width="400" height="400" alt="pi05_libero_collision" src="https://github.com/user-attachments/assets/f97ecc1f-8cf8-43e8-8cf8-711cf9b33749" />


The hard part is setting up scalable augmentations and getting a model to generalize obstacle avoidance.

## Approach 

**Scalably Augmenting Demonstrations**
We want to augment existing demonstrations in order to increase the amount of data for this task of obstacle avoidance while avoiding the very expensive costs of collecting data. Demonstrations are very important for manipulation because manipulaiton behaviors are very hard to guide out of an exploring RL agent; as opposed to more navigation based tasks like traversing a maze where a simple explore/exploit agent can often achieve success.

To augment demonstrations, we recreate the simulation environment from the original demonstration and add an obstacle. We break the original robot trajectory into key poses: start, pick, and place. Then, we change the intermediate path with a motion planner. The idea is that this simple approach captures the manipulation control, which is the important part of a human demonstration, and rewrites the motion between those segments, which is less important.

I tried to implement my own simple motion planner in simulation, then tested several old and nonfunctional planing libraries for the Mujoco simulator before finding one that worked. While we built off this planning library designed broadly for the simulator, key implementation challenges were getting our robot model and obstacles to format correctly, approximating the joint space poses from the planner into tool-space delta actions that the simulator wanted, and interpolating the sparse poses from the planner to have a dense path. 

**Training a model to learn avoidance**
We want to see if the traditional motion planner, which depends on privileged information from the simulator like the ground truth location of obstacles, can be learned into a robot policy while it is also learning to complete another task. We use the successful, non-colliding demonstrations from the augmentation, while ignoring augmentations that didn't go too well. Architecture-wise, we used basic behavior cloning from the LIBERO benchmark [7] we are testing on and didn't do anything new. 



## Evaluation and Results

We are trying to augment demonstrations scalably to avoid obstacles and complete a pick-and-place task, then train a model to do the same. To do this, we need to test if our augmentation process manages to succeed and avoid collision, and then test our trained model in those aspects as well.

**Augmenting Demonstrations Results**
Starting from 50 demonstrations with randomized obstacles inserted, half of them returned a valid motion plan. An invalid motion plan means that the inserted obstacle may have made it impossible to reach the key poses. Getting this signal allows for (even in case of failure to plan) randomly varying the obstacle, checking for a motion plan, and iterating without manually tuning the environment and obstacle too much. This could also allow future work to vary between simple and complex obstacle arrangements using heuristics like number of samples for the motion planner to reach a solution. 

Of the amount with a valid motion plan, 60% were both successful and avoided hitting any obstacles. 36% were unsuccessful due to unwanted change in the arm position during key poses that stem from our lossy conversion of motion planner outputs to arm control inputs. Only 4% had a collision, with similar causes. This is promising because we did not tune the motion planner at all, and it shows that we could easily (and algorithmically) change parameters like clearance distance, interpolation methods, and constraints that are typical when using motion planners, and get more successful no-collision trajectories by using a bit more compute. As a note, the motion planner is very fast and runs in a few seconds on a CPU for a few hundred steps of robot motion, so the scaling cost is very low. 

See below for an original demonstration and the same demonstration augmented.

<img width="400" height="400" alt="ep_00_agent_replay" src="https://github.com/user-attachments/assets/4fc2f860-05a7-4bbb-9566-0914fd859e0a" />

<img width="400" height="400" alt="ep_00_agent" src="https://github.com/user-attachments/assets/d8cae979-ba2a-4f41-b9f4-cefc33dcff9f" />

**Model Eval**
Our model performance was pretty poor. Evaluated on obstacles with a similar placement as the training data(between start and pick), 22% of trajectories succeeded and 0% collided. Evaluated on different obstacles (between pick and place), 0% succeeded and 98% collided. We will discuss more below.

See below for a success on a similar obstacle and complete failure on a different obstacle.

<img width="400" height="400" alt="ep_13_agent" src="https://github.com/user-attachments/assets/e70877cc-843d-4ce6-ba48-8181b2f73ef8" />

<img width="400" height="400" alt="ep_25_agent" src="https://github.com/user-attachments/assets/a474f09e-392c-424a-bdb4-839165ba2fb3" />

## Discussion

It seems to be extremely hard to learn avoidance alongside manipulation. The behavior cloning that we used has obvious flaws like averaging multimodal actions, but it is widely used for manipulation again because manipulation is hard to explore. Another learning method, maximum entropy RL [2], has some ability to be robust to obstacles because it tries to complete a task in as random of a way as possible, but avoidance is only by chance and not from actual understanding.

See below for max entropy trained with no obstacle and evaluated without an obstacle [2].

<img width="600" height="450" alt="maxent" src="https://github.com/user-attachments/assets/60301e82-e076-4584-8113-23c25ae70519" />


Learning a negative task of avoidance next to a positive task of manipulation seems very hard. Mobile robots are able to focus on navigation, like a self driving car, a wheeled bot that navigates and takes pictues, or a drone that navigates and drops a delivery. Manipulation robots dynamically change their collision body throughout a task, so the problem seems quite different and challenging. 

Going forward, the spectrum seems to go from a traditional motion planner with an explicit world map that rejects actions if they would cause collision, to the other extreme of physical AI that knows its own limitations and latently avoids hitting anything. It's unclear where the field will move forward as robots come into more contact with humans, and I foresee cheap heuristics like not moving while people are around or if the environment is changing too much.

## References

[1] Borquez, Javier, et al. "On Safety and Liveness Filtering Using Hamilton-Jacobi Reachability Analysis." arXiv, 2024, https://arxiv.org/abs/2312.15347.

[2] Eysenbach, Benjamin, and Sergey Levine. "Maximum Entropy RL (Provably) Solves Some Robust RL Problems." arXiv, 2022, https://arxiv.org/abs/2103.06257.

[3] Murali, Adithyavairavan, et al. "GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training." arXiv, 2025, https://arxiv.org/abs/2507.13097.

[4] Physical Intelligence, et al. "π0.5: A Vision-Language-Action Model with Open-World Generalization." arXiv, 2025, https://arxiv.org/abs/2504.16054.

[5] Spitznagel, Martin, Jan Vaillant, and Janis Keuper. "PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?" arXiv, 2025, https://arxiv.org/abs/2503.05333.

[6] Wang, Wenhao, et al. "FieldGen: From Teleoperated Pre-Manipulation Trajectories to Field-Guided Data Generation." arXiv, 2025, https://arxiv.org/abs/2510.20774.

[7] Liu, Bo, et al. "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning." arXiv, 2023, https://arxiv.org/abs/2306.03310.







