Augmenting Demonstrations with Motion Planning to Help Vision-Language-Action Models Avoid Collision

Josh Bowden / jjosh

Summary:
From prior experience, state-of-the-art pi0.5 VLA sometimes knocks over non-target objects in the scene, which can lead to safety issues and poor performance. We are going to add obstacles in the way of pick-and-place tasks in the LIBERO robot simulation benchmark.  We will test robot policies trained with demonstrations from the original tasks without obstacles. We will augment the existing demonstrations using motion planning algorithms to route around the obstacles, then use those demonstrations to train a policy. Finally, we will test the original policy against the augmented policy, aiming for fewer collisions and higher task success.

https://github.com/user-attachments/assets/2973123b-7d19-4bd5-8a80-f0fdc818b92f



https://github.com/user-attachments/assets/10d2b0e7-071c-4a2c-84af-2d4be15f4cfc



Inputs and outputs:
Inputs are human demonstrations controlling a robot arm completing various pick-and-place tasks in simulation. Outputs are augmented demonstrations that are automatically routed around an obstacle using motion planning algorithms. Both inputs and outputs are used to train robot policies. 

Task list: 

Milestone 1

- (DONE) Clone openpi codebase for evaluating pi0.5 VLA in LIBERO simulation **(my code will be in ./examples/libero)**
- (DONE) Verify problem exists by observing pi0.5 VLA collide and knock over object, leading to task failures 
    (video above)
- (DONE) Run pi0.5 VLA in LIBERO simulation at scale, making sure we can use and modify the automated success detector,
detect object collisions, and add arbitrary obstacles to the environment
    (video, stats from 500 episodes, and script at episode_stats.txt and object_rollout_collection_script.py)

Milestone 2

- (DONE) Systematically evaluate pi0.5 VLA and/or a smaller model on the task with an obstacle in the way of the place phase. Result = 0% success, 100% collision. See avoid.py
- (DONE) Use simple heuristic of moving robot arm 1 meter to the right to avoid obstacle, using VLA for the rest and evaluate. This shows if motion planning would work. Result = 60% success, 0% collision. See baseline_IK.py. 

Motion planning can have higher success because we can depend less on the VLA succeeding from OOD states, although some combination is probably needed for more complex tasks.

Final submission

- Augment LIBERO demonstrations with more complex motion planning around an obstacle in the way of the pick phase
- Use augmented demonstrations to train pi0.5 VLA and/or a smaller model
- Evaluate augmented model on the task with an obstacle in the way of the pick phase
- Evaluate generalization to obstacles in different places
- Evaluate generalization to an obstacle in the way of the place phase

Stretch:
- Do more tasks and varying obstacles
- Augment demonstrations with obstacles in the place phase
- Try other simulation benchmarks or real arm

Expected deliverables and/or evaluation:
Ideally, a good policy will pick and place successfully and avoid colliding with other objects.
- A graph showing success of the base policy vs the augmented policy on the pick-and-place task
- A graph showing collision number of the base policy vs the augmented policy on the pick-and-place task
- Similar graphs showing success and collisions generalizing to when an obstacle is added in the place phase
- Video walkthrough of episodes showing success and failure modes

Risks: 
- Finetuning large VLA is too time/compute extensive
    - Derisk: I have experience finetuning pi0.5, and will use a smaller task-specific policy if prohibitive

- Challenges setting up motion planning on demonstrations
    - Derisk: I have experience using motion planners on other arms, and Panda in LIBERO is very well supported. Simulation allows for using ground truth obstacle coordinates. 

References:
https://fieldgen.github.io/  - augment grasp demonstrations with scripted movement towards the grasp position
https://lujieyang.github.io/physicsgen/ - transfer hand contact to other embodiments
https://www.alphaxiv.org/abs/2309.08821 - use motion planners for safer navigation

