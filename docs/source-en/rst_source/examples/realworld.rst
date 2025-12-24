Real-World RL with Franka
============================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide to launching and managing the 
a CNN policy training task within the RLinf framework, 
focusing on training a ResNet-based CNN policy from scratch for robotic manipulation in the real world setup. 

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Action Generation**: Producing precise robotic actions (position, rotation), possibly with gripper control.
3. **Reinforcement Learning**: Optimizing the policy via the SAC with environment feedback.

Environment
-----------

**Real World Environment**

- **Environment**: Real world setup.
  - Franka Emika Panda robotic arm
  - Realsense cameras
  - Possibly use spacemouse for teleoperation data collection or human intervention.
- **Task**: Currently we support the peg-insertion task and the charger task. 
- **Observation**: 
    - RGB images (128x128) from a wrist camera or a third-person camera.
- **Action Space**: 6 or 7-dimensional continuous actions, depending on whether gripper control is included:
  - 3D position control (x, y, z)
  - 3D rotation control (roll, pitch, yaw)
  - Gripper control (open/close)

**Data Structure**

- **Images**: RGB tensors ``[batch_size, 128, 128, 3]``
- **Actions**: Normalized continuous values ``[-1, 1]`` for each action dimension
- **Rewards**: Step-level rewards based on task completion


Algorithm
-----------------------------------------

**Core Algorithm Components**

1. **SAC (Soft Actor-Critic)**

   - Learning Q-values by Bellman backups and entropy regularization.

   - Learning policy to maximize entropy-regularized Q.

   - Learning temperature parameter for exploration-exploitation trade-off.

2. **Cross-Q**

   - A variant of SAC that removes the target Q network.

   - Concating curr-obs and next-obs in one batch, incorporating BatchNorm for stable training for Q.

3. **RLPD (Reinforcement Learning with Prior Data)**

   - A variant of SAC that incorporates prior data for improved learning efficiency.

   - High update-to-data ratio to leverage collected data effectively.

4. **CNN Policy Network**

   - ResNet-based architecture for processing visual inputs.

   - MLP layers for fusing images and states to output actions.

   - Q heads for critic functions.

Dependency Installation
-----------------------
TODO

Visualization and Results
-------------------------

**1. Tensorboard Logging**

At the ray head node, run:

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. Key Metrics Tracked**

- **Environment Metrics**:

  - ``env/episode_len``: Number of environment steps elapsed in the episode (unit: step).
  - ``env/return``: Episode return.
  - ``env/reward``: Step-level reward.  
  - ``env/success_once``: Recommended metric to monitor training performance. It directly reflects the unnormalized episodic success rate.

- **Training Metrics**:

  - ``train/sac/critic_loss``: Loss of the Q-function.
  - ``train/critic/grad_norm``: Gradient norm of the Q-function.

  - ``train/sac/actor_loss``: Loss of the policy.
  - ``train/actor/entropy``: Entropy of the policy.
  - ``train/actor/grad_norm``: Gradient norm of the policy.

  - ``train/sac/alpha_loss``: Loss of the temperature parameter.
  - ``train/sac/alpha``: Value of the temperature parameter.
  - ``train/alpha/grad_norm``: Gradient norm of the temperature parameter.

  - ``train/replay_buffer/size``: Current size of the replay buffer.
  - ``train/replay_buffer/max_reward``: Maximum reward stored in the replay buffer.
  - ``train/replay_buffer/min_reward``: Minimum reward stored in the replay buffer.
  - ``train/replay_buffer/mean_reward``: Average reward stored in the replay buffer.
  - ``train/replay_buffer/std_reward``: Standard deviation of rewards stored in the replay buffer.
  - ``train/replay_buffer/utilization``: Utilization rate of the replay buffer.

Real World Results
~~~~~~~~~~~~~~~~~~
Here we provide demo videos and training curves for the task peg-insertion and charger task, respectively. Within 1 hour of training, the robot is able to learn a policy that can continuously successfully complete the task.

.. raw:: html

  <div style="flex: 0.8; text-align: center;">
      <img src="https://github.com/RLinf/misc/raw/main/pic/realworld-curve.png" style="width: 100%;"/>
      <p><em>Training Curve</em></p>
    </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://github.com/RLinf/misc/raw/main/pic/peg-insertion-compressed.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>Peg Insertion</em></p>
  </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://github.com/RLinf/misc/raw/main/pic/charger-compressed.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>Charger</em></p>
  </div>