Simulators, Robots, and Models
==============================

This page summarizes the simulators, real-world robotic platforms, and VLA/WAM models
supported by RLinf for embodied reinforcement learning.

Supported Simulators
--------------------

RLinf supports a wide range of GPU and CPU-based simulators through
standardized RL interfaces. Use the value in parentheses as
``env.train.env_type`` / ``env.eval.env_type``.

* **ManiSkill3** (``maniskill``) — GPU-parallelized robotic manipulation simulator with diverse tasks and object sets.
* **LIBERO** (``libero``) — Lifelong Robot Learning benchmark with 130 language-conditioned manipulation tasks.
* **IsaacLab** (``isaaclab``) — NVIDIA Isaac Lab for high-fidelity robot simulation, including GR00T workflows.
* **MetaWorld** (``metaworld``) — Classic robotic manipulation benchmark with 50 distinct tabletop tasks.
* **CALVIN** (``calvin``) — Long-horizon language-conditioned benchmark with 4-DOF manipulation.
* **RoboCasa** (``robocasa``) — Large-scale simulation of daily household manipulation tasks.
* **RoboTwin 2.0** (``robotwin``) — Dual-arm manipulation benchmark with 50 diverse tasks.
* **RoboVerse** (``roboverse``) — Unified simulation platform integrating multiple environments and embodiments.
* **FrankaSim** (``frankasim``) — Franka arm simulation environment with MLP/CNN policy support.
* **Behavior** (``behavior``) — Interactive simulation benchmark with complex household activities.
* **EmbodiChain** (``embodichain``) — Gym-style environment for chain-based manipulation tasks.
* **Habitat** (``habitat``) — Embodied navigation and interaction environments.
* **Genesis** (``genesis``) — GPU-accelerated robotics and physics simulation.
* **D4RL** (``d4rl``) — Offline RL benchmark tasks.
* **Polaris** (``polaris``) — Polaris evaluation and embodied training environments.

For simulator-specific training examples, see :doc:`../examples/simulators_index`.

Real-World Robotics
-------------------

RLinf supports real-world RL training through ``realworld`` and related robot
integrations:

* **Franka Arm** — Franka Research 3 robotic arm with RealSense/ZED cameras and standard or Robotiq grippers.
* **XSquare Turtle2** — Dual-arm robot with SAC + CNN policy training.
* **GimArm** — 6-DOF robotic arm with SocketCAN communication and Pinocchio-based forward kinematics.
* **Dexmal DOS-W1** — Dual-arm robot with flow-matching + SAC pick tasks.
* **Franka + Dexterous Hand** — Franka arm combined with Ruiyan five-finger dexterous hand for complex manipulation.

For detailed setup guides, see :doc:`../guides/realworld_robot` and the :doc:`Franka examples <../examples/embodied/franka>`.

Supported Embodied Models
--------------------------

RLinf supports the following VLA and embodied policy models. Use the value in
parentheses as ``model.model_type``.

Vision-Language-Action (VLA) Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **OpenVLA** (``openvla``) — 7B-parameter open-source VLA model for general-purpose robot manipulation.
* **OpenVLA-OFT** (``openvla_oft``) — Fine-tuned OpenVLA with LoRA adaptation for improved task-specific performance.
* **π₀ / π₀.₅** (``openpi``) — Flow-matching based VLA models from Physical Intelligence.
* **GR00T** (``gr00t``, ``gr00t_n1d6``, ``gr00t_n1d7``) — NVIDIA VLA models for generalist robot control.
* **StarVLA** (``starvla``) — Vision-language-action model with spatial-temporal reasoning.
* **Dexbotic** (``dexbotic_pi``, ``dexbotic_dm0``) — Dexterous manipulation models based on π₀.₅-style architectures.
* **Lingbot-VLA** (``lingbotvla``) — VLA model optimized for language-conditioned manipulation.
* **ABOT-M0** (``abot_m0``) — Embodied VLA model for robot manipulation.
* **DreamZero** (``dreamzero``) — SFT / world-model workflow for embodied policy learning.

World Action Models (WAMs)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **OpenSora** (``opensora_wm``) — Video generation world model used as a simulator for RL training.
* **Wan** (``wan_wm``) — Large-scale video generation model for world-model-based RL.

Policy Networks
~~~~~~~~~~~~~~~

* **MLP** (``mlp_policy``) — Simple multi-layer perceptron policy for state-based RL.
* **CNN** (``cnn_policy``) — Convolutional neural network policy for visual RL tasks.
* **Flow Policy** (``flow_policy``) — Flow-matching policy for continuous control.
* **CMA Policy** (``cma``) — CMA-style policy baseline.
* **ResNet Reward** (``resnet``) — Pretrained ResNet models for image-based reward modeling.
* **Config / Value Models** (``cfg_model``, ``value_model``) — Internal model wrappers for configuration and value modeling paths.

For model-specific training examples, see :doc:`../examples/vla_wam_index`.
