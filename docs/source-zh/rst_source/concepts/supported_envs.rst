支持的模拟器、真机平台与模型
==============================

本页面总结 RLinf 支持的模拟器、真机平台和 VLA/WAM 模型，供具身强化学习使用。

支持的模拟器
------------

RLinf 通过标准化 RL 接口支持多种 GPU 和 CPU 模拟器。括号中的值用于
``env.train.env_type`` / ``env.eval.env_type``。

* **ManiSkill3** (``maniskill``) — GPU 并行化的机器人操作模拟器，提供丰富的任务和物体集合。
* **LIBERO** (``libero``) — 终身机器人学习基准，包含 130 个语言条件操作任务。
* **IsaacLab** (``isaaclab``) — NVIDIA Isaac Lab 高保真机器人仿真环境，支持 GR00T 工作流。
* **MetaWorld** (``metaworld``) — 经典机器人操作基准，包含 50 个桌面操作任务。
* **CALVIN** (``calvin``) — 长时域语言条件基准，4 自由度操作任务。
* **RoboCasa** (``robocasa``) — 大规模日常家庭操作任务仿真环境。
* **RoboTwin 2.0** (``robotwin``) — 双臂操作基准，包含 50 个多样化任务。
* **RoboVerse** (``roboverse``) — 统一仿真平台，集成多种环境和机器人形态。
* **FrankaSim** (``frankasim``) — Franka 机械臂仿真环境，支持 MLP/CNN 策略。
* **Behavior** (``behavior``) — 交互式仿真基准，包含复杂家庭活动任务。
* **EmbodiChain** (``embodichain``) — 面向链式操作任务的 Gym 环境。
* **Habitat** (``habitat``) — 具身导航与交互环境。
* **Genesis** (``genesis``) — GPU 加速的机器人与物理仿真环境。
* **D4RL** (``d4rl``) — 离线 RL 基准任务。
* **Polaris** (``polaris``) — Polaris 评测与具身训练环境。

各模拟器的具体训练示例见 :doc:`基于模拟器的具身强化学习 <../examples/simulators_index>`。

真机机器人
----------

RLinf 通过 ``realworld`` 与相关机器人集成支持真机 RL 训练：

* **Franka 机械臂** — Franka Research 3 机械臂，搭配 RealSense/ZED 相机和标准或 Robotiq 夹爪。
* **XSquare Turtle2** — 双臂机器人，支持 SAC + CNN 策略训练。
* **GimArm** — 6 自由度机械臂，SocketCAN 通信，基于 Pinocchio 的正向运动学。
* **Dexmal DOS-W1** — 双臂机器人，支持 flow-matching + SAC 抓取任务。
* **Franka + 灵巧手** — Franka 机械臂配合睿研五指灵巧手，实现复杂操作。

详细配置指南见 :doc:`../guides/realworld_robot` 和 :doc:`Franka 示例 <../examples/embodied/franka>`。

支持的具身模型
--------------

RLinf 支持以下 VLA 和具身策略模型。括号中的值用于 ``model.model_type``。

视觉-语言-动作（VLA）模型
~~~~~~~~~~~~~~~~~~~~~~~~~

* **OpenVLA** (``openvla``) — 7B 参数开源 VLA 模型，通用机器人操作。
* **OpenVLA-OFT** (``openvla_oft``) — 经过微调的 OpenVLA，通过 LoRA 适配提升特定任务性能。
* **π₀ / π₀.₅** (``openpi``) — Physical Intelligence 基于 flow-matching 的 VLA 模型。
* **GR00T** (``gr00t``、``gr00t_n1d6``、``gr00t_n1d7``) — NVIDIA 通用机器人控制 VLA 模型。
* **StarVLA** (``starvla``) — 具备时空推理能力的视觉-语言-动作模型。
* **Dexbotic** (``dexbotic_pi``、``dexbotic_dm0``) — 基于 π₀.₅ 风格架构的灵巧操作模型。
* **Lingbot-VLA** (``lingbotvla``) — 面向语言条件操作优化的 VLA 模型。
* **ABOT-M0** (``abot_m0``) — 面向机器人操作的具身 VLA 模型。
* **DreamZero** (``dreamzero``) — 面向具身策略学习的 SFT / world-model 工作流。

世界动作模型（WAMs）
~~~~~~~~~~~~~~~~~~~~

* **OpenSora** (``opensora_wm``) — 视频生成世界模型，用作 RL 训练的仿真器。
* **Wan** (``wan_wm``) — 大规模视频生成模型，用于基于世界模型的 RL。

策略网络
~~~~~~~~

* **MLP** (``mlp_policy``) — 简单多层感知机策略，用于状态空间 RL。
* **CNN** (``cnn_policy``) — 卷积神经网络策略，用于视觉 RL 任务。
* **Flow Policy** (``flow_policy``) — 用于连续控制的 flow-matching 策略。
* **CMA Policy** (``cma``) — CMA 风格策略基线。
* **ResNet Reward** (``resnet``) — 预训练 ResNet 模型，用于图像奖励建模。
* **Config / Value Models** (``cfg_model``、``value_model``) — 配置与 value modeling 路径的内部模型包装。

各模型的具体训练示例见 :doc:`VLA / WAM 模型的强化学习 <../examples/vla_wam_index>`。
