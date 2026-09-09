AgileX Piper 配置与控制
=======================

.. figure:: https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/asserts/pictures/piper_urdf_zero.png
   :align: center
   :width: 70%

   Piper 机械臂与夹爪模型。图片来源：AgileX Robotics，piper_ros。

将 AgileX Piper 机械臂接入 RLinf，读取关节与夹爪状态，并通过机器人接口发送控制指令。先在本机连接 CAN 总线，按需加入相机，再参考任务扩展指南定义自己的真机任务。现有 reach 环境和 mock SAC 配置用于检查集成流程，尚无经过验证的任务或训练方案。

概览
----

RLinf 通过 ``pyAgxArm`` SDK 控制 Piper 机械臂及其 AgxGripper 夹爪。当前测试配置采用 MLP policy，通过 mock SDK 检查训练流程，无需物理硬件。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP policy（集成测试）

   .. grid-item-card:: 算法
      :text-align: center

      SAC（集成测试）

   .. grid-item-card:: 任务
      :text-align: center

      自定义任务；reach 示例框架

   .. grid-item-card:: 硬件
      :text-align: center

      Piper · CAN · AgxGripper

| **操作流程：** 安装依赖 → 配置 CAN → 读取反馈 → 测试关节与夹爪 → 新增任务。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Piper 硬件 · Linux 控制主机 · CAN 适配器。

任务
~~~~

先通过现有文件了解硬件与环境之间的 contract，再定义需要机器人学习的行为。

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - 用途
     - 入口
     - 范围
   * - 本机硬件操作
     - ``PiperRobot`` 与 ``PiperArm``
     - 读取状态，控制机械臂与夹爪。
   * - 环境示例框架
     - ``PiperReachEnv-v1``；``env/piper_reach.yaml``
     - 展示关节目标、复位和奖励的接入方式。
   * - 集成测试
     - ``piper_mock_sac_mlp_reach``
     - 使用 mock SDK 运行 SAC，不代表物理任务性能。

观测与动作
~~~~~~~~~~

机器人返回嵌套字典。reach 示例框架将这些数据转换为测试 policy 使用的一维状态和动作数组。

.. list-table::
   :header-rows: 1
   :widths: 23 77

   * - 字段
     - Contract
   * - Observation
     - ``arm.arm_joint_position``：6 个关节角，单位为弧度；``arm.tcp_pose``：基座坐标系中的 ``[x, y, z, qx, qy, qz, qw]``，位置单位为米；``arm.end_effector.state``：1 个夹爪开度，范围为 ``[0, 1]``。带夹爪时，示例框架的一维状态包含 14 个数值。
   * - Action
     - ``arm.joint_position``：6 个绝对关节目标，单位为弧度；``arm.end_effector.target``：1 个夹爪开度，``0`` 为闭合，``1`` 为全开。示例框架的一维动作包含 7 个数值，关节目标不是增量。
   * - Images
     - 相机可选。示例框架对每张图像居中裁剪并缩放为 ``(128, 128, 3)``，存入 ``frames.wrist_1``、``frames.wrist_2`` 等 key。没有相机时不包含 ``frames``；mock MLP 不使用图像。
   * - Reward
     - 示例框架按关节距离计算稠密奖励，或按逐关节容差计算稀疏奖励。实际任务需要自行定义奖励与成功条件。
   * - Prompt
     - 示例框架返回 ``reach a joint configuration``。

硬件配置
--------

先准备控制主机及其连接的机械臂，再安装软件。本机硬件检查在控制主机上运行；只有可选的 mock SAC 集成测试需要 GPU。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 要求
   * - 机械臂与夹爪
     - AgileX Piper 及其电源；若安装夹爪，则使用 AgxGripper。
   * - 控制主机
     - Linux 计算机、处于 SocketCAN 模式的 USB-to-CAN 适配器，以及连接机械臂的 CAN 线缆。
   * - 相机（可选）
     - 连接到控制主机的 Intel RealSense 相机，用于图像检查。
   * - GPU（可选）
     - mock SAC 测试需要一张受支持的 GPU，本机硬件检查无需 GPU。

将机械臂固定在稳定的台面上，并为小幅关节运动留出空间。上电前，按厂商说明连接电源与 CAN 线缆。

安装
----

在控制主机上按下方自定义环境流程安装机器人依赖。后续检查均在同一仓库目录和已激活的环境中运行。

机器人控制节点
~~~~~~~~~~~~~~

在连接机器人的主机上克隆 RLinf，然后安装对应环境。

.. include:: _setup_common.rst
   :end-before: 然后，使用

在仓库根目录安装机器人依赖：

.. code-block:: bash

   # Mainland China users can add --use-mirror for faster downloads.
   bash requirements/install.sh embodied --env piper
   source .venv/bin/activate
   export REPO_PATH="$PWD"

后续命令均在此仓库目录中运行，并保持环境已激活。安装脚本会安装 ``PiperArm`` 使用的固定版本 ``pyAgxArm`` 驱动；平台要求见 :doc:`安装 </rst_source/start/installation>`。接口示例和 mock MLP 测试均无需下载模型 checkpoint。

准备 CAN 连接
^^^^^^^^^^^^^

运行 ``ip -brief link`` 查找 SocketCAN 接口。处于 SocketCAN 模式的 USB-to-CAN 适配器通常显示为 ``can0``；如果名称不同，请替换下列命令中的接口名。连接机器人前，将接口配置为 1 Mbit/s：

.. code-block:: bash

   sudo ip link set can0 down
   sudo ip link set can0 type can bitrate 1000000
   sudo ip link set can0 up
   ip -details link show can0

输出中应显示接口已启用，且包含 ``bitrate 1000000``。机械臂上电后，运行 ``candump -n 5 can0`` 查看 5 帧反馈报文。如果一直没有报文，按 Ctrl+C 退出，检查电源、CAN 接线、适配器和波特率，再继续操作。SDK 不会配置 CAN 接口；适配器配置与硬件支持说明见 `pyAgxArm 文档 <https://github.com/agilexrobotics/pyAgxArm>`_。

运行
----

CAN 配置完成后，先在当前 Python 进程中使用机械臂。``PiperArm`` 声明记录设备参数，但不打开硬件；``PiperRobot`` 将这条机械臂命名为 ``arm``。夹爪与机械臂共用 CAN 连接，访问路径为 ``arm.end_effector``。

测试配置
~~~~~~~~

先在无硬件模式下检查测试工具：

.. code-block:: bash

   python -m toolkits.realworld_check.test_piper_controller --mock --read-only

命令打印 6 个实测关节角、包含 7 个数值的工具位姿以及 1 个夹爪开度，然后退出。mock 数值仅用于验证软件流程。随后对已连接的机械臂执行相同检查：

.. code-block:: bash

   python -m toolkits.realworld_check.test_piper_controller \
     --channel can0 --read-only

此命令使能电机并读取反馈，不发送位置目标，也不复位机械臂。日志中会显示检测到的固件 profile。若 AgxGripper 行程为 0.1 m，添加 ``--gripper-max-width 0.1``；未安装夹爪时，添加 ``--no-gripper``。

去掉 ``--read-only`` 即可测试小幅运动。逐条输入下列指令，每次运动结束后再继续：

.. code-block:: bash

   python -m toolkits.realworld_check.test_piper_controller \
     --channel can0 --speed-percent 10

.. code-block:: text

   where
   joint 1 2
   where
   joint 1 -2
   grip 0.5
   where
   quit

``joint 1 2`` 以关节 1 的实测位置为起点，增加 2 度。脚本接受关节编号 1 至 6，单次增量不超过 5 度；机械臂驱动还会按关节行程裁剪目标。``grip 0.5`` 将夹爪目标设为完整行程的一半。``where`` 读取反馈，夹爪停止后可再次执行。``quit``、EOF 或 Ctrl+C 关闭 CAN 会话，电机继续保持位置。添加 ``--mock`` 可在无硬件模式下演练同一组指令。

.. warning::

   即使指定 ``--read-only``，连接时也会使能电机。连接前清空运动区域，执行关节与夹爪指令时保持手部远离机器人。退出测试不等同于急停，也不会切断电机电源。

使用机器人接口
~~~~~~~~~~~~~~

在自己的 Python 程序中可使用以下接口示例。``connect()`` 打开连接并使能机械臂，``get_observation()`` 返回嵌套状态。示例读取当前关节位置和夹爪开度，再将这些数值作为目标发送回机器人。

.. code-block:: python

   from rlinf.robotics import PiperRobot
   from rlinf.robotics.parts.arms.piper import PiperArm
   from rlinf.utils.logging import get_logger

   logger = get_logger()
   robot = PiperRobot(arm=PiperArm.declare("can0", speed_percent=10))
   robot.connect()
   try:
       reading = robot.get_observation()["arm"]
       logger.info("Joint positions: %s", reading["arm_joint_position"])
       robot.send_action(
           {
               "arm": {
                   "joint_position": reading["arm_joint_position"],
                   "end_effector": {"target": reading["end_effector"]["state"]},
               }
           }
       )
   finally:
       robot.disconnect()

``send_action()`` 分发嵌套目标，返回已分发动作值的字典；如需测量执行结果，应再次读取观测。``disconnect()`` 释放机器人连接，包括机械臂与夹爪共用的连接。不指定 ``node_rank`` 时，上述示例在本机进程中运行，无需 Ray 集群。

``gripper_max_width`` 表示所装夹爪的完整行程，单位为米，默认值为 ``0.07``。没有夹爪时，声明 ``with_gripper=False``，并移除夹爪动作与观测访问。此时示例框架包含 6 个动作值和 13 个状态值。

加入相机
~~~~~~~~

如果需要同时读取图像和机械臂状态，可将上例的机器人声明替换为以下组合。``CameraInfo`` 描述物理相机，``Camera.declare()`` 返回按名称组织、尚未连接的相机零部件。机器人安装脚本在 x86 Linux 上包含 RealSense 支持。

通过 ``Camera.backend("realsense")`` 选择相机驱动，再调用 ``discover()`` 列出已连接相机的序列号：

.. code-block:: python

   from rlinf.robotics import Camera

   logger.info("RealSense serials: %s", Camera.backend("realsense").discover())

将下例中的 ``CAMERA_SERIAL`` 替换为列出的一个序列号，再连接并读取图像：

.. code-block:: python

   from rlinf.robotics import Camera, CameraInfo

   robot = PiperRobot(
       arm=PiperArm.declare("can0", speed_percent=10),
       **Camera.declare(
           {"wrist_1": CameraInfo(name="wrist_1", serial_number="CAMERA_SERIAL")}
       ),
   )
   robot.connect()
   try:
       image = robot.get_observation()["wrist_1"]["frame"]
       logger.info("Camera image shape: %s", image.shape)
   finally:
       robot.disconnect()

默认相机 backend 为 RealSense。接口返回相机配置分辨率下的图像，由环境负责缩放到 policy 输入所需的尺寸。机器人统一打开和关闭相机与机械臂；其他组合方式见 :doc:`机器人接口 </rst_source/concepts/robotics>`。

无硬件集成检查
~~~~~~~~~~~~~~

理解接口后，可在配有一张受支持 GPU 的主机上运行现有 mock 配置，检查 scheduler、环境、rollout 与 actor 的完整调用流程：

.. code-block:: bash

   bash tests/e2e_tests/embodied/run.sh piper_mock_sac_mlp_reach \
     runner.logger.log_path="$REPO_PATH/logs/piper_mock"

配置名包含 ``mock``，启动脚本因此启用 mock SDK。测试使用实际环境实现，设置 ``is_dummy: False``，接入两台假相机，并采用 ``action_dim: 7``、``obs_dim: 14`` 的 MLP。这只是短时软件集成检查，不会驱动物理机械臂，也不验证任务效果。

新增真机任务
------------

硬件检查通过后，参考 :doc:`新增真机任务 </rst_source/extending/new_task>` 定义任务观测、复位流程、奖励和成功条件。Piper 当前具备硬件接口与 reach 示例框架，尚无经过验证的真机任务方案。

从 ``rlinf/envs/real/piper/base.py`` 中的 ``PiperEnv`` 与 ``PiperEnvConfig`` 开始。相邻的 ``reach.py`` 展示了如何从 ``override_cfg`` 构造任务配置并传给基类构造函数；指南中的 ``CONFIG_CLS`` 简写仅适用于 Franka。将新任务类加入 ``rlinf/envs/real/piper/__init__.py`` 的 ``TASKS``，再复制 ``examples/embodiment/config/env/piper_reach.yaml``，通过 ``init_params.id`` 选择新注册的 Gymnasium ID。

硬件参数放在集群的 ``hardware`` 配置中，设置 ``type: Piper``；任务参数放在 ``env.train.override_cfg`` 中。现有 mock 配置展示了如何将 ``env`` 放置到携带机器人硬件的 node group。控制器位于另一台主机时，参考 :doc:`多节点训练 </rst_source/guides/multi_node>` 和 :doc:`机器人架构 </rst_source/concepts/robotics_architecture>`。当前没有已注册的遥操作设备能生成 Piper 所需的关节布局，添加兼容设备前请保持 ``teleop: none``。

可视化与结果
------------

mock 运行将 TensorBoard 日志写入 ``logs/piper_mock``，用于检查软件执行流程，不代表物理任务成功率。此示例尚无经过验证的任务结果或预训练任务权重。完成自己的任务后，可按 :doc:`训练指标 </rst_source/reference/metrics>` 和 :doc:`日志记录 </rst_source/guides/logger>` 配置日志与视频。
