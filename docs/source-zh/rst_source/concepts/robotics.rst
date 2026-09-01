机器人接口
============

RLinf 通过具名零部件提供统一的观测和动作接口。机械臂、末端执行器、相机和移动底盘等零部件都有稳定的访问路径；观测按这些路径组织，可控零部件的动作也使用相同路径。

本页介绍任务和 env 实际使用的机器人接口：先读取和控制已支持的机器人，再说明访问路径如何反映硬件连接关系，以及本地与跨节点部署为何共用一套调用方式。硬件 session、资源归属和 worker placement 等实现细节见 :doc:`机器人架构 <robotics_architecture>`。

使用机器人接口
------------------

构建机器人后，可在打开硬件前检查其零部件路径和部署位置。实际运行时，建议将 ``disconnect()`` 置于 ``finally`` 块中：

.. code-block:: python

   from rlinf.robotics import Arm, Camera, build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # 机器人未上电或尚不可达时也可以执行。
   print(robot.describe())

   arm = robot.child("arm", Arm)
   cameras = robot.parts_of_type(Camera)

   robot.connect()
   try:
       if not arm.is_robot_up():
           raise RuntimeError("The arm is connected but not ready.")
       if not all(camera.is_ready() for camera in cameras.values()):
           raise RuntimeError("A camera is not delivering frames.")

       observation = robot.get_observation()
       tcp_pose = observation["arm"]["tcp_pose"]
       gripper_state = observation["end_effector"]["state"]

       robot.send_action(
           {
               "arm": {"tcp_pose": target},
               "end_effector": {"target": width},
           }
       )
   finally:
       robot.disconnect()

``get_observation()`` 会读取一次整个机器人，并按零部件路径返回嵌套字典。``send_action()`` 接收层级相同的动作字典；每次只需指定本次要控制的分支。

初始化或复位逻辑需要调用某类零部件的通用方法时，使用 ``child(name, ExpectedType)``。该方法会立即检查实际类型，编辑器也能据此推断返回值。例如，所有 ``Arm`` 都提供 ``is_robot_up()``、``clear_errors()`` 和 ``reset_joint()``，不受 backend 或 placement 影响。如果任务需要遍历所有相机、但不依赖具体名称，可使用 ``parts_of_type(Camera)``。

按路径访问零部件
--------------------

零部件名称定义对外数据接口。单臂 Franka 的 ``arm`` 和 ``end_effector`` 位于同一层级，因为机械臂和 Franka Hand 分别打开独立端点：

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

``via`` 不同表示两个零部件各自持有 connection，因此可以分别调整部署位置、恢复连接或替换实现。

有些硬件在连接层面不可分开。SO-101 的五个机械臂关节和夹爪共用一条伺服总线，因此末端执行器位于机械臂下一级，两条路径的 connection 相同：

.. code-block:: text

   SO101Robot
   └── arm               SO101Arm             node=0     via SO101Arm#1
       └── end_effector  MethodEndEffector    node=0     via SO101Arm#1

观测和动作也遵循这一层级：

.. code-block:: python

   joint_position = observation["arm"]["arm_joint_position"]
   gripper_state = observation["arm"]["end_effector"]["state"]

   robot.send_action(
       {
           "arm": {
               "joint_position": joint_target,
               "end_effector": {"target": gripper_target},
           }
       }
   )

``describe()`` 中的 ``via`` 适合用于排查组合和资源归属问题，日常任务代码只需使用访问路径和对应数据。完整输出字符串不是稳定的序列化格式，不应存储或解析。

组合零部件时保持接口一致
----------------------------

能够返回观测的零部件可以直接加入机器人。例如，``Robot(base=base, arm=arm, end_effector=hand)`` 中的三个关键字参数名会分别成为访问路径。如果共享控制器只是 ``Connection``、本身不能观测，则需先通过 ``session.part("left")`` 取出它支持的可读零部件。

一个零部件也可以承载其他零部件，这些零部件会自动出现在它的下一级。因此，SO-101 的夹爪使用 ``arm.end_effector`` 路径。双臂机器人可以在外层增加 ``left`` 和 ``right`` 两个 group，内部仍使用相同的零部件接口：

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

RLinf 不预设 ``arms`` 或 ``cameras`` 等固定字段。新增移动底盘、升降机构、云台或第三条机械臂时，只需为它选择稳定名称，无需再增加一套机器人 API。

在 env 中统一读写机器人
------------------------

env 应通过组合后的机器人读写硬件，不应直接访问厂商 driver。每次调用 ``robot.get_observation()`` 都会为 env 提供一份完整读取结果，用于构造 policy 需要的 ``state`` 和 ``frames``。在同一次读取中，共享 connection 的零部件会复用同一份底层状态快照。

对于不属于单步动作流的操作，可以直接使用具体零部件。当前真机 env 通过类型明确的 ``Arm`` 检查就绪状态、恢复错误并复位关节；相机则通过 ``parts_of_type(Camera)`` 查找，其 placement 和生命周期仍由机器人负责。

现有 policy 如果仍使用扁平向量，可在 env 边界通过 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter`` 转换。机器人接口本身仍保留具名嵌套结构。

任务代码不依赖部署位置
----------------------

访问路径不包含零部件的运行位置。位于 env 进程的相机和运行在其他节点的机械臂，仍出现在同一份观测中。使用不同 connection 的零部件可以并行读写；共享 connection 的分支则按声明顺序执行，避免并发访问不支持该模式的厂商 SDK。

因此，任务和 policy 代码只依赖零部件名称及其数据，无需感知 Ray actor、RPC、串口或厂商 session。调整 connection 的部署位置时，不需修改任务接口。

分离硬件能力与任务逻辑
----------------------

零部件负责读取传感器或控制执行器，不判断 rollout 是否成功。奖励、终止条件、任务特有的复位流程和 Gymnasium space 应由 ``RobotTask`` 或具体的真机 env 定义；遵循通用任务接口时，可使用 ``RobotTaskEnv`` 组合任务与机器人。

这一边界使同一台机器人可以运行多个任务，也使任务在硬件调整到其他节点后仍能保持 policy 侧数据格式稳定。

后续阅读
--------

- :doc:`新增真机任务 <../extending/new_task>`：在已支持的真机上添加任务。
- :doc:`添加机器人 <../extending/new_robot>`：接入本地传感器、执行器或整台机器人。
- :doc:`机器人架构 <robotics_architecture>`：了解 ``parts`` 与 ``children`` 的区别，以及共享 connection、``PartGroup``、资源生命周期和 worker placement。
- :doc:`遥操作 <../guides/teleoperation>`：组合多种操作者设备。
