机器人接口
============

本页说明任务和 env 如何通过机器人接口检查、连接、读取并控制真机。内容先按照实际调用顺序介绍如何构建和检查机器人、按类型获取零部件、管理连接和读写数据，再解释访问路径如何产生、组合为何不改变接口，以及 env 应在哪一层转换机器人数据。硬件 session、资源归属和 worker placement 等实现机制见 :doc:`机器人架构 <robotics_architecture>`。

机器人接口为机械臂、末端执行器、相机和移动底盘等零部件提供稳定的名称和访问路径，观测和动作沿用同一组路径。任务侧无需直接依赖具体 driver；只要路径保持不变，更换 backend 或调整部署节点也无需修改调用代码。

使用机器人接口
------------------

一次完整调用可以分为四个阶段：构建并检查机器人，取得初始化所需的零部件，连接后读写数据，最后释放硬件。下面的示例将这四个阶段放在同一段代码中：

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

       applied = robot.send_action(
           {
               "arm": {"tcp_pose": target},
               "end_effector": {"target": width},
           }
       )
   finally:
       robot.disconnect()

``build_robot()`` 根据注册名找到对应的机器人 builder，并返回尚未连接的 ``Robot``。构建阶段只记录硬件参数和 placement，不导入厂商 SDK，也不打开设备。``describe()`` 读取这些声明，因此机器人尚未上电或网络不可达时，也能检查访问路径、部署节点和连接归属。

初始化代码随后取得所需能力。``child("arm", Arm)`` 返回 ``arm`` 路径上的零部件，并立即检查它是否属于 ``Arm``；编辑器也会将返回值推断为 ``Arm``。``parts_of_type(Camera)`` 则遍历完整组合，返回以完整路径为 key 的所有相机。任务依赖固定路径时使用 ``child()``；只关心设备类别而不依赖相机名称时，使用 ``parts_of_type()``。

``connect()`` 会为每条 owner connection 打开一次资源；如果后续 connection 打开失败，它会回滚此前已经打开的资源。连接完成后，可通过 ``Arm.is_robot_up()`` 和 ``Camera.is_ready()`` 检查设备是否可用；``clear_errors()`` 与 ``reset_joint()`` 等初始化操作位于单步动作流之外。

step 循环只需两个接口。``get_observation()`` 对组合后的机器人执行一次完整读取，并按访问路径返回嵌套字典。``send_action()`` 接收层级相同的动作，只下发本次提供的分支，并将各零部件实际发送的动作作为 ``applied`` 返回。``disconnect()`` 最后按相反顺序关闭 connection；将它置于 ``finally`` 中，可以保证读取或控制失败后仍完成清理，而且重复调用不会产生额外影响。

所有机器人都遵循这条调用链。不同机器人之间的差异，体现在观测和动作字典包含哪些路径，以及这些路径如何嵌套。

按路径访问零部件
--------------------

理解前面的观测和动作字典，需要先明确访问路径从何而来。组合时使用的名称构成第一段路径；某个零部件承载的下级零部件会继续增加路径层级。Franka 与 SO-101 分别对应这两种情况。

单臂 Franka 的 ``arm`` 和 ``end_effector`` 位于同一层级，因为机械臂和 Franka Hand 分别打开独立端点：

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

``via`` 不同表示两个零部件各自持有 connection；两个顶层名称因此对应两项可以独立管理的能力。

SO-101 则共用一条伺服总线。五个机械臂关节和夹爪由同一个 connection 驱动，因此末端执行器位于机械臂下一级：

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

嵌套动作并不是另一套 API，只是在同一个 ``send_action()`` 调用中增加了一层路径。``describe()`` 中的 ``via`` 用于解释层级和资源归属，任务代码仍只使用路径和对应数据。完整输出属于诊断信息，不是稳定的序列化格式。

组合零部件时保持接口一致
----------------------------

上一节从调用方角度说明了访问路径，本节进一步说明这些路径如何在组合阶段产生。组合只负责将可读零部件放到稳定名称下，不会改变后续读写方式。

``RobotPart`` 可以直接加入机器人。例如，``Robot(base=base, arm=arm, end_effector=hand)`` 中的三个关键字参数名会分别成为顶层路径。裸 ``Connection`` 只管理共享硬件 session，本身没有可供任务读取的观测；遇到这种情况，应先调用 ``session.part("left")``，再将返回的 ``RobotPart`` 加入机器人。

零部件还可以承载共用同一 connection 的下级零部件，它们会自动出现在下一层，因此 SO-101 的夹爪使用 ``arm.end_effector``。多项零部件需要组成可复用单元时，使用 ``PartGroup`` 再增加一层名称。双臂机器人可以在外层加入 ``left`` 和 ``right`` 两个 group，内部仍沿用同一套机械臂接口：

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

RLinf 不预设 ``arms`` 或 ``cameras`` 等固定字段。新增移动底盘、升降机构、云台或第三条机械臂时，只需确定稳定名称，调用方仍使用 ``get_observation()`` 与 ``send_action()``。组合结构确定后，env 就可以将整台机器人作为一个硬件边界使用。

在 env 中统一读写机器人
------------------------

env 需要把具名机器人数据转换为 policy 使用的观测和动作格式。它应从完整机器人接口进入，而不是在旁路直接读取 driver：每个 step 通过一次 ``robot.get_observation()`` 同时构造 ``state`` 和 ``frames``，再通过 ``robot.send_action()`` 将动作送回硬件。共用 connection 的零部件会在这次读取中复用同一份底层状态快照。

连接前取得的具类型零部件仍用于 step 循环之外的初始化。当前真机 env 通过 ``Arm`` 检查就绪状态、恢复错误并复位关节，通过 ``parts_of_type(Camera)`` 查找画面来源。相机的 placement 和生命周期仍由机器人管理，env 只消费观测结果。

现有 policy 如果使用扁平向量，可在这一边界通过 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter`` 转换。adapter 只改变表示形式，机器人接口仍保留具名嵌套结构。除常规的观测和动作转换外，长时间运行的 rollout 还需要在同一边界上统一检查并恢复硬件故障。

故障诊断与自动恢复
------------------

长时间真机运行中，控制器可能退出流式控制模式、触发关节位置或速度保护阈值，或在 episode 中途触发碰撞保护。处理这类中断需要解决两个问题：如何在不影响正常高频控制的前提下检测并恢复硬件故障，以及如何处理被故障打断的当前轨迹。机器人接口将故障检测与原位硬件恢复收敛在机器人边界内，而把轨迹处理留给 env。

调用方既可以在 ``Robot`` 或任意子组合与零部件上显式调用 ``diagnose()`` 与 ``recover()``，也可以交由底层连接在每次下发控制指令前自动完成检测与恢复：

.. code-block:: python

   status = robot.diagnose()
   if not status.healthy:
       robot.recover()

故障检测：对高频控制零额外开销
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果在每个控制周期都通过同步服务调用查询控制器状态，50–100 Hz 的高频流式控制会被直接阻塞。因此，底层连接采用“后台异步订阅 + 内存标志位快检”的检测机制：连接在后台订阅控制器的运行模式、系统状态、错误码和各关节轴使能状态等遥测流并实时更新内存缓存；每次下发控制指令前的健康检查只读取内存中的状态位。当机器人处于正常状态时，单步检测仅包含几次内存变量比较，不发起任何 RPC 调用，对高频控制回路没有额外延迟。

只有当内存标志位反映出异常，或调用方主动调用 ``robot.diagnose()`` 时，连接才会汇总完整的 ``HealthStatus`` 诊断快照，并通过 ``FaultLevel`` 将硬件状态归为四类：

- ``FaultLevel.OK``：控制器健康且正常跟踪指令。
- ``FaultLevel.MODE_DROP``：未锁存硬件故障，但控制器退出了流式控制模式，或在跟踪关节目标时发生停滞。
- ``FaultLevel.RECOVERABLE_FAULT``：锁存了运动故障或有关节轴下掉使能（如关节限位、速度超限或自碰撞保护），可通过软件清除。
- ``FaultLevel.UNRECOVERABLE_FAULT``：触发急停、安全锁或连接断开，无法通过软件清除，需要人工介入。

控制器上报的错误码、轴掉使能和模式回退属于确定性硬件信号，不存在误判，因此自动故障恢复（``enable_fault_recovery``）默认开启。相比之下，关节跟踪停滞检测（``enable_stall_detection``）是一种可选的启发式内存看门狗：当关节跟踪误差连续 ``stall_steps_threshold`` 步大于 ``stall_position_threshold`` 且实际关节速度小于 ``stall_velocity_threshold`` 时判定为停滞。由于精细慢速靠近或柔顺接触也可能满足这一条件，``enable_stall_detection`` 默认设为 ``false``，需要时可在硬件配置中开启并调节阈值：

.. code-block:: yaml

   hardware:
     type: R2
     config:
       enable_fault_recovery: true
       enable_stall_detection: false
       stall_position_threshold: 0.05
       stall_velocity_threshold: 0.002
       stall_steps_threshold: 15
       max_recovery_attempts: 3
       recovery_cooldown_window: 10.0

分级恢复机制及其对控制的影响
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

检测到故障后，``recover()`` 会根据故障等级执行最小必要的恢复动作，避免轻微中断触发完整复位：

- 针对 ``FaultLevel.MODE_DROP``，恢复流程只重新切回流式控制模式，不重置驱动器。
- 针对 ``FaultLevel.RECOVERABLE_FAULT``，恢复流程依次清除控制器故障、重新为下掉的关节轴上使能，并切回流式控制模式。
- 针对 ``FaultLevel.UNRECOVERABLE_FAULT``，或在 ``recovery_cooldown_window`` 秒内连续触发超过 ``max_recovery_attempts`` 次自动恢复时，滑动时间窗熔断器会立即停止自动恢复并抛出 ``RuntimeError`` 交由人工处理。

为了防止恢复瞬间机械臂发生抽动或跳变，底层连接在重新上使能或切回流式控制模式前，都会先将内部缓存的目标关节角强制同步为机器人当前的实测关节角。即便故障期间上层算法的目标位置已经拉开差距，恢复后的第一帧指令仍会让机器人平滑保持在当前实测姿态，而不会朝故障前的旧目标猛冲。

从控制时序上看，无故障运行时控制回路没有任何额外耗时；仅在故障实际触发的那一个 step，清错、上使能和切模式的同步服务调用会让指令流短暂暂停约 50–300 ms（期间机械臂保持原位），随后立即恢复控制。对于遥操作或评估脚本，这意味着偶发限位或模式回退后可以直接继续操作而不必重启程序；但对于强化学习训练采样，这几十到几百毫秒的停顿与目标同步会破坏该步状态转移的马尔可夫连续性。因此，env 会在每个 step 结束后读取本次动作是否发生过自动恢复，将故障详情记入 ``info["robot_fault"]``，并在开启 ``truncate_on_fault_recovery`` 时将该 step 标记为 ``truncated=True``，交由 rollout runner 干净地截断当前轨迹并在下一轮采样前执行复位。将底层硬件恢复与 episode 截断分离后，任务代码同样无需依赖部署位置。

任务代码不依赖部署位置
----------------------

访问路径标识机器人能力，不表示承载它的进程。相机可以位于 env 进程，机械臂可以运行在其他节点，两者仍出现在同一份观测中；具体位置记录在各自的 owner connection 上。使用不同 connection 的零部件可以并行读写，共享 connection 的分支则按声明顺序执行，避免并发访问不支持该模式的厂商 SDK。

任务和 policy 代码只依赖零部件名称及其数据，无需感知 Ray actor、RPC、串口或厂商 session。调整 connection 的部署位置时，前述调用和数据结构保持不变。机器人接口还需要划清最后一项职责：它说明硬件如何工作，但不定义 rollout 要完成什么目标。

分离硬件能力与任务逻辑
----------------------

零部件说明如何读取传感器或控制执行器，任务则解释这些读数和动作为什么重要。奖励、终止条件、任务特有的复位流程和 Gymnasium space 应由该任务注册的真机 env 定义，env 通过同一套组合调用驱动硬件，并负责其生命周期。

最终形成两项相互独立的 contract：机器人路径在任务和 placement 变化时保持稳定，任务则可以调整 policy 侧 schema，而无需修改 driver。

后续阅读
--------

根据下一步需要修改的边界选择文档：

- :doc:`新增真机任务 <../extending/new_task>`：在已支持的真机上添加任务。
- :doc:`添加机器人 <../extending/new_robot>`：接入本地传感器、执行器或整台机器人。
- :doc:`机器人架构 <robotics_architecture>`：了解 ``parts`` 与 ``children`` 的区别，以及共享 connection、``PartGroup``、资源生命周期和 worker placement。
- :doc:`遥操作 <../guides/teleoperation>`：组合多种操作者设备。
