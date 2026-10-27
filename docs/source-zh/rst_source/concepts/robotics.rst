机器人接口
============

本页说明任务和 env 如何通过机器人接口检查、连接、读取并控制真机。内容按从底层调用到上层集成的顺序展开：先介绍单台机器人的四阶段生命周期调用链，再说明访问路径与多部件组合规则，随后解释连接层如何完成运动轨迹平滑与多部件协同批处理、如何在运行中诊断并自动恢复硬件故障，最后说明真机强化学习环境与跨节点部署如何复用这一接口边界。硬件 session、资源归属和 worker placement 的内部实现见 :doc:`机器人架构 <robotics_architecture>`。

机器人接口为机械臂、末端执行器、相机和移动底盘等零部件提供稳定的名称和访问路径，观测和动作沿用同一组路径。任务侧无需直接依赖具体 driver；只要路径保持不变，更换 backend 或调整部署节点也无需修改调用代码。

连接、读取与控制机器人
----------------------

一次完整调用分为四个阶段：构建并检查机器人，取得初始化所需的零部件，连接后读写数据，最后释放硬件。下面的示例将这四个阶段放在同一段代码中：

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

按路径寻址与组合零部件
----------------------

理解 ``get_observation()`` 与 ``send_action()`` 使用的嵌套字典，需要先明确访问路径如何产生，以及多个零部件如何组装为一台完整机器人。组合阶段只负责将可读零部件挂载到稳定名称下，不会改变后续读写方式。

独立连接与共享总线的路径层级
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

组合时使用的名称构成第一段路径；某个零部件承载的下级零部件会继续增加路径层级。Franka 与 SO-101 分别对应独立端点与共享总线两种情况。

单臂 Franka 的 ``arm`` 和 ``end_effector`` 位于同一层级，因为机械臂和 Franka Hand 分别打开独立端点：

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

``via`` 不同表示两个零部件各自持有 connection；两个顶层名称因此对应两项独立管理的能力。

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

组合零部件与部件组
^^^^^^^^^^^^^^^^^^

``RobotPart`` 可以直接加入机器人。例如，``Robot(base=base, arm=arm, end_effector=hand)`` 中的三个关键字参数名会分别成为顶层路径。裸 ``Connection`` 只管理共享硬件 session，本身没有可供任务读取的观测；遇到这种情况，应先调用 ``session.part("left")``，再将返回的 ``RobotPart`` 加入机器人。

零部件还可以承载共用同一 connection 的下级零部件，它们会自动出现在下一层，因此 SO-101 的夹爪使用 ``arm.end_effector``。多项零部件需要组成可复用单元时，使用 ``PartGroup`` 再增加一层名称。双臂机器人可以在外层加入 ``left`` 和 ``right`` 两个 group，内部仍沿用同一套机械臂接口：

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

RLinf 不预设 ``arms`` 或 ``cameras`` 等固定字段。新增移动底盘、升降机构、云台或第三条机械臂时，只需确定稳定名称，调用方仍使用 ``get_observation()`` 与 ``send_action()``。当多个部件或双臂共用同一条底层硬件连接时，要让真机平稳受控，还需要由连接层统一管理目标平滑与多部件指令合并。

平滑运动控制与多部件协同下发
----------------------------

高频流式控制器（通常运行在 50–100 Hz）无法直接承受单步跨度过大的目标跳变、负载受阻时目标角与实测角的持续累积偏差，或者在双臂等共享总线上先后发起两段独立的全身体轨迹流。机器人接口在 ``Connection`` 层通过可配置的运动平滑参数与多部件指令批处理统一解决这三类问题。

轨迹插值、漂移重锚定与速度脉冲
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

每个 ``Connection`` 都持有一个 ``MotionProfile`` 实例（可在构造时传入或通过 ``configure_motion_profile()`` 动态修改，控制频率 ``control_rate_hz`` 与脉冲频率 ``pulse_rate_hz`` 默认均为 ``50.0`` Hz，不与特定硬件频率绑定），并可通过 ``JointGroupSpec`` 为腰部、双臂、颈部等不同惯量的关节组分别指定步长上限与位置软限位。

每次收到新目标时，底层连接会先通过 ``reanchor()`` 防止指令积分饱和（wind-up）。如果关节因重力、外部阻力或短暂暂停未能跟上先前的目标，内部缓存的目标角就会逐渐偏离实测角；一旦阻力消失，累积的偏差会导致关节猛冲。当缓存目标与实测角的偏差超过 ``max_drift`` 时，连接会自动将插值起点拉回实测角附近，再计算下一步位移。

如果单次目标变化在单步上限 ``max_step`` 以内，连接会直接下发，不引入额外延迟。当目标跨度较大（例如关键帧动作、IK 跳变或调用 ``reset_joint()``）时，``generate_trajectory()`` 会按各关节组的最大相对位移自动计算插值步数，生成首尾速度均为零的余弦平滑（:math:`C^1` smoothstep）轨迹，并在终点追加 ``hold_steps`` 帧稳态保持指令使控制器平稳收敛。对于设有硬件看门狗、要求连续发送速度帧的移动底盘，``generate_velocity_pulse()`` 会按 ``pulse_rate_hz`` 连续发送平滑加速、匀速巡航和余弦减速脉冲，并在末尾补发显式零速帧，避免底盘因单帧超时骤停或急停点头。

共享总线上的多部件指令批处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在双臂或全身共用同一个 ``Connection`` 的机器人上，如果依次调用 ``left_arm.reset_joint(...)`` 和 ``right_arm.reset_joint(...)``，默认会串行发起两段阻塞式轨迹流：第一段在移动左臂的同时将右臂锁在旧目标，紧接着切换到第二段全身指令流时极易触发控制器流式模式掉线或运动控制器报错。

将多个部件的直接控制调用包裹在 ``with robot.batch_commands():`` 或 ``connection.batch_commands()`` 上下文中，可以将多次部件调用合并为一次同步全身轨迹下发。在 ``with`` 块内部，各个部件只更新连接缓存目标中对应的关节切片；退出 ``with`` 块时，``_flush_command_batch()`` 会一次性对所有更新过的关节组做联合平滑插值并同步发送：

.. code-block:: python

   # 按需在运行时调整控制频率或单步最大步长
   robot.child("left_arm").connection.configure_motion_profile(
       control_rate_hz=100.0,
       max_step=0.006,
   )

   # 将左右臂复位合并为同一条全身平滑轨迹同步下发
   with robot.batch_commands():
       robot.child("left_arm").reset_joint(left_home_qpos)
       robot.child("right_arm").reset_joint(right_home_qpos)

在每步策略交互中，``robot.send_action({...})`` 内部已经自动为涉及的所有连接开启 ``batch_commands()``，因此多部件策略动作天然会合并下发；当在初始化或回合复位阶段直接调用 ``reset_joint()`` 等部件方法同时控制多个部件时，使用 ``with robot.batch_commands():`` 即可保证多部件协同平滑运动。

运动平滑与批处理消除了软件侧指令跳变与并发冲突，但在真实物理交互中，外部接触或边界条件仍可能触发硬件保护。

硬件故障诊断与自动恢复
----------------------

真机运行过程中，控制器可能因通信抖动退出流式控制模式，也可能在运动中触发关节限位、速度超限或碰撞保护，导致电机掉使能、机器人停止响应。你可以在 ``Robot`` 或任意零部件上直接调用 ``diagnose()`` 检查健康状态、调用 ``recover()`` 执行恢复；开启自动恢复后，底层连接也会在每次下发控制指令前自动完成检查与恢复：

.. code-block:: python

   status = robot.diagnose()
   if not status.healthy:
       robot.recover()

零热路径开销的故障检测与分级
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果每次下发 50–100 Hz 控制指令前都额外发起一次同步远程调用或串口往返去查询控制器状态，通信耗时就会阻塞控制回路。为此，``Connection`` 将故障的“采集上报”与下发指令前的“状态检查”解耦，通过内存中的健康状态缓存同时兼容 ROS 和非 ROS 机器人：

1. **异步状态回调或 SDK 后台线程**\ （如 ROS/RFrame 状态话题、UDP 状态广播或 SocketCAN 监听线程）：当后台回调收到错误帧或控制模式切换时，调用 ``self.report_health(...)`` 将 ``HealthStatus`` 记入内存。
2. **随观测帧捎带的状态字节**\ （如 RS-485/TTL 串口舵机或 CANopen PDO 报文，在读取关节角的同时已在返回帧中包含错误位图）：当 ``get_observation()`` 通过 ``get_state()`` 读取最新硬件状态时，``Connection.read_state()`` 会自动把该状态传给 ``_inspect_state_health(state)`` 并更新内存中的健康状态，无需发起第二次总线读取。

每次下发控制指令前，框架只检查内存里缓存的健康状态。正常运行时，单步检查只是一次内存读取，不增加任何 RPC 或总线通信耗时。当内存缓存记录了故障，或者外部主动调用 ``robot.diagnose()`` 时，连接会返回 ``HealthStatus`` 诊断结果，并按 ``FaultLevel`` 将当前状态分为四类：

- ``FaultLevel.OK``：控制器正常，能够响应控制指令。
- ``FaultLevel.MODE_DROP``：未触发硬件报错，但控制器退出了流式控制模式。
- ``FaultLevel.RECOVERABLE_FAULT``：触发了关节限位、速度超限或自碰撞等保护，或有关节轴掉使能，可通过软件清除。
- ``FaultLevel.UNRECOVERABLE_FAULT``：触发急停、安全锁或连接断开，无法由软件清除，需要人工处理。

由于控制器上报的错误码、轴掉使能和控制模式退出都是明确的硬件状态，不会产生误报，因此自动故障恢复（``enable_fault_recovery``）默认开启，可在硬件 YAML 配置或运行时通过 ``robot.configure_fault_recovery(...)`` 调整：

.. code-block:: yaml

   hardware:
     type: R2
     config:
       enable_fault_recovery: true
       max_recovery_attempts: 3
       recovery_cooldown_window: 10.0

分级恢复与防跳变目标同步
^^^^^^^^^^^^^^^^^^^^^^^^

检测到异常后，``recover()`` 会根据 ``FaultLevel`` 的等级执行对应的恢复步骤，而不是每次都重置整个控制器：

- ``FaultLevel.MODE_DROP``：只需重新切回流式控制模式，不重置电机驱动器。
- ``FaultLevel.RECOVERABLE_FAULT``：依次清除控制器错误码、重新为掉使能的关节轴上使能，再切回流式控制模式。
- ``FaultLevel.UNRECOVERABLE_FAULT``：直接抛出 ``RuntimeError`` 停止运行；对于可恢复故障，如果在 ``recovery_cooldown_window`` 秒内连续触发恢复超过 ``max_recovery_attempts`` 次，说明故障原因仍未消除，熔断机制也会停止重试并抛出 ``RuntimeError``。

恢复过程中还需要防止位置跳变：故障发生后机器人停在原地，但上层策略可能还在继续更新目标关节角。如果重新上使能后直接发送旧目标，机械臂就会突然朝旧目标猛冲。为了避免跳变，底层连接在重新上使能或切回流式控制模式前，会先用机器人当前的实测关节角覆盖内部缓存的目标关节角，使恢复后的第一帧指令平稳保持在当前姿态。

无故障运行时，这套机制没有任何额外耗时；只有在故障实际触发的那一个 step，清错、上使能和切模式这几次同步恢复调用会让控制流暂停约 50–300 ms，期间机器人保持原位，随后即可继续接收后续指令。

为新机器人驱动接入故障检测与恢复
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为新机器人驱动（任何继承自 ``Connection`` 或 ``ControllablePart`` 的类）接入这套故障检测与自动恢复接口时，只需根据硬件通信方式二选一上报故障，并实现 ``clear_errors()``：

- **方式一（异步回调或后台线程）**：在后台状态监听回调中调用 ``self.report_health(HealthStatus(...))``。
- **方式二（观测帧捎带状态位）**：重写 ``_inspect_state_health(self, state) -> HealthStatus | None``，从 ``get_state()`` 返回的硬件状态中提取错误字段。
- **恢复动作**：实现 ``clear_errors(self) -> bool``，在其中将内部目标角同步到实测角、清除驱动器报错并恢复控制模式（若控制模式掉线可单独切回而不必清错，还可按需重写 ``_recover_mode(self) -> str``）。

.. code-block:: python

   from rlinf.robotics import ControllablePart, FaultLevel, HealthStatus


   class MyArmDriver(ControllablePart):
       # 方式二：在每次读取观测状态时检查随帧捎带的错误码
       def _inspect_state_health(self, state: dict) -> HealthStatus:
           err = int(state.get("error_code", 0))
           if err != 0:
               return HealthStatus(
                   healthy=False,
                   level=FaultLevel.RECOVERABLE_FAULT,
                   error_code=err,
                   error_message=f"drive fault {err}",
               )
           return HealthStatus(healthy=True)

       # 或方式一：在 SDK 异步状态回调中调用 self.report_health(...)
       def _on_sdk_error_callback(self, err_code: int, msg: str) -> None:
           self.report_health(
               HealthStatus(
                   healthy=(err_code == 0),
                   level=FaultLevel.RECOVERABLE_FAULT
                   if err_code
                   else FaultLevel.OK,
                   error_code=err_code,
                   error_message=msg,
               )
           )

       def clear_errors(self) -> bool:
           self._sync_cmd_to_actual()
           self._sdk.reset_faults()
           return True

实现上述方法后，``robot.diagnose()``、``robot.recover()``、下发指令前的自动检查与分级恢复、冷却窗口熔断以及 ``robot.consume_recovery_event()`` 都会自动生效，无需再编写额外调度逻辑。

对接真机环境与跨节点部署
------------------------

访问路径、运动平滑与故障恢复都封装在机器人边界内部后，上层强化学习 env 与集群调度器可以把组合后的机器人当作单一设备边界使用，将硬件能力与任务定义和进程部署彻底解耦。

在真机环境中转换观测、动作与故障事件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

零部件说明如何读取传感器或控制执行器，任务则定义这些读数和动作为什么重要。奖励、终止条件、任务特有的复位流程和 Gymnasium space 由该任务注册的真机 env 定义，env 通过组合机器人接口驱动硬件并管理其生命周期：

- **观测与动作转换**：每个 step 通过一次 ``robot.get_observation()`` 同时构造 policy 侧的 ``state`` 和 ``frames``，共用 connection 的零部件会在同一次读取中复用底层状态快照，并通过 ``robot.send_action()`` 作为唯一下发通道。使用扁平向量的现有 policy 可在这一边界通过 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter`` 转换，机器人接口本身仍保留具名嵌套结构。
- **单步循环外的初始化**：连接前取得的具类型零部件引用用于回合初始化与复位，例如通过 ``Arm`` 检查就绪状态、恢复错误并复位关节，通过 ``parts_of_type(Camera)`` 发现图像来源。
- **被故障打断的回合截断**：在遥操作或独立评估脚本中，经历 50–300 ms 的自动恢复后操作可以直接继续；但在强化学习数据采集中，中间的停顿会使当前 step 的状态转移不再连续。因此 env 会在每个 step 结束时调用 ``robot.consume_recovery_event()``，将故障详情记入 ``info["robot_fault"]``，并在开启 ``truncate_on_fault_recovery`` 时将该 step 标记为 ``truncated=True``，通知 rollout runner 结束当前轨迹并在下一轮采样前复位机器人。

对任务代码透明的跨进程与跨节点部署
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

访问路径标识机器人能力，不表示承载它的进程。相机可以位于 env 进程，机械臂可以运行在其他节点，两者仍出现在同一份观测中，因为具体部署位置记录在各自的 owner connection 上。使用不同 connection 的零部件可以并行读写，共享同一 connection 的分支则按声明顺序执行，避免并发访问厂商 session。

任务和 policy 代码因此只依赖零部件名称及其数据，无需感知 Ray actor、RPC、串口或厂商 session。调整 connection 的部署位置时，本页介绍的所有调用方式和数据结构均保持不变。

后续阅读
--------

根据下一步需要修改的边界选择文档：

- :doc:`新增真机任务 <../extending/new_task>`：在已支持的真机上添加任务。
- :doc:`添加机器人 <../extending/new_robot>`：接入本地传感器、执行器或整台机器人。
- :doc:`机器人架构 <robotics_architecture>`：了解 ``parts`` 与 ``children`` 的区别，以及共享 connection、``PartGroup``、资源生命周期和 worker placement。
- :doc:`遥操作 <../guides/teleoperation>`：组合多种操作者设备。
