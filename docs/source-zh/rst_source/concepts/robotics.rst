机器人接口
============

本页介绍任务和 ``env`` 代码如何通过 ``rlinf.robotics`` 连接、读取和控制真机。全文按从单机调用到系统集成的顺序展开：

1. 单台机器人的连接、读取观测与下发动作流程。
2. 访问路径规则，以及如何用 ``RobotPart`` 和 ``PartGroup`` 组装多部件机器人。
3. 如何通过 ``MotionProfile`` 和 ``batch_commands()`` 平滑大幅度运动并合并多部件指令。
4. 如何通过 ``diagnose()`` 和 ``recover()`` 诊断并自动恢复硬件故障。
5. ``RealWorldEnv`` 与多节点集群如何接入并复用这套机器人接口。

硬件 session、资源归属与 worker placement 的底层实现见 :doc:`机器人架构 <robotics_architecture>`。

连接、读取与控制机器人
----------------------

调用一台机器人分为四步：构建并查看声明、获取初始化所需的 part、连接硬件并运行 ``step`` 循环，最后断开连接。下面的示例展示了完整流程：

.. code-block:: python

   from rlinf.robotics import Arm, Camera, build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # 机器人未上电或网络未连通时也可以安全调用。
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

``build_robot()`` 根据注册名查找对应的 builder，返回一个尚未连接的 ``Robot`` 对象。构建阶段只记录硬件参数和部署节点，不会导入厂商 SDK，也不会打开硬件设备。``describe()`` 直接读取这些声明信息，因此即使机器人还没上电，也能查看各部件的访问路径、所在节点和 ``Connection`` 归属。

连接前，初始化代码可以先获取需要单独操作的部件。``child("arm", Arm)`` 获取 ``arm`` 路径上的部件并校验其类型为 ``Arm``，IDE 也会将其静态类型推断为 ``Arm``。``parts_of_type(Camera)`` 则遍历整棵机器人结构树，返回以路径为 key 的所有相机。固定路径的部件用 ``child()`` 获取；只按设备类别查找、不关心具体命名时用 ``parts_of_type()``。

``connect()`` 会依次打开每条底层 ``Connection``；如果某条连接打开失败，会自动回滚前面已打开的连接。连接成功后，可以用 ``Arm.is_robot_up()`` 和 ``Camera.is_ready()`` 检查硬件是否就绪，或在进入 ``step`` 循环前调用 ``clear_errors()`` 和 ``reset_joint()`` 完成清错与关节复位。

进入控制循环后，只需调用两个方法：``get_observation()`` 一次性读取所有部件，返回以路径为 key 的嵌套字典；``send_action()`` 接收相同层级结构的动作字典，只下发本次传入的分支，并将各部件实际执行的动作作为 ``applied`` 返回。将 ``disconnect()`` 放在 ``finally`` 块中，可以确保哪怕读写中途报错也能按相反顺序安全释放所有连接，且重复调用 ``disconnect()`` 是安全的。

所有机器人都使用这套相同的调用流程，不同机器人的区别只在于观测和动作字典里的路径结构。

访问路径与多部件组合
--------------------

``get_observation()`` 和 ``send_action()`` 字典里的 key，直接由组装机器人时的命名和连接关系决定。增加部件或对部件分组只会增加路径前缀，不会改变读写接口。

独立连接与共享总线
^^^^^^^^^^^^^^^^^^

组装时赋给部件的参数名成为第一段路径；挂在该部件下面的子部件会继续增加下一段路径。对比 Franka 和 SO-101 可以看清为什么有的路径扁平、有的路径嵌套。

单臂 Franka 的机械臂控制器与 Franka Hand 夹爪分别连接各自的通信端点，因此 ``arm`` 和 ``end_effector`` 处于同一层级：

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

``describe()`` 输出中不同的 ``via`` 编号表示它们各自拥有独立的 ``Connection``。

SO-101 的五个机械臂关节和夹爪则共用同一条串口舵机总线。由于夹爪通过机械臂的 ``Connection`` 收发指令，它挂在 ``arm`` 下面成为 ``arm.end_effector``，且两者的 ``via`` 都是 ``SO101Arm#1``：

.. code-block:: text

   SO101Robot
   └── arm               SO101Arm             node=0     via SO101Arm#1
       └── end_effector  MethodEndEffector    node=0     via SO101Arm#1

读写数据时的字典层级与路径完全一致：

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

使用 RobotPart 与 PartGroup 组装机器人
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

组装 ``Robot`` 时，任何 ``RobotPart`` 都可以直接作为关键字参数传入，例如 ``Robot(base=base, arm=arm, end_effector=hand)`` 会直接生成 ``base``、``arm`` 和 ``end_effector`` 三个顶层路径。如果一条共享 ``Connection`` 同时驱动多个子系统（如全身控制器或双臂控制器），连接本身并不直接对应单个观测部件，此时先调用 ``session.part("left")`` 取出对应的 ``RobotPart``，再传给 ``Robot``。

当多个部件共同组成一个逻辑单元（例如左臂加上左夹爪）时，可以用 ``PartGroup`` 将它们包成一组，在外层增加 ``left`` 或 ``right`` 等路径前缀：

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

``Robot`` 不限制必须有哪些固定槽位。无论是接入移动底盘、腰部、头部还是第三条机械臂，都只需赋予一个路径名，上层依然通过 ``get_observation()`` 和 ``send_action()`` 读写。不过，当双臂或全身多个部件共用同一条底层 ``Connection`` 时，要让硬件运动平稳且不互相打架，还需要由 ``Connection`` 统一处理轨迹平滑与多部件指令合并。

运动平滑与多部件指令合并
------------------------

高频流式控制器（通常为 50–100 Hz）要求每帧传入平滑、连续的目标值。在实际运行中，以下三种情况很容易引起机身猛冲或控制器报错退出流式模式：

1. 单步下发的目标跨度过大，超过电机单帧允许的最大位移。
2. 关节受重力或外部碰撞阻挡而停留在原位，但下发的目标角仍在继续累加；一旦阻力消失，累积的巨大偏差会让机械臂突然猛冲。
3. 在双臂或全身共用一条 ``Connection`` 的机器人上，先后调用 ``left_arm`` 和 ``right_arm`` 的运动方法，导致底层串行发送两段阻塞式全身轨迹流。

``Connection`` 通过 ``MotionProfile``、``JointGroupSpec`` 和 ``batch_commands()`` 在接口层统一解决这三类问题。

使用 MotionProfile 平滑轨迹与底盘脉冲
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

每个 ``Connection`` 都持有一个 ``MotionProfile`` 对象（可在初始化时传入，或通过 ``configure_motion_profile()`` 动态修改；控制频率 ``control_rate_hz`` 与脉冲频率 ``pulse_rate_hz`` 默认均为 ``50.0`` Hz，不绑定固定频率），同时可以用 ``JointGroupSpec`` 为腰部、双臂、颈部等不同惯量的关节组分别设置步长上限与关节限位。

收到控制目标时，``MotionProfile`` 和 ``JointGroupSpec`` 会自动完成三项保护：

- **限制目标累积偏差（``reanchor()``）**：如果关节因阻力或暂停未能跟上先前的指令，内部缓存的目标角会偏离实测角。只要两者差距超过 ``max_drift``，``reanchor()`` 就会先把插值起点拉回实测角附近，防止偏差持续累积后突然释放。
- **大幅度位移自动插值（``generate_trajectory()``）**：如果单次目标变化在单步上限 ``max_step`` 以内，连接会直接下发，不增加任何延迟。如果目标跨度较大（如调用 ``reset_joint()`` 或大跨度关键帧），``generate_trajectory()`` 会自动将其拆分为首尾速度均为零的余弦平滑（:math:`C^1` smoothstep）轨迹，并在到达终点后补发 ``hold_steps`` 帧保持指令，让控制器平稳停稳。
- **底盘连续速度脉冲（``generate_velocity_pulse()``）**：许多移动底盘带有硬件看门狗，只发单帧速度指令会被立刻超时急停。``generate_velocity_pulse()`` 会按 ``pulse_rate_hz`` 连续发送平滑加速、匀速巡航和余弦减速曲线，并在结束时补发零速帧使底盘平稳停下。

使用 batch_commands() 合并多部件指令
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在双臂或全身共用同一个 ``Connection`` 的机器人上，如果依次调用 ``left_arm.reset_joint(...)`` 和 ``right_arm.reset_joint(...)``，默认会串行执行两段独立轨迹：第一段在移动左臂时把右臂锁在旧目标，紧接着切换到第二段全身指令流时很容易触发控制器报错或退出流式模式。

把多个部件的调用放在 ``with robot.batch_commands():`` 或 ``connection.batch_commands()`` 中，就能将它们合并为同一条轨迹。在 ``with`` 块内，每个部件只更新 ``Connection`` 缓存目标中属于自己的关节切片；退出 ``with`` 块时，``_flush_command_batch()`` 会对所有更新过的关节组一起做平滑插值并同步下发：

.. code-block:: python

   # 按需在运行时修改控制频率或单步最大步长
   robot.child("left_arm").connection.configure_motion_profile(
       control_rate_hz=100.0,
       max_step=0.006,
   )

   # 将左右臂复位合并为一次同步全身轨迹
   with robot.batch_commands():
       robot.child("left_arm").reset_joint(left_home_qpos)
       robot.child("right_arm").reset_joint(right_home_qpos)

在 ``step`` 循环中，``robot.send_action({...})`` 内部已经自动为涉及的所有 ``Connection`` 开启 ``batch_commands()``，因此 ``policy`` 输出的多部件动作天然会合并下发；只有在初始化或 ``reset()`` 阶段直接跨多个部件调用 ``reset_joint()`` 等方法时，才需要显式使用 ``with robot.batch_commands():``。

硬件故障诊断与自动恢复
----------------------

运动平滑避免了指令跳变，但真机交互中的碰撞或关节超限仍可能触发硬件保护，导致电机掉使能或控制器退出流式模式。你可以在 ``Robot`` 或任意部件上手动调用 ``diagnose()`` 和 ``recover()``，也可以保持自动恢复开启，让 ``Connection`` 在每次发指令前自动检查并恢复：

.. code-block:: python

   status = robot.diagnose()
   if not status.healthy:
       robot.recover()

在不阻塞控制回路的前提下检测故障
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果在每次下发 50–100 Hz 控制指令前都额外发起一次同步 RPC 或串口查询去问控制器状态，通信延迟会直接卡住控制循环。因此，``Connection`` 在内存中维护一份 ``HealthStatus`` 缓存，并通过两条非阻塞通道更新它：

1. **异步状态回调（``report_health()``）**：当后台监听线程（如 ROS/RFrame 状态订阅、UDP 广播或 SocketCAN 线程）收到错误码或控制模式变化时，调用 ``self.report_health(...)`` 更新内存中的 ``HealthStatus``。
2. **随状态读取一并检查（``_inspect_state_health()``）**：对于每次读取关节状态时已经在返回包里自带错误位图的硬件（如 RS-485/TTL 串口舵机或 CANopen PDO 帧），``Connection.read_state()`` 会在 ``get_state()`` 拿到新状态后直接调用 ``_inspect_state_health(state)`` 更新健康缓存，不需要发起第二次总线请求。

每次发指令前，``Connection`` 只读取内存里的 ``HealthStatus``，正常运行时没有任何额外通信开销。当检测到故障或主动调用 ``robot.diagnose()`` 时，返回的 ``HealthStatus`` 会用 ``FaultLevel`` 标明故障等级：

- ``FaultLevel.OK``：控制器健康，正常接收指令。
- ``FaultLevel.MODE_DROP``：没有硬件报错，但控制器退出了流式控制模式。
- ``FaultLevel.RECOVERABLE_FAULT``：触发了关节限位、速度限制或自碰撞等保护，或存在关节轴掉使能，可通过软件清错恢复。
- ``FaultLevel.UNRECOVERABLE_FAULT``：急停按下、安全锁触发或连接断开，无法由软件恢复，需人工介入。

自动故障恢复（``enable_fault_recovery``）默认开启，可在硬件 YAML 配置或运行时通过 ``robot.configure_fault_recovery(...)`` 调整参数：

.. code-block:: yaml

   hardware:
     type: R2
     config:
       enable_fault_recovery: true
       max_recovery_attempts: 3
       recovery_cooldown_window: 10.0

按故障等级恢复并防止位置跳变
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

触发恢复时，``recover()`` 会根据 ``FaultLevel`` 执行对应的恢复动作：

- ``FaultLevel.MODE_DROP``：直接切回流式控制模式，不重置电机驱动器。
- ``FaultLevel.RECOVERABLE_FAULT``：依次清除控制器错误码、重新为掉使能的关节轴上使能，再切回流式控制模式。
- ``FaultLevel.UNRECOVERABLE_FAULT``：直接抛出 ``RuntimeError``；对于可恢复故障，如果在 ``recovery_cooldown_window`` 秒内连续恢复超过 ``max_recovery_attempts`` 次，说明故障根因仍未消除，熔断机制也会停止重试并抛出 ``RuntimeError``。

为了防止恢复瞬间机械臂猛冲，``Connection`` 在重新上使能或切回流式模式前，会先用当前的实测关节角覆盖内部缓存的目标关节角。这样即使故障停机期间上层 ``policy`` 已经把目标角推到了别处，恢复后的第一帧指令也会让机器人平稳保持在当前位置。

正常运行时这套机制零延迟；只有在某个 ``step`` 实际触发故障恢复时，清错、上使能和切模式会使控制流暂停约 50–300 ms，恢复完成后后续指令继续正常执行。

在新驱动中接入故障恢复
^^^^^^^^^^^^^^^^^^^^^^

为新机器人驱动（继承自 ``Connection`` 或 ``ControllablePart`` 的类）接入故障恢复时，只需按硬件通信方式二选一上报状态，并实现 ``clear_errors()``：

- **方式一（异步回调或后台线程）**：在状态回调中调用 ``self.report_health(HealthStatus(...))``。
- **方式二（随状态读取检查）**：重写 ``_inspect_state_health(self, state) -> HealthStatus | None``，检查 ``get_state()`` 返回字典中的错误字段。
- **清错动作**：实现 ``clear_errors(self) -> bool``，在其中将缓存目标同步到实测角、清除硬件报错并恢复控制模式（若模式退出可单独恢复，还可按需重写 ``_recover_mode(self) -> str``）。

.. code-block:: python

   from rlinf.robotics import ControllablePart, FaultLevel, HealthStatus


   class MyArmDriver(ControllablePart):
       # 方式二：检查每次 get_state() 返回字典中自带的错误码
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

       # 或方式一：在 SDK 异步状态回调中上报健康状态
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

定义上述方法后，``robot.diagnose()``、``robot.recover()``、下发指令前的自动恢复、冷却窗口熔断以及 ``robot.consume_recovery_event()`` 都会自动生效。

在真机环境与多节点集群中使用机器人
----------------------------------

访问路径、运动平滑与故障恢复都由 ``Robot`` 和 ``Connection`` 处理后，上层的 ``RealWorldEnv`` 与集群调度器就可以用统一方式驱动任意组合的机器人。

对接 RealWorldEnv 与处理故障截断
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``RobotPart`` 负责读写传感器与执行器，而 ``RealWorldEnv`` 子类负责定义任务奖励、终止条件、``episode`` 复位流程以及 Gymnasium spaces：

- **观测与动作转换**：每个 ``step`` 调用一次 ``robot.get_observation()`` 构造 ``state`` 和相机 ``frames``，共用 ``Connection`` 的部件在同一次读取中复用同一份硬件状态快照，并通过 ``robot.send_action()`` 下发动作。如果 ``policy`` 使用扁平向量，可通过 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter`` 在扁平数组与嵌套字典之间转换。
- **episode 初始化与复位**：连接前通过 ``child()`` 或 ``parts_of_type()`` 获取的部件引用，在 ``reset()`` 中用于检查就绪状态、清除报错以及将机械臂复位到初始姿态。
- **故障打断时的 episode 截断**：在遥操作或独立评估脚本中，经历 50–300 ms 的自动恢复后可以直接继续操作；但在强化学习数据采集中，中间这几百毫秒的停顿会破坏当前 ``step`` 状态转移的连续性。因此 ``RealWorldEnv`` 会在每个 ``step`` 结束时调用 ``robot.consume_recovery_event()``，把故障信息记录到 ``info["robot_fault"]``，并在开启 ``truncate_on_fault_recovery`` 时将该 ``step`` 标记为 ``truncated=True``，通知 rollout runner 结束当前 ``episode`` 并在下一轮采样前复位机器人。

跨进程与跨节点部署
^^^^^^^^^^^^^^^^^^

访问路径只表示机器人具备什么部件，不绑定驱动运行在哪个进程。相机可以运行在 ``env`` 进程，机械臂可以运行在另一台机器的 ``worker`` 上，但两者依然出现在同一个 ``robot.get_observation()`` 字典里，因为 ``node_rank`` 和 ``worker_name`` 配置在各自的 ``Connection`` 上。不同 ``Connection`` 之间会并行读写，而共用同一个 ``Connection`` 的部件则按声明顺序依次执行，避免并发调用底层 SDK。

把某个部件迁移到其他进程或节点时，只需修改其 ``Connection`` 的部署参数，``env`` 和 ``policy`` 代码完全不需要改动。

后续阅读
--------

- :doc:`新增真机任务 <../extending/new_task>`：在已支持的真机硬件上定义新的 Gymnasium 任务。
- :doc:`添加机器人 <../extending/new_robot>`：实现并注册新的传感器、执行器或整机组合。
- :doc:`机器人架构 <robotics_architecture>`：了解 ``parts``、``children``、共享 ``Connection``、生命周期与 worker placement 的实现细节。
- :doc:`遥操作 <../guides/teleoperation>`：将操作员输入设备与机器人组合使用。
