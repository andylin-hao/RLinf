机器人架构
==========

编写任务时，只需知道机器人由哪些具名零部件组成，以及观测和动作使用哪些访问路径。接入新设备、让多个零部件共享硬件连接，或排查远程部署与资源回收问题时，再参考本页的实现细节。若尚未了解零部件名称与访问路径的关系，请先阅读 :doc:`robotics`。

从任务使用的访问路径出发
--------------------------

先看最常见的一对一情况：一条硬件连接只对应一个零部件。移动底盘可以直接加入 ``Robot``：

.. code-block:: python

   base = ExampleMobileBase(
       "tcp://mobile-base:7000",
       node_rank=0,
       worker_name="ExampleMobileBase-0-0",
   )
   robot = Robot(base=base)

构造底盘会创建一个尚未连接的 ``MobileBase`` 实例。该实例保存设备参数，``Connection`` 的 metaclass 则记录 ``node_rank`` 和 ``worker_name``，供后续 placement 使用。在调用 ``robot.connect()`` 前，代码不会导入厂商 SDK，也不会打开硬件。移动底盘本身能返回观测，policy 可直接访问，因此可以 ``base=base`` 传给 ``Robot``；参数名 ``base`` 会成为底盘的访问路径。

一条硬件连接也可以同时支持多个零部件。例如，GimArm 的关节和夹爪共用一条 CAN 总线，一条链路同时应答两者，任务仍应通过两条独立路径访问它们：

.. code-block:: text

   robot
   └── arm
       └── end_effector

这两种结构分别回答不同的问题：

- ``Robot`` 的组合关系决定 policy 可以观测和控制哪些零部件，以及访问它们的路径。
- 硬件连接描述哪些资源需要在同一节点打开，并且只能释放一次。

这两组名称分别保存在两个属性中。即使名称相同，两个属性的用途也不同：

- ``PartGroup`` 或 ``Robot`` 的 ``children`` 保存组合时传入的直接成员。每个 key 都会成为观测和动作路径的一段，例如 ``left.arm``；任务、policy 和数据集使用这些路径访问零部件。
- ``Connection`` 的 ``parts`` 列出同一硬件 session 支持的零部件。如果 connection 本身就是 ``RobotPart``，该 mapping 表示安装在它上的其他零部件；如果 connection 只表示共享 session，该 mapping 则列出可通过 ``part(name)`` 取出的零部件。这些名称属于 driver 内部，不会自动成为机器人的访问路径。

组合时，``Robot(arm=connection)`` 将能返回观测的机械臂以 ``arm`` 为名加入 ``Robot``。机械臂承载的其他零部件会同时出现在下一级；例如 GimArm 的夹爪位于 ``arm.end_effector``，因为它没有自己的链路，只能通过机械臂应答。决定层级的是连接方式，而不是安装位置：Franka Hand 同样固定在机械臂上，但它有独立的通信端点，因此与机械臂并列，而不在其下。访问路径、placement 和资源归属在 ``connect()`` 前已经确定，因此即使当前机器没有连接真机，也可以先检查组合结果。

``connection.part(name)`` 用于从一条本身不能返回观测的 connection 中选出某个零部件，例如从双臂 session 中选出左臂。只有这种情况才需要调用该方法。

应根据待组合对象选择对应写法：

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - 对象
     - 组合方式
     - 结果
   * - 不承载其他零部件的 ``RobotPart``，例如相机
     - ``Robot(wrist=camera)``
     - 相机的访问路径为 ``wrist``。
   * - 承载其他零部件的 ``RobotPart``，例如夹爪与关节共用总线的机械臂
     - ``Robot(arm=connection)``
     - 机械臂的访问路径为 ``arm``，夹爪的访问路径为 ``arm.end_effector``。
   * - 各自持有链路的两个零部件，例如机械臂与 Franka Hand
     - ``Robot(arm=arm, end_effector=hand)``
     - 两者各自成为顶层零部件，各自打开自己的连接。
   * - 本身不能返回观测的 ``Connection``，例如双臂 session
     - ``Robot(left=session.part("left"))``
     - 选出的零部件的访问路径为 ``left``。
   * - 已组合好的 ``PartGroup``
     - ``Robot(left=PartGroup(...))``
     - 该 group 位于 ``left``，其中的具名零部件继续使用下一级路径。

``part(name)`` 返回的就是一个 ``RobotPart``，中间没有任何需要机器人开发者构造或标注的类型。``PartGroup`` 接收 ``RobotPart`` 或另一个 ``PartGroup``；传入本身不能返回观测的裸 ``Connection`` 会被拒绝，并指出出错的参数名。

在组合结构中，``children`` 始终表示当前对象的直接下一级。对 ``RobotPart`` 而言，它表示该零部件承载的其他零部件；对 ``PartGroup`` 而言，它表示组合该 group 时传入的具名成员。因此，查找相机、生成结构说明或读取观测时，无需根据当前对象的具体类型切换遍历方式。

因此，机器人定义中只需声明机械臂：

.. code-block:: python

   class ExampleRobot(Robot):
       @classmethod
       def build_arms(cls, **config):
           return {"arm": ExampleArm(config["robot_ip"], node_rank=config["node_rank"])}

机器人 builder 为每条连接命名一个零部件。夹爪搭在机械臂链路上时，只声明机械臂即可：builder 再次声明夹爪，会使它变成机械臂的同级零部件，同时多出一份需要同步维护的清单；机械臂在运行时决定是否安装夹爪时，这份清单容易与实际硬件不一致。

Franka 属于另一种情况。它的末端执行器自行打开 session，因此 builder 同时声明机械臂和末端执行器，两者互不归属：

.. code-block:: python

   class FrankaRobot(Robot):
       @classmethod
       def build_arms(cls, *, robot_ip, node_rank, **config):
           return {
               "arm": cls.declare_arm(robot_ip, node_rank=node_rank, name=...),
               "end_effector": cls.declare_end_effector(
                   robot_ip, node_rank=node_rank, name=..., **config
               ),
           }

两种情况遵循同一条规则：为持有链路的对象命名；搭在链路上的零部件，随持有者一并加入。

因此，driver 的 ``parts`` mapping 只包含该零部件承载的其他零部件，不包含它自身。系统会拒绝将零部件自身加入 ``parts``，从而避免形成无法终止的递归结构。

核心类型
--------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 类型
     - 用途
   * - ``Connection``
     - 一条硬件连接。它记录运行节点，打开厂商 session，并在结束时释放资源；裸 ``Connection`` 不一定可以读取观测。
   * - ``RobotPart``
     - 可读取的 ``Connection``，实现 ``get_observation()``，并通过 ``observation_features`` 声明观测接口。
   * - ``ControllablePart``
     - 可接收动作的 ``RobotPart``，还实现 ``send_action()`` 和 ``action_features``。
   * - ``PartGroup``
     - 由具名 ``children`` 组成的可读、可控单元，可表示机械臂总成、躯干或其他嵌套结构。
   * - ``Robot``
     - 最外层的 ``PartGroup``，还管理注册，并知道每条连接运行在哪个节点上。

跨节点运行的零部件不使用单独的 API 类型。带 ``node_rank`` 的 connection 会在目标节点的 worker 中重新构造，本地对象则切换为由原 driver class 合成的子类。对象 identity 保持不变，``isinstance`` 仍会匹配原 driver 和设备类别，公开方法与 property 则转发到远程 worker。``Camera``、``MobileBase`` 等类别不需要为 placement 另外注册 proxy。

通过配置选择具体实现
--------------------

假设相机配置中写有 ``camera_type: zed``，机器人 builder 不应再维护一个导入所有相机 driver 的分支表。每个 driver 直接注册配置中使用的名称，再由设备类别完成查找：

.. code-block:: python

   @Camera.register("example")
   class ExampleCamera(BaseCamera):
       ...


   camera_cls = Camera.backend(camera_info.camera_type)
   camera = camera_cls(camera_info, node_rank=2)

所有设备类别都会继承 ``Connection.register()`` 和 ``backend()``，而且 registry 属于具体设备类别，例如 ``Camera``、``Arm`` 和 ``EndEffector``，因为配置里写的是一类设备，而不是某个基类。backend 名称不区分大小写；两个 class 注册同一名称时会直接报错。

如果某类设备具有固定的配置结构，还可以提供构建入口：``Camera.of()`` 接收 ``CameraInfo`` 并从中读出 backend；``EndEffector.of()`` 接收名称，以及安装它的机械臂所能提供的接入方式；``Arm.declare()`` 把机器人层面的机械臂配置映射到某个 backend 自己的构造函数上。这些映射都写在驱动里、紧挨着它所服务的构造函数，因此新增一个 backend 不需要改动任何负责选择 backend 的代码。

机械臂尤其适合采用这套机制，因为同一套硬件可能支持多种 backend。Franka 可以通过 libfranka 或 ROS 控制，因此两种实现都注册到 ``Arm``，机器人只需指定其中一种：

.. code-block:: python

   class FrankaRobot(Robot):
       BACKEND = "franka_ros"


   class DualFrankaRobot(FrankaRobot):
       BACKEND = "franky"

切换时只需修改 backend 名称。每个 backend 在自己的 ``declare()`` 中，将标准机械臂配置映射到相应构造函数；机器人无需了解某套实现启动 ROS package，而另一套实现打开 libfranka session。机械臂只接受机械臂自身的配置：向 ``declare()`` 传入 ``gripper_type`` 会被直接拒绝，而不是静默丢弃，因为这类配置属于与它并列组合的末端执行器。

Franka Hand 是本项目中唯一有两种驱动方式的设备 —— 经由 ROS topic，或经由自己的 libfranka session —— 因此 ``FrankaRobot`` 用 ``HAND_BACKENDS`` 记录从机械臂 backend 到末端执行器 driver 的对应关系。这一判断放在组合层，因为只有这里同时知道两者。如果配置直接写明 driver，例如 ``end_effector_type: franky_gripper``，则以配置为准。

支持硬件枚举的 driver 还可以通过 ``SDK`` 声明厂商模块，并实现 ``discover()``。公共 discovery 流程会据此报告缺失的 SDK，并在持有设备的节点上校验相机 ID。厂商模块仍应在 ``_open()`` 或 ``discover()`` 中导入，不应在模块导入阶段加载。

robotics 代码中有两种 registry，它们所命名的对象不同：

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - 命名对象
     - 公开 API
     - 用途
   * - 单个设备 backend
     - ``Camera.register()`` / ``Arm.register()`` 与 ``backend()``
     - 根据配置选择 ``realsense``、``franky`` 等 driver。
   * - 完整机器人类型
     - ``Robot.register_type()`` 与 ``Robot.of_type()``
     - 根据名称选择机器人组合及其 ``RobotConfig``；未传入自定义 class 时，同时创建标准 discovery 流程。

注册操作会关联 robot class、config class、discovery class 和 builder，但不会自动将 ``RobotConfig`` 实例转换为 builder 参数。``Robot.of_type()`` 和 ``build_robot()`` 会将接收到的关键字参数直接传给 ``build()``。因此，机器人的 builder 应提供明确的参数签名；如果 env 从 ``RobotInfo`` 获取硬件配置，应在一处显式完成参数转换。

env 层也使用相同的注册风格，但遥操作使用独立的 ``TeleopBackend`` registry。一个遥操作名称对应设备及其 binding，不代表机器人零部件。该 registry 位于 ``rlinf/envs/real/wrappers/teleop``，避免 Gymnasium 配置进入 robotics 层。

将共享连接中的零部件加入机器人
------------------------------

一条硬件 session 同时支持多个供 policy 使用的零部件时，通过 ``parts`` 声明它们。下面的机械臂本身可以读取观测，同时将夹爪作为单独的零部件提供给 ``Robot``：

.. code-block:: python

   class ExampleArm(ControllablePart):
       @property
       def parts(self) -> dict[str, RobotPart]:
           return {
               "end_effector": MethodEndEffector(
                   self, state_field="gripper_position"
               ),
           }

``end_effector`` 是 driver 内部的名称。机械臂本身不能再出现在 ``parts`` 中；将机械臂传给 ``Robot`` 时，组合已经确定它的位置。任务使用的访问路径由组合时的参数名决定：

.. code-block:: python

   connection = ExampleArm(
       "10.0.0.2",
       node_rank=1,
       worker_name="ExampleArm-0-0",
   )
   robot = Robot(
       arm=connection,
   )

传给 ``Robot`` 的关键字参数会进入 ``robot.children``。上述机器人的顶层路径为 ``arm``，末端执行器则位于 ``arm.end_effector``。裸 ``Connection`` 本身不能返回观测，因此没有 ``children``；``PartGroup`` 的组成项已保存在 ``children`` 中，因此其 ``parts`` 为空。当共享 session 本身不能返回观测时，通过 ``connection.part(...)`` 取出需要组合的零部件。

通过 ``part(name)`` 取出零部件时，共享 connection 会成为该 view 的 owner。因此 view 不需要实现 ``_open()``，也不应覆盖 ``connect()``。``parts`` 适用于这类借用共享 connection 的 view。

两种形式的区别，取决于框架对 class 的一个判断，而不取决于配置：它是否实现了 ``_open()``。实现了的零部件持有自己的链路，保留自己的 owner 和 ``node_rank``，需要显式加入 ``Robot`` 或某个 ``PartGroup``；没有实现的零部件，则由声明它的 connection 接管。通过 USB 连接的腕部相机和 Franka Hand 属于前者，基于机械臂自身状态的 ``MethodEndEffector`` 属于后者。

如果共享 session 本身没有可供 policy 使用的观测，应直接继承 ``Connection``，而不是 ``RobotPart``。Turtle2 的联动控制器采用这种形式：

.. code-block:: python

   connection = Turtle2Connection(
       50,
       tuple(camera_ids),
       node_rank=0,
       worker_name="Turtle2Connection-0-0",
   )
   robot = Turtle2Robot(
       left=PartGroup(
           arm=connection.part("left"),
           end_effector=connection.part("left_end_effector"),
       ),
       right=PartGroup(
           arm=connection.part("right"),
           end_effector=connection.part("right_end_effector"),
       ),
       wrist=connection.part("wrist_1"),
   )

通过 ``part(name)`` 取出的所有零部件都指向同一个 connection 实例，因此底层控制器只会打开和释放一次。

有些资源属于进程，而不属于某个 connection 对象。ROS 1 就是这里会遇到的情况：一个进程只有一个 node，因此同时使用 ROS 的机械臂和末端执行器，无论如何都会落在同一个 node 上。若由机械臂把 session 传给末端执行器，就会在两个本可独立的零部件之间引入依赖，因此基于 ROS 的零部件改为自行获取：

.. code-block:: python

   from rlinf.robotics.parts.transports.ros import ROSController


   class ExampleROSGripper(BaseGripper):
       def _open(self):
           self._ros = ROSController.shared()
           self._ros.connect_ros_channel(self._state_channel, JointState, self._on_state)
           return self._ros

``ROSController.shared()`` 会在文件锁保护下启动 ``roscore``（若尚未运行）并初始化 node，之后的调用方复用同一 controller。该 session 不会被关闭：``rospy`` 没有受支持的方式在 node 关闭后重新启动它。各零部件订阅和发布的 topic 互不相同，因此加入机械臂已经打开的 session 只是增加订阅，不会产生争用。
先确定部署位置，再打开硬件
----------------------------

placement 参数与硬件构造参数一起传入。构造过程不会访问设备；``Robot.connect()`` 会根据目标节点，选择在本进程打开现有对象，或在远程 worker 中按相同参数重新构造：

.. code-block:: python

   arm_connection = ExampleArm(
       "10.0.0.2",
       node_rank=1,
       worker_name="ExampleArm-0-0",
   )
   robot = Robot(
       arm=arm_connection,
       scene=RealSenseCamera(camera_info, node_rank=3),
   )

   print(robot.describe())
   robot.connect()
   try:
       observation = robot.get_observation()
   finally:
       robot.disconnect()

执行 ``connect()`` 时，机器人会为每个不同的 ``Connection`` 打开一次资源。没有 ``node_rank`` 的 connection 在本进程打开；带 ``node_rank`` 的 connection 在目标节点的 scheduler worker 中重新构造，组合中现有对象的具体 class 则切换为合成子类，公开方法和 property 转发到该 worker。对象 identity 不变，任务代码因而无需区分部署位置，``isinstance`` 的结果也与连接前一致。

资源归属由对象 identity 决定。每个零部件的 ``owner`` 都表示谁为它打开连接：自带链路的机械臂的 owner 是自身，搭在共享 session 上的 view 的 owner 则是该 session。机器人连接 owner，而不是逐个连接零部件，因此一条连接只打开一次、只释放一次。位于不同连接上的零部件可以并行调用；共用一条连接的零部件按声明顺序调用，避免并发访问不支持该模式的厂商 SDK。Franka 的机械臂和末端执行器各自持有链路，因此一次整机读取会同时取回两者，而不是依次等待。

零部件各自持有链路，也意味着它们可能在同一台机器的不同进程中打开。只要访问的是不同端点，这就是正常情况：libfranka 的机械臂控制和末端执行器本来就在不同端口上。问题出在两个零部件访问同一端点时，此时报错通常只提到 socket，而不会指出真正的原因。因此，独占某个端点的零部件会按端点申请 ``DeviceClaim``：第二个申请者会立即被拒绝，并被告知当前持有者是谁。

连接前检查组合结果
------------------

``Robot.describe()`` 读取已完成的组合结构，因此在打开任何硬件之前就能查看零部件路径、部署节点和资源归属：

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

``via`` 相同的行共用一个 ``Connection``。上面两行的 ``via`` 不同，由此可以最直接地看出末端执行器可以单独连接和恢复。连接后，跨节点零部件会显示合成 class 名称，例如 RemoteFrankaROSArm。零部件路径、``node`` 和 owner 保持不变，但完整输出字符串不是稳定的序列化格式，不应存储或解析该字符串。

``describe()`` 目前只显示组合结构、节点和资源归属，不显示 observation/action schema。若需检查字段和 shape，请使用 :doc:`添加机器人 <../extending/new_robot>` 中的 conformance 检查；该检查会通过 mock SDK 或真机打开连接后进行验证。

资源生命周期
------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 阶段
     - 执行内容
   * - 组合
     - 构造函数记录硬件参数和 placement，不导入厂商 SDK，也不打开硬件。
   * - 连接
     - 机器人在每条连接指定的节点上把它打开一次。跨节点的连接在该节点重新构造，本地对象随即变成它的 view。
   * - 使用
     - 机器人按名称读取、复位和控制各零部件，并根据资源归属决定能否并行。
   * - 断开
     - ``Connection`` 释放 ``_open()`` 实际返回的设备。跨节点的连接先在它所在的节点上关闭，再停掉 worker，然后恢复成一个未打开的普通对象。

``_open()`` 返回的厂商对象会原样传给 ``_release(device)``。清理逻辑应释放参数 ``device``，不要重新从 ``self`` 读取。

driver 应实现 ``_open()`` 与 ``_release()``，不要覆盖 ``connect()`` 和 ``disconnect()``。后两个方法负责选择设备的运行节点；覆盖它们会绕过跨节点部署流程。例如，在 ``super().connect()`` 返回后启动线程，线程会运行在持有零部件对象的进程，而非实际持有设备的 worker。设备类别如需在 driver 生命周期外增加处理，可实现 ``_opened()`` 和 ``_closing()``。``BaseCamera`` 使用这两个 hook 启停取流循环，确保循环与相机运行在同一节点。

如果后续 connection 打开失败，``Robot.connect()`` 会回滚此前已成功打开的 connection。但如果 driver 在 ``_open()`` 内部只完成了部分初始化就抛出异常，driver 仍需要自行释放已获取的资源，因为此时尚没有可供机器人关闭的完整 connection。排除故障后，可对同一对象再次调用 ``connect()``。已成功关闭的 connection 也可重复调用 ``disconnect()``，并保持可重新连接状态。

调用设备专有方法
--------------------

placement 的两端都由 driver class 派生而来，因此标准零部件接口之外的公开方法也会跨节点转发。通过零部件的 ``owner`` 取得其所在的连接，再调用所需方法：

.. code-block:: python

   controller = robot.child("arm").owner
   controller.is_robot_up()
   controller.reset_joint(home_qpos)

无论这条机械臂在本机还是在其他节点上，写法都相同，也无需额外拆包返回值。任务代码仍应通过标准观测和动作路径访问零部件；初始化、诊断或无法纳入通用零部件接口的厂商操作才直接使用连接。

保持导入边界
------------

零部件模块不能导入 ``rlinf.scheduler`` 和 Gymnasium。只有当一条连接指定了节点时，``Connection.connect()`` 才会延迟加载桥接层 ``rlinf/robotics/placement/handles.py``。

这条规则的意义在于依赖方向。scheduler 是通用框架，robotics 只是它的一个扩展，因此 scheduler 从不导入本包：它按配置里写明的名字导入硬件策略模块，再调用这些模块注册进来的 discovery 类。Gymnasium 则位于另一侧，属于消费机器人的 env 层。只有组合层——placement、discovery、机器人构建器——会反向导入这两者，驱动因而能够作为纯粹的硬件代码被阅读和测试。

上述限制不包括 Ray。Ray 是 RLinf 的基础依赖，运行节点已经安装该依赖。该规则也不保证导入零部件模块时不会加载其他模块；零部件仍可使用 ``rlinf.utils`` 中的 ``get_logger`` 等公共工具。``tests/unit_tests/test_robotics.py`` 会检查两个方向的导入边界。

代码位置
--------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 路径
     - 内容
   * - ``robotics/parts/base.py``
     - ``Connection``、``RobotPart``、``ControllablePart``、``PartGroup``，以及组合阶段的类型检查和 driver registry。
   * - ``robotics/parts/arms/``
     - ``Arm`` 类别与 ``BaseArm``，以及注册到它们之上的各个 backend：Franky、Franka ROS、GimArm 和联动控制器。
   * - ``robotics/parts/cameras/``
     - 相机生命周期，以及 RealSense、ZED 和 Lumos 实现。
   * - ``robotics/parts/end_effectors/``
     - 夹爪和灵巧手。
   * - ``robotics/parts/mobility/``
     - ``MobileBase`` 类别与移动平台驱动。
   * - ``robotics/parts/views.py``
     - 将共享厂商 session 转换为 ``RobotPart`` 接口的 ``MethodArm``、``MethodEndEffector`` 和 ``MethodCamera``。
   * - ``robotics/placement/``
     - 承载连接的 worker，以及连接跨节点后变成的 view，两者都由 driver class 合成。
   * - ``robotics/robot.py``
     - 最外层组合、``describe()`` 和生命周期。
   * - ``robotics/discovery/``
     - 机器人类型注册、标准硬件枚举、环境变量补全与配置查找。

后续阅读
--------

- :doc:`添加机器人 <../extending/new_robot>`：按照实际接入顺序应用本页介绍的类型和机制。
- :doc:`placement <placement>`：了解调度器资源如何映射到节点和 GPU。
- :doc:`遥操作 <../guides/teleoperation>`：了解环境侧如何组合操作者设备与 binding。
