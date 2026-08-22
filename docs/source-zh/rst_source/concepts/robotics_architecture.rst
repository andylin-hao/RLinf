机器人架构
==========

编写任务时，只需将机器人理解为一棵按名称组织的部件树。当需要接入设备、让多个部件共享硬件连接，或排查远程部署与资源回收问题时，再参考本页的实现模型。若尚未了解部件树，请先阅读 :doc:`robotics`。

从对外公开的部件树出发
------------------------

先看最常见的一对一情况：一条硬件连接只对应一个逻辑部件。移动底盘可以直接加入机器人树：

.. code-block:: python

   base = ExampleMobileBase(
       "tcp://mobile-base:7000",
       node_rank=0,
       worker_name="ExampleMobileBase-0-0",
   )
   robot = Robot(base=base)

构造底盘会创建一个尚未连接的 ``MobileBase`` 实例。该实例保存设备参数，``Connection`` 的 metaclass 则记录 ``node_rank`` 和 ``worker_name``，供后续 placement 使用。在调用 ``robot.connect()`` 前，代码不会导入厂商 SDK，也不会打开硬件。移动底盘本身就是 policy 需要访问的逻辑部件，因此可直接以 ``base=base`` 传给 ``Robot``；参数名 ``base`` 即它在公开部件树中的路径。

一条硬件连接也可以对应多个逻辑部件。例如，机械臂和夹爪共用一个 ROS session 时，任务仍应看到两个独立部件：

.. code-block:: text

   robot
   └── arm
       └── end_effector

这两种结构分别回答不同的问题：

- 机器人树描述 policy 可以观测和控制哪些部件。
- 硬件连接描述哪些资源需要在同一节点打开，并且只能释放一次。

这两组名称分别保存在两个属性中。即使名称相同，两个属性的用途也不同：

- ``PartGroup`` 或 ``Robot`` 将公开部件树保存在 ``children`` 中。每个 key 都会成为观测和动作路径，例如 ``left.arm``；任务、policy 和数据集使用这些名称访问部件。
- ``Connection`` 通过 ``parts`` 列出同一硬件 session 支持的逻辑部件。对可读取的部件而言，该映射表示安装在它上面的其他部件；对不可读取的共享 session 而言，该映射列出可通过 ``part(name)`` 取出的部件。这些名称属于 driver 内部，不会自动成为机器人路径。

组合时，``Robot(arm=connection)`` 为一个部件指定名称，而搭在这个部件上的部件会同时进入树中，并位于它的下一层。机械臂的夹爪出现在 ``arm.end_effector``，因为夹爪安装在机械臂上。公开路径、placement 和资源归属在 ``connect()`` 前已经确定，因此即使当前机器没有连接真机，也可以先检查组合结果。

``connection.part(name)`` 用于从一条本身不是部件的链路里挑出某个部件，例如驱动两条机械臂的 session。只有这种情况才需要它。

应根据待组合对象选择对应写法：

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - 对象
     - 组合方式
     - 结果
   * - 不包含子部件的部件，例如相机
     - ``Robot(wrist=camera)``
     - 该部件以 ``wrist`` 为名称进入机器人树。
   * - 承载其他部件的部件，例如带夹爪的机械臂
     - ``Robot(arm=connection)``
     - 机械臂进入 ``arm``，它的夹爪进入 ``arm.end_effector``。
   * - 本身不是部件的链路，例如双臂 session
     - ``Robot(left=session.part("left"))``
     - 指定的部件以 ``left`` 为名称进入机器人树。
   * - 已经组合好的子树
     - ``Robot(left=PartGroup(...))``
     - ``PartGroup`` 及其具名部件共同进入 ``left`` 路径。

``part(name)`` 返回的就是一个 ``RobotPart``，中间没有任何需要机器人开发者构造或标注的类型。``PartGroup`` 接收 ``RobotPart`` 或另一个 ``PartGroup``；传入不可读取的裸 ``Connection`` 会被拒绝，并指出出错的参数名。

``children`` 始终表示部件树中的下一层：对部件而言，它表示该部件承载的子部件；对 ``PartGroup`` 而言，它表示组合该 group 时传入的具名成员。因此，在描述部件树、查找相机或读取观测时，无需根据当前对象的具体类型切换遍历方式。

因此，机器人定义中只需声明机械臂：

.. code-block:: python

   class ExampleRobot(Robot):
       @classmethod
       def build_arms(cls, **config):
           return {"arm": ExampleArm(config["robot_ip"], node_rank=config["node_rank"])}

机械臂通过 ``parts`` 声明夹爪后，组合机械臂时会一并加入该夹爪。如果机器人再次声明夹爪，夹爪会成为机械臂的同级部件，而不是其子部件，同时还会形成一份需要同步维护的重复清单。机械臂在运行时决定是否安装夹爪，或后续增加新部件时，这份清单容易遗漏实际存在的部件。

因此，driver 的 ``parts`` 映射只包含该部件承载的子部件，不包含部件自身。系统会拒绝将部件自身加入 ``parts``，从而避免形成无法终止的递归结构。

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
     - 由具名 ``children`` 组成的可读、可控子树，可表示机械臂组件、躯干或其他嵌套单元。
   * - ``Robot``
     - 最外层的 ``PartGroup``，还管理注册，并知道每条连接运行在哪个节点上。

跨节点运行的部件没有单独的公开类型。带 ``node_rank`` 的 connection 会在目标节点的 worker 中重新构造，本地对象则切换为由原 driver class 合成的子类。对象 identity 保持不变，``isinstance`` 仍会匹配原 driver 和设备类别，公开方法与 property 则转发到远程 worker。``Camera``、``MobileBase`` 等类别不需要为 placement 另外注册 proxy。

通过配置选择具体实现
--------------------

假设相机配置中写有 ``camera_type: zed``，机器人 builder 不应再维护一个导入所有相机 driver 的分支表。每个 driver 直接注册配置中使用的名称，再由设备类别完成查找：

.. code-block:: python

   @Camera.register("example")
   class ExampleCamera(BaseCamera):
       ...


   camera_cls = Camera.backend(camera_info.camera_type)
   camera = camera_cls(camera_info, node_rank=2)

所有设备类别都会继承 ``Connection.register()`` 和 ``backend()``，而且 registry 属于**类别**本身——``Camera``、``Arm``、``EndEffector``——因为配置里写的是一类设备，而不是某个基类。backend 名称不区分大小写；两个 class 注册同一名称时会直接报错。

如果某类设备具有固定的配置结构，还可以提供构建入口：``Camera.of()`` 接收 ``CameraInfo`` 并从中读出 backend；``EndEffector.of()`` 接收名称，以及安装它的机械臂所能提供的接入方式；``Arm.declare()`` 把机器人层面的机械臂配置映射到某个 backend 自己的构造函数上。这些映射都写在驱动里、紧挨着它所服务的构造函数，因此新增一个 backend 不需要改动任何负责选择 backend 的代码。

机械臂尤其适合采用这套机制，因为同一套硬件可能支持多种 backend。Franka 可以通过 libfranka 或 ROS 控制，因此两种实现都注册到 ``Arm``，机器人只需指定其中一种：

.. code-block:: python

   class FrankaRobot(Robot):
       BACKEND = "franka_ros"


   class DualFrankaRobot(FrankaRobot):
       BACKEND = "franky"

切换时只需修改 backend 名称。每个 backend 在自己的 ``declare()`` 中，将标准机械臂配置映射到相应构造函数；机器人无需了解某套实现需要 ROS package，而另一套实现需要夹爪串口。backend 无法满足的配置项应直接拒绝，不能静默丢弃，否则实际使用的末端执行器可能与配置不一致。

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
     - 根据名称选择机器人树及其 ``RobotConfig``；未传入自定义 class 时，同时创建标准 discovery 流程。

注册操作会关联 robot class、config class、discovery class 和 builder，但不会自动将 ``RobotConfig`` 实例转换为 builder 参数。``Robot.of_type()`` 和 ``build_robot()`` 会将接收到的关键字参数直接传给 ``build()``。因此，机器人的 builder 应提供明确的参数签名；如果 env 从 ``RobotInfo`` 获取硬件配置，应在一处显式完成参数转换。

env 层也使用相同的注册风格，但遥操作使用独立的 ``TeleopBackend`` registry。一个遥操作名称对应设备及其 binding，而不是机器人部件。该 registry 位于 ``rlinf/envs/real/wrappers/teleop``，避免 Gymnasium 配置进入 robotics 层。

将共享硬件连接映射到机器人树
----------------------------

一条硬件 session 对应多个面向 policy 的部件时，通过 ``parts`` 声明这些部件。下面的机械臂本身可以读取观测，同时将夹爪呈现为单独的部件：

.. code-block:: python

   class ExampleArm(ControllablePart):
       @property
       def parts(self) -> dict[str, RobotPart]:
           return {
               "end_effector": MethodEndEffector(
                   self, state_field="gripper_position"
               ),
           }

``end_effector`` 是 driver 内部的名称。机械臂本身不能再出现在 ``parts`` 中；将机械臂传给 ``Robot`` 时，组合已经建立了它在树中的位置。对外公开的路径由组合时的参数名决定：

.. code-block:: python

   connection = ExampleArm(
       "10.0.0.2",
       node_rank=1,
       worker_name="ExampleArm-0-0",
   )
   robot = Robot(
       arm=connection,
   )

传给 ``Robot`` 的关键字参数会进入 ``robot.children``。上述机器人的顶层路径为 ``arm``，末端执行器则位于 ``arm.end_effector``。裸 ``Connection`` 不属于部件树，因此没有 ``children``；``PartGroup`` 的组成项已保存在 ``children`` 中，因此其 ``parts`` 为空。当共享 session 本身不可读取时，通过 ``connection.part(...)`` 取出需要组合的部件。

通过 ``part(name)`` 取出部件时，共享 connection 会成为该 view 的 owner。因此 view 不需要实现 ``_open()``，也不应覆盖 ``connect()``。``parts`` 应用于这类借用共享 connection 的 view。如果设备拥有独立链路，例如通过 USB 连接的腕部相机，应将其显式组合为 ``Robot`` 或某个 ``PartGroup`` 的 child。这样相机会保留自己的 owner，``Robot.connect()`` 也会在其指定节点上打开该设备。

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

所有部件选择都指向同一个 connection 实例，因此底层控制器只会打开和释放一次。

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

执行 ``connect()`` 时，机器人会为每个不同的 ``Connection`` 打开一次资源。没有 ``node_rank`` 的 connection 在本进程打开；带 ``node_rank`` 的 connection 在目标节点的 scheduler worker 中重新构造，树中已有对象的具体 class 则切换为合成子类，公开方法和 property 转发到该 worker。对象 identity 不变，任务代码因而无需区分部署位置，``isinstance`` 的结果也与连接前一致。

资源归属由对象 identity 决定。每个部件都用 ``owner`` 回答“谁代它打开连接”：自带链路的机械臂回答自己，搭在共享 session 上的 view 回答那个 session。机器人连接的是 owner 而不是部件，因此一条连接只打开一次、只释放一次。不同连接上的部件可以并行调用；共用一条连接的部件按声明顺序调用，避免并发访问不支持该模式的厂商 SDK。

连接前检查组合结果
------------------

``Robot.describe()`` 读取已组合的部件树，因此在打开任何硬件之前就能查看部署节点和资源归属：

.. code-block:: text

   FrankaRobot
   └── arm                 FrankaROSArm         node=1     via FrankaROSArm#1
       └── end_effector    MethodEndEffector    node=1     via FrankaROSArm#1

``via`` 相同的行共用一个 ``Connection``。连接后，跨节点部件会显示合成 class 名称，例如 RemoteFrankaROSArm。部件路径、``node`` 和 owner 保持不变，但完整输出字符串不是稳定的序列化格式，不应存储或解析该字符串。

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
     - 机器人按名称读取、复位和控制部件树，并根据资源归属决定能否并行。
   * - 断开
     - ``Connection`` 释放 ``_open()`` 实际返回的设备。跨节点的连接先在它所在的节点上关闭，再停掉 worker，然后恢复成一个未打开的普通对象。

``_open()`` 返回的厂商对象会原样传给 ``_release(device)``。清理逻辑应释放参数 ``device``，不要重新从 ``self`` 读取。

driver 应实现 ``_open()`` 与 ``_release()``，不要覆盖 ``connect()`` 和 ``disconnect()``。后两个方法负责选择设备的运行节点；覆盖它们会绕过跨节点部署流程。例如，在 ``super().connect()`` 返回后启动线程，线程会运行在持有部件对象的进程，而非实际持有设备的 worker。设备类别如需在 driver 生命周期外增加处理，可实现 ``_opened()`` 和 ``_closing()``。``BaseCamera`` 使用这两个 hook 启停取流循环，确保循环与相机运行在同一节点。

如果后续 connection 打开失败，``Robot.connect()`` 会回滚此前已成功打开的 connection。但如果 driver 在 ``_open()`` 内部只完成了部分初始化就抛出异常，driver 仍需要自行释放已获取的资源，因为此时尚没有可供机器人关闭的完整 connection。排除故障后，可对同一对象再次调用 ``connect()``。已成功关闭的 connection 也可重复调用 ``disconnect()``，并保持可重新连接状态。

直接在部件上调用设备专有方法
----------------------------

placement 的两端都由 driver class 派生而来，因此标准部件接口之外的公开方法同样会跨节点转发，理由和 driver 本身拥有这些方法完全一样。先问部件它搭在哪条连接上，再调用即可：

.. code-block:: python

   controller = robot.child("arm").owner
   controller.is_robot_up()
   controller.reset_joint(home_qpos)

无论这条机械臂在本机还是在别的节点上，写法都一样，也没有返回值需要额外拆包。任务代码仍应通过标准 observation/action tree 访问部件；初始化、诊断或无法纳入通用部件接口的厂商操作才直接使用连接。

保持导入边界
------------

部件模块不能导入 ``rlinf.scheduler`` 和 Gymnasium。只有当一条连接指定了节点时，``Connection.connect()`` 才会延迟加载桥接层 ``rlinf/robotics/placement/handles.py``。

这条规则的意义在于依赖方向。scheduler 是通用框架，robotics 只是它的一个扩展，因此 scheduler 从不导入本包：它按配置里写明的名字导入硬件策略模块，再调用这些模块注册进来的 discovery 类。Gymnasium 则位于另一侧，属于消费机器人的 env 层。只有组合层——placement、discovery、机器人构建器——会反向导入这两者，驱动因而能够作为纯粹的硬件代码被阅读和测试。

上述限制不包括 Ray。Ray 是 RLinf 的基础依赖，运行节点已经安装该依赖。该规则也不保证导入部件模块时不会加载其他模块；部件仍可使用 ``rlinf.utils`` 中的 ``get_logger`` 等公共工具。``tests/unit_tests/test_robotics.py`` 会检查两个方向的导入边界。

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
     - 将共享厂商 session 呈现为部件的 ``MethodArm``、``MethodEndEffector`` 和 ``MethodCamera``。
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
