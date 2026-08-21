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
   ├── arm
   └── end_effector

这两种结构分别回答不同的问题：

- 机器人树描述 policy 可以观测和控制哪些部件。
- 硬件连接描述哪些资源需要在同一节点打开，并且只能释放一次。

这两组名称分别保存在两个属性中：

- ``PartGroup`` 或 ``Robot`` 将公开部件树保存在 ``children`` 中。每个 key 都会成为观测和动作路径，例如 ``left.arm``；任务、policy 和数据集使用这些名称访问部件。
- ``Connection`` 将同一硬件 session 对应的逻辑部件保存在 ``parts`` 中。这些名称属于驱动内部，不会自动成为机器人路径。

组合时，``connection.part("arm")`` 记录需要从连接中选择的部件，``Robot(arm=...)`` 再为该部件指定公开名称 ``arm``。这个选择会一直保持未解析状态，直到 ``Robot.connect()`` 打开连接并确定其中实际包含的部件。

应根据待组合对象选择对应写法：

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - 对象
     - 组合方式
     - 结果
   * - 一个可读取的完整部件，例如移动底盘或相机
     - ``Robot(base=base)``
     - 该部件以 ``base`` 为名称进入机器人树。
   * - 一条对应多个部件的硬件连接
     - ``Robot(arm=connection.part("arm"))``
     - 选中的部件以 ``arm`` 为名称进入机器人树。
   * - 已经组合好的子树
     - ``Robot(left=PartGroup(...))``
     - ``PartGroup`` 及其具名部件共同进入 ``left`` 路径。

``part(name)`` 返回的是内部使用的延迟选择，机器人开发者无需构造或标注其具体类型。``PartGroup`` 可以接收该选择、一个 ``RobotPart`` 或另一个 ``PartGroup``。如果传入不可读取的裸 ``Connection``，构造函数会拒绝该值，并指出出错的参数名。

有些对象既是硬件连接，也是可读取的部件。例如，机械臂 session 可以返回机械臂自身的观测，同时包含一个末端执行器。此类对象可以直接组合；如果其中包含多个部件，连接后会在对应参数名下解析为 ``PartGroup``。需要将这些部件设为同级路径或分别重命名时，应使用 ``part(name)`` 显式选择。

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

跨节点运行的部件没有单独的类型。带 ``node_rank`` 的连接会在目标节点的 worker 中重新构造，而手上已有的那个对象随即变成它的一个 view：class 不变，``isinstance`` 不变，只是所有公开调用都改为跨进程执行。``Camera``、``MobileBase`` 这类设备类别不需要为此注册任何东西，因为 view 本身就是由 driver class 派生出来的。

通过配置选择具体实现
--------------------

假设相机配置中写有 ``camera_type: zed``，机器人 builder 不应再维护一个导入所有相机 driver 的分支表。每个 driver 直接注册配置中使用的名称，再由设备类别完成查找：

.. code-block:: python

   @BaseCamera.register("example")
   class ExampleCamera(BaseCamera):
       ...


   camera_cls = BaseCamera.backend(camera_info.camera_type)
   camera = camera_cls(camera_info, node_rank=2)

``BaseCamera``、``MobileBase`` 等设备类别都会继承 ``Connection.register()`` 和 ``backend()``。backend 名称不区分大小写；两个 class 注册同一名称时会直接报错。如果某类设备具有固定的配置结构，还可以提供 ``of()`` 或 ``declare()`` 作为便捷入口。例如，``BaseCamera.of()`` 读取 ``CameraInfo.camera_type``，``BaseCamera.declare()`` 则返回可直接加入机器人树的相机。

支持硬件枚举的 driver 还可以通过 ``SDK`` 声明厂商模块，并实现 ``discover()``。公共 discovery 流程会据此报告缺失的 SDK，并在持有设备的节点上校验相机 ID。厂商模块仍应在 ``_open()`` 或 ``discover()`` 中导入，不应在模块导入阶段加载。

robotics 代码中有三种 registry，它们所命名的对象不同：

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - 命名对象
     - 公开 API
     - 用途
   * - 单个设备 backend
     - ``BaseCamera.register()`` 与 ``BaseCamera.backend()``
     - 根据设备配置选择 ``realsense``、``zed`` 等 driver。
   * - 完整机器人类型
     - ``Robot.register_type()`` 与 ``Robot.of_type()``
     - 根据名称选择机器人树及其 ``RobotConfig``；未传入自定义 class 时，同时创建标准 discovery 流程。

将共享硬件连接映射到机器人树
----------------------------

一条硬件 session 对应多个面向 policy 的部件时，通过 ``parts`` 声明这些部件。下面的机械臂本身可以读取观测，同时将夹爪呈现为单独的部件：

.. code-block:: python

   class ExampleArm(ControllablePart):
       @property
       def parts(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(
                   self, state_field="gripper_position"
               ),
           }

``arm`` 和 ``end_effector`` 是驱动内部的名称。组合机器人时，再决定对外公开的路径：

.. code-block:: python

   connection = ExampleArm(
       "10.0.0.2",
       node_rank=1,
       worker_name="ExampleArm-0-0",
   )
   robot = Robot(
       arm=connection.part("arm"),
       end_effector=connection.part("end_effector"),
   )

传给 ``Robot`` 的关键字参数会进入 ``robot.children``，因此上述机器人最终公开 ``arm`` 和 ``end_effector``。裸 ``Connection`` 不组合任何部件，所以没有 ``children``；``PartGroup`` 中的部件已经位于 ``children``，因此其 ``parts`` 为空。``connection.part(...)`` 是两套命名体系相交的位置。

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
       arm=arm_connection.part("arm"),
       end_effector=arm_connection.part("end_effector"),
       scene=RealSenseCamera(camera_info, node_rank=3),
   )

   print(robot.describe())
   robot.connect()
   try:
       observation = robot.get_observation()
   finally:
       robot.disconnect()

执行 ``connect()`` 时，机器人会为每个不同的 ``Connection`` 打开一次资源。没有 ``node_rank`` 的连接在本进程打开；带 ``node_rank`` 的连接在目标节点的调度器 worker 中重新构造，树里那个对象则变成自身 class 的一个合成子类，公开方法和 property 全部转发到该 worker。树中没有任何对象被替换，任务代码因而不必区分部署位置，``isinstance`` 的结果也和连接之前一致。

资源归属由对象 identity 决定。每个部件都用 ``owner`` 回答“谁代它打开连接”：自带链路的机械臂回答自己，搭在共享 session 上的 view 回答那个 session。机器人连接的是 owner 而不是部件，因此一条连接只打开一次、只释放一次。不同连接上的部件可以并行调用；共用一条连接的部件按声明顺序调用，避免并发访问不支持该模式的厂商 SDK。

连接前检查组合结果
------------------

``Robot.describe()`` 读取组合好的部件树，而这棵树在 ``connect()`` 前后持有的是同一批对象，因此没有机器人在场时也能回答节点和资源归属：

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  MethodGripper        node=1     via FrankaROSArm#1

``via`` 相同的行共用一个 ``Connection``。连接之后，跨节点的部件会显示它的 view class，例如上面的 ``FrankaROSArm`` 会变成 ``RemoteFrankaROSArm``，一眼就能看出哪些部件跑在别的机器上。

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

要实现的始终是这两个方法，而不是 ``connect()`` 和 ``disconnect()``。后面这一对决定设备在哪里运行，覆盖它们等于放弃跨节点部署的能力：在 ``super().connect()`` 之后启动的线程会跑在持有部件的机器上，而不是持有设备的那台。设备类别若要在 driver 之外再包一层，用 ``_opened()`` 和 ``_closing()``：``BaseCamera`` 的取流循环就在这里启停，无论相机落在哪个节点，循环都跟着相机走。

连接过程要么全部成功，要么全部回滚。如果后续连接启动失败，``Robot.connect()`` 会关闭此前已经打开的资源，并恢复原始组合。排除故障后，可以对同一对象再次调用 ``connect()``；``disconnect()`` 也支持重复调用，并将机器人恢复到可重新连接的状态。

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

Ray 不在此列。它是 RLinf 的基础依赖，运行 RLinf 的机器上一定有 Ray，禁止这个名字并无收益。这条规则约束的也只是部件源码可以写出哪些名字，而不保证导入一个部件不会加载任何东西：部件可以使用 ``rlinf.utils`` 里的 ``get_logger`` 等工具，它们会触及更深的层次。``tests/unit_tests/test_robotics.py`` 会检查两个方向的导入边界。

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
     - 机械臂与联动控制器实现。
   * - ``robotics/parts/cameras/``
     - 相机生命周期，以及 RealSense、ZED 和 Lumos 实现。
   * - ``robotics/parts/end_effectors/``
     - 夹爪和灵巧手。
   * - ``robotics/parts/mobility/``
     - ``MobileBase`` 类别与移动平台驱动。
   * - ``robotics/parts/views.py``
     - 将共享厂商 session 呈现为部件的 ``MethodArm``、``MethodGripper`` 和 ``MethodCamera``。
   * - ``robotics/placement/``
     - 承载连接的 worker，以及连接跨节点后变成的 view，两者都由 driver class 合成。
   * - ``robotics/robot.py``
     - 最外层组合、声明快照、``describe()`` 和生命周期。
   * - ``robotics/discovery/``
     - 机器人类型注册、标准硬件枚举、环境变量补全与配置查找。

后续阅读
--------

- :doc:`添加机器人 <../extending/new_robot>`：按照实际接入顺序应用本页介绍的类型和机制。
- :doc:`placement <placement>`：了解调度器资源如何映射到节点和 GPU。
- :doc:`遥操作 <../guides/teleoperation>`：了解环境侧如何组合操作者设备与 binding。
