机器人模型
==========

把机器人拆成一组具名部件后，RLinf 就能统一处理节点放置、并行读取和资源回收，单臂、
双臂等变体也不必各写一套逻辑。先看几个 Franka 的例子，直观理解这种拆法能省掉哪些
工作，再往下看完整模型。

组装一台机器人
--------------

接入一款机器人时，先写清它带有哪些部件。连接、放置和清理沿用公共逻辑，机器人类只
描述硬件本身：

.. code-block:: python

   class FrankaRobot(Robot):
       ROBOT_TYPE = "Franka"
       BACKEND = "franka_ros"

       @classmethod
       def build_arms(cls, *, robot_ip, node_rank, **config) -> dict[str, RobotPart]:
           return {"arm": cls.declare_arm(robot_ip, node_rank=node_rank, name="arm")}


   FrankaRobot.register(FrankaConfig, FrankaDiscovery)

``build`` 汇总各个 ``build_*`` 方法返回的部件；随后，``connect`` 在指定节点上构建并
连接这些部件，``disconnect`` 再统一释放资源。这套流程写在公共层，每接一款机器人都
能直接复用。

单臂和多臂是同一套代码
----------------------

双臂 Franka 不需要另一套机器人框架。它继承单臂版本，只改 ``build_arms`` 返回的条目：

.. code-block:: python

   class DualFrankaRobot(FrankaRobot):
       ROBOT_TYPE = "DualFranka"
       BACKEND = "franky"

       @classmethod
       def build_arms(cls, *, left_robot_ip, right_robot_ip,
                      left_node_rank, right_node_rank, **config):
           return {
               "left": cls.declare_arm(left_robot_ip, node_rank=left_node_rank,
                                       name="left"),
               "right": cls.declare_arm(right_robot_ip, node_rank=right_node_rank,
                                        name="right"),
           }

``declare_arm``、``build_cameras``、``build`` 以及放置、并行读取、资源回收都直接继承。
以后要加第三条机械臂，也只是多一个条目，不会再分叉出一套“三臂版”流程。公共行为始终
只有一份，机械臂数量不会带来额外维护成本。

切换控制后端只需一行
--------------------

设置 ``BACKEND``，即可选择声明最终构建哪种机械臂。所有机器人变体都通过
``declare_arm`` 创建机械臂，所以后端只接一次，单臂和六臂可以共用：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``BACKEND``
     - 它声明的机械臂部件
   * - ``"franka_ros"``
     - ``FrankaROSArm``：通过 ROS 做笛卡尔阻抗控制。
   * - ``"franky"``
     - ``FrankyArm``：通过 libfranka 做关节和笛卡尔控制。

要接第三种后端，只需新增一个机械臂部件，并在 ``FRANKA_BACKENDS`` 中登记。机器人类
从未写死具体机械臂类名，因此不用跟着修改；选择后端这件事只留在映射表中。

机械臂整体到位
--------------

Franka 夹爪与机械臂共用连接，所以应当归在机械臂下面。把机械臂装进机器人时，夹爪会
作为它的子部件一起出现：

.. code-block:: python

   robot = FrankaRobot.build(robot_ip="10.0.0.1", node_rank=1, ...)

   robot.parts                  # {"arm": ...}
   robot.part("arm").parts      # {"arm": ..., "gripper": ...}

这里的 ``build`` 完全不用关心夹爪。机械臂描述这条连接能访问哪些组件，机器人只选择
要带哪些顶层硬件。连接细节因此不会散落到各个机器人定义里。

任何部件都能放在任何节点
------------------------

部件放在哪里，取决于设备接在哪台机器上，与部件类型无关。任何部件都可以指定
``node_rank``：

.. code-block:: python

   robot = FrankaRobot(
       arm=FrankaROSArm.at(robot_ip, node_rank=1),      # 放在机械臂那台 NUC 上
       wrist=RealSenseCamera.at(info, node_rank=3),     # 放在它插接的机器上
   )

使用独立连接的部件可以并行读取；共用连接的部件则按声明顺序访问。厂商 SDK 通常不
允许在同一条链路上并发调用，这种安排既吃到了安全的并行度，也不用每个机器人自己写
调度逻辑。

核心思想
--------

**把每个物理组件都建模为部件（part）。** 机械臂、夹爪、相机和移动底盘都属于
``RobotPart``。部件知道如何连接硬件、读取观测；可控部件还接收动作。这里不再额外套一
层“驱动”，因为部件本身已经表达了上层真正需要的硬件能力。

硬件连接与物理组件并不总是一一对应。比如，一台联动双臂控制器可能通过一条 ROS 连接
操纵两条机械臂、两个夹爪和两个腕部相机。此时，让连接对应的部件列出它暴露的所有
组件即可，不必再发明一种“持有连接的对象”：

.. code-block:: python

   @property
   def parts(self) -> dict[str, RobotPart]:
       return {
           "left": MethodArm(self, commands={"tcp_pose": "move_left_arm"}),
           "right": MethodArm(self, commands={"tcp_pose": "move_right_arm"}),
           "left_end_effector": MethodGripper(self, state_field="follow1_pos"),
       }

是否持有连接，只是某些部件的内部特征，不需要进入类型体系。

**用具名部件组合机器人。** **把任何部件放到所需节点。** 连同第一条规则，这套模型用
三件事就把硬件访问、组合结构和集群放置拆开，不需要再增加中间层。

抽象一览
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 抽象
     - 含义
   * - ``RobotPart``
     - 所有物理部件的基类，定义 ``connect``、``get_observation``、``disconnect`` 和
       ``reset``；``observation_features`` 描述观测的数据结构。
   * - ``ControllablePart``
     - 可以通过 ``send_action`` 接收命令，并用 ``action_features`` 描述命令格式的部件。
   * - ``Camera`` / ``EndEffector`` / ``MobileBase`` / ``LeggedBase``
     - 更具体的部件类型。组合结构或远程代理需要区分设备类别时，会用到这些类型。
   * - ``parts``
     - 当前部件暴露的具名子部件。硬件用它列出一条连接驱动的组件，``Group`` 用它返回
       组合进来的部件，叶子部件则返回 ``{}``。
   * - ``Group``
     - 由其他具名部件组成的部件。同一种结构既能表示机械臂或躯干，也能表示整台机器人。
   * - ``Robot``
     - 最外层的组。它带有注册类型，可以按硬件配置组装自身，并管理放置时创建的连接。
   * - ``at()`` / ``PartSpec``
     - 一条尚未执行的声明，记录部件类、构造参数和目标节点。
   * - ``PartHandle``
     - 指向部件的引用。无论部件在本地还是 worker 中，调用接口都一样。
   * - ``MethodArm`` / ``MethodGripper`` / ``MethodCamera``
     - 把 ``open_gripper``、``get_camera(id)`` 等硬件方法包装成部件接口的视图。

组合，而非机器人类型
--------------------

机器人不是一组固定的机械臂、相机和底盘槽位，而是一棵由具名部件组成的树。因此，碰到
升降机构或云台这类设备时，不必扩展框架概念；给它起个名字，按普通部件组合即可：

.. code-block:: python

   one = FrankaRobot(arm=arm, gripper=gripper)
   two = FrankaRobot(left=Group(arm=l, gripper=lg), right=Group(arm=r, gripper=rg))
   lifted = FrankaRobot(left=..., right=..., lift=lift, head=head_camera)

观测和动作沿用同一棵树。每个部件名都是一段路径，嵌套多少层都遵循这条规则：

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 路径
     - 含义
   * - ``<name>``
     - 机器人的一个部件，名字就是你组合时给它的名字。
   * - ``<group>.<name>``
     - 组内的部件，组合有多深就能嵌多深。

各机械臂使用独立连接时，``Robot`` 可以并行重置、读取和下发命令。双臂观测只占一个
往返时间，策略侧也看不到这层调度细节。

放置是部件的属性
----------------

用 ``at()`` 记下部件应该运行在哪个节点，不需要再单独调用放置函数。
``Robot.connect`` 会找到这些声明，并在指定节点上构建部件。

.. code-block:: python

   robot = FrankaRobot(
       left=FrankaROSArm.at("10.0.0.1", node_rank=1),
       scene=RealSenseCamera.at(info, node_rank=3),
   )
   robot.connect()

声明刻意不产生副作用，调用 ``connect`` 之前不会访问真实硬件。真正调用 ``connect``
后，RLinf 才会逐一放置不同的声明，并把句柄挂到 ``robot.handles[<name>]``。若后续部件
启动失败，之前放置的内容会全部回收；正常结束时则由 ``disconnect`` 完成同样的清理。

遇到共享硬件时，只声明一次连接，再引用它暴露的部件。一条连接即使支撑两条机械臂和
两个相机，也只会打开一次：

.. code-block:: python

   hardware = Turtle2Hardware.at(50, camera_ids, node_rank=0)
   robot = Turtle2Robot(
       left=Group(
           arm=hardware.part("left"), gripper=hardware.part("left_end_effector")
       ),
       right=Group(
           arm=hardware.part("right"), gripper=hardware.part("right_end_effector")
       ),
       wrist_1=hardware.part("wrist_1"),
   )

所有部件都遵循同一套放置规则，不只有机械臂。相机可以留在实际插接的机器上，策略则在
另一台机器运行。通过 ``declare_cameras({name: info}, node_rank=...)`` 指定节点后，
机器人会在 ``connect`` 时打开相机，在 ``disconnect`` 时关闭。底层的 ``spawn()`` 会
立刻放置部件；只有调试脚本这类没有机器人管理生命周期的场景，才直接调用它。

每种硬件不必再配一个 worker 类。RLinf 会根据部件类自动合成
（``type(name, (Worker, PartCls), ...)``），再由 ``WorkerGroup`` 把公有方法绑定为
RPC。若要调用标准部件接口以外的方法，直接走句柄；本地和远程写法一致::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

worker 与节点、GPU 的映射方式见 :doc:`放置策略 <placement>`。

组合所有部件
------------

可以把机器人看成一棵树：先组合顶层部件，每个部件再带上同一连接暴露的子部件。如果
机械臂和夹爪共用连接，把机械臂装进来就够了，机器人不用再单独声明夹爪：

.. code-block:: python

   robot = FrankaRobot(
       arm=FrankaROSArm.at(robot_ip, node_rank=1),
       wrist=RealSenseCamera.at(info, node_rank=3),
   )

   robot.part("arm").parts     # {"arm": ..., "gripper": ...}

只有一条硬件连接暴露多个同级组件时，才需要从声明中选取子部件。比如，联动控制器能
驱动两条机械臂，但它本身并不是机械臂；这时用 ``part(...)`` 选出要组合的那一个。

按部件类别拆分构建逻辑。``build`` 汇总各个 ``build_*`` 方法的结果，机械臂数量不同的
机器人只需改写 ``build_arms``，其余构建流程继续继承：

.. code-block:: python

   class FrankaRobot(Robot):
       @classmethod
       def build_arms(cls, **config) -> dict[str, RobotPart]:
           return {"arm": cls.declare_arm(...)}

       @classmethod
       def build(cls, **config) -> "FrankaRobot":
           return cls(**cls.build_arms(**config), **cls.build_cameras(...))


   class DualFrankaRobot(FrankaRobot):
       @classmethod
       def build_arms(cls, **config) -> dict[str, RobotPart]:
           return {"left": ..., "right": ...}      # 唯一的差别

这里无需为每种部件单独定义配置类。相机通过
``declare_cameras({name: info}, node_rank=...)`` 声明，机械臂使用 ``at(...)`` 加构造
参数；所需字段统一放在机器人已有的 ``RobotConfig`` 中，硬件发现本来就要读取这份
YAML 结构。

生命周期
--------

生命周期分成四个明确阶段。把“声明”和“连接”拆开后，组装机器人不会意外访问硬件，
启动失败时也有清晰的回滚点。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 步骤
     - 发生了什么
   * - 声明
     - ``at()`` 只记录部件类、构造参数和所属节点，不构建对象，也不访问硬件。
   * - 连接
     - ``Robot.connect`` 把每条不同的声明构建到它的节点上，连接所有部件，并把句柄
       发布为 ``robot.handles[<name>]``。
   * - 使用
     - ``get_observation`` 和 ``send_action`` 会在彼此独立的连接上并行调用。
   * - 断开
     - ``Robot.disconnect`` 先断开各个部件，再释放底层连接。

构建机器人时只组装声明，不会连接硬件。``Robot.build`` 返回后，要先调用 ``connect``，
再读取观测或下发命令。在此之前，``is_connected`` 仍为 ``False``，机器人各个槽位中
保存的也是声明，而不是已经连接的部件。

某个部件连接失败时，RLinf 会回收此前已经放置或连接的内容，并把各个槽位恢复为声明。
排除故障后，可以直接再次调用 ``connect``。正常执行 ``disconnect`` 也会回到这个状态，
因此同一个机器人对象可以安全地连接、断开，再重新连接。


边界
----

部件实现中不要导入 Ray、Gymnasium 或 ``rlinf.scheduler``。加载设备驱动不应顺带把
调度器拉进进程，这样 ``toolkits/realworld_check`` 里的调试脚本才能脱离集群，直接在
硬件机器上运行。

只有 ``rlinf/robotics/placement.py`` 跨过这条边界，而且 ``spawn`` 只在真正需要放置时
才惰性导入它。调度器不会反向导入 robotics。
``tests/unit_tests/test_robotics_boundaries.py`` 同时检查这两个方向。

代码位置
--------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 路径
     - 内容
   * - ``parts/base.py``
     - 核心部件类型：``RobotPart``、``ControllablePart``、``Camera``、
       ``EndEffector``、``Group``、``MobileBase``、``LeggedBase``。
   * - ``parts/arms/``
     - 各硬件系列的机械臂实现和 state 数据类。
   * - ``parts/cameras/``
     - RealSense、ZED、Lumos。
   * - ``parts/end_effectors/``
     - ``grippers/`` 与 ``hands/``。
   * - ``parts/teleop/``
     - 主臂和输入设备，包括 GELLO、数据手套、键盘、Pico、SpaceMouse。
   * - ``parts/transports/``
     - ROS 等共享传输层。传输层只为部件传递消息，本身不算部件。
   * - ``robots/``
     - 每台机器人对应一个模块，配置、发现逻辑和构建函数放在一起。
   * - ``specs.py``
     - ``PartSpec`` 与 ``SubpartRef``：前者表示部件声明，后者引用其暴露的子部件。
   * - ``placement.py``
     - ``PartHandle`` 与自动合成的 worker。整个 robotics 包只有这里导入调度器。
   * - ``views.py``
     - ``Method*`` 系列视图。
   * - ``robot.py``、``discovery.py``、``adapters.py``、``config.py``
     - 机器人组合、注册、旧策略适配器和环境变量配置。

任务不进入硬件代码
------------------

部件只关心怎么运动、能感知什么，不应该判断任务是否成功。把重置行为、奖励、终止条件
和 Gymnasium 空间写进 ``RobotTask``，再通过 ``RobotTaskEnv`` 与 ``Robot`` 组合。若
现有策略使用扁平向量和 ``state``/``frames`` 观测，就在边界上接入
``LegacyObservationAdapter`` 和 ``VectorActionAdapter``。这样，同一套硬件代码既不
依赖策略数据结构，也能复用到不同任务中。

下一步
------

- :doc:`添加机器人 <../extending/new_robot>`：按步骤接入新机器人。
- :doc:`放置策略 <placement>`：了解 worker 如何映射到节点和 GPU。
