机器人模型
==========

先把机器人看成一棵由具名部件组成的树。``Robot.connect`` 根据声明把各部件放到硬件
所在的节点上，调用方拿到的则是一棵完整的观测与动作树。下面先从 Franka 入手，把它
从声明、连接一路跑起来；后半页再回头梳理各个抽象和生命周期。

组装一台机器人
--------------

我们先从头组装一台 Franka。机器人类只说明它带哪些机械臂和相机，等整台机器人组好
之后，调用方再统一连接和放置这些部件。

.. code-block:: python

   class FrankaRobot(Robot):
       ROBOT_TYPE = "Franka"       # build_robot() 按这个名字查找
       BACKEND = "franka_ros"      # declare_arm 据此选择机械臂实现

       @classmethod
       def build_arms(cls, *, robot_ip, node_rank, **config) -> dict[str, RobotPart]:
           # declare_arm 返回一个部件：位于 robot_ip、待在 node_rank 上构建的机械臂，
           # 连同这条连接所暴露的末端执行器。这里只是把描述记下来，不碰硬件。
           return {"arm": cls.declare_arm(robot_ip, node_rank=node_rank, name="arm")}

       @classmethod
       def build_cameras(cls, cameras=None, *, node_rank=None) -> dict[str, RobotPart]:
           # 换一类部件，约定完全一样。
           return declare_cameras(cameras, node_rank=node_rank)


   FrankaRobot.register(FrankaConfig, FrankaDiscovery)

每个 ``build_*`` 方法处理一类硬件，返回 ``{名字: 部件}`` 映射。``build`` 只合并这些
映射；Franka 变体若只更换机械臂数量，便只需改对应的方法：

.. code-block:: python

   @classmethod
   def build(cls, *, cameras=None, camera_node_rank=None, **config) -> "FrankaRobot":
       return cls(
           **cls.build_arms(**config),
           **cls.build_cameras(cameras, node_rank=camera_node_rank),
       )

映射中的键会成为部件名。组装完成后，``connect`` 在各自指定的节点上构建部件，
``disconnect`` 再释放相同的资源。

怎么用一台机器人
----------------

声明写完后，就可以按注册名构建 Franka。调用方不必了解部件如何拼装，只需连接一次，
然后读取状态、发送动作，最后断开连接：

.. code-block:: python

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)
   robot.connect()

   observation = robot.get_observation()
   observation["arm"]["arm"]["tcp_pose"]            # 末端笛卡尔位姿
   observation["arm"]["end_effector"]["state"]      # 夹爪开合

   robot.send_action(
       {"arm": {"arm": {"tcp_pose": target}, "end_effector": {"target": width}}}
   )

   robot.reset()
   robot.disconnect()

观测和动作都是嵌套字典，键来自前面的部件名。换成双臂机器人后，调用方式不变，只是
树上多了 ``left`` 和 ``right`` 两个分支：

.. code-block:: python

   observation["left"]["arm"]["arm_joint_position"]
   observation["right"]["end_effector"]["state"]

这段调用代码不关心部件运行在哪个节点。远端机械臂与本地机械臂的读数出现在同一路径；
两条机械臂若使用独立连接，``Robot.get_observation`` 会同时发起读取，因此调用方只等
一次往返。

还没接上硬件时，可以先读 ``observation_features`` 和 ``action_features``，看看这棵
树最终会是什么形状。``RobotTaskEnv`` 也根据这两份描述创建 Gymnasium 空间。策略若
只接收扁平向量，后面的 `任务不进入硬件代码`_ 会说明适配器放在哪里。

单臂和多臂是同一套代码
----------------------

现在把单臂 Franka 换成双臂。``DualFrankaRobot`` 继承原来的类，只替换
``build_arms`` 返回的映射：

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

``declare_arm``、``build_cameras`` 和 ``build`` 仍用父类版本，放置与资源回收流程也
没有变化。以后增加第三条机械臂时，在这个映射里再加一个条目即可。

切换控制后端只需一行
--------------------

Franka 有两套控制后端，机器人类通过 ``BACKEND`` 选择其中一套。``declare_arm`` 每次
声明机械臂都会读取这个属性，不论当前型号有一条还是多条机械臂：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``BACKEND``
     - 它声明的机械臂部件
   * - ``"franka_ros"``
     - ``FrankaROSArm``：通过 ROS 做笛卡尔阻抗控制。
   * - ``"franky"``
     - ``FrankyArm``：通过 libfranka 做关节和笛卡尔控制。

接入第三种后端时，新增一个机械臂部件，再把它登记到 ``FRANKA_BACKENDS``。机器人类
继续调用 ``declare_arm``，不直接写具体的机械臂类名。

机械臂整体到位
--------------

接下来看看夹爪。Franka 夹爪与机械臂共用一条连接，因此它属于机械臂；把机械臂加入
机器人时，夹爪已经作为 ``end_effector`` 子部件包含在内：

.. code-block:: python

   robot = FrankaRobot.build(robot_ip="10.0.0.1", node_rank=1, ...)

   robot.parts                  # {"arm": ...}
   robot.part("arm").parts      # {"arm": ..., "end_effector": ...}

``build`` 没有提到夹爪。机械臂列出这条连接能够访问的组件，机器人只选择顶层硬件。

任何部件都能放在任何节点
------------------------

完整的放置流程稍后再讲。这里先看结果：``node_rank`` 写在部件声明上，所以机械臂和
相机可以分别运行在不同节点。

.. code-block:: python

   robot = FrankaRobot(
       arm=FrankaROSArm.at(robot_ip, node_rank=1),      # 放在机械臂那台 NUC 上
       wrist=RealSenseCamera.at(info, node_rank=3),     # 放在它插接的机器上
   )

``Robot`` 会并行读取相互独立的连接；共用同一连接的部件仍按声明顺序访问，因为厂商
SDK 往往不允许在一条链路上并发调用。Franka 类本身不参与这层调度。

核心思想
--------

前面的例子都遵循同一条规则：**把每个物理组件都建模为部件（part）。** 机械臂、夹爪、
相机和移动底盘都是 ``RobotPart``。部件知道怎么连接硬件、读取观测；可控部件还接收
动作。设备侧的这些行为直接写在部件上，不再另设一层“驱动”抽象。

一条硬件连接不一定只对应一个物理组件。不妨设想一台联动双臂控制器：它通过一条 ROS
连接操纵两条机械臂、两个夹爪和两个腕部相机。此时，由连接对应的部件通过 ``parts``
列出所有组件：

.. code-block:: python

   @property
   def parts(self) -> dict[str, RobotPart]:
       return {
           "left": MethodArm(self, commands={"tcp_pose": "move_left_arm"}),
           "right": MethodArm(self, commands={"tcp_pose": "move_right_arm"}),
           "left_end_effector": MethodGripper(self, state_field="follow1_pos"),
       }

是否持有连接，只是部件内部的实现细节。

另外两条规则是：**用具名部件组合机器人。** **把任何部件放到所需节点。** 前面示例中
的嵌套字典和 ``node_rank`` 声明，都由这两条规则推导而来。

抽象一览
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 抽象
     - 含义
   * - ``RobotPart``
     - 物理部件的基类，定义 ``connect``、``get_observation``、``disconnect`` 和
       ``reset``。``observation_features`` 描述返回的数据。
   * - ``ControllablePart``
     - 带有 ``send_action`` 的部件；``action_features`` 描述它接受的命令格式。
   * - ``Camera`` / ``EndEffector`` / ``MobileBase`` / ``LeggedBase``
     - 更具体的部件类型。组合结构或远程代理需要保留设备类别时使用。
   * - ``parts``
     - 当前部件暴露的具名组件。硬件用它列出一条连接驱动的所有组件，``Group`` 返回
       组内成员，叶子部件返回 ``{}``。
   * - ``Group``
     - 由其他具名部件组成的部件，可以表示机械臂、躯干或整台机器人。
   * - ``Robot``
     - 最外层的 ``Group``，带有注册类型，按硬件配置组装自身，并管理放置时创建的连接。
   * - ``at()`` / ``PartSpec``
     - 尚未执行的声明，记录部件类、构造参数和目标节点。
   * - ``PartHandle``
     - 指向部件的引用；本地部件和 worker 中的部件使用相同调用接口。
   * - ``MethodArm`` / ``MethodGripper`` / ``MethodCamera``
     - 把 ``open_gripper``、``get_camera(id)`` 等硬件方法呈现为部件接口的视图。

组合，而非机器人类型
--------------------

理解了这些抽象后，我们不妨先忘掉“机器人必须有哪些槽位”。机器人只是一棵具名部件
树，升降机构或云台也和机械臂一样，起一个名字后直接放进组合：

.. code-block:: python

   one = FrankaRobot(arm=arm, gripper=gripper)
   two = FrankaRobot(left=Group(arm=l, gripper=lg), right=Group(arm=r, gripper=rg))
   lifted = FrankaRobot(left=..., right=..., lift=lift, head=head_camera)

观测和动作沿用同一棵树。每个部件名增加一段路径，``Group`` 内部继续按相同规则嵌套：

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 路径
     - 含义
   * - ``<name>``
     - 机器人的一个部件，名字就是你组合时给它的名字。
   * - ``<group>.<name>``
     - 组内的部件，组合有多深就能嵌多深。

各机械臂使用独立连接时，``Robot`` 会并行重置、读取和下发命令。策略仍只读取上面的
嵌套字典，不需要协调两个请求。

放置是部件的属性
----------------

组合结构说明机器人有什么，放置声明则告诉 ``Robot.connect`` 去哪里创建这些部件。
调用 ``at()`` 时写入目标节点，再连接组装好的机器人：

.. code-block:: python

   robot = FrankaRobot(
       left=FrankaROSArm.at("10.0.0.1", node_rank=1),
       scene=RealSenseCamera.at(info, node_rank=3),
   )
   robot.connect()

上面的 ``at()`` 只记录声明，不会创建 worker，也不会打开硬件。调用
``Robot.connect`` 后，每条不同的声明只放置一次，所得句柄挂在
``robot.handles[<name>]`` 下。后续部件若启动失败，已经放置的部件会依次回收；正常
退出时，``disconnect`` 走同一套清理流程。

共享硬件还要多考虑一步：连接只声明一次，组合时再引用它通过 ``parts`` 暴露的组件。
下面这一条声明同时支撑两条机械臂和相机：

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

相机和机械臂走同一套放置流程。比如，相机留在实际插接的机器上，策略运行在别处；此时
把节点传给 ``declare_cameras({name: info}, node_rank=...)``，``Robot.connect`` 会打开
相机，``Robot.disconnect`` 会关闭它。更底层的 ``spawn()`` 会立即放置部件，只适合
调试脚本等没有 ``Robot`` 管理生命周期的场景。

放置部件时不必另写 worker 类。放置代码根据部件类合成
``type(name, (Worker, PartCls), ...)``，``WorkerGroup`` 再把公有方法绑定为 RPC。厂商
特有的方法即使不在标准部件接口中，也可以通过句柄调用；本地与远程使用同一种写法::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

接下来若要配置 worker 与节点、GPU 的映射，请读 :doc:`放置策略 <placement>`。

组合所有部件
------------

部件放在哪个节点，并不会改变组合树。我们先放入顶层部件，再由这些部件列出同一连接上
的组件。机械臂和夹爪共用连接时，加入机械臂后，``end_effector`` 已经在树中：

.. code-block:: python

   robot = FrankaRobot(
       arm=FrankaROSArm.at(robot_ip, node_rank=1),
       wrist=RealSenseCamera.at(info, node_rank=3),
   )

   robot.part("arm").parts     # {"arm": ..., "end_effector": ...}

一条硬件声明若暴露多个同级组件，再用 ``part(...)`` 逐个选取。比如，联动控制器可以
驱动两条机械臂，但它本身不是机械臂；机器人应从声明中选出左右臂。

机器人类的构建逻辑按部件类别拆开。``build`` 汇总各个 ``build_*`` 映射，机械臂数量
不同的变体只改写 ``build_arms``：

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

不必给每种部件单独定义配置类。相机通过
``declare_cameras({name: info}, node_rank=...)`` 声明，机械臂使用 ``at(...)`` 和构造
参数；两者都从机器人现有的 ``RobotConfig`` 读取字段，硬件发现也使用这份配置。

生命周期
--------

讲到这里，需要把“组装声明”和“连接硬件”分清。机器人的生命周期分为四个阶段，第一个
阶段不会打开任何硬件：

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

``Robot.build`` 返回时，机器人各个槽位中仍是声明，不是已经连接的部件，
``is_connected`` 也仍为 ``False``。读取观测或下发命令之前，必须先调用 ``connect``。

某个部件连接失败时，``Robot.connect`` 会回收此前已经放置或连接的部件，并把各个槽位
恢复为声明。排除硬件故障后，可以再次调用 ``connect``。正常执行 ``disconnect`` 也会
回到同一状态，之后还可以重新连接这个机器人对象。


边界
----

实现部件时还要守住一条导入边界：部件模块不能导入 Ray、Gymnasium 或
``rlinf.scheduler``。否则，仅仅加载设备驱动就会连带加载集群依赖。
``toolkits/realworld_check`` 中的调试脚本依靠这条边界，才能脱离集群直接运行在硬件机器
上。

只有 ``rlinf/robotics/placement.py`` 可以跨过这条边界，``spawn`` 也只在确实需要放置
时才惰性导入它。另一个方向同样受限：调度器不会导入 robotics。
``tests/unit_tests/test_robotics_boundaries.py`` 会检查这两条规则。

代码位置
--------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 路径
     - 内容
   * - ``parts/base.py``
     - 核心部件类型，包括 ``RobotPart``、``ControllablePart``、``Camera``、
       ``EndEffector``、``Group``、``MobileBase``、``LeggedBase``。
   * - ``parts/arms/``
     - 各硬件系列的机械臂实现，以及对应的 state 数据类。
   * - ``parts/cameras/``
     - RealSense、ZED、Lumos。
   * - ``parts/end_effectors/``
     - ``grippers/`` 与 ``hands/``。
   * - ``parts/transports/``
     - ROS 等共享传输层。它替部件传递消息，但本身不是部件。
   * - ``robots/``
     - 每台机器人一个模块，其中包含配置、发现逻辑和构建函数。
   * - ``specs.py``
     - ``PartSpec`` 表示部件声明，``SubpartRef`` 引用声明暴露的子部件。
   * - ``placement.py``
     - ``PartHandle`` 与自动合成的 worker；整个 robotics 包只有这里导入调度器。
   * - ``views.py``
     - ``Method*`` 系列视图。
   * - ``robot.py``、``discovery.py``、``adapters.py``、``config.py``
     - 组合、注册、旧策略适配器和环境变量配置。

任务不进入硬件代码
------------------

最后一条边界位于硬件与任务之间。部件只描述怎么运动、能感知什么，不判断一轮任务是否
成功。把重置行为、奖励、终止条件和 Gymnasium 空间写进 ``RobotTask``，再通过
``RobotTaskEnv`` 与 ``Robot`` 组合。现有策略若使用扁平向量和 ``state``/``frames``
观测，就在这里加入 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter``。换任务时，
机器人设备代码无需随之修改。

下一步
------

- :doc:`添加机器人 <../extending/new_robot>`：按步骤接入新机器人。
- :doc:`放置策略 <placement>`：了解 worker 如何映射到节点和 GPU。
