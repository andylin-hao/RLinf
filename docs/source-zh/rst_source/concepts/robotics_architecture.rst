机器人架构
============

:doc:`robotics` 把机器人介绍为一棵具名部件树。当你开始写设备驱动、让多个部件共用
一条连接，或者排查远程放置与资源回收问题时，再来读这一页。

从部件树走到硬件连接
----------------------

先看一个常见情况：机械臂和夹爪共用一个 ROS 会话。策略并不关心这条会话，它仍然
看到两个独立的部件：

.. code-block:: text

   robot
   ├── arm
   └── end_effector

这里其实有两个不同的问题：

- 机器人树回答“策略能观测什么、能控制什么”。
- 硬件连接回答“哪些资源只能打开一次、放在哪里、最后由谁释放”。

RLinf 用 ``children`` 和 ``exports`` 分别回答这两个问题。理清这对概念后，后面的
放置和生命周期就容易理解了。

先认识几个类型
------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 类型
     - 用途
   * - ``Endpoint``
     - RLinf 在某台机器上打开、之后再关闭的对象。它有运行位置和生命周期，
       但不一定能读取观测。
   * - ``RobotPart``
     - 可以读取的 ``Endpoint``。它实现 ``get_observation()``，并用
       ``observation_features`` 描述返回的数据。
   * - ``ControllablePart``
     - 可以接收动作的 ``RobotPart``。它还实现 ``send_action()`` 和
       ``action_features``。
   * - ``Connection``
     - 一次硬件会话，可以支撑多个部件。它自己不是部件，因此不会出现在观测树中。
   * - ``Group``
     - 由具名 ``children`` 组成的部件。它可以表示一套机械臂、一个躯干，也可以表示整台机器人。
   * - ``Robot``
     - 最外层的 ``Group``。除了组合部件，它还管理注册、放置和 ``connect()`` 期间创建的句柄。
   * - ``PartSpec``
     - 调用部件类的 ``at()`` 后得到的延迟声明，记录部件类、构造参数、节点和可选的 worker 名称。
   * - ``PartHandle``
     - 指向已连接部件的统一句柄。本地部件和 worker 中的远端部件使用同一套调用方式。

``Camera``、``EndEffector``、``MobileBase`` 和 ``LeggedBase`` 是更具体的部件类型。当组合
或远程代理需要保留设备类别时，使用这些类型。

别把 ``exports`` 和 ``children`` 混在一起
---------------------------------------------------

``exports`` 说明一次硬件会话能够对外暴露哪些能力。例如，一条机械臂连接可以同时
暴露机械臂本身和夹爪视图：

.. code-block:: python

   class ExampleArm(ControllablePart):
       @property
       def exports(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(
                   self, state_field="gripper_position"
               ),
           }

不过，``exports`` 中的名字只属于这条连接。机器人最终对外公布什么名字，要在组合时
明确写出来：

.. code-block:: python

   connection = ExampleArm.at("10.0.0.2", node_rank=1)
   robot = Robot(
       arm=connection.export("arm"),
       end_effector=connection.export("end_effector"),
   )

此时，``robot.children`` 包含 ``arm`` 和 ``end_effector``。``Connection`` 没有
``children``，``Group`` 也不向外 ``export`` 任何东西。可以这样记：
``exports`` 属于硬件会话，``children`` 属于具名组合树，两者只在组合这一步相遇。

联动控制器也遵循同样的规则。控制器不必假装成其中某条机械臂：

.. code-block:: python

   connection = Turtle2Connection.at(50, camera_ids, node_rank=0)
   robot = Turtle2Robot(
       left=Group(
           arm=connection.export("left"),
           end_effector=connection.export("left_end_effector"),
       ),
       right=Group(
           arm=connection.export("right"),
           end_effector=connection.export("right_end_effector"),
       ),
       wrist=connection.export("wrist_1"),
   )

所有引用都指向同一份声明，因此这条控制器连接只会打开一次。

先声明放置，再连接硬件
----------------------

``at()`` 只记录部件应当在哪个节点构建，不会创建对象，也不会打开硬件：

.. code-block:: python

   arm_connection = ExampleArm.at("10.0.0.2", node_rank=1)
   robot = Robot(
       arm=arm_connection.export("arm"),
       end_effector=arm_connection.export("end_effector"),
       scene=RealSenseCamera.at(camera_info, node_rank=3),
   )

   print(robot.describe())
   robot.connect()
   try:
       observation = robot.get_observation()
   finally:
       robot.disconnect()

``connect()`` 会逐个解析 ``PartSpec``，同一份声明只解析一次。本地声明得到
``LocalPartHandle``，远端声明由调度器放进 worker，得到 ``RemotePartHandle``。两种
句柄的公有方法一致，调用方无需根据放置方式分支。

放置层也记录资源归属。即使三条部件路径都引用同一份 ``PartSpec``，机器人也只会创建
一个句柄，最后也只释放一次。不同句柄上的部件可以并行访问；共用句柄的部件按声明顺序
访问，避免对不支持并发的厂商会话发起多个请求。

只有在没有 ``Robot`` 管理资源时，例如写独立工装脚本，才直接使用 ``spawn()``。它会立即
放置部件，句柄的回收也交给调用方。在机器人内部优先使用 ``at()``，让回滚和清理保持自动。

连硬件之前先检查声明
--------------------

``Robot.describe()`` 读取机器人保存的声明快照，而不是从已连接的代理上重新推断。
因此，连接前后看到的节点和资源归属保持一致：

.. code-block:: text

   FrankaRobot
   ├── arm           declared      node=1     via FrankaROSArm#1
   └── end_effector  declared      node=1     via FrankaROSArm#1

``via`` 相同的行共用一个 ``Endpoint``。直接声明的部件在连接前就能报告自己的类别。
如果某一行引用的是未打开 ``Connection`` 中的能力，结果会显示为 ``declared``。连接打开前
还不知道它会创建哪个具体部件，``describe()`` 不会拿连接自身的类别来充数。

目前，``describe()`` 只展示组合结构、放置节点和资源归属，还不会列出观测与动作的数据描述。
尤其是尚未打开的 ``Connection``，此时还无法说明它最终会暴露哪些具体部件。需要检查这些
数据描述时，请使用 :doc:`添加机器人 <../extending/new_robot>` 中的一致性检查，通过 fake SDK
或真机打开连接后再验证。

生命周期怎么走
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 阶段
     - 发生了什么
   * - 声明
     - ``at()`` 记录构造参数和放置节点。这时不导入厂商 SDK，也不打开硬件。
   * - 连接
     - 机器人放置每份声明，打开 ``Endpoint``，解析它暴露的部件，并将句柄记入
       ``robot.handles``。
   * - 使用
     - 机器人按具名树读取、复位和发送动作，再根据资源是否共享决定能否并行。
   * - 断开
     - 部件释放实际打开的设备，关闭句柄，机器人再恢复最初的声明树。

``_open()`` 返回厂商对象，``_release(device)`` 收到的就是同一个对象。把设备显式传入释放钩子，
可以避免清空 ``_device`` 的时机影响资源回收。

连接是一个整体。后面的部件启动失败时，``Robot.connect()`` 会回收已经放置或打开的资源，
然后恢复声明树。排除硬件问题后，可以对同一个机器人对象再次调用 ``connect()``。
``disconnect()`` 重复调用也是安全的。

通过句柄调用设备专有方法
------------------------

放置层会根据部件类创建 worker 接口，因此每接一种设备都不必再写一个 worker 类。标准部件接口
之外的公有方法，仍然可以通过句柄调用：

.. code-block:: python

   robot.handles["arm"].is_robot_up().wait()[0]
   robot.handles["arm"].reset_joint(home_qpos).wait()

本地和远端部件的写法相同。任务逻辑应当继续使用观测与动作树；只有初始化、诊断，或者确实没有
通用部件方法可表达的厂商操作，才需要使用句柄。

守住导入边界
--------------

部件模块不导入 Ray、Gymnasium 或 ``rlinf.scheduler``。这样，硬件机器可以不启动集群，
直接导入和测试设备驱动。``rlinf/robotics/placement/handles.py`` 是唯一的桥接层，只有在
需要远程放置时才惰性导入。

反方向也一样：调度器不导入 robotics。``tests/unit_tests/test_robotics.py`` 会检查这两条边界。

去哪里找实现
----------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 路径
     - 内容
   * - ``robotics/parts/base.py``
     - ``Endpoint``、各类部件、``Group`` 和 ``Connection``。
   * - ``robotics/parts/arms/``
     - 机械臂与联动控制器实现。
   * - ``robotics/parts/cameras/``
     - 相机生命周期，以及 RealSense、ZED 和 Lumos 实现。
   * - ``robotics/parts/end_effectors/``
     - 夹爪和灵巧手。
   * - ``robotics/parts/views.py``
     - 将共享厂商会话呈现为部件的 ``MethodArm``、``MethodGripper`` 和 ``MethodCamera``。
   * - ``robotics/placement/``
     - 延迟声明、本地与远端句柄、worker 放置。
   * - ``robotics/robot.py``
     - 最外层组合、声明快照、``describe()`` 和生命周期。
   * - ``robotics/discovery/``
     - 机器人注册、发现与配置查找。

下一步
--------

- :doc:`添加机器人 <../extending/new_robot>` 按实际接入顺序使用这些概念。
- :doc:`放置策略 <placement>` 说明调度器资源如何映射到节点和 GPU。
- :doc:`遥操作 <../guides/teleoperation>` 说明环境侧如何组合操作者设备与 binding。
