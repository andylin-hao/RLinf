机器人模型
==========

先掌握 RLinf 的机器人模型，再接入硬件或排查真机运行问题。你会明确三件事：策略如何
理解组件、如何把组件组合成机器人，以及每个组件在哪里运行。

核心思想
--------

**把每个物理组件都建模为部件（part）。** 机械臂、夹爪、相机和移动底盘都是
``RobotPart``。每个部件都能连接并报告观测。可控部件还能接收动作。无需再加一层独立
的“驱动”抽象。

硬件与组件通常不是一一对应。例如，一台联动双臂控制器可能通过一条 ROS 连接驱动两条
机械臂、两个夹爪和两个腕部相机。遇到这种情况，直接让部件声明它暴露的组件。不要为
“持有连接的对象”再定义一层抽象：

.. code-block:: python

   def subparts(self) -> dict[str, RobotPart]:
       return {
           "left": MethodArm(self, commands={"tcp_pose": "move_left_arm"}),
           "right": MethodArm(self, commands={"tcp_pose": "move_right_arm"}),
           "left_end_effector": MethodGripper(self, state_field="follow1_pos"),
       }

把“持有连接”视为部件的属性，不要定义成另一种类型。

**用具名部件组合机器人。** **把任何部件放到所需节点。** 加上第一条规则，这三条规则
共同构成机器人模型。

抽象一览
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 抽象
     - 含义
   * - ``RobotPart``
     - 任意物理组件。它包含 ``connect``、``get_observation``、``disconnect`` 和
       ``reset``。``observation_features`` 描述返回的数据。
   * - ``ControllablePart``
     - 还能通过 ``send_action`` 接收命令，并用 ``action_features`` 描述命令的部件。
   * - ``Camera`` / ``EndEffector`` / ``MobileBase`` / ``LeggedBase``
     - 更具体的部件类型。组合层和远程代理可以据此区分它们。
   * - ``subparts()``
     - 部件暴露的具名组件。叶子部件返回 ``{}``。
   * - ``Arm``
     - 组合机械臂本体、可选末端执行器和腕部相机的部件。
   * - ``Robot``
     - 由具名机械臂、机器人级相机、附加部件及其持有的句柄组成。
   * - ``at()`` / ``PartSpec``
     - 一条声明：部件类、构造参数，以及它要运行的节点。
   * - ``PartHandle``
     - 指向部件的引用。部件无论在本地还是 worker 中运行，接口都相同。
   * - ``MethodArm`` / ``MethodGripper`` / ``MethodCamera``
     - 把 ``open_gripper``、``get_camera(id)`` 等方法转换成部件的视图。

组合，而非机器人类型
--------------------

机器人只有一个 ``arms`` 映射，它的大小就是机械臂数量。不需要在单臂和双臂两种变体
之间选择，也没有“最多两条”的限制：

.. code-block:: python

   one = FrankaRobot(arms={"arm": Arm(arm, gripper)})
   two = FrankaRobot(arms={"left": Arm(left, lg), "right": Arm(right, rg)})
   three = FrankaRobot(arms={"left": ..., "right": ..., "third": ...})

用组合结构确定策略看到的数据形状。名称会成为路径：

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 路径
     - 含义
   * - ``arms.<name>.state``
     - 该机械臂本体的观测。
   * - ``arms.<name>.arm``
     - 该机械臂本体的动作。
   * - ``arms.<name>.end_effector``
     - 其末端执行器的观测与动作。
   * - ``cameras.<name>`` / ``parts.<name>``
     - 机器人级相机与其他组件。

各机械臂使用独立连接时，``Robot`` 会并行重置、读取和下发命令。读取双臂观测只需
一个往返时间，而不是两个。

放置是部件的属性
----------------

用 ``at()`` 声明部件运行在哪个节点。没有人需要调用放置函数：``Robot.connect``
会把每条声明构建到它所属的节点上。

.. code-block:: python

   robot = FrankaRobot(
       arms={"left": Arm(FrankaROSArm.at("10.0.0.1", node_rank=1))},
       cameras={"scene": RealSenseCamera.at(info, node_rank=3)},
   )
   robot.connect()

声明本身不做任何事。在 ``connect`` 之前不会碰硬件。``connect`` 会把每条不同的声明
放置一次，把句柄发布为 ``robot.handles[<name>]``，并在后续部件失败时拆掉已经放置的
部分。``disconnect`` 负责释放它们。

共享连接只声明一次，再引用它的 subparts。一条连接支撑两条机械臂和两个相机时，
只会被打开一次，而不是四次：

.. code-block:: python

   hardware = Turtle2Hardware.at(50, camera_ids, node_rank=0)
   robot = Turtle2Robot(
       arms={
           side: Arm(
               hardware.subpart(side), hardware.subpart(f"{side}_end_effector")
           )
           for side in ("left", "right")
       },
       cameras={"wrist_1": hardware.subpart("wrist_1")},
   )

放置适用于所有部件，不只是机械臂。机器人持有自己的相机，并像放置其他部件一样放置
它们，因此相机可以运行在它所插接的机器上，而策略运行在别处。用
``CameraConfig(info=info, node_rank=...)`` 指定节点，机器人会在 ``connect`` 时打开
它、在 ``disconnect`` 时关闭它。``spawn()`` 是底层的即时形式，只在机器人之外使用，例如调试脚本。

不要为每种硬件编写 worker 类。RLinf 会根据部件类自动合成一个
（``type(name, (Worker, PartCls), ...)``）。``WorkerGroup`` 随后把每个公有方法绑定
为 RPC。通过句柄调用部件接口之外的方法。本地和远程的调用形式相同::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

要了解 worker 如何映射到节点和 GPU，请阅读 :doc:`放置策略 <placement>`。

组合所有部件，不只是机械臂
--------------------------

机械臂、末端执行器和相机是彼此独立的部件。机器人的 ``build`` 负责把它们组合起来，
每个部件都带有自己的 ``node_rank``：

.. code-block:: python

   arm = FrankaROSArm.at(robot_ip, node_rank=1)
   robot = FrankaRobot(
       arms={
           "arm": Arm(
               arm,
               RobotiqGripper.at(port="/dev/ttyUSB0", node_rank=2),
               cameras={"wrist": RealSenseCamera.at(info, node_rank=3)},
           )
       }
   )

Robotiq 夹爪本身就是一台串口设备，相机也独占自己的 USB 链路，因此它们都不必和机械臂
待在同一台机器上。只有当末端执行器确实依附于机械臂的连接时（例如 Franka 手），才用
``arm.subpart("end_effector")`` 取它。

让配置自己声明部件。继承 ``PartConfig``，写明要构建哪个类、用什么参数，
``declare_all`` 就能把一组配置变成声明，与部件种类无关：

.. code-block:: python

   @dataclass
   class CameraConfig(PartConfig):
       info: CameraInfo = None

       def part_cls(self):
           return camera_cls(self.info.camera_type)

       def part_args(self):
           return (self.info,)


   cameras = declare_all(configs, default_node_rank=node_rank)

节点默认值、命名和声明都由基类继承，因此机器人的 ``build`` 只剩下组合。

生命周期
--------

四个步骤，顺序固定。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 步骤
     - 发生了什么
   * - 声明
     - ``at()`` 记录部件类、构造参数和所属节点。不构建任何东西，也不碰硬件。
   * - 连接
     - ``Robot.connect`` 把每条不同的声明构建到它的节点上，连接所有部件，并把句柄
       发布为 ``robot.handles[<name>]``。
   * - 使用
     - ``get_observation`` 和 ``send_action`` 在彼此独立的连接上并行展开。
   * - 断开
     - ``Robot.disconnect`` 先断开部件，再释放它们背后的连接。

构建机器人不会连接它。``Robot.build`` 只是组合声明并返回；在读取或下发之前，必须先
调用 ``connect``。在此之前 ``is_connected`` 为 ``False``，各个槽位里仍然是声明。

如果某个部件在连接时失败，已经放置和连接的部分会被拆掉，各个槽位也会恢复成声明，
因此你可以排除原因后重新调用 ``connect``。``disconnect`` 同样会恢复声明，所以机器人
可以连接、断开、再连接。

只有当机械臂本身不会打开末端执行器时，才能把末端执行器声明为独立部件。Franka 系列的
机械臂会在 ``connect`` 时用自己的连接构建夹爪，此时再声明一个 Robotiq 夹爪就会重复
打开同一个串口；``compose_arms`` 会直接报错，而不是留到真机上才失败。


边界
----

不要在部件中导入 Ray、Gymnasium 或 ``rlinf.scheduler``。导入部件时，不能把调度器
加载到当前进程。这样，``toolkits/realworld_check`` 中的调试脚本即使没有集群也能
运行。

只有 ``rlinf/robotics/placement.py`` 跨过这条边界。``spawn`` 会惰性导入它。调度器
不会导入 robotics。``tests/unit_tests/test_robotics_boundaries.py`` 会检查两个方向。

代码位置
--------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 路径
     - 内容
   * - ``parts/base.py``
     - 部件类型体系：``RobotPart``、``ControllablePart``、``Camera``、
       ``EndEffector``、``Arm``、``MobileBase``、``LeggedBase``。
   * - ``parts/arms/``
     - 机械臂硬件，以及每个系列的 state 数据类。
   * - ``parts/cameras/``
     - RealSense、ZED、Lumos。
   * - ``parts/end_effectors/``
     - ``grippers/`` 与 ``hands/``。
   * - ``parts/teleop/``
     - 主从臂与输入设备：GELLO、数据手套、键盘、Pico、SpaceMouse。
   * - ``parts/transports/``
     - ROS 等共享传输层。它们不是部件，只为部件传递消息。
   * - ``robots/``
     - 每台机器人对应一个模块，其中包含配置、发现逻辑和构建函数。
   * - ``specs.py``
     - ``PartSpec``、``SubpartRef`` 与 ``PartConfig``：部件声明、指向其 subpart
       的引用，以及能声明自身部件的配置。
   * - ``placement.py``
     - ``PartHandle`` 与自动合成的 worker。只有这个模块导入调度器。
   * - ``views.py``
     - ``Method*`` 系列视图。
   * - ``robot.py``、``discovery.py``、``adapters.py``、``config.py``
     - 组合、注册、旧策略适配器和环境变量配置。

任务不进入硬件代码
------------------

把任务逻辑留在硬件代码之外。部件只知道如何运动和能感知什么，不判断是否成功。把
重置行为、奖励、终止条件和 Gymnasium 空间写入 ``RobotTask``。再通过
``RobotTaskEnv`` 与 ``Robot`` 组合。用 ``LegacyObservationAdapter`` 和
``VectorActionAdapter`` 把组合接口转换成现有策略所需的扁平向量与
``state``/``frames`` 观测。硬件代码无需了解策略的数据结构。

下一步
------

- :doc:`添加机器人 <../extending/new_robot>`：按步骤接入新机器人。
- :doc:`放置策略 <placement>`：了解 worker 如何映射到节点和 GPU。
