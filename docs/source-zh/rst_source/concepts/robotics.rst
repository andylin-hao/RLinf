机器人模型
==========

在接入硬件或排查真机运行问题之前，先理解 RLinf 如何为一台物理机器人建模。这一层回答
三个问题：一个组件对策略而言 **是什么**、组件如何组合成机器人、以及每个组件在哪里
运行。

核心思想
--------

**任何物理组件都是部件（part）。** 机械臂、夹爪、相机、移动底盘都是 ``RobotPart``：
可以连接、报告观测，若可控还能接收动作。底下没有另一套独立的“驱动”概念。

这一点很重要，因为硬件很少与组件一一对应。一台联动的双臂控制器可能通过单条 ROS
连接驱动两条机械臂、两个夹爪和两个腕部相机。与其为“持有连接的那个东西”再发明一层
抽象，不如让这样的部件直接声明它暴露了什么：

.. code-block:: python

   def subparts(self) -> dict[str, RobotPart]:
       return {
           "left": MethodArm(self, commands={"tcp_pose": "move_left_arm"}),
           "right": MethodArm(self, commands={"tcp_pose": "move_right_arm"}),
           "left_end_effector": MethodGripper(self, state_field="follow1_pos"),
       }

“持有一条连接”只是部分部件具备的属性，而不是另一类事物。

**机器人是部件的具名组合**，而且 **任何部件都可以被放置到某个节点上**。整个模型就是
这三句话。

抽象一览
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 抽象
     - 含义
   * - ``RobotPart``
     - 任何物理组件：``connect``、``get_observation``、``disconnect``、``reset``，
       以及描述返回内容的 ``observation_features``。
   * - ``ControllablePart``
     - 还能接收命令的部件：``send_action`` 与 ``action_features``。
   * - ``Camera`` / ``EndEffector`` / ``MobileBase`` / ``LeggedBase``
     - 更具体的类别，便于组合层和远程代理区分它们。
   * - ``subparts()``
     - 一个部件所暴露的具名组件。叶子部件返回 ``{}``。
   * - ``Arm``
     - 组合本体、可选末端执行器和腕部相机的部件。
   * - ``Robot``
     - 具名机械臂、机器人级相机、附加部件，以及它所持有的句柄。
   * - ``PartHandle``
     - 指向部件的引用；无论其运行在本地还是 worker 中，接口都一致。
   * - ``MethodArm`` / ``MethodGripper`` / ``MethodCamera``
     - 把方法接口（``open_gripper``、``get_camera(id)``）转换成部件的视图。

组合，而非机器人类型
--------------------

机械臂数量只是映射的大小。单臂与双臂机器人属于同一个类：

.. code-block:: python

   single = FrankaRobot.single_arm(Arm(arm, gripper))
   dual = FrankaRobot.dual_arm(Arm(left, left_gripper), Arm(right, right_gripper))

组合方式同时决定了策略看到的数据结构。名称会成为路径：

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

由于各机械臂位于彼此独立的连接上，``Robot`` 会并行执行重置、读取和下发：双臂观测
只需一个往返时间，而不是两个。

放置是部件的属性
----------------

``RobotPart.spawn`` 是唯一的放置入口。

.. code-block:: python

   local = RealSenseCamera.spawn(camera_info)                    # 本地
   remote = RealSenseCamera.spawn(camera_info, node_rank=2)      # 放到节点 2

两者都返回 API 完全相同的 ``PartHandle``，因此调用方无需区分放置方式。这不限于机械
臂：相机可以运行在它所插接的机器上，而策略运行在别处。

不存在按硬件编写的 worker 类。RLinf 会依据部件类自动合成一个
（``type(name, (Worker, PartCls), ...)``），``WorkerGroup`` 随即把每个公有方法绑定
为 RPC。部件接口之外的方法仍可通过句柄调用，且本地与远程的调用形式一致::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

worker 如何映射到节点和 GPU，参见 :doc:`放置策略 <placement>`。

边界
----

部件不得导入 Ray、Gymnasium 或 ``rlinf.scheduler``。导入一个部件不应把调度器带入
进程——正是这一点让 ``toolkits/realworld_check`` 下的调试脚本可以在完全没有集群的
机器上运行。

只有 ``rlinf/robotics/placement.py`` 一个模块跨越这条边界，且 ``spawn`` 以惰性方式
导入它。反方向上，调度器从不导入 robotics。两个方向都由
``tests/unit_tests/test_robotics_boundaries.py`` 强制检查。

代码位置
--------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 路径
     - 内容
   * - ``parts/base.py``
     - 类型体系：``RobotPart``、``ControllablePart``、``Camera``、
       ``EndEffector``、``Arm``、``MobileBase``、``LeggedBase``。
   * - ``parts/arms/``
     - 机械臂硬件，以及各系列的 state 数据类。
   * - ``parts/cameras/``
     - RealSense、ZED、Lumos。
   * - ``parts/end_effectors/``
     - ``grippers/`` 与 ``hands/``。
   * - ``parts/teleop/``
     - 主从臂与输入设备：GELLO、数据手套、键盘、Pico、SpaceMouse。
   * - ``parts/transports/``
     - ROS 等共享传输层。它们不是部件，只为部件传递消息。
   * - ``robots/``
     - 每台机器人一个模块：配置、发现逻辑和 builder。
   * - ``placement.py``
     - ``PartHandle`` 与自动合成的 worker。唯一导入调度器的模块。
   * - ``views.py``
     - ``Method*`` 系列视图。
   * - ``robot.py``、``discovery.py``、``adapters.py``、``config.py``
     - 组合、注册、旧策略适配器、环境变量配置。

任务不进入硬件代码
------------------

部件知道如何运动、能感知什么，但不知道什么算成功。重置行为、奖励、终止条件和
Gymnasium 空间属于 ``RobotTask``，由 ``RobotTaskEnv`` 与 ``Robot`` 组合在一起。
``LegacyObservationAdapter`` 和 ``VectorActionAdapter`` 负责把组合后的接口翻译成
既有策略所需的扁平向量与 ``state``/``frames`` 观测，因此硬件代码永远不必了解策略的
数据结构。

下一步
------

- :doc:`添加机器人 <../extending/new_robot>` —— 分步操作指南。
- :doc:`放置策略 <placement>` —— worker 如何映射到节点和 GPU。
