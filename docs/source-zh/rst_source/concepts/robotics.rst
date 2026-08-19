机器人模型
==========

理解 RLinf 的机器人模型，可以先从一句话开始：机器人是一棵具名部件树。每个部件
都会返回观测，其中一部分还能接收动作。

如果你正在写任务、接策略或调试已有机器人，记住这一点就够了。机械臂、夹爪、相机
和移动底盘各占树上的一条路径；部件若能接收动作，动作也沿用这条路径。

本页会带你跑一遍已有的 Franka，说明怎样读部件树，以及机器人代码在哪里与任务分开。读完这一页
不需要先掌握放置 worker 和硬件会话；如果你准备扩展 robotics，页末会引导你继续阅读架构页。

先用一台已有机器人
----------------------

先按注册名构建机器人，看清它的组成，再连接硬件。建议把 ``disconnect()`` 放进
``finally``；即使 rollout 或调试命令中途报错，硬件资源也能正常释放。

.. code-block:: python

   from rlinf.robotics import build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # 这一步不会打开任何硬件。
   print(robot.describe())

   robot.connect()
   try:
       observation = robot.get_observation()
       tcp_pose = observation["arm"]["tcp_pose"]
       gripper_width = observation["end_effector"]["state"]

       robot.send_action(
           {"arm": {"tcp_pose": target}, "end_effector": {"target": width}}
       )
   finally:
       robot.disconnect()

``get_observation()`` 返回一个嵌套字典。``send_action()`` 接收同样的部件分支，
再返回实际执行的动作。每次只发送需要控制的分支即可，不必补齐整棵树。

怎么看 ``describe()``
-------------------------

部件名是面向策略和数据集的公开约定。单臂 Franka 的顶层有 ``arm`` 和
``end_effector`` 两个部件：

.. code-block:: text

   FrankaRobot
   ├── arm           declared      node=1     via FrankaROSArm#1
   └── end_effector  declared      node=1     via FrankaROSArm#1

两行的 ``via`` 相同，说明机械臂和夹爪共用一条 Franka 连接。使用机器人时不必
处理这层细节；它出现在这里，是为了让你在真机运动前就能发现放置或资源归属错误。

换成双臂机器人后，使用方式不变。原来的部件名只是多了 ``left`` 和 ``right``
两层前缀：

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

组合树可以继续向下嵌套。RLinf 没有固定的 ``arms`` 或 ``cameras`` 槽位，因此新增升降
机构、云台或第三条机械臂时，不用再设计一套机器人接口。

本地与远端部件的用法相同
--------------------------

路径只说明部件在机器人里的位置，不说明它运行在哪台机器上。接在本机的相机和由其他
节点控制的机械臂，仍会出现在同一棵观测树中。RLinf 会并行访问相互独立的硬件连接，
共用连接的部件则保持声明顺序。

因此，任务和策略代码只处理名字与数值，不接触 Ray actor、RPC、串口或厂商会话。现有策略
若仍需要扁平向量，可以在环境边界使用 ``LegacyObservationAdapter`` 和
``VectorActionAdapter``。

任务与硬件各管一侧
--------------------

部件知道怎么读传感器、怎么让执行器运动，但它不判断一个 rollout 是否成功。复位行为、奖励、
终止条件和 Gymnasium 空间属于 ``RobotTask``，``RobotTaskEnv`` 再把任务与机器人组合起来。

这条边界让同一台机器人可以换任务而不改设备代码，也让部件改变运行节点后，策略仍看到原来的
数据结构。

接下来读什么
----------------

- 想在已支持的硬件上新增任务，继续阅读 :doc:`新增任务 <../extending/new_task>`。
- 想先接入一个本地传感器或执行器，继续阅读 :doc:`添加机器人 <../extending/new_robot>`。
- 想了解共享连接、``Endpoint``、远程句柄、失败回滚和 worker 放置，阅读
  :doc:`机器人架构 <robotics_architecture>`。
- 想组合多种操作者设备，阅读 :doc:`遥操作 <../guides/teleoperation>`。
