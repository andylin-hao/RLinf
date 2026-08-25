机器人模型
==========

在 RLinf 中，机器人由机械臂、夹爪、相机和移动底盘等零部件组成。每个零部件都有稳定的名称；层级化的名称既是观测数据的 key，也是发送动作时使用的路径。所有零部件都能返回观测，可控零部件还可以接收动作。

如果仅需编写任务、接入 policy 或调试现有机器人，了解上述模型即可。以下以 Franka 为例，说明如何查看机器人的组成、读写具名零部件，以及划分任务代码与硬件代码的边界。接入新设备或排查底层问题时，再参阅架构文档中的 worker、远程调用和硬件连接机制。

使用现有机器人
--------------

通过注册名构建机器人后，可先调用 ``describe()`` 检查零部件名称、访问路径和部署位置。该方法不会连接硬件，因此可在机械臂上电前运行。实际使用时，建议将 ``disconnect()`` 置于 ``finally`` 块中，确保 rollout 或调试命令发生异常时仍能关闭硬件连接。

.. code-block:: python

   from rlinf.robotics import build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # 这一步不会打开任何硬件。
   print(robot.describe())

   robot.connect()
   try:
       observation = robot.get_observation()
       tcp_pose = observation["arm"]["tcp_pose"]
       gripper_width = observation["arm"]["end_effector"]["state"]

       robot.send_action(
           {
               "arm": {
                   "tcp_pose": target,
                   "end_effector": {"target": width},
               }
           }
       )
   finally:
       robot.disconnect()

``get_observation()`` 返回嵌套字典，其层级与机器人中零部件的组合关系一致。调用 ``send_action()`` 时沿用相同路径，只需指定本次要控制的零部件，不必传入其他分支。

以零部件名称访问观测和动作
--------------------------

单臂 Franka 的顶层只有 ``arm``。末端执行器安装在机械臂上，因此位于 ``arm`` 分支下：

.. code-block:: text

   FrankaRobot
   └── arm                 FrankaROSArm         node=1     via FrankaROSArm#1
       └── end_effector    MethodEndEffector    node=1     via FrankaROSArm#1

两行输出中的 ``via`` 值相同，表示机械臂和夹爪共用一条 Franka connection。读取观测和发送动作时无需处理这条 connection；排查重复连接或节点配置时，可结合 ``node`` 和 ``via`` 确认资源归属。

这一结构也说明了何时可以直接向 ``Robot`` 传入零部件，何时需要调用 ``part(name)``。``FrankaROSArm`` 本身就是能返回观测的 ``RobotPart``，因此可通过 ``Robot(arm=arm)`` 直接组合；安装在机械臂上的末端执行器也会同时出现在 ``arm.end_effector``。如果某个共享控制器只是 ``Connection``、本身不能返回观测，则需先通过 ``session.part("left")`` 取出其中的零部件，再将其加入 ``Robot``。

双臂机器人在顶层增加 ``left`` 和 ``right`` 两个分支，各零部件的名称保持不变：

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["arm"]["end_effector"]["state"]

这种组合关系可以继续嵌套。RLinf 不预设 ``arms``、``cameras`` 等固定分组；新增升降机构、云台或第三条机械臂时，只需为新零部件命名，并将其加入合适的层级。

调用方式与部署位置解耦
----------------------

零部件的访问路径只表示其在机器人中的位置，不包含部署信息。本机相机和远端机械臂仍使用同一套观测层级。RLinf 会并行访问相互独立的硬件连接；共用同一连接的零部件则按声明顺序访问。

任务和 policy 代码只依赖零部件的名称和数据，不需要区分底层使用的是 Ray actor、RPC、串口还是厂商 SDK。现有 policy 如果仍接收扁平向量，可在环境边界使用 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter`` 完成格式转换。

区分硬件驱动与任务逻辑
------------------------

硬件 driver 读取传感器并控制执行器。rollout 的成功条件、奖励、终止条件、复位流程和 Gymnasium space 由 ``RobotTask`` 定义；``RobotTaskEnv`` 再将任务与机器人组合。为同一台机器人切换任务时，无需修改底层设备代码。

后续阅读
--------

- :doc:`新增真机任务 <../extending/new_task>`：在已支持的真机上添加任务。
- :doc:`添加机器人 <../extending/new_robot>`：接入本地传感器、执行器或整台机器人。
- :doc:`机器人架构 <robotics_architecture>`：了解 ``parts`` 和 ``children`` 分别表示什么，以及共享 connection、``PartGroup``、资源生命周期和 worker placement。
- :doc:`遥操作 <../guides/teleoperation>`：组合多种操作者设备。
