添加机器人
==========

接入新硬件时，建议按“先本地、再组合、后远端”的顺序来做。本页先让一台设备在当前进程
中跑通，再将它组合进机器人，接着处理共享连接，最后才把部件放到其他节点。这样出现问题时，
可以先判断是设备本身还是放置逻辑。

开始前，先读一遍简短的 :doc:`机器人模型 <../concepts/robotics>`。如果硬件已经支持，你只需要
新的奖励、复位或成功判定，请直接阅读 :doc:`new_task`；这种情况不用新增机器人类。

1. 先接入一个本地部件
----------------------

先选一个能独立验证的设备。传感器继承 ``RobotPart``；设备若还要接收命令，则继承
``ControllablePart``。这一步先不写集群配置，只确认部件在当前进程能正常连接、读取和断开。

每个部件都回答同样的三个问题：``_open`` 连上硬件，``get_observation`` 读它，
``_release`` 放开它。连接和断开由基类统一处理，所以写一个部件就是说清楚它的硬件是什么。
厂商 SDK 应当在 ``_open`` 内导入：这样只有真正打开连接的节点需要安装 SDK，其他节点
仍可导入部件模块。

.. code-block:: python

   import numpy as np

   from rlinf.robotics import ControllablePart


   class ExampleArm(ControllablePart):
       def __init__(self, endpoint: str):
           self.endpoint = endpoint

       def _open(self):
           from example_robot_sdk import Client

           return Client(self.endpoint)

       def _release(self, device) -> None:
           device.close()

       @property
       def observation_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       @property
       def action_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       def reset(self) -> None:
           self._device.move_home()

       def get_observation(self) -> dict[str, np.ndarray]:
           return {"joint_position": self._device.get_joint_position()}

       def send_action(
           self, action: dict[str, np.ndarray]
       ) -> dict[str, np.ndarray]:
           if set(action) != {"joint_position"}:
               raise KeyError("Expected only 'joint_position'.")
           self._device.move_joints(action["joint_position"])
           return action

组合机器人之前，先直接跑一遍这个部件：

.. code-block:: python

   arm = ExampleArm("tcp://left-arm:5000")
   arm.connect()
   try:
       print(arm.get_observation())
   finally:
       arm.disconnect()

``_open()`` 返回的对象会保存在 ``self._device`` 中。断开时，``_release(device)`` 收到
的就是同一个对象；清理代码应当释放传入的 ``device``，不要再从 ``self`` 上查找。

把打开设备的逻辑放在 ``connect()`` 阶段，而不是 ``__init__`` 中，之后才能在一台机器上声明部件、
再到另一台机器上构建。如果机械臂连接后还需要回零，可以重写 ``connect()`` 和
``disconnect()``，但两个方法都要支持重复调用，这样失败回滚和重新连接才是安全的。

设备若符合 ``Camera``、``EndEffector``、``MobileBase`` 或 ``LeggedBase`` 的接口，
就继承对应的具体类型。后续组合部件或创建远程代理时，代码仍能识别它的设备类别。

2. 硬件共享连接时再写 ``exports``
--------------------------------------

本地部件跑通后，再看它的连接是否还管理其他硬件。一个网络连接、一条 CAN 总线或一个 ROS
节点可能同时驱动多个物理组件。这时用 ``exports`` 列出它能对外暴露的部件。调用方可以分别
命名和访问它们，底层连接仍然只打开一次。对于机械臂，约定用 ``"arm"`` 暴露机械臂本身。

.. code-block:: python

   from rlinf.robotics import MethodGripper, RobotPart


   class ExampleArm(ControllablePart):
       ...

       @property
       def exports(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(self, state_field="gripper_position"),
           }

有些 SDK 用 ``open_gripper``、``move_left_arm``、``get_camera(id)`` 等方法暴露硬件
能力。此时用 ``MethodGripper``、``MethodArm`` 或 ``MethodCamera`` 包一层视图，并把
视图写在所适配的方法旁边。到了机器人组合这一步，看到的便是普通部件，而不是厂商 API
中的方法名。

3. 组合对外稳定的部件名
--------------------------

此时可以先手动组合一台机器人，还不需要写硬件发现和 YAML。共享连接只声明一次，再从它的
``exports`` 中选出需要的部件，给它们安排清晰的名字：

.. code-block:: python

   from rlinf.robotics import Robot


   class Bench(Robot):
       ROBOT_TYPE = "Bench"


   connection = ExampleArm.at("tcp://left-arm:5000")
   robot = Bench(
       arm=connection.export("arm"),
       end_effector=connection.export("end_effector"),
   )
   print(robot.describe())
   robot.connect()
   try:
       print(robot.get_observation())
   finally:
       robot.disconnect()

这个小例子不需要硬件配置和 discovery。它们的用途是让配置文件可以根据类型名构建机器人；
工装脚本可以先保持这种简单写法。

部件名是策略和数据集会长期使用的字段名，命名时要把它当作公开接口。数据开始采集后再改名，
就会同时改变观测、动作和数据集结构。

.. code-block:: python

   from rlinf.robotics import Group, Robot


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"


   left = ExampleArm.at("tcp://left-arm:5000")
   right = ExampleArm.at("tcp://right-arm:5000")
   robot = ExampleRobot(
       left=Group(
           arm=left.export("arm"),
           end_effector=left.export("end_effector"),
       ),
       right=Group(
           arm=right.export("arm"),
           end_effector=right.export("end_effector"),
       ),
   )
   robot.connect()
   try:
       observation = robot.get_observation()
       robot.send_action(
           {
               "left": {"arm": {"joint_position": left_target}},
               "right": {"arm": {"joint_position": right_target}},
           }
       )
   finally:
       robot.disconnect()

路径完全由上面的名字组成，中间不会自动加入 ``arms`` 或 ``cameras`` 之类的层级。左臂
位于 ``left.arm``，左夹爪位于 ``left.end_effector``；如果相机的组合名为 ``wrist``，
它的路径就是 ``wrist``。

``Robot`` 会并行重置、读取和控制使用独立连接的机械臂。因此，读取双臂观测只需等待
一次往返；机器人子类中无需另写并发代码。

4. 本地组合跑通后，再放到远端
------------------------------

先确认本地组合可以正常连接、读取和断开。之后只需在同一份声明上加入 ``node_rank``，
部件实现和机器人树都不用改：

.. code-block:: python

   connection = ExampleArm.at("tcp://left-arm:5000", node_rank=0)
   robot = Robot(
       arm=connection.export("arm"),
       end_effector=connection.export("end_effector"),
   )
   robot.connect()
   try:
       print(robot.get_observation())
   finally:
       robot.disconnect()

这个 ``at()`` 只记录一份目标节点为 0 的 ``ExampleArm`` 声明。直到 ``connect()``
执行时，系统才创建 worker、导入 SDK 并打开设备。所得句柄保存在 ``robot.handles`` 中，
``disconnect()`` 会在退出时回收它。

这种声明并不只适用于机械臂。例如，相机可以留在实际插接的机器上::

   scene=RealSenseCamera.at(info, node_rank=2)

一条连接支撑多个组件时，只声明一次连接，再从中选取暴露的部件::

   connection = ExampleConnection.at(node_rank=0)
   Group(arm=connection.export("left"), gripper=connection.export("left_end_effector"))

更底层的 ``spawn()`` 会立即放置部件。它适合机器人之外的场景，例如调试脚本；脚本
此时也要自行管理句柄的生命周期。

部件不需要配套手写 worker 类。放置代码根据部件类自动合成 worker，再由
``WorkerGroup`` 把公有方法绑定为 RPC。标准部件接口之外的方法仍可通过句柄调用，
本地与远程使用同一种写法::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

5. 让配置文件也能构建这台机器人
----------------------------------

部件定义完成后，把机器人的硬件输入写进 ``RobotConfig`` 数据类，再由 ``build()``
完成组装。连接地址和节点放置写在这里；重置位姿、奖励、回合长度仍留在任务配置中，
因为它们描述的是一轮任务，而不是硬件发现。

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import Group, RobotConfig


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str = ""
       right_endpoint: str = ""


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"

       @classmethod
       def build(
           cls,
           *,
           left_endpoint: str,
           right_endpoint: str,
           node_rank: int = 0,
           **_,
       ) -> "ExampleRobot":
           arms = {}
           for side, endpoint in (
               ("left", left_endpoint),
               ("right", right_endpoint),
           ):
               connection = ExampleArm.at(
                   endpoint,
                   node_rank=node_rank,
                   name=f"ExampleArm-{side}",
               )
               arms[side] = Group(
                   arm=connection.export("arm"),
                   end_effector=connection.export("end_effector"),
               )
           return cls(**arms)

单臂型号可以沿用同样的写法，只返回一个 ``Group``。``build()`` 应当只组合
``PartSpec``，不要在这里打开硬件。后面由 ``Robot.connect()`` 统一启动；某个部件失败时，
它会回收先前已放置的资源，再恢复声明树。

.. warning::

   ``build()`` 只组合声明，不会访问硬件。读取观测或发送命令前，先对返回的机器人调用
   ``connect()``；结束时再调用 ``disconnect()``。真机环境会在硬件初始化和清理流程中
   执行这些调用。

注册机器人
----------

配置、构建方法和发现逻辑都写好后，在机器人自己的模块中完成注册，不需要再去修改中央
注册表。

.. code-block:: python

   from typing import Optional

   from rlinf.robotics import RobotDiscovery, RobotInfo
   from rlinf.scheduler.hardware import HardwareConfig, HardwareResource


   class ExampleRobotDiscovery(RobotDiscovery):
       HW_TYPE = ExampleRobot.ROBOT_TYPE

       @classmethod
       def enumerate(
           cls,
           node_rank: int,
           configs: Optional[list[HardwareConfig]] = None,
       ) -> Optional[HardwareResource]:
           matching = [
               config
               for config in configs or []
               if isinstance(config, ExampleRobotConfig)
               and config.node_rank == node_rank
           ]
           if not matching:
               return None
           return HardwareResource(
               type=cls.HW_TYPE,
               infos=[
                   RobotInfo(type=cls.HW_TYPE, model=cls.HW_TYPE, config=config)
                   for config in matching
               ],
           )


   ExampleRobot.register(ExampleRobotConfig, ExampleRobotDiscovery)

把注册调用放在模块末尾，等配置类和发现类都定义完成后再执行。它会关联机器人类、配置、
发现逻辑和 ``build``。之后，调用方使用 ``build_robot("ExampleRobot", ...)`` 即可按
名称组装机器人，无需导入具体类。

新硬件若只是现有机器人的变体，可以直接继承。例如，``DualFrankaRobot`` 在
``FrankaRobot`` 上修改 ``build_arms`` 和 ``BACKEND``，而 ``build`` 与生命周期方法仍
使用父类版本。

构造 ``Cluster`` 前必须先导入注册模块，否则硬件发现还不认识这款机器人。已经注册的
硬件策略模块也会传给各节点探测，所以每个节点配置的 Python 环境都要能导入该模块。

配置集群
--------

最后，把构建所需的硬件信息写入现有的 ``cluster.node_groups.hardware`` 结构。注册的
配置类解析每一项，再由构建方法组装机器人。endpoint、node rank 等部署信息应当留在
YAML 中，不要写死在 Python 代码里：

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement: {}
     node_groups:
       - label: example_robot
         node_ranks: 0
         hardware:
           type: ExampleRobot
           configs:
             - node_rank: 0
               left_endpoint: tcp://left-arm:5000
               right_endpoint: tcp://right-arm:5000

分离任务与兼容逻辑
------------------

做到这里，机器人已经知道去哪里找硬件、怎样操作硬件，但它不应该定义任务。重置、奖励、
成功判定、截断和 Gymnasium 空间属于 ``RobotTask`` 或真机环境，再通过
``RobotTaskEnv`` 与 ``Robot`` 组合。现有策略若使用扁平动作向量和
``state``/``frames`` 观测，就在这条边界上加入 ``LegacyObservationAdapter`` 和
``VectorActionAdapter``。

.. warning::

   引入规范接口时，既有 Gym ID、动作维度、观测字段、相机名称和数据集字段都不能变。
   用适配器和回归测试保证兼容，避免破坏已经训练好的策略和现有数据集。

检查组合结果
------------

连接任何硬件之前，先让机器人把组合结果列出来::

   >>> print(robot.describe())
   ExampleRobot
   ├── left.arm             declared      node=0     via ExampleArm#1
   ├── left.end_effector    declared      node=0     via ExampleArm#1
   ├── right.arm            declared      node=0     via ExampleArm#2
   └── right.end_effector   declared      node=0     via ExampleArm#2

每一行都列出一个部件、它将要运行的节点，以及它来自哪份声明。``via`` 相同的部件
共用一条连接，因此它们只打开一次，访问时也按声明顺序执行。``connect()`` 之后再调用
``describe()``，节点和资源归属仍然不变，因为这些信息来自最初保存的声明。

测试集成
--------

接下来，在 ``tests/unit_tests/`` 中新增测试，把一致性检查（conformance suite）指向刚写好的
部件、连接和机器人。这些 contract 类会检查框架依赖的约定，也可以沿用工装检查中的假 SDK：

.. code-block:: python

   from robot_contracts import ConnectionContract, PartContract, RobotContract


   def test_my_arm_conforms():
       PartContract(
           lambda: MyArm("10.0.0.1"),
           action={"joint_position": np.zeros(6)},
       ).assert_kept()


   def test_my_link_conforms():
       ConnectionContract(MyConnection).assert_kept()


   def test_my_robot_conforms():
       RobotContract(lambda: MyRobot.build(robot_ip="10.0.0.1")).assert_kept()

它们放在 ``tests/robot_contracts``，和 ``tests/robot_mocks`` 里的假 SDK 挨在一起：
它们是用来检查 RLinf 的，而不是 RLinf 的一部分。

这些检查会连接、读取和断开设备，然后重复一次完整生命周期，最后再断开一次。它们会对照声明
检查观测字段和 shape。``PartContract`` 若收到一份示例动作，还会确认该动作能够执行，未知字段会被拒绝。
``RobotContract`` 会在 ``connect()`` 期间注入失败，并确认机器人不会报告一棵只连接了一部分的树。

上面每一条都是这个包真实出过的 bug。失败信息说的是「哪条约定没被守住」，而不是某个
断言，而且会一次列全：

.. code-block:: text

   ConformanceError: MyArm does not keep 2:
     - MyArm: reconnecting raised RuntimeError: threads can only be started
       once; stall recovery closes an endpoint and opens it again
     - MyArm observes tcp_pose with shape (7,), declares (6,)

接入逻辑的其余部分也用同样的方式测试。至少覆盖部件接口、组合路径、句柄生命周期、
注册与硬件发现，以及旧策略依赖的完整数据结构：

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py tests/unit_tests/test_real_env.py

这组测试覆盖调度器导入边界、单臂与双臂组合、任务与机器人的分界，以及所有内置真机
环境暴露给策略的数据结构；运行时不需要真实硬件。

用假 SDK 跑一遍
~~~~~~~~~~~~~~~

部件只在打开时才导入厂商 SDK，不在模块导入时导入，所以只要往 ``sys.modules`` 里放一份
假的，真实的部件类就能在线缆另一端空无一物的情况下跑起来。``tests/robot_mocks``
为每个 SDK 提供了一份。

让机器人的组合过程跑在假 SDK 上：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MyRobot --mock \
       --arg robot_ip=10.0.0.1 --arg node_rank=0

它会列出机器人由哪些部件组成、每个部件挂在哪条连接上、被放到了哪个节点，然后逐个读取
观测并断开。以下情况会判定失败：某个部件返回了它没有声明过的观测；某个观测的形状与
声明不符；连接本身出现在部件树里；断开之后仍有东西声称自己是连着的。

其中形状检查最不显眼，却最值得有：环境是照着部件的声明来搭观测空间的，因此某个值多出
一个数时，它是以数据的形式、而不是以报错的形式抵达策略。

加上 ``--remote``，部件会被托管到调度器 worker 里，而不是留在当前进程。这一步专门用来
暴露根本放不出去的部件：方法名和 worker 自己的撞车，或者状态跨不过进程边界。

完整的训练同样可以这么跑。配置名里带 ``mock`` 时，``run.sh`` 会先装好这些假 SDK：

.. code-block:: bash

   bash tests/e2e_tests/embodied/run.sh realworld_mock_sac_cnn

每种机器人都有一份，因此组合、wrapper 栈、观测空间和 runner 都会按上真机时的样子跑一遍：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 机器人
     - 配置
   * - Franka
     - ``realworld_mock_sac_cnn``
   * - 双臂 Franka
     - ``realworld_dual_franka_mock_sac_cnn``
   * - GimArm
     - ``gim_arm_mock_sac_cnn``
   * - Turtle2
     - ``realworld_xsquare_turtle2_mock_sac_cnn``
   * - DOSW1
     - ``dosw1_mock_sac_mlp_pick``

用真机跑一遍
~~~~~~~~~~~~

剩下的部分必须有硬件才能验证：时序、标定，以及设备文档里没写的那些行为。等机器人上电、
网络可达之后，去掉 ``--mock``，同一条检查就跑在它上面：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MyRobot \
       --arg robot_ip=10.0.0.1 --arg node_rank=1
