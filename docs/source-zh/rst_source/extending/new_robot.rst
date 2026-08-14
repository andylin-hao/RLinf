添加机器人
==========

接入真实机器人，同时把设备 SDK、集群放置和任务逻辑分开。编写各部件，将它们组合成
``Robot``，完成注册，再让集群配置指向它。RLinf 会把部件托管到正确的机器上，并行
读取各部件，再向策略暴露它们。

开始前先阅读 :doc:`机器人模型 <../concepts/robotics>`。这篇文档说明本指南采用的
设计。每个物理组件都是 ``RobotPart``。驱动多个组件的硬件用 ``subparts()`` 声明
这些组件。``Robot`` 是具名组合，``spawn()`` 可以把任何部件放到节点上。文档还介绍
了 ``rlinf/robotics`` 的代码结构。

实现部件
--------

让只采集观测的设备继承 ``RobotPart``。让能接收命令的设备继承
``ControllablePart``。把厂商 SDK 的导入语句放在 ``connect()`` 内。这样，未安装该
SDK 的节点也能导入模块。

.. code-block:: python

   import numpy as np

   from rlinf.robotics import ControllablePart


   class ExampleArm(ControllablePart):
       def __init__(self, endpoint: str):
           self.endpoint = endpoint
           self._client = None

       @property
       def is_connected(self) -> bool:
           return self._client is not None

       @property
       def observation_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       @property
       def action_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       def connect(self) -> None:
           from example_robot_sdk import Client

           self._client = Client(self.endpoint)

       def reset(self) -> None:
           self._client.move_home()

       def get_observation(self) -> dict[str, np.ndarray]:
           return {"joint_position": self._client.get_joint_position()}

       def send_action(
           self, action: dict[str, np.ndarray]
       ) -> dict[str, np.ndarray]:
           if set(action) != {"joint_position"}:
               raise KeyError("Expected only 'joint_position'.")
           self._client.move_joints(action["joint_position"])
           return action

       def disconnect(self) -> None:
           if self._client is not None:
               self._client.close()
               self._client = None

如果设备符合更具体的接口，请使用 ``Camera``、``EndEffector``、``MobileBase`` 或
``LeggedBase``。

在一条连接上暴露多个组件
------------------------

如果一个套接字、CAN 总线或 ROS 节点驱动多个组件，请用 ``subparts()`` 声明这些
组件。按约定，把部件自身作为 ``"arm"`` 条目。

.. code-block:: python

   from rlinf.robotics import MethodGripper, RobotPart


   class ExampleArm(ControllablePart):
       ...

       def subparts(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(self, state_field="gripper_position"),
           }

如果硬件通过 ``open_gripper``、``move_left_arm``、``get_camera(id)`` 等命名方法
暴露能力，请用 ``MethodGripper``、``MethodArm`` 和 ``MethodCamera`` 将它适配成
部件。这样，组合层只处理统一接口。在 Python 中，把这些视图声明在所包装的方法旁边。

组合机器人
----------

用 ``Arm`` 包装每个机械臂本体。保持机械臂名称稳定，因为它们会成为规范观测和动作
路径。

.. code-block:: python

   from rlinf.robotics import Arm, Robot


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"


   robot = ExampleRobot.dual_arm(
       left_arm=Arm(ExampleArm("tcp://left-arm:5000")),
       right_arm=Arm(ExampleArm("tcp://right-arm:5000")),
   )
   robot.connect()
   observation = robot.get_observation()
   robot.send_action(
       {
           "arms": {
               "left": {"arm": {"joint_position": left_target}},
               "right": {"arm": {"joint_position": right_target}},
           }
       }
   )

左臂本体的规范观测路径是 ``arms.left.state.joint_position``。动作路径是
``arms.left.arm``。末端执行器动作使用 ``arms.<name>.end_effector``。机器人级相机
使用 ``cameras.<name>``，其他部件使用 ``parts.<name>``。

``Robot`` 会并行重置、读取和控制彼此独立的机械臂。读取双臂观测只需一个往返时间，
而不是两个。

在节点上放置部件
----------------

用 ``at()`` 声明部件运行在哪个节点。把这条声明放在你原本放置部件的位置。
``Robot.connect`` 会把它构建到该节点上，你无需调用任何放置函数。

.. code-block:: python

   from rlinf.robotics import Arm, Robot

   robot = Robot.single_arm(Arm(ExampleArm.at("tcp://left-arm:5000", node_rank=0)))
   robot.connect()

这段代码的作用：1) 为节点 0 声明 ``ExampleArm``；2) 机器人 connect 时构建并连接它；
3) 若该部件本身带末端执行器，就用同一条连接填充机械臂的末端执行器槽位。``connect``
会把每个句柄发布为 ``robot.handles[<name>]``，``disconnect`` 负责释放。

声明适用于所有部件，不只是机械臂。相机可以运行在它所插接的机器上::

   cameras={"scene": RealSenseCamera.at(info, node_rank=2)}

当一条连接支撑多个组件时，只声明一次并引用它的 subparts，这条连接就只会打开一次::

   hardware = ExampleHardware.at(node_rank=0)
   Arm(hardware.subpart("left"), hardware.subpart("left_end_effector"))

``spawn()`` 是底层的即时形式。只在机器人之外使用，例如调试脚本，此时句柄由你自己
管理。

无需为每种机器人编写 worker 类。RLinf 会依据部件类自动合成一个，``WorkerGroup``
随即把该类的每个公有方法绑定为 RPC。部件接口之外的方法仍可通过句柄调用，且本地与
远程的调用形式一致::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

描述并构建机器人
----------------

把连接和放置参数写入 ``RobotConfig`` 数据类。再写一个构建函数，用这些字段组合
``Robot``。把重置位姿、奖励和回合长度留在任务配置中。

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import Arm, RobotConfig


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str = ""
       right_endpoint: str = ""


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"

       @classmethod
       def build(cls, *, config: ExampleRobotConfig) -> "ExampleRobot":
           arms = {
               side: Arm(
                   ExampleArm.at(
                       endpoint,
                       node_rank=config.node_rank,
                       name=f"ExampleArm-{side}",
                   )
               )
               for side, endpoint in (
                   ("left", config.left_endpoint),
                   ("right", config.right_endpoint),
               )
           }
           return cls(arms=arms)

单臂型号沿用同一个 ``build()``，只是改为返回 ``ExampleRobot.single_arm(...)``。如果后续部件
启动失败，请先断开已经放置的句柄，再抛出错误。不要返回不完整的机器人。

注册机器人
----------

在机器人模块中，一次注册配置、组合、发现逻辑和构建函数。无需修改中央注册表。

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

把这次调用放在模块末尾，确保配置类和发现类都已定义。它会一次性注册机器人类、
配置、发现逻辑和 ``build``。注册完成后，用 ``build_robot("ExampleRobot", ...)``
按名称组合机器人，无需直接导入这个类。

继承已有机器人即可复用它的构建逻辑。``DualFrankaRobot`` 继承 ``FrankaRobot``，
原样沿用 ``declare_arms``，只覆盖 ``BACKEND`` 和 ``build``。

构造 ``Cluster`` 前，先导入注册模块。RLinf 会将已注册的硬件策略模块传给各节点的
探测流程。确保每个节点配置的 Python 环境都能导入该模块。

配置集群
--------

沿用现有的 ``cluster.node_groups.hardware`` 数据结构。注册的配置类会解析每一项。
注册的构建函数随后组合机器人。

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

把重置、奖励、成功判定、截断和 Gymnasium 空间写在 ``RobotTask`` 或真机环境中。
通过 ``RobotTaskEnv`` 组合任务与 ``Robot``。如果现有策略需要扁平动作向量以及
``state``/``frames`` 观测，请使用 ``LegacyObservationAdapter`` 和
``VectorActionAdapter``。

.. warning::

   引入规范接口时，保留既有的 Gym ID、动作维度、观测键、相机名称和数据集字段。
   请添加适配器和回归测试，不要改动这些内容。

测试集成
--------

在没有厂商 SDK 的环境中测试部件。还要测试组合路径、句柄生命周期、发现注册，以及
与旧策略完全一致的数据结构。

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py \
     tests/unit_tests/test_robotics_boundaries.py \
     tests/unit_tests/test_robot_task_env.py \
     tests/unit_tests/test_realworld_robotics_compatibility.py

这条命令会验证调度器边界、单臂与双臂组合、任务与机器人的分离，以及所有内置真机
环境面向策略的数据结构。这些测试都不需要真实硬件。
