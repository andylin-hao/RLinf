添加机器人
==========

添加真机时，将设备 SDK、调度器放置和任务逻辑解耦。实现纯部件，描述物理布局，组合 ``Robot``，再将其规范接口适配到现有策略数据结构。

架构
----

让每一层只承担一种职责。

.. list-table::
   :header-rows: 1

   * - 层
     - 职责
   * - ``RobotPart`` 和 ``ControllablePart``
     - 管理一个设备连接，并公开规范观测与动作。
   * - ``Arm``
     - 组合一个机械臂驱动、可选 ``EndEffector`` 和命名腕部相机。
   * - ``Robot``
     - 组合命名机械臂、机器人级相机，以及底盘等可选部件。
   * - ``RobotSpec``
     - 描述物理机械臂、相机、末端执行器、连接信息和节点 rank。
   * - ``RobotDiscovery``
     - 将调度器硬件配置转换为通用硬件资源。
   * - ``RobotRuntime``
     - 协调本地和远程部件，包括并行多臂操作。
   * - ``RobotTask`` 和 ``RobotTaskEnv``
     - 管理重置、奖励、终止、Gymnasium 空间和策略兼容逻辑。

依赖方向必须保持严格：驱动不得导入 Ray、Gymnasium 或 ``rlinf.scheduler``。运行时宿主可以导入调度器 API。环境使用组合后的运行时，并将任务语义保留在设备驱动之外。

实现纯驱动
----------

仅提供观测的设备继承 ``RobotPart``。机械臂、底盘和其他可接收命令的设备继承 ``ControllablePart``。将可选厂商 SDK 放在 ``connect()`` 内导入，使未安装该 SDK 的节点也能导入模块。

.. code-block:: python

   import numpy as np

   from rlinf.robotics import ControllablePart


   class ExampleArmDriver(ControllablePart):
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

当存在更具体的接口时，使用 ``Camera``、``EndEffector``、``MobileBase`` 或 ``LeggedBase``。内置实现位于 ``rlinf/robotics/cameras``、``drivers``、``end_effectors``、``grippers`` 和 ``hands``。

组合机器人
----------

将每个机械臂驱动放入 ``Arm``。请使用稳定的机械臂名称，因为它们会成为规范观测和动作路径。单臂与双臂机器人使用相同结构。

.. code-block:: python

   from rlinf.robotics import Arm, Robot, RobotRuntime


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"


   robot = ExampleRobot.dual_arm(
       left_arm=Arm(ExampleArmDriver("tcp://left-arm:5000")),
       right_arm=Arm(ExampleArmDriver("tcp://right-arm:5000")),
   )
   runtime = RobotRuntime(robot)
   runtime.connect()
   observation = runtime.get_observation()
   runtime.send_action(
       {
           "arms": {
               "left": {
                   "arm": {"joint_position": left_target},
               },
               "right": {
                   "arm": {"joint_position": right_target},
               },
           }
       }
   )

左机械臂驱动的规范观测路径为 ``arms.left.state.joint_position``。末端执行器动作使用 ``arms.<name>.end_effector``。机器人级相机使用 ``cameras.<name>``；其他部件使用 ``parts.<name>``。

描述物理硬件
------------

将 ``RobotConfig.to_spec()`` 作为现有扁平 YAML 数据结构到规范物理布局的转换边界。连接和放置信息放入 ``RobotSpec``。重置位姿、奖励和回合长度保留在任务配置中。

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import ArmSpec, RobotConfig, RobotSpec


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str
       right_endpoint: str

       def to_spec(self) -> RobotSpec:
           return RobotSpec(
               robot_type=ExampleRobot.ROBOT_TYPE,
               node_rank=self.node_rank,
               arms=(
                   ArmSpec(
                       name="left",
                       driver="example",
                       node_rank=self.node_rank,
                       connection={"endpoint": self.left_endpoint},
                   ),
                   ArmSpec(
                       name="right",
                       driver="example",
                       node_rank=self.node_rank,
                       connection={"endpoint": self.right_endpoint},
                   ),
               ),
           )

机器人级或腕部相机使用 ``CameraSpec``，机械臂工具使用 ``EndEffectorSpec``，底盘、头部、升降机构或其他可选组件使用 ``PartSpec``。

注册发现逻辑
------------

让调度器发现逻辑与 ``Robot`` 分离。实现 ``RobotDiscovery.enumerate()``，并将发现类、物理配置类和组合类一起注册。

.. code-block:: python

   from typing import Optional

   from rlinf.robotics import (
       RobotDiscovery,
       RobotInfo,
       register_robot,
   )
   from rlinf.scheduler.hardware import (
       HardwareConfig,
       HardwareResource,
   )


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
                   RobotInfo(
                       type=cls.HW_TYPE,
                       model=cls.HW_TYPE,
                       config=config,
                   )
                   for config in matching
               ],
           )


   register_robot(ExampleRobotConfig, ExampleRobot)(ExampleRobotDiscovery)

在构造 ``Cluster`` 之前导入注册模块。RLinf 会将已注册硬件策略模块传播到节点探针，因此每个节点配置的 Python 环境都必须能导入该模块。

配置物理硬件
------------

保留现有 ``cluster.node_groups.hardware`` 数据结构。注册的配置类负责解析每一项，``to_spec()`` 提供规范布局。

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

远程托管驱动
------------

使用 ``ArmRuntime`` 在 RLinf ``Worker`` 中构造并连接纯机械臂驱动。将单 worker group 包装为 ``RemoteControllablePart``，再组合到本地 ``Robot``。

.. code-block:: python

   from rlinf.robotics import (
       Arm,
       ArmRuntime,
       RemoteControllablePart,
       Robot,
       RobotRuntime,
   )
   from rlinf.scheduler import NodePlacementStrategy

   group = ArmRuntime.create_group(
       ExampleArmDriver,
       {"endpoint": "tcp://left-arm:5000"},
   ).launch(
       cluster=cluster,
       placement_strategy=NodePlacementStrategy(node_ranks=[0]),
       name="ExampleArmRuntime",
   )
   remote_driver = RemoteControllablePart(group)
   runtime = RobotRuntime(Robot.single_arm(Arm(remote_driver)))
   runtime.connect()

``RobotRuntime`` 会并行执行彼此独立的机械臂重置、观测和动作。内置硬件通过 ``launch_franka_runtime``、``launch_dual_franka_runtime``、``launch_gim_arm_runtime``、``launch_turtle2_runtime`` 和 ``build_dosw1_runtime`` 使用同一边界。

分离任务与兼容逻辑
------------------

在 ``RobotTask`` 或真机环境中实现重置、奖励、成功判定、截断和 Gymnasium 空间。使用 ``RobotTaskEnv`` 组合任务与 ``RobotRuntime``。当现有策略需要扁平动作向量以及 ``state``/``frames`` 观测时，使用 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter``。

.. warning::

   引入规范接口时，不要修改现有 Gym ID、动作维度、观测键、相机名称或数据集字段。请添加适配器和回归测试。

测试集成
--------

测试无厂商 SDK 的纯驱动构造、组合路径、远程运行时生命周期、物理规格转换、发现注册，以及准确的旧策略数据结构。

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py \
     tests/unit_tests/test_robotics_boundaries.py \
     tests/unit_tests/test_robot_task_env.py \
     tests/unit_tests/test_realworld_robotics_compatibility.py

该命令无需物理硬件即可验证调度器边界、嵌套单臂与双臂组合、任务与运行时分离，以及所有内置真机环境的兼容性。
