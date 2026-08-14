添加机器人
==========

通过实现可复用部件、将部件组合为 ``Robot`` 并注册发现配置来添加真机。将任务逻辑保留在真机环境中，不要放入机器人驱动。

架构
----

每一层只承担一种职责：

.. list-table::
   :header-rows: 1

   * - 层
     - 职责
   * - ``RobotPart``
     - 连接一个可观测设备并返回其观测。
   * - ``ControllablePart``
     - 为机械臂、夹爪或移动底盘等部件添加动作。
   * - ``Robot``
     - 组合命名部件，并公开带命名空间的观测和动作。
   * - ``RobotConfig`` 和 ``RobotInfo``
     - 描述物理连接配置和调度器发现结果。
   * - 真机环境
     - 定义重置、奖励、终止条件以及面向策略的空间。
   * - ``PartRuntime``
     - 可选地通过 RLinf ``Worker`` 在其他节点托管部件。

不要在部件实现中导入 Ray、Gymnasium 或调度器 API。这样，同一个部件可以在本地、组合机器人或 ``PartRuntime`` 中复用。

实现部件
--------

对于相机等仅提供观测的设备，继承 ``RobotPart``。对于还接收动作的设备，继承 ``ControllablePart``。

.. code-block:: python

   import numpy as np

   from rlinf.robotics import ControllablePart


   class ExampleArm(ControllablePart):
       def __init__(self, endpoint: str):
           self.endpoint = endpoint
           self._connected = False

       @property
       def is_connected(self) -> bool:
           return self._connected

       @property
       def observation_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       @property
       def action_features(self) -> dict:
           return {"joint_target": {"shape": (6,), "dtype": "float32"}}

       def connect(self) -> None:
           self._connected = True

       def get_observation(self) -> dict[str, np.ndarray]:
           return {"joint_position": np.zeros(6, dtype=np.float32)}

       def send_action(
           self, action: dict[str, np.ndarray]
       ) -> dict[str, np.ndarray]:
           # 在此处通过厂商 SDK 发送 action["joint_target"]。
           return action

       def disconnect(self) -> None:
           self._connected = False

如果只有机器人节点安装厂商 SDK，请将厂商 SDK 导入放在 ``connect()`` 或构造函数内部。

组合机器人
----------

为每个部件指定稳定名称。这些名称会成为观测和动作的顶层键，因此同一结构可以支持单臂、双臂、相机、灵巧手、轮式底盘或腿式底盘。

.. code-block:: python

   from rlinf.robotics import Robot

   robot = Robot(
       parts={
           "left_arm": ExampleArm("tcp://left-arm:5000"),
           "right_arm": ExampleArm("tcp://right-arm:5000"),
       }
   )
   robot.connect()
   observation = robot.get_observation()
   robot.send_action(
       {
           "left_arm": {"joint_target": left_target},
           "right_arm": {"joint_target": right_target},
       }
   )

注册发现逻辑
------------

为每种硬件类型注册一个 ``RobotConfig`` 和发现策略。调度器只使用通用的 ``HardwareResource`` 结果；机器人专用验证应保留在此模块中。

.. code-block:: python

   from dataclasses import dataclass
   from typing import Optional

   from rlinf.robotics import Robot, RobotConfig, RobotInfo
   from rlinf.scheduler.hardware import (
       HardwareConfig,
       HardwareInfo,
       HardwareResource,
   )


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       endpoint: str


   @Robot.register_robot(ExampleRobotConfig)
   class ExampleRobot(Robot):
       HW_TYPE = "ExampleRobot"

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

           infos: list[HardwareInfo] = [
               RobotInfo(
                   type=cls.HW_TYPE,
                   model=cls.HW_TYPE,
                   config=config,
               )
               for config in matching
           ]
           return HardwareResource(type=cls.HW_TYPE, infos=infos)

在构造 ``Cluster`` 之前导入集成模块。注册在导入时发生，并且该模块必须能在每个集群节点的 Python 环境中导入。

.. code-block:: python

   import my_project.example_robot  # 注册 ExampleRobot。

   from rlinf.scheduler import Cluster

   cluster = Cluster(cluster_cfg=cfg.cluster)

配置物理硬件
------------

将物理连接配置放在 ``cluster.node_groups.hardware`` 下。将重置位姿、奖励和回合长度等任务参数保留在 ``env`` 下。

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
               endpoint: tcp://robot-arm:5000

远程托管部件
------------

当设备必须在其他节点上运行时，使用 ``PartRuntime``。它是 RLinf ``Worker``，因此其放置和生命周期与其他 RLinf 组件使用相同的调度器 API。

.. code-block:: python

   from rlinf.robotics import PartRuntime
   from rlinf.scheduler import NodePlacementStrategy

   arm_runtime = PartRuntime.create_group(
       ExampleArm,
       {"endpoint": "tcp://robot-arm:5000"},
   ).launch(
       cluster=cluster,
       placement_strategy=NodePlacementStrategy(node_ranks=[0]),
       name="ExampleArmRuntime",
   )
   arm_runtime.initialize().wait()

测试集成
--------

使用模拟 SDK 或回环传输，在没有硬件的情况下测试部件。覆盖连接清理、观测键、动作分发、配置解析，以及每种支持布局的发现结果。

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py

该命令会在不运行完整训练的情况下执行内置的组合与注册契约测试。请为你的部件添加同等的单元测试；如果验证需要物理设备或厂商 SDK，请再添加一个受硬件条件限制的端到端测试。
