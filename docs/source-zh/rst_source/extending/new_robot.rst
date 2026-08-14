添加机器人
==========

接入一台真实机器人，而不必把它的设备 SDK、集群放置和任务逻辑耦合在一起。你只需实现
部件、组合成 ``Robot``、完成注册，再让集群配置指向它。其余的事情——把部件托管到正确
的机器上、并行读取、暴露给策略——都由这一层负责。

开始之前，请先阅读 :doc:`机器人模型 <../concepts/robotics>`，了解本指南所依据的
设计：每个物理组件都是 ``RobotPart``；驱动多个组件的硬件用 ``subparts()`` 声明它们；
``Robot`` 是具名组合；任何部件都可以用 ``spawn()`` 放置到节点上。该页同时给出了
``rlinf/robotics`` 的代码结构。

实现部件
--------

仅提供观测的设备继承 ``RobotPart``；可接收命令的设备继承 ``ControllablePart``。
把厂商 SDK 放在 ``connect()`` 内导入，使未安装该 SDK 的节点也能导入该模块。

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

当存在更具体的接口时，使用 ``Camera``、``EndEffector``、``MobileBase`` 或
``LeggedBase``。

在一条连接上暴露多个组件
------------------------

当一个套接字、CAN 总线或 ROS 节点驱动多个组件时，用 ``subparts()`` 声明它们。
按约定，部件自身对应 ``"arm"`` 条目。

.. code-block:: python

   from rlinf.robotics import MethodGripper, RobotPart


   class ExampleArm(ControllablePart):
       ...

       def subparts(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(self, state_field="gripper_position"),
           }

``MethodGripper``、``MethodArm`` 和 ``MethodCamera`` 负责把以命名方法暴露能力的
硬件（``open_gripper``、``move_left_arm``、``get_camera(id)``）转换成部件，使组合
层看到统一接口。请在 Python 中、紧挨着被包装的方法处声明它们。

组合机器人
----------

将每个本体放入 ``Arm``。机械臂名称会成为规范观测和动作路径，因此要保持稳定。

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

左侧本体的规范观测路径为 ``arms.left.state.joint_position``，其动作路径为
``arms.left.arm``。末端执行器动作使用 ``arms.<name>.end_effector``；机器人级相机
使用 ``cameras.<name>``；其他部件使用 ``parts.<name>``。

``Robot`` 会并行执行彼此独立的机械臂重置、读取和下发，因此双臂观测只需一个往返
时间，而不是两个。

在节点上放置部件
----------------

``RobotPart.spawn`` 是唯一的放置入口，每个部件都具备。不传 ``node_rank`` 时部件在
当前进程内构造；传入时则托管在该节点的调度器 worker 中。两者返回的句柄 API 完全
相同，因此调用方无需区分放置方式。这不限于机械臂：相机可以运行在它所插接的机器上，
而策略运行在别处。

.. code-block:: python

   from rlinf.robotics import Arm, Robot

   handle = ExampleArm.spawn(
       "tcp://left-arm:5000",
       node_rank=0,
       name="ExampleArm-0",
   )
   robot = Robot.single_arm(
       Arm(handle.subpart("arm"), handle.subpart("end_effector")),
       handles={"arm": handle},
   )
   robot.connect()

这段代码的作用：1) 在节点 0 上构造 ``ExampleArm`` 并连接；2) 返回其 subparts 的
代理；3) 组合成持有该句柄的机器人。把机器人持有的句柄通过 ``handles=`` 传入，
``Robot.disconnect`` 会在所有部件断开后释放它们。

无需为每种机器人编写 worker 类。RLinf 会依据部件类自动合成一个，``WorkerGroup``
随即把该类的每个公有方法绑定为 RPC。部件接口之外的方法仍可通过句柄调用，且本地与
远程的调用形式一致::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

.. warning::

   ``WorkerGroup`` 保留了 ``launch``、``execute_on``、``from_group_name`` 和
   ``WorkerRank`` 这几个名字。若部件的公有方法与其重名，将无法被托管，请改名。

描述并构建机器人
----------------

把连接信息和放置信息放进 ``RobotConfig`` 数据类，并为它提供一个 builder，将这些
字段组合成 ``Robot``。重置位姿、奖励和回合长度则保留在任务配置中。

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import Arm, RobotConfig


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str = ""
       right_endpoint: str = ""


   def build_example_robot(config: ExampleRobotConfig) -> ExampleRobot:
       handles = {
           side: ExampleArm.spawn(
               endpoint,
               node_rank=config.node_rank,
               name=f"ExampleArm-{side}",
           )
           for side, endpoint in (
               ("left", config.left_endpoint),
               ("right", config.right_endpoint),
           )
       }
       return ExampleRobot.dual_arm(
           Arm(handles["left"].subpart("arm")),
           Arm(handles["right"].subpart("arm")),
           handles=handles,
       )

单臂型号使用同一个 builder，只是返回 ``ExampleRobot.single_arm(...)``。若后续某个
部件启动失败，请在抛出错误前断开已经放置的句柄，避免返回一个不完整的机器人。

注册机器人
----------

在机器人自己的模块中，用一次调用完成配置、组合、发现逻辑和 builder 的注册。无需
改动任何中心化的表。

.. code-block:: python

   from typing import Optional

   from rlinf.robotics import RobotDiscovery, RobotInfo, register_robot
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


   register_robot(
       ExampleRobotConfig, ExampleRobot, build=build_example_robot
   )(ExampleRobotDiscovery)

请把这次调用放在模块末尾，这样它才能引用到 builder。注册完成后，
``build_robot("ExampleRobot", ...)`` 即可按名字组合出机器人，无需直接导入它的
builder。

请在构造 ``Cluster`` 之前导入该注册模块。RLinf 会把已注册的硬件策略模块传播到各
节点的探测流程，因此该模块必须能在每个节点配置的 Python 环境中被导入。

配置集群
--------

保留现有的 ``cluster.node_groups.hardware`` 数据结构。注册的配置类负责解析每一项，
注册时提供的 builder 负责组合出机器人。

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

在 ``RobotTask`` 或真机环境中实现重置、奖励、成功判定、截断和 Gymnasium 空间。
使用 ``RobotTaskEnv`` 组合任务与 ``Robot``。当现有策略需要扁平动作向量以及
``state``/``frames`` 观测时，使用 ``LegacyObservationAdapter`` 和
``VectorActionAdapter``。

.. warning::

   在引入规范接口时，请勿更改既有的 Gym ID、动作维度、观测键、相机名称或数据集
   字段。应改为增加适配器和回归测试。

测试集成
--------

在没有厂商 SDK 的情况下测试部件、组合路径、句柄生命周期、发现注册，以及与旧策略
完全一致的数据结构。

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py \
     tests/unit_tests/test_robotics_boundaries.py \
     tests/unit_tests/test_robot_task_env.py \
     tests/unit_tests/test_realworld_robotics_compatibility.py

这段命令的作用：验证调度器边界、单臂与双臂组合、任务与机器人的分离，以及所有内置
真机环境面向策略的数据结构；全部无需真实硬件。
