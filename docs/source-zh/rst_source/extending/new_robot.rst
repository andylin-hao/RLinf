添加机器人
==========

接入真实机器人时，先把设备 SDK、集群放置和任务逻辑分开。你需要把硬件建模成部件，
组合成 ``Robot``，完成注册，再从集群配置中引用它。边界理顺之后，RLinf 可以让部件
就近运行在硬件所在的机器上，并行读取独立连接，同时给策略一套完整、统一的接口。

动手前先读 :doc:`机器人模型 <../concepts/robotics>`，了解这套设计为什么这样拆。简要
来说，每个物理组件都对应一个 ``RobotPart``；一套硬件若驱动多个组件，就通过
``parts`` 列出它们。``Robot`` 为这些部件确定稳定的名字，``spawn()`` 则能把任意部件
放到其他节点。查找具体实现时，也可以参考其中的 ``rlinf/robotics`` 目录说明。

实现部件
--------

先按设备能力选择最窄的接口：只采集观测就继承 ``RobotPart``，还要接收命令则继承
``ControllablePart``。厂商 SDK 放到 ``connect()`` 内再导入。这样，只有实际连接硬件
的节点需要安装依赖，其他节点仍能正常导入这个模块。

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

如果设备符合 ``Camera``、``EndEffector``、``MobileBase`` 或 ``LeggedBase`` 这类更
具体的接口，优先继承对应类型。组合部件或创建远程代理时，设备类别便不会丢失。

在一条连接上暴露多个组件
------------------------

一条 socket、CAN 总线或 ROS 节点可能同时驱动多个物理组件。此时用 ``parts`` 把它们
全部列出来：底层连接只打开一次，上层仍能分别访问每项能力。按照约定，部件自身放在
``"arm"`` 条目下。

.. code-block:: python

   from rlinf.robotics import MethodGripper, RobotPart


   class ExampleArm(ControllablePart):
       ...

       @property
       def parts(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(self, state_field="gripper_position"),
           }

有些 SDK 通过 ``open_gripper``、``move_left_arm``、``get_camera(id)`` 等方法暴露
能力。用 ``MethodGripper``、``MethodArm`` 和 ``MethodCamera`` 把这些方法包装成普通
部件，组合层就不必理解厂商 API。视图声明放在对应方法旁边，日后修改映射时也更容易找。

组合机器人
----------

给部件命名时，把它们当成公开 API 字段来考虑。这些名字会进入规范观测和动作路径；
策略和数据集开始依赖之后，就不要随意改动。

.. code-block:: python

   from rlinf.robotics import Group, Robot


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"


   robot = ExampleRobot(
       left=ExampleArm("tcp://left-arm:5000"),
       right=ExampleArm("tcp://right-arm:5000"),
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

在这套结构中，左臂本体的规范观测路径是 ``arms.left.state.joint_position``，动作
路径是 ``arms.left.arm``。末端执行器动作放在 ``arms.<name>.end_effector`` 下，
机器人级相机使用 ``cameras.<name>``，其他组件使用 ``parts.<name>``。统一遵循这些
约定，不同机器人暴露给策略的数据结构才容易对齐。

``Robot`` 知道哪些机械臂使用独立连接，因此可以并行重置、读取和下发命令。读取双臂
观测只占一个往返时间，机器人实现中也不用再手写并发逻辑。

在节点上放置部件
----------------

用 ``at()`` 记下部件应该运行在哪个节点，再把这条声明放到原本组合部件的位置。
``Robot.connect`` 会在目标节点上构建它，不需要额外协调一次放置调用。

.. code-block:: python

   from rlinf.robotics import Group, Robot

   robot = Robot(arm=ExampleArm.at("tcp://left-arm:5000", node_rank=0))
   robot.connect()

这段代码分两步工作：1）记录一条 ``ExampleArm`` 声明，目标是节点 0；2）机器人连接
时，再构建并连接这条机械臂。末端执行器和相机要显式组合，机械臂只包含交给它的部件。
启动时，``connect`` 把句柄挂到 ``robot.handles[<name>]``；结束时，``disconnect`` 回收
这些句柄。

所有部件都能这样声明，不只有机械臂。例如，相机可以留在实际插接的机器上::

   scene=RealSenseCamera.at(info, node_rank=2)

一条连接支撑多个组件时，只声明一次，再引用它暴露的部件。这样底层设备只会打开一次::

   hardware = ExampleHardware.at(node_rank=0)
   Group(arm=hardware.part("left"), gripper=hardware.part("left_end_effector"))

底层的 ``spawn()`` 会立刻放置部件。只有机器人之外的场景才直接调用它，比如调试
脚本；此时句柄的生命周期也要由脚本自己管理。

每种机器人不必再写一个 worker 类。RLinf 根据部件类自动合成，再由 ``WorkerGroup``
把公有方法绑定为 RPC。标准部件接口之外的方法仍能通过句柄访问，而且本地与远程的
调用形式一致::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

描述并构建机器人
----------------

把硬件连接和放置参数写进 ``RobotConfig`` 数据类，再在机器人类中编写构建方法，根据
这些字段组合出 ``Robot``。重置位姿、奖励和回合长度仍放在任务配置里；它们描述
机器人怎么用，不是硬件去哪里找。

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
       def build(cls, *, config: ExampleRobotConfig) -> "ExampleRobot":
           return cls(
               **{
                   side: ExampleArm.at(
                       endpoint,
                       node_rank=config.node_rank,
                       name=f"ExampleArm-{side}",
                   )
                   for side, endpoint in (
                       ("left", config.left_endpoint),
                       ("right", config.right_endpoint),
                   )
               }
           )

单臂型号可以沿用同一个 ``build()``，只返回一个条目。启动过程应当要么全部成功，要么
完整回滚：若后续部件失败，先断开此前放置的句柄，再向上抛出错误。否则，策略看到的
数据结构会随着启动顺序变化。

.. warning::

   ``build()`` 只组合声明，不会访问硬件。读取观测或发送命令前，先对返回的机器人调用
   ``connect()``；结束时再调用 ``disconnect()``。真机环境会在硬件初始化和清理流程中
   执行这些调用。

注册机器人
----------

配置、发现逻辑和构建函数写好后，直接在机器人自己的模块中注册。相关信息集中在一处，
新增机器人也不必修改中央注册表。

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

把这次调用放在模块末尾，等配置类和发现类都定义完成后再执行。注册会把机器人类、配置、
发现逻辑和 ``build`` 绑定在一起。之后调用 ``build_robot("ExampleRobot", ...)``，即可
按名称组合机器人，各个调用点无需直接导入具体类。

硬件结构大体相同时，优先继承现有机器人，复用构建逻辑。``DualFrankaRobot`` 继承
``FrankaRobot``，只改 ``build_arms`` 和 ``BACKEND``；构建入口和生命周期等通用行为
继续共用。

构造 ``Cluster`` 前先导入注册模块，硬件发现开始时才能识别这款机器人。RLinf 会把已
注册的硬件策略模块交给各节点探测，因此每个节点配置的 Python 环境都必须能导入它。

配置集群
--------

硬件信息沿用现有的 ``cluster.node_groups.hardware`` 结构。注册的配置类解析每一项，
构建函数再把配置变成机器人组合。endpoint 和 node rank 这类部署信息因此留在 YAML
中，不会写死在 Python 代码里。

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

重置、奖励、成功判定、截断和 Gymnasium 空间属于 ``RobotTask`` 或真机环境。通过
``RobotTaskEnv`` 把任务与 ``Robot`` 组合起来。现有策略若使用扁平动作向量和
``state``/``frames`` 观测，就在边界上接入 ``LegacyObservationAdapter`` 和
``VectorActionAdapter``，不要让硬件代码适配某个策略的数据格式。

.. warning::

   引入规范接口时，既有 Gym ID、动作维度、观测键、相机名称和数据集字段都不能变。
   用适配器和回归测试保证兼容，避免破坏已经训练好的策略和现有数据集。

测试集成
--------

大部分接入逻辑不需要厂商 SDK 或真实硬件就能测试。请覆盖部件接口、组合路径、句柄
生命周期、注册与硬件发现，以及旧策略依赖的完整数据结构。

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py \
     tests/unit_tests/test_robotics_boundaries.py \
     tests/unit_tests/test_robot_task_env.py \
     tests/unit_tests/test_realworld_robotics_compatibility.py

这条命令检查调度器边界、单臂与双臂组合、任务与机器人的边界，以及所有内置真机环境
暴露给策略的数据结构。整组测试都不依赖真实硬件。
