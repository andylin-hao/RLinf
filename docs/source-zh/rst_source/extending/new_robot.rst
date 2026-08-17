添加机器人
==========

接入真实机器人时，我们先把硬件拆成部件，再将它们组合为 ``Robot``。这篇指南会从一个
部件开始，依次完成机器人构建、注册和集群配置。设备 SDK 只在部件中加载，任务逻辑留给
``RobotTask``，节点位置则写进部件声明。

如果还不熟悉部件模型，建议先读 :doc:`机器人模型 <../concepts/robotics>`。后文会反复
用到其中三点：每个物理组件都是 ``RobotPart``；一条连接可以通过 ``parts`` 暴露多个
组件；``Robot`` 为整棵树确定稳定的名字。概念页还列出了 ``rlinf/robotics`` 的代码
位置，并说明 ``spawn()`` 如何把部件放到其他节点。

实现部件
--------

先挑一台设备，从最窄的接口写起。只采集观测的设备继承 ``RobotPart``；还要接收命令，
则继承 ``ControllablePart``。

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

       def _release(self) -> None:
           self._device.close()

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

``_open`` 返回什么，就能通过 ``self._device`` 拿到什么，``is_connected`` 也据此判断。
在 ``_open`` 而不是 ``__init__`` 里打开硬件，部件才能在一台机器上声明、到另一台机器上
构建。如果某个部件的生命周期不止“打开一个设备”——比如机械臂上电后必须先回零——可以
改为重写 ``connect`` 和 ``disconnect``。

设备若符合 ``Camera``、``EndEffector``、``MobileBase`` 或 ``LeggedBase`` 的接口，
就继承对应的具体类型。后续组合部件或创建远程代理时，代码仍能识别它的设备类别。

在一条连接上暴露多个组件
------------------------

接着处理硬件连接。一个 socket、一条 CAN 总线或一个 ROS 节点可能同时驱动多个物理
组件，这时通过 ``parts`` 把它们逐项列出。底层连接只打开一次，调用方仍可分别访问每个
组件。对于机械臂，约定把部件自身放在 ``"arm"`` 条目下。

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

有些 SDK 用 ``open_gripper``、``move_left_arm``、``get_camera(id)`` 等方法暴露硬件
能力。此时用 ``MethodGripper``、``MethodArm`` 或 ``MethodCamera`` 包一层视图，并把
视图写在所适配的方法旁边。到了机器人组合这一步，看到的便是普通部件，而不是厂商 API
中的方法名。

组合机器人
----------

机器人就是一组具名部件，没有别的东西。直接构造的话，全部内容就是这些：

.. code-block:: python

   from rlinf.robotics import Robot


   class Bench(Robot):
       ROBOT_TYPE = "Bench"


   robot = Bench(arm=ExampleArm.at("10.0.0.2", node_rank=1))
   robot.connect()

这里没有硬件配置，也没有 discovery。它们的存在是为了仅凭类型名就能组装出机器人——
配置文件需要这一点，脚本不需要。本页后面的内容才会用到它们。


部件准备好后，就可以开始组装机器人。命名时请把每个名字当成公开 API 字段，因为它会
进入规范观测和动作路径；一旦策略和数据集开始使用这些路径，再改名就等于修改数据结构。

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

按这套结构，左臂本体的规范观测路径是 ``arms.left.state.joint_position``，动作路径是
``arms.left.arm``。末端执行器动作位于 ``arms.<name>.end_effector``，机器人级相机位于
``cameras.<name>``，其他组件则位于 ``parts.<name>``。环境、策略和数据集应当共用这些
名字。

``Robot`` 会并行重置、读取和控制使用独立连接的机械臂。因此，读取双臂观测只需等待
一次往返；机器人子类中无需另写并发代码。

在节点上放置部件
----------------

现在再决定每个部件运行在哪个节点。调用 ``at()`` 写入节点，把返回的声明放到原本应该
放本地部件的位置；之后 ``Robot.connect`` 会在正常连接流程中把它构建到目标节点上。

.. code-block:: python

   from rlinf.robotics import Group, Robot

   robot = Robot(arm=ExampleArm.at("tcp://left-arm:5000", node_rank=0))
   robot.connect()

这里的 ``at()`` 只记录一条目标为节点 0 的 ``ExampleArm`` 声明。直到 ``connect``
执行，机械臂才真正构建并连接，所得句柄挂在 ``robot.handles[<name>]`` 下；
``disconnect`` 会在退出时回收它。这个示例中的机械臂只包含显式传入的部件，因此末端
执行器和相机也要显式组合。

这种声明并不只适用于机械臂。例如，相机可以留在实际插接的机器上::

   scene=RealSenseCamera.at(info, node_rank=2)

一条连接支撑多个组件时，只声明一次连接，再从中选取暴露的部件::

   hardware = ExampleHardware.at(node_rank=0)
   Group(arm=hardware.part("left"), gripper=hardware.part("left_end_effector"))

更底层的 ``spawn()`` 会立即放置部件。它适合机器人之外的场景，例如调试脚本；脚本
此时也要自行管理句柄的生命周期。

部件不需要配套手写 worker 类。放置代码根据部件类自动合成 worker，再由
``WorkerGroup`` 把公有方法绑定为 RPC。标准部件接口之外的方法仍可通过句柄调用，
本地与远程使用同一种写法::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

描述并构建机器人
----------------

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

单臂型号也可以沿用这种构建方式，只返回一个条目。启动必须要么全部成功，要么完整
回滚：后续部件失败时，先断开已经放置的句柄，再向上抛出错误。若返回只启动了一部分的
机器人，调用方看到的数据结构会取决于故障发生的先后顺序。

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

   引入规范接口时，既有 Gym ID、动作维度、观测键、相机名称和数据集字段都不能变。
   用适配器和回归测试保证兼容，避免破坏已经训练好的策略和现有数据集。

测试集成
--------

大部分接入逻辑在厂商 SDK 和真实机器人到位之前就能测试。至少覆盖部件接口、组合路径、
句柄生命周期、注册与硬件发现，以及旧策略依赖的完整数据结构：

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py tests/unit_tests/test_real_env.py

这组测试覆盖调度器导入边界、单臂与双臂组合、任务与机器人的分界，以及所有内置真机
环境暴露给策略的数据结构；运行时不需要真实硬件。

剩下的部分必须有机器人才能验证。等机器人上电、网络可达之后，同一条路径可以直接跑在
它上面：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MyRobot \
       --arg robot_ip=10.0.0.1 --arg node_rank=1

它会列出机器人由哪些部件组成、每个部件挂在哪条连接上、被放到了哪个节点，然后逐个读取
观测并断开。以下情况会判定失败：某个部件返回了它没有声明过的观测；连接本身出现在部件
树里；断开之后仍有东西声称自己是连着的。这些错误用假部件复现不出来，只有真设备接在
另一端时才会显形。
