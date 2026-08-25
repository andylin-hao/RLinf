添加机器人
==========

本指南以新增移动底盘为例，先在本地验证底盘的观测与动作，再将其与 RLinf 已有的 Franka 机械臂组合为移动操作机器人，最后接入真机 Gymnasium 环境和集群配置。按照这一顺序排查问题，可以先确认设备接口，再处理组合和部署。

开始前，请阅读 :doc:`机器人模型 <../concepts/robotics>`。如果 RLinf 已经能够连接目标硬件，而变更仅涉及奖励、复位流程或成功条件，请参阅 :doc:`新增真机任务 <new_task>`。

1. 在本地实现移动底盘
----------------------

移动底盘是一个可控零部件：它报告自身位姿，并接收速度命令。实现时继承 ``MobileBase``，让类本身直接表明设备类别。厂商 SDK 应在 ``_open()`` 中导入；只导入该模块而不连接硬件的节点无需安装 SDK。

.. code-block:: python

   import numpy as np

   from rlinf.robotics import MobileBase


   @MobileBase.register("example")
   class ExampleMobileBase(MobileBase):
       def __init__(self, endpoint: str):
           self.endpoint = endpoint

       def _open(self):
           from example_mobile_sdk import Client

           return Client(self.endpoint)

       def _release(self, device) -> None:
           try:
               device.stop()
           finally:
               device.close()

       @property
       def observation_features(self) -> dict:
           return {"pose": {"shape": (3,), "dtype": "float32"}}

       @property
       def action_features(self) -> dict:
           return {"velocity": {"shape": (2,), "dtype": "float32"}}

       def reset(self) -> None:
           self._device.stop()

       def get_observation(self) -> dict[str, np.ndarray]:
           pose = np.asarray(self._device.get_pose(), dtype=np.float32)
           return {"pose": pose}

       def send_action(
           self, action: dict[str, np.ndarray]
       ) -> dict[str, np.ndarray]:
           if set(action) != {"velocity"}:
               raise KeyError("Expected only 'velocity'.")
           velocity = np.asarray(action["velocity"], dtype=np.float32)
           if velocity.shape != (2,):
               raise ValueError(f"Expected velocity shape (2,), got {velocity.shape}.")
           self._device.set_velocity(
               linear=float(velocity[0]),
               angular=float(velocity[1]),
           )
           return {"velocity": velocity}

装饰器将该 driver 注册为 ``example`` backend。已经持有具体 class 的代码仍可直接构造实例；从配置构建机器人时，则可通过 ``MobileBase.backend("example")`` 解析 driver。这个 registry 用于选择可互换的设备 driver，不代表完整机器人类型；后文会使用 ``register_type()`` 注册机器人。

这里约定 ``pose`` 为 ``[x, y, yaw]``，``velocity`` 为 ``[linear_velocity, angular_velocity]``。这些名称和单位会进入任务、policy 与数据集，因此应采用长期稳定的规范字段，而不是直接暴露厂商 SDK 的方法名。

与其他零部件组合前，先单独连接并控制底盘：

.. code-block:: python

   base = ExampleMobileBase("tcp://mobile-base:7000")
   base.connect()
   try:
       print(base.get_observation())
       base.send_action(
           {"velocity": np.array([0.1, 0.0], dtype=np.float32)}
       )
   finally:
       base.disconnect()

``_open()`` 的返回值会保存到 ``self._device``，断开时同一对象作为参数传给 ``_release(device)``。清理逻辑应停止并释放参数 ``device``，不要重新从 ``self`` 读取。基类已经保证 ``connect()`` 和 ``disconnect()`` 可重复调用；如果设备需要自定义生命周期，也应保留这一性质。

.. warning::

   Python 进程失去连接后，移动底盘仍可能继续运动。硬件控制器必须配置命令超时和速度限制；``_release()`` 中的停止命令只能作为最后一道软件保护。

2. 与现有机械臂组合
--------------------

移动底盘不应附带另一套机械臂驱动。可以直接复用现有 Franka connection 提供的零部件，将底盘、机械臂和末端执行器组合为一个移动操作机器人：

.. code-block:: python

   from rlinf.robotics import FrankaRobot, Robot


   class MobileManipulator(Robot):
       ROBOT_TYPE = "MobileManipulator"


   base = ExampleMobileBase(
       "tcp://mobile-base:7000",
       node_rank=0,
       worker_name="ExampleMobileBase-0-0",
   )
   arm_connection = FrankaRobot.declare_arm(
       "10.0.0.2",
       node_rank=0,
       name="FrankaArm-0-0",
   )
   robot = MobileManipulator(
       base=base,
       arm=arm_connection,
   )

底盘和 Franka 机械臂都是尚未连接的 ``RobotPart``，因此组合方式相同：直接传给 ``Robot``，由参数名决定访问路径。``base=base`` 建立 ``base`` 路径，``arm=arm_connection`` 建立 ``arm`` 路径。

两者的区别在于是否还承载其他零部件。底盘没有直接挂载的零部件；Franka 机械臂则通过同一个硬件 session 承载末端执行器。因此，直接组合机械臂时，末端执行器会自动出现在 ``arm.end_effector``，机器人 builder 无需再维护一份末端执行器清单。

只有当共享 session 本身不能返回观测时，才需要调用 ``part(name)``。例如，双臂控制器可以通过 ``session.part("left")`` 取出左臂，再将返回的 ``RobotPart`` 加入机器人的组合结构。

``PartGroup`` 会在组合阶段检查这一边界。构造函数只接受 ``RobotPart`` 或另一个 ``PartGroup``；如果传入本身不能返回观测的裸 ``Connection``，异常会直接指出出错的参数名。

如果沿用 Franka 的标准零部件名称，可以直接复用已有的 ``build_arms``：

.. code-block:: python

   arm_parts = FrankaRobot.build_arms(
       robot_ip="10.0.0.2",
       node_rank=0,
       worker_rank=0,
       env_idx=0,
   )
   robot = MobileManipulator(base=base, **arm_parts)

将 ``FrankaRobot.build_arms`` 替换为其他机器人系列提供的零部件构建方法，或者使用 ``PartGroup`` 组合多条机械臂，都不需要修改移动底盘。机器人由所选零部件及其名称构成，无需为移动操作机器人增加专用字段或基类。

打开硬件前，可以检查零部件路径、部署节点和连接归属：

.. code-block:: text

   >>> print(robot.describe())
   MobileManipulator
   ├── base                ExampleMobileBase    node=0     via ExampleMobileBase#1
   └── arm                 FrankaROSArm         node=0     via FrankaROSArm#2
       └── end_effector    MethodEndEffector    node=0     via FrankaROSArm#2

``arm`` 与 ``arm.end_effector`` 的 ``via`` 相同，表示二者共用一条 Franka connection；底盘使用另一条独立 connection。连接后，零部件路径、节点和资源归属保持不变；如果 connection 位于远程节点，class 名称会显示为 RemoteFrankaROSArm 等合成类型。

连接后，观测和动作按照组合时定义的名称访问：

.. code-block:: python

   robot.connect()
   try:
       observation = robot.get_observation()
       base_pose = observation["base"]["pose"]
       arm_pose = observation["arm"]["tcp_pose"]

       # 只发送底盘动作。
       robot.send_action(
           {"base": {"velocity": np.array([0.1, 0.0], dtype=np.float32)}}
       )

       # 任务也可以同时控制底盘和现有机械臂。
       robot.send_action(
           {
               "base": {"velocity": base_velocity},
               "arm": {
                   "tcp_pose": arm_target,
                   "end_effector": {"target": gripper_target},
               },
           }
       )
   finally:
       robot.disconnect()

``PartGroup.send_action`` 接受只包含部分路径的动作字典，因此导航任务只需发送 ``base`` 动作，无需为机械臂补充保持当前位置的命令。动作同时包含底盘和机械臂时，RLinf 可以并行调用两条独立连接；机械臂与末端执行器共用连接，仍会按照声明顺序调用。

3. 在真机环境中使用组合机器人
------------------------------

硬件代码定义底盘如何运动，任务代码则定义目标位置、成功条件以及 policy 实际控制的零部件。下面的 ``RobotTask`` 只向 policy 提供底盘观测和动作；同一机器人中已经组合的机械臂保持空闲：

.. code-block:: python

   import gymnasium as gym

   from rlinf.envs.real import RobotTask, RobotTaskEnv


   class DriveToTarget(RobotTask):
       def __init__(self, target_xy: np.ndarray):
           self.target_xy = np.asarray(target_xy, dtype=np.float32)

       @property
       def description(self) -> str:
           return "drive the mobile manipulator to the target"

       @property
       def observation_space(self) -> gym.Space:
           return gym.spaces.Dict(
               {
                   "base": gym.spaces.Dict(
                       {
                           "pose": gym.spaces.Box(
                               -np.inf, np.inf, shape=(3,), dtype=np.float32
                           )
                       }
                   )
               }
           )

       @property
       def action_space(self) -> gym.Space:
           return gym.spaces.Dict(
               {
                   "base": gym.spaces.Dict(
                       {
                           "velocity": gym.spaces.Box(
                               low=np.array([-0.5, -1.0], dtype=np.float32),
                               high=np.array([0.5, 1.0], dtype=np.float32),
                           )
                       }
                   )
               }
           )

       @staticmethod
       def observe(robot: Robot) -> dict:
           return {"base": robot.get_observation()["base"]}

       def reset(self, robot: Robot, *, seed=None, options=None):
           del seed, options
           robot.reset()
           return self.observe(robot), {}

       def step(self, robot: Robot, action: dict):
           robot.send_action(action)
           observation = self.observe(robot)
           distance = float(
               np.linalg.norm(observation["base"]["pose"][:2] - self.target_xy)
           )
           reached = distance < 0.05
           return observation, float(reached), reached, False, {"distance": distance}


   env = RobotTaskEnv(robot, DriveToTarget(np.array([1.0, 0.0])))
   try:
       observation, info = env.reset()
       observation, reward, terminated, truncated, info = env.step(
           {"base": {"velocity": np.array([0.1, 0.0], dtype=np.float32)}}
       )
   finally:
       env.close()

创建 ``RobotTaskEnv`` 时会连接整个组合机器人，``close()`` 则负责断开。移动操作任务可以在 observation space、action space 和动作字典中加入 ``arm`` 与 ``arm.end_effector`` 路径，无需修改底盘 driver 或机器人组合。

如需通过 RLinf 分布式 ``RealWorldEnv`` 启动该任务，应先注册 Gymnasium ID，并在 env YAML 中设置 ``env_type: real`` 和对应 ID。当前 rollout 接口使用面向 policy 的 ``state`` 与 ``frames`` 观测；已有 policy 采用该表示时，请在环境边界配置 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter``。任务注册、YAML、wrapper 与兼容性检查请参阅 :doc:`新增真机任务 <new_task>`。

4. 将同一组合部署到硬件节点
----------------------------

placement 只决定各条连接在哪个节点打开，不改变任务访问零部件的路径。例如，可以将底盘控制器放在节点 0，将 Franka 控制器放在节点 1：

.. code-block:: python

   base = ExampleMobileBase(
       "tcp://mobile-base:7000",
       node_rank=0,
       worker_name="ExampleMobileBase-0-0",
   )
   arm_connection = FrankaRobot.declare_arm(
       "10.0.0.2",
       node_rank=1,
       name="FrankaArm-0-0",
   )
   robot = MobileManipulator(
       base=base,
       arm=arm_connection,
   )

构造这些对象时只会记录参数、``node_rank`` 和 ``worker_name``，不会导入厂商 SDK 或打开设备。``Connection`` 的 metaclass 会先取走 placement 参数，再调用驱动自身的 ``__init__``，因此新驱动的构造函数只需声明硬件相关参数。``robot.connect()`` 会在每条连接指定的节点上将其打开一次。跨节点的连接会在目标节点重新构造，机器人组合中的原对象则切换为对应的 view。机器人仍持有相同的零部件对象，调用代码无需区分部署位置。

如果后续 connection 打开失败，机器人会关闭此前已经成功打开的 connection。driver 如果在 ``_open()`` 内部获取部分资源后抛出异常，仍需自行完成清理；由于该 connection 尚未完成连接，机器人无法代为回滚。

5. 通过配置构建组合机器人
-------------------------

调试脚本中的组合确认无误后，再使用 ``RobotConfig`` dataclass 描述硬件输入，并实现 ``build()``。机械臂继续复用已有声明，不应复制其 SDK 或生命周期代码：

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import MobileBase, RobotConfig


   @dataclass
   class MobileManipulatorConfig(RobotConfig):
       base_backend: str = "example"
       base_endpoint: str | None = None
       arm_ip: str | None = None
       controller_node_rank: int | None = None


   class MobileManipulator(Robot):
       ROBOT_TYPE = "MobileManipulator"

       @classmethod
       def build(
           cls,
           *,
           base_backend: str = "example",
           base_endpoint: str | None,
           arm_ip: str | None,
           node_rank: int,
           controller_node_rank: int | None = None,
           worker_rank: int = 0,
           env_idx: int = 0,
       ) -> "MobileManipulator":
           if not base_endpoint:
               raise ValueError("MobileManipulator requires base_endpoint.")
           base_cls = MobileBase.backend(base_backend)
           base = base_cls(
               base_endpoint,
               node_rank=node_rank,
               worker_name=f"{base_cls.__name__}-{worker_rank}-{env_idx}",
           )
           arm_node_rank = (
               node_rank
               if controller_node_rank is None
               else controller_node_rank
           )
           arm_connection = FrankaRobot.declare_arm(
               arm_ip,
               node_rank=arm_node_rank,
               name=f"FrankaArm-{worker_rank}-{env_idx}",
           )
           return cls(
               base=base,
               arm=arm_connection,
           )

``MobileBase.backend()`` 根据硬件配置中的名称找到对应 driver。``build()`` 随后组合尚未连接的零部件，以及从共享连接中选出的零部件，不会打开任何设备。连接地址、backend 选择和 placement 写入机器人配置；目标位置、奖励、复位姿态与 episode 长度仍属于任务配置。

``build()`` 应保留明确的参数签名。``Robot.of_type()`` 和 ``build_robot()`` 会将关键字参数直接传给 ``build()``；注册过程不会自动展开 ``RobotConfig`` 实例，也不应丢弃 builder 无法识别的字段。如果配置中出现未支持的 key，应在这一边界直接报错，而不是由 ``**kwargs`` 吸收后静默忽略。

.. warning::

   读取观测或发送命令前必须调用 ``connect()``，清理阶段必须调用 ``disconnect()``。由 ``RobotTaskEnv`` 持有机器人时，这两个生命周期操作分别在环境创建和 ``close()`` 中完成。

6. 注册机器人类型
-----------------

大多数机器人无需单独实现 discovery class。在模块末尾注册机器人及其配置即可；``register_type()`` 会创建标准 discovery class，并关联当前机器人的 ``build()``：

.. code-block:: python

   MobileManipulator.register_type(MobileManipulatorConfig)

标准 discovery 流程会筛选属于当前节点的配置，通过同名大写环境变量补全未设置字段，并为每项配置返回一条硬件记录。如果配置包含相机字段，该流程还会复用公共的相机发现与校验逻辑。只有机器人的枚举方式确实不同时，才需要将自定义 ``RobotDiscovery`` 子类作为第二个参数传给 ``register_type()``。

``Connection.register()`` 与 ``Robot.register_type()`` 对应两个不同的 registry：前者注册单个设备 driver，后者注册整台机器人的组合。完成机器人类型注册后，调用方既可以使用 ``Robot.of_type("MobileManipulator", ...)``，也可以调用便捷函数 ``build_robot("MobileManipulator", ...)``。两种方式都需要提供 builder 声明的参数；注册操作不会自动将硬件配置转换为这些参数。

项目内置实现应放在 ``rlinf/robotics/robots/`` 下，并由该目录的 ``__init__.py`` 导入。这样，无论构造 ``Cluster`` 还是运行检查脚本，导入 ``rlinf.robotics.robots`` 时都会先完成注册。项目外部的集成则需在自己的 entry point 中显式导入注册模块。node probe 也会导入已注册的机器人模块，因此每个节点配置的 Python 环境都必须能够导入该模块。

7. 配置集群
-----------

将物理硬件信息写入 ``cluster.node_groups.hardware``。每项配置都会由已注册的 config class 解析，并在硬件 discovery 过程中生成 ``RobotInfo``。连接地址和 ``node_rank`` 等部署参数应保留在 YAML 中，不应硬编码到 Python 代码中：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement: {}
     node_groups:
       - label: mobile_manipulator
         node_ranks: 0
         hardware:
           type: MobileManipulator
           configs:
             - node_rank: 0
               base_backend: example
               base_endpoint: tcp://mobile-base:7000
               arm_ip: 10.0.0.2
               controller_node_rank: 1
       - label: arm_controller
         node_ranks: 1

这里的 ``node_rank`` 表示配置中的机器人资源由哪个节点持有，``controller_node_rank`` 则将复用的 Franka connection 部署到对应控制节点。env 配置另行选择 Gym ID，因此同一套硬件组合可以服务于导航、移动操作或数据采集任务。

env 收到 ``RobotInfo`` 后，需要显式调用已注册的 builder。scheduler 提供的 env worker rank 等运行时信息也在这一边界加入：

.. code-block:: python

   hardware = robot_info.config
   robot = build_robot(
       "MobileManipulator",
       base_backend=hardware.base_backend,
       base_endpoint=hardware.base_endpoint,
       arm_ip=hardware.arm_ip,
       node_rank=hardware.node_rank,
       controller_node_rank=hardware.controller_node_rank,
       worker_rank=worker_info.rank,
       env_idx=env_idx,
   )

显式保留这次调用，可以避免硬件 registry 隐式转换不同层的配置结构。如果多个 env 共用同一种机器人，应将这段转换逻辑放入公共的硬件初始化代码，而不是复制到每个任务中。

8. 测试集成
-----------

运行 contract 前，应先在 ``tests/robot_mocks/`` 中为示例 SDK 添加一个 fake，并将其加入 ``sdk_modules()``。这个 fake 只需实现上文实际调用的 ``get_pose()``、``set_velocity()``、``stop()`` 和 ``close()``。统一注册后，单元测试、检查脚本和远程 mock worker 会使用同一份 fake。

随后在 ``tests/unit_tests/`` 中新增测试，并使用 ``PartContract`` 和 ``RobotContract`` 检查新底盘及组合机器人：

.. code-block:: python

   from robot_contracts import PartContract, RobotContract
   from robot_mocks import mocked_sdks


   def test_mobile_base_conforms():
       assert MobileBase.backend("example") is ExampleMobileBase
       with mocked_sdks():
           PartContract(
               lambda: ExampleMobileBase("tcp://mobile-base:7000"),
               action={"velocity": np.zeros(2, dtype=np.float32)},
           ).assert_kept()


   def test_mobile_manipulator_conforms():
       with mocked_sdks():
           RobotContract(
               lambda: MobileManipulator.build(
                   base_endpoint="tcp://mobile-base:7000",
                   arm_ip="10.0.0.2",
                   node_rank=0,
                   controller_node_rank=0,
               )
           ).assert_kept()

这些 contract 位于 ``tests/robot_contracts``，mock SDK 位于 ``tests/robot_mocks``。二者仅用于测试，不会随 ``rlinf`` 包发布。

这些 contract 会重复执行连接和断开流程。``PartContract`` 检查单个零部件的观测字段及 shape；提供速度样例后，还会验证该动作可以执行，并拒绝未知动作字段。``RobotContract`` 检查机器人能否在连接前输出结构说明，验证顶层组合中的叶子节点，并在 ``connect()`` 中注入失败以检查回滚行为。

目前，``RobotContract`` 会将承载其他零部件的 ``RobotPart`` 视为一个叶子节点，不会继续逐项检查其 ``children``。因此，新增机器人时还应直接检查本次引入的路径和 owner：

.. code-block:: python

   robot = MobileManipulator.build(
       base_endpoint="tcp://mobile-base:7000",
       arm_ip="10.0.0.2",
       node_rank=0,
       controller_node_rank=0,
   )
   assert set(robot.named_parts) == {"base", "arm", "arm.end_effector"}
   end_effector = robot.child("arm").child("end_effector")
   assert end_effector.owner is robot.child("arm").owner
   assert len(robot.owners()) == 2

只有当新增 SDK session 同时支持多个零部件时，才需要增加 ``ConnectionContract``。该 contract 会检查 session 生命周期，以及 ``parts`` 中各零部件的观测。还应逐项断言 ``connection.part(name).owner is connection``；目前 contract 不会自行调用 ``part(name)`` 检查 owner 绑定。本例新增的是单个叶子 ``MobileBase``，并复用已经测试过的 Franka connection，因此无需为底盘增加共享 session 测试。

这些检查覆盖了项目中实际出现过的故障。contract 检查失败时，错误信息会列出具体原因，例如：

.. code-block:: text

   ConformanceError: ExampleMobileBase does not keep 2:
     - ExampleMobileBase: reconnecting raised RuntimeError: threads can only be started
       once; stall recovery closes a connection and opens it again
     - ExampleMobileBase observes pose with shape (4,), declares (3,)

测试还应覆盖组合路径、connection 生命周期、注册、硬件发现，以及现有 policy 依赖的数据结构：

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py tests/unit_tests/test_conformance.py \
       tests/unit_tests/test_real_env.py

上述测试不依赖真实硬件。

使用 mock SDK 验证
~~~~~~~~~~~~~~~~~~

设备 SDK 只在 ``_open()`` 中导入。``tests/robot_mocks`` 中的 mock SDK 允许真实的零部件类在没有硬件和厂商 SDK 的机器上运行。

首先检查机器人的组合结构和生命周期：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MobileManipulator --mock \
       --arg base_endpoint=tcp://mobile-base:7000 \
       --arg arm_ip=10.0.0.2 --arg node_rank=0 --arg controller_node_rank=0

脚本会列出零部件路径、共享连接和部署节点，然后读取观测并断开。未声明的观测、错误的 shape、误将裸 ``Connection`` 加入机器人组合，以及断开后仍报告已连接的资源，都会导致检查失败。

环境根据零部件的声明创建 observation space，因此观测 shape 必须与声明一致。

添加 ``--remote`` 后，mock 测试会保留各 connection 声明的 ``node_rank``。声明了节点的 connection 会部署到 scheduler worker 中；没有 ``node_rank`` 的 connection 仍在当前进程打开。该模式可进一步发现 worker 方法名称冲突和状态无法跨进程传递等问题。

还可以使用 mock 配置运行完整训练。``run.sh`` 会先安装相应的 mock SDK；Turtle2 是现有实现中最接近移动操作机器人的示例，可以作为新增配置的参考：

.. code-block:: bash

   bash tests/e2e_tests/embodied/run.sh realworld_xsquare_turtle2_mock_sac_cnn

每种内置机器人均提供对应配置：

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

连接真机
~~~~~~~~

mock 测试通过后，再检查真机的时序、标定和 SDK 行为。确认机器人已上电且网络可达后，移除 ``--mock`` 并运行同一命令：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MobileManipulator \
       --arg base_endpoint=tcp://mobile-base:7000 \
       --arg arm_ip=10.0.0.2 --arg node_rank=0 --arg controller_node_rank=1
