新增真机任务
============

当 RLinf 已支持目标真机，而变更仅涉及任务目标、奖励或复位方式时，应新增真机任务，无需修改机器人实现。一个真机任务包含任务配置、env 类、Gymnasium ID 和对应的 YAML 配置，以下步骤将依次定义并验证这些内容。

本页所述真机任务专指 ``rlinf/envs/real`` 下的任务模块。如需为模拟器或 benchmark 新增 task，请参阅 :doc:`new_env`。

如果硬件本身尚未接入，请先按照 :doc:`new_robot` 实现零部件，并确认观测和动作能够正常传递，再添加任务。

实施步骤
--------

以下示例为 Franka 添加 ``WipeEnv-v1``。后续步骤会直接引用前面定义的配置类和 Gymnasium ID，因此应按顺序完成。

1. 定义任务配置
~~~~~~~~~~~~~~~

在 ``rlinf/envs/real/<robot>/<task>.py`` 中新建模块，并继承该机器人的配置 dataclass。配置类只需添加擦拭任务所需的字段：

.. code-block:: python

   from dataclasses import dataclass, field

   import numpy as np

   from .base import FrankaEnv, FrankaRobotConfig, compliance


   @dataclass
   class WipeConfig(FrankaRobotConfig):
       task_description: str = "wipe the surface"
       target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
       reward_threshold: np.ndarray = field(
           default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.2, 0.2, 0.2])
       )
       random_xy_range: float = 0.03

       def __post_init__(self):
           self.compliance_param = compliance(
               translational_stiffness=800,   # 降低刚度以保持接触
               translational_clip_z=0.02,
           )
           self.target_ee_pose = np.array(self.target_ee_pose)
           self.action_scale = np.array([0.02, 0.1, 1])

阻抗参数只需声明与默认值不同的部分。``compliance()`` 会将其合并到 ``COMPLIANCE_DEFAULTS``；字段名错误或控制器不支持相应参数时，该函数会立即报错。

2. 定义 env 类
~~~~~~~~~~~~~~

env 类首先指定前一步定义的配置类型。对于多数任务，仅需以下定义：

.. code-block:: python

   class WipeEnv(FrankaEnv):
       CONFIG_CLS = WipeConfig

仅当任务行为确有差异时才覆盖相应方法。最常见的是 ``go_to_rest``：插销任务在返回初始位姿前需要先抬高末端，避免插销卡在插孔中。

.. code-block:: python

       def go_to_rest(self, joint_reset=False):
           reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
           reset_pose[2] += 0.05
           self._interpolate_move(reset_pose, timeout=1)
           super().go_to_rest(joint_reset)

如果任务沿用原有动作空间，则无需实现 ``action_parts``。如果任务修改动作空间，则必须声明每段动作对应的零部件及其语义；遥操作根据这些语义匹配设备，而不能只根据维度判断。

.. code-block:: python

       def action_parts(self):
           return (
               ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),
               ActionPart("end_effector", 1, ActionKind.GRIPPER),
           )

各段宽度之和必须与动作空间维度一致，否则系统会在构建阶段报错。

3. 注册任务
~~~~~~~~~~~~

在 ``rlinf/envs/real/<robot>/__init__.py`` 的 ``TASKS`` 映射中加入 env 类：

.. code-block:: python

   from .wipe import WipeEnv

   TASKS = {
       ...
       "WipeEnv-v1": WipeEnv,
   }

``register_tasks`` 根据该映射生成 entry point，并将其注册到 Gymnasium。wrapper 由 env 自行声明，无需在此重复配置。Gym ID 会写入用户配置和数据集元数据，因此数据采集开始后不应再修改 ID。

4. 添加环境配置
~~~~~~~~~~~~~~~

在 ``examples/embodiment/config/env/`` 下新增 YAML 文件。``init_params.id`` 使用刚注册的 ID，任务字段写入 ``override_cfg``：

.. code-block:: yaml

   env_type: real
   init_params:
     id: "WipeEnv-v1"      # 上一步注册的 gym id
     num_envs: null
   teleop: spacemouse
   override_cfg:
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
     random_xy_range: 0.03

5. 验证注册结果
~~~~~~~~~~~~~~~

连接硬件前，先确认 ID 已完成注册且 entry point 可以解析：

.. code-block:: python

   from rlinf.envs.real import RealWorldEnv  # 触发全部任务注册
   from gymnasium.envs.registration import registry

   assert "WipeEnv-v1" in registry

``tests/unit_tests/test_real_env.py`` 会检查所有内置任务。请将新 ID 加入 ``EXPECTED_IDS``。

复用现有基础设施
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 需求
     - 现有实现
   * - 连接硬件、部署零部件
     - ``Robot.connect``，见 :doc:`../concepts/robotics`。
   * - 遥操作
     - 环境配置中的 ``teleop`` 选择设备，wrapper 栈负责组装。
   * - 手动标记奖励、手动结束 episode
     - 环境配置里的 ``keyboard_reward_wrapper``。
   * - 相对坐标系、欧拉角转换、夹爪维度裁剪
     - ``real/wrappers/transforms/``，由 wrapper 栈加载。
   * - 各任务共用的阻抗参数
     - ``COMPLIANCE_DEFAULTS``，任务只需声明差异项。

新增遥操作设备
--------------

一个遥操作设备就是 ``rlinf/robotics/parts/teleop/`` 下的一个类。它回答三个问题：如何连接硬件、操作者正在做什么、机器人应当如何响应。

.. code-block:: python

   @TeleopDevice.register("pedal")
   class Pedal(TeleopDevice):
       PRODUCES = {"end_effector": ActionKind.GRIPPER}

       def __init__(self, port: str) -> None:
           self._port = port

       def _open(self):
           from rlinf.robotics.parts.teleop.pedal import PedalReader

           return PedalReader(port=self._port)

       @property
       def observation_features(self):
           return {"pressed": {"shape": (1,), "dtype": "bool"}}

       def get_observation(self):
           return {"pressed": np.asarray([self._device.is_pressed()])}

       def action(self, reading, context):
           pressed = bool(reading["pressed"])
           return TeleopAction(
               parts={"end_effector": np.array([-1.0 if pressed else 1.0])},
               driving=pressed,
           )

``__init__`` 只保存端口等构造参数；``_open`` 在连接阶段打开硬件，返回值保存在 ``self._device``。因此设备可以在当前进程声明，再由目标节点创建。

``PRODUCES`` 同时声明目标零部件的名称和动作语义，因此 twist 不会被错误地接到关节角动作上。``action`` 用一个 ``TeleopAction`` 一次性给出全部结果：``driving`` 与动作在同一个返回值中，避免为判断接管状态而再次读取设备或保存中间状态。

``register`` 中的名称就是配置里书写的名称。配置到构造参数的转换由 ``from_config`` 完成，其默认实现直接把设备自身的选项传给构造函数，因此上面的例子无需编写这一部分。若要读取更外层的 env 配置，或根据被驱动的机器人调整行为，可以覆盖它：

.. code-block:: python

   @classmethod
   def from_config(cls, cfg, options, facts):
       port = options.get("port") or cfg.get("pedal_port")
       if port is None:
           raise ValueError("teleop device 'pedal' requires a port")
       return TeleopEntry(cls(port=port), drives=options.get("drives"))

最后把 ``pedal`` 加入对应 env 的 ``TELEOP`` 元组，声明该 env 能够表示这种设备产生的动作。这一步不会重复注册设备，公共 builder 会通过 ``TeleopDevice`` 查找该名称。如果机器人不包含 ``end_effector``，系统会在构建阶段报错。

如果同一套硬件需要第二种映射方式，例如输出关节角而非笛卡尔量，继承已有设备并覆盖 ``action`` 即可，``GelloJoint`` 与 ``Gello`` 就是这样的关系。

新增 wrapper
------------

如果新逻辑改变的是 rollout 周边行为，应新增 wrapper，而不是任务：动作接管放入 ``teleop/``，表示转换放入 ``transforms/``，rollout 起止和评分放入 ``episode/``。遥操作设备仍按上述方式实现为一个设备类；新的键盘模式应继承 ``KeyboardSession``。各目录的职责见 :doc:`../concepts/realworld_envs`。
