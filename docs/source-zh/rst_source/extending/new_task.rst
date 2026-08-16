新增任务
========

在 RLinf 已支持的硬件上新增任务，需要改三处：写一份配置 dataclass，定义一个 env 类，
再向任务表加一行。任务里说明目标、阻抗参数、成功条件和复位规则；机器人怎样构建与
放置，外层套哪些 wrapper，都沿用现有逻辑。

如果连机器人也是新的，先按 :doc:`new_robot` 接入硬件，再来定义任务。

步骤
----

1. 写配置
~~~~~~~~~

在 ``rlinf/envs/real/<robot>/<task>.py`` 新建模块，与这台机器人的其他任务放在一起。
从机器人配置 dataclass 继承，再补上当前任务所需的字段：

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
               translational_stiffness=800,   # 放软一些，保持接触
               translational_clip_z=0.02,
           )
           self.target_ee_pose = np.array(self.target_ee_pose)
           self.action_scale = np.array([0.02, 0.1, 1])

阻抗参数只写与默认值不同的项。``compliance()`` 会把它们合并到
``COMPLIANCE_DEFAULTS`` 上，遇到控制器不接受的参数就直接抛错。参数名一旦拼错，会在
这里停下来，不会传到阻抗控制器后被静默忽略。

2. 写 env 类
~~~~~~~~~~~~

接下来把 env 类的配置类型设为刚才的 dataclass。对很多任务来说，整个类只有这两行：

.. code-block:: python

   class WipeEnv(FrankaEnv):
       CONFIG_CLS = WipeConfig

只有任务行为确实不同时，才覆盖相应钩子。最常见的是 ``go_to_rest``，因为回原位的路径
取决于任务结束时的位姿。比如插销任务要先抬离插孔，否则销子会在上升途中卡住。

.. code-block:: python

       def go_to_rest(self, joint_reset=False):
           reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
           reset_pose[2] += 0.05
           self._interpolate_move(reset_pose, timeout=1)
           super().go_to_rest(joint_reset)

3. 注册
~~~~~~~

到 ``rlinf/envs/real/<robot>/__init__.py`` 的 ``TASKS`` 表里加一行，写明 env 类：

.. code-block:: python

   from .wipe import WipeEnv

   TASKS = {
       ...
       "WipeEnv-v1": WipeEnv,
   }

``register_tasks`` 会生成 entry point 并把这个 id 注册到 Gymnasium。这里不写 wrapper
栈：env 在上面已经声明过，``build_stack`` 读的就是那份声明。

用户配置和数据集元数据都会保存 gym id。后续改名会让这些引用失效，采集数据前先把名字
定下来。

4. 加环境配置
~~~~~~~~~~~~~

在 ``examples/embodiment/config/env/`` 下新增 YAML。注册好的 id 按下面的路径填写，任务
字段放进 ``override_cfg``；这些字段来自前面的配置 dataclass：

.. code-block:: yaml

   env_type: realworld
   init_params:
     id: "WipeEnv-v1"      # 上一步注册的 gym id
     num_envs: null
   teleop: spacemouse
   override_cfg:
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
     random_xy_range: 0.03

5. 验证
~~~~~~~

接硬件之前，先确认 id 已经注册，并且 entry point 可以解析：

.. code-block:: python

   from rlinf.envs.real import RealWorldEnv  # 触发全部任务注册
   from gymnasium.envs.registration import registry

   assert "WipeEnv-v1" in registry

``tests/unit_tests/test_real_env.py`` 会对每个内置任务做同样的检查。把新 id 加进
``EXPECTED_IDS``。

这些你都不用写
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 需求
     - 已经在哪里
   * - 连接硬件、放置部件
     - ``Robot.connect``，见 :doc:`../concepts/robotics`。
   * - 遥操作
     - 环境配置里的 ``teleop`` 选设备，wrapper 栈把它装上。
   * - 手动标记奖励、手动结束 episode
     - 环境配置里的 ``keyboard_reward_wrapper``。
   * - 相对坐标系、欧拉角转换、夹爪维度裁剪
     - ``real/wrappers/transforms/``，由 wrapper 栈套上。
   * - 各任务共用的阻抗参数
     - ``COMPLIANCE_DEFAULTS``，你只写差异项。

新增遥操作设备
--------------

一种新的遥操作设备需要三样东西：读取硬件的部件、把读数映射到具名机器人动作部件的
binding，以及把两者配对的注册表条目。

部件写在 ``rlinf/robotics/parts/teleop/devices.py``，只读取设备硬件：

.. code-block:: python

   class Pedal(TeleopPart):
       def _open(self):
           from .readers.pedal import PedalReader

           return PedalReader(port=self._port)

       @property
       def observation_features(self):
           return {"pressed": {"shape": (1,), "dtype": "bool"}}

       def get_observation(self):
           return {"pressed": np.asarray([self._reader.is_pressed()])}

``_open`` 在部件连接时打开硬件，``__init__`` 只记录声明。因此，设备可以在一台机器上
声明，再到另一台机器上构建。

binding 写在 ``rlinf/robotics/teleop/bindings.py``。它通过 ``PRODUCES`` 列出要填的
机器人动作部件，再由 ``action`` 返回这些具名部件的值：

.. code-block:: python

   class PedalGripperBinding(TeleopBinding):
       PRODUCES = ("end_effector",)

       def action(self, reading, context):
           return {"end_effector": np.array([-1.0 if reading["pressed"] else 1.0])}

       def is_driving(self, reading):
           return bool(reading["pressed"])

在 ``rlinf/envs/real/wrappers/teleop/builder.py`` 的 ``DEVICES`` 中登记部件与 binding 的
组合。能够使用该设备的环境也要加入这个设备名：

.. code-block:: python

   DEVICES = {..., "pedal": _pedal}

stack builder 不需要增加设备专用分支。配置中写入设备名后，系统会选中对应的注册表条目；
机器人若没有 ``end_effector``，装置会在构建时报错。

如果要加的是 wrapper
--------------------

如果要改的是 rollout 周围的行为，就该新增 wrapper，而不是任务。替换动作的逻辑放进
``teleop/``，观测或动作的表示转换放进 ``transforms/``；rollout 的起止与得分归
``episode/``。新增遥操作设备时实现 ``read``，新增键盘模式时继承
``KeyboardSession``。两个扩展点都在 :doc:`../concepts/realworld_envs` 中说明。
