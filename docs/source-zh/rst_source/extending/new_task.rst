新增任务
========

如果 RLinf 已经能连接这台机器人，你需要改的是“让它完成什么”，请从本页开始。完成后，你会得到
一份配置 dataclass、一个简短的 env 类、一个已注册的 Gymnasium ID，以及可以直接启动的 YAML 配置。

这个过程不改机器人的构建、设备放置和遥操作。目标、阻抗参数、成功判定和复位行为写在任务中，
机器人与 wrapper 栈继续沿用现有实现。如果硬件本身也是新的，先阅读 :doc:`new_robot`；等观测和动作能够
通过部件后，再回到这里。

步骤
----

下面以给 Franka 新增 ``WipeEnv-v1`` 为例。请按顺序完成：后面会继续使用第一步定义的配置类，
也会引用前面选定的 Gymnasium ID。

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

如果任务沿用机器人原有的动作空间，动作的读法也一并继承。任务若改动了动作空间，就要把改动
声明出来：遥操作设备匹配的是每个部件的含义，而不是它有多宽。

.. code-block:: python

       def action_parts(self):
           return (
               ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),
               ActionPart("end_effector", 1, ActionKind.GRIPPER),
           )

声明的宽度之和必须与动作空间完全一致；对不上会直接报错，而不是让某一段悄悄落到别处。

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

   env_type: real
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
       def __init__(self, port: str) -> None:
           self._port = port

       def _open(self):
           from .readers.pedal import PedalReader

           return PedalReader(port=self._port)

       @property
       def observation_features(self):
           return {"pressed": {"shape": (1,), "dtype": "bool"}}

       def get_observation(self):
           return {"pressed": np.asarray([self._device.is_pressed()])}

``_open`` 在部件连接时打开硬件，并返回与硬件对话的那个对象；它就是 ``self._device``。
``__init__`` 只记录声明。因此，设备可以在一台机器上声明，再到另一台机器上构建。

binding 写在 ``rlinf/robotics/teleop/bindings.py``。``PRODUCES`` 把它要填的每个动作
部件映射到这些数字的\ **含义**\ ，因此把 twist 交给关节空间机械臂的设备会被拒绝，而不是
被照单执行。``action`` 用一个 ``TeleopAction`` 一次性回答关于这次读数的全部问题：

.. code-block:: python

   class PedalGripperBinding(TeleopBinding):
       PRODUCES = {"end_effector": ActionKind.GRIPPER}

       def action(self, reading, context):
           pressed = bool(reading["pressed"])
           return TeleopAction(
               parts={"end_effector": np.array([-1.0 if pressed else 1.0])},
               driving=pressed,
           )

``driving`` 属于这一次回答，而不是另开一次调用：如果 binding 的答案依赖它刚算出来的
状态，分成两次就得把状态留在中间，而且没有任何东西能保证两次调用的先后顺序。

先写一个条目构造函数把部件和 binding 配起来，再在
``rlinf/envs/real/wrappers/teleop/builder.py`` 的 ``DEVICES`` 中登记：

.. code-block:: python

   def _pedal(cfg, options, facts):
       return TeleopEntry(
           Pedal(port=options["port"]),
           PedalGripperBinding(),
           drives=options.get("drives"),
       )

   DEVICES = {..., "pedal": _pedal}

stack builder 不需要增加设备专用分支。配置中写入设备名后，系统会选中对应的注册表条目；
机器人若没有 ``end_effector``，这套遥操组合会在构建时报错。

如果要加的是 wrapper
--------------------

如果要改的是 rollout 周围的行为，就该新增 wrapper，而不是任务。替换动作的逻辑放进
``teleop/``，观测或动作的表示转换放进 ``transforms/``；rollout 的起止与得分归
``episode/``。新增遥操作设备靠的是上面那对部件与 binding，而不是 wrapper；新增键盘
模式时继承 ``KeyboardSession``。两个扩展点都在 :doc:`../concepts/realworld_envs` 中说明。
