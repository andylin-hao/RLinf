新增任务
========

任务说明的是「让机器人做什么」：目标在哪、沿途要多柔顺、怎样算成功、每一条之间场景
如何随机化。机器人、放置和 wrapper 栈都已经有人管了，所以在 RLinf 已支持的硬件上新增
一个任务，只需要写一份配置 dataclass、一个 env 类，再在表里加一行。

如果机器人本身也是新的，先做 :doc:`new_robot`，任务总得跑在什么东西上面。

步骤
----

1. 写配置
~~~~~~~~~

在这台机器人的任务旁边新建模块 ``rlinf/envs/real/<robot>/<task>.py``，从机器人的配置
dataclass 继承，补上任务需要的字段：

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

只写和默认值不同的那几项阻抗参数。``compliance()`` 会把它们合并到
``COMPLIANCE_DEFAULTS`` 上，遇到控制器不接受的参数直接抛错；否则拼错的键会一路传到
阻抗控制器，在那里被静默忽略。

2. 写 env 类
~~~~~~~~~~~~

把 env 类指向你的配置。很多时候整个类就这么两行：

.. code-block:: python

   class WipeEnv(FrankaEnv):
       CONFIG_CLS = WipeConfig

任务行为不同时再覆盖对应的钩子。最常覆盖的是 ``go_to_rest``：从任务结束位姿回原位这件
事本身就和任务有关，比如插销任务要先抬离插孔，不然销子会在抬升时卡住。

.. code-block:: python

       def go_to_rest(self, joint_reset=False):
           reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
           reset_pose[2] += 0.05
           self._interpolate_move(reset_pose, timeout=1)
           super().go_to_rest(joint_reset)

3. 注册
~~~~~~~

在 ``rlinf/envs/real/<robot>/__init__.py`` 的 ``TASKS`` 表里加一行，写明 env 类和它的
动作空间所需的 wrapper 栈：

.. code-block:: python

   from .wipe import WipeEnv

   TASKS = {
       ...
       "WipeEnv-v1": (WipeEnv, apply_single_arm_wrappers),
   }

``register_tasks`` 会生成 entry point 并把这个 id 注册到 Gymnasium。单臂 Franka 和
Turtle2 用 ``apply_single_arm_wrappers``，双臂 Franka 用
``apply_dual_franka_joint_wrappers``。

gym id 会写进用户配置和数据集，取好之后就别再改了。

4. 加环境配置
~~~~~~~~~~~~~

在 ``examples/embodiment/config/env/`` 下新增 YAML，描述硬件和任务字段。
``override_cfg`` 里放的就是你在配置 dataclass 里定义的那些字段：

.. code-block:: yaml

   env_type: realworld
   init_params:
     id: "WipeEnv-v1"      # 上一步注册的 gym id
     num_envs: null
   teleop_device: spacemouse
   override_cfg:
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
     random_xy_range: 0.03

5. 验证
~~~~~~~

接硬件之前，先确认 id 已注册、entry point 能解析：

.. code-block:: python

   from rlinf.envs.real import RealWorldEnv  # 触发全部任务注册
   from gymnasium.envs.registration import registry

   assert "WipeEnv-v1" in registry

``tests/unit_tests/test_real_env.py`` 会对所有内置任务做这项检查，把你的 id 加进
``EXPECTED_IDS`` 即可。

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
     - 环境配置里的 ``teleop_device`` 选设备，wrapper 栈把它装上。
   * - 手动标记奖励、手动结束 episode
     - 环境配置里的 ``keyboard_reward_wrapper``。
   * - 相对坐标系、欧拉角转换、夹爪维度裁剪
     - ``real/transforms/``，由 wrapper 栈套上。
   * - 各任务共用的阻抗参数
     - ``COMPLIANCE_DEFAULTS``，你只写差异项。

如果要加的是 wrapper
--------------------

如果你要加的不是新任务，而是 rollout 周围的某种行为，就按「它改什么」放进对应的包：
改动作放 ``teleop/``，改观测或动作的表示方式放 ``transforms/``，决定 rollout 何时开始、
何时结束、拿多少分放 ``episode/``。新增遥操作设备实现 ``read``，新增键盘模式继承
``KeyboardSession``。两者都在 :doc:`../concepts/realworld_envs` 中有说明。
