真机环境模型
============

一个真机环境由三部分组成：机器人、任务，以及一层层 wrapper。机器人管运动和感知，
任务判断怎样算成功，wrapper 则涵盖人在一次 rollout 周围所做的事情：用主臂接管、标记
这一条成功、换一种方式表示位姿。下面先跟着一个任务从配置走到 gym id，再看看各类
wrapper 各自放在哪里、为什么这样分。

机器人本身如何由部件组合、如何放到节点上，请先看 :doc:`robotics`。这一页从环境这一层
接着讲。

一个任务 = 一份配置 + 几个覆盖
------------------------------

打开 ``rlinf/envs/real/franka/``，里面就是 Franka 能做的各项任务，一个任务一个模块，
旁边是它们共用的 ``base.py``。任务通常是一个 dataclass：写清目标在哪、机械臂沿途要
多柔顺，再加上这个任务特有的复位动作。

.. code-block:: python

   @dataclass
   class PegInsertionConfig(FrankaRobotConfig):
       task_description: str = "peg and insertion"
       target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
       random_xy_range: float = 0.05

       def __post_init__(self):
           # 只写和公共阻抗参数不一样的那几项。
           self.compliance_param = compliance(translational_stiffness=2000)
           ...


   class PegInsertionEnv(FrankaEnv):
       CONFIG_CLS = PegInsertionConfig

       def go_to_rest(self, joint_reset=False):
           # 先抬离插孔再回原位，否则销子会卡住。
           ...

``compliance()`` 把你写的这几项合并到 ``COMPLIANCE_DEFAULTS`` 上，遇到控制器不认识的
参数直接报错，拼错的键不会一路传到阻抗控制器再被悄悄丢掉。插销任务只改一项，
bin relocation 改十一项。配置里剩下的部分就是任务本身：位姿、奖励阈值，以及复位时的
随机化范围。

注册只要一行
------------

每个任务的构建方式都一样：用 worker 传来的配置构造 env 类，再套上这台机器人动作空间
所需的 wrapper。所以任务需要声明的也就这两样：

.. code-block:: python

   TASKS = {
       "FrankaEnv-v1": (FrankaEnv, apply_single_arm_wrappers),
       "PegInsertionEnv-v1": (PegInsertionEnv, apply_single_arm_wrappers),
       "DualFrankaTCPEnv-v1": (DualFrankaTCPEnv, apply_dual_franka_joint_wrappers),
   }

   _ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

``register_tasks`` 会生成对应的 Gymnasium entry point 并完成注册。gym id 会写进用户
配置和数据集，所以取名时想清楚，之后尽量别改。

三类 wrapper
------------

套在任务 env 外面的东西都能归进三类，归哪一类取决于这个 wrapper 改的是什么：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 包
     - 它的 wrapper 改什么
   * - ``teleop/``
     - 改动作本身。人接管的这段时间里，他的指令替换掉策略的输出。
   * - ``transforms/``
     - 改观测或动作的表示方式，不改含义。相对坐标系只是把同一个运动换到末端坐标下。
   * - ``episode/``
     - 决定一次 rollout 何时开始、何时结束、拿到多少分。这些判断只有旁边的人知道，
       传感器不会告诉你。

``wrappers.py`` 把它们组装起来：读环境配置，确定用哪种遥操作设备，返回整个栈。

.. code-block:: python

   env = apply_single_arm_wrappers(PegInsertionEnv(...), cfg)

遥操作：一个 wrapper，多种设备
------------------------------

所有遥操作设备回答的都是同一个问题：此刻操作者想让机器人做什么。拿到答案之后的处理
也完全一样。所以设备只需要实现「怎么读」这一件事：

.. code-block:: python

   class SpaceMouseTeleop(TeleopDevice):
       def read(self, env, policy_action) -> TeleopSample:
           expert, buttons = self.expert.get_action()
           return TeleopSample(
               action=expert,
               active=bool(np.linalg.norm(expert) > 0.001),
               info={"left": buttons[0], "right": buttons[1]},
           )

其余都交给 ``TeleopIntervention``：两次采样之间维持接管状态的保持窗口、松手后的回退
动作，以及数据采集要读的 ``intervene_action``。

判断「人在操作」看的是 ``active``，不是动作本身。设备总会有细小的抖动读数，所以阈值由
每个设备自己定。像 PICO 扳机这种按住才生效的设备则把 ``timeout`` 设为 0：按键已经精确
表示了操作区间，再多留半秒，就会在人松手之后继续给机器人发指令。

主臂要跟得顺，从臂必须持续收到目标，频率远高于 ``env.step``。这类设备用
:class:`StreamingTeleopDevice`，它自带一个线程，也替你收拾线程带来的那些麻烦事：环境
复位时暂停、发第一个目标前先对齐、退出时 join。

选设备就是一个配置项：

.. code-block:: yaml

   env:
     eval:
       teleop_device: spacemouse   # 也可以是 gello、pico、gello_joint、none
       gello_port: /dev/serial/by-id/...

设备读取与动作转换分开
----------------------

``teleop/`` 内部还有一层划分，正是它让工装脚本能单独跑起来。``devices/`` 只放读取代码，
也就是和串口、头显打交道的部分，不 import Gymnasium；``adapters.py`` 才把读数换算成
某个环境的动作。

于是接机器人之前，可以先确认主臂接线是否正常：

.. code-block:: bash

   python -m rlinf.envs.real.teleop.devices.gello --port /dev/ttyUSB0

遥操作设备不是 :class:`~rlinf.robotics.parts.base.RobotPart`。部件描述的是「这个组件对
策略意味着什么」，而主臂给不出这个答案：没有策略会观测它，也没有机器人会把它组合进来。
它读的是人，不是机器人，所以归在环境这一侧。

episode 控制不算遥操作
----------------------

标记成功、放弃这一条、rollout 中途切换策略：这些同样只有人知道，但都不碰动作。相关
wrapper 放在 ``episode/``，共用 :class:`KeyboardSession`。它持有监听器，在防抖窗口内
丢掉重复按键，并在复位时清空队列，免得回原位时踩到的踏板把下一个 episode 提前打开。

新增一种模式，只要读 ``presses()`` 并规定每个键的含义：

.. code-block:: python

   class KeyboardRLTPolicySwitchWrapper(KeyboardSession):
       def step(self, action):
           obs, reward, terminated, truncated, info = self.env.step(action)
           for key in self.presses():
               if key == "b":
                   self._rlt_switch_flags = True
           info["rlt_switch_flags"] = self._rlt_switch_flags
           return obs, reward, terminated, truncated, info

代码位置
--------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 路径
     - 内容
   * - ``real/<robot>/``
     - 一个任务一个模块，另有存放公共逻辑的 ``base.py``，以及写着 ``TASKS`` 表的
       ``__init__.py``。
   * - ``real/teleop/devices/``
     - GELLO、数据手套、键盘、PICO、SpaceMouse 的读取实现，不含 Gymnasium。
   * - ``real/teleop/``
     - ``intervention.py``、``adapters.py``、``streaming.py``、``pico.py``，以及选设备用的
       ``config.py``。
   * - ``real/transforms/``
     - 相对坐标系、四元数转欧拉角、夹爪维度裁剪。
   * - ``real/episode/``
     - 各类键盘会话：奖励与结束、开始与结束、评测控制、策略切换、主从臂。
   * - ``real/wrappers.py``
     - 把三类 wrapper 组装起来的构建函数。
   * - ``real/registry.py``
     - ``task_factory`` 与 ``register_tasks``。
   * - ``real/robot_task_env.py``
     - ``RobotTask`` 和 ``RobotTaskEnv``，把任务逻辑挡在硬件之外。

下一步
------

- :doc:`新增任务 <../extending/new_task>`：按步骤接入一个新任务。
- :doc:`机器人模型 <robotics>`：下面那台机器人是怎么组合出来的。
