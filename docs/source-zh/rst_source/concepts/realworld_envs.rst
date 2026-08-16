真机环境模型
============

真机环境把机器人和任务接到一起，外面再套上 rollout 期间需要的 wrapper。机器人给出
运动与感知能力，目标位姿、奖励规则和复位行为则写在任务里。人来接管、手动标记结果，
或要转换位姿表示时，都由 wrapper 处理。

如果还不清楚机器人怎样组合部件、怎样放到各个节点，先看 :doc:`robotics`。这里我们只看
环境这一层。

一个任务 = 一份配置 + 几个覆盖
------------------------------

打开 ``rlinf/envs/real/franka/``，每个 Franka 任务各占一个模块，旁边的 ``base.py``
存放公共逻辑。大多数任务先用 dataclass 写下目标和阻抗参数，env 类再补充这个任务特有
的复位行为。

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

``compliance()`` 把这些覆盖项合并到 ``COMPLIANCE_DEFAULTS`` 上。控制器不接受的参数
会直接抛错，拼错的键也不会传到阻抗控制器后被静默忽略。插销任务只改一项，bin
relocation 改十一项。配置的其余部分描述任务本身，包括位姿、奖励阈值和复位时的随机化
范围。

注册只要一行
------------

任务表的每一行只记两件事：用 worker 配置构造哪个 env 类，以及这台机器人的动作空间
需要哪套 wrapper。新增任务时加一行即可：

.. code-block:: python

   TASKS = {
       "FrankaEnv-v1": (FrankaEnv, apply_single_arm_wrappers),
       "PegInsertionEnv-v1": (PegInsertionEnv, apply_single_arm_wrappers),
       "DualFrankaTCPEnv-v1": (DualFrankaTCPEnv, apply_dual_franka_joint_wrappers),
   }

   _ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

``register_tasks`` 会生成对应的 Gymnasium entry point 并完成注册。用户配置和数据集
元数据都会保存 gym id；以后改名，这些引用也得跟着改。最好在采集数据前把名字定下来。

三类 wrapper
------------

看 wrapper 改了什么，就知道它该放在哪个包：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 包
     - 它的 wrapper 改什么
   * - ``teleop/``
     - 改动作本身。操作者接管期间，他的指令会替换策略输出。
   * - ``transforms/``
     - 改观测或动作的表示，不改含义。相对坐标系只是把同一个运动写到末端坐标下。
   * - ``episode/``
     - 决定 rollout 的起止和得分。这些判断来自现场操作者，传感器不会上报。

``wrappers.py`` 读取环境配置，选定遥操作设备，再把这些 wrapper 组装成完整的栈。

.. code-block:: python

   env = apply_single_arm_wrappers(PegInsertionEnv(...), cfg)

遥操作：一个 wrapper，多种设备
------------------------------

遥操作设备各有不同的读取方式，返回的却是同一种信息：操作者此刻想让机器人做什么。
设备只实现读取这一步：

.. code-block:: python

   class SpaceMouseTeleop(TeleopDevice):
       def read(self, env, policy_action) -> TeleopSample:
           expert, buttons = self.expert.get_action()
           return TeleopSample(
               action=expert,
               active=bool(np.linalg.norm(expert) > 0.001),
               info={"left": buttons[0], "right": buttons[1]},
           )

接下来的事交给 ``TeleopIntervention``。它用保持窗口维持两次采样之间的接管状态，处理
松手后的回退动作，并写入数据采集器要读取的 ``intervene_action``。

``active`` 表示操作者是否正在接管，动作本身不作这个判断。大多数设备静止时仍有细小
抖动，因此各自设定阈值。PICO 握把或扳机这类按住才生效的设备能准确标出接管区间，直接
把 ``timeout`` 设为 0。如果再延续半秒，操作者松手后机器人仍会收到指令。

主臂要跟得顺，从臂就得持续接收目标，更新频率远高于 ``env.step``。
:class:`StreamingTeleopDevice` 为这类设备单独开一个线程。环境复位时，线程先暂停；发送
第一个目标前还会完成对齐，退出时再 join。

选设备就是一个配置项：

.. code-block:: yaml

   env:
     eval:
       teleop_device: spacemouse   # 也可以是 gello、pico、gello_joint、none
       gello_port: /dev/serial/by-id/...

设备读取与动作转换分开
----------------------

``teleop/`` 内部还把设备 I/O 与环境动作转换分开。``devices/`` 只和串口、头显打交道，
不导入 Gymnasium；``adapters.py`` 再把设备读数换算成某个环境的动作。工装脚本因此可以
脱离环境单独运行。

接机器人之前，先用下面的命令检查主臂接线：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

遥操作设备不是 :class:`~rlinf.robotics.parts.base.RobotPart`。部件描述策略看到的物理
组件，主臂不在这棵部件树里：策略不观测它，机器人也不会组合它。主臂读取的是操作者
指令，代码归在环境这一侧。

episode 控制不算遥操作
----------------------

操作者还会标记成功、放弃当前这条，或者在 rollout 中途切换策略。这些选择都不改动作，
相关 wrapper 放在 ``episode/``，并共用 :class:`KeyboardSession`。它持有按键监听器，
防抖窗口内的重复输入会被丢弃。环境复位时，队列也会清空，免得回原位期间踩到的踏板
提前开始下一个 episode。

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
   * - ``robotics/parts/teleop/``
     - 操作者驱动的设备，本身是部件；底下 ``readers/`` 里的厂商读取实现不导入
       Gymnasium。
   * - ``real/wrappers/teleop/``
     - ``intervention.py``、``adapters.py``、``streaming.py``、``pico.py``，以及选设备用的
       ``config.py``。
   * - ``real/wrappers/transforms/``
     - 相对坐标系、四元数转欧拉角、夹爪维度裁剪。
   * - ``real/wrappers/episode/``
     - 各类键盘会话：奖励与结束、开始与结束、评测控制、策略切换、主从臂。
   * - ``real/wrappers/__init__.py``
     - 把三类 wrapper 组装起来的构建函数。
   * - ``real/registry.py``
     - ``task_factory`` 与 ``register_tasks``。
   * - ``real/env.py``
     - ``RealWorldEnv``，框架根据 ``env_type: realworld`` 创建的向量化环境。
   * - ``real/task_env.py``
     - ``RobotTask`` 和 ``RobotTaskEnv`` 划定任务逻辑与硬件代码的边界。

下一步
------

- :doc:`新增任务 <../extending/new_task>`：按步骤接入一个新任务。
- :doc:`机器人模型 <robotics>`：下面那台机器人是怎么组合出来的。
