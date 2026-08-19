真机环境模型
============

真机环境先把机器人与一个任务组合起来，再加上 rollout 期间需要、但不应进入前两者的行为，
例如操作者接管、手动标记结果，以及观测和动作的表示转换。

拿不准一段新逻辑应该放在哪里时，可以按顺序问三个问题：它是否直接读写硬件？是否定义成功、奖励或
复位？还是只改变 rollout 的控制和表示方式？这三种情况分别属于部件、任务和 wrapper。

如果还不熟悉具名部件树，请先读 :doc:`robotics`。本页从环境边界开始，看一个任务如何逐层组成
完整的 wrapper 栈。

一个任务通常只需要少量定制
----------------------------

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
会直接抛错，字段名拼错也不会继续传到阻抗控制器。插销任务只改一项，bin
relocation 改十一项。配置的其余部分描述任务本身，包括位姿、奖励阈值和复位时的随机化
范围。

注册只要一行
------------

任务表的每一行只记一件事：用 worker 配置构造哪个 env 类。外面套哪些 wrapper 由
env 自己声明，所以新增任务只要加一行：

.. code-block:: python

   TASKS = {
       "FrankaEnv-v1": FrankaEnv,
       "PegInsertionEnv-v1": PegInsertionEnv,
       "DualFrankaTcpEnv-v1": DualFrankaTcpEnv,
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

``build_stack`` 读取环境配置，按 env 自己的声明把 wrapper 逐层套上；无论哪台机器人
都走这一条路径。

.. code-block:: python

   env = build_stack(PegInsertionEnv(...), cfg)

遥操作：留在这一侧的部分
------------------------

:doc:`遥操作指南 <../guides/teleoperation>` 说明怎样选设备、怎样用 binding 解释读数。到了环境边界，
还剩两个问题：接管动作保持多久，以及怎样把各部件的动作拼成环境所需的扁平向量。

``TeleopIntervention`` 会在设备的相邻采样之间短暂保留最新操作者动作。没有这个窗口，
操作者还在移动时，控制权可能在策略与操作者之间反复跳转。PICO grip 采用按住接管，
控制件本身已经精确标出接管区间，因此把 ``timeout`` 设为 0。松开后若继续保留动作，
机器人仍会收到操作者的指令。数据采集器从 ``intervene_action`` 读取仲裁后的动作。

接管关系确定后，还要把动作整理成环境接收的形状。group 按部件名产出动作，环境却只接收
一个扁平向量；``ComposedTeleop`` 按环境声明的布局写入各部件。没有设备驱动的部件保留
策略给出的值，摆好的手也就停在上一个指令位置。

这份声明不只说明每个部件的位置，还说明它的含义。``FrankaEnv`` 把前六个数读作位姿增量，
``GimArmEnv`` 把前六个数读作关节角，所以 SpaceMouse 能驱动前者而不能驱动后者。两者宽度
完全相同，因此宽度不能作为判断依据。设备产生的指令若会被机器人误读，遥操组合在构建时就会报错；
没有声明的环境则根本无法接入遥操作。

单台设备只需一个配置项；多台设备组合使用时，改用列表：

.. code-block:: yaml

   env:
     eval:
       teleop: spacemouse   # 也可以是 gello、pico、none
       gello_port: /dev/serial/by-id/...

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

把设备读取与动作含义分开
------------------------

遥操作会穿过几层代码，每一层只做一件事：

- ``robotics/parts/teleop/readers/`` 直接读串口、HID 设备和头显。
- ``robotics/parts/teleop/devices.py`` 把 reader 包成 ``TeleopPart``，与其他部件一样连接、
  返回观测并断开。
- ``robotics/teleop/bindings.py`` 说明一次读取应当填入机器人的哪个动作部件。
- ``real/wrappers/teleop/composed.py`` 把具名动作写入环境声明的扁平动作向量。

分开之后，可以先排查线缆和权限，不必立即启动机器人；同一台物理设备也可以通过另一个 binding
获得不同的动作含义。

接机器人之前，先用下面的命令检查主臂接线：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

遥操作设备本身也是 :class:`~rlinf.robotics.parts.base.RobotPart`。
:class:`~rlinf.robotics.parts.teleop.devices.TeleopPart` 直接继承该类，因此使用同一套生命周期和放置方式。
``SpaceMouse.at(node_rank=1)`` 会把设备放到它实际插接的机器上，用法与放置机械臂一致。

它不属于的是\ **机器人**\ 。主臂不在机器人的组合里，策略也不会观测它：它读的是操作者，
不是机器人。它填机器人动作的哪些部件由 binding 决定，那部分代码归在环境这一侧。

episode 控制不算遥操作
----------------------

操作者还会标记成功、放弃当前这条，或者在 rollout 中途切换策略。这些选择都不改动作，
相关 wrapper 放在 ``episode/``，并共用 :class:`KeyboardSession`。它持有按键监听器，
防抖窗口内的重复输入会被丢弃。环境复位时，队列也会清空，免得回原位期间踩到的踏板
提前开始下一个 episode。

新增一种模式，只要读 ``presses()`` 并规定每个按键的含义：

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
     - 操作者设备及其部件接口；底层 ``readers/`` 不导入 Gymnasium。
   * - ``robotics/teleop/``
     - binding、动作含义和 ``TeleopGroup`` 组合。
   * - ``real/wrappers/teleop/``
     - 设备选择、策略与操作者动作仲裁、扁平动作布局，以及可选的直接推送路径。
   * - ``real/wrappers/transforms/``
     - 相对坐标系、四元数转欧拉角、夹爪维度裁剪。
   * - ``real/wrappers/episode/``
     - 各类键盘会话：奖励与结束、开始与结束、评测控制、策略切换、主从臂。
   * - ``real/wrappers/__init__.py``
     - 把三类 wrapper 组装起来的构建函数。
   * - ``real/registry.py``
     - ``task_factory`` 与 ``register_tasks``。
   * - ``real/env.py``
     - ``RealWorldEnv``，框架根据 ``env_type: real`` 创建的向量化环境。
   * - ``real/task_env.py``
     - ``RobotTask`` 和 ``RobotTaskEnv`` 划定任务逻辑与硬件代码的边界。

下一步
------

- :doc:`新增任务 <../extending/new_task>`：按步骤接入一个新任务。
- :doc:`机器人模型 <robotics>`：下面那台机器人是怎么组合出来的。
