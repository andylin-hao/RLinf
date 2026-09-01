真机任务与环境
==============

真机环境将机器人与任务组合，并在外层添加遥操作、人工结果标记和数据转换等 wrapper。

新增逻辑时，可根据职责确定代码位置：直接读写硬件的逻辑属于机器人零部件；奖励、成功条件和复位流程属于任务；只改变 rollout 控制方式或数据表示的逻辑属于 wrapper。若尚未了解零部件如何组合，请先阅读 :doc:`robotics`。

任务由配置和少量覆盖组成
--------------------------

``rlinf/envs/real/franka/`` 中的每个任务对应一个模块，公共逻辑位于 ``base.py``。通常只需新增一个配置 dataclass；任务需要特殊复位流程时，再覆盖相应的 env 方法。

.. code-block:: python

   @dataclass
   class PegInsertionConfig(FrankaRobotConfig):
       task_description: str = "peg and insertion"
       target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
       random_xy_range: float = 0.05

       def __post_init__(self):
           # 仅覆盖与公共阻抗参数不同的项。
           self.compliance_param = compliance(translational_stiffness=2000)
           ...


   class PegInsertionEnv(FrankaEnv):
       CONFIG_CLS = PegInsertionConfig

       def go_to_rest(self, joint_reset=False):
           # 先抬离插孔，再返回初始位姿，避免插销卡住。
           ...

``compliance()`` 将任务参数合并到 ``COMPLIANCE_DEFAULTS``。字段名错误或控制器不支持相应参数时，该函数会立即报错。插销任务只覆盖一项参数，bin relocation 覆盖十一项；位姿、奖励阈值和复位随机范围也分别保存在对应的任务配置中。

注册任务
--------

``TASKS`` 只记录 Gymnasium ID 与 env 类之间的对应关系。wrapper 由 env 自行声明，无需写入注册表：

.. code-block:: python

   TASKS = {
       "FrankaEnv-v1": FrankaEnv,
       "PegInsertionEnv-v1": PegInsertionEnv,
       "DualFrankaTcpEnv-v1": DualFrankaTcpEnv,
   }

   _ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

``register_tasks`` 根据该映射生成 Gymnasium entry point。用户配置和数据集元数据都会保存 Gym ID，因此数据采集开始后不应随意修改 ID。

通过机器人接口读写硬件
----------------------

env 持有一台组合完成的机器人。它构建机械臂、末端执行器和相机，调用 ``robot.connect()`` 打开硬件，并在 ``close()`` 中释放同一台机器人。每一步的观测和动作都通过 ``robot.get_observation()`` 和 ``robot.send_action()`` 完成，不直接访问 driver 或厂商 SDK。

不同硬件结构使用同一边界。Franka 的机械臂和末端执行器分别打开连接，因此使用并列路径；SO-101 的夹爪是机械臂总线上的另一个伺服，因此使用 ``arm.end_effector``。``SO101ReachEnv-v1`` 仍通过这套嵌套接口读写硬件，再将数据转换为 policy 使用的六维关节与夹爪向量。

不属于单步动作流的初始化操作，可以通过类型明确的零部件完成：

.. code-block:: python

   from rlinf.robotics import Arm, Camera

   arm = robot.child("arm", Arm)
   cameras = robot.parts_of_type(Camera)

   if not arm.is_robot_up():
       raise RuntimeError("The arm is not ready.")
   arm.reset_joint(reset_qpos)
   ready = all(camera.is_ready() for camera in cameras.values())

相机的 placement 和生命周期仍由机器人负责。env 可以保留相机引用用于处理画面，但不应为同一设备构建或关闭第二个对象。env 还应复用一次整机读取结果来构造当前步的状态和画面，避免混入后续 SDK 读取的数据。

按职责组织 wrapper
-------------------

wrapper 根据所承担的职责划分目录：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 包
     - wrapper 的职责
   * - ``teleop/``
     - 修改动作本身。在操作者接管期间，以操作者指令替换 policy 输出。
   * - ``transforms/``
     - 转换观测或动作的表示形式，不改变其语义。例如，将同一运动转换到末端坐标系。
   * - ``episode/``
     - 控制 rollout 的起止和评分。这些信息由现场操作者提供，而非由传感器产生。

``build_stack`` 根据 env 声明组装 wrapper，所有机器人共用同一套构建流程。

.. code-block:: python

   env = build_stack(PegInsertionEnv(...), cfg)

在环境侧处理动作接管
--------------------

:doc:`遥操作指南 <../guides/teleoperation>` 介绍设备选择和 binding。环境侧处理两个问题：在相邻采样之间保持接管状态，以及将以零部件名称为 key 的动作写入环境使用的扁平 action vector。

``TeleopIntervention`` 会在短时间内保留最近一次操作者动作，避免两次设备采样之间切回 policy。PICO 使用 grip 明确标识接管区间，因此将 ``timeout`` 设为 0，并在松开后立即交还控制权。数据采集器从 ``intervene_action`` 读取仲裁后的动作。

``TeleopGroup`` 按零部件名称返回动作，而环境接收扁平向量。``ComposedTeleop`` 根据环境声明的布局，将各零部件的动作写入对应区间；未由操作者控制的部分保留 policy 输出。

布局还声明动作语义。``FrankaEnv`` 将前六个数解释为位姿增量，``GimArmEnv`` 则将其解释为关节角。两者宽度相同，但 SpaceMouse 只能匹配前者。binding 与环境声明的动作类型不一致时，系统会在构建阶段报错。

单台设备使用一个配置项；组合多台设备时，将配置改为列表：

.. code-block:: yaml

   env:
     eval:
       teleop: spacemouse   # 也可以是 gello、pico、none
       gello_port: /dev/serial/by-id/...

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

分离设备读取与动作映射
----------------------

遥操作代码按以下职责分层：

- ``robotics/parts/teleop/<device>.py``：读取一台串口设备、HID 设备或头显，并声明其读数对应哪些机器人动作及各自的语义。它本身是 ``RobotPart``，因此拥有一致的连接、观测和断开接口，也可以放置到其他节点。
- ``robotics/parts/teleop/base.py``：保存 registry 和各设备的公共逻辑；``group.py`` 将多台设备的动作合并为一个动作。
- ``real/wrappers/teleop/builder.py``：解析配置中的设备名称；``composed.py`` 再根据零部件名称，将动作写入 env 声明的扁平 action vector。

设备层不依赖 Gymnasium，因此排查线缆或设备权限时可以单独运行设备，无需启动机器人。将具名动作写入某个 env 的扁平 action vector 仍由 env 层负责，因为只有它知道该布局。

连接机器人前，可使用以下命令检查主臂接线：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.gello --port /dev/ttyUSB0

遥操作设备本身也是 :class:`~rlinf.robotics.parts.base.RobotPart`。:class:`~rlinf.robotics.parts.teleop.base.TeleopPart` 直接继承该类，因此沿用标准连接生命周期。构造设备时不会访问硬件；wrapper stack 启动后，``TeleopGroup.connect()`` 才会依次打开设备。

遥操作设备在类型上继承 ``RobotPart``，但不会加入 ``Robot`` 的组合结构。主臂读取操作者输入，而非机器人状态，因此 policy 不会观测该设备。设备控制哪些机器人零部件，由环境侧的 binding 决定。这个边界也影响 placement：内置遥操作构建器在 env 进程中打开设备，不会经过 ``Robot.connect()``。手动部署独立设备前，请先阅读 :doc:`遥操作指南 <../guides/teleoperation>`。

将 episode 控制置于独立层
-------------------------

标记成功、放弃当前 rollout 或切换 policy 都不会修改机器人动作。此类 wrapper 位于 ``episode/``，并共用 :class:`KeyboardSession`。该类负责键盘监听、防抖，以及在 reset 时清空队列，避免机械臂返回初始位姿期间的输入影响下一个 episode。

新增控制模式时，读取 ``presses()`` 并定义各按键的含义：

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
     - 每个任务对应一个模块；``base.py`` 保存公共逻辑，``__init__.py`` 保存 ``TASKS`` 映射。
   * - ``robotics/parts/teleop/``
     - 每个操作者设备对应一个模块；``base.py`` 保存公共逻辑，``group.py`` 将多个设备合成一个动作。
   * - ``robotics/actions.py``
     - 动作向量中每一段的语义，env 与设备共用。
   * - ``real/wrappers/teleop/``
     - 设备选择、policy 与操作者动作仲裁、扁平动作布局，以及可选的直接推送路径。
   * - ``real/wrappers/transforms/``
     - 相对坐标系、四元数转欧拉角、夹爪维度裁剪。
   * - ``real/wrappers/episode/``
     - 各类键盘会话：奖励与结束、开始与结束、评测控制、policy 切换、主从臂。
   * - ``real/wrappers/__init__.py``
     - 组装三类 wrapper 的构建函数。
   * - ``real/registry.py``
     - ``task_factory`` 与 ``register_tasks``。
   * - ``real/env.py``
     - ``RealWorldEnv``，框架根据 ``env_type: real`` 创建的向量化环境类。
   * - ``real/task_env.py``
     - ``RobotTask`` 和 ``RobotTaskEnv`` 划定任务逻辑与硬件代码的边界。

后续阅读
--------

- :doc:`新增真机任务 <../extending/new_task>`：按步骤接入新的真机任务。
- :doc:`机器人接口 <robotics>`：了解如何读取和控制底层机器人。
