遥操作
======

遥操作允许操作者在 rollout 中接管 policy，可用于采集示教、纠正失败动作或运行 DAgger。接入时，建议先验证单台设备能否稳定读取数据，确认无误后再组合多台设备。本页按这一顺序展开，最后说明独立设备与 env 管理设备在 placement 上的差异。若尚未了解零部件名称与动作路径的关系，请先阅读 :doc:`../concepts/robotics`。

选择设备
--------

仅使用一台设备时，在配置中填写设备名称：

.. code-block:: yaml

   env:
     eval:
       teleop: spacemouse

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - 设备
     - 操作方式
     - 额外配置
   * - ``spacemouse``
     - 通过 6 自由度操作球控制机械臂，并使用按键锁定夹爪状态。
     - 无
   * - ``gello``
     - 操作主臂，使从臂跟随至相同位姿。
     - ``gello_port``
   * - ``gello_joint``
     - 操作主臂，使从臂逐关节跟随；每条手臂配置一项。
     - ``left_gello_port`` / ``right_gello_port``
   * - ``pico``
     - 使用手持 VR 控制器，按住 grip 时接管机器人；双臂机器人需为每条手臂分别配置。
     - ``pico:`` 段
   * - ``glove``
     - 弯曲手指控制灵巧手，与驱动机械臂的设备配合使用。
     - ``glove_config:`` 段
   * - ``so101_leader``
     - 操作 SO-101 主臂，其夹爪同时控制从臂夹爪。
     - ``so101_leader_port``
   * - ``none``
     - 不接操作者设备，由 policy 独立控制。
     - 无

设备输出必须与环境声明的动作类型一致。例如，双臂 Franka 不提供单臂笛卡尔动作，因此配置 ``spacemouse`` 时会直接报错，并列出当前环境支持的设备。

验证设备读数
------------

由后台线程持续轮询的设备——``gello``、``gello_joint`` 和 ``spacemouse``——不依赖机器人或集群，可独立运行：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.gello --port /dev/ttyUSB0

如果主臂只返回零值或 SpaceMouse 没有响应，请先用该命令排查接线和设备权限。完整机器人可使用 ``toolkits/realworld_check`` 检查；``check_robot_parts`` 会依次验证组合、读取和断开流程。

组合多台设备
------------

确认每台设备均可独立工作后，将 ``teleop`` 改为列表。每一项只控制对应的机器人零部件：

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

在上述配置中，SpaceMouse 控制机械臂，数据手套控制灵巧手。按住 SpaceMouse 的第二个按键时，手套开始接管；松开后，灵巧手保持最后一个位姿。

设备级参数保留在对应的配置段中。例如，项目内置的灵巧手配置通过 ``glove_config`` 指定数据手套端口和标定文件：

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]
       glove_config:
         left_port: /dev/ttyACM0
         frequency: 60
         config_file: null

当机器人包含两个同类分支时，使用 ``drives`` 指定每台设备控制的分支。该字段是遥操作配置中唯一直接引用零部件名称的位置：

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

如果设备声明的零部件不在机器人中，``TeleopGroup`` 会跳过对应动作。如果某台设备无法匹配任何零部件，或两台设备同时声明控制同一零部件，系统会在构建阶段报错。

明确设备的资源归属
------------------

内置遥操作构建器会在 env 进程中创建设备，再由 ``TeleopGroup.connect()`` 直接打开。遥操作设备不会加入 ``Robot`` 的组合结构，因此 ``Robot.connect()`` 不会处理它的 placement。通过 ``env.*.teleop`` 配置设备时，应将设备接到 env worker 所在的机器。

遥操作设备本身也是一条 ``Connection``，因此同样接受 ``node_rank``，连接时就在该节点打开：

.. code-block:: python

   leader = Gello("/dev/ttyUSB0", node_rank=1)
   leader.connect()
   try:
       print(leader.get_observation())
   finally:
       leader.disconnect()

该写法适用于独立诊断，也适用于 ``TeleopGroup``，因为 group 通过同一套 connection 接口打开每台设备。主臂的运行节点由构造时传入的 ``node_rank`` 决定。``env.*.teleop`` 配置目前尚未提供该字段，因此通过该配置创建的设备均在 env 进程中打开。

每次采样对每台独立设备只读取一次。同一台设备即使同时控制两个零部件，也只会打开一次；因此，SpaceMouse 同时控制机械臂和夹爪时仍只占用一个 HID 句柄。

.. _teleop-rate:

提高主从臂跟随频率
------------------

如果主臂目标仅按 policy 的执行频率下发，从臂可能出现跟随延迟。直推模式使用独立线程，以约 1 kHz 的频率向控制器发送关节目标；``env.step`` 仍会读取状态，但不再转发运动指令：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         teleop_direct_stream: true

仅在跟随延迟明显时启用该选项。启用后，``env.step`` 不再发送关节目标；如果配置有误，机器人将保持静止。

新增设备
--------

一台设备对应 ``robotics/parts/teleop/`` 下的一个模块。继承 ``TeleopDevice``，为它注册一个配置名称，并声明它填充哪些动作：

.. code-block:: python

   @TeleopDevice.register("example")
   class ExampleDevice(TeleopDevice):
       PRODUCES = {"arm": ActionKind.JOINT_POSITION}
       NEEDS = ("joint_positions",)

       def __init__(self, port: str) -> None:
           self._port = port

       def _open(self):
           return ExampleSDK(self._port).open()

       @property
       def observation_features(self) -> Features:
           return {"joints": {"shape": (7,), "dtype": "float32"}}

       def get_observation(self) -> Observation:
           return {"joints": self._device.read()}

       def action(self, reading, context) -> TeleopAction:
           moved = np.linalg.norm(reading["joints"] - context["joint_positions"][0])
           return TeleopAction({"arm": reading["joints"]}, driving=bool(moved > 0.01))

``PRODUCES`` 声明设备填充哪些动作零部件及其语义，env 因此可以在打开硬件之前完成校验。``NEEDS`` 声明设备需要的机器人状态；无论有几台设备请求同一项状态，每次采样都只读取一次，并通过 ``context`` 传入。

``_open()`` 负责连接硬件并返回句柄，设备随后通过 ``self._device`` 读取；``_release()`` 负责关闭。这与其他机器人零部件使用同一套 connection 生命周期，``node_rank`` 也由同一套机制处理，设备无需为此编写任何代码——所以上面的构造函数只接收自己的参数。

``observation_features`` 在打开硬件之前声明读数结构，因此可以离线描述一套设备。该方法是抽象方法，未实现的设备无法实例化。

``action()`` 返回 ``TeleopAction``，其中包含设备填充的动作零部件，以及本次采样操作者是否正在接管。若本次不产生动作，返回 ``driving=False``，控制权保留给 policy。

面向配置的行为
~~~~~~~~~~~~~~

``from_config()`` 的默认实现会将该列表项自身的 options 作为关键字参数传给构造函数。只要设备的配置 key 与构造参数同名，就无需再写任何代码：上面的设备已经可以通过 ``{example: {port: /dev/ttyUSB0}}`` 使用。

如果需要读取设备级配置段，或根据机器人的动作语义调整行为，则覆盖该方法：

.. code-block:: python

   @classmethod
   def from_config(cls, cfg, options, facts):
       settings = dict(cfg.get("example_config", {}))
       settings.update({k: v for k, v in options.items() if k != "drives"})
       if "port" not in settings:
           raise ValueError("teleop device 'example' requires a port")
       return TeleopEntry(cls(**settings), drives=options.get("drives"))

``cfg`` 是完整的 env 配置段，用于读取设备级配置；``options`` 只属于当前列表项，可单独指定端口或 ``drives``。应在此处校验允许的 key，避免拼写错误的硬件参数被静默忽略。``facts`` 描述 env 的动作布局和语义，例如机械臂接收绝对位姿还是增量，设备可据此调整，无需导入某个具体 env class。

只有当某台设备还要在独立线程中直接下发指令时，才需要覆盖 ``streamer()``；除 ``gello_joint`` 外，其余设备均使用默认实现并返回 ``None``，参见 :ref:`提高主从臂跟随频率 <teleop-rate>`。builder 会先构建所有 entry，再创建 streamer，因此 streamer 可以接管并非由自身构造的设备。

最后，将注册名加入对应 env 的 ``TELEOP`` 元组，声明该 env 能够表示该设备产生的动作。这里无需再次注册设备。

已废弃的配置项
--------------

旧配置使用 ``teleop_device`` 指定设备，或通过 ``use_spacemouse``、``use_gello``、``use_gello_joint`` 和 ``use_pico`` 启用设备。这些字段仍可读取，但会产生 deprecation warning。如果旧字段与 ``teleop`` 同时出现，系统以 ``teleop`` 为准，并在 warning 中列出被覆盖的字段。

后续阅读
--------

- :doc:`机器人组成 <../concepts/robotics>`：了解设备所填充的零部件路径。
- :doc:`真机任务与环境 <../concepts/realworld_envs>`：了解遥操作在 wrapper 栈中的位置。
- :doc:`数据采集 <data_collection>`：记录操作者动作。
