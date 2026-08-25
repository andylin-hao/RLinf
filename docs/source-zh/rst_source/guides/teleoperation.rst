遥操作
======

遥操作允许操作者在 rollout 中接管 policy，可用于采集示教、纠正失败动作或运行 DAgger。接入时，建议先验证单台设备能否稳定读取数据，再配置 binding，最后组合多台设备。本页还会说明独立设备与 env 管理设备在 placement 上的差异。若尚未了解零部件名称与动作路径的关系，请先阅读 :doc:`../concepts/robotics`。

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
   * - ``none``
     - 不接操作者设备，由 policy 独立控制。
     - 无

设备输出必须与环境声明的动作类型一致。例如，双臂 Franka 不提供单臂笛卡尔动作，因此配置 ``spacemouse`` 时会直接报错，并列出当前环境支持的设备。

验证设备读数
------------

reader 不依赖机器人或集群，可独立运行：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

如果主臂只返回零值或 SpaceMouse 没有响应，请先通过 reader 排查接线和设备权限。完整机器人可使用 ``toolkits/realworld_check`` 检查；``check_robot_parts`` 会依次验证组合、读取和断开流程。

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

如果 binding 声明的零部件不在机器人中，``TeleopGroup`` 会跳过对应动作。如果某台设备无法匹配任何零部件，或两台设备同时声明控制同一零部件，系统会在构建阶段报错。

明确设备的资源归属
------------------

内置遥操作构建器会在 env 进程中创建设备，再由 ``TeleopGroup.connect()`` 直接打开。遥操作设备不会加入 ``Robot`` 的组合结构，因此 ``Robot.connect()`` 不会处理它的 placement。通过 ``env.*.teleop`` 配置设备时，应将设备接到 env worker 所在的机器。

遥操作设备本身也是一条 ``Connection``，因此同样接受 ``node_rank``，连接时就在该节点打开：

.. code-block:: python

   leader = TeleopLeaderArm("/dev/ttyUSB0", node_rank=1)
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

新增遥操作设备时，需要实现两个职责明确的扩展点。首先，在 ``robotics/parts/teleop`` 中将硬件 reader 封装为 ``TeleopPart``，沿用标准 connection 生命周期，且不依赖 Gymnasium。随后，在 ``real/wrappers/teleop/backends.py`` 中实现面向配置的 ``TeleopBackend``，将设备与 binding 组合起来。该 registry 位于 env 层，因为 backend 需要读取 env 配置，并根据 env 声明的动作语义选择 binding。

backend 应在实现该组合的同一文件中完成注册：

.. code-block:: python

   @TeleopBackend.register("example")
   class ExampleBackend(TeleopBackend):
       @classmethod
       def entry(cls, cfg, options, facts):
           device_cfg = dict(cfg.get("example_config", {}))
           unknown = set(options) - {"port", "drives"}
           if unknown:
               raise ValueError(f"Unsupported example options: {sorted(unknown)}")
           port = options.get("port", device_cfg.get("port"))
           if port is None:
               raise ValueError("teleop device 'example' requires a port")
           return TeleopEntry(
               ExampleDevice(port=port),
               ExampleBinding(),
               drives=options.get("drives"),
           )

``cfg`` 是完整的 env 配置段，用于读取设备级配置；``options`` 只属于当前列表项，可单独指定端口或 ``drives``。backend 应在此处校验允许的 key，避免拼写错误的硬件参数被静默忽略。

``entry()`` 返回 ``TeleopEntry``，其中包含设备、解释设备读数的 binding，以及可选的目标分支。``facts`` 描述 env 的动作布局和语义，例如机械臂接收绝对位姿还是增量。backend 可据此选择正确的 binding，无需导入某个具体 env class。

只有当某台设备还要在独立线程中直接下发指令时，才需要覆盖 ``streamer()``；除 ``gello_joint`` 外，其余设备均使用默认实现并返回 ``None``，参见 :ref:`提高主从臂跟随频率 <teleop-rate>`。

builder 会先构建所有 backend entry，再创建 streamer。streamer 可能接管并非由自身构造的设备；按照这一顺序创建，可确保 streamer 初始化时已经取得本次配置请求的全部设备和 binding。

最后，将注册名加入对应 env 的 ``TELEOP`` 元组，声明该 env 能够表示该设备产生的动作。这里无需再次注册设备；公共 builder 会通过 ``TeleopBackend`` 查询注册名并构建相应的 entry。

已废弃的配置项
--------------

旧配置使用 ``teleop_device`` 指定设备，或通过 ``use_spacemouse``、``use_gello``、``use_gello_joint`` 和 ``use_pico`` 启用设备。这些字段仍可读取，但会产生 deprecation warning。如果旧字段与 ``teleop`` 同时出现，系统以 ``teleop`` 为准，并在 warning 中列出被覆盖的字段。

后续阅读
--------

- :doc:`机器人模型 <../concepts/robotics>`：了解 binding 所引用的零部件路径。
- :doc:`真机环境模型 <../concepts/realworld_envs>`：了解遥操作在 wrapper 栈中的位置。
- :doc:`数据采集 <data_collection>`：记录操作者动作。
