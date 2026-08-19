遥操作
======

遥操作允许操作者在 rollout 途中替换策略动作，可用于采集示教、挽救失败动作或运行 DAgger。
建议先接入一台设备，确认读取正常；然后用 binding 把读数对应到机器人的具名动作部件；这条路径
跑通后，再组合多台设备或将设备放到其他机器上。

本页就按这个顺序展开。如果还不熟悉具名部件树，请先读 :doc:`../concepts/robotics`。

选一个设备
----------

只用一台设备时，在配置中写设备名：

.. code-block:: yaml

   env:
     eval:
       teleop: spacemouse

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - 设备
     - 操作者怎么用
     - 额外配置
   * - ``spacemouse``
     - 推动 6 自由度操作球来控制机械臂，用按键锁定夹爪状态。
     - 无
   * - ``gello``
     - 摆动主臂，从臂跟随到相同位姿。
     - ``gello_port``
   * - ``gello_joint``
     - 摆动主臂，从臂逐关节跟随；每条手臂各写一项。
     - ``left_gello_port`` / ``right_gello_port``
   * - ``pico``
     - 手持 VR 控制器，按住 grip 时接管机器人；双臂机器人每条手臂各写一项。
     - ``pico:`` 段
   * - ``glove``
     - 弯曲手指控制灵巧手，与驱动机械臂的设备配合使用。
     - ``glove_config:`` 段
   * - ``none``
     - 不接操作者设备，由策略独立控制。
     - 无

环境只接受与机器人控制通路匹配的设备。例如，双臂 Franka 没有单臂笛卡尔通路，配置
``spacemouse`` 会直接报错，不会被静默忽略；报错信息也会列出该环境接受的设备。

先单独检查设备
--------------

每种设备的 reader 都能脱离机器人和集群单独运行：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

主臂只报告零值或操作球没有响应时，这条命令可以把接线、权限问题与环境配置问题分开。
``toolkits/realworld_check`` 中的脚本则检查整台机器人，其中 ``check_robot_parts`` 会从组装一路走到断开。

单台设备跑通后再组合
----------------------

每台设备都能单独读取后，再把配置写成列表。每一项只生成自己能驱动的具名部件动作：

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

在这套灵巧手遥操组合中，操作球控制机械臂，数据手套控制手。只有按住 SpaceMouse 的第二个按键，
手套才会接管；松开后，手停在上一个位姿。

机器人有两个同类分支时，用 ``drives`` 指定每台设备驱动哪一支。配置中只有这里会出现
机器人部件名：

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

binding 声明的部件若不在机器人上，``TeleopGroup`` 会跳过该部件。如果一台设备最终匹配不到
任何部件，或者两台设备声明了同一个部件，构建遥操组合时都会报错。

把设备放到它插着的机器上
------------------------

遥操作设备也是部件，使用相同的放置模型。操作者的硬件与策略不在同一台机器上时，为设备
指定 ``node_rank``：

.. code-block:: python

   leader = TeleopLeaderArm.at("/dev/ttyUSB0", node_rank=1)

系统会并行读取连接互相独立的设备。同一台设备即使为两个部件生成动作也只打开一次；
SpaceMouse 同时控制机械臂和夹爪时，仍只占用一个 HID 句柄。

如果问题出在频率
----------------

主臂目标若只按策略步频下发，从臂的跟随会不稳定。直推模式把这条路径移到独立线程，以约
1 kHz 向控制器推送关节目标；``env.step`` 继续读取状态，但不再转发运动指令：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         teleop_direct_stream: true

只有跟随明显滞后时才需要直推模式。开启后，``env.step`` 不再下发关节目标；遥操配置若有
错误，机器人会保持静止，不会收到错误的运动指令。

已废弃的写法
------------

``teleop_device`` 只能指定一台设备，``use_spacemouse``、``use_gello``、
``use_gello_joint``、``use_pico`` 则各自开启一台。这些写法仍然可用，但会触发告警。
配置分层时，上层的 ``teleop`` 常常叠在仍用旧写法的基础配置上，此时以 ``teleop``
为准，告警里会说明它取代了哪些旧配置项。

下一步
------

- :doc:`机器人模型 <../concepts/robotics>`：设备、binding 和 group。
- :doc:`真机环境模型 <../concepts/realworld_envs>`：遥操作在 wrapper 栈里的位置。
- :doc:`数据采集 <data_collection>`：把操作者的动作记录下来。
