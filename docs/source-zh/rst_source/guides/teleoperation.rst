遥操作
======

配置好操作者要用的设备，人就能在 rollout 中途从策略手里接管：采示教数据、从失败里救
回来，或者跑 DAgger。这篇讲怎么在配置里选设备、几个设备怎么一起用、接机器人之前怎么
单独验一台，以及怎么把设备放到它插着的那台机器上。

背后的模型见 :doc:`../concepts/robotics`。

选一个设备
----------

一个配置项就够了：

.. code-block:: yaml

   env:
     eval:
       teleop_device: spacemouse

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - 设备
     - 操作者怎么用
     - 额外配置
   * - ``spacemouse``
     - 推动 6 自由度的操作球控制机械臂，按键控制夹爪开合。
     - 无
   * - ``gello``
     - 手动摆主臂，从臂跟着走到同一个位姿。
     - ``gello_port``
   * - ``gello_joint``
     - 手动摆主臂，从臂逐关节跟随。
     - ``left_gello_port`` / ``right_gello_port``
   * - ``pico``
     - 握 VR 手柄，扳机表示此刻是否在操作。
     - ``pico:`` 段
   * - ``none``
     - 无人接管，只跑策略。
     - 无

不是每台机器人都能用每种设备。双臂 Franka 没有单臂笛卡尔通路，所以在那里写
``spacemouse`` 会直接报错，而不是被忽略。每个环境自己声明能接哪些，报错信息里会列出来。

几个设备一起用
--------------

一个设备只填它能驱动的部件，几个设备就能各填各的。把名字换成列表：

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

灵巧手那套装置就是这样：操作球管机械臂，数据手套管手，按住 SpaceMouse 的第二个键才
把控制权交给手套。松手之后，手就停在你摆好的位置。

机器人上有两个同类部件时，要说明每个设备驱动哪一支：

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

某个设备最后什么都没驱动，会在装置构建时报错，而不是等机器人动起来才发现；两个设备
抢同一个部件也一样。

先单独验一台设备
----------------

每种设备都能单独读，不需要机器人，也不需要集群：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

主臂读数为零、操作球没反应时先跑这个，能把接线和权限问题跟配置问题分开。
``toolkits/realworld_check`` 里的脚本对整台机器人做同样的事。

把设备放到它插着的机器上
------------------------

遥操作设备也是部件，和别的部件一样可以指定节点。操作者的硬件不在跑策略那台机器上时，
给它一个 ``node_rank``：

.. code-block:: python

   leader = TeleopLeaderArm.at("/dev/ttyUSB0", node_rank=1)

连接互相独立的设备会并行读取。同一个设备被两个部件用到时只打开一次，所以 SpaceMouse
同时管机械臂和夹爪时，不会自己跟自己抢 HID 句柄。

如果问题出在频率
----------------

从臂只在策略的步频上收到主臂的目标，跟随就会发飘。打开直推模式后，会有一个线程以约
1 kHz 把关节目标直接推给控制器，``env.step`` 仍然读状态，但不再转发运动指令：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         teleop_direct_stream: true

跟随没有明显发飘时就别开。开了之后 ``env.step`` 不再下发关节目标，所以装置配错的表现
是机器人不动，而不是动得不对。

已废弃的写法
------------

``use_spacemouse``、``use_gello``、``use_gello_joint``、``use_pico`` 仍然可用，但会告警。
如果一份配置里既有废弃开关又有 ``teleop_device``，而且两者指向不同设备，会直接报错：
悄悄挑一个，就等于在机器人已经动起来的时候把错的设备交到别人手里。

下一步
------

- :doc:`机器人模型 <../concepts/robotics>`：设备、binding 和 group。
- :doc:`真机环境模型 <../concepts/realworld_envs>`：遥操作在 wrapper 栈里的位置。
- :doc:`数据采集 <data_collection>`：把操作者的动作记录下来。
