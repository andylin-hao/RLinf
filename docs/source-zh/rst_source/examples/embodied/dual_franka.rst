双 Franka 真机：GELLO 数据采集、π₀.₅ SFT 与部署
====================================================

本指南是 RLinf 中 **双臂 Franka** 真机的端到端流程：双节点环境搭建、
1 kHz GELLO 关节空间双臂数据采集、π₀.₅ 在 20 维 tcp_rot6d 动作空间上的
SFT 微调，以及通过脚踏开关将训练好的策略部署回真机。

阅读本页前请先阅读：

* :doc:`franka` — 单臂 Franka 基础、Ray cluster 搭建、RealSense +
  SpaceMouse 数据采集路径。如果尚不熟悉 ``FrankaController`` /
  ``FCI`` / ``RLINF_NODE_RANK``，请先完整阅读该页。
* :doc:`franka_gello` — GELLO 硬件安装、Dynamixel SDK、
  ``gello-teleop`` 包、USB-FTDI 权限。


硬件拓扑
--------

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - 节点
     - 角色
     - 节点上的硬件
   * - **node 0**\ （head）
     - Ray head；env worker；左 ``FrankyController``；
       部署阶段的 actor / rollout；所有相机和 GELLO 采集
     - 1× GPU（如 RTX 4090，仅 SFT 与部署阶段使用）；
       左 Franka FR3 直连一张网卡，对接 FCI 端口；
       左 Robotiq 2F-85（USB-RS485 Modbus）；
       **左右两台 GELLO** Dynamixel 链（USB-FTDI）；
       **三台相机全部在此**\ —— base RealSense D435i（第三人称）+
       左腕 Lumos USB-3 + 右腕 Lumos USB-3；
       PCsensor 3 键脚踏（放在 node 0）
   * - **node 1**\ （worker）
     - Ray worker；只跑右 ``FrankyController``
     - 可选 GPU（推理不需要）；
       右 Franka FR3 直连自己的网卡，对接 FCI 端口；
       右 Robotiq 2F-85

.. note::

   两臂的 FCI IP 与网卡名按你机器实际网络情况填写到下文
   Hardware YAML 中即可。

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - 相机槽位
     - 后端
     - 用途
   * - ``base_0_rgb``
     - RealSense D435i
     - 第三人称视角，左右臂共用
   * - ``left_wrist_0_rgb``
     - Lumos USB 3（XVisio vSLAM）
     - 左臂腕相机，作为 π₀.₅ 主 ``image``
   * - ``right_wrist_0_rgb``
     - Lumos USB 3（XVisio vSLAM）
     - 右臂腕相机

脚踏（PCsensor 3 键 FootSwitch）必须接在 node 0，键码 ``a`` / ``b`` /
``c`` 用厂家 Windows 工具刷一次进固件。


安装（每个节点都执行）
----------------------

以下步骤需在 ``node 0`` 和 ``node 1`` 上**分别执行一次**。两个节点是
独立 checkout、独立 venv，只共享 LAN 网络。

1. 检查 Franka 固件版本
~~~~~~~~~~~~~~~~~~~~~~~~

在机器人管理网页（一般为 ``http://<robot_ip>/desk``）中，点击
``SETTINGS`` 选项卡，在 ``DashBoard`` 中查看 ``Control`` 后面的版本号，
如下所示。请记录该固件版本号，后续步骤会用到。

.. raw:: html

  <div style="flex: 1; text-align: center;">
      <img src="https://github.com/RLinf/misc/blob/main/pic/franka_firmware.png?raw=true" style="width: 60%;"/>
  </div>

确认固件版本号后到 Franka 官方 `compatibility matrix
<https://frankarobotics.github.io/docs/compatibility.html>`_ 查与该
固件兼容的 libfranka 版本，下一节 "RLinf + franky" 会用到。

2. PREEMPT_RT 内核与 rtprio 限额
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

按 `Setting up the real-time kernel
<https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html>`_
启动 PREEMPT_RT 内核（验证过 ``5.15.133-rt69``\ ）。验证：

.. code-block:: bash

   uname -a | grep -o PREEMPT_RT

把以下写进 ``/etc/security/limits.d/99-realtime.conf`` 然后重新登录：

.. code-block:: text

   *  -  rtprio    99
   *  -  memlock   unlimited

退出登录再重新登录让 PAM 重新读取限额；然后 ``ulimit -r`` 应当返回
``99`` 或 ``unlimited``\ ，``ulimit -l`` 应当返回 ``unlimited``\ 。
否则 ``FrankyController.__init__`` 会打印 ``SCHED_FIFO denied`` /
``mlockall failed`` 并 fallback 到默认调度——控制器仍能运行，但
RT 抖动会回来。

.. note::

   这些限额由
   ``rlinf/envs/realworld/franka/franky_controller.py`` 中的
   ``_apply_rt_hardening()`` 在启动时检查；如果 ``SCHED_FIFO``
   被拒绝或 ``mlockall`` 失败，控制器会以 best-effort 模式继续
   运行并打 warning，而不会直接退出，warning 文本里附带具体的
   修复指引。

3. 每次开机的 RT 调优
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done'
   sudo sysctl -w kernel.sched_rt_runtime_us=-1
   sudo ethtool -C eno1 rx-usecs 0 tx-usecs 0   # eno1 换成你的网卡

4. RLinf + franky
~~~~~~~~~~~~~~~~~

按 "1. 检查 Franka 固件版本" 一节查到的 libfranka 版本导出
``LIBFRANKA_VERSION``，然后跑安装脚本：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   # 按 "1. 检查 Franka 固件版本" 一节里查到的 libfranka 版本设这个变量。
   export LIBFRANKA_VERSION=0.15.0       # 或 0.19.0、...

   # 一次装齐：系统依赖（rt-tests, ethtool, eigen, pinocchio，由
   # install.sh 内部调 franky_install.sh 处理）+ RLinf Python 依赖
   # + 与 LIBFRANKA_VERSION 对应的 franky-control wheel。
   # 非 root 用户在系统依赖安装那一步会提示输 sudo 密码。
   bash requirements/install.sh embodied --env franka-franky --use-mirror
   source .venv/bin/activate

``--env franka-franky`` 走 franky 路径 —— 从
``Brunch-Life/franky`` fork 的 ``wheels-libfranka-<LIBFRANKA_VERSION>``
release 按当前 Python ABI 挑对应的 ``franky-control`` wheel
（cp39..cp314，x86_64 manylinux_2_28，**libfranka 内嵌在 wheel 里**），
**跳过** :doc:`franka` 使用的 ``serl_franka_controllers`` ROS / catkin
编译流。``--use-mirror`` 面向国内用户（自动切换 PyPI / GitHub /
HuggingFace 镜像）。

.. note::

   ``requirements/install.sh embodied --env franka-franky`` **一条命令
   搞定**：uv venv → 内部调 ``franky_install.sh`` 装系统级依赖
   （``rt-tests``、``ethtool``、``cmake``、``libeigen3-dev``、
   ``libpoco-dev``、``libfmt-dev``、pinocchio 等）→ 拉对应 libfranka
   版本的 ``franky-control`` wheel。**不需要单独跑** ``franky_install.sh``。

.. warning::

   **请避开 libfranka 0.18.0**。Franka 官方 0.18.0 release notes 标注
   了阻抗 / 笛卡尔控制路径的回归 bug。

5. GELLO（env worker 所在节点）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

两台 GELLO 的 USB-FTDI 都接 node 0，跨 LAN 走会破坏 1 kHz 实时性。

按 :doc:`franka_gello` 在 node 0 装 ``gello`` + ``gello-teleop`` +
USB-FTDI 权限到同一 venv。

6. 脚踏
~~~~~~~

PCsensor FootSwitch 用厂家 Windows 工具把 3 个踏板刷成键码 ``a`` /
``b`` / ``c``\ （写入固件，一次永久）。

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd       # 期望: usb-PCsensor_FootSwitch-event-kbd → ../eventXX
   sudo chmod 666 /dev/input/eventXX
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX   # 必须在 `ray start` 之前


硬件验证
--------

启动 Ray 之前每个节点单测各硬件。

相机
~~~~

.. code-block:: bash

   rs-enumerate-devices | grep -E "Name|Serial|USB Type"   # RealSense
   ls /dev/v4l/by-id/                                       # 两个 Lumos 节点
   lsusb -t                                                 # 期望 5000M，480M = USB-2 fallback

GELLO（找串口 + 验关节）
~~~~~~~~~~~~~~~~~~~~~~~~

每条 GELLO 对应 ``/dev/serial/by-id/usb-FTDI_..._<unique_id>-if00-port0``\ 。
分辨左右用拔插对照法：

.. code-block:: bash

   # 先只插左 → 记一遍；再插右 → 新出现的就是右
   ls /dev/serial/by-id/ | grep -i ftdi

把两条 by-id 写进
``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``
的 ``env.eval.left_gello_port`` / ``right_gello_port``\ 。

读数实时性：

.. code-block:: bash

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   python -m gello_teleop.gello_expert \
       --port /dev/serial/by-id/usb-FTDI_..._<LEFT_ID>-if00-port0

如果数值阻塞或突然跳 ±2π，跑下一节的 ``calibrate``\ 。右臂同上。

每台机械臂单独验
~~~~~~~~~~~~~~~~

每个节点对自己那台跑：

.. code-block:: bash

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   FRANKA_ROBOT_IP=172.16.0.2 \
   FRANKA_GRIPPER_TYPE=robotiq \
   FRANKA_GRIPPER_PORT=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<id>-if00-port0 \
       python toolkits/realworld_check/test_franky_controller.py

REPL 关键命令：``getjoint`` / ``home`` / ``hold 30``\ （静置听嗡鸣）/
``stream 4 0.001 500``\ （1 kHz preemption 压测）/ ``open`` / ``close``\ 。

两边都通过前不要启动 Ray。


GELLO 标定
----------

每台 GELLO 都要单独标定一次（更换电机后再标）。在左右臂上的标定结果
一致，**两台 GELLO 都在 node 0 直连的左臂上做即可**。

对每台 GELLO 按 "calibrate → align-sequential 验证" 两步走一遍；第一台
通过后把 ``GELLO_PORT`` 改成第二台的 by-id 路径，再跑一遍同样的两步。

1. **标定**：

   .. code-block:: bash

      export PYTHONPATH=$PWD:${PYTHONPATH:-}
      export GELLO_PORT=/dev/serial/by-id/usb-FTDI_..._<ID>-if00-port0
      python toolkits/realworld_check/test_gello.py calibrate

   脚本会将机器人安全地依次移动到两个已知姿态（``POSE_A`` =
   Franka 原点，``POSE_B`` = π/4 倍数），让操作员将 GELLO 各
   摆成相同姿态，然后从两次差值解出 ``joint_signs`` 和
   ``joint_offsets``，最后打印一段可直接粘贴到
   ``gello_software/gello/agents/gello_agent.py`` 的
   ``DynamixelRobotConfig`` 块::

       "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_<id>-if00-port0":
           DynamixelRobotConfig(
               joint_ids=(1, 2, 3, 4, 5, 6, 7),
               joint_offsets=(...),
               joint_signs=(...),
               gripper_config=(8, ..., ...),
           ),

   python toolkits/realworld_check/test_gello.py calibrate

2. **对齐验证**：标完直接跑 align-sequential 走一遍 J1 → J7，确认每个
   关节都能稳稳进 ±0.10 rad 容差。如果某关节始终对不上 / 残差超过预期，
   回到上一步重标。

**对齐**\ （leader 跟机械臂位姿对不上时跑）：

      export PYTHONPATH=$PWD:${PYTHONPATH:-}
      python toolkits/realworld_check/test_gello.py align-sequential

   python toolkits/realworld_check/test_gello.py align-sequential

两个脚本都会用 glob ``/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_*-if00-port0``
自动找到本机的 Robotiq 串口（node 0 上接的就是左臂的 Robotiq），不需要
手动配置。


硬件 YAML
---------

双 Franka 的硬件配置写在
``examples/embodiment/config/env/realworld_dual_franka_joint.yaml``
（采集）和
``examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml``
（tcp_rot6d 部署）。参考这两份示例改即可，需要按本机替换的占位符：

* ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP`` —— 左右臂 FCI IP（如
  ``172.16.0.2``）。
* ``BASE_CAMERA_SERIAL`` —— base 相机 serial（RealSense 用
  ``rs.context().devices`` 报告的；按 ``base_camera_type`` 后端
  改成对应 SDK 的 serial）。
* ``LEFT_CAMERA_SERIAL`` / ``RIGHT_CAMERA_SERIAL`` —— 两腕相机
  serial（Lumos 用 ``/dev/v4l/by-id/usb-XVisio_..._video-index0``
  路径；按 ``*_camera_type`` 后端改）。
* ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION``
  —— Robotiq 2F-85 的 RS-485 串口，固定用
  ``/dev/serial/by-id/usb-FTDI_..._<id>-if00-port0``，**不要**
  用 ``/dev/ttyUSB*``\ （重启 / 热插拔后会换号）。
* ``LEFT_GELLO_PORT`` / ``RIGHT_GELLO_PORT`` —— GELLO 主手的
  ``/dev/serial/by-id`` 路径（两个都插在 env worker 所在节点，
  即 ``node_rank: 0``）。
* override 段内的 ``ee_pose_limit_min`` / ``ee_pose_limit_max``
  —— 按本机工作空间安全箱调；行 0 是左臂、行 1 是右臂，每行
  ``[x, y, z, roll, pitch, yaw]``。

``left_controller_node_rank`` / ``right_controller_node_rank``
（默认 ``0`` / ``1``\ ，每节点各管一台）和 ``node_rank``\ （env
worker + 相机所在节点）通常不用改。


Ray cluster 启动
-----------------

Ray 在 ``ray start`` 时快照当前已 export 的环境变量，未 export 的
worker 永远拿不到。先 export 完再 ``ray start``。

.. code-block:: bash

   # node 0（head）
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

   ray stop --force
   ray start --head --port=6379 --node-ip-address=<HEAD_IP>

.. code-block:: bash

   # node 1（worker）
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=1
   ray stop --force
   ray start --address=<HEAD_IP>:6379 --node-ip-address=<WORKER_IP>

node 0 验证 ``ray status`` 两个节点 ALIVE。

.. warning::

   两个节点是独立 checkout。node 0 改完代码要
   ``rsync -av --delete RLinf/ <node1>:/path/to/RLinf/`` 再在
   node 1 重启 Ray。否则会出现 worker ImportError 或不一致行为。


数据采集（GELLO 关节空间）
--------------------------

env = ``DualFrankaJointEnv-v1``\ ；``teleop_direct_stream: true`` 开
1 kHz 守护线程读 GELLO 直接推到 ``FrankyController``\ ；``env.step``
以 10 Hz 只读 state + 抓相机帧，不发 motion。这样数据集记录的是 1 kHz
真实操作员动作而非 100 ms 网格截断后的轨迹。

配置
~~~~

``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``\ ，
常改字段：

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 字段
     - 含义
   * - ``runner.num_data_episodes``
     - 总目标（resume 时跨会话累计）。
   * - ``env.eval.left_gello_port`` / ``right_gello_port``
     - 临时换 GELLO 单元时覆盖。
   * - ``env.eval.override_cfg.task_description``
     - 每帧 ``task`` 字段的 prompt。
   * - ``env.eval.override_cfg.joint_action_mode``
     - 采集用 ``absolute``\ 。
   * - ``env.eval.override_cfg.teleop_direct_stream``
     - 必须 ``true``\ 。
   * - ``data_collection.save_dir``
     - 数据集根目录，多次会话指同一目录可累积。
   * - ``data_collection.resume``
     - ``true`` 从已有 ``id_*`` shard 继续。

启动
~~~~

Ray 起来后开两个终端（都在 node 0）：

.. code-block:: bash

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   bash examples/embodiment/collect_data.sh \
        realworld_collect_data_gello_joint_dual_franka 2>&1 | tee logs/collect.log

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   python toolkits/realworld_check/collect_monitor.py logs/collect.log

monitor 单独存在是因为 Ray 的 log monitor 缓冲会破坏 tqdm 原位刷新；
它单独 tail 日志渲染干净的进度条 + 脚踏事件 + 最近 reward。

每个 episode 的工作流
~~~~~~~~~~~~~~~~~~~~~~

确认 ``align-sequential`` 报告 ``ALL JOINTS ALIGNED`` 之后：

1. (pre) reset 跳过 home，机械臂保持在操作员当前位置。
2. 踩 ``a`` —— 开始录第 0 帧。
3. 演示任务。机械臂 1 kHz 跟 GELLO，相机 10 Hz 抓帧。
4. 踩 ``b`` —— ``segment_id`` +1（1 s 防抖），标 approach/grasp/...
5. 踩 ``c`` —— 成功：reward=1.0、写 LeRobot shard。
6. 录制中再踩 ``a`` —— 中止：丢 buffer，回 pre，机械臂不 home。

输出格式
~~~~~~~~

LeRobot v2.1，每会话一个 shard：``<save_dir>/rank_0/id_{N}/``\ 。
``meta/info.json`` 里 joint 数据为 ``state=[68]``\ 、\ ``actions=[16]``\ ，
tcp_rot6d 数据为 ``state=[20]``\ 、\ ``actions=[20]``\ 。

关键帧字段：

* ``state`` ——
  ``[L_grip, R_grip, joint_position(14), joint_velocity(14),
  tcp_force(6), tcp_pose(14), tcp_torque(6), tcp_vel(12)]`` = 68
* ``actions`` —— joint 模式
  ``[L_jpos(7), L_grip, R_jpos(7), R_grip]``
* ``image`` —— ``left_wrist_0_rgb``\ （主图像）
* ``extra_view_image-0`` / ``-1`` —— **顺序锁死**
  ``(base_0_rgb, right_wrist_0_rgb)``\ ，重命名会断言报错
* ``is_success`` —— 整条 episode 都为 True 当且仅当踩 ``c`` 结束
* ``segment_id`` —— uint8，踩 ``b`` 自增

断点续采
~~~~~~~~

``data_collection.resume: true`` + 原 ``save_dir`` 重跑：扫已有
``id_*`` shard 累计计数，新会话写到新 ``id_{N}``\ 。
``num_data_episodes`` 是跨会话累计目标。


回填 tcp_rot6d 与 norm_stats
-----------------------------

采集的是 16 维 joint actions + 68 维 joint-env state；π₀.₅ SFT 要求
20 维 tcp_rot6d state/actions（actions 为
``[xyz(3) + rot6d(6) + grip(1)] × 2``\ ）。先离线回填，再算 norm_stats。

``<repo_id>`` 是数据集相对 ``HF_LEROBOT_HOME`` 的路径，
``joint_v1`` / ``tcp_rot6d_v1`` 是版本子目录。

.. code-block:: bash

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   python toolkits/dual_franka/backfill_tcp_rot6d.py \
       --src $HF_LEROBOT_HOME/<repo_id>/joint_v1 \
       --dst $HF_LEROBOT_HOME/<repo_id>/tcp_rot6d_v1

回填脚本：从已有 tcp_pose 列切出并重建 20 维 tcp_rot6d state（不跑
FK）；actions 扩到 20 维，xyz/rot6d 用**下一帧** tcp_pose 当当前目标，
gripper 槽位沿用原信号。重复回填会直接报错。

.. code-block:: bash

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_dualfranka_tcp_rot6d \
       --repo-id <repo_id>/tcp_rot6d_v1

输出 ``<openpi_assets_dirs>/pi05_dualfranka_tcp_rot6d/<repo_id>/norm_stats.json``\ 。
**回填后才算**\ ，否则 stats 对的是绝对目标而非 body-frame delta。


SFT（π₀.₅，tcp_rot6d_v1）
-------------------------

SFT 跑在远端 GPU 训练集群上，不在 node 0 / node 1。把 tcp_rot6d 数据集
推到训练集群、在那边训练、把 ckpt + norm_stats 拉回 node 0 给部署用。

1. 推数据集（node 0）：

   .. code-block:: bash

      rsync -av $HF_LEROBOT_HOME/<repo_id>/tcp_rot6d_v1/ \
          <train>:$HF_LEROBOT_HOME/<repo_id>/tcp_rot6d_v1/

2. 装环境 + 训练（在 ``<train>``\ ）。开训前在
   ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``
   里把 ``runner.logger.wandb_entity`` 和
   ``cluster.component_placement`` 改成你的值。

   .. code-block:: bash

      # 安装脚本要求填 --env，不实际使用。
      bash requirements/install.sh embodied --model openpi --env maniskill_libero --use-mirror
      source .venv/bin/activate

      export PYTHONPATH=$PWD:${PYTHONPATH:-}
      export HF_LEROBOT_HOME=/path/to/lerobot_root
      export DUAL_FRANKA_DATA_ROOT=$HF_LEROBOT_HOME/<repo_id>/tcp_rot6d_v1
      export PI05_BASE_CKPT=/path/to/pi05/torch

      python toolkits/lerobot/calculate_norm_stats.py \
          --config-name pi05_dualfranka_tcp_rot6d \
          --repo-id <repo_id>/tcp_rot6d_v1
      mkdir -p $PI05_BASE_CKPT/<repo_id>
      mv <openpi_assets_dirs>/pi05_dualfranka_tcp_rot6d/<repo_id>/norm_stats.json \
         $PI05_BASE_CKPT/<repo_id>/norm_stats.json

      bash examples/sft/run_vla_sft.sh realworld_sft_openpi_dual_franka_tcp_rot6d

   Ckpt 落在
   ``<log_path>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``\ 。

3. 把 ckpt + norm_stats 拉回 node 0：

   .. code-block:: bash

      CKPT=<train_log>/checkpoints/global_step_<N>
      mkdir -p $CKPT/actor/model_state_dict $CKPT/<repo_id>
      rsync -av <train>:$CKPT/actor/model_state_dict/full_weights.pt \
          $CKPT/actor/model_state_dict/full_weights.pt
      rsync -av <train>:$PI05_BASE_CKPT/<repo_id>/norm_stats.json \
          $CKPT/<repo_id>/norm_stats.json

   部署时读 ``$CKPT/actor/model_state_dict/full_weights.pt`` 和
   ``$CKPT/<repo_id>/norm_stats.json``\ 。

真机部署
--------

跟采集用同一套 Ray cluster，换入口脚本 + 配置。

安装
~~~~

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env franka-franky --use-mirror
   source .venv/bin/activate

配置
~~~~

``examples/embodiment/config/realworld_eval_dual_franka.yaml``。
占位符标注为 ``# Replace:``。最常修改的字段：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 字段
     - 设成
   * - ``rollout.model.model_path``
     - ``<sft_log>/checkpoints/global_step_<N>/`` —— 必须包含
       ``actor/model_state_dict/full_weights.pt`` 和
       ``<data_config.repo_id>/norm_stats.json`` （
       ``data_config.repo_id`` 怎么算见下文"ckpt / norm_stats
       锁步"）。
   * - ``actor.model.openpi_data.repo_id``
     - 作为 ``data_kwargs`` 传给 ``get_openpi_config`` ，会覆盖
       ``data_config.repo_id`` ；这个 ``repo_id`` 就是部署时
       ``norm_stats.json`` 的查找 key。和
       ``calculate_norm_stats.py --repo-id`` 时给的值保持一致。
   * - ``env.eval.override_cfg.task_description``
     - 跟训练 prompt 一致。
   * - ``env.eval.override_cfg.target_ee_pose``
     - 跟采集时的 workspace 对齐。

硬件 ``configs`` 与采集 yaml 完全一致 —— 同 IP、同相机 serial、
同 gripper 串口。Wrapper 是按 ``env.eval.use_*`` flag 装的，所以
采集 vs 部署的 yaml 差别只有 2 个：

* ``use_gello_joint: false``\ （采集是 ``true``）
* ``keyboard_reward_wrapper: eval_control``\ （采集是 ``start_end``）

启动
~~~~

.. code-block:: bash

   # node 0：先把代码同步到 node 1
   rsync -av --delete --exclude=results --exclude='.venv*' --exclude=.git \
       --exclude=__pycache__ --exclude='*.pyc' --exclude=wandb \
       ./ <node1>:/path/to/RLinf/

.. code-block:: bash

   # node 0（head）
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

   ray stop --force
   ray start --head --port=6379 --node-ip-address=<HEAD_IP>

.. code-block:: bash

   # node 1（worker）
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=1
   ray stop --force
   ray start --address=<HEAD_IP>:6379 --node-ip-address=<WORKER_IP>

.. code-block:: bash

   # node 0
   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka

   # Hydra override 示例：
   #   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka \
   #        rollout.model.model_path=/sft/global_step_5000 \
   #        env.eval.override_cfg.task_description="pour water"

每个 episode 的部署工作流
~~~~~~~~~~~~~~~~~~~~~~~~~

``KeyboardEvalControlWrapper`` 把脚踏 wrapper 切成自主推理模式：

1. ``env.reset()`` 之后两台机械臂保持在 reset 位姿。``env.step()``
   被截到 **idle** 模式 —— 不向内层 env 转发（impedance 控制器
   保持上一次 reset 时的目标，机械臂原地静止），但 wrapper 仍会把
   最近一次 obs 返回，让策略的 chunked rollout 循环空转，不下发
   任何关节指令。
2. 踩下 ``a`` —— wrapper 切到 **running**。下一步 ``env.step``
   开始向内层 env 转发策略输出。
3. 踩下 ``c`` —— 成功：``terminated=True``、``reward=1.0``、
   ``info["eval_result"]="success"``。Wrapper 内部立刻调
   ``env.reset()`` 让机械臂回 home，然后回到 idle 等下一次 ``a``
   —— 这是脚踏可连续操作的关键，即使 eval ``env_worker``
   是 ``auto_reset=False``。
4. 踩下 ``b`` —— 失败：行为同 ``c``，但 ``reward=0.0``、
   ``info["eval_result"]="failure"``。
5. running 阶段，wrapper 强制把 ``terminated`` / ``truncated``
   置 False，除非脚踏触发 —— env 自己的 ``max_episode_steps``
   不会切断策略。把 ``max_episode_steps`` 设大一点（仓库 yaml
   是 ``10000``），让脚踏始终是边界 owner。

ckpt / norm_stats 锁步
~~~~~~~~~~~~~~~~~~~~~~~

Rollout 读
``<rollout.model.model_path>/<actor.model.openpi_data.repo_id>/norm_stats.json``\ 。
``repo_id`` 必须跟 SFT 时的 ``calculate_norm_stats.py --repo-id`` 一致，
否则会走 fallback 路径并打 warning：
``"norm_stats fallback: ... verify they match training or inference
will be wrong"``\ 。不匹配不会崩，但策略会塌缩成固定动作。

启动前自检：

.. code-block:: bash

   find <model_path> -maxdepth 3 -name norm_stats.json
   ls <model_path>/actor/model_state_dict/full_weights.pt


故障排查
--------

**GELLO 守护线程未启动**
   GELLO 重新上电、FTDI 重插，然后用
   ``python -m gello_teleop.gello_expert --port /dev/...`` 验证
   两侧都能持续输出 Dynamixel 读数。

**Ray worker 静默死在 import**
   在跑 ``ray start`` 的同一 shell 里执行
   ``which python && python -c "import franky, gello, gello_teleop"``
   确认 venv 和已装包一致；具体报错看
   ``/tmp/ray/session_latest/logs/worker-*.err``。

**有一台机械臂 reset 时挂住**
   在 controller 节点 ``ping -c 100 <robot_ip>``，若丢包就重启
   该机械臂再跑。

**开机后 ``move_joints`` 一直报错**
   释放白色急停按钮 → Desk 网页（\ ``http://<robot_ip>/desk/``\ ）
   点 *Activate FCI* → 等关节 LED 由白转蓝 → 再启动。

**GELLO 守护线程和 env reset 互相 race**
   reset 期间把 GELLO leader 放稳在支架上，等
   ``KeyboardStartEndWrapper`` 报告 reset 结束再继续操作。

**脚踏报 "Permission denied"**
   ``sudo chmod 666 /dev/input/eventXX``；要持久化就写 udev rule
   （``KERNEL=="event*", SUBSYSTEM=="input",
   ATTRS{name}=="PCsensor FootSwitch", MODE="0666"``）。

**RealSense 退到 USB 2.x**
   换 USB 线缆，插到主板的蓝色 USB-3 端口，``lsusb -t`` 确认显示
   ``5000M`` 而不是 ``480M``。

**Lumos 冷启动第一次失败**
   重新插拔 USB 线。

**部署时 idle 一直不响应**
   确认 ``RLINF_KEYBOARD_DEVICE`` 指向正确的
   ``/dev/input/eventXX`` 且 ``chmod 666`` 仍生效，然后踩 ``a``
   触发。

**部署阶段跟踪抖动**
   降 ``RLINF_CART_K_R``、提高 ``RLINF_CART_GAINS_TC``、把
   ``RLINF_CART_MAX_STEP_RAD`` 收紧；仍不行就缩短策略 chunk
   长度。

**部署时找不到 ``norm_stats.json``**
   把 ``calculate_norm_stats.py`` 写出的
   ``<openpi_assets_dirs>/<repo_id>/`` 复制到
   ``<model_path>/<repo_id>/``；先 grep
   ``"norm_stats fallback"`` warning 判断是否走到 fallback 路径。

**collect_monitor 无进展**
   launcher 加 ``2>&1 | tee logs/collect.log``；env worker 在另一
   节点时给 monitor 加 ``--source=worker``。

**controller 启动时输出 ``sched_setaffinity failed`` warning**
   换 6+ 核机器，或对 venv 解释器执行
   ``sudo setcap cap_sys_nice=eip $(which python)``。

**reset 时两台机械臂都动了，但之后只有一根跟踪 GELLO**
   每台 GELLO 单独跑
   ``python toolkits/realworld_check/test_gello.py align-check``
   确认都在持续输出读数，再重启。
