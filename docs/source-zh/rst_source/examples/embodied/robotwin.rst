基于RoboTwin评测平台的强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://robotwin-platform.github.io/assets/images/teaser.png
   :align: center
   :width: 90%

   RoboTwin 2.0 的双臂操作任务（图片来源：`RoboTwin <https://robotwin-platform.github.io>`__）。

`RoboTwin 2.0 <https://robotwin-platform.github.io>`__ 是一个任务丰富、规模可观的双臂操作基准。
RLinf 提供 ``RoboTwinEnv`` 环境，在其上对视觉-语言-动作（VLA）策略进行强化学习微调，常常能把较弱的
SFT 检查点提升为接近饱和的策略。

概览
----

在 RoboTwin 2.0 上对 VLA 进行强化学习微调；OpenVLA-OFT + GRPO 平均任务成功率提升约 +57%。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅ · Lingbot-VLA

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO · DAgger

   .. grid-item-card:: 任务
      :text-align: center

      46 个双臂操作任务

   .. grid-item-card:: 硬件
      :text-align: center

      1–2 节点 · 8–16 张 GPU

| **你将完成：** 安装依赖 → 克隆 RoboTwin 与资产 → 下载 SFT 模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · RoboTwin 仓库与资产 · 一个 SFT 检查点（见下文步骤）。

任务
~~~~

``RoboTwinEnv`` 基于 RoboTwin 2.0 仿真平台，目前支持 46 个操作任务，可按需选择任务进行训练。

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - 类别
     - 任务
     - 描述
   * - 放置类任务
     - ``adjust_bottle``
     - 使用正确的手臂将桌上的瓶子拾起并保持瓶口朝上
   * - 
     - ``place_a2b_left``
     - 使用合适的手臂将物体 A 放置在物体 B 的左侧
   * - 
     - ``place_a2b_right``
     - 使用合适的手臂将物体 A 放置在物体 B 的右侧
   * - 
     - ``place_bread_basket``
     - 若桌上有一个面包，用单臂抓取并放入篮子；若有两个面包，用双臂同时抓取并放入篮子
   * - 
     - ``place_bread_skillet``
     - 用单臂抓取桌上的面包并放入平底锅
   * - 
     - ``place_burger_fries``
     - 使用双臂抓取汉堡和薯条并放置到托盘上
   * - 
     - ``place_can_basket``
     - 一只手臂将易拉罐放入篮子，另一只手臂提起篮子
   * - 
     - ``place_cans_plasticbox``
     - 使用双臂将易拉罐抓取并放入塑料箱
   * - 
     - ``place_container_plate``
     - 将容器放置到盘子上
   * - 
     - ``place_empty_cup``
     - 使用单臂将空杯放置到杯垫上
   * - 
     - ``place_mouse_pad``
     - 抓取鼠标并放置到彩色垫子上
   * - 
     - ``place_object_basket``
     - 一只手臂将目标物体放入篮子，另一只手臂抓起篮子并向外移动
   * - 
     - ``place_object_stand``
     - 使用合适的手臂将物体放置到支架上
   * - 
     - ``place_phone_stand``
     - 抓取手机并放置到手机支架上
   * - 
     - ``place_shoe``
     - 使用单臂从桌上抓取鞋子并放到垫子上
   * - 
     - ``place_dual_shoes``
     - 使用双臂抓取两只鞋并放入鞋盒，且鞋头朝左
   * - 抓取类任务
     - ``pick_dual_bottles``
     - 用双臂分别抓取两个瓶子
   * - 
     - ``pick_diverse_bottles``
     - 用双臂分别抓取两个不同的瓶子
   * - 
     - ``move_can_pot``
     - 用单臂抓取易拉罐并移动到锅旁
   * - 
     - ``move_pillbottle_pad``
     - 用单臂抓取药瓶并放到垫子上
   * - 
     - ``move_playingcard_away``
     - 抓取扑克牌并将其朝远离桌面的方向移动
   * - 
     - ``move_stapler_pad``
     - 使用合适的手臂将订书机移动到彩色垫子上
   * - 
     - ``grab_roller``
     - 使用双臂抓取桌上的滚轴
   * - 
     - ``lift_pot``
     - 使用双臂抬起锅
   * - 
     - ``put_bottles_dustbin``
     - 抓取瓶子并放入桌子左侧的垃圾桶
   * - 堆叠类任务
     - ``stack_blocks_two``
     - 将绿色积木堆叠在红色积木上
   * - 
     - ``stack_blocks_three``
     - 将蓝色积木叠在绿色积木上，再将绿色积木叠在红色积木上
   * - 
     - ``stack_bowls_two``
     - 将两个碗上下堆叠
   * - 
     - ``stack_bowls_three``
     - 将三个碗上下堆叠
   * - 排序类任务
     - ``blocks_ranking_rgb``
     - 按红、绿、蓝顺序从左到右排列积木
   * - 
     - ``blocks_ranking_size``
     - 将积木从左到右按由大到小排列
   * - 使用工具类任务
     - ``click_alarmclock``
     - 按下闹钟顶部中央按钮
   * - 
     - ``click_bell``
     - 按下铃铛顶部中央
   * - 
     - ``beat_block_hammer``
     - 抓起锤子敲击积木
   * - 
     - ``open_microwave``
     - 用单臂打开微波炉
   * - 
     - ``press_stapler``
     - 用单臂按压订书机
   * - 
     - ``stamp_seal``
     - 抓取印章并盖在指定颜色的垫子上
   * - 
     - ``turn_switch``
     - 用机械臂拨动开关
   * - 交接类任务
     - ``handover_block``
     - 左臂抓取红色积木并交接给右臂，随后放置到蓝色垫子上
   * - 
     - ``handover_mic``
     - 单臂抓取麦克风并交接给另一只手臂
   * - 倾倒、投掷与摇晃任务
     - ``shake_bottle``
     - 使用合适的手臂摇晃瓶子
   * - 
     - ``shake_bottle_horizontally``
     - 使用合适的手臂水平摇晃瓶子
   * - 
     - ``dump_bin_bigbin``
     - 抓取小箱并将其中物体倒入大箱中
   * - 悬挂与特殊任务
     - ``hanging_mug``
     - 左臂抓取杯子并调整姿态，右臂再次抓取并将杯子挂到挂架上
   * - 
     - ``scan_object``
     - 一只手臂持扫描器，另一只手臂持物体并完成扫描
   * - 
     - ``rotate_qrcode``
     - 抓取二维码板并旋转，使二维码朝向机器人

.. note::
   目前有四个任务尚未支持，分别是 ``place_fan``， ``open_laptop``， ``place_object_scale`` 和 ``put_object_cabinet`` 。另外，dense reward 奖励函数还在开发中，后续将逐步扩展到所有任务。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - ``images``
     - 头部相机 RGB，``[B, 224, 224, 3]`` uint8（中心裁剪）。
   * - ``wrist_images``
     - 可选的左/右腕部相机 RGB，``[B, n, 224, 224, 3]`` uint8，或 ``None``。
   * - ``states``
     - 本体感知，``[B, 14]`` float32（末端执行器位姿；``proprio_dim=14``）。
   * - ``task_descriptions``
     - 自然语言任务描述，长度为 ``B``。
   * - 动作 (Action)
     - 14 维连续 ``float32``：3D 位置 + 3D 旋转 + 1D 夹爪 + 7D 关节位置。

安装
----

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 如果希望在中国大陆更快地下载，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

RLinf 提供了预配置的 RoboTwin 环境 Docker 镜像，镜像中已包含所有必需的依赖，可以直接使用，**无需进行后续安装步骤**。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-robotwin
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-robotwin

.. note::
   Docker 镜像已包含：
   
   - RLinf RoboTwin 环境相关依赖
   - 兼容性补丁已应用
   - 支持 OpenVLA-OFT、OpenPI 模型

   **使用 Docker 镜像后，可以直接跳转到** `RoboTwin 代码克隆 和 Assets 下载`_ **，** `模型下载`_ **和** `运行脚本`_ **章节，无需进行后续安装步骤。**

**选项 2：自建环境**

在本地环境直接安装依赖，运行以下命令。根据要训练的模型，将 ``--model openvla-oft`` 参数替换为对应的模型名称（``openvla-oft`` 或 ``OpenPI``）：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openvla-oft --env robotwin
   source .venv/bin/activate

该脚本会自动完成：

- 安装 RLinf RoboTwin 环境相关依赖
- 应用 RoboTwin 兼容性补丁（修复 sapien 和 mplib 的兼容性问题）
- 安装对应 VLA 模型的依赖包


RoboTwin 代码克隆 和 Assets 下载
-----------------------------------------

RoboTwin Assets 是 RoboTwin 环境所需的资产文件，需要从 HuggingFace 下载。

.. code-block:: bash

   # 1. 克隆 RoboTwin 仓库
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
   
   # 2. 下载并解压 Assets 文件
   bash script/_download_assets.sh


模型下载
-----------------------

在开始训练之前，您需要下载相应的SFT模型：

.. code-block:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup --local-dir RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

下载后，请确保在配置 yaml 文件中正确指定模型路径（``actor.model.model_path``）。

运行脚本
-------------------

请确保您在运行下面的命令前已激活正确的 Python 虚拟环境（venv）。
如果您使用的是官方 Docker 镜像，请根据模型类型切换环境：

- OpenVLA-OFT：``source switch_env openvla-oft``
- OpenPI（π\ :sub:`0`\ / π\ :sub:`0.5`\ ）：``source switch_env OpenPI``

**1. 关键参数配置**

**1.1 OpenVLA-OFT + GRPO**

以 OpenVLA-OFT 模型为例，在 ``actor.model`` 中需要配置以下关键参数：

.. code-block:: yaml

   actor:
     model:
       model_path: "/path/to/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup"  # SFT 模型路径
       model_type: "openvla_oft"                                             # 模型类型设置为openvla_oft
       implement_version: "official"                                         # openvla_oft实现版本（RLinf OpenVLA-OFT模型的实现接入了oft官方版本和rlinf sft微调版本，RoboTwin环境使用官方版本）
       action_dim: 14                                                        # RoboTwin 动作维度（14维）
       use_proprio: True                                                     # 是否使用本体感觉信息
       proprio_dim: 14                                                       # 本体感觉维度
       use_film: False                                                       # 是否使用 FiLM 层
       num_images_in_input: 1                                                # 输入图像数量
       num_action_chunks: 25                                                 # 动作块数量
       unnorm_key: "place_empty_cup"                                         # 动作归一化键（需与SFT训练时使用的unnorm_key一致）

**1.2** :math:`\pi_0` **+ PPO**

RoboTwin 中的 π\ :sub:`0`\ + PPO 训练推荐沿用 OpenPI 的 RoboTwin 配置，并切换为 actor-critic 形式：

.. code-block:: yaml

   actor:
     model:
      model_path: "/path/to/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
      num_action_chunks: 50 # interface for the env
      add_value_head: True
      action_dim: 14
      OpenPI:
         config_name: "pi0_aloha_robotwin"
         num_images_in_input: 3
         detach_critic_input: True


**1.3** :math:`\pi_0.5` **+ PPO**

π\ :sub:`0.5`\ 在 RoboTwin 中已提供现成的 PPO 训练配置，示例如下：

.. code-block:: yaml

   actor:
      model:
         model_path: "/path/to/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
         num_action_chunks: 50 # interface for the env
         action_dim: 14
         add_value_head: True
         OpenPI:
            config_name: "pi05_aloha_robotwin"
            num_images_in_input: 3
            detach_critic_input: True


**2. 环境配置**

在环境配置文件中，需要设置以下关键参数：

.. code-block:: yaml

   env/train: robotwin_place_empty_cup
   env/eval: robotwin_place_empty_cup
   
   # 在 env/train/robotwin_place_empty_cup.yaml 中：
   env_type: robotwin
   assets_path: "/path/to/robotwin_assets"
   
   task_config:
     task_name: place_empty_cup  # 或其他任务名称
     step_lim: 200
     embodiment: [piper, piper, 0.6]
     camera:
       head_camera_type: D435
       wrist_camera_type: D435
       collect_head_camera: true
       collect_wrist_camera: false

对于 OpenPI（π\ :sub:`0`\ / π\ :sub:`0.5`\ ）配置，还需要额外注意：

- ``env.train.center_crop: False`` 和 ``env.eval.center_crop: False``：关闭中心裁剪
- ``env.*.task_config.embodiment: [aloha-agilex]``：切换到 AgileX 机器人配置
- ``env.*.task_config.camera.collect_wrist_camera: true``：启用腕部相机输入

**3. 配置文件**

RoboTwin 当前可直接参考的配置文件如下：

- **OpenVLA-OFT + GRPO**：``examples/embodiment/config/robotwin_place_empty_cup_grpo_openvlaoft.yaml``
- **π₀ + PPO**：``examples/embodiment/config/robotwin_adjust_bottle_ppo_OpenPI.yaml``
- **π₀ Eval**：``examples/embodiment/config/robotwin_adjust_bottle_ppo_OpenPI_eval.yaml``
- **π₀.₅ + PPO**：``examples/embodiment/config/robotwin_adjust_bottle_ppo_OpenPI_pi05.yaml``
- **π₀.₅ Eval**：``examples/embodiment/config/robotwin_adjust_bottle_ppo_OpenPI_pi05_eval.yaml``

**4. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   # 设置ROBOT_PLATFORM环境变量
   export ROBOT_PLATFORM=ALOHA
   # 设置ROBOTWIN_PATH环境变量
   export ROBOTWIN_PATH=/path/to/RoboTwin

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，在 RoboTwin 环境中使用 GRPO 训练 OpenVLA-OFT 模型：

.. code-block:: bash

   # 设置ROBOT_PLATFORM环境变量
   export ROBOT_PLATFORM=ALOHA
   # 设置ROBOTWIN_PATH环境变量
   export ROBOTWIN_PATH=/path/to/RoboTwin

   bash examples/embodiment/run_embodiment.sh robotwin_place_empty_cup_grpo_openvlaoft

例如，使用 PPO 训练 π\ :sub:`0.5`\ 模型：

.. code-block:: bash

   export ROBOT_PLATFORM=ALOHA
   export ROBOTWIN_PATH=/path/to/RoboTwin

   bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_ppo_OpenPI_pi05

.. admonition:: 进一步配置
   :class: note

   - 放置与吞吐 → :doc:`放置 </rst_source/tutorials/usage/placement>` 与 :doc:`执行模式 </rst_source/tutorials/usage/execution_modes>`
   - 全部配置项 → :doc:`配置 </rst_source/tutorials/configuration/index>`
   - 指标定义与日志后端 → :doc:`训练指标 </rst_source/tutorials/configuration/metrics>`
   - 从检查点恢复 → :doc:`断点续训 </rst_source/tutorials/configuration/resume>`
   - 卡住或显存不足（OOM）？ → :doc:`FAQ </rst_source/faq>`

可视化与结果
------------

启动 TensorBoard 实时查看训练：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

最值得关注的指标是 **``env/success_once``**。每个日志指标的含义见
:doc:`训练指标 </rst_source/tutorials/configuration/metrics>`。

训练和评估过程中的视频会自动保存。配置如下：

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train  # 训练视频
     # 或
     video_base_dir: ${runner.logger.log_path}/video/eval   # 评估视频

评估结果
~~~~~~~~~~~~~~~~~~~

.. list-table:: **OpenVLA-OFT 模型在七个 RoboTwin 任务上的评估结果**
   :header-rows: 1

   * - Task
     - OpenVLA-OFT (SFT)
     - OpenVLA-OFT (RLinf-GRPO)
     - OpenVLA-OFT (RLinf-PPO)
   * - beat_block_hammer
     - |huggingface| `10.15% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer>`_
     - |huggingface| `96.09% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-beat_block_hammer>`__
     - ---
   * - pick_dual_bottles
     - |huggingface| `20.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-pick_dual_bottles>`_
     - |huggingface| `92.96% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-pick_dual_bottles>`__
     - ---
   * - place_empty_cup
     - |huggingface| `75.78% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup>`_
     - |huggingface| `94.53% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_empty_cup>`__
     - |huggingface| `92.97% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-PPO-place_empty_cup>`_
   * - place_container_plate
     - |huggingface| `54.69% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_container_plate>`_
     - |huggingface| `95.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_container_plate>`__
     - ---
   * - move_can_pot
     - |huggingface| `9.37% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot>`_
     - |huggingface| `83.59% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-move_can_pot>`__
     - ---
   * - lift_pot
     - |huggingface| `3.13% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot>`_
     - |huggingface| `70.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-lift_pot>`__
     - ---
   * - handover_block
     - |huggingface| `28.13% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-handover_block>`_
     - |huggingface| `70.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-handover_block>`__
     - ---
   * - Average
     - 28.79%
     - **86.16%**
     - ---
   * - Δ Avg.
     - ---
     - **+57.37%**
     - ---


.. list-table:: **OpenPI 在 RoboTwin 任务上的评估结果**
   :header-rows: 1

   * - Task
     - Pi0 (SFT)
     - Pi0 (RLinf-PPO)
     - Pi0.5 (SFT)
     - Pi0.5 (RLinf-PPO)
   * - adjust_bottle
     - |huggingface| `76.56% <https://huggingface.co/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle>`_
     - |huggingface| `98.44% <https://huggingface.co/RLinf/RLinf-Pi0-RoboTwin-PPO-adjust_bottle>`_
     - |huggingface| `85.94% <https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle>`_
     - |huggingface| `96.09% <https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-PPO-adjust_bottle>`_
   * - Average
     - 76.56%
     - 98.44%
     - 85.94%
     - 96.09%
   * - Δ Avg.
     - ---
     - **21.88%**
     - ---
     - **10.15%**

.. note::
   **OpenVLA-OFT** 模型的所有任务都在 **demo_randomized** 设置下进行训练；
   **OpenPI** 模型的所有任务都在 **demo_clean** 设置下进行训练。
   更多信息请参考 `RoboTwin 参数配置文档 <https://robotwin-platform.github.io/doc/usage/configurations.html>`_。

评估脚本
~~~~~~~~~~~~~~~~~~~

本节介绍如何在 RoboTwin 评测平台上对不同 VLA 模型进行评估（Eval）。
在 RLinf 中，模型评估与训练复用同一套配置文件（YAML），
通常只需在对应 YAML 中将 ``runner.only_eval`` 设置为 ``True``，即可进入评估模式。

1. **OpenVLA-OFT 模型评估**

   请确保在运行前已激活正确的 Python 虚拟环境。  
   若使用官方 Docker 镜像，需要通过：

   .. code-block:: bash

      source switch_env openvla-oft

   以 GRPO 算法、``place_empty_cup`` 任务为例，对应配置文件为：

   - ``examples/embodiment/config/robotwin_place_empty_cup_grpo_openvlaoft.yaml``

2. **π₀ 模型评估**

   请确保在运行前已激活正确的 Python 虚拟环境。  
   若使用官方 Docker 镜像，需要通过：

   .. code-block:: bash

      source switch_env OpenPI

   以 PPO 算法、``adjust_bottle`` 任务为例，对应配置文件为：

   - ``examples/embodiment/config/robotwin_adjust_bottle_ppo_OpenPI_eval.yaml``

3. **π₀.₅ 模型评估**

   请确保在运行前已激活正确的 Python 虚拟环境。  
   若使用官方 Docker 镜像，需要通过：

   .. code-block:: bash

      source switch_env OpenPI

   以 PPO 算法、``adjust_bottle`` 任务为例，对应配置文件为：

   - ``examples/embodiment/config/robotwin_adjust_bottle_ppo_OpenPI_pi05_eval.yaml``

4. **评估模式设置**

   在上述任一配置文件中，将 ``runner.only_eval`` 设置为 ``True``：

   .. code-block:: yaml

      runner:
        task_type: embodied
        logger:
          log_path: "../results"
          project_name: rlinf
          experiment_name: "robotwin_grpo_openvlaoft"
          logger_backends: ["tensorboard"]

        max_epochs: 1000
        max_steps: -1
        only_eval: True

5. **启动评估**

   .. code-block:: bash

      export ROBOT_PLATFORM=ALOHA
      export ROBOTWIN_PATH=/path/to/RoboTwin

      bash examples/embodiment/eval_embodiment.sh CHOSEN_CONFIG

6. **注意事项**

   - OpenVLA-OFT 模型目前使用 ``[piper, piper, 0.6]`` 机械臂配置  
   - π\ :sub:`0`\ 和 π\ :sub:`0.5`\ 模型目前使用 ``[aloha-agilex]`` 机械臂配置  
   - 其余详细参数请参考下一节 **配置说明**

配置说明
-----------------------

OpenVLA-OFT关键配置
~~~~~~~~~~~~~~~~~~~

1. **模型配置**：

   - ``actor.model.model_type: "openvla_oft"``：使用 OpenVLA-OFT 模型
   - ``actor.model.implement_version: "official"``：使用 OpenVLA-OFT 官方版本
   - ``actor.model.action_dim: 14``：14 维动作空间（包含本体感觉）
   - ``actor.model.use_proprio: True``：启用本体感觉输入
   - ``actor.model.proprio_dim: 14``：本体感觉维度
   - ``actor.model.num_action_chunks: 25``：动作块数量

2. **算法配置**：

   - ``algorithm.reward_type: chunk_level``：chunk 级别的奖励
   - ``algorithm.logprob_type: token_level``：token 级别的对数概率
   - ``algorithm.n_chunk_steps: 8``：每个 chunk 的步数

3. **环境配置**：

   - ``env.train.task_config.task_name``：任务名称（如 ``place_empty_cup``）
   - ``env.train.task_config.embodiment``：机器人配置
   - ``env.train.task_config.camera``：相机配置

π\ :sub:`0`\ 和 π\ :sub:`0.5`\关键配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **模型配置**：

   - ``actor.model.num_action_chunks: 50``：动作块数量
   - ``actor.model.action_dim: 14``：动作维度
   - ``actor.model.add_value_head: True``：PPO 训练需要 value head
   - ``actor.model.OpenPI.num_images_in_input: 3``：输入图像数量

2. **模型配置名称**：

   - π\ :sub:`0`：``actor.model.OpenPI.config_name: "pi0_aloha_robotwin"``
   - π\ :sub:`0.5`：``actor.model.OpenPI.config_name: "pi05_aloha_robotwin"``

3. **算法配置**：

   - ``algorithm.reward_type: chunk_level``：chunk 级奖励
   - ``algorithm.logprob_type: chunk_level``：chunk 级对数概率
   - ``algorithm.adv_type: gae``：使用 GAE 估计优势
   - ``algorithm.loss_type: actor_critic``：使用 actor-critic 损失

4. **环境配置**：

   - ``env.train.center_crop: False`` 与 ``env.eval.center_crop: False``：关闭中心裁剪
   - ``env.*.task_config.embodiment: [aloha-agilex]``：使用 AgileX 机器人配置，而非oft中使用的[piper, piper, 0.6]
   - ``env.*.task_config.camera.collect_wrist_camera: true``：启用腕部相机
   - ``fsdp.gradient_checkpointing: False``：OpenPI 当前不支持开启梯度检查点

更多关于 RoboTwin 配置的详细信息，请参考 `RoboTwin 配置文档 <https://robotwin-platform.github.io/doc/usage/configurations.html>`_。

注意事项
-----------------------

1. **资源路径**：确保 ``assets_path`` 路径正确
2. **ROBOT_PLATFORM 环境变量**：确保 ``ROBOT_PLATFORM`` 变量设置为 ``ALOHA``
3. **RoboTwin Repo**：确保正确设置 ``ROBOTWIN_PATH``，如 ``export ROBOTWIN_PATH=/path/to/RoboTwin``
4. **GPU 内存**：RoboTwin 环境可能需要较多 GPU 内存，建议使用 ``enable_offload: True``
5. **任务配置**：根据具体任务修改 ``task_config`` 中的参数
