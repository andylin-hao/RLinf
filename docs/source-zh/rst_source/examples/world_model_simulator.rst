基于世界模型模拟器的强化学习训练
=========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档描述如何在 **RLinf** 框架内启动与管理 **Vision-Language-Action Models (VLAs)** 训练任务，并以 **Action-conditioned Opensora 世界模型模拟器** （下文简称 Opensora）作为环境后端。
目标是在无需真实机器人或传统物理仿真器的情况下，通过视觉生成模型模拟环境随动作的动态变化，
为策略优化提供一个稳定、可控的训练闭环。

使用方式与在 LIBERO 环境中微调 VLA 类似，本指南侧重介绍如何在基于 Opensora 的模拟环境中
运行强化学习训练任务，并阐述模型在该框架中具备的关键能力。

Opensora 主要希望赋予模型以下能力：

1. **视觉理解**：Opensora 借助当前观测图像与给定的动作序列生成未来视频帧，为策略提供连续的视觉反馈，使模型能够处理来自真实机器人相机的 RGB 图像。 
2. **语言理解**：理解自然语言的任务描述。  
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。 
4. **策略提升**：借助 Opensora 生成的“想象”轨迹，利用 PPO 等强化学习方法对 VLA 策略进行优化。

与 LIBERO 环境下微调 VLA 的流程类似，本文档重点介绍如何在基于 Opensora 的模拟环境中运行 RL 训练任务。

环境
-----------------------
Opensora 作为一个世界模型，理论上可以拟合任意环境的任意任务，并保持接口一致。以 **LIBERO 环境** 为例子，环境各种接口与定义如下：

**Opensora 模拟 LIBERO 环境**

- **Environment**：视觉生成模型
- **Task**：指挥一台 7 自由度机械臂完成多种家居操作技能（抓取放置、叠放、开抽屉、空间重排等）  
- **Observation**：视觉生成模型返回的图像 
- **Action Space**：7 维连续动作  
  - 末端执行器三维位置控制（x, y, z）  
  - 三维旋转控制（roll, pitch, yaw）  
  - 夹爪控制（开/合）

**Opensora 模拟 LIBERO 环境重置**

不同于真实仿真器可以直接通过 reset() 进行环境重置，Opensora 需要接收初始帧和任务描述进行初始化并重置。故我们需要提前下载对应的用于初始化的数据集，并指定初始化数据集的路径。

**数据结构**

- **Images**：RGB 张量 ``[batch_size, 256, 256, 3]``  
- **Task Descriptions**：自然语言指令  
- **Actions**：归一化的连续值，转换为离散 tokens  
- **Rewards**：由世界模型中的奖励判定器给出，为 0-1 奖励

算法
-----------------------------------------

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用 GAE（Generalized Advantage Estimation）进行优势估计  
   - 基于比率的策略裁剪  
   - 价值函数裁剪  
   - 熵正则化

2. **GRPO（Group Relative Policy Optimization）**

   - 对于每个状态/提示，策略生成 *G* 个独立动作  
   - 以组内平均奖励为基线，计算每个动作的相对优势

3. **Vision-Language-Action 模型**

   - OpenVLA 架构，多模态融合  
   - 动作 token 化与反 token 化  
   - 带 Value Head 的 Critic 功能


依赖安装
---------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env openvla-oft

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令
   # 首先需要正常安装所使用的算法 (openvla-oft) 和仿真环境 (maniskill_libero) 的依赖
   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   # 接着安装 opensora 的依赖
   bash requirements/install.sh embodied --model opensora --env maniskill_libero
   source .venv/bin/activate

VLA 模型下载
--------------

在开始训练之前，你需要下载相应的预训练模型：

.. code:: bash

   # 使用下面任一方法下载模型
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

下载完成后，请确保在配置yaml文件中正确指定模型路径。

.. code:: yaml

   rollout:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   actor:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora

WM (World Model) 模型下载
---------------------------------

除 VLA 模型之外，你还需要下载 Opensora 的相应权重、以及用于仿真初始化的数据集，目前 RLinf 暂时只提供 libero-spatial 和 libero-object 的权重和数据，具体下载方法如下：

.. code:: bash

   # 分别下载 Opensora 中的模型权重和用于初始化的数据
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/jzndd/Opensora_for_libero

   # 方法 2: 使用 huggingface-hub
   pip install huggingface-hub
   hf download jzndd/Opensora_for_libero

Opensora_for_libero 的目录结构如下：

.. code-block:: text

    Opensora_for_libero/
    └── libero_spatial/  (或 libero_object)
        ├── best_wm_ckpt/
        │   └── base_policy/
        │       ├── model/                      # 世界模型权重文件
        │       └── dataset_statistics.json     # 数据集归一化统计信息
        ├── best_rm_ckpt/
        │   └── resnet_rm.pth                   # 奖励模型权重文件
        └── dataset_for_rlinf_world_model_init/ # 仿真初始化数据集
            └── base_policy_rollout_buffer/
                ├── traj0.npy                   # 初始状态帧数据
                ├── traj1.npy
                ├── ...
                └── trajN.npy
        └── vae/                                # VAE 模型权重文件

下载完成后，请确保在配置yaml文件中正确指定模型路径。

.. code:: yaml

    env:
        train:
            opensora_wm_hf_ckpt_path: /Pathto/dataset/Opensora_for_libero/

运行脚本
-------------------

**1. 关键参数配置**

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-7
         rollout: 8-15
         actor: 0-15

   rollout:
      pipeline_stage_num: 2

你可以灵活配置 env、rollout、actor 三个组件使用的 GPU 数量。   
此外，在配置中设置 `pipeline_stage_num = 2`，可实现 **rollout 与 env** 之间的流水线重叠，从而提升 rollout 效率。

.. code-block:: yaml
   
   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

你也可以重新配置 Placement，实现 **完全共享**：env、rollout、actor 三个组件共享全部 GPU。

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 8-15

你还可以重新配置 Placement，实现 **完全分离**：env、rollout、actor 各用各的 GPU、互不干扰，  
这样就不需要 offload 功能。。

**2. 配置文件**

   支持 **OpenVLA-OFT** 模型， **GRPO** 算法。  
   对应配置文件：

   - **OpenVLA-OFT + GRPO**：``examples/embodiment/config/opensora_libero_spatial_grpo_openvlaoft.yaml``

**3. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，使用 opensora 模拟 libero-spatial 环境并使用 GRPO 训练 OpenVLA-OFT 模型：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh opensora_libero_spatial_grpo_openvlaoft

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- **训练指标**：

  - ``train/actor/approx_kl``: 近似 KL，用于监控策略更新幅度
  - ``train/actor/clip_fraction``: 触发 PPO 的 clip 样本的比例
  - ``train/actor/clipped_ratio``: 被裁剪后的概率比均值，用来衡量策略更新受到 clip 的影响程度
  - ``train/actor/grad_norm``: 梯度范数
  - ``train/actor/lr``: 学习率
  - ``train/actor/policy_loss``: PPO/GRPO的策略损失
  - ``train/critic/value_loss``: 价值函数的损失
  - ``train/critic/value_clip_ratio``: PPO-style value function clipping 中触发 clip 的比例
  - ``train/critic/explained_variance``: 衡量价值函数拟合程度，越接近 1 越好
  - ``train/entropy_loss``: 策略熵
  - ``train/loss``: 策略损失 + 价值损失 + 熵正则的总和  (actor_loss + critic_loss + entropy_loss regularization)

- **Rollout 指标**：

  - ``rollout/advantages_max``: 优势函数的最大值
  - ``rollout/advantages_mean``: 优势函数的均值
  - ``rollout/advantages_min``: 优势函数的最小值
  - ``rollout/rewards``: 一个chunk的奖励 （参考 libero_env.py 的414行）

- **环境指标**：

- **环境指标（Environment Metrics）**：

  - ``env/episode_len``：该回合实际经历的环境步数（单位：step）。
  - ``env/return``：回合总回报。在 LIBERO 的稀疏奖励设置中，该指标并不具有参考价值，因为奖励在回合中几乎始终为 0，只有在成功结束时才会给出 1。
  - ``env/reward``：环境的 step-level 奖励（在任务未完成的步骤中为 0，仅在成功终止时为 1）。  
    日志中的数值会按回合步数进行归一化，因此无法直接反映实际的任务完成表现。
  - ``env/success_once``：建议使用该指标来监控训练效果，它直接表示未归一化的任务成功率，更能反映策略的真实性能。

**3. 视频生成**

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eva

**4. 训练日志工具集成**

.. code-block:: yaml

   runner:
      task_type: embodied
      logger:
         log_path: "../results"
         project_name: rlinf
         experiment_name: "libero_10_grpo_openvlaoft"
         logger_backends: ["tensorboard"] # wandb, swanlab

LIBERO 部分结果
~~~~~~~~~~~~~~~~~~~

目前，我们仅在 libero-spatial, libero-object 环境下使用 Opensora 进行模拟，并训练 VLA 模型，更多环境还在测试中。

对于每个 LIBERO 套件，我们评估所有 task_id 与 trial_id 的组合。对于 Object 和 Spatial 套件，我们共评估 500 个环境（10 个任务 × 50 个试次）。

我们根据模型的训练配置来设置评估的超参：
对于 SFT 训练（LoRA-base）模型，我们设置 `do_sample = False`。
对于 RL 训练的模型，我们设置 `do_sample = True`、`temperature = 1.6`，并启用 `rollout_epoch=2` 以激发 RL 调优策略的最佳性能。

.. note:: 
    
    具体选择 Opensora 作为 world model simulator 的动机来源于 `WMPO <https://arxiv.org/abs/2511.09515>`_，在实际世界模型的训练中，我们参考了 `WMPO <https://arxiv.org/abs/2511.09515>`_ 和 `Opensora <https://github.com/jzndd/opensora>`_ 的代码。

.. list-table:: **Evaluation results of the unified model on the two LIBERO task groups**
    :header-rows: 1

    * - 环境
      - 任务
      - Spatial
    * - |huggingface| `OpenVLA-OFT (LoRA-base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora>`_
      - 50.20%
      - 51.61%
    * - OpenVLA-OFT (RLinf-GRPO with Opensora as world model simulator)
      - 75.5%
      - 64.5%
    * - 效果提升
      - +25.3%
      - +12.9%