基于MetaWorld评测平台的强化学习训练
======================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/metaworld.png
   :align: center
   :width: 90%

   Meta-World 基准（图片来源：`Meta-World <https://metaworld.farama.org>`__）。

`Meta-World <https://metaworld.farama.org>`__ 是一个基于 MuJoCo 的多任务操作基准：一台 7 自由度
机械臂完成 50 个多样的桌面任务。RLinf 借助它对视觉-语言-动作（VLA）策略进行强化学习微调，并评测
分布外（OOD）泛化。

概览
----

在 Meta-World 的 50 个任务上对 VLA 进行强化学习微调；π₀ + PPO 平均成功率约 78%。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      MT50 · ML45（5 个 OOD）

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 张 GPU

| **你将完成：** 安装依赖 → 下载 SFT 模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · SFT 检查点（见下文步骤）。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - 套件
     - 任务数
     - 设置
   * - MT50
     - 50
     - 在全部 50 个任务上进行多任务训练与评测。
   * - ML45
     - 45 + 5
     - 在 45 个任务上训练；在 5 个留出（OOD）任务上评测。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测 (Observation)
     - 工作区周围离屏相机的 RGB 图像（480×480）。
   * - 动作 (Action)
     - 4 维连续动作：3D 末端执行器位置（x, y, z）+ 夹爪开合。
   * - 奖励 (Reward)
     - 稀疏奖励——基于任务完成。


安装
----

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
      rlinf/rlinf:agentic-rlinf0.2-metaworld
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-metaworld

请使用内置的 `switch_env` 工具切换到相应的虚拟环境：

.. code:: bash

   # 使用OpenPi模型训练
   source switch_env openpi
   # 使用OpenVLA-OFT模型训练
   # source switch_env openvla-oft

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   # 使用OpenPi模型训练
   bash requirements/install.sh embodied --model openpi --env metaworld

   # 使用OpenVLA-OFT模型训练
   # bash requirements/install.sh embodied --model openvla-oft --env metaworld
   
   source .venv/bin/activate


下载模型
--------

在开始训练之前，您需要下载相应的预训练模型：

.. code:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-SFT
   git clone https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Metaworld-SFT

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-MetaWorld-SFT --local-dir RLinf-Pi0-MetaWorld-SFT
   hf download RLinf/RLinf-Pi05-MetaWorld-SFT --local-dir RLinf-Pi05-MetaWorld-SFT
   hf download RLinf/RLinf-OpenVLAOFT-Metaworld-SFT --local-dir RLinf-OpenVLAOFT-Metaworld-SFT

下载后，请确保在配置 yaml 文件中正确指定模型路径。

运行
----

**1. 关键集群配置**

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

您可以灵活配置 env、rollout 和 actor 组件的 GPU 数量。
此外，通过在配置中设置 ``pipeline_stage_num = 2``，
您可以实现 rollout 和 env 之间的管道重叠，提高 rollout 效率。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

您也可以重新配置布局以实现完全共享，
其中 env、rollout 和 actor 组件都共享所有 GPU。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

您也可以重新配置布局以实现完全分离，
其中 env、rollout 和 actor 组件各自使用自己的 GPU，无
干扰，消除了卸载功能的需要。


**2. 配置文件**
MetaWorld MT50 多任务联合训练配置文件 （在该任务设定下，训练和推理阶段均在多任务环境当中进行）：

- π\ :sub:`0`\ + PPO:
  ``examples/embodiment/config/metaworld_50_ppo_openpi.yaml``

- π\ :sub:`0.5`\ + PPO:
  ``examples/embodiment/config/metaworld_50_ppo_openpi_pi05.yaml``

- OpenVLA-OFT + GRPO:
  ``examples/embodiment/config/metaworld_50_grpo_openvlaoft.yaml``

MetaWorld ML45 联合训练配置文件 （在该任务设定下，训练在45个任务中进行，推理在OOD的5个任务中进行：

- π\ :sub:`0`\ + PPO:
  ``examples/embodiment/config/metaworld_45_ppo_openpi.yaml``

**3. 启动命令**

要使用选定的配置开始训练，请运行以下
命令：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，要在 MetaWorld 环境中使用 PPO 算法训练 π\ :sub:`0`\ 模型，请运行：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh metaworld_50_ppo_openpi


可视化与结果
------------

**1. TensorBoard 日志记录**

.. code:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键指标**

最值得关注的指标是 **``env/success_once``** —— 任务成功率。每个日志指标的含义见
:doc:`训练指标 </rst_source/tutorials/configuration/metrics>`。

**3. 视频生成**

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

**4. WandB 集成**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "metaworld_50_ppo_openpi"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


MetaWorld 结果
-------------------------
下表Diffusion Policy, TinyVLA和SmolVLA的结果参考 `SmolVLA 论文 <https://arxiv.org/abs/2403.04880>`_ 论文得到。π\ :sub:`0`\ 和 π\ :sub:`0.5`\ 的SFT结果是通过LeRobot官方提供的 `数据集 <https://huggingface.co/datasets/lerobot/metaworld_mt50>`_ 重新训练所得。

.. list-table:: **MetaWorld-MT50 性能对比（Success Rate, %）**
   :widths: 15 10 10 10 10 10
   :header-rows: 1

   * - **Methods**
     - **Easy**
     - **Medium**
     - **Hard**
     - **Very Hard**
     - **Avg.**
   * - Diffusion Policy
     - 23.1
     - 10.7
     - 1.9
     - 6.1
     - 10.5
   * - TinyVLA
     - 77.6
     - 21.5
     - 11.4
     - 15.8
     - 31.6
   * - SmolVLA
     - 87.1
     - 51.8
     - 70.0
     - 64.0
     - 68.2
   * - π\ :sub:`0`\
     - 77.9
     - 51.8
     - 53.3
     - 20.0
     - 50.8
   * - π\ :sub:`0`\  + PPO
     - **92.1**
     - **74.6**
     - 61.7
     - **84.0**
     - **78.1**
   * - π\ :sub:`0.5`\
     - 68.2
     - 37.3
     - 41.7
     - 28.0
     - 43.8
   * - π\ :sub:`0.5`\  + PPO
     - 86.4
     - 55.5
     - **75.0**
     - 66.0
     - 70.7