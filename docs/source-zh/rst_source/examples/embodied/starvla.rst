StarVLA 模型强化学习训练
=========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/starvla.png
   :align: center
   :width: 70%

   StarVLA：模块化的 VLM backbone + action head。

使用 RLinf 对 **StarVLA** 模型进行强化学习微调。StarVLA 是一个开源的
Vision-Language-Action 工具箱，支持将 VLM backbone 与 action head 以模块化方式组合；
本示例采用 **QwenOFT** 设置，在 **LIBERO** 上使用 GRPO 训练。

概览
----

在 LIBERO Spatial 上用 GRPO 微调 StarVLA（QwenOFT）。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO

   .. grid-item-card:: 算法
      :text-align: center

      GRPO

   .. grid-item-card:: 任务
      :text-align: center

      LIBERO Spatial

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · GPU

| **你将完成：** 安装 → 下载 StarVLA checkpoint 与 base VLM → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 StarVLA LIBERO checkpoint 与 Qwen2.5-VL 基座（见下文）。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 30 42

   * - 套件
     - 配置
     - 重点
   * - LIBERO Spatial
     - ``libero_spatial_grpo_starvla``
     - 空间关系与桌面重排。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测
     - 多视角 RGB（``main_images``，可选 ``wrist_images`` / ``extra_view_images``），可选本体状态 ``states``。
   * - 动作
     - 连续动作块 ``[B, T, D_action]``；LIBERO 上常见为 7 维（末端位姿增量 6D + 夹爪 1D）。
   * - 任务提示
     - 环境提供的自然语言任务描述，直接作为 VLM 输入。
   * - 机器人平台
     - 通过 ``ROBOT_PLATFORM`` 选择（本文默认 ``ROBOT_PLATFORM=libero``）。

接口约定
~~~~~~~~

在 RLinf 的 StarVLA wrapper 中，``env_obs`` 为 batch-first 的 dict（第 0 维为 batch size ``B``）。

必选字段：

* ``main_images``：主视角 RGB，``torch.uint8``，形状 ``[B, H, W, 3]``（常用 ``H=W=224``）。
* ``states``：本体状态，``torch.float32``，形状 ``[B, D_state]``。
* ``task_descriptions``：自然语言任务描述，``list[str]``，长度为 ``B``。

可选字段：

* ``wrist_images``：腕部视角 RGB，``torch.uint8``，形状 ``[B, H, W, 3]``。
* ``extra_view_images``：其他视角 RGB，推荐形状 ``[B, V, H, W, 3]``（``V`` 为额外视角数）。若仅提供单个额外视角，也允许 ``[B, H, W, 3]``，等价视为 ``V=1``。

在 LIBERO 的默认实现中，``states`` 的常见定义为末端位置 ``(x, y, z)``（3 维）、
末端姿态轴角 ``(rx, ry, rz)``（3 维）与夹爪状态（原始 2 维），因此常见
``D_state = 3 + 3 + 2 = 8``。若 checkpoint 期望 7 维状态，wrapper 会将 2 维夹爪状态压缩为
``[x, y, z, rx, ry, rz, g_mean]``，其中 ``g_mean = 0.5 * (g0 + g1)``。

StarVLA 推理输出动作块 ``[B, T, D_action]``，其中
``T = actor.model.num_action_chunks``（planning horizon），
``D_action = actor.model.action_dim``（LIBERO 常用 7）。Rollout 采用 receding-horizon：
每次 forward 产生长度 ``T`` 的动作序列，环境执行前 ``N`` 步（``1 <= N <= T``）后重新规划。

依赖安装
--------

.. include:: _setup_common.rst

**选项 1：Docker 镜像** —— 镜像标签 ``agentic-rlinf0.2-maniskill_libero``：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 国内镜像加速：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

   # 进入容器后，切换到 StarVLA 虚拟环境：
   source switch_env starvla

**选项 2：自定义环境** —— 安装套件 ``--env maniskill_libero``：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 --use-mirror。
   bash requirements/install.sh embodied --model starvla --env maniskill_libero
   source .venv/bin/activate

下载模型
~~~~~~~~

下载 StarVLA checkpoint 与 base VLM：

.. code-block:: bash

   # 方式1：使用 git clone
   git lfs install
   git clone https://huggingface.co/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1
   git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

   # 方式2：使用 huggingface-hub（国内可设置 HF_ENDPOINT=https://hf-mirror.com）
   pip install -U huggingface-hub
   hf download StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1 --local-dir ./Qwen2.5-VL-OFT-LIBERO-4in1
   hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

.. note::

   下载完成后，请修改 ``Qwen2.5-VL-OFT-LIBERO-4in1/config.yaml`` 中的
   ``framework.qwenvl.base_vlm``，使其指向 ``Qwen2.5-VL-3B-Instruct`` 的本地路径。

运行
----

**1. 配置**

StarVLA + GRPO + LIBERO Spatial 使用
``examples/embodiment/config/libero_spatial_grpo_starvla.yaml``。将模型路径指向你的下载，并设置动作接口：

.. code-block:: yaml

   defaults:
      - env/libero_spatial@env.train
      - env/libero_spatial@env.eval

   rollout:
     model:
       model_path: "/path/to/model"

   actor:
     model:
       model_path: "/path/to/model"
       action_dim: 7
       num_action_chunks: 8
       action_stats_source: "minmax"
       starvla:
         framework_name: "QwenOFT"
         expected_action_dim: ${actor.model.action_dim}
         expected_num_action_chunks: ${actor.model.num_action_chunks}
         enable_state_input: False

**2. 启动**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_starvla

评估建议采用 RLinf 统一的评估流程，详见 :doc:`LIBERO 评测指南 <../../evaluations/guides/libero>`。

可视化与结果
------------

关注任务成功率指标 ``env/success_once``。各项指标的含义见
:doc:`训练指标 </rst_source/tutorials/configuration/metrics>`。

参考曲线（采用的模型来自
`LIBERO_BASELIEN_FORJINHUI_10K_QWENOFT <https://huggingface.co/JasonYang66/LIBERO_BASELIEN_FORJINHUI_10K_QWENOFT>`_）：

.. image:: https://github.com/RLinf/misc/raw/main/pic/libero_goal_starvla_baseline.png
   :alt: LIBERO Goal StarVLA baseline result curve
   :width: 95%
   :align: center

.. image:: https://github.com/RLinf/misc/raw/main/pic/libero_object_starvla_baseline.png
   :alt: LIBERO Object StarVLA baseline result curve
   :width: 95%
   :align: center

.. image:: https://github.com/RLinf/misc/raw/main/pic/libero_spatial_starvla_baseline.png
   :alt: LIBERO Spatial StarVLA baseline result curve
   :width: 95%
   :align: center
