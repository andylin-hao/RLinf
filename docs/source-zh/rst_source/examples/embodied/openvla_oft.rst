OpenVLA-OFT 强化学习训练
========================

.. figure:: https://openvla-oft.github.io/static/images/libero_task_performance_results.png
   :align: center
   :width: 90%

   原始 OFT 微调研究的 LIBERO 结果（图片来源：`OpenVLA-OFT 项目 <https://openvla-oft.github.io/>`__）。

使用 RLinf 对 OpenVLA-OFT 进行强化学习微调。本页先介绍 NVIDIA 上的 LIBERO + GRPO 训练流程，再说明如何在 AMD ROCm 和华为昇腾 CANN 上安装并运行同一模型。原始 OpenVLA 模型的训练流程见 :doc:`maniskill`。

概览
----

选择与任务匹配的 checkpoint 和配置，在 LIBERO 上训练 OpenVLA-OFT。其他环境的训练流程见下方链接的模拟器页面。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO · ManiSkill · RoboTwin · MetaWorld · BEHAVIOR · OpenSora · Wan

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      语言条件操作任务

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA · :ref:`AMD ROCm <openvla-oft-amd>` · :ref:`华为昇腾 CANN <openvla-oft-ascend>` （LIBERO）

| **你将完成：** 安装 → 下载 LIBERO-Goal checkpoint → 设置模型路径 → 启动 GRPO → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 所选后端的硬件和驱动。

任务
~~~~

可先运行 LIBERO-Goal；AMD 和昇腾的硬件 e2e 作业也使用这个任务。其他模拟器的完整流程保留在各自页面中。

.. list-table::
   :header-rows: 1
   :widths: 18 22 36 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - :doc:`LIBERO <libero>`
     - Goal
     - ``libero_goal_grpo_openvlaoft``
     - 目标条件操作。
   * - :doc:`ManiSkill <maniskill>`
     - Plate-25
     - ``maniskill_ppo_openvlaoft``
     - 桌面操作。
   * - :doc:`RoboTwin <robotwin>`
     - Place empty cup
     - ``robotwin_place_empty_cup_grpo_openvlaoft``
     - 双臂操作。
   * - :doc:`MetaWorld <metaworld>`
     - MT50
     - ``metaworld_50_grpo_openvlaoft``
     - 多任务操作。
   * - :doc:`BEHAVIOR <behavior>`
     - 家居任务
     - ``behavior_ppo_openvlaoft``
     - 长程活动。
   * - :doc:`OpenSora <opensora>` / :doc:`Wan <wan>`
     - LIBERO Spatial
     - ``opensora_libero_spatial_grpo_openvlaoft`` / ``wan_libero_spatial_grpo_openvlaoft``
     - 使用世界模型训练。

观测与动作
~~~~~~~~~~

LIBERO 训练流程根据图像和任务提示生成动作块。

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - Observation
     - 模型配置指定的 RGB 图像。
   * - Action
     - 由 7 维末端与夹爪动作组成的动作块。
   * - Reward
     - 按训练配置缩放的 LIBERO 任务成功奖励。
   * - Prompt
     - 当前任务的自然语言指令。

安装
----

默认流程使用下方的 NVIDIA 安装步骤。AMD 或昇腾用户请先完成 :ref:`对应后端的安装 <openvla-oft-hardware>`，再下载模型。

.. include:: _setup_common.rst

在仓库根目录启动 NVIDIA 容器：

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g --network host \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero bash
   source switch_env openvla-oft

也可以在本地仅安装 OpenVLA-OFT 和 LIBERO 的依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env libero
   source .venv/bin/activate

下载模型
--------

下载该配置对应的 LIBERO-Goal SFT checkpoint：

.. code-block:: bash

   hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 \
      --local-dir checkpoints/Openvla-oft-SFT-libero-goal-traj1

.. include:: _model_path.rst

在 ``examples/embodiment/config/libero_goal_grpo_openvlaoft.yaml`` 中设置这些路径。保留 ``actor.model.unnorm_key: libero_goal_no_noops``，以匹配 checkpoint 的动作统计数据。其他套件需要对应的 checkpoint 和 key，详见 :doc:`libero`。

运行
----

激活模型环境并设置路径后，启动 LIBERO-Goal 训练：

.. code-block:: bash

   ROBOT_PLATFORM=LIBERO bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openvlaoft

脚本加载指定的 YAML，按 ``cluster.component_placement`` 创建 actor、rollout 和环境 worker，然后启动 GRPO。请按设备资源调整 placement 和 batch size，参见 :doc:`../../concepts/placement` 与 :doc:`../../resources/faq`。

.. _openvla-oft-hardware:

在不同硬件后端上运行
--------------------

NVIDIA 使用上面的安装与启动流程。``.github/workflows/embodied-e2e-tests.yml`` 中有 AMD ROCm 和华为昇腾 CANN 的 OpenVLA-OFT + LIBERO-Goal GRPO e2e 作业。以下后端说明仅覆盖 LIBERO，概览中列出的其他环境需要分别验证。

.. _openvla-oft-amd:

AMD ROCm
~~~~~~~~

启动 ROCm LIBERO 镜像，并将 AMD 设备开放给容器：

.. code-block:: bash

   docker run -it --rm \
      --device=/dev/kfd --device=/dev/dri --group-add video \
      --ipc=host --shm-size 20g --network host \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4 bash
   source switch_env openvla-oft

ROCm 7.2.3 对应的镜像 tag 为 ``agentic-rlinf0.3-libero-rocm7.2.3``。中国大陆用户可使用 ``docker.1ms.run/rlinf/rlinf`` 下的同名 tag。如需从当前代码构建，在宿主机执行以下命令，再将上面容器命令中的镜像替换为 ``rlinf-libero-rocm6.4``：

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=amd \
      --build-arg ROCM_VER=6.4 \
      --build-arg 'ROCM_ARCHS=gfx90a;gfx942' \
      --build-arg BUILD_TARGET=embodied-libero \
      -t rlinf-libero-rocm6.4 .

.. warning::

   ``ROCM_ARCHS`` 必须与目标 GPU 匹配。Docker 构建期间可能无法访问设备，``flash-attn`` 等扩展需要显式指定架构。Dockerfile 会将这些值传给 ROCm 构建工具。

若宿主机已安装 ROCm，也可以直接安装依赖：

.. code-block:: bash

   bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openvla-oft --env libero
   source .venv/bin/activate

省略 ``--rocm`` 可自动检测已安装的版本；中国大陆用户可添加 ``--use-mirror``。完成后按 :ref:`下方步骤启动 LIBERO <openvla-oft-backend-launch>`。

.. _openvla-oft-ascend:

华为昇腾 CANN
~~~~~~~~~~~~~

可以使用容器，也可以在已具备 CANN 和 NPU 驱动的宿主机上安装。

.. include:: _ascend_libero.rst

进入 RLinf 容器后，激活模型环境：

.. code-block:: bash

   source switch_env openvla-oft

本地安装时，通过昇腾选项安装 CPU PyTorch 及匹配的 ``torch-npu``：

.. code-block:: bash

   bash requirements/install.sh --platform ascend embodied --model openvla-oft --env libero
   source .venv/bin/activate

中国大陆用户可添加 ``--use-mirror``。安装脚本在昇腾上会跳过 CUDA flash-attention 的构建。

.. _openvla-oft-backend-launch:

在 AMD 或昇腾上启动 LIBERO
~~~~~~~~~~~~~~~~~~~~~~~~~~

完成任一后端的安装后，按前面的说明下载 LIBERO-Goal checkpoint 并设置模型路径。在同一 shell 中启用软件渲染。

.. include:: _libero_osmesa.rst

启动已配置的 GRPO 训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openvlaoft

若要用硬件 CI 配置做一次短程检查，将两个 worker 都指向本地 checkpoint：

.. code-block:: bash

   export REPO_PATH="$PWD"
   bash tests/e2e_tests/embodied/run.sh libero_goal_grpo_openvlaoft osmesa \
      actor.model.model_path="$PWD/checkpoints/Openvla-oft-SFT-libero-goal-traj1" \
      rollout.model.model_path="$PWD/checkpoints/Openvla-oft-SFT-libero-goal-traj1"

测试脚本的第二个参数选择渲染后端；训练脚本的第二个参数选择 robot platform。使用训练脚本时，通过环境变量指定 OSMesa。

可视化与结果
------------

在训练日志中观察 ``env/success_once``。指标定义见 :doc:`训练指标 <../../reference/metrics>`，已发布的训练结果见 :doc:`LIBERO 结果 <libero>`。独立评测请按 :doc:`LIBERO 评测指南 <../../evaluations/guides/libero>` 操作。
