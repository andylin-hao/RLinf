ABot-M0 模型强化学习训练
=========================

本文档介绍如何将 ABot-M0 安装到 RLinf，并在 **LIBERO-10** 与 **LIBERO-Plus** 上完成端到端的具身强化学习训练与评测。

本页内容包括：

* **依赖链路验证**：确认 RLinf + ABot-Manipulation + VGGT 可在同一环境中导入。
* **Rollout 验证**：确认 ABot-M0 可在 RLinf rollout worker 内生成动作块。
* **Actor-Rollout 同步验证**：确认策略权重同步与训练循环可正常运行。
* **PPO 训练与评测**：在 LIBERO-10（标准）与 LIBERO-Plus（扰动变体）上端到端跑通 PPO 训练与独立评测。

算法
----

**核心组件**

* **PPO (actor_critic)**
   * 基于 GAE（Generalized Advantage Estimation）的优势估计。
   * 策略比率裁剪（ratio clipping）。
   * 价值函数裁剪（value clipping）。
   * 熵正则项（entropy regularization）。

* **ABot-M0 策略**
   * 面向机器人操作的通用 VLA 模型，支持跨 embodiment 训练。
   * 采用 AML（Action Manifold Learning）以提升连续动作预测的效率与稳定性。
   * 模块化感知设计，可结合 VLM 语义与可选 3D 先验（通过 ABot-Manipulation 与 VGGT）。
   * 在 RLinf 中通过原生 wrapper 支持 rollout 动作生成与训练期 logprob/value 重算。

安装
--------

1. 依赖安装（ABot 与 VGGT）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/amap-cvlab/ABot-Manipulation.git
   git clone https://github.com/facebookresearch/vggt.git

   cd <path_to_RLinf>
   export ABOT_PATH=<path_to_ABot-Manipulation>
   export VGGT_PATH=<path_to_vggt>

2. 环境安装
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env maniskill_libero --install-rlinf
   source .venv/bin/activate

2.1 LIBERO-Plus 安装（仅 LIBERO-Plus 场景需要）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果你只需要在标准 LIBERO-10 上运行，可跳过本节。运行 LIBERO-Plus 时，请在
同一份 ``.venv`` 中追加安装并下载资产：

.. code-block:: bash

   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env liberoplus

   LIBERO_PLUS_DIR=$(python -c "import liberoplus.liberoplus as p, pathlib; print(pathlib.Path(p.__file__).parent)")
   hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip --local-dir "${LIBERO_PLUS_DIR}"
   unzip -o "${LIBERO_PLUS_DIR}/assets.zip" -d "${LIBERO_PLUS_DIR}"

LIBERO-Plus 的完整说明见 :doc:`liberoplus_pro`。

3. 下载 ABot-M0 权重
~~~~~~~~~~~~~~~~~~~~

从以下地址下载 ABot-M0 LIBERO 权重：
``https://huggingface.co/acvlab/ABot-M0-LIBERO/tree/main``

使用 huggingface-cli 示例：

.. code-block:: bash

   pip install -U "huggingface_hub[cli]"
   huggingface-cli download acvlab/ABot-M0-LIBERO \
     --local-dir <path_to_ABot-M0-LIBERO>

3.1 更新 ABot checkpoint 配置中的 ``base_vlm``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ABot-M0 checkpoint 自带 ``config.yaml``。在部分版本中，下面这个字段可能是开发者本地绝对路径，
在其他机器上会失效。

下载 ABot-M0 后，请打开：
``<path_to_ABot-M0-LIBERO>/config.yaml``

找到并修改：

.. code-block:: yaml

   qwenvl:
     base_vlm: /some/developer/local/path/Qwen3-VL-4B-Instruct-Action

改为你机器上真实可用的 Qwen3-VL backbone 本地路径。

模型来源：

* Qwen3-VL backbone：``https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action``
* ABot-M0-LIBERO checkpoints：``https://huggingface.co/acvlab/ABot-M0-LIBERO``

3.2 离线加载 VGGT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ABot 当前默认使用以下方式初始化 VGGT：
``VGGT.from_pretrained("facebook/VGGT-1B")``

如果运行环境无法访问 Hugging Face，请先离线下载：
``https://huggingface.co/facebook/VGGT-1B/``

之后可采用以下方式之一：

* 放入本地 Hugging Face cache 目录，或
* 在 ABot 安装代码中将 VGGT 加载路径显式改为本地目录。

本地路径示例：

.. code-block:: python

   self.spatial_model = spatial_model = VGGT.from_pretrained('/workspace/models/VGGT-1B')

4. 在 ABot 训练配置中设置 ``model_path``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

针对两个 benchmark 各提供一份配置：

* LIBERO-10：    ``examples/embodiment/config/libero_10_ppo_abot_m0.yaml``
* LIBERO-Plus： ``examples/embodiment/config/libero_10_plus_ppo_abot_m0.yaml``

在你将要使用的配置中，将以下两项都设置为本地 ABot-M0 权重路径：

* ``rollout.model.model_path``
* ``actor.model.model_path``

5. 导入完整性验证
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -c "import rlinf; import ABot; import vggt; print('IMPORT_SMOKE_OK')"

若输出 ``IMPORT_SMOKE_OK``，说明包级依赖链路正常。

6. 评测
~~~~~~~

在启动 RL 训练之前，可以先用已有的 ABot-M0 权重跑一轮独立评测，用于验证
rollout 流程、录制可视化视频、并对任务成功率做快速 sanity check。

评测入口是 ``examples/embodiment/eval_embodied_agent.py``。两个 benchmark
共用同一套启动流程，差异只在 ``LIBERO_TYPE`` 与配置文件名。

通用环境变量：

.. code-block:: bash

   source .venv/bin/activate

   export REPO_PATH=$(pwd)
   export EMBODIED_PATH=$(pwd)/examples/embodiment
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export ROBOT_PLATFORM=LIBERO

   ray stop || true
   ray start --head --port=6379

**LIBERO-10（标准）：**

.. code-block:: bash

   export LIBERO_TYPE=standard

   python examples/embodiment/eval_embodied_agent.py \
     --config-name libero_10_ppo_abot_m0 \
     actor.model.model_path=<path_to_abot_m0_ckpt> \
     rollout.model.model_path=<path_to_abot_m0_ckpt> \
     env.eval.total_num_envs=8 \
     env.eval.video_cfg.save_video=true \
     algorithm.eval_rollout_epoch=1 \
     runner.logger.experiment_name=abot_m0_libero10_eval

**LIBERO-Plus：**

.. code-block:: bash

   export LIBERO_TYPE=plus

   python examples/embodiment/eval_embodied_agent.py \
     --config-name libero_10_plus_ppo_abot_m0 \
     actor.model.model_path=<path_to_abot_m0_ckpt> \
     rollout.model.model_path=<path_to_abot_m0_ckpt> \
     env.eval.total_num_envs=8 \
     env.eval.video_cfg.save_video=true \
     algorithm.eval_rollout_epoch=1 \
     runner.logger.experiment_name=abot_m0_liberoplus_eval

注意事项：

* ``actor.model.model_path`` 与 ``rollout.model.model_path`` 必须指向同一份
  你要评测的 ABot-M0 checkpoint 目录或 ``.pt`` 文件。

7. PPO 训练
~~~~~~~~~~~

PPO 训练与评测共用同一套启动流程，差异只在 ``LIBERO_TYPE`` 与配置文件名。

通用环境变量：

.. code-block:: bash

   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   export EMBODIED_PATH=$(pwd)/examples/embodiment
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export ROBOT_PLATFORM=LIBERO

   ray stop || true
   ray start --head --port=6379

**LIBERO-10（标准）：**

.. code-block:: bash

   export LIBERO_TYPE=standard
   python examples/embodiment/train_embodied_agent.py --config-name libero_10_ppo_abot_m0

**LIBERO-Plus：**

.. code-block:: bash

   export LIBERO_TYPE=plus
   python examples/embodiment/train_embodied_agent.py --config-name libero_10_plus_ppo_abot_m0

8. 可视化
~~~~~~~~~

.. code-block:: bash

   tensorboard --logdir logs --port 6006
