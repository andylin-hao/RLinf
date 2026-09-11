RL on π\ :sub:`0`\  and π\ :sub:`0.5`\  Models
==================================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 35%

   The π\ :sub:`0`\  / π\ :sub:`0.5`\  flow-based VLA models.

Fine-tune the **π**\ :sub:`0`\  and **π**\ :sub:`0.5`\  flow-based VLA models with
reinforcement learning (PPO / GRPO) across several simulators using RLinf. For the full
method, see the paper
`πRL: Online RL Fine-Tuning for Flow-Based Vision-Language-Action Models <https://arxiv.org/abs/2510.25889>`__.

Overview
--------

RL-fine-tune π\ :sub:`0`\  / π\ :sub:`0.5`\  on LIBERO, ManiSkill, MetaWorld, and CALVIN with PPO or GRPO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO · ManiSkill · MetaWorld · CALVIN

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      Spatial · Object · Goal · Long

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA CUDA · :ref:`Moore Threads MUSA <pi0-hardware>` (π₀.₅, LIBERO / ManiSkill)

| **You'll do:** install → download an SFT checkpoint → pick a config → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a π\ :sub:`0`\  / π\ :sub:`0.5`\  SFT checkpoint (steps below).

Tasks
~~~~~

Select the model page by matching the environment, task family, and config or checkpoint artifact.

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - LIBERO
     - Spatial · Object · Goal · Long
     - ``libero_spatial_ppo_openpi_pi05`` / ``libero_10_grpo_openpi_pi05``
     - Fine-tune π0 / π0.5 on LIBERO manipulation suites.
   * - ManiSkill3
     - PickCube and related tasks
     - ``maniskill_ppo_openpi_pi05``
     - Fine-tune π0.5 on ManiSkill3 robot-control tasks.
   * - MetaWorld
     - MT50
     - ``metaworld_50_ppo_openpi_pi05``
     - Evaluate generalization across MetaWorld manipulation tasks.
   * - CALVIN
     - ABC-D
     - ``calvin_abc_d_ppo_openpi_pi05``
     - Train on long-horizon language-conditioned manipulation.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - Main-view and wrist-view RGB plus robot state from LIBERO, ManiSkill3, MetaWorld, or CALVIN.
   * - Action
     - 7-D continuous control for end-effector position, rotation, and gripper state.
   * - Reward
     - Environment success or shaped reward used by PPO / GRPO.
   * - Prompt
     - Environment-provided natural-language task description consumed by the VLA processor.

π\ :sub:`0`\  / π\ :sub:`0.5`\  train with PPO (actor-critic; GAE, ratio clipping, value clipping, entropy regularization) or GRPO (group-relative advantages over *G* sampled actions).

Installation
------------

Use the NVIDIA setup below, or follow :ref:`the backend-specific setup <pi0-hardware>` for your hardware.

.. include:: _setup_common.rst

**Option 1: Docker image** — image tag ``agentic-rlinf0.4-maniskill_libero``:

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

--------------

Download the Model
------------------

Before starting training, you need to download the corresponding pretrained models. For example, for Spatial, Object, Goal task types in the LIBERO environment, you can download them as follows:

.. code:: bash

   # Download the Spatial-Object-Goal model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

Alternatively, you can download the model from ModelScope: https://www.modelscope.cn/models/RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal.

Of course, RLinf also provides pretrained models for other environments. The model list is as follows:

.. list-table:: **π**\ :sub:`0`\  **Model List**
   :header-rows: 1
   :widths: 15 25 15 12 12

   * - Environment
     - Task Description
     - SFT Model
     - Flow-SDE
     - Flow-Noise

   * - LIBERO
     - Spatial, Object, Goal
     - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT>`__
     - -
     - -

   * - LIBERO
     - Long
     - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Long-SFT>`__
     - -
     - -

   * - ManiSkill3
     - Multi-task
     - |huggingface| `38.4% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-SFT>`__
     - |huggingface| `78.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowSDE>`__
     - |huggingface| `77.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowNoise>`__

   * - MetaWorld
     - MT50
     - |huggingface| `50.8% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-SFT>`__
     - |huggingface| `78.1% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowSDE>`__
     - |huggingface| `85.8% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowNoise>`__

   * - CALVIN
     - ABC-D
     - |huggingface| `57.5% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT>`__
     - |huggingface| `61.7% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowSDE>`__
     - |huggingface| `59.9% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowNoise>`__

.. list-table:: **π**\ :sub:`0.5`\  **Model List**
   :header-rows: 1
   :widths: 15 25 15 12 12
   :align: left

   * - Environment
     - Task Description
     - SFT Model
     - Flow-SDE
     - Flow-Noise

   * - LIBERO
     - Spatial, Object, Goal, Long
     - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__
     - -
     - -

   * - ManiSkill3
     - Multi-task
     - |huggingface| `40.1% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT>`__
     - |huggingface| `90.9% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE>`__
     - |huggingface| `89.7% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise>`__

   * - MetaWorld
     - MT50
     - |huggingface| `43.8% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT>`__
     - |huggingface| `70.7% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowSDE>`__
     - |huggingface| `66.1% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowNoise>`__

   * - CALVIN
     - ABC-D
     - |huggingface| `61.3% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT>`__
     - |huggingface| `87.0% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE>`__
     - |huggingface| `84.5% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowNoise>`__

After downloading, please make sure to specify the model path correctly in your configuration file.

Run It
------

**1. Key Cluster Configuration**

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

Here you can flexibly configure the GPU count for env, rollout, and
actor components.
Additionally, by setting ``pipeline_stage_num = 2`` in the
configuration, you can achieve pipeline overlap between rollout and
env, improving rollout efficiency.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also reconfigure the placement to achieve complete sharing,
where env, rollout, and actor components all share all GPUs.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

You can also reconfigure the placement to achieve complete separation,
where env, rollout, and actor components each use their own GPUs without
interference, eliminating the need for offload functionality.

--------------

**2. Model Key Parameter Configuration**

**2.1 Model Parameters**

.. code:: yaml

   openpi:
     noise_level: 0.5 # default noise intensity for flow_sde
     noise_logvar_range: [0.08, 0.16] # default learnable noise range for flow_noise
     action_chunk: ${actor.model.num_action_chunks}
     num_steps: ${actor.model.num_steps}
     train_expert_only: True
     action_env_dim: ${actor.model.action_dim}
     noise_method: "flow_sde" # flow_sde, flow_noise
     add_value_head: False
     pi05: False
     value_after_vlm: False

- Set different flow-matching steps via ``num_steps``.

- Use different noise injection methods by modifying ``noise_method``. We provide two options:
  `flow_sde <https://arxiv.org/abs/2505.05470>`__ and
  `flow_noise <https://arxiv.org/abs/2505.22094>`__.
  ``noise_level`` controls the noise intensity for ``flow_sde``, and ``noise_logvar_range`` controls the learnable noise range for ``flow_noise``.

- Enable π\ :sub:`0.5`\  model by setting ``pi05: True``.

- Control the critic position via ``value_after_vlm``: when True, the critic is connected after the VLM module output; when False, the critic input is from the action expert module output.

**2.2 Algorithm Configuration**

In the paper, we provide two technical approaches, flow-noise and flow-sde, to fine-tune π\ :sub:`0`\  and π\ :sub:`0.5`\  models. Specifically, you can choose different technical approaches by switching the following configuration:

.. code:: yaml

   algorithm:
      entropy_bonus: 0.0 # entropy regularization coefficient, set to 0.0 for flow-sde, 0.005 for flow-noise
   openpi:
     noise_method: "flow_sde" # [flow_sde,flow_noise] noise injection method, flow-sde introduces noise through ode-sde transformation, flow-noise introduces noise through noise network
     noise_level: 0.5 # noise intensity for flow-sde
     noise_logvar_range: [0.08, 0.16] # learnable noise range for flow-noise
     joint_logprob: False # whether to optimize joint probability density function. For flow-sde, please set to False. For flow-noise, please set to True.

For example, for complete parameter settings of flow-sde, please refer to ``libero_spatial_ppo_openpi.yaml``; for complete parameter settings of flow-noise, please refer to ``maniskill_ppo_openpi.yaml``.

**2.3 LoRA Settings**

.. code:: yaml

   model:
     is_lora: True
     lora_rank: 8
     gradient_checkpointing: False

If you want to use LoRA (Low-Rank Adaptation) to fine-tune the VLM part, please set ``is_lora: True`` and configure the ``lora_rank`` parameter. Note that gradient checkpointing is currently **not supported**, please keep ``gradient_checkpointing: False``.

⭐ **2.4 Minimum Test Case** ⭐

If you encounter OOM errors or want to implement a minimum test case with as few resources as possible, you can refer to ``libero_spatial_ppo_openpi_quickstart.yaml``.
Compared to the standard task configuration, we have made the following modifications:

.. code:: yaml

   env.train.rollout_epoch: 8 -> 2
   env.train.total_num_envs: 64 -> 32
   actor.micro_batch_size: 128 -> 64
   actor.global_batch_size: 2048 -> 256
   actor.optim.lr: 5e-6 -> 1e-6
   actor.enable_offload: False -> True
   rollout.enable_offload: False -> True

On 4 H100 GPUs, we compared the results of standard parameters and minimum test parameters, and found that their performance is almost the same at the same time: (minimum test parameters optimize faster per round, but converge slower)

.. image:: https://github.com/user-attachments/assets/80d098f6-5286-4ff4-89be-547f43a4dc86
   :alt: Minimum test case comparison
   :width: 95%
   :align: center

If you still encounter OOM issues under the minimum parameter configuration, we provide the following solutions:

**If OOM occurs during the rollout stage:**

- Try replacing the rendering engine from ``egl`` to ``osmesa``
- Further reduce ``env.train.total_num_envs`` from 32 to 16, but increase ``env.train.rollout_epoch`` from 2 to 4 to ensure the total number of environments per rollout round remains consistent
- Check if actor's ``enable_offload`` is enabled, and set it to ``True`` if it is ``False``

**If OOM occurs during the actor stage:**

- Try reducing ``micro_batch_size`` from 64 to 32, keeping ``global_batch_size`` at 256
- Check if rollout's ``enable_offload`` is enabled, and set it to ``True`` if it is ``False``

.. note::

   If you encounter a mismatch between ``micro_batch_size`` and ``global_batch_size``, ensure that ``global_batch_size`` is an integer multiple of ``micro_batch_size`` × number of GPUs.

**2.5 Model Evaluation**

For models after SFT or RL training, we provide two evaluation methods:

- Use RLinf's unified evaluation script; see :doc:`evaluation <../../evaluations/index>` for evaluation. This method supports parallel environment evaluation, which is fast, but only supports outputting the success rate of the entire task.
- To hide inference latency during evaluation or deployment, π\ :sub:`0.5`\  is also integrated with RTC, which overlaps action-chunk execution with the next chunk's inference; see :doc:`RTC <../../guides/rtc>`.

.. note::

   ``Metaworld`` currently do not support the evaluation mode with ``env.eval.auto_reset=True``. It is recommended to use individual script files for model evaluation.

- Use individual script files for model evaluation, refer to the example `README.md <https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/README.md>`__. This method's evaluation scripts are consistent with the official evaluation scripts provided by ``openpi``, supporting output of success rates for each subtask, but it is slower.

**3. Configuration Files**

Using libero-10 as an example, the configuration files for π\ :sub:`0`\  and π\ :sub:`0.5`\  are:

- π\ :sub:`0`\ + PPO:
   ``examples/embodiment/config/libero_10_ppo_openpi.yaml``
- π\ :sub:`0`\ + GRPO:
   ``examples/embodiment/config/libero_10_grpo_openpi.yaml``
- π\ :sub:`0.5`\ + PPO:
   ``examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml``
- π\ :sub:`0.5`\ + GRPO:
   ``examples/embodiment/config/libero_10_grpo_openpi_pi05.yaml``

--------------

**4. Launch Command**

To start training with a chosen configuration, run the following
command:

::

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the π\ :sub:`0`\  model using the PPO algorithm in
the LIBERO environment, run:

::

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi_quickstart

--------------

.. _pi0-hardware:

Run on Different Hardware Backends
----------------------------------

NVIDIA uses the installation and launch steps above. Moore Threads MUSA has
e2e jobs for **π₀.₅ + PPO** on **LIBERO-10** and **ManiSkill plate-25**. These
jobs cover π₀.₅; they do not establish MUSA support for π₀ or the other
environments on this page.

Moore Threads MUSA
~~~~~~~~~~~~~~~~~~

Install OpenPI inside a Moore Threads container. The installer reuses the
image's Python, PyTorch, and ``torch_musa`` through a virtual environment with
system site-packages enabled. It preserves the vendor torch packages and skips
CUDA-only dependencies, including the vLLM and SGLang builds. These embodied
recipes use ``rollout.generation_backend: huggingface``.

For LIBERO, start from the training-suite image on the host:

.. code-block:: bash

   export MUSA_IMAGE=registry.mthreads.com/mcctest/ai/training-suite:v2.1.5-musa4.3.7
   docker run -it --rm --runtime=mthreads \
      --ipc=host --shm-size=100g \
      -e MTHREADS_VISIBLE_DEVICES=all \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      "$MUSA_IMAGE" bash

Inside the container, verify device access and install OpenPI with LIBERO:

.. code-block:: bash

   mthreads-gmi
   python -c "import torch, torch_musa; print(torch.musa.device_count())"
   bash requirements/install.sh --platform musa embodied --model openpi --env libero
   source .venv/bin/activate

Add ``--use-mirror`` for downloads from mainland China. RLinf detects the MUSA
devices and assigns them to workers using the existing placement config.

.. warning::

   The container needs ``--runtime=mthreads`` to expose the driver libraries.
   A zero device count must be resolved before training. Keep the image's
   PyTorch and ``torch_musa`` pair; installing a generic torch wheel can replace
   the MUSA build. If you select a Transformers model path that requests
   ``attn_implementation: flash_attention_2``, use ``sdpa`` when the vendor
   flash-attention package is not detected by Transformers.

To build an RLinf image from your checkout, use BuildKit on the host:

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=musa \
      --build-arg MUSA_VER=v2.1.5-musa4.3.7 \
      --build-arg BUILD_TARGET=embodied-maniskill_libero \
      -t rlinf:embodied-maniskill_libero-musa .

Use the resulting tag as ``MUSA_IMAGE`` in the container command, then activate
``source switch_env openpi``. BuildKit resolves the selected platform stages
without fetching unrelated platform bases.

.. warning::

   The image build cannot import torch before the runtime injects the MUSA
   driver libraries. Asset downloads that require ManiSkill are deferred to
   the running container. Building the public ``maniskill_libero`` target does
   not supply the vendor simulator modifications required below.

LIBERO on MUSA
~~~~~~~~~~~~~~~

Download ``RLinf/RLinf-Pi05-LIBERO-SFT`` from the model list above. Set both model
paths in ``examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`` and enable
software rendering in the active environment.

.. include:: _libero_osmesa.rst

Launch the LIBERO PPO recipe:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

For a short check, the MUSA CI config uses two devices and small environment
counts. Point its model paths to your download:

.. code-block:: bash

   export REPO_PATH="$PWD"
   bash tests/e2e_tests/embodied/run.sh libero_10_ppo_openpi_pi05_musa osmesa \
      actor.model.model_path=/path/to/RLinf-Pi05-LIBERO-SFT \
      rollout.model.model_path=/path/to/RLinf-Pi05-LIBERO-SFT

The test runner's second argument selects OSMesa; the training wrapper uses
``MUJOCO_GL`` and ``PYOPENGL_PLATFORM`` instead.

ManiSkill on MUSA
~~~~~~~~~~~~~~~~~

.. TODO(agent): Pin the vendor RLinf image tag when a reproducible tag is recorded in the repository.

Use a vendor RLinf image from ``registry.mthreads.com/lgpublic/rlinf`` with
the MUSA-adapted simulator packages. Set ``MUSA_IMAGE`` to the exact tag supplied
by the vendor, start it with the container command above, and activate its
OpenPI environment.

.. warning::

   ManiSkill on MUSA requires both a modified SAPIEN and a matching ManiSkill.
   Stock SAPIEN cannot create the required render system; public ManiSkill
   ``v3.0.0b22`` also imports a sensor missing from the vendor SAPIEN build.
   ``install.sh --env maniskill_libero`` installs the public packages, so it
   cannot provide this simulator stack. Preserve the vendor packages when
   preparing this workflow.

Fetch the assets inside the running vendor container:

.. code-block:: bash

   source switch_env openpi
   download_assets --assets maniskill

The MUSA test config selects CPU physics and an explicit render device for
both training and evaluation:

.. code-block:: yaml

   env:
     train:
       total_num_envs: 2
       init_params:
         sim_backend: cpu
         render_backend: "pci:0000:00:00.0"
     eval:
       total_num_envs: 2
       init_params:
         sim_backend: cpu
         render_backend: "pci:0000:00:00.0"

These settings come from
``tests/e2e_tests/embodied/maniskill_async_ppo_openpi_pi05_musa.yaml``.
It places actor, rollout, and env on devices ``0-1`` with
``rollout.pipeline_stage_num: 1``. CPU physics runs one environment per env
worker in this setup, so keep each ``total_num_envs`` equal to the env worker
count when changing placement. CUDA PhysX simulation is unavailable on MUSA;
OSMesa variables alone do not configure SAPIEN's renderer.

Download ``RLinf/RLinf-Pi05-ManiSkill-25Main-SFT`` from the model list above,
then run the small configuration with your checkpoint paths:

.. code-block:: bash

   export REPO_PATH="$PWD"
   ROBOT_PLATFORM=BRIDGE bash tests/e2e_tests/embodied/run.sh maniskill_async_ppo_openpi_pi05_musa osmesa \
      actor.model.model_path=/path/to/RLinf-Pi05-ManiSkill-25Main-SFT \
      rollout.model.model_path=/path/to/RLinf-Pi05-ManiSkill-25Main-SFT

Despite the config's name, the MUSA CI job launches it with the synchronous
``run.sh`` entry point. Use that same command for this check. To extend the run,
adjust ``runner.max_epochs`` and the episode limits in the config after the
small setup works.

Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code:: bash

   # Launch TensorBoard
   tensorboard --logdir ./logs --port 6006

--------------

**2. Key Metrics**

Watch **``env/success_once``** for the task success rate. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

--------------

**3. Video Generation**

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

--------------

**4. WandB Integration**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "libero_10_ppo_openpi"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

--------------

LIBERO Results
~~~~~~~~~~~~~~

We trained π\ :sub:`0`\  and π\ :sub:`0.5`\  with PPO and GRPO in the LIBERO environment.
The results achieved through RL training are shown below:

.. list-table:: **π**\ :sub:`0`\  **model results on LIBERO**
   :header-rows: 1

   * - Model
     - Spatial
     - Object
     - Goal
     - Long
     - Average
     - Δ Avg.

   * - π\ :sub:`0`\ (few-shot)
     - 65.3%
     - 64.4%
     - 49.8%
     - 51.2%
     - 57.6%
     - ---

   * - +GRPO
     - 97.8%
     - 97.8%
     - 83.2%
     - 81.4%
     - 90.0%
     - +32.4

   * - +PPO
     - **98.4%**
     - **99.4%**
     - **96.2%**
     - **90.2%**
     - **96.0%**
     - **+38.4**

.. list-table:: **π**\ :sub:`0.5`\  **model results on LIBERO**
   :header-rows: 1

   * - Model
     - Spatial
     - Object
     - Goal
     - Long
     - Average
     - Δ Avg.

   * - π\ :sub:`0.5`\ (few-shot)
     - 84.6%
     - 95.4%
     - 84.6%
     - 43.9%
     - 77.1%
     - ---

   * - +GRPO
     - 97.4%
     - 99.8%
     - 91.2%
     - 77.6%
     - 91.5%
     - +14.4

   * - +PPO
     - **99.6%**
     - **100%**
     - **98.8%**
     - **93.0%**
     - **97.9%**
     - **+20.8**

MetaWorld Results
~~~~~~~~~~~~~~~~~
For MetaWorld results, please check :doc:`MetaWorld Page <metaworld>`.

CALVIN Results
~~~~~~~~~~~~~~~~~
For CALVIN results, please check :doc:`CALVIN Page <calvin>`.
