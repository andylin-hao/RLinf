RL with Behavior Benchmark
==========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
   :align: center
   :width: 90%

   The BEHAVIOR benchmark (image: `BEHAVIOR <https://behavior.stanford.edu>`__).

`BEHAVIOR <https://behavior.stanford.edu>`__ is a benchmark of everyday household
activities built on NVIDIA IsaacSim / OmniGibson. A dual-arm R1 Pro robot performs
long-horizon manipulation; RLinf uses it to RL-fine-tune vision-language-action (VLA)
policies.

Overview
--------

RL-finetune a VLA on BEHAVIOR household tasks with PPO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      50 BEHAVIOR-1K tasks

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · ray-tracing GPUs

| **You'll do:** install IsaacSim deps → download assets + base model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · IsaacSim 4.5 + BEHAVIOR-1K assets (>30 GB) · a base checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Detail
   * - Tasks
     - 50 household manipulation tasks from BEHAVIOR-1K (select via ``task_idx`` 0–49).
   * - Robot
     - Dual-arm R1 Pro on IsaacSim / OmniGibson.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Head-camera RGB (720×720) plus left/right wrist RealSense RGB (480×480).
   * - Action
     - 23-dim continuous: 3-DOF base (x, y, rz), 4-DOF torso, 2×7-DOF arms, and 2×1-DOF parallel-jaw grippers.


Installation
------------

.. warning::

   Check the IsaacSim software and hardware requirements before installing:

   - https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html
   - https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   Hopper-generation GPUs need NVIDIA driver 570 or newer. GPUs without ray tracing
   support, such as A100 or H100, can render BEHAVIOR scenes with severe artifacts.
   Prefer RTX 30/40 series or newer GPUs for visual quality and training stability.

.. include:: _setup_common.rst

**Option 1: Docker image** — image tag ``agentic-rlinf0.2-behavior``:

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-behavior
      # Mainland China mirror: docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-behavior

   # Inside the container, switch to the model's virtual environment:
   source switch_env openvla-oft        # or: source switch_env openpi

**Option 2: Custom environment** — install bundle ``--env behavior``:

.. code-block:: bash

   # Add --use-mirror for faster downloads in mainland China.
   bash requirements/install.sh embodied --model openvla-oft --env behavior
   # Or install the OpenPI environment:
   # bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate


Download the Assets
-------------------

Download IsaacSim 4.5 and set ``ISAAC_PATH`` before every run:

.. code-block:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   mkdir -p $ISAAC_PATH && cd $ISAAC_PATH
   curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o isaac-sim.zip
   unzip isaac-sim.zip && rm isaac-sim.zip

Download BEHAVIOR-1K assets and set ``OMNIGIBSON_DATA_PATH`` before every run:

.. code-block:: bash

   # The datasets require more than 30 GB.
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   mkdir -p $OMNIGIBSON_DATA_PATH

   # Run these inside the active venv. Set HF_ENDPOINT=https://hf-mirror.com in mainland China.
   python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
   python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"

**What this does:**

1. Downloads the IsaacSim runtime that OmniGibson uses.
2. Downloads the BEHAVIOR robot assets, task assets, and 2025 challenge instances.
3. Creates the two environment variables that the training and evaluation scripts need.

Download the Model
------------------

Download the checkpoint for your model family (either method works):

.. code-block:: bash

   # Method 1: git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Behavior
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Behavior

   # Method 2: huggingface-hub (set HF_ENDPOINT=https://hf-mirror.com in mainland China)
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-Behavior --local-dir RLinf-OpenVLAOFT-Behavior
   hf download RLinf/RLinf-Pi0-Behavior --local-dir RLinf-Pi0-Behavior

.. include:: _model_path.rst

Run It
------

.. warning::

   Place BEHAVIOR env workers on GPUs starting from 0. IsaacSim can hang when env
   workers start on later GPU ranks.

Each recipe is a YAML config under ``examples/embodiment/config/``:

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - Model / purpose
     - Algorithm
     - Config
   * - OpenVLA-OFT
     - PPO
     - ``behavior_ppo_openvlaoft.yaml``
   * - π₀
     - PPO
     - ``behavior_ppo_openpi.yaml``
   * - π₀.₅
     - PPO
     - ``behavior_ppo_openpi_pi05.yaml``

Launch a config with ``run_embodiment.sh``:

.. code-block:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft

**What this command does:**

1. Loads ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml`` and its shared env config ``examples/embodiment/config/env/behavior_r1pro.yaml``.
2. Starts Ray workers for the actor, rollout, and BEHAVIOR env placement.
3. Runs PPO training and writes logs/checkpoints under ``runner.logger.log_path``.

.. admonition:: Configure further
   :class: note

   - BEHAVIOR throughput → increase env GPU count first, then tune ``env.num_env_subprocess`` and ``env.train.total_num_envs``.
   - Each BEHAVIOR process can use roughly 10 GiB of VRAM; tune subprocess count for your GPU memory.
   - Cached task instances → generate them with ``rlinf/envs/behavior/instance_generator.py`` and ``examples/embodiment/config/env/behavior_r1pro.yaml``.
   - Placement and throughput → :doc:`Placement </rst_source/tutorials/usage/placement>` and :doc:`Execution modes </rst_source/tutorials/usage/execution_modes>`
   - Metric definitions and logging backends → :doc:`Training metrics </rst_source/tutorials/configuration/metrics>`

.. warning::

   Known issue: under the current BEHAVIOR setup, training success rate
   (``env/success_once``) may stay at 0 for OpenVLA-OFT / π₀.
   This issue will be fixed in a later release.

Evaluate with ``behavior_openpi_pi05_eval.yaml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In principle, any ``pi05`` checkpoint that has non-zero success rate on
Behavior and has been converted to PyTorch format can be used for evaluation
with this config. We use OpenPI-Comet only as an example source:

- https://huggingface.co/sunshk/openpi_comet/tree/main

After download, you can use the following repository to convert weights to
PyTorch format:

- https://github.com/mli0603/openpi-comet

Thanks to the OpenPI-Comet authors for open-sourcing the model and tools, which
helps reproducibility and evaluation in RLinf.

After conversion, update ``behavior_openpi_pi05_eval.yaml`` as follows:

1. Set ``actor.model.model_path`` and ``rollout.model.model_path`` to the converted model directory.
2. Increase ``max_episode_steps`` and ``max_steps_per_rollout_epoch`` in both
   ``env.train`` and ``env.eval`` (for example, ``4096``).

.. code-block:: yaml

   env:
     train:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096
     eval:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096

Run evaluation with:

.. code-block:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval

For the full evaluation workflow, see :doc:`BEHAVIOR-1K evaluation guide <../../evaluations/guides/behavior>`.


Visualization and Results
-------------------------

Launch TensorBoard to watch training live:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

The key signal to watch is **``env/success_once``** — the task success rate. For every
logged metric, see :doc:`Training metrics </rst_source/tutorials/configuration/metrics>`.

To save evaluation videos, enable them in the config:

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval


For the Behavior experiment, we were inspired by 
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_, 
with only minor modifications. We thank the authors for releasing their open-source code.
