RL with MetaWorld Benchmark
===========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/metaworld.png
   :align: center
   :width: 90%

   The Meta-World benchmark (image: `Meta-World <https://metaworld.farama.org>`__).

`Meta-World <https://metaworld.farama.org>`__ is a multi-task manipulation benchmark on
MuJoCo: a 7-DoF arm performs 50 diverse tabletop tasks. RLinf uses it to RL-fine-tune
vision-language-action (VLA) policies, including held-out (OOD) generalization.

Overview
--------

RL-finetune a VLA across Meta-World's 50 tasks; pi0 + PPO reaches ~78% average success.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      MT50 · ML45 (5 OOD)

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install deps → download the SFT model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · an SFT checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Suite
     - Tasks
     - Setting
   * - MT50
     - 50
     - Multi-task training and evaluation across all 50 tasks.
   * - ML45
     - 45 + 5
     - Train on 45 tasks; evaluate on 5 held-out (OOD) tasks.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - RGB (480×480) from off-screen cameras around the workspace.
   * - Action
     - 4-dim continuous: 3D end-effector position (x, y, z) + gripper open/close.
   * - Reward
     - Sparse — based on task completion.


Installation
------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-metaworld
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-metaworld

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   # To train OpenPi models
   source switch_env openpi

   # To train OpenVLA-OFT models
   # source switch_env openvla-oft

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   # To train OpenPi models
   bash requirements/install.sh embodied --model openpi --env metaworld

   # To train OpenVLA-OFT models
   # bash requirements/install.sh embodied --model openvla-oft --env metaworld

   source .venv/bin/activate

Download the Model
------------------

Before starting training, you need to download the corresponding pretrained model:

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-SFT
   git clone https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-MetaWorld-SFT

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-MetaWorld-SFT --local-dir RLinf-Pi0-MetaWorld-SFT
   hf download RLinf/RLinf-Pi05-MetaWorld-SFT --local-dir RLinf-Pi05-MetaWorld-SFT
   hf download RLinf/RLinf-OpenVLAOFT-MetaWorld-SFT --local-dir RLinf-OpenVLAOFT-MetaWorld-SFT

Alternatively, you can also download the model from ModelScope at https://www.modelscope.cn/models/RLinf/RLinf-Pi0-MetaWorld.

After downloading, make sure to correctly specify the model path in the configuration yaml file.

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

You can flexibly configure the GPU count for env, rollout, and actor components.
Additionally, by setting ``pipeline_stage_num = 2`` in the configuration,
you can achieve pipeline overlap between rollout and env, improving rollout efficiency.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also reconfigure the layout to achieve full sharing,
where env, rollout, and actor components all share all GPUs.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

You can also reconfigure the layout to achieve full separation,
where env, rollout, and actor components each use their own GPUs with no
interference, eliminating the need for offloading functionality.



**2. Configuration Files**

MetaWorld MT50 multi-task joint training configuration files (In this task setting, both training and inference are performed in a multi-task environment):

- π\ :sub:`0`\ + PPO:
  ``examples/embodiment/config/metaworld_50_ppo_openpi.yaml``

- π\ :sub:`0.5`\ + PPO:
  ``examples/embodiment/config/metaworld_50_ppo_openpi_pi05.yaml``

- OpenVLA-OFT + GRPO:
  ``examples/embodiment/config/metaworld_50_grpo_openvlaoft.yaml``
  
MetaWorld ML45 joint training configuration files (In this task setting, training is performed on 45 tasks, and inference is performed on 5 OOD tasks):

- π\ :sub:`0`\ + PPO:
  ``examples/embodiment/config/metaworld_45_ppo_openpi.yaml``



**3. Launch Commands**

To start training with the selected configuration, run the following
command:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the π\ :sub:`0`\ model using the PPO algorithm in the MetaWorld environment, run:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh metaworld_50_ppo_openpi


Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code:: bash

   # Launch TensorBoard
   tensorboard --logdir ./logs --port 6006


**2. Key metrics**

The key signal to watch is **``env/success_once``** — the task success rate. For every
logged metric, see :doc:`Training metrics </rst_source/tutorials/configuration/metrics>`.

**3. Video Generation**

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

**4. WandB Integration**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "metaworld_50_ppo_openpi"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


MetaWorld Results
-------------------------
The results for Diffusion Policy, TinyVLA, and SmolVLA in the table below are referenced from the `SmolVLA paper <https://arxiv.org/abs/2403.04880>`_. The SFT results for π\ :sub:`0`\  and π\ :sub:`0.5`\  are obtained by retraining using the official `dataset <https://huggingface.co/datasets/lerobot/metaworld_mt50>`_ provided by LeRobot.

.. list-table:: **MetaWorld-MT50 Performance Comparison (Success Rate, %)**
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