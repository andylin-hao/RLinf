RL on ABot-M0 Model
====================

This document describes how to install ABot-M0 and run end-to-end embodied RL training and evaluation on both **LIBERO-10** and **LIBERO-Plus**.

This page covers:

* **Dependency Wiring**: Verify RLinf + ABot-Manipulation + VGGT are importable in one environment.
* **Rollout**: Verify ABot-M0 can generate action chunks inside RLinf rollout workers.
* **Actor-Rollout Sync**: Verify policy weight sync and training loop run without parameter mismatch errors.
* **PPO Training & Evaluation**: Run end-to-end PPO and standalone evaluation on LIBERO-10 (standard) and LIBERO-Plus (perturbation variants).

Algorithm
---------

**Core Components**

* **PPO (actor_critic)**
   * Advantage estimation using GAE (Generalized Advantage Estimation).
   * Policy clipping with ratio limits.
   * Value function clipping.
   * Entropy regularization.

* **ABot-M0 Policy**
   * General-purpose VLA model for robotic manipulation with cross-embodiment training.
   * Action Manifold Learning (AML) for efficient and stable continuous action prediction.
   * Modular perception design that integrates VLM semantics and optional 3D priors (via ABot-Manipulation and VGGT).
   * RLinf-native wrapper for rollout action generation and training-time logprob/value recomputation.

Installation
------------

1. Dependency Installation (ABot and VGGT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/amap-cvlab/ABot-Manipulation.git
   git clone https://github.com/facebookresearch/vggt.git

   cd <path_to_RLinf>
   export ABOT_PATH=<path_to_ABot-Manipulation>
   export VGGT_PATH=<path_to_vggt>

2. Environment Setup
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env maniskill_libero --install-rlinf
   source .venv/bin/activate

2.1 LIBERO-Plus Setup (only required for LIBERO-Plus)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Skip this section if you only want to run on standard LIBERO-10. For
LIBERO-Plus, install it into the same ``.venv`` and download its assets:

.. code-block:: bash

   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env liberoplus

   LIBERO_PLUS_DIR=$(python -c "import liberoplus.liberoplus as p, pathlib; print(pathlib.Path(p.__file__).parent)")
   hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip --local-dir "${LIBERO_PLUS_DIR}"
   unzip -o "${LIBERO_PLUS_DIR}/assets.zip" -d "${LIBERO_PLUS_DIR}"

See :doc:`liberoplus_pro` for full LIBERO-Plus details.

3. Download ABot-M0 Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download ABot-M0 LIBERO checkpoint from:
``https://huggingface.co/acvlab/ABot-M0-LIBERO/tree/main``

Example with huggingface-cli:

.. code-block:: bash

   pip install -U "huggingface_hub[cli]"
   huggingface-cli download acvlab/ABot-M0-LIBERO \
     --local-dir <path_to_ABot-M0-LIBERO>

3.1 Update ``base_vlm`` in ABot checkpoint config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ABot-M0 checkpoints include ``config.yaml``. In some releases, the field below may point to a developer-local path,
which is invalid on other machines.

After downloading ABot-M0, open:
``<path_to_ABot-M0-LIBERO>/config.yaml``

Find and update:

.. code-block:: yaml

   qwenvl:
     base_vlm: /some/developer/local/path/Qwen3-VL-4B-Instruct-Action

to your actual local path of the downloaded Qwen3-VL backbone.

Model sources:

* Qwen3-VL backbone: ``https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action``
* ABot-M0-LIBERO checkpoints: ``https://huggingface.co/acvlab/ABot-M0-LIBERO``

3.2 Offline VGGT loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ABot currently initializes VGGT with:
``VGGT.from_pretrained("facebook/VGGT-1B")``

If your runtime cannot access Hugging Face, pre-download:
``https://huggingface.co/facebook/VGGT-1B/``

Then either:

* place the model in your local Hugging Face cache, or
* explicitly set VGGT loading to a local directory in your ABot installation.

Example local override:

.. code-block:: python

   self.spatial_model = spatial_model = VGGT.from_pretrained('/workspace/models/VGGT-1B')

4. Configure ``model_path`` in ABot Training YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two configs are provided, one per benchmark:

* LIBERO-10:    ``examples/embodiment/config/libero_10_ppo_abot_m0.yaml``
* LIBERO-Plus:  ``examples/embodiment/config/libero_10_plus_ppo_abot_m0.yaml``

In whichever config(s) you plan to use, set both fields to your local
ABot-M0 checkpoint path:

* ``rollout.model.model_path``
* ``actor.model.model_path``

5. Import Sanity Check
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -c "import rlinf; import ABot; import vggt; print('IMPORT_SMOKE_OK')"

If the command prints ``IMPORT_SMOKE_OK``, the package-level dependency wiring is valid.

6. Evaluation
~~~~~~~~~~~~~

Before launching RL training, you can run a standalone evaluation pass with an
existing ABot-M0 checkpoint to verify the rollout pipeline, record videos, and
sanity-check task success rates.

The eval entrypoint is ``examples/embodiment/eval_embodied_agent.py``. Both
benchmarks share the same launch flow; the only differences are
``LIBERO_TYPE`` and the config name.

Common environment setup:

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

**LIBERO-10 (standard):**

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

**LIBERO-Plus:**

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

Notes:

* ``actor.model.model_path`` and ``rollout.model.model_path`` must point to the
  same ABot-M0 checkpoint directory or ``.pt`` file you want to evaluate.

7. PPO Training
~~~~~~~~~~~~~~~

PPO training uses the same launch flow as evaluation; only ``LIBERO_TYPE`` and
the config name differ.

Common environment setup:

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

**LIBERO-10 (standard):**

.. code-block:: bash

   export LIBERO_TYPE=standard
   python examples/embodiment/train_embodied_agent.py --config-name libero_10_ppo_abot_m0

**LIBERO-Plus:**

.. code-block:: bash

   export LIBERO_TYPE=plus
   python examples/embodiment/train_embodied_agent.py --config-name libero_10_plus_ppo_abot_m0

8. Visualization
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   tensorboard --logdir logs --port 6006
