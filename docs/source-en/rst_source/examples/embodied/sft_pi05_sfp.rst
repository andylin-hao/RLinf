Streaming Flow Policy SFT for Pi0.5
===================================

This guide fine-tunes a Pi0.5 checkpoint on LIBERO with the Streaming Flow
Policy (SFP) objective from `StreamingVLA <https://github.com/gen-robot/StreamingVLA>`_,
covering dataset conversion, normalization statistics, configuration, and the
training launch.

SFP changes only how the action expert is trained, not what the model is. An
ordinary Pi0.5 denoises a whole action chunk at once, and its flow time is a
denoising coordinate unrelated to robot time, so nothing is executable until
every denoising step has run. SFP integrates the chunk into a trajectory and
maps the flow time onto it instead: ``t=0`` is the action state the robot has
reached, ``t=1`` is the end of the chunk, and the action expert regresses the
trajectory velocity at ``t`` from a single suffix token. A policy trained this
way can be integrated forward from wherever it currently is, which is what lets
a caller overlap action generation with execution.

Because the architecture is untouched, training starts from a stock Pi0.5
checkpoint and the result loads back into one. What the objective does need is
an extra per-frame field, ``action_states``: the cumulative sum of the actions
taken so far, which is where each chunk's trajectory begins. The steps below
produce it.

RLinf implements the SFP training objective only. Rollout, simulator
evaluation, and RL are not available for an SFP checkpoint yet, because they
would need a trajectory sampler that does not exist; ``actor.model.openpi.task``
values other than ``sft`` reject ``use_sfp`` rather than silently running the
flow-matching sampler on it.


Install dependencies
--------------------

SFP reuses the OpenPI environment. From the RLinf repository root:

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env libero
   source .venv/bin/activate


Convert LIBERO to LeRobot
-------------------------

Download the public `OpenVLA modified LIBERO RLDS dataset
<https://huggingface.co/datasets/openvla/modified_libero_rlds>`_:

.. code:: bash

   hf download openvla/modified_libero_rlds \
       --repo-type dataset \
       --local-dir /data/libero-rlds

Choose where LeRobot datasets live, then convert the four LIBERO suites into
one dataset:

.. code:: bash

   export HF_LEROBOT_HOME=/data/lerobot

   python toolkits/lerobot/convert_libero_data_to_lerobot.py \
       --data-dir /data/libero-rlds \
       --repo-name local/libero_sfp

The converter records ``action_states`` for every frame alongside the images,
state, and action deltas, and writes the result to
``$HF_LEROBOT_HOME/local/libero_sfp``. Pass ``--raw-dataset-names`` to convert a
subset of the suites, and ``--overwrite`` to replace an existing dataset at that
path. ``--repo-name`` must be relative to ``HF_LEROBOT_HOME``; an absolute path
is rejected. Keep ``HF_LEROBOT_HOME`` set to the same value for the remaining
steps.


Compute normalization statistics
--------------------------------

.. code:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_libero_sfp \
       --repo-id local/libero_sfp \
       --output-dir /data/assets/libero_sfp

This writes ``/data/assets/libero_sfp/norm_stats.json`` with statistics for
``state``, ``actions``, and ``action_states``. The last entry is a copy of
``actions``: SFP sums the initial state and the action deltas into one
trajectory, so both must be scaled by the same statistics, and measuring the
cumulative states separately would put them on a different scale.

For the same reason, SFP normalizes by scale alone (dividing by
``max(|q01|, |q99|)``) rather than with the affine quantile map OpenPI applies
elsewhere. Pure scaling commutes with the cumulative sum, so accumulating
normalized deltas gives the same trajectory as normalizing the accumulated one.


Prepare the Pi0.5 checkpoint
----------------------------

Training starts from the official ``pi05_libero`` weights in RLinf's PyTorch
layout, that is a directory containing ``model.safetensors``. In an `OpenPI
<https://github.com/Physical-Intelligence/openpi>`_ checkout:

.. code:: bash

   python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"

   python examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero" \
       --config_name pi05_libero \
       --output_path /data/checkpoints/pi05_libero_pytorch


Configure and launch
--------------------

Point ``examples/sft/config/libero_sft_pi05_sfp.yaml`` at the three paths
created above:

.. code:: yaml

   data:
     train_data_paths: local/libero_sfp

   actor:
     model:
       model_path: /data/checkpoints/pi05_libero_pytorch
       openpi_data:
         norm_stats_path: /data/assets/libero_sfp/norm_stats.json

The objective itself is selected in ``examples/sft/config/model/pi0_5_sfp.yaml``
by ``actor.model.openpi.use_sfp``, with ``sfp_sigma`` and ``sfp_noise_decay``
controlling how much noise is injected at the start of the trajectory and how
quickly it decays along it. ``config_name: pi05_libero_sfp`` selects both the
transform pipeline that carries ``action_states`` and, through its name, the
matching SFT data loader.

The supplied recipe runs full-parameter FSDP SFT on four GPUs with fp32 master
weights and bf16 compute. ``global_batch_size`` must stay a multiple of
``micro_batch_size`` times the world size.

Launch from the RLinf repository root:

.. code:: bash

   source .venv/bin/activate
   export HF_LEROBOT_HOME=/data/lerobot
   bash examples/sft/run_vla_sft.sh libero_sft_pi05_sfp

``run_vla_sft.sh`` overrides ``runner.logger.log_path`` with a timestamped
directory under ``logs/``, so checkpoints appear in
``logs/<timestamp>-libero_sft_pi05_sfp/checkpoints/global_step_<N>/``. To
resume, set ``runner.resume_dir`` to one of those directories and relaunch.
