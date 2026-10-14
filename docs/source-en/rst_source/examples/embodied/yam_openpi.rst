Pi0.5 Training and Evaluation on YAM
====================================

Collect demonstrations with :doc:`yam`, compute normalization statistics,
fine-tune OpenPI Pi0.5, and evaluate the checkpoint on a dual-arm YAM setup.

Observation and Action
----------------------

``pi05_yam_joint`` uses 14-D states and absolute-position actions in the order
``[left joints (6), left gripper, right joints (6), right gripper]``.
Joints are in radians; grippers are normalized to ``[0, 1]``.
The three RGB views are the top, left-wrist, and right-wrist cameras, in that order.

The data factory learns state-relative deltas for the 12 joints and absolute
gripper targets. Shared OpenPI transforms normalize the 14-D vectors and then
pad them to the model's 32-D width. The model predicts 30 actions; evaluation
executes the first 5 before requesting the next prediction.

Installation
------------

On the GPU training node and the YAM controller, use the same RLinf revision and
install the model and robot dependencies in one environment:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openpi --env yam
   source .venv/bin/activate
   uv pip check

Configure the CAN interfaces and identify the three camera serials using
:doc:`yam`. Use the ``embodied-yam`` Docker build target for a GPU training
image, or install natively on machines that access the robot hardware.

Dataset and Normalization
-------------------------

Collect successful demonstrations following :doc:`yam`. The collector prints
the run directory containing ``lerobot_dataset/``. Set ``YAM_COLLECTION`` to
that directory. The following commands merge its shards into a new local dataset:

.. code-block:: bash

   export YAM_COLLECTION=/absolute/path/to/collection/run
   export YAM_DATASET="$HOME/yam-data/merged"
   mkdir -p "$(dirname "$YAM_DATASET")"
   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir "$YAM_COLLECTION/lerobot_dataset" --output-dir "$YAM_DATASET" \
       --only-success --dry-run
   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir "$YAM_COLLECTION/lerobot_dataset" --output-dir "$YAM_DATASET" \
       --only-success
   head -n 5 "$YAM_DATASET/meta/tasks.jsonl"
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_yam_joint --repo-id "$YAM_DATASET"
   test -s "$YAM_DATASET/norm_stats.json"

Use the recorded task text from ``meta/tasks.jsonl`` as the evaluation prompt.
Keep the resulting statistics with this dataset and use the same file for
training and evaluation.

Pretrained Model
----------------

The common installer provides ``gsutil``. Download the official Pi0.5 base
checkpoint and convert it to the OpenPI PyTorch layout used by RLinf:

.. code-block:: bash

   export YAM_MODELS="$HOME/yam-models"
   mkdir -p "$YAM_MODELS/jax"
   gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base "$YAM_MODELS/jax/"
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
       --input-model "$YAM_MODELS/jax/pi05_base" \
       --input-norm-stats "$YAM_DATASET/norm_stats.json" \
       --output-model "$YAM_MODELS/pi05_base_openpi" \
       --output-norm-stats "$YAM_MODELS/pi05_base_openpi/yam/norm_stats.json" \
       --action-dim 32 --action-horizon 30 --max-token-len 200
   test -s "$YAM_MODELS/pi05_base_openpi/model.safetensors"
   test -s "$YAM_MODELS/pi05_base_openpi/config.json"

Supervised Fine-Tuning
----------------------

Edit ``examples/sft/config/realworld_dual_yam_sft_openpi_pi05.yaml`` with the
absolute paths obtained above:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Field
     - Value
   * - ``data.train_data_paths``
     - The merged ``YAM_DATASET`` directory.
   * - ``actor.model.model_path``
     - ``YAM_MODELS/pi05_base_openpi``.
   * - ``actor.model.openpi_data.norm_stats_path``
     - ``YAM_DATASET/norm_stats.json``.

``cluster.component_placement.actor: all`` uses the allocated GPUs. Run
``nvidia-smi`` to check the available devices and adjust ``actor.micro_batch_size``
and ``actor.global_batch_size`` for them; the global batch must be divisible by
the micro batch times the number of actor workers.

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh realworld_dual_yam_sft_openpi_pi05
   tensorboard --logdir logs --port 6006
   find logs -path '*/actor/model_state_dict/full_weights.pt'

The launcher prints the run's log directory. Select a saved
``checkpoints/global_step_<N>`` directory from the ``find`` output: it contains
``actor/model_state_dict/full_weights.pt``, which the evaluation loader accepts
directly. ``<N>`` is the saved training step.

Evaluation
----------

.. warning::

   Evaluation opens both CAN devices and all three cameras on its first reset.
   Keep an operator at the emergency stop and supervise the entire run. Before
   the episode ends or the process is stopped, ensure both arms can safely remain
   supported when motor output is released. This recipe leaves reset and parking
   to the operator.

Use a GPU node with rank ``0`` and the YAM controller with rank ``1``. Install
the same dependencies on both nodes before starting Ray, as described in
:doc:`../../guides/multi_node`. Check ``ray status`` for an existing cluster;
use machines allocated to this run rather than stopping another user's cluster.

On both machines, run ``hostname -I`` and choose addresses reachable from each
other. Export ``GPU_IP`` and ``CONTROL_IP`` with those addresses. Set
``YAM_TASK_DESCRIPTION`` to the task text in the dataset. On the controller,
export the CAN channels and top/left/right camera serials identified in
:doc:`yam` before starting Ray.

On the GPU node:

.. code-block:: bash

   source .venv/bin/activate
   export YAM_TASK_DESCRIPTION='pick the block'
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address="$GPU_IP"

On the controller, use the same task description:

.. code-block:: bash

   source .venv/bin/activate
   export YAM_TASK_DESCRIPTION='pick the block'
   export RLINF_NODE_RANK=1
   ray start --address="$GPU_IP:6379" --node-ip-address="$CONTROL_IP"

On the GPU node, ``ray status`` must list two alive nodes. Confirm the robot's
joint limits, camera views, emergency stop, and clear workspace using the
collection guide before enabling policy control. The operator positions both
arms for the task; reset and parking remain operator-controlled by default.

Set ``YAM_CHECKPOINT`` to the saved ``global_step_<N>`` directory selected above.
Run evaluation on the GPU node:

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_dual_yam_openpi_pi05_eval \
       rollout.model.model_path="$YAM_CHECKPOINT" \
       rollout.model.openpi_data.norm_stats_path="$YAM_DATASET/norm_stats.json"

The launcher prints the evaluation log directory; videos are saved under that
run. The YAM environment reports execution metrics and a zero step reward;
the operator judges task completion from the robot execution and video.

RPent Deployment
----------------

RPent's YAM OpenPI integration uses this same ``pi05_yam_joint`` contract:
14-D joint targets, top/left/right cameras, a 30-step prediction, and 5 executed
steps at 30 Hz. Use the fine-tuned checkpoint and its dataset statistics with
the `RPent YAM deployment instructions
<https://github.com/RLinf/RPent/blob/main/docs/source-en/rst_source/real_world_robots/yam.rst>`_.
Keep the RLinf revision installed here on both RPent nodes. Stop the native
evaluation before starting RPent's controller so only one process owns the arms.
