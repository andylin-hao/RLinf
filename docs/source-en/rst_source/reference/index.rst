Reference
=========

Use Reference when you need exact API, algorithm, configuration, or evaluation
details.

API
---

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - What you get
   * - :doc:`API Overview <api/index>`
     - A tour of the core API interfaces and how they fit together.
   * - :doc:`Actor API <api/actor>`
     - The actor worker interface.
   * - :doc:`Channel API <api/channel>`
     - The asynchronous communication channel interface.
   * - :doc:`Cluster API <api/cluster>`
     - The cluster and resource interface.
   * - :doc:`Data API <api/data>`
     - Data structures exchanged between workers.
   * - :doc:`Embodied Data API <api/embodied_data>`
     - Embodied env / rollout data structures.
   * - :doc:`Environment API <api/env>`
     - The environment interface.
   * - :doc:`Placement API <api/placement>`
     - The placement-strategy interface.
   * - :doc:`Replay Buffer API <api/replay_buffer>`
     - The replay buffer interface.
   * - :doc:`Rollout API <api/rollout>`
     - The rollout worker interface.
   * - :doc:`Worker API <api/worker>`
     - The base worker and WorkerGroup interface.

Algorithms
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - Summary
   * - :doc:`PPO <algorithms/ppo>`
     - Proximal Policy Optimization.
   * - :doc:`GRPO <algorithms/grpo>`
     - Group Relative Policy Optimization.
   * - :doc:`DAPO <algorithms/dapo>`
     - Decoupled-clip and dynamic-sampling policy optimization.
   * - :doc:`Reinforce++ <algorithms/reinforce>`
     - An enhanced REINFORCE baseline.
   * - :doc:`SAC <algorithms/sac>`
     - Soft Actor-Critic.
   * - :doc:`CrossQ <algorithms/crossq>`
     - Sample-efficient off-policy RL without target networks.
   * - :doc:`RLPD <algorithms/rlpd>`
     - RL with prior data.
   * - :doc:`IQL <algorithms/iql>`
     - Implicit Q-Learning for offline RL.
   * - :doc:`Async PPO <algorithms/async_ppo>`
     - Asynchronous, pipelined PPO.

Configuration and Evaluation
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - What you get
   * - :doc:`Training Configuration <configuration>`
     - Hydra YAML structure and the most-tuned config keys.
   * - :doc:`Training Metrics <metrics>`
     - The ``train/``, ``eval/``, ``env/``, ``rollout/``, ``time/`` namespaces.
   * - :doc:`Evaluation Configuration <../evaluations/reference/configuration>`
     - Eval YAML structure and required fields.
   * - :doc:`Evaluation CLI <../evaluations/reference/cli>`
     - ``run_eval.sh`` usage and Hydra overrides.
   * - :doc:`Evaluation Models <../evaluations/reference/models>`
     - Supported models and example eval configs.
   * - :doc:`Evaluation Results <../evaluations/reference/results>`
     - Logs, metrics, and video output.

.. toctree::
   :hidden:

   API Overview <api/index>
   Actor API <api/actor>
   Channel API <api/channel>
   Cluster API <api/cluster>
   Data API <api/data>
   Embodied Data API <api/embodied_data>
   Environment API <api/env>
   Placement API <api/placement>
   Replay Buffer API <api/replay_buffer>
   Rollout API <api/rollout>
   Worker API <api/worker>
   PPO <algorithms/ppo>
   GRPO <algorithms/grpo>
   DAPO <algorithms/dapo>
   Reinforce++ <algorithms/reinforce>
   SAC <algorithms/sac>
   CrossQ <algorithms/crossq>
   RLPD <algorithms/rlpd>
   IQL <algorithms/iql>
   Async PPO <algorithms/async_ppo>
   Training Configuration <configuration>
   Training Metrics <metrics>
   Evaluation Configuration <../evaluations/reference/configuration>
   Evaluation CLI <../evaluations/reference/cli>
   Evaluation Models <../evaluations/reference/models>
   Evaluation Results <../evaluations/reference/results>
