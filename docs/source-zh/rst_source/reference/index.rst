参考
====

当你需要精确的 API、算法、配置或评测细节时，使用参考页。

API
---

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 页面
     - 内容
   * - :doc:`API 概览 <api/index>`
     - 核心 API 接口概览及其协同方式。
   * - :doc:`Actor API <api/actor>`
     - actor worker 接口。
   * - :doc:`Channel API <api/channel>`
     - 异步通信通道接口。
   * - :doc:`Cluster API <api/cluster>`
     - 集群与资源接口。
   * - :doc:`Data API <api/data>`
     - worker 间交换的数据结构。
   * - :doc:`Embodied Data API <api/embodied_data>`
     - 具身 env / rollout 数据结构。
   * - :doc:`Environment API <api/env>`
     - 环境接口。
   * - :doc:`Placement API <api/placement>`
     - placement 策略接口。
   * - :doc:`Replay Buffer API <api/replay_buffer>`
     - replay buffer 接口。
   * - :doc:`Rollout API <api/rollout>`
     - rollout worker 接口。
   * - :doc:`Worker API <api/worker>`
     - 基础 worker 与 WorkerGroup 接口。

算法
----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 算法
     - 简介
   * - :doc:`PPO <algorithms/ppo>`
     - Proximal Policy Optimization。
   * - :doc:`GRPO <algorithms/grpo>`
     - Group Relative Policy Optimization。
   * - :doc:`DAPO <algorithms/dapo>`
     - 解耦裁剪与动态采样的策略优化。
   * - :doc:`Reinforce++ <algorithms/reinforce>`
     - 增强版 REINFORCE 基线。
   * - :doc:`SAC <algorithms/sac>`
     - Soft Actor-Critic。
   * - :doc:`CrossQ <algorithms/crossq>`
     - 无需 target 网络的高样本效率离策略 RL。
   * - :doc:`RLPD <algorithms/rlpd>`
     - 利用先验数据的强化学习。
   * - :doc:`IQL <algorithms/iql>`
     - 面向离线 RL 的 Implicit Q-Learning。
   * - :doc:`Async PPO <algorithms/async_ppo>`
     - 异步流水线化的 PPO。

配置与评测
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 页面
     - 内容
   * - :doc:`训练配置 <configuration>`
     - Hydra YAML 结构与最常调的配置项。
   * - :doc:`训练指标 <metrics>`
     - ``train/``、``eval/``、``env/``、``rollout/``、``time/`` 命名空间。
   * - :doc:`评测配置 <../evaluations/reference/configuration>`
     - 评测 YAML 结构与必填字段。
   * - :doc:`评测 CLI <../evaluations/reference/cli>`
     - ``run_eval.sh`` 用法与 Hydra 覆盖。
   * - :doc:`评测模型 <../evaluations/reference/models>`
     - 支持的模型与示例评测配置。
   * - :doc:`评测结果 <../evaluations/reference/results>`
     - 日志、指标与视频输出。

.. toctree::
   :hidden:

   API 概览 <api/index>
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
   训练配置 <configuration>
   训练指标 <metrics>
   评测配置 <../evaluations/reference/configuration>
   评测 CLI <../evaluations/reference/cli>
   评测模型 <../evaluations/reference/models>
   评测结果 <../evaluations/reference/results>
