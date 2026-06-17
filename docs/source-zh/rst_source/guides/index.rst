指南
====

指南面向实际操作流程：配置训练、扩展到多节点、恢复训练，以及调试性能。

配置
----

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`基础配置 <basic_config>`
     - 配置文件的结构与覆盖方式。
   * - :doc:`具身智能配置 <embodiment_config>`
     - 具身训练专用的配置项。
   * - :doc:`智能体配置 <agentic_config>`
     - 智能体 / 推理训练专用的配置项。
   * - :doc:`异构集群 <hetero>`
     - 配置混合硬件与节点分组。
   * - :doc:`日志 <logger>`
     - 接入 TensorBoard / wandb / swanlab 后端。
   * - :doc:`恢复训练 <resume>`
     - Checkpoint 频率与断点续训。

运维
----

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`多节点训练 <multi_node>`
     - 在多个节点上启动训练。
   * - :doc:`Checkpoint 转换 <convertor>`
     - 在不同格式之间转换 checkpoint。
   * - :doc:`智能体指南 <agentic>`
     - 智能体任务的操作流程。
   * - :doc:`真机机器人 <realworld_robot>`
     - 在真实机器人硬件上运行 RL。
   * - :doc:`云边协同 <cloud_edge>`
     - 在云端与边缘之间拆分推理与训练。
   * - :doc:`数据采集 <data_collection>`
     - 采集并预处理示教数据。
   * - :doc:`AMD ROCm <amd_rocm>`
     - 在 AMD ROCm 加速器上运行。
   * - :doc:`Ascend CANN <ascend_cann>`
     - 在 Ascend CANN 加速器上运行。
   * - :doc:`LoRA <lora>`
     - 使用 LoRA 适配器训练。
   * - :doc:`5D 并行 <5D>`
     - 为大模型配置 5D 并行。
   * - :doc:`SGLang 版本切换 <version>`
     - 在不同 SGLang 版本之间切换。
   * - :doc:`Profiling <profile>`
     - 对 Ray worker 进程进行系统级 Profiling。
   * - :doc:`动态调度 <dynamic_scheduling>`
     - 训练过程中动态调度资源。
   * - :doc:`自动 Placement <auto_placement>`
     - 为训练负载自动选择最优 placement。

.. toctree::
   :hidden:

   基础配置 <basic_config>
   具身智能配置 <embodiment_config>
   智能体配置 <agentic_config>
   异构集群 <hetero>
   日志 <logger>
   恢复训练 <resume>
   多节点训练 <multi_node>
   Checkpoint 转换 <convertor>
   智能体指南 <agentic>
   真机机器人 <realworld_robot>
   云边协同 <cloud_edge>
   数据采集 <data_collection>
   AMD ROCm <amd_rocm>
   Ascend CANN <ascend_cann>
   LoRA <lora>
   5D 并行 <5D>
   SGLang 版本切换 <version>
   Profiling <profile>
   动态调度 <dynamic_scheduling>
   自动 Placement <auto_placement>
