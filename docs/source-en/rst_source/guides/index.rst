Guides
======

Use Guides for operational workflows: configure a run, scale it, resume it, and
debug performance.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
   * - :doc:`Basic Configuration <basic_config>`
     - How config files are structured and overridden.
   * - :doc:`Embodied Configuration <embodiment_config>`
     - Config keys specific to embodied training.
   * - :doc:`Agentic Configuration <agentic_config>`
     - Config keys specific to agentic / reasoning training.
   * - :doc:`Heterogeneous Clusters <hetero>`
     - Configure mixed hardware and node groups.
   * - :doc:`Logging <logger>`
     - Wire up TensorBoard / wandb / swanlab backends.
   * - :doc:`Resume Training <resume>`
     - Checkpoint cadence and resuming a run.

Operations
----------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
   * - :doc:`Multi-Node Training <multi_node>`
     - Launch a run across multiple nodes.
   * - :doc:`Checkpoint Conversion <convertor>`
     - Convert checkpoints between formats.
   * - :doc:`Agentic Guides <agentic>`
     - Operational workflows for agentic tasks.
   * - :doc:`Real-World Robots <realworld_robot>`
     - Run RL on physical robot hardware.
   * - :doc:`Cloud-Edge Collaboration <cloud_edge>`
     - Split inference and training across cloud and edge.
   * - :doc:`Data Collection <data_collection>`
     - Collect and preprocess demonstration data.
   * - :doc:`AMD ROCm <amd_rocm>`
     - Run on AMD ROCm accelerators.
   * - :doc:`Ascend CANN <ascend_cann>`
     - Run on Ascend CANN accelerators.
   * - :doc:`LoRA <lora>`
     - Train with LoRA adapters.
   * - :doc:`5D Parallelism <5D>`
     - Configure 5D parallelism for large models.
   * - :doc:`SGLang Version Switching <version>`
     - Switch between SGLang versions.
   * - :doc:`Profiling <profile>`
     - System-level profiling of Ray worker processes.
   * - :doc:`Dynamic Scheduling <dynamic_scheduling>`
     - Dynamically schedule resources during training.
   * - :doc:`Auto Placement <auto_placement>`
     - Auto-select the best placement for a workload.

.. toctree::
   :hidden:

   Basic Configuration <basic_config>
   Embodied Configuration <embodiment_config>
   Agentic Configuration <agentic_config>
   Heterogeneous Clusters <hetero>
   Logging <logger>
   Resume Training <resume>
   Multi-Node Training <multi_node>
   Checkpoint Conversion <convertor>
   Agentic Guides <agentic>
   Real-World Robots <realworld_robot>
   Cloud-Edge Collaboration <cloud_edge>
   Data Collection <data_collection>
   AMD ROCm <amd_rocm>
   Ascend CANN <ascend_cann>
   LoRA <lora>
   5D Parallelism <5D>
   SGLang Version Switching <version>
   Profiling <profile>
   Dynamic Scheduling <dynamic_scheduling>
   Auto Placement <auto_placement>
