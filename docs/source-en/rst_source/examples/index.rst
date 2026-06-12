Example Gallery
===============

This section presents the collection of **examples currently supported by RLinf**,
showcasing how the framework can be applied across different scenarios and
demonstrating its efficiency in practice.
This example gallery is continuously expanding, covering new scenarios and tasks to highlight RLinf's flexibility and efficiency.

Embodied intelligence is RLinf's primary focus, and the embodied gallery is split into five entry points so you can pick the one that matches your starting question:

- :doc:`simulators/index`: **RL with Embodied Simulators** — pick this when the simulator / benchmark (LIBERO, ManiSkill, RoboTwin, IsaacLab, …) is your starting point.

- :doc:`real_world/index`: **RL with Real-world Robotics** — pick this when you have access to physical hardware (Franka, dexterous hands, mobile dual-arm platforms, …).

- :doc:`vla_wam/index`: **RL on VLA / WAM Models** — pick this when you want to RL-fine-tune a specific model family (π₀, GR00T, Lingbot-VLA, OpenSora, Wan, …).

- :doc:`sft/index`: **SFT for VLA / WAM Models** — supervised fine-tuning recipes that produce strong RL cold-start checkpoints.

- :doc:`methods/index`: **Training Solutions for Embodiment** — algorithm-centric examples (DAgger, RECAP, DSRL, IQL offline RL, sim-real co-training, MLP/SAC-Flow policies, …).

Beyond embodiment:

- :doc:`agentic/index`: **Agentic Scenarios** — math reasoning and agentic AI workflows, including both single-agent and multi-agent settings.

- :doc:`system/index`: **System-level Optimizations** — flexible and dynamic scheduling of computing resources and assignment to the most suitable hardware devices.

.. toctree::
   :hidden:
   :maxdepth: 2

   simulators/index
   real_world/index
   vla_wam/index
   sft/index
   methods/index
   agentic/index
   system/index
