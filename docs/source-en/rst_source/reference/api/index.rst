APIs
==========


Walk through RLinf's most essential API interfaces and how to use them.
These key APIs are exposed to users to simplify the complex data flows of RL, allowing them to focus on higher-level abstractions without needing to worry about the underlying implementations.

This API documentation proceeds bottom-up, starting with the foundational APIs of RLinf, including:

- :doc:`worker` — A unified interface for workers and worker groups.
- :doc:`placement` — An introduction to RLinf’s GPU placement strategies.
- :doc:`cluster` — Support for distributed training via clusters.
- :doc:`channel` — Low-level communication primitives, including a producer–consumer queue abstraction.

After that, we introduce the upper-layer APIs used to implement different stages of RL:

- :doc:`actor` — Actor wrappers based on FSDP and Megatron.
- :doc:`rollout` — Rollout wrappers built on Huggingface and SGLang.
- :doc:`env` — Environment wrappers for embodied intelligence scenarios.
- :doc:`data` — Encapsulation of the data structure for transmission between different workers.
- :doc:`embodied_data` — Embodied Env/Rollout data structures.
- :doc:`replay_buffer` — Trajectory replay buffer design and sampling.

.. toctree::
   :hidden:
   :maxdepth: 1

   worker
   placement
   cluster
   channel

   actor
   rollout
   env
   data
   embodied_data
   replay_buffer

