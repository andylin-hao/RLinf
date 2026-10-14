Blog
====

The blog collects RLinf system designs, performance analyses, and application studies. Choose an article for the problem you are working on: embodied-model and simulator throughput, model training and evaluation integration, or online RL in an application.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Embodied Models on Ascend
      :link: ascend_embodied_optimization
      :link-type: doc

      Follow complete-call dependencies to understand metadata reuse, communication overlap, and graph execution.

   .. grid-item-card:: BEHAVIOR Simulation Optimization
      :link: behavior_system_optimization
      :link-type: doc

      Examine how observations, simulation work, and pipeline parallelism affect environment-interaction throughput.

   .. grid-item-card:: Cosmos3 Training and Evaluation
      :link: cosmos3_sft_and_sglang_evaluate
      :link-type: doc

      Explore Cosmos3-Nano fine-tuning, SGLang inference, and parallel simulator evaluation.

   .. grid-item-card:: World Models and VLM Rewards
      :link: worldloop_topreward
      :link-type: doc

      Use training-free visual-language rewards with WoVR and examine results across tasks.

   .. grid-item-card:: Online RL for Code Completion
      :link: build_a_coding_online_rl_case
      :link-type: doc

      Follow an end-to-end case from editor interaction to training feedback.

   .. grid-item-card:: Benchmarking against VeRL
      :link: compare_with_verl
      :link-type: doc

      Look up the environment, configuration, and results for collocated training comparisons.

.. toctree::
   :hidden:
   :maxdepth: 1

   Embodied Models on Ascend <ascend_embodied_optimization>
   behavior_system_optimization
   cosmos3_sft_and_sglang_evaluate
   worldloop_topreward
   build_a_coding_online_rl_case
   compare_with_verl
