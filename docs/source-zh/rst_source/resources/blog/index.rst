博客
====

博客收录 RLinf 的系统设计、性能分析和应用实践。根据要解决的问题，选择下列文章：具身模型与仿真吞吐优化、模型训练与评测集成，或在线强化学习的实际应用。

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 具身模型的 Ascend 优化
      :link: ascend_embodied_optimization
      :link-type: doc

      从完整调用的数据依赖，解释控制信息复用、通信重叠和执行图的设计与收益。

   .. grid-item-card:: BEHAVIOR 仿真优化
      :link: behavior_system_optimization
      :link-type: doc

      分析观测生成、仿真执行与流水线并行如何影响环境交互吞吐。

   .. grid-item-card:: Cosmos3 的训练与评测
      :link: cosmos3_sft_and_sglang_evaluate
      :link-type: doc

      介绍 Cosmos3-Nano 微调、SGLang 推理与并行仿真评测的集成设计。

   .. grid-item-card:: 世界模型与 VLM 奖励
      :link: worldloop_topreward
      :link-type: doc

      在 WoVR 世界模型训练中使用无需训练的视觉语言奖励，并分析跨任务结果。

   .. grid-item-card:: 在线代码强化学习
      :link: build_a_coding_online_rl_case
      :link-type: doc

      从编辑器交互到训练反馈，说明代码补全在线强化学习的完整案例。

   .. grid-item-card:: 与 VeRL 的基准对比
      :link: compare_with_verl
      :link-type: doc

      查看共享式训练配置下的环境搭建、配置和性能结果。

.. toctree::
   :hidden:
   :maxdepth: 1

   Ascend 具身模型优化 <ascend_embodied_optimization>
   behavior_system_optimization
   cosmos3_sft_and_sglang_evaluate
   worldloop_topreward
   build_a_coding_online_rl_case
   compare_with_verl
