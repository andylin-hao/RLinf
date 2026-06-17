扩展
====

当你要向 RLinf 添加模型、环境、算法、bridge 或权重同步路径时，使用扩展页。

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 扩展概览
      :link: overview
      :link-type: doc

      各扩展点位于何处，以及各部分如何协同。

   .. grid-item-card:: 新环境
      :link: new_env
      :link-type: doc

      添加一个新的 RL 环境并接入环境注册表。

   .. grid-item-card:: FSDP 新模型
      :link: new_model_fsdp
      :link-type: doc

      在 FSDP 后端上添加 HuggingFace 模型。

   .. grid-item-card:: Megatron 新模型
      :link: new_model_megatron
      :link-type: doc

      在 Megatron+SGLang 后端上添加 HuggingFace 模型。

   .. grid-item-card:: 新 SFT 模型
      :link: new_model_sft
      :link-type: doc

      将新模型接入 SFT 训练流程。

   .. grid-item-card:: Megatron-Bridge
      :link: mbridge
      :link-type: doc

      使用 Megatron-Bridge actor 后端。

   .. grid-item-card:: 权重同步
      :link: weight_syncer
      :link-type: doc

      优化具身训练中 actor 到 rollout 的权重同步。

   .. grid-item-card:: 奖励模型流程
      :link: reward_model
      :link-type: doc

      使用图像分类与 VLM 奖励模型。

.. toctree::
   :hidden:

   扩展概览 <overview>
   新环境 <new_env>
   FSDP 新模型 <new_model_fsdp>
   Megatron 新模型 <new_model_megatron>
   新 SFT 模型 <new_model_sft>
   Megatron-Bridge <mbridge>
   权重同步 <weight_syncer>
   奖励模型流程 <reward_model>
