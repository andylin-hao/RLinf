Extending
=========

Use Extending when you add a model, environment, algorithm, bridge, or weight
synchronization path to RLinf.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Extending Overview
      :link: overview
      :link-type: doc

      Where each extension point lives and how the pieces fit together.

   .. grid-item-card:: New Environment
      :link: new_env
      :link-type: doc

      Add a new RL environment and wire it into the env registry.

   .. grid-item-card:: New Model with FSDP
      :link: new_model_fsdp
      :link-type: doc

      Add a HuggingFace model on the FSDP backend.

   .. grid-item-card:: New Model with Megatron
      :link: new_model_megatron
      :link-type: doc

      Add a HuggingFace model on the Megatron+SGLang backend.

   .. grid-item-card:: New SFT Model
      :link: new_model_sft
      :link-type: doc

      Plug a new model into the SFT training flow.

   .. grid-item-card:: Megatron-Bridge
      :link: mbridge
      :link-type: doc

      Use the Megatron-Bridge actor backend.

   .. grid-item-card:: Weight Synchronization
      :link: weight_syncer
      :link-type: doc

      Optimize actor-to-rollout weight sync in embodied training.

   .. grid-item-card:: Reward Model Workflow
      :link: reward_model
      :link-type: doc

      Use image-classification and VLM reward models.

.. toctree::
   :hidden:

   Extending Overview <overview>
   New Environment <new_env>
   New Model with FSDP <new_model_fsdp>
   New Model with Megatron <new_model_megatron>
   New SFT Model <new_model_sft>
   Megatron-Bridge <mbridge>
   Weight Synchronization <weight_syncer>
   Reward Model Workflow <reward_model>
