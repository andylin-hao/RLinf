.. _home:

Welcome to RLinf
================

.. raw:: html

   <div class="rlinf-hero">
     <img class="rlinf-hero-logo" src="_static/svg/logo_white.svg" alt="RLinf logo" />
     <h1 class="rlinf-hero-title">Welcome to RLinf</h1>
     <p class="rlinf-hero-subtitle">Scalable RL post-training for foundation models and embodied agents</p>
   </div>

RLinf is a flexible, scalable open-source infrastructure for post-training
foundation models with reinforcement learning. The "inf" stands for
*Infrastructure* — a robust backbone for next-generation training — and for
*Infinite*, capturing open-ended learning and continuous generalization.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Get Started
      :link: rst_source/start/index
      :link-type: doc
      :text-align: center

   .. grid-item-card:: Install
      :link: rst_source/start/installation
      :link-type: doc
      :text-align: center

   .. grid-item-card:: Examples
      :link: rst_source/examples/index
      :link-type: doc
      :text-align: center

   .. grid-item-card:: Evaluation
      :link: rst_source/evaluations/index
      :link-type: doc
      :text-align: center

Choose Your Path
----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 🤖 Embodied RL
      :link: rst_source/start/vla
      :link-type: doc

      Fine-tune a VLA on LIBERO, ManiSkill, RoboTwin, and more with PPO or GRPO.

   .. grid-item-card:: 🧠 Agentic / Reasoning RL
      :link: rst_source/start/llm
      :link-type: doc

      Train math-reasoning and agentic tasks with GRPO on Qwen / DeepSeek models.

   .. grid-item-card:: 🧩 Bring Your Own
      :link: rst_source/extending/index
      :link-type: doc

      Add a model, environment, or algorithm and plug it into RLinf.

   .. grid-item-card:: 🚀 Scale to a Cluster
      :link: rst_source/concepts/execution_modes
      :link-type: doc

      Collocated, disaggregated, and hybrid placement across GPUs and nodes.

Why RLinf
---------

- **Fast** — hybrid fine-grained pipelining delivers 120%+ throughput over comparable frameworks, plus automatic online scaling.
- **Flexible** — switch FSDP + Hugging Face for prototyping or Megatron + SGLang for large-scale training, with no code changes.
- **Proven** — built-in PPO, GRPO, DAPO, and Reinforce++, with SOTA recipes for embodied and reasoning tasks.

:doc:`Learn more about RLinf's design and benchmarks → <rst_source/resources/why_rlinf>`

.. toctree::
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Get Started <rst_source/start/index>
  Examples <rst_source/examples/index>
  Evaluation <rst_source/evaluations/index>
  Concepts <rst_source/concepts/index>
  Guides <rst_source/guides/index>
  Reference <rst_source/reference/index>
  Extending <rst_source/extending/index>
  Resources <rst_source/resources/index>
