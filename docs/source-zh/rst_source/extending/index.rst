扩展
====

先选择与改动范围相匹配的扩展点。RLinf 若已能连接硬件，只新增任务，不要改 robotics。如果需要
接入新的传感器、执行器或机器人，先让一个本地部件跑通，再考虑远程放置。下面的指南会从最小实现一直
带到注册和测试。

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 扩展概览
      :link: overview
      :link-type: doc

      各扩展点位于何处，以及各部分如何协同。

   .. grid-item-card:: 新环境
      :link: new_env
      :link-type: doc

      添加一个新的 RL 环境并接入环境注册表。

   .. grid-item-card:: 新任务
      :link: new_task
      :link-type: doc

      在 RLinf 已支持的硬件上新增一个任务。

   .. grid-item-card:: 新机器人
      :link: new_robot
      :link-type: doc

      先接入一个本地部件，再组合机器人并放到远端。

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

   .. grid-item-card:: 高级集成
      :link: advanced-integrations/index
      :link-type: doc

      添加 Megatron-Bridge 与权重同步工作流。

   .. grid-item-card:: SGLang 具身模型
      :link: sglang_embodied_model
      :link-type: doc

      把具身模型使用 sglang 后端适配到 RLinf 的 rollout worker 中，使用各式各样的仿真器来进行模型评测。

.. toctree::
   :hidden:

   扩展概览 <overview>
   新环境 <new_env>
   新任务 <new_task>
   新机器人 <new_robot>
   FSDP 新模型 <new_model_fsdp>
   Megatron 新模型 <new_model_megatron>
   新 SFT 模型 <new_model_sft>
   高级集成 <advanced-integrations/index>
   SGLang 具身模型 <sglang_embodied_model>
