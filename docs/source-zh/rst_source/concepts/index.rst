概念
====

在调优 placement、worker 或通信之前，先阅读概念页了解 RLinf 的执行、调度与环境模型。

选择概念区域
------------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 执行模型
      :link: execution-model/index
      :link-type: doc

      理解任务流程、worker、cluster、channel 与 collective。

   .. grid-item-card:: 调度模型
      :link: scheduling-model/index
      :link-type: doc

      理解 placement 策略、执行模式与 replay buffer。

   .. grid-item-card:: 环境模型
      :link: environment-model/index
      :link-type: doc

      理解环境接口与支持的模拟器。

.. toctree::
   :hidden:

   执行模型 <execution-model/index>
   调度模型 <scheduling-model/index>
   环境模型 <environment-model/index>
