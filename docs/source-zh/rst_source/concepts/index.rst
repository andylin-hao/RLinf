概念
====

在调优 placement、worker 或通信之前，先阅读概念页了解 RLinf 的执行模型。

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: RLinf 执行流程
      :link: execution_flow
      :link-type: doc

      一个任务如何端到端运行——代码流程、进程与核心概念。

   .. grid-item-card:: Worker 与 WorkerGroup
      :link: worker
      :link-type: doc

      计算的基本单元，以及驱动一组 worker 的句柄。

   .. grid-item-card:: M2Flow 编程流程
      :link: flow
      :link-type: doc

      将逻辑与调度解耦的宏观到微观模型。

   .. grid-item-card:: Channel
      :link: channel
      :link-type: doc

      用于 worker 间数据交换的异步通道。

   .. grid-item-card:: Placement
      :link: placement
      :link-type: doc

      worker 如何映射到节点与 GPU。

   .. grid-item-card:: 执行模式
      :link: execution_modes
      :link-type: doc

      共享式、分离式与混合式部署及其权衡。

   .. grid-item-card:: Cluster
      :link: cluster
      :link-type: doc

      集群抽象与资源模型。

   .. grid-item-card:: 集合通信
      :link: collective
      :link-type: doc

      集合通信操作与异步工作句柄。

   .. grid-item-card:: 支持的环境
      :link: supported_envs
      :link-type: doc

      环境接口以及 RLinf 支持的模拟器。

   .. grid-item-card:: Replay Buffer
      :link: replay_buffer
      :link-type: doc

      轨迹回放缓冲区的设计与采样。

.. toctree::
   :hidden:

   RLinf 执行流程 <execution_flow>
   Worker 与 WorkerGroup <worker>
   M2Flow 编程流程 <flow>
   Channel <channel>
   Placement <placement>
   执行模式 <execution_modes>
   Cluster <cluster>
   集合通信 <collective>
   支持的环境 <supported_envs>
   Replay Buffer <replay_buffer>
