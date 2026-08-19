概念
====

建议先从自己正在使用的那一层读起，建立整体认识后再调优 placement、worker 和通信。
只有在准备扩展或排查底层问题时，才需要继续追到对应的架构页。

选择概念区域
------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 执行模型
      :link: execution-model/index
      :link-type: doc

      理解任务流程、worker、cluster、channel 与 collective。

   .. grid-item-card:: 调度模型
      :link: scheduling-model/index
      :link-type: doc

      理解 placement 策略、执行模式与 replay buffer。

   .. grid-item-card:: 机器人模型
      :link: robotics
      :link-type: doc

      把机器人作为一棵具名观测与动作部件树来使用。

   .. grid-item-card:: 机器人架构
      :link: robotics_architecture
      :link-type: doc

      深入了解共享连接、生命周期和远程放置。

   .. grid-item-card:: 真机环境模型
      :link: realworld_envs
      :link-type: doc

      理解任务、遥操作与各类 wrapper 如何围绕机器人组织。

.. toctree::
   :hidden:

   执行模型 <execution-model/index>
   调度模型 <scheduling-model/index>
   机器人模型 <robotics>
   机器人架构 <robotics_architecture>
   真机环境模型 <realworld_envs>
