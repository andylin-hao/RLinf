快速开始
==========

欢迎使用 RLinf。本指南带你从安装到第一次训练运行，并指引接下来的步骤。

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 安装
      :link: installation
      :link-type: doc

      使用 Docker 或自定义环境安装具身智能栈。

   .. grid-item-card:: 快速上手
      :link: vla
      :link-type: doc

      在 ManiSkill3 上使用 PPO 训练 OpenVLA。

   .. grid-item-card:: 启动与扩展
      :link: ../guides/launch-scale/index
      :link-type: doc

      从单机扩展到多节点或真实机器人运行。

环境要求
--------

以下是经过充分测试的配置。

.. list-table:: 硬件
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 配置
   * - GPU
     - 每个节点 8 块 H100
   * - CPU
     - 每个节点 192 核心
   * - 内存
     - 每个节点 1.8TB
   * - 网络
     - NVLink + RoCE / IB，带宽 3.2 Tbps
   * - 存储
     - | 单节点实验使用 1TB 本地存储
       | 分布式实验使用 10TB 共享存储（NAS）

.. list-table:: 软件
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 版本
   * - 操作系统
     - Ubuntu 22.04
   * - NVIDIA 驱动
     - 535.183.06
   * - CUDA
     - 12.4
   * - Docker
     - 26.0.0
   * - NVIDIA Container Toolkit
     - 1.17.8

接下来
------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 示例
      :link: ../examples/index
      :link-type: doc

      在示例库中浏览端到端配方。

   .. grid-item-card:: 评测
      :link: ../evaluations/index
      :link-type: doc

      在基准上评测成功率。

   .. grid-item-card:: 概念
      :link: ../concepts/index
      :link-type: doc

      理解 RLinf 的执行模型。

   .. grid-item-card:: 指南
      :link: ../guides/index
      :link-type: doc

      配置启动、日志、检查点与集群。

   .. grid-item-card:: 为什么选择 RLinf
      :link: ../resources/why_rlinf
      :link-type: doc

      RLinf 背后的设计理念、性能与 SOTA 结果。

命令参考
--------

.. grid:: 1 1 1 1
   :gutter: 2

   .. grid-item-card:: 速查表
      :link: cheat_sheet
      :link-type: doc

      熟悉流程后，直接查找最常用命令。

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   vla
   cheat_sheet
