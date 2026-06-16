使用 PPO 训练 Math 推理任务
=============================

在 :doc:`reasoning` 中，我们已经介绍了使用 GRPO 来训练数学推理模型，而 RLinf 同样支持使用 PPO 算法来训练同样的任务，本文将介绍如何使用 PPO 来进行这一任务的训练。由于 GRPO 可以看做是标准 PPO 的一种变体，而且我们尽量使得 PPO 与 GRPO 复用大部分代码和配置项，因此本文略去了较为重复的部分，阅读本文前建议先阅读 :doc:`reasoning`。

概述
----

当你希望在与 GRPO 示例相同的数学推理任务上使用 actor-critic PPO 时，请使用本配方。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-1.5B

   .. grid-item-card:: 算法
      :text-align: center

      PPO，使用 GAE 优势估计与 critic

   .. grid-item-card:: 数据
      :text-align: center

      AReaL-boba 数学推理数据

   .. grid-item-card:: 硬件
      :text-align: center

      多 GPU Megatron 训练

数据集
-------------

我们同样采用 boba 数据集，细节请参考 :doc:`reasoning`。

算法
---------

我们采用标准的 PPO（Proximal Policy Optimization）算法，算法参考见 :doc:`PPO </rst_source/tutorials/rlalg/ppo>`。

运行
----

**1. 配置文件**

推荐配置示例：  

- ``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml``

**2. 启动命令**

PPO 训练与 GRPO 训练的启动命令基本相同，也是使用 ``run_main_grpo_math.sh`` 作为入口脚本，RLinf 会通过 yaml 配置文件中是否存在 critic 相关配置以及 adv_type 的取值（PPO 通常使用 gae 作为优势函数）来自动判断是否使用 PPO 训练。


训练曲线
~~~~~~~~~~~~~~

我们基于 Qwen2.5-1.5B-Instruct 模型，使用 PPO 算法进行了训练，下面展示训练曲线。橙色为RLinf，蓝色为作为对照的VeRL，两者运行相同的算法配置。

由于 Qwen2.5-1.5B-Instruct 模型基础能力较弱，所以整体 reward 数值较低，但是随着训练的进行，reward 数值明显上升。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ppo_rlinf_vs_verl.jpg" style="width: 50%;"/>
       <p><em>MATH 1.5B PPO</em></p>
     </div>
   </div>
