PPO training for Math Reasoning
==================================

In :doc:`reasoning`, we have already introduced how to use GRPO to train a math reasoning model. RLinf also supports using the PPO algorithm for the same task. This page explains how to train this task with PPO. Since GRPO can be regarded as a variant of standard PPO, and we try to make PPO and GRPO share most of the code and configuration items, this page omits the parts that are largely repetitive. We recommend reading :doc:`reasoning` first.

Overview
--------

Use this recipe when you want actor-critic PPO for the same math-reasoning setup
used by the GRPO example.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-1.5B

   .. grid-item-card:: Algorithm
      :text-align: center

      PPO with GAE advantages and a critic

   .. grid-item-card:: Data
      :text-align: center

      AReaL-boba math reasoning data

   .. grid-item-card:: Hardware
      :text-align: center

      Multi-GPU Megatron training

Dataset
-------

We also use the boba dataset. For details, see :doc:`reasoning`.

Algorithm
---------

We use the standard PPO (Proximal Policy Optimization) algorithm. For the algorithm reference, see :doc:`PPO </rst_source/tutorials/rlalg/ppo>`.

Run It
------

**1. Config file**

Recommended config example:  

- ``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml``

**2. Launch command**

The launch command for PPO training is basically the same as for GRPO training. We also use ``run_main_grpo_math.sh`` as the entry script. RLinf automatically determines whether to use PPO training based on whether there are critic-related configurations in the YAML config file and the value of ``adv_type`` (PPO typically uses ``gae`` as the advantage function).


Training Curve
--------------

We fine-tune the Qwen2.5-1.5B-Instruct model with PPO, and the training curve is shown below. The orange line is RLinf, and the blue line is VeRL as a baseline; both are run with the same algorithm configuration.

Since the base capability of the Qwen2.5-1.5B-Instruct model is relatively weak, the overall reward values are low. However, as training progresses, the reward values increase significantly.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ppo_rlinf_vs_verl.jpg" style="width: 50%;"/>
       <p><em>MATH 1.5B PPO</em></p>
     </div>
   </div>
