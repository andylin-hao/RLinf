Franka真机强化学习
============================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon


本文档给出在 RLinf 框架内启动在 Franka 机械臂真机环境中训练任务的完整指南，
重点介绍如何从零开始训练基于 ResNet 的 CNN 策略以完成机器人操作任务。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像。  
2. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。  
3. **强化学习**：结合环境反馈，使用 SAC 优化策略。

环境
-----------

**真实世界环境**

- **Environment**: 真机设置
  - Franka Emika Panda 机械臂
  - Realsense 相机
  - 可能使用空间鼠标进行数据采集和人类干预
- **Task**: 目前支持插块插入（Peg Insertion）和充电器插电（Charger）任务
- **Observation**: 腕部或第三人称相机的 RGB 图像（128×128）
- **Action Space**: 6 维或 7 维连续动作，取决于是否包含夹爪控制：
  - 三维位置控制（x, y, z）
  - 三维旋转控制（roll, pitch, yaw）
  - 夹爪控制（开/合）

**数据结构**

- **Images**: RGB 张量 ``[batch_size, 128, 128, 3]``
- **Actions**:归一化取值在 ``[-1, 1]`` 的连续值
- **Rewards**: 基于任务完成度的逐步奖励


算法
-----------------------------------------

**核心算法组件**

1. **SAC (Soft Actor-Critic)**

   - 通过 Bellman 公式和熵正则化学习 Q 值。

   - 学习策略网络以最大化熵正则化的 Q 值。

   - 学习温度参数以平衡探索与利用。

2. **Cross-Q**

   - SAC 的一种变体，去除了目标 Q 网络。

   - 在一个批次中连接当前观测和下一个观测，结合 BatchNorm 实现 Q 的稳定训练。

3. **RLPD (Reinforcement Learning with Prior Data)**

   - SAC 的一种变体，结合离线数据和在线数据进行训练。

   - 使用较大的网络更新与数据更新比例，以提高数据效率。

4. **CNN Policy Network**

   - 基于 ResNet 的视觉输入处理架构。

   - 使用 MLP 层融合图像和状态以输出动作。

   - 用多个 Q-head 实现 Critic 功能。

依赖安装
-----------------------
TODO

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- **环境指标**:

  - ``env/episode_len``：该回合实际经历的环境步数（单位：step）
  - ``env/return``：回合总回报。在 LIBERO 的稀疏奖励设置中，该指标并不具有参考价值，因为奖励在回合中几乎始终为 0，只有在成功结束时才会给出 1
  - ``env/reward``：环境的 step-level 奖励
  - ``env/success_once``：建议使用该指标来监控训练效果，它直接表示未归一化的任务成功率，更能反映策略的真实性能

- **Training Metrics**:

  - ``train/sac/critic_loss``: Q 函数的损失
  - ``train/critic/grad_norm``: Q 函数的梯度范数

  - ``train/sac/actor_loss``: 策略损失
  - ``train/actor/entropy``: 策略熵
  - ``train/actor/grad_norm``: 策略的梯度范数

  - ``train/sac/alpha_loss``: 温度参数的损失
  - ``train/sac/alpha``: 温度参数的值
  - ``train/alpha/grad_norm``: 温度参数的梯度范数

  - ``train/replay_buffer/size``: 当前重放缓冲区的大小
  - ``train/replay_buffer/max_reward``: 重放缓冲区中存储的最大奖励
  - ``train/replay_buffer/min_reward``: 重放缓冲区中存储的最小奖励
  - ``train/replay_buffer/mean_reward``: 重放缓冲区中存储的平均奖励
  - ``train/replay_buffer/std_reward``: 重放缓冲区中存储的奖励标准差
  - ``train/replay_buffer/utilization``: 重放缓冲区的利用率

真实世界结果
~~~~~~~~~~~~~~~~~~
以下提供了插块插入任务和充电器任务的演示视频和训练曲线。在 1 小时的训练时间内，机器人能够学习到一套能够持续成功完成任务的策略。

.. raw:: html

  <div style="flex: 0.8; text-align: center;">
      <img src="https://github.com/RLinf/misc/raw/main/pic/realworld-curve.png" style="width: 100%;"/>
      <p><em>训练曲线</em></p>
    </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://github.com/RLinf/misc/raw/main/pic/peg-insertion-compressed.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>插块插入（Peg Insertion）</em></p>
  </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://github.com/RLinf/misc/raw/main/pic/charger-compressed.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>充电器插电（Charger）</em></p>
  </div>