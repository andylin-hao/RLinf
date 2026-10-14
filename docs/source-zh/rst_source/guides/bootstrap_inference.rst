终止观测的 value 推理
========================

通过合并终止观测的推理请求，以及直接计算 OpenVLA 的 value，可以减少 rollout 边界处的推理开销。这两个选项用于 HuggingFace 具身 rollout worker 的训练流程。先启用批量推理；如果使用带 value head 的 OpenVLA，再启用直接计算 value 的路径。

合并终止观测
------------

episode 结束或 rollout 收尾时，actor 可能需要动作执行后观测的 value 来计算回报。环境自动 reset 后，新 episode 的初始观测与上一 episode 的终止观测不同。环境会单独发送终止观测，rollout worker 用它计算 bootstrap value，再将结果交回对应的轨迹。

即使每个环境进程只运行一个环境，rollout worker 也可能同时收到多个请求。要在一次模型调用中处理这些请求的终止观测，设置：

.. code-block:: yaml

   rollout:
     batch_bootstrap_inference: true

worker 只合并当前收到的数据中需要终止推理的观测，并按原请求的 batch 大小拆分计算结果，再发送给 actor。该选项默认为 ``false``。RLT 和 DAgger 继续逐请求推理，保留现有的特征路由和 expert 选择行为。

直接计算 OpenVLA 的 value
-------------------------

合并请求减少了模型调用次数，但 OpenVLA 的动作生成路径仍会在每次终止推理中解码动作 token。它的 value head 使用 prompt 最后一个位置的 hidden state；模型生成第一个动作 token 之前，这个 hidden state 就已计算完成。对于设置了 ``actor.model.add_value_head: true`` 的 OpenVLA，可以进一步启用：

.. code-block:: yaml

   rollout:
     batch_bootstrap_inference: true
     value_only_bootstrap: true

``rollout.value_only_bootstrap`` 默认为 ``false``，要求 ``model_type: openvla``。模型的 ``predict_value_batch(env_obs)`` 方法沿用动作推理的图像和 prompt 预处理流程，返回 shape 为 ``[B, 1]`` 的 tensor。存在 value head 时，worker 直接计算 value；RLT、DAgger 和未配置 value head 的模型仍使用动作推理路径。

验证训练配置
------------

分别记录动作推理与终止 value 推理的实际 batch 大小和耗时，并为每种负载单独排除预热调用。批量推理可能改变浮点计算的舍入结果；直接计算 value 则省去了生成无用终止动作时的随机采样，因此相同 seed 下后续的动作序列可能发生变化。选择训练配置时，需要比较 bootstrap value 和学习效果，再测量完整训练迭代的耗时，确认实际收益。
