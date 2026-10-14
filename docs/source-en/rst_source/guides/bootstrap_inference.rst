.. Copyright 2026 The RLinf Authors.
   Licensed under the Apache License, Version 2.0.

Terminal Value Inference
========================

Reduce the work needed to estimate values at rollout boundaries by batching terminal observations and, for OpenVLA, evaluating the value head directly. These options apply to HuggingFace embodied rollout workers during training. Start with batching, then enable direct value inference for an OpenVLA model with a value head.

Batch Terminal Observations
---------------------------

When an episode ends or a rollout closes, the actor may need the value of the post-action observation to compute returns. An environment that resets automatically sends its terminal observation separately from the next episode's initial observation. The rollout worker uses that terminal observation for bootstrap inference and returns each value to its original trajectory.

Several environment requests can arrive together even when each environment contains only one instance. To process their terminal observations in one model call, set:

.. code-block:: yaml

   rollout:
     batch_bootstrap_inference: true

The worker batches only observations that request terminal inference within the current received input. It splits the resulting values by the original request sizes before publishing them to the actor. The default is ``false``. RLT and DAgger retain their per-request inference paths so feature routing and expert selection keep their existing behavior.

Evaluate OpenVLA Values Directly
--------------------------------

Batching reduces the number of calls, but OpenVLA's action-generation path still decodes action tokens during each terminal call. Its value head reads the prompt's final hidden state, which is already available before the first action token is sampled. For an OpenVLA model with ``actor.model.add_value_head: true``, enable:

.. code-block:: yaml

   rollout:
     batch_bootstrap_inference: true
     value_only_bootstrap: true

``rollout.value_only_bootstrap`` defaults to ``false`` and requires ``model_type: openvla``. The model's ``predict_value_batch(env_obs)`` method uses the same image and prompt preprocessing as action inference and returns a tensor with shape ``[B, 1]``. The worker uses direct value inference when a value head is present and retains action inference for RLT, DAgger, and models without a value head.

Validate a Training Configuration
---------------------------------

Measure action inference and terminal value inference separately, recording their actual batch sizes and excluding warm-up calls for each workload. Batching can change floating-point rounding, and direct value inference does not consume the random draws used to generate discarded terminal actions. A fixed seed can therefore produce a different action sequence later in the run. Compare bootstrap values and learning results when selecting a training configuration, then measure the complete training iteration to confirm the practical benefit.
