Pi0.5 的 Streaming Flow Policy 监督微调
=======================================

本文介绍如何在 LIBERO 上用 `StreamingVLA <https://github.com/gen-robot/StreamingVLA>`_ 提出的 Streaming Flow Policy（SFP）目标微调 Pi0.5，内容依次是数据集转换、归一化统计量、配置修改和训练启动。

SFP 改变的只是 action expert 的训练目标，模型本身没有变。普通 Pi0.5 一次性对整个 action chunk 去噪，它的 flow 时间是一个去噪坐标，和机器人的物理时间无关，因此必须走完全部去噪步才能拿到可执行的动作。SFP 则把 chunk 累加成一条轨迹，并把 flow 时间映射到轨迹上：``t=0`` 是机器人当前累积到的动作状态，``t=1`` 是 chunk 末尾，action expert 用一个 suffix token 回归 ``t`` 处的轨迹速度。这样训练出的 policy 可以从当前位置沿速度场继续积分，调用方也就能把动作生成和动作执行重叠起来。

因为网络结构没动，训练可以直接从官方 Pi0.5 权重开始，产出的权重也能按原样载回。SFP 额外需要的是一个逐帧字段 ``action_states``，即此前所有动作的累积和，每个 chunk 的轨迹就从这里起步。下面的步骤会生成它。

RLinf 目前只实现了 SFP 的训练目标。SFP 权重还不能用于 rollout、仿真评测和 RL，因为这些路径需要一个尚不存在的轨迹采样器；``actor.model.openpi.task`` 取 ``sft`` 以外的值时会直接拒绝 ``use_sfp``，而不是拿 flow matching 的采样器去跑它。


安装依赖
--------

SFP 复用 OpenPI 环境。在 RLinf 仓库根目录执行：

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env libero
   source .venv/bin/activate


转换 LIBERO 数据
----------------

下载公开的 `OpenVLA modified LIBERO RLDS 数据集 <https://huggingface.co/datasets/openvla/modified_libero_rlds>`_：

.. code:: bash

   hf download openvla/modified_libero_rlds \
       --repo-type dataset \
       --local-dir /data/libero-rlds

设置 LeRobot 数据集的存放目录，然后把四个 LIBERO 任务套件合并转换成一个数据集：

.. code:: bash

   export HF_LEROBOT_HOME=/data/lerobot

   python toolkits/lerobot/convert_libero_data_to_lerobot.py \
       --data-dir /data/libero-rlds \
       --repo-name local/libero_sfp

转换脚本会在图像、state 和动作增量之外，为每一帧记录 ``action_states``，结果写入 ``$HF_LEROBOT_HOME/local/libero_sfp``。用 ``--raw-dataset-names`` 可以只转换其中几个套件，``--overwrite`` 用于覆盖该路径下已有的数据集。``--repo-name`` 必须是相对于 ``HF_LEROBOT_HOME`` 的路径，传绝对路径会被拒绝。后续步骤要保持 ``HF_LEROBOT_HOME`` 不变。


计算归一化统计量
----------------

.. code:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_libero_sfp \
       --repo-id local/libero_sfp \
       --output-dir /data/assets/libero_sfp

命令生成 ``/data/assets/libero_sfp/norm_stats.json``，其中包含 ``state``、``actions`` 和 ``action_states`` 三项统计量。最后一项是 ``actions`` 的副本：SFP 把初始状态和动作增量累加成同一条轨迹，两者必须用同一套统计量缩放，单独统计累积状态会把它们放到不同尺度上。

同样的原因决定了 SFP 只做纯缩放归一化（除以 ``max(|q01|, |q99|)``），而不用 OpenPI 在别处采用的仿射 quantile 映射。纯缩放是线性的，与累加可交换，因此“先归一化再累加”和“先累加再归一化”结果一致。


准备 Pi0.5 初始权重
-------------------

训练从官方 ``pi05_libero`` 权重开始，需要转换成 RLinf 使用的 PyTorch 布局，也就是一个包含 ``model.safetensors`` 的目录。在 `OpenPI <https://github.com/Physical-Intelligence/openpi>`_ 仓库中执行：

.. code:: bash

   python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"

   python examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero" \
       --config_name pi05_libero \
       --output_path /data/checkpoints/pi05_libero_pytorch


修改配置并启动训练
------------------

把 ``examples/sft/config/libero_sft_pi05_sfp.yaml`` 指向上面生成的三个路径：

.. code:: yaml

   data:
     train_data_paths: local/libero_sfp

   actor:
     model:
       model_path: /data/checkpoints/pi05_libero_pytorch
       openpi_data:
         norm_stats_path: /data/assets/libero_sfp/norm_stats.json

训练目标本身由 ``examples/sft/config/model/pi0_5_sfp.yaml`` 中的 ``actor.model.openpi.use_sfp`` 选择，``sfp_sigma`` 和 ``sfp_noise_decay`` 分别控制轨迹起点注入的噪声大小、以及它沿轨迹衰减的快慢。``config_name: pi05_libero_sfp`` 同时决定了两件事：携带 ``action_states`` 的 transform 管线，以及通过名字匹配到的 SFT 数据加载器。

配套配置在四张卡上做全参数 FSDP 微调，权重主副本为 fp32，计算用 bf16。``global_batch_size`` 必须是 ``micro_batch_size`` 乘以 world size 的整数倍。

在 RLinf 仓库根目录启动：

.. code:: bash

   source .venv/bin/activate
   export HF_LEROBOT_HOME=/data/lerobot
   bash examples/sft/run_vla_sft.sh libero_sft_pi05_sfp

``run_vla_sft.sh`` 会把 ``runner.logger.log_path`` 覆盖成 ``logs/`` 下带时间戳的目录，因此 checkpoint 出现在 ``logs/<时间戳>-libero_sft_pi05_sfp/checkpoints/global_step_<N>/``。需要继续训练时，把 ``runner.resume_dir`` 指向其中一个目录再启动即可。
