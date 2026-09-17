在昇腾 950 NPU 上本地安装时，在 ``embodied`` 之前添加 ``--torch 2.11.0``。昇腾安装脚本默认使用 PyTorch 和 ``torch-npu`` 2.6.0，而 ``torch-npu`` 2.6.0 初始化 950 NPU 时会报 ``Unsupported soc version``。PyTorch 和 ``torch-npu`` 2.11.0 已在 NPU 驱动 25.7.rc1、CANN 9.1.1 的昇腾 950 上验证通过。

.. warning::

   其他昇腾 NPU 请保持默认的 PyTorch 版本。RLinf 在同卡部署的 worker 之间通过 NPU IPC 传递权重。在 CI 使用的昇腾 910B 主机上，``torch-npu`` 2.10 至 2.12 无法在已安装的驱动上使用 NPU IPC：权重同步会报 ``entry in cache has missing shared_ptr``，训练随之停滞。只有在同卡部署的训练能跑完第一个训练步后，才在这类主机上使用 ``--torch 2.11.0``。
