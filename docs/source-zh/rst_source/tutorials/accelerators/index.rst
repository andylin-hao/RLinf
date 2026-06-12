支持的加速器
============

RLinf 主要面向 NVIDIA GPU，但部分具身示例同样可以在 **AMD ROCm** 与
**华为 Ascend CANN** 加速器上运行。本节文档侧重于加速器特定的依赖安装与
运行时环境变量，训练任务本身（任务说明、PPO/GRPO 算法、模型下载、配置文件、
指标与结果）与平台无关，可参考对应的示例页。

- :doc:`amd_rocm`
   在 AMD ROCm 上运行 LIBERO 强化学习示例 —— ROCm 依赖安装、OSMesa CPU
   渲染以及 ROCm 专属 Docker 构建参数。

- :doc:`ascend_cann`
   在华为 Ascend CANN 上运行 LIBERO 强化学习示例 —— Ascend 依赖安装、
   宿主机驱动挂载以及 OSMesa CPU 渲染。

.. toctree::
   :hidden:
   :maxdepth: 1

   amd_rocm
   ascend_cann
