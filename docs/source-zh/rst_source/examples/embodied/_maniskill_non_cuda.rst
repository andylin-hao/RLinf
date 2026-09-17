ManiSkill 的 PhysX 仿真可以在 CPU 上运行，不依赖承载模型的 accelerator。请在所选 ManiSkill 配置的 ``env.train`` 与 ``env.eval`` 中使用以下设置：

.. code-block:: yaml

   env:
     train:
       total_num_envs: 2
       init_params:
         sim_backend: cpu
         render_backend: "pci:0000:00:00.0"
     eval:
       total_num_envs: 2
       init_params:
         sim_backend: cpu
         render_backend: "pci:0000:00:00.0"

``sim_backend: cpu`` 会将 PhysX 移到 CPU，VLA 仍在所选 accelerator 上运行。``render_backend`` 通过完整 PCI 地址选择 SAPIEN Vulkan renderer；RLinf 会在创建环境前保留 ``pci:<domain>:<bus>:<slot>.<function>`` 格式。若容器内的 renderer 使用其他地址，请替换示例值。

在 AMD 平台上，安装后的环境会让 Vulkan 使用 Mesa 的 RADV 驱动，在 Radeon GPU 上渲染。RADV 不支持仅用于计算的 AMD accelerator，这类设备在 ``lspci`` 中显示为 ``Processing accelerators``。在这类设备上，SAPIEN 会报 ``RuntimeError: cannot create image`` 或 ``failed to find device``，升级 Mesa 也无法解决。此时请在激活环境后执行以下命令，改用 Mesa 的 lavapipe CPU renderer：

.. code-block:: bash

   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

lavapipe 在 CPU 上渲染，因此每个环境每一步占用的 CPU 时间会比 GPU 渲染更多。

.. warning::

   CPU simulation backend 无法在单个进程内向量化多个环境。请将训练与评测的 ``total_num_envs`` 分别设为对应 env worker 的 rank 数量，使每个 worker 只运行一个环境。
