安装说明
========

使用 Docker 获得最可复现的安装，或在 Docker 镜像与机器不匹配时使用自定义 Python
环境。无论哪种方式，先选择与你的工作负载匹配的安装目标。

选择安装目标
------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 目标
     - 适用场景
   * - ``embodied``
     - 运行 VLA / 机器人示例。需要传入 ``--model`` 和 ``--env``。
   * - ``agentic``
     - 运行需要 Megatron、SGLang 或 vLLM 的智能体 / 推理示例。
   * - ``docs``
     - 在本地构建 Sphinx 文档。

Docker
------

每个镜像都打包了开箱即用的栈（例如 ``agentic-rlinf0.2-maniskill_libero``）。拉取
并运行镜像，然后在容器内选择模型环境：

.. code-block:: bash

   docker pull rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
   docker run -it --gpus all \
      --shm-size 100g \
      --net=host \
      --name rlinf \
      -e NVIDIA_DRIVER_CAPABILITIES=all \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero /bin/bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   source switch_env openvla

- 保留 ``-e NVIDIA_DRIVER_CAPABILITIES=all``，GPU 渲染需要该能力。
- 不要覆盖容器内的 ``/root`` 或 ``/opt``；镜像中的资源文件和虚拟环境位于这些目录。
- 如果平台会修改 ``$HOME`` 或挂载 ``/root``，请在容器内运行 ``link_assets`` 后再启动示例。
- 使用 ``source switch_env openvla``、``source switch_env openvla-oft`` 或
  ``source switch_env openpi`` 切换模型环境。

自定义环境
----------

当 Docker 镜像与机器不匹配时，用 ``install.sh`` 构建虚拟环境。传入安装目标以及
对应的专用参数：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

更多目标与组合：

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   bash requirements/install.sh embodied --model openpi --env libero
   bash requirements/install.sh agentic
   bash requirements/install.sh docs

运行 ``bash requirements/install.sh --help`` 可查看完整模型和环境列表。

- 使用 ``--venv <dir>`` 指定虚拟环境目录。
- 使用 ``--use-mirror`` 加速中国大陆环境下的下载。
- 仅在依赖需要时使用 ``--python <version>``。默认版本是 Python 3.11.14；部分环境
  如 ``behavior`` 和 ``d4rl`` 需要 Python 3.10。
- 仅在需要不同 PyTorch wheel 时使用 ``--torch <version>``。
- 使用 ``--platform amd`` 或 ``--platform ascend`` 进行实验性的非 NVIDIA 安装。
  参见 :doc:`../guides/amd_rocm` 和 :doc:`../guides/ascend_cann`。

验证
----

激活环境后，确认 RLinf 和 Ray 可用：

.. code-block:: bash

   python -c "import rlinf; print(rlinf.__file__)"
   ray --version

下一步
------

- :doc:`运行 VLA 快速上手 <vla>`。
- :doc:`扩展到多机 <../guides/launch-scale/index>`。
- 如果只需要命令，打开 :doc:`速查表 <cheat_sheet>`。
