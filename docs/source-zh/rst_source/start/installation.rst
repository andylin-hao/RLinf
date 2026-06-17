安装说明
========

除非你明确需要智能体 / 推理依赖，否则建议从具身智能栈开始。它是运行本节快速上手示例的最短路径。

具身智能快速安装
----------------

如果希望环境最可复现，优先使用 Docker：

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

如果 Docker 镜像与你的机器不匹配，可以使用自定义 Python 环境：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

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

常见具身智能安装命令：

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   bash requirements/install.sh embodied --model openpi --env libero

运行 ``bash requirements/install.sh --help`` 可查看完整模型和环境列表。

详细选项
--------

Docker
~~~~~~

- 保留 ``-e NVIDIA_DRIVER_CAPABILITIES=all``，具身环境渲染需要 GPU graphics 能力。
- 不要覆盖容器内的 ``/root`` 或 ``/opt``；镜像中的资源文件和虚拟环境位于这些目录。
- 如果平台会修改 ``$HOME`` 或挂载 ``/root``，请在容器内运行 ``link_assets`` 后再启动示例。
- 使用 ``source switch_env openvla``、``source switch_env openvla-oft`` 或
  ``source switch_env openpi`` 切换模型环境。

自定义环境
~~~~~~~~~~

- 使用 ``--venv <dir>`` 指定虚拟环境目录。
- 使用 ``--use-mirror`` 加速中国大陆环境下的下载。
- 仅在依赖需要时使用 ``--python <version>``。默认版本是 Python 3.11.14；部分环境
  如 ``behavior`` 和 ``d4rl`` 需要 Python 3.10。
- 仅在需要不同 PyTorch wheel 时使用 ``--torch <version>``。
- 使用 ``--platform amd`` 或 ``--platform ascend`` 进行实验性的非 NVIDIA 安装。
  参见 :doc:`../guides/amd_rocm` 和 :doc:`../guides/ascend_cann`。

智能体 / 推理
~~~~~~~~~~~~~

只有在计划运行智能体或推理示例时，才安装 agentic 栈：

.. code-block:: bash

   bash requirements/install.sh agentic

文档
~~~~

安装文档构建依赖：

.. code-block:: bash

   bash requirements/install.sh docs

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
