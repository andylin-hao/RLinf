速查表
======

当你已经熟悉流程，只需要最短可运行命令时，使用本页。

安装
----

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openvla_oft --env maniskill

启动 Ray
--------

单节点运行可以在本机启动 Ray。

.. code-block:: bash

   ray start --head

多节点运行时，必须在每个节点执行 ``ray start`` 之前设置 ``RLINF_NODE_RANK``。

运行训练
--------

通过配置名启动具身智能训练。

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft

从对应示例目录启动智能体或推理训练。

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh

运行评测
--------

使用统一评测入口运行具身智能 benchmark。

.. code-block:: bash

   bash evaluations/run_eval.sh libero/libero_spatial_openpi_pi05_eval

下一步
------

- :doc:`安装 <installation>` — 安装 RLinf 和可选依赖。
- :doc:`具身智能快速开始 <vla>` — 运行 VLA 训练示例。
- :doc:`智能体快速开始 <llm>` — 运行推理训练示例。
- :doc:`评测 <../evaluations/index>` — 运行独立具身智能评测。
