异构软硬件集群配置
=====================

RLinf 支持在具有异构硬件和软件环境的集群上运行。例如，你可以：

* 在支持光线追踪的 GPU（如 RTX 4090）上运行高保真模拟器；
* 在大显存计算 GPU（如 A100）上进行训练；
* 在无 GPU 的节点上运行搜索 Agent。
* 在存在特殊硬件（如机械臂）的节点上运行机器人控制器

要搭建这样的异构环境，只需要在 YAML 配置文件中正确配置 ``cluster`` 段落即可。


集群配置总览
------------

``cluster`` 段落描述了：**你拥有哪些机器**，以及 **RLinf 应该如何在这些机器上放置各个组件（actor、rollout、env、agent 等）**。

从高层来看，你需要指定：

* 集群中节点（node）的总数量；
* 一组 *节点组（node group）*，每个节点组内部拥有相同的硬件 / 运行环境；
* 一条 *组件放置（component placement）* 规则，用来把逻辑组件映射到具体的硬件资源（GPU、机器人或仅仅是节点）。


示例配置
~~~~~~~~

下面的示例（改写自 ``ClusterConfig`` 的文档字符串）展示了一个具有异构硬件、并通过
``env_configs`` 配置每个节点软件环境的集群：

.. code-block:: yaml

	cluster:
	  num_nodes: 18

	  component_placement:
	    actor:
	      node_group: a800
	      placement: 0-63           # 在 ``a800`` 节点组内部的硬件编号
	    rollout:
	      node_group: 4090
	      placement: 0-63           # 在 ``4090`` 节点组内部的硬件编号
	    env:
	      node_group: franka
	      placement: 0-1            # 在 ``franka`` 节点组内部的机器人硬件编号
	    agent:
	      node_group: node
	      placement: 0-1:0-199,2-3:200-399  # 节点编号 : 进程编号

	  node_groups:
	    - label: a800
	      node_ranks: 0-7
	      env_configs:
	        - node_ranks: 0-7
	          python_interpreter_path: /opt/venv/openpi/bin/python3
	          env_vars:
	            - GLOO_SOCKET_IFNAME: "eth0"

	    - label: 4090
	      node_ranks: 8-15
	      env_configs:
	        - node_ranks: 8-15
	          env_vars:
	            - GLOO_SOCKET_IFNAME: "eth1"

	    - label: franka
	      node_ranks: 16-17
	      hardware:
	        type: Franka
	        configs:
	          - robot_ip: "10.10.10.1"
	            node_rank: 16
	            camera_serials:
	              - "322142001230"
	              - "322142001231"
	          - robot_ip: "10.10.10.2"
	            node_rank: 17
	            camera_serials:
	              - "322142001232"
	              - "322142001233"


配置解释
--------

上面的配置表达了如下含义：

* ``num_nodes: 18`` —— 集群中共有 18 个节点。节点编号从 0 开始，
  并且需要在每个节点启动 Ray 之前，通过环境变量 ``RLINF_NODE_RANK``
  指定对应的节点编号。

* ``node_groups`` —— 每一项定义了一个 **节点组（node group）**，
  表示一组具备相同硬件与环境的节点。节点组包含：

  - ``label``：在 ``component_placement`` 中引用该节点组时使用的唯一字符串标识，
    例如 ``a800``、``4090``、``franka``。标签区分大小写。

    标签 ``cluster`` 与 ``node`` 由调度器保留，用户不能自定义使用。
    其中 ``node`` 是一个特殊的节点组，表示“所有节点但不关心硬件”，
    适合放置只依赖 CPU 的进程（如 agent 等）。

  - ``node_ranks``：属于该节点组的全局节点编号列表或范围。
    在示例中，``a800`` 覆盖 ``0-7``，``4090`` 覆盖 ``8-15``，``franka`` 覆盖 ``16-17``。

  - ``env_configs`` （可选）：用于描述该节点组内部不同节点子集的软件环境。
    每一项是一个 ``NodeGroupEnvConfig`` ，包含其自身的 ``node_ranks``、``env_vars`` 与 ``python_interpreter_path``：

	* 每个 ``env_configs`` 项的 ``node_ranks`` 必须是父节点组 ``node_ranks`` 的子集；同一节点组中不同 ``env_configs`` 的 ``node_ranks`` 之间不能重叠；

	* ``env_vars`` 是一组只包含单个键值对的字典列表；在同一节点上，环境变量的键必须是唯一的（不能在多个 ``env_configs`` 或不同节点组中重复设置同一个键）；
	
	* ``python_interpreter_path`` 用于指定这些节点上使用的 Python 解释器。每个节点最多只能配置一个解释器路径。

  - ``hardware``（可选）：描述该节点组上 *非加速卡* 硬件（如机器人）的结构化配置。
    其具体字段由 ``type`` 决定（例如 ``Franka``）。当指定了 ``hardware`` 时，
    该节点组被视为仅包含这一类硬件资源，并在组内为其定义 **硬件编号** （0, 1, ...）。

* 如果节点组 **未** 指定 ``hardware`` 字段，RLinf 的行为如下：

  - 如果节点上检测到加速卡硬件（GPU、NPU 等），则这些加速卡成为默认资源，
    它们的本地索引用作硬件编号；
  - 如果没有任何加速卡存在，则每个 **节点本身** 被视为一个硬件资源，
    在本节点内的硬件编号为 0。

当你在 ``component_placement`` 中引用某个 ``node_group`` 时，
其 ``placement`` 字段中写下的始终是 **该节点组内部的硬件编号** ：

* 若节点组配置了 ``hardware``，则编号指的是该类型硬件的编号
  （例如机器人 0–3）；
* 若未配置 ``hardware`` 但存在加速卡，则编号指的是自动检测到的
  加速卡索引；
* 若也没有加速卡，则编号对应的是节点本身。

当在 ``component_placement`` 中使用保留标签 ``node`` 时，
调度器不再进行任何硬件层面的放置，而是直接把编号解释为“节点编号”。
这非常适合将与硬件无关的进程（例如 agent 或只用 CPU 的 worker）
精确放到某些节点上，而不考虑 GPU 分布。


组件放置（component placement）
------------------------------

组件放置描述的是：**每个组件要启动多少个进程，以及这些进程应该放到哪里**。
这通过 ``cluster.component_placement`` 字段来完成。

对应的解析逻辑由 :class:`rlinf.scheduler.placement.ComponentPlacement`
实现，下面对其语法进行总结。


基本形式
~~~~~~~~

有两种等价的写法。

1. **简写形式**——直接把组件名映射到资源编号：

.. code-block:: yaml

	cluster:
	  num_nodes: 1
	  component_placement:
	    actor,inference: 0-7

在这里，``actor`` 与 ``inference`` *共用同一条放置规则*。
字符串 ``0-7`` 被解释为一段 **资源编号** 范围。
RLinf 会为每个组件分别创建 8 个进程（进程编号为 ``0-7``），
并将它们平均映射到资源 ``0-7`` 上。

2. **节点组形式**——显式选择某个节点组：

.. code-block:: yaml

	cluster:
	  num_nodes: 2
	  component_placement:
	    actor:
	      node_group: a800
	      placement: 0-8
	    rollout:
	      node_group: 4090
	      placement: 0-8
	    env:
	      node_group: robot
	      placement: 0-3:0-7
	    agent:
	      node_group: node
	      placement: 0-1:0-200,2-3:201-511

其含义是：

* ``actor`` 使用节点组 ``a800`` 中编号为 ``0-8`` 的加速卡；
* ``rollout`` 使用节点组 ``4090`` 中编号为 ``0-8`` 的加速卡；
* ``env`` 使用节点组 ``robot`` 中编号为 ``0-3`` 的机器人硬件，
  进程编号 ``0-7`` 被平均分配到这些机器人上（每台机器人对应 2 个进程）；
* ``agent`` 使用特殊节点组 ``node``。进程 ``0-200`` 放在节点编号
  ``0-1`` 上，进程 ``201-511`` 放在节点编号 ``2-3`` 上。


resource_ranks 与 process_ranks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

每个放置条目的通用形式为::

	resource_ranks[:process_ranks]

* ``resource_ranks``：要使用的物理资源（GPU、机器人或节点）的编号，
  支持的写法包括：

  - ``a-b`` —— 闭区间整数范围，例如 ``0-3`` 表示 0,1,2,3；
  - ``all`` —— 表示所选节点组中的全部有效资源。

  “资源”的具体含义由节点组决定：

  - 若节点组声明了某类 ``hardware``，则编号指这一类硬件
    （例如机器人索引）；
  - 若未声明 ``hardware`` 但存在加速卡，则编号指加速卡索引；
  - 若既没有声明 ``hardware`` 又没有加速卡，则编号指节点索引。

* ``process_ranks``：要放置到这些资源上的 **组件进程编号**，
  语法与 ``resource_ranks`` 相同，但不能写成 ``all``。

  如果省略 ``process_ranks``，RLinf 会自动分配一段连续的进程编号，
  其长度与资源数量相同。例如有如下两段::

	 0-3,4-7

  则第一段隐式对应进程 ``0-3``，第二段对应进程 ``4-7``。

可以通过逗号连接多段配置，并混合使用显式/隐式 ``process_ranks``::

	 0-1:0-3,3-5,7-10:7-14

其含义为：

* 进程 ``0-3`` 均匀分配到资源 ``0-1`` 上；
* 进程 ``4-6`` 被隐式分配到资源 ``3-5``（每个资源对应一个进程，
  由调度器自动推断）；
* 进程 ``7-14`` 均匀分配到资源 ``7-10`` 上。

对于同一组件，所有进程编号必须从 ``0`` 到 ``N-1`` **连续且互不重复**
（其中 ``N`` 为该组件的总进程数），否则放置解析会报错。

此外，对于每一对 ``resource_ranks:process_ranks``，资源数量与进程数量
必须满足“互为整数倍”的关系：

* 要么一个进程占用多个资源；
* 要么多个进程共享同一个资源。


放置策略如何执行
------------------

在内部，:class:`rlinf.scheduler.placement.ComponentPlacement`
会解析放置字符串，并根据所选节点组是否有硬件资源来决定使用哪种具体策略：

* 如果节点组 **没有专门的硬件资源**（既无机器人也无加速卡），
  RLinf 使用 :class:`rlinf.scheduler.placement.NodePlacementStrategy`
  仅通过节点编号进行放置。每个节点被视为一个资源，单个进程不能跨节点运行。

* 如果节点组 **有硬件资源**（如加速卡或自定义硬件），
  RLinf 使用 :class:`rlinf.scheduler.placement.FlexiblePlacementStrategy`
  将每个进程映射到某个节点上的一个或多个本地硬件编号上。

无论使用哪种策略，最终都会生成一组底层的
:class:`rlinf.scheduler.placement.Placement` 对象，其中包含：

* 该进程在整个集群中的全局进程编号；
* 该进程所在节点的全局节点编号及其在节点上的本地索引；
* 分配给该进程的硬件编号及其本地索引；
* 是否通过 ``CUDA_VISIBLE_DEVICES`` 等环境变量隐藏未分配给
  该 worker 的加速卡。

在典型使用场景下，你只需要正确编写 ``cluster`` 段落和
``component_placement`` 配置。RLinf 会基于它们自动完成异构集群中
各类组件的放置工作。