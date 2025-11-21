Heterogenous Software and Hardware Setup
==========================================

RLinf supports running on nodes with heterogeneous hardware and software setup, e.g., running simulators on ray tracing-capable GPUs (like RTX 4090), training on compute GPUs with larger GPU memory (like A100), search agent on CPU-only nodes, and/or robot controllers on nodes with special hardware like robotic arms.

To set up such a heterogeneous environment, all you need to do is configuring the `cluster` section of the YAML config file as follows.


Cluster configuration
---------------------

The ``cluster`` section describes **what machines you have** and **how RLinf
should place each component (actor, rollout, env, agent, etc.) on them**.

At a high level, you specify:

* the total number of nodes in the cluster,
* a set of *node groups* that share the same hardware / environment, and
* a *component placement* rule that maps logical components to hardware
	resources (GPUs, robots, or just nodes).

An example
~~~~~~~~~~~~~~~~~

The following example (adapted from the ``ClusterConfig`` docstring) shows a
cluster with heterogeneous hardware and per-node software environments
configured via ``env_configs``:

.. code-block:: yaml

	cluster:
	  num_nodes: 18

	  component_placement:
	    actor:
	      node_group: a800
	      placement: 0-63           # hardware ranks within ``a800``
	    rollout:
	      node_group: 4090
	      placement: 0-63           # hardware ranks within ``4090``
	    env:
	      node_group: franka
	      placement: 0-1            # robot hardware ranks within ``franka``
	    agent:
	      node_group: node
	      placement: 0-1:0-199,2-3:200-399  # node ranks : process ranks

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

Interpretation
~~~~~~~~~~~~~~

The above configuration encodes the following ideas:

* ``num_nodes: 18`` – total number of nodes in the cluster. Node ranks
	are zero-indexed and specified via the ``RLINF_NODE_RANK`` environment
	variable when starting Ray on each node.

* ``node_groups`` – each entry defines a **node group**: a set of nodes
	with the same hardware and environment. A node group has:

	- ``label``: a unique string identifier used later in
		``component_placement`` (e.g., ``a800``, ``4090``, ``franka``).
		Labels are case sensitive.

		The labels ``cluster`` and ``node`` are reserved by the scheduler.
		``node`` is a special group that covers *all* nodes and is used for
		hardware-agnostic placement (CPU-only processes, agents, etc.).

	- ``node_ranks``: a list or range of global node ranks that belong to
		this group. In the example, ``a800`` covers ``0-7``, ``4090`` covers
		``8-15``, and ``franka`` covers ``16-17``.

	- ``env_configs`` (optional): a list of software environment
		configurations for subsets of nodes in the group. Each entry is a
		``NodeGroupEnvConfig`` with its own ``node_ranks``,
		``env_vars``, and ``python_interpreter_path``:

		* ``node_ranks`` must be a subset of the parent group's
		``node_ranks``, and different ``env_configs`` in the same group must not overlap.

		* ``env_vars`` is a list of one-key dicts; environment variable
		keys must be unique within a node group for a node.

		* ``python_interpreter_path`` is the interpreter to use on the
		specified nodes.

	- ``hardware`` (optional): structured description of *non-accelerator
		hardware* (such as robots). The structure depends on the hardware
		``type`` (for example, ``Franka``). When ``hardware`` is present,
		this node group is treated as owning exactly one hardware *type*, and
		that type defines **hardware ranks** (0, 1, ...) within the group.

* If ``hardware`` is **not** specified for a node group, RLinf behaves as
	follows:

	- If accelerator hardware (GPUs, NPUs, etc.) is detected on the nodes,
		those accelerators become the default resources, and their local
		indices are used as hardware ranks.
	- If no accelerators are present, each **node itself** is treated as a
		single hardware resource with rank 0 within that node.

When you reference a ``node_group`` in ``component_placement``, the
``placement`` string is always written in terms of **hardware ranks
within that group**:

* If ``hardware`` is present, these are explicit hardware ranks of that
	type (e.g., robots ``0-3``).
* Otherwise, they are automatically detected accelerators, if any.
* If there is no accelerator, the node itself is considered a hardware
	resource.

Using the reserved ``node`` group in ``component_placement`` disables
hardware placement entirely and interprets ranks as node ranks only. This
is useful for placing hardware-agnostic processes (such as agents or
CPU-only workers) on particular nodes regardless of available GPUs.


Component placement
-------------------

Component placement tells RLinf **how many processes each component
should run and where they should be placed**. This is done via
``cluster.component_placement``.

The syntax supported by :class:`rlinf.scheduler.placement.ComponentPlacement`
is summarized below.


Basic formats
~~~~~~~~~~~~~

There are two equivalent styles:

1. **Short form** – directly map components to resources:

.. code-block:: yaml

	cluster:
	  num_nodes: 1
	  component_placement:
	    actor,inference: 0-7

Here, ``actor`` and ``inference`` *share the same placement rule*.
The string ``0-7`` is interpreted as a range of **resource ranks**.
RLinf will create 8 processes (ranks ``0-7``) for each of these
components and evenly map them to resources ``0-7``.

2. **Node-group form** – explicitly select a node group:

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

The meaning is:

* ``actor`` uses accelerators ``0-8`` in node group ``a800``.
* ``rollout`` uses accelerators ``0-8`` in node group ``4090``.
* ``env`` uses robot hardware ``0-3`` in group ``robot``; process
	ranks ``0-7`` are evenly shared across these robots (2 processes per
	robot).
* ``agent`` uses the special group ``node``. Processes
	``0-200`` are placed on node ranks ``0-1``, and processes
	``201-511`` are placed on node ranks ``2-3``.


resource_ranks and process_ranks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each placement entry has the general form::

	 resource_ranks[:process_ranks]

* ``resource_ranks``: which physical resources to use (GPUs, robots, or
	nodes). The format supports:

	- ``a-b`` – inclusive integer range; e.g. ``0-3`` means 0,1,2,3.
	- ``all`` – all valid resources in the selected node group.

	The meaning of "resource" depends on the node group:

	- If a hardware type is specified in the node group, the ranks refer
		to that hardware (e.g., robot indices).
	- If no hardware type is specified but accelerators exist, the ranks
		are accelerator (GPU) indices.
	- If no accelerators exist, the ranks become node indices.

* ``process_ranks``: which **process ranks** of that component should be
	assigned to these resources. It uses the same range syntax as
	``resource_ranks`` but must **not** be ``all``.

	If ``process_ranks`` is omitted, RLinf automatically assigns a
	continuous block of process ranks of the same length as the number of
	resources. For example, with two entries::

		 0-3,4-7

	the first part implicitly uses process ranks ``0-3``, the second part
	uses process ranks ``4-7``.

You can combine multiple segments with commas, possibly mixing parts with
and without explicit ``process_ranks``::

	 0-1:0-3,3-5,7-10:7-14

This means:

* Processes ``0-3`` are evenly assigned to resources ``0-1``.
* Processes ``4-6`` are implicitly assigned to resources ``3-5`` (one
	process per resource, deduced by the scheduler).
* Processes ``7-14`` are evenly assigned to resources ``7-10``.

All process ranks for a component must be **continuous** from
``0`` to ``N-1`` (where ``N`` is the total number of processes for that
component), and each process rank must appear exactly once. Violating
this will raise an assertion error in the placement parser.

Additionally, for each ``resource_ranks:process_ranks`` pair, the number
of resources and the number of processes must be compatible: one must be
an integer multiple of the other. This ensures that either one process
uses multiple resources, or multiple processes share one resource.


How placement is executed
-------------------------

Internally, :class:`rlinf.scheduler.placement.ComponentPlacement`
parses the placement strings and chooses a concrete placement strategy
based on whether the selected node group has hardware:

* If the node group has **no dedicated hardware** (no robots and no
	accelerators), RLinf uses
	:class:`rlinf.scheduler.placement.NodePlacementStrategy` to place
	processes purely by node rank. Each node is treated as a single
	resource; a process cannot span multiple nodes.

* If the node group has **hardware resources** (accelerators or custom
	hardware), RLinf uses
	:class:`rlinf.scheduler.placement.FlexiblePlacementStrategy` to map
	each process to one or more local hardware ranks on exactly one node.

Both strategies produce a list of low-level :class:`rlinf.scheduler.placement.Placement`
objects, which encode:

* the global process rank in the cluster,
* the global node rank and local node index for the process,
* the selected hardware ranks and their local indices,
* whether accelerators not allocated to the worker are hidden via
	``CUDA_VISIBLE_DEVICES`` and related environment variables.

In typical user workflows, you only need to write the ``cluster``
section and ``component_placement`` correctly. RLinf will then use these
strategies to automatically realize the desired heterogeneous placement
for you.
