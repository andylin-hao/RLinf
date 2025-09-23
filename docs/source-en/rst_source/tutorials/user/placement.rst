GPU Resource Placement Strategy
========================================

The placement module defines how workers are distributed across the hardware resources (nodes and GPUs) of the cluster. 
A **PlacementStrategy** encapsulates a particular policy for assigning workers to GPUs and nodes. 
This allows flexibility in how a set of workers (e.g., a `WorkerGroup`) utilizes the available GPUs. 
For each strategy, a list of `Placement` objects is returned by the `get_placement(cluster, isolate_accelerator)` method. Each `Placement` includes:

+---------------------------+-----------------------------------------------+
| Property                  | Description                                   |
+===========================+===============================================+
| `rank`                    | Unique global worker index                    | 
+---------------------------+-----------------------------------------------+
| `node_id`                 | Identifier of the node where the worker runs  |
+---------------------------+-----------------------------------------------+
| `node_rank`               | Index of the node within the cluster          |
+---------------------------+-----------------------------------------------+
| `local_accelerator_id`    | Accelerator index of the worker on that node  |
+---------------------------+-----------------------------------------------+
| `local_rank`              | Worker’s index among workers on the same node |
+---------------------------+-----------------------------------------------+
| `local_world_size`        | Total number of workers on the same node      |
+---------------------------+-----------------------------------------------+
| `visible_accelerators`    | List of accelerator IDs visible to the worker |
+---------------------------+-----------------------------------------------+
| `isolate_accelerator`     | Whether worker is restricted to assigned accelerators  |
+---------------------------+-----------------------------------------------+

PackedPlacementStrategy
-----------------------

This placement strategy can:

* **Pack accelerators contiguously** (`stride = 1`) – the classic “close-packed”
  behaviour; or
* **Assign accelerators in a fixed-stride pattern** (`stride > 1`) – the former
  “strided” mode, e.g. `0, 2, 4` for `stride = 2`.

Required inputs
~~~~~~~~~~~~~~~~~

* ``start_accelerator_id`` – first *global* accelerator index to consider.  
* ``end_accelerator_id`` – last *global* accelerator index (inclusive).  
* ``num_accelerators_per_process`` – number of accelerators given to **each** process.  
* ``stride`` – distance between successive accelerators *inside one process*
  (``1`` = contiguous; ``>1`` = strided).  
* ``isolate_accelerator`` – whether to set ``CUDA_VISIBLE_DEVICES`` so the
  process only “sees” its assigned accelerators (defaults to ``True``).  

Placement principle
~~~~~~~~~~~~~~~~~~~~~

Starting at ``start_accelerator_id`` the scheduler walks forward through the
accelerator IDs:

1. **Allocate a block** of
   ``num_accelerators_per_process × stride`` consecutive global IDs.  
2. **Select every stride-th** ID inside that block; those become the
   accelerators for the current rank  
   (e.g. ``[0, 1, 2, 3]`` → stride 2 → ``[0, 2]``).  
3. Repeat until **all** IDs up to ``end_accelerator_id`` are consumed, wrapping
   to the next node when a node’s accelerator count is exceeded.

The constructor enforces  
``total_accelerators % (num_accelerators_per_process × stride) == 0`` so every process
obtains a full accelerator set without spilling across node boundaries.

When ``isolate_accelerator=True`` the generated placement also sets
``CUDA_VISIBLE_DEVICES`` to the local (node-relative) accelerator list, ensuring
library calls see only those devices.

Purpose
~~~~~~~~~~~~~

* **Contiguous mode** (`stride = 1`) remains the default for
  data-parallel or per-rank model-parallel jobs that expect sequential
  device IDs.
* **Strided mode** (`stride > 1`) is useful to colocation placement of rollout and training models that places model parallel source 
  ranks on the same accelerators, enabling fast zero-copy-cudaIPC-based weight synchronization

FlexiblePlacementStrategy
-----------------------

This placement strategy allows arbitrary accelerator IDs to be assigned to each process, specified as a list of accelerator ID lists, each accelerator ID list contains the global accelerator IDs assigned to a worker process.

Required inputs
~~~~~~~~~~~~~~~~~

* ``accelerator_id_lists`` – a list of lists of global accelerator IDs, each inner list specifies the accelerators assigned to a worker process.

Placement principle
~~~~~~~~~~~~~~~~~~~~~
The scheduler iterates through the provided ``accelerator_id_lists``, and for each inner list, it assigns the specified global accelerator IDs to a worker process. The node ID is determined by the first accelerator ID in the inner list, and the local accelerator IDs are calculated relative to that node.

Purpose
~~~~~~~~~~~~~
This strategy provides maximum flexibility, allowing users to define exactly which accelerators each worker should use, regardless of contiguity or stride. It is particularly useful in scenarios where specific accelerator assignments are required due to hardware topology or other constraints.


Example
---------

.. autoclass:: rlinf.scheduler.placement.PlacementStrategy
   :no-members:
   :no-inherited-members:
   :exclude-members: __init__, __new__

Summary
--------

In summary, the **placement** component ensures that workers are deployed in a way that matches the desired parallel execution pattern and resource usage policy, which is crucial for performance and correctness in distributed training.

