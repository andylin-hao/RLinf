Concepts
========

Use Concepts to understand the execution model before you tune placement,
workers, or communication.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: RLinf Execution Flow
      :link: execution_flow
      :link-type: doc

      How a job runs end to end — code flow, processes, and core concepts.

   .. grid-item-card:: Worker and WorkerGroup
      :link: worker
      :link-type: doc

      The unit of computation and the handle that drives a group of workers.

   .. grid-item-card:: M2Flow Programming Flow
      :link: flow
      :link-type: doc

      The macro-to-micro model that decouples logic from scheduling.

   .. grid-item-card:: Channel
      :link: channel
      :link-type: doc

      Asynchronous channels for inter-worker data exchange.

   .. grid-item-card:: Placement
      :link: placement
      :link-type: doc

      How workers map onto nodes and GPUs.

   .. grid-item-card:: Execution Modes
      :link: execution_modes
      :link-type: doc

      Collocated, disaggregated, and hybrid placement and their trade-offs.

   .. grid-item-card:: Cluster
      :link: cluster
      :link-type: doc

      The cluster abstraction and resource model.

   .. grid-item-card:: Collective Communication
      :link: collective
      :link-type: doc

      Collective operations and asynchronous work handles.

   .. grid-item-card:: Supported Environments
      :link: supported_envs
      :link-type: doc

      The environment interface and the simulators RLinf supports.

   .. grid-item-card:: Replay Buffer
      :link: replay_buffer
      :link-type: doc

      Trajectory replay buffer design and sampling.

.. toctree::
   :hidden:

   RLinf Execution Flow <execution_flow>
   Worker and WorkerGroup <worker>
   M2Flow Programming Flow <flow>
   Channel <channel>
   Placement <placement>
   Execution Modes <execution_modes>
   Cluster <cluster>
   Collective Communication <collective>
   Supported Environments <supported_envs>
   Replay Buffer <replay_buffer>
