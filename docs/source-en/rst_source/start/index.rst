Get Started
===========

Welcome to RLinf. This guide takes you from install to your first training run,
then points you to what comes next.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Install
      :link: installation
      :link-type: doc

      Install the embodied stack with Docker or a custom environment.

   .. grid-item-card:: Quick Start
      :link: vla
      :link-type: doc

      Train OpenVLA on ManiSkill3 with PPO.

   .. grid-item-card:: Launch & Scale
      :link: ../guides/launch-scale/index
      :link-type: doc

      Move from a single machine to multi-node or real-world runs.

Requirements
------------

The following configuration has been extensively tested.

.. list-table:: Hardware
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Configuration
   * - GPU
     - 8xH100 per node
   * - CPU
     - 192 cores per node
   * - Memory
     - 1.8TB per node
   * - Network
     - NVLink + RoCE / IB 3.2 Tbps
   * - Storage
     - | 1TB local storage for single-node experiments
       | 10TB shared storage (NAS) for distributed experiments

.. list-table:: Software
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Version
   * - Operating System
     - Ubuntu 22.04
   * - NVIDIA Driver
     - 535.183.06
   * - CUDA
     - 12.4
   * - Docker
     - 26.0.0
   * - NVIDIA Container Toolkit
     - 1.17.8

What's Next
-----------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Examples
      :link: ../examples/index
      :link-type: doc

      Browse end-to-end recipes in the example gallery.

   .. grid-item-card:: Evaluation
      :link: ../evaluations/index
      :link-type: doc

      Measure success rates on benchmarks.

   .. grid-item-card:: Concepts
      :link: ../concepts/index
      :link-type: doc

      Understand the execution model.

   .. grid-item-card:: Guides
      :link: ../guides/index
      :link-type: doc

      Configure launches, logging, checkpoints, and clusters.

   .. grid-item-card:: Why RLinf
      :link: ../resources/why_rlinf
      :link-type: doc

      The design, performance, and SOTA results behind RLinf.

Command Reference
-----------------

.. grid:: 1 1 1 1
   :gutter: 2

   .. grid-item-card:: Cheat Sheet
      :link: cheat_sheet
      :link-type: doc

      Jump to the most-used commands after you know the workflow.

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   vla
   cheat_sheet
