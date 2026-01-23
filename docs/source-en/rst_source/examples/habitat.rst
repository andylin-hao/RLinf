RL with Habitat Benchmark
=========================

This document provides a comprehensive guide to launching and managing the Vision-Language-Navigation Models (VLNs) training task within the RLinf framework, focusing on finetuning a VLN model in the `Habitat <https://aihabitat.org/>`_ environment.

The primary objective is to develop a model capable of performing robotic navigation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (navigation control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with environment feedback.

Dependency Installation
--------------------------------------

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --env habitat
   source .venv/bin/activate


VLN-CE Dataset Preparation
--------------------------

Download the scene dataset:

- For **R2R**, **RxR**: Download the MP3D scenes from `http://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip`
  and put them into the ``VLN-CE/scene_dataset`` folder.

Download the VLN-CE episodes:

- `r2r <https://drive.google.com/file/d/18DCrNcpxESnps1IbXVjXSbGLDzcSOqzD/view>`_
  (Rename ``R2R_VLNCE_v1/`` -> ``r2r/``)
- `rxr <https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view>`_
  (Rename ``RxR_VLNCE_v0/`` -> ``rxr/``)

Put them into the ``VLN-CE/datasets`` folder.

Download the GT actions:

- For every episode in the eval split, we use ``ShortestPathFollower`` to generate
  the ground-truth actions. Please download them from
  `here <https://drive.google.com/drive/folders/1RkTKfu2SIiSPQ7hvfA0Ylr6lFnu7yBB3?usp=sharing>`_ and
  put them into the ``VLN-CE/actions`` folder.

Dataset structure:

.. code:: bash

   VLN-CE
   |-- datasets
   |   |-- r2r
   |   |-- rxr
   `-- scene_dataset
       |-- mp3d
   `-- actions


Test Habitat Environment
------------------------

You can test the Habitat environment setup with:

.. code:: bash

   export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
   python examples/embodiment/debug_habitat_env.py
