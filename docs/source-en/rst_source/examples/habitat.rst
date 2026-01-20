RL with Habitat Benchmark
=========================

This document provides a comprehensive guide to launching and managing the Vision-Language-Navigation Models (VLNs) training task within the RLinf framework, focusing on finetuning a VLN model in the `Habitat <https://aihabitat.org/>`_ environment.

The primary objective is to develop a model capable of performing robotic navigation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (navigation control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with environment feedback.

Habitat-Sim & Habitat-Lab Installation
--------------------------------------

**1. Prepare Habitat Environment**

.. code:: bash

   # Prepare habitat env
   cd /opt/venv
   uv venv habitat --python 3.10
   source /opt/venv/habitat/bin/activate
   cd /data
   git clone https://github.com/RLinf/RLinf.git
   cd /data/RLinf
   uv sync --active --extra embodied

**2. Clone Required Repositories**

.. code:: bash

   # Clone Required Repositories
   cd /opt
   git clone https://github.com/facebookresearch/habitat-sim.git
   git clone https://github.com/facebookresearch/habitat-lab.git

**3. Initialize and Patch Habitat-Sim**

.. code:: bash

   cd /opt/habitat-sim
   git submodule update --init --recursive
   git checkout v0.3,3

   # Correct the CMake File
   sed -i 's/^cmake_minimum_required.*$/cmake_minimum_required(VERSION 3.5)/' src/deps/zstd/build/cmake/CMakeLists.txt
   sed -i 's/^cmake_minimum_required.*$/cmake_minimum_required(VERSION 3.5)/' src/deps/assimp/CMakeLists.txt

**4. Install System-Level Ninja and Configure CMake**

.. code:: bash

   # Install System-Level Ninja
   apt-get update && apt-get install -y ninja-build
   uv pip install ninja
   export CMAKE_MAKE_PROGRAM=/usr/bin/ninja
   export CMAKE_POLICY_VERSION_MINIMUM=3.5

**5. Install Habitat-Sim**

.. code:: bash

   # Habitat-Sim Installation
   uv pip install . --config-settings="--build-option=--headless" --config-settings="--build-option=--with-bullet"
   uv pip install build/deps/magnum-bindings/src/python/

**6. Install Habitat-Lab and Habitat-Baselines**

.. code:: bash

   # Habitat-lab Installation
   cd /opt/habitat-lab
   git checkout v0.3.3
   uv pip install -e habitat-lab
   uv pip install -e habitat-baselines


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
