Installation
============

Start with the embodied stack unless you know you need the agentic / reasoning
dependencies. The embodied path is the shortest route to the quickstart in this
section.

Fast Path: Embodied
-------------------

Use Docker when you want the most reproducible setup:

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

Use a custom Python environment when the Docker image does not match your
machine:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

Choose an Install Target
------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Target
     - Use it when
   * - ``embodied``
     - You run VLA / robotics examples. Pass ``--model`` and ``--env``.
   * - ``agentic``
     - You run agentic or reasoning examples that need Megatron, SGLang, or vLLM.
   * - ``docs``
     - You build the Sphinx documentation locally.

Common embodied examples:

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   bash requirements/install.sh embodied --model openpi --env libero

Run ``bash requirements/install.sh --help`` for the complete model and
environment list.

Detailed Options
----------------

Docker
~~~~~~

- Keep ``-e NVIDIA_DRIVER_CAPABILITIES=all`` for GPU rendering in embodied
  environments.
- Do not mount over ``/root`` or ``/opt``; those directories contain assets and
  virtual environments in the image.
- If your platform changes ``$HOME`` or remounts ``/root``, run ``link_assets``
  inside the container before launching an example.
- Switch model environments with ``source switch_env openvla``,
  ``source switch_env openvla-oft``, or ``source switch_env openpi``.

Custom Environment
~~~~~~~~~~~~~~~~~~

- Use ``--venv <dir>`` to choose the virtual environment directory.
- Use ``--use-mirror`` for faster downloads from mainland China.
- Use ``--python <version>`` only when a package requires it. The default is
  Python 3.11.14; some environments such as ``behavior`` and ``d4rl`` require
  Python 3.10.
- Use ``--torch <version>`` only when you need a different PyTorch wheel.
- Use ``--platform amd`` or ``--platform ascend`` for experimental non-NVIDIA
  installs. See :doc:`../guides/amd_rocm` and :doc:`../guides/ascend_cann`.

Agentic / Reasoning
~~~~~~~~~~~~~~~~~~~

Install the agentic stack only when you plan to run agentic or reasoning
examples:

.. code-block:: bash

   bash requirements/install.sh agentic

Documentation
~~~~~~~~~~~~~

Install the documentation dependencies with:

.. code-block:: bash

   bash requirements/install.sh docs

Verify
------

After activation, verify that RLinf and Ray are visible in the environment:

.. code-block:: bash

   python -c "import rlinf; print(rlinf.__file__)"
   ray --version

Next Steps
----------

- :doc:`Run the VLA quickstart <vla>`.
- :doc:`Scale beyond one machine <../guides/launch-scale/index>`.
- :doc:`Open the cheat sheet <cheat_sheet>` when you only need commands.
