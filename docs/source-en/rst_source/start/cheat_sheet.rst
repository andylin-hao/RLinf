Cheat Sheet
===========

Use this page when you already know the workflow and need the shortest path to a
working command.

Install
-------

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openvla_oft --env maniskill

Start Ray
---------

Single-node runs can start Ray locally.

.. code-block:: bash

   ray start --head

For multi-node runs, set ``RLINF_NODE_RANK`` before ``ray start`` on every node.

Run Training
------------

Launch an embodied recipe by config name.

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft

Launch an agentic or reasoning recipe from its example directory.

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh

Evaluate
--------

Use the unified evaluation entry point for embodied benchmarks.

.. code-block:: bash

   bash evaluations/run_eval.sh libero/libero_spatial_openpi_pi05_eval

Next Steps
----------

- :doc:`Installation <installation>` — set up RLinf and optional dependencies.
- :doc:`Embodied Quickstart <vla>` — run a VLA training recipe.
- :doc:`Agentic Quickstart <llm>` — run a reasoning training recipe.
- :doc:`Evaluation <../evaluations/index>` — run standalone embodied evaluation.
