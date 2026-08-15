Adding a Task
=============

A task says what the robot is being asked to do: where the target is, how the arm
should comply on the way, what counts as success, and how the scene is
randomized between episodes. The robot, its placement, and the wrapper stack are
already handled, so a new task on hardware RLinf already supports is a config
dataclass, an env class, and one line in a table.

If the robot itself is new, do :doc:`new_robot` first -- a task needs something
to run on.

Steps
-----

1. Write the config
~~~~~~~~~~~~~~~~~~~

Add a module beside the other tasks for that robot, at
``rlinf/envs/real/<robot>/<task>.py``. Start from the robot's config dataclass
and add the fields your task needs:

.. code-block:: python

   from dataclasses import dataclass, field

   import numpy as np

   from .base import FrankaEnv, FrankaRobotConfig, compliance


   @dataclass
   class WipeConfig(FrankaRobotConfig):
       task_description: str = "wipe the surface"
       target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
       reward_threshold: np.ndarray = field(
           default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.2, 0.2, 0.2])
       )
       random_xy_range: float = 0.03

       def __post_init__(self):
           self.compliance_param = compliance(
               translational_stiffness=800,   # softer, to keep contact
               translational_clip_z=0.02,
           )
           self.target_ee_pose = np.array(self.target_ee_pose)
           self.action_scale = np.array([0.02, 0.1, 1])

State only the gains that differ. ``compliance()`` merges them onto
``COMPLIANCE_DEFAULTS`` and raises on a gain the controller does not take, which
catches a typo that would otherwise reach the impedance controller and be
ignored there.

2. Write the env
~~~~~~~~~~~~~~~~

Point the env class at your config. Often that is the whole class:

.. code-block:: python

   class WipeEnv(FrankaEnv):
       CONFIG_CLS = WipeConfig

Override a hook when the task needs different behavior. ``go_to_rest`` is the
common one, because homing from a task's end pose is task-specific -- peg
insertion lifts clear of the slot first, or the peg catches on the way up:

.. code-block:: python

       def go_to_rest(self, joint_reset=False):
           reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
           reset_pose[2] += 0.05
           self._interpolate_move(reset_pose, timeout=1)
           super().go_to_rest(joint_reset)

3. Register it
~~~~~~~~~~~~~~

Add one entry to the robot's ``TASKS`` table in
``rlinf/envs/real/<robot>/__init__.py``, naming the env class and the wrapper
stack its action space needs:

.. code-block:: python

   from .wipe import WipeEnv

   TASKS = {
       ...
       "WipeEnv-v1": (WipeEnv, apply_single_arm_wrappers),
   }

``register_tasks`` builds the entry point and registers the id with Gymnasium.
Single-arm Franka and Turtle2 envs take ``apply_single_arm_wrappers``; the
dual-arm Franka envs take ``apply_dual_franka_joint_wrappers``.

The gym id goes into user configs and dataset metadata, so pick it once and leave
it alone.

4. Add the env config
~~~~~~~~~~~~~~~~~~~~~

Add a YAML under ``examples/embodiment/config/env/`` describing the hardware and
the task fields. ``override_cfg`` carries whatever your config dataclass defines:

.. code-block:: yaml

   env_type: realworld
   init_params:
     id: "WipeEnv-v1"      # the gym id you registered
     num_envs: null
   teleop_device: spacemouse
   override_cfg:
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
     random_xy_range: 0.03

5. Check it
~~~~~~~~~~~

Confirm the id registers and its entry point resolves before involving hardware:

.. code-block:: python

   from rlinf.envs.real import RealWorldEnv  # registers every task
   from gymnasium.envs.registration import registry

   assert "WipeEnv-v1" in registry

``tests/unit_tests/test_real_env.py`` asserts this for every shipped task; add
yours to ``EXPECTED_IDS`` there.

What You Do Not Need to Write
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Concern
     - Where it already lives
   * - Connecting and placing hardware
     - ``Robot.connect``; see :doc:`../concepts/robotics`.
   * - Teleoperation
     - ``teleop_device`` in the env config selects one; the wrapper stack builds
       it.
   * - Marking reward or ending an episode by hand
     - ``keyboard_reward_wrapper`` in the env config.
   * - Relative frames, Euler conversion, gripper narrowing
     - ``real/transforms/``, applied by the wrapper stack.
   * - Impedance gains that every task shares
     - ``COMPLIANCE_DEFAULTS``; state only your deltas.

Adding a Wrapper Instead
------------------------

If what you need is behavior around a rollout rather than a new task, add it to
the family that matches what it changes: ``teleop/`` if it replaces the action,
``transforms/`` if it rewrites how an observation or action is expressed, and
``episode/`` if it decides when a rollout starts, ends, or what it scored. A new
teleop device implements ``read``; a new keyboard mode subclasses
``KeyboardSession``. :doc:`../concepts/realworld_envs` describes both.
