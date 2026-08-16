Adding a Task
=============

On hardware RLinf already supports, a new task takes a config dataclass, an env
class, and one row in the task table. The task records the target and compliance
settings as well as success and reset rules. Robot construction and placement
remain unchanged, as does the wrapper stack.

If the robot itself is new, add it through :doc:`new_robot` before defining its
task.

Steps
-----

1. Write the config
~~~~~~~~~~~~~~~~~~~

Create ``rlinf/envs/real/<robot>/<task>.py`` beside the other tasks for that
robot. Inherit its config dataclass and add the fields required by your task:

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
``COMPLIANCE_DEFAULTS`` and raises on any gain the controller does not accept. A
misspelled gain stops here instead of reaching the impedance controller and
being ignored.

2. Write the env
~~~~~~~~~~~~~~~~

Set the env class's config type. For many tasks, that is the whole class:

.. code-block:: python

   class WipeEnv(FrankaEnv):
       CONFIG_CLS = WipeConfig

Override a hook only when the task needs different behavior. ``go_to_rest`` is
the common case because homing depends on the task's end pose. Peg insertion,
for example, lifts clear of the slot first; otherwise the peg catches on the way
up:

.. code-block:: python

       def go_to_rest(self, joint_reset=False):
           reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
           reset_pose[2] += 0.05
           self._interpolate_move(reset_pose, timeout=1)
           super().go_to_rest(joint_reset)

3. Register it
~~~~~~~~~~~~~~

Add one entry to the robot's ``TASKS`` table in
``rlinf/envs/real/<robot>/__init__.py``. Name the env class and the wrapper stack
required by its action space:

.. code-block:: python

   from .wipe import WipeEnv

   TASKS = {
       ...
       "WipeEnv-v1": (WipeEnv, apply_single_arm_wrappers),
   }

``register_tasks`` builds the entry point and registers the id with Gymnasium.
Single-arm Franka and Turtle2 envs take ``apply_single_arm_wrappers``; the
dual-arm Franka envs take ``apply_dual_franka_joint_wrappers``.

User configs and dataset metadata store the gym id. Changing it later breaks
those references. Choose the name before collecting data.

4. Add the env config
~~~~~~~~~~~~~~~~~~~~~

Add a YAML file under ``examples/embodiment/config/env/``. Keep the registered
id at the path shown below and describe the task fields in ``override_cfg``;
these fields come from your config dataclass:

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

Before connecting hardware, confirm that the id is registered and its entry
point resolves:

.. code-block:: python

   from rlinf.envs.real import RealWorldEnv  # registers every task
   from gymnasium.envs.registration import registry

   assert "WipeEnv-v1" in registry

``tests/unit_tests/test_real_env.py`` makes the same assertion for every shipped
task. Add your id to ``EXPECTED_IDS`` there.

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
     - ``real/wrappers/transforms/``, applied by the wrapper stack.
   * - Impedance gains that every task shares
     - ``COMPLIANCE_DEFAULTS``; state only your deltas.

Adding a Teleop Device
----------------------

A device and what its readings mean are two separate things, so adding one is a
part, a binding, and an entry pairing them.

The part reads hardware and nothing else, in
``rlinf/robotics/parts/teleop/devices.py``:

.. code-block:: python

   class Pedal(TeleopPart):
       def _open(self):
           from .readers.pedal import PedalReader

           return PedalReader(port=self._port)

       @property
       def observation_features(self):
           return {"pressed": {"shape": (1,), "dtype": "bool"}}

       def get_observation(self):
           return {"pressed": np.asarray([self._reader.is_pressed()])}

Opening in ``_open`` rather than ``__init__`` is what lets the device be
declared on one machine and built on another.

The binding says which parts of a robot's action it fills, in
``rlinf/robotics/teleop/bindings.py``:

.. code-block:: python

   class PedalGripperBinding(TeleopBinding):
       PRODUCES = ("end_effector",)

       def action(self, reading, context):
           return {"end_effector": np.array([-1.0 if reading["pressed"] else 1.0])}

       def is_driving(self, reading):
           return bool(reading["pressed"])

Then pair them in ``DEVICES`` in
``rlinf/envs/real/wrappers/teleop/builder.py``, and name the device in any env
that can drive it:

.. code-block:: python

   DEVICES = {..., "pedal": _pedal}

Nothing else changes. The stack builder never learns the device exists; a config
naming it gets it, and a robot without an ``end_effector`` refuses the rig rather
than half-building one.

Adding a Wrapper Instead
------------------------

When the new behavior surrounds a rollout, add a wrapper. Put action replacement
in ``teleop/`` and representation changes in ``transforms/``;
rollout boundaries and scores belong in ``episode/``. A new teleop device
implements ``read``. For a new keyboard mode, subclass ``KeyboardSession``.
:doc:`../concepts/realworld_envs` describes both extension points.
