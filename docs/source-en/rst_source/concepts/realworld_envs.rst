Real-World Environment Model
============================

A real-world environment joins a robot to a task, then wraps that pair with the
behavior needed during a rollout. The robot supplies motion and sensing. The
task owns target poses and reward rules, including reset behavior. Wrappers
handle operator intervention and manual outcome labels; they also adapt how
observations or actions are represented.

If you need the part and placement model underneath the robot, start with
:doc:`robotics`. Here we stay at the environment boundary.

A Task Is a Config and a Few Overrides
--------------------------------------

Look in ``rlinf/envs/real/franka/``. Each Franka task has its own module beside
the shared ``base.py``. Most tasks start with a dataclass that records the target
and compliance settings; the env class adds any reset behavior specific to that
task:

.. code-block:: python

   @dataclass
   class PegInsertionConfig(FrankaRobotConfig):
       task_description: str = "peg and insertion"
       target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
       random_xy_range: float = 0.05

       def __post_init__(self):
           # Only what differs from the shared impedance gains.
           self.compliance_param = compliance(translational_stiffness=2000)
           ...


   class PegInsertionEnv(FrankaEnv):
       CONFIG_CLS = PegInsertionConfig

       def go_to_rest(self, joint_reset=False):
           # Lift clear of the slot before homing, or the peg catches.
           ...

``compliance()`` merges your overrides onto ``COMPLIANCE_DEFAULTS`` and rejects
any gain the controller does not accept. A misspelled key fails here instead of
reaching the impedance controller and being ignored. Peg insertion states one
gain; bin relocation states eleven. Everything else in the config describes the
task itself: poses, reward thresholds, and reset randomization.

Registering It Is One Line
--------------------------

The task table records two things for each task: the env class constructed from
the worker's config and the wrapper stack required by that robot's action space.
One row is enough:

.. code-block:: python

   TASKS = {
       "FrankaEnv-v1": (FrankaEnv, apply_single_arm_wrappers),
       "PegInsertionEnv-v1": (PegInsertionEnv, apply_single_arm_wrappers),
       "DualFrankaTCPEnv-v1": (DualFrankaTCPEnv, apply_dual_franka_joint_wrappers),
   }

   _ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

``register_tasks`` generates each Gymnasium entry point and registers it. User
configs and dataset metadata both store the gym id. Renaming it later leaves
those references stale.

Three Kinds of Wrapper
----------------------

What a wrapper changes determines its package:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Package
     - What its wrappers change
   * - ``teleop/``
     - The action itself. An operator takes over and their command replaces the
       policy's for as long as they are driving.
   * - ``transforms/``
     - How an observation or action is written down, never what it means. A
       relative frame moves the same motion into end-effector coordinates.
   * - ``episode/``
     - When a rollout starts or ends, and what it scored. These are judgements
       made by the person watching; no sensor reports them.

``wrappers.py`` reads the env config, selects the teleop device, and composes the
wrapper stack:

.. code-block:: python

   env = apply_single_arm_wrappers(PegInsertionEnv(...), cfg)

Teleop: One Wrapper, Many Devices
---------------------------------

Teleop devices differ in how they read the operator, but they all return the
same answer: what the operator would do right now. A device only implements that
read:

.. code-block:: python

   class SpaceMouseTeleop(TeleopDevice):
       def read(self, env, policy_action) -> TeleopSample:
           expert, buttons = self.expert.get_action()
           return TeleopSample(
               action=expert,
               active=bool(np.linalg.norm(expert) > 0.001),
               info={"left": buttons[0], "right": buttons[1]},
           )

``TeleopIntervention`` handles the rest. It keeps control active between samples
for a hold window and chooses the fallback after release. Dataset collectors
read the resulting action from ``intervene_action``.

``active`` says whether the operator is driving; the action itself does not.
Most devices report small residual motion even at rest, and each device sets its
own threshold. A held control such as a PICO grip or trigger marks the interval
exactly and sets ``timeout = 0``. Carrying that signal over for another half
second would continue commanding the robot after release.

A leader arm needs to send targets continuously, much faster than ``env.step``
runs. :class:`StreamingTeleopDevice` gives it a dedicated thread. The thread
pauses while the env drives the robot home and aligns before sending its first
target; shutdown joins it.

Choosing a device is one config key:

.. code-block:: yaml

   env:
     eval:
       teleop_device: spacemouse   # or gello, pico, gello_joint, none
       gello_port: /dev/serial/by-id/...

Devices and Readers Are Separate
--------------------------------

Inside ``teleop/``, device I/O is kept apart from env-specific action
conversion. ``devices/`` contains the readers that talk to a serial port or a
headset and imports no Gymnasium. ``adapters.py`` turns their readings into
actions for a particular env.

You can therefore check a leader arm's wiring without involving a robot:

.. code-block:: bash

   python -m rlinf.envs.real.teleop.devices.gello --port /dev/ttyUSB0

A teleop device is not a :class:`~rlinf.robotics.parts.base.RobotPart`. A part
describes a component as the policy sees it. A leader arm never appears in that
view: policies do not observe it, and robots do not include it in their
composition. It reads the operator, not the robot, and belongs on the
environment side.

Episode Control Is Not Teleop
-----------------------------

The operator may also mark a success or abort a take, and can switch policies
mid-rollout. None of those choices changes the action. Their wrappers live in
``episode/`` and share :class:`KeyboardSession`. It owns the listener and drops
repeat presses inside a debounce window; on reset, it clears the queue before a
pedal tapped during homing can start the next episode.

A new mode reads ``presses()`` and decides what each key means:

.. code-block:: python

   class KeyboardRLTPolicySwitchWrapper(KeyboardSession):
       def step(self, action):
           obs, reward, terminated, truncated, info = self.env.step(action)
           for key in self.presses():
               if key == "b":
                   self._rlt_switch_flags = True
           info["rlt_switch_flags"] = self._rlt_switch_flags
           return obs, reward, terminated, truncated, info

Where the Code Lives
--------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Contents
   * - ``real/<robot>/``
     - One module per task, plus ``base.py`` with the machinery they share and
       ``__init__.py`` with the ``TASKS`` table.
   * - ``real/teleop/devices/``
     - Readers for GELLO, gloves, keyboards, PICO, and SpaceMouse. No Gymnasium.
   * - ``real/teleop/``
     - ``intervention.py``, ``adapters.py``, ``streaming.py``, ``pico.py``, and
       ``config.py`` for device selection.
   * - ``real/transforms/``
     - Relative frames, quaternion-to-Euler, gripper narrowing.
   * - ``real/episode/``
     - Keyboard sessions: reward and done, start and end, eval control, policy
       switch, leader-follower.
   * - ``real/wrappers.py``
     - The stack builders that compose the three families.
   * - ``real/registry.py``
     - ``task_factory`` and ``register_tasks``.
   * - ``real/env.py``
     - ``RealWorldEnv``, the vectorized env the framework instantiates from
       ``env_type: realworld``.
   * - ``real/robot_task_env.py``
     - ``RobotTask`` and ``RobotTaskEnv`` define the boundary between task logic
       and hardware.

Next
----

- :doc:`Adding a Task <../extending/new_task>`: follow the step-by-step guide.
- :doc:`Robotics Model <robotics>`: how the robot underneath is composed.
