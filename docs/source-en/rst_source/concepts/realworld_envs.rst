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

The task table records one thing for each task: the env class to construct from
the worker's config. Which wrappers go around it is the env's own declaration,
so a task is one row:

.. code-block:: python

   TASKS = {
       "FrankaEnv-v1": FrankaEnv,
       "PegInsertionEnv-v1": PegInsertionEnv,
       "DualFrankaTcpEnv-v1": DualFrankaTcpEnv,
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

``build_stack`` reads the env config and wraps the env in what the env declares,
whichever robot it runs on:

.. code-block:: python

   env = build_stack(PegInsertionEnv(...), cfg)

Teleop: What Stays on This Side
-------------------------------

Device selection and the meaning of each reading belong to :doc:`robotics`. The
environment boundary handles two remaining decisions: how long an intervention
stays active, and how named part actions enter the environment's flat vector.

``TeleopIntervention`` keeps the latest operator action active for a short window
between samples. Without that window, the action could flicker between operator
and policy while the person is still moving. A held control such as a PICO grip
already marks the interval exactly and uses ``timeout = 0``. Any hold after
release would continue to command the robot. Dataset collectors read the action
selected by this arbitration from ``intervene_action``.

Once arbitration selects the action, its shape still has to change. A group
produces named parts, while an environment accepts one flat vector.
``ComposedTeleop`` writes each part into the layout declared by the environment.
Parts that nobody drives retain the values requested by the policy; a posed hand
stays at its last commanded position.

That declaration says what each part means, not only where it sits.
``FrankaEnv`` reads its first six numbers as a twist and ``GimArmEnv`` reads its
first six as joint angles, so a spacemouse can drive one and not the other. The
widths are identical, which is why the width is not what decides. A device whose
commands the robot would misread is refused when the group is built, and an
environment that declares nothing cannot be teleoperated at all.

A single device uses one config key. When several devices share the rig, use a
list:

.. code-block:: yaml

   env:
     eval:
       teleop: spacemouse   # or gello, pico, none
       gello_port: /dev/serial/by-id/...

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

Devices and Readers Are Separate
--------------------------------

Inside ``teleop/``, device I/O is kept apart from env-specific action
conversion. ``devices/`` contains the readers that talk to a serial port or a
headset and imports no Gymnasium. ``adapters.py`` turns their readings into
actions for a particular env.

You can therefore check a leader arm's wiring without involving a robot:

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

A teleop device *is* a :class:`~rlinf.robotics.parts.base.RobotPart` --
:class:`~rlinf.robotics.parts.teleop.devices.TeleopPart` inherits it -- which is
what gives it a lifecycle and a node for free: ``SpaceMouse.at(node_rank=1)``
puts a device on the machine it is plugged into, exactly as an arm is placed.

What it is not is part of the *robot*. A leader arm never appears in the robot's
composition, and a policy never observes it: it reads the operator, not the
robot. Which parts of the robot's action it fills is the binding's business, and
that lives on the environment side.

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
   * - ``robotics/parts/teleop/``
     - The devices an operator drives, as parts, over vendor readers in
       ``readers/`` that import no Gymnasium.
   * - ``real/wrappers/teleop/``
     - ``intervention.py``, ``adapters.py``, ``streaming.py``, ``composed.py``,
       and ``config.py`` for device selection.
   * - ``real/wrappers/transforms/``
     - Relative frames, quaternion-to-Euler, gripper narrowing.
   * - ``real/wrappers/episode/``
     - Keyboard sessions: reward and done, start and end, eval control, policy
       switch, leader-follower.
   * - ``real/wrappers/__init__.py``
     - The stack builders that compose the three families.
   * - ``real/registry.py``
     - ``task_factory`` and ``register_tasks``.
   * - ``real/env.py``
     - ``RealWorldEnv``, the vectorized env the framework instantiates from
       ``env_type: real``.
   * - ``real/task_env.py``
     - ``RobotTask`` and ``RobotTaskEnv`` define the boundary between task logic
       and hardware.

Next
----

- :doc:`Adding a Task <../extending/new_task>`: follow the step-by-step guide.
- :doc:`Robotics Model <robotics>`: how the robot underneath is composed.
