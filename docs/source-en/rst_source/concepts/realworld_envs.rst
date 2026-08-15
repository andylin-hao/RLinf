Real-World Environment Model
============================

A real-world environment is a robot, a task, and a stack of wrappers. The robot
knows how to move and what it senses; the task knows what counts as success; the
wrappers cover everything a person adds around a rollout -- taking over with a
leader arm, marking an episode successful, changing how a pose is written down.
We'll follow one task from its config to its gym id, then look at where each kind
of wrapper lives and why the split is where it is.

For how a robot is composed from parts and placed on nodes, start with
:doc:`robotics`. This page picks up where that one stops, at the environment.

A Task Is a Config and a Few Overrides
--------------------------------------

Open ``rlinf/envs/real/franka/`` and you see the tasks a Franka can be asked to
do, one module each, beside the ``base.py`` they share. A task is usually a
dataclass saying where the target is and how the arm should comply, plus whatever
reset behavior that task needs:

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

``compliance()`` merges your overrides onto ``COMPLIANCE_DEFAULTS`` and rejects a
gain the controller does not take, so a misspelled key is an error rather than a
setting that quietly does nothing. Peg insertion states one gain; bin relocation
states eleven. The rest of each config is the task: poses, reward thresholds, and
the randomization applied at reset.

Registering It Is One Line
--------------------------

Every task is built the same way -- construct the env class with the worker's
config, then wrap it in the stack that robot's action space needs -- so that is
all a task declares:

.. code-block:: python

   TASKS = {
       "FrankaEnv-v1": (FrankaEnv, apply_single_arm_wrappers),
       "PegInsertionEnv-v1": (PegInsertionEnv, apply_single_arm_wrappers),
       "DualFrankaTCPEnv-v1": (DualFrankaTCPEnv, apply_dual_franka_joint_wrappers),
   }

   _ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

``register_tasks`` generates each Gymnasium entry point and registers it. The gym
id is what a config and a dataset refer to, so it is worth choosing once and
leaving alone.

Three Kinds of Wrapper
----------------------

Everything wrapped around a task env falls into one of three groups, and the
group is decided by what the wrapper changes:

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
     - When a rollout starts, when it ends, and what it scored -- the judgements
       a person watching makes, which no sensor reports.

``wrappers.py`` composes them. It reads the env config, decides which teleop
device is in play, and returns the stack:

.. code-block:: python

   env = apply_single_arm_wrappers(PegInsertionEnv(...), cfg)

Teleop: One Wrapper, Many Devices
---------------------------------

Every teleop device answers the same question -- what would the operator do right
now -- and the answer is handled the same way whatever produced it. So the
reading is the only thing a device implements:

.. code-block:: python

   class SpaceMouseTeleop(TeleopDevice):
       def read(self, env, policy_action) -> TeleopSample:
           expert, buttons = self.expert.get_action()
           return TeleopSample(
               action=expert,
               active=bool(np.linalg.norm(expert) > 0.001),
               info={"left": buttons[0], "right": buttons[1]},
           )

``TeleopIntervention`` owns everything around it: the hold window that keeps the
operator in control between samples, the fallback when they let go, and the
``intervene_action`` key a dataset collector reads back.

The ``active`` flag, not the action, is what says the operator is driving.
Devices report small residual motion constantly, so each one sets its own
threshold. A device the operator holds down -- a PICO grip, a trigger -- sets
``timeout = 0`` instead, because the button already says exactly when they are
driving and half a second of carry-over would keep commanding the robot after
they let go.

A leader arm tracks well only if the follower receives targets continuously, far
faster than ``env.step`` runs. :class:`StreamingTeleopDevice` gives such a device
its own thread and owns the awkward parts of having one: pause while the env
drives the robot home, align before the first target, join on shutdown.

Choosing a device is one config key:

.. code-block:: yaml

   env:
     eval:
       teleop_device: spacemouse   # or gello, pico, gello_joint, none
       gello_port: /dev/serial/by-id/...

Devices and Readers Are Separate
--------------------------------

Inside ``teleop/`` there is one more split, and it is the reason a bench script
works. ``devices/`` holds the readers -- the code that talks to a serial port or
a headset -- and imports no Gymnasium. ``adapters.py`` turns a reading into an
action for a particular env.

That is what lets you check a leader arm is wired correctly before involving a
robot at all:

.. code-block:: bash

   python -m rlinf.envs.real.teleop.devices.gello --port /dev/ttyUSB0

A teleop device is not a :class:`~rlinf.robotics.parts.base.RobotPart`. A part
answers what a component means to the policy, and a leader arm has no such
answer: no policy observes one, and no robot composes one. It reads the operator,
not the robot, which is why it lives on the environment side.

Episode Control Is Not Teleop
-----------------------------

Marking a success, aborting a take, switching policies mid-rollout: the operator
is the only one who knows, but none of it touches the action. Those wrappers live
in ``episode/`` and share :class:`KeyboardSession`, which owns the listener, drops
repeat presses inside a debounce window, and clears the queue on reset so a pedal
tapped while the arm homes does not start the next episode.

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
   * - ``real/robot_task_env.py``
     - ``RobotTask`` and ``RobotTaskEnv``, which keep task logic out of hardware.

Next
----

- :doc:`Adding a Task <../extending/new_task>`: follow the step-by-step guide.
- :doc:`Robotics Model <robotics>`: how the robot underneath is composed.
