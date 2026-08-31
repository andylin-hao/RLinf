Teleoperation
=============

Teleoperation lets an operator replace the policy's action during a rollout to
collect demonstrations, recover from a failure, or run DAgger. Start with one
device and verify its readings; only after that works should you combine
devices. This page follows that order, and closes with the placement boundary
for standalone and environment-owned devices. For the underlying robot
composition, see :doc:`../concepts/robotics`.

Choose a Device
---------------

For a single device, set its name:

.. code-block:: yaml

   env:
     eval:
       teleop: spacemouse

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - Device
     - What the operator does
     - Extra config
   * - ``spacemouse``
     - Moves the arm with a 6-DoF puck; its buttons latch the gripper.
     - None
   * - ``gello``
     - Poses a leader arm for the follower to track.
     - ``gello_port``
   * - ``gello_joint``
     - Poses a leader arm for joint-by-joint tracking, one entry per arm.
     - ``left_gello_port`` / ``right_gello_port``
   * - ``pico``
     - Uses a handheld VR controller whose grip marks when it is driving; one
       entry per arm on a two-armed robot.
     - ``pico:`` block
   * - ``glove``
     - Bends the fingers of a dexterous hand, alongside a device on the arm.
     - ``glove_config:`` block
   * - ``so101_leader``
     - Poses an SO-101 leader arm; its own gripper commands the follower's.
     - ``so101_leader_port``
   * - ``none``
     - Leaves the policy in control with no operator device.
     - None

An environment accepts only devices that match the robot's control paths. A
dual-arm Franka, for example, has no single-arm Cartesian path, so it rejects
``spacemouse`` instead of ignoring the setting. The error also lists the devices
that the environment accepts.

Check a Device First
--------------------

The devices whose readers poll on their own thread -- ``gello``,
``gello_joint`` and ``spacemouse`` -- run without a robot or cluster:

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.gello --port /dev/ttyUSB0

When a leader arm reports only zeros or a spacemouse does not respond, this
command isolates wiring and permission problems from environment configuration.
``toolkits/realworld_check`` does the same for a complete robot;
``check_robot_parts`` walks one from composition through to disconnect.

Compose Devices After One Works
-------------------------------

Once each device works alone, put them in a list. Each entry contributes actions
for the named robot parts it can drive:

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

In this dexterous-hand rig, the puck drives the arm and the glove drives the
hand. The glove takes control only while the spacemouse's second button is held;
releasing it leaves the hand at its last pose.

Device-wide settings remain in their named config block. For example, the
shipped dexterous-hand configs set the glove ports and calibration under
``glove_config``:

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]
       glove_config:
         left_port: /dev/ttyACM0
         frequency: 60
         config_file: null

When a robot has two branches of the same kind, ``drives`` selects the branch for
each device. This is the only configuration field that names a robot part:

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

A device leaves any part the robot does not have unfilled. The builder rejects
the rig if a device matches no part at all, or if two devices claim the same
part.

Keep Device Ownership Explicit
------------------------------

The built-in teleop builder creates devices in the environment process, and
``TeleopGroup.connect()`` opens them there. A teleop device does not belong to
the robot tree, so ``Robot.connect()`` does not place it. Plug devices configured
through ``env.*.teleop`` into the machine that runs the environment worker.

A teleop device is a ``Connection`` like any other, so it accepts a
``node_rank`` and opens on that node when something connects it:

.. code-block:: python

   leader = Gello("/dev/ttyUSB0", node_rank=1)
   leader.connect()
   try:
       print(leader.get_observation())
   finally:
       leader.disconnect()

That is useful for a standalone diagnostic, and it also works inside a
``TeleopGroup``, which opens each of its devices the same way. What decides
where a leader arm runs is the ``node_rank`` its constructor was given; the
``env.*.teleop`` config does not yet expose one, so devices configured through
it open in the environment process.

Each distinct device is read once per sample. If one device contributes to two
parts, it is still opened only once; a spacemouse driving both an arm and a
gripper therefore uses a single HID handle.

.. _teleop-rate:

When the Rate Is the Problem
----------------------------

A follower tracks poorly when it receives leader-arm targets only at the
policy's step rate. Direct streaming moves that path onto a thread that pushes
joint targets to the controllers at roughly 1 kHz. ``env.step`` continues to
read state but no longer forwards motion:

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         teleop_direct_stream: true

Enable direct streaming only when tracking visibly lags. Because ``env.step`` no
longer dispatches joint targets in this mode, a misconfigured rig remains still
instead of receiving malformed motion.

Add a Device
------------

A device is one module under ``robotics/parts/teleop/``. Subclass
``TeleopDevice``, register a config name for it, and say what it fills:

.. code-block:: python

   @TeleopDevice.register("example")
   class ExampleDevice(TeleopDevice):
       PRODUCES = {"arm": ActionKind.JOINT_POSITION}
       NEEDS = ("joint_positions",)

       def __init__(self, port: str) -> None:
           self._port = port

       def _open(self):
           return ExampleSDK(self._port).open()

       @property
       def observation_features(self) -> Features:
           return {"joints": {"shape": (7,), "dtype": "float32"}}

       def get_observation(self) -> Observation:
           return {"joints": self._device.read()}

       def action(self, reading, context) -> TeleopAction:
           moved = np.linalg.norm(reading["joints"] - context["joint_positions"][0])
           return TeleopAction({"arm": reading["joints"]}, driving=bool(moved > 0.01))

``PRODUCES`` names the action parts the device fills and the meaning of each, so
an environment can check the device against its own layout before any hardware
is opened. ``NEEDS`` lists the robot state the device wants; each name is read
once per sample and arrives in ``context``, whether one device asks for it or
five do.

``_open()`` reaches the hardware and returns the handle, which the device then
reads through ``self._device``; ``_release()`` closes it. That is the same
connection lifecycle every robot part follows, and the same machinery accepts
``node_rank`` without the device writing anything for it -- which is why the
constructor above takes only its own arguments.

``observation_features`` declares the reading before any hardware is open, so a
rig can be described offline. It is abstract, so a device that omits it cannot
be instantiated.

``action()`` returns a ``TeleopAction``: the parts it fills, and whether the
operator is currently driving. A device that fills nothing this sample returns
``driving=False`` and leaves the policy in control.

Config-Facing Behaviour
~~~~~~~~~~~~~~~~~~~~~~~

The default ``from_config()`` passes the entry's own options through as keyword
arguments, which is all a device needs whose config keys match its constructor.
The example above is already reachable as ``{example: {port: /dev/ttyUSB0}}``
with nothing further to write.

Override it to read a key from the wider env config, or to choose behaviour from
what the robot can accept:

.. code-block:: python

   @classmethod
   def from_config(cls, cfg, options, facts):
       settings = dict(cfg.get("example_config", {}))
       settings.update({k: v for k, v in options.items() if k != "drives"})
       if "port" not in settings:
           raise ValueError("teleop device 'example' requires a port")
       return TeleopEntry(cls(**settings), drives=options.get("drives"))

``cfg`` is the complete env section and holds the device-wide config block.
``options`` belongs to this one list entry, so it can select a port or a
``drives`` branch without changing other instances. Validate these keys here; a
misspelled hardware option should not be silently ignored. ``facts`` describes
the env's action layout and semantics -- notably whether an arm takes an
absolute pose or a delta -- so a device can adapt without importing a concrete
env class.

Override ``streamer()`` only for a device that also commands the robot on its
own thread; the default returns nothing, which is what every device but
``gello_joint`` wants. See :ref:`When the Rate Is the Problem <teleop-rate>`.
The builder creates every entry before it creates any streamer, so a streamer
may take over devices it did not construct itself.

Then list the registered name in the env's ``TELEOP`` tuple. This declaration
states that the env can represent the device's action; it does not register the
device again.

Retired Spellings
-----------------

``teleop_device`` named a single device, and the booleans ``use_spacemouse``,
``use_gello``, ``use_gello_joint``, and ``use_pico`` each switched one on. All
of them still work and warn. Where ``teleop`` appears alongside one of them --
which happens when a run config sits on an older base -- ``teleop`` is the one
that takes effect, and the warning names what it replaced.

Next
----

- :doc:`Robot Composition <../concepts/robotics>`: the named robot paths a
  device fills.
- :doc:`Real-World Tasks and Environments <../concepts/realworld_envs>`: where teleop
  sits in the wrapper stack.
- :doc:`Data Collection <data_collection>`: recording what an operator drove.
