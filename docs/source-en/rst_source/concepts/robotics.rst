Robotics Interface
==================

This page explains how task and environment code uses ``rlinf.robotics`` to
inspect, connect, read from, and control a physical robot. It follows the
workflow from single-robot calls to full environment integration:

1. Connect, read observations, and send actions on a single robot.
2. Address parts by path and compose multi-part robots with ``RobotPart`` and
   ``PartGroup``.
3. Smooth large motion steps and batch multi-part commands on shared
   connections with ``MotionProfile`` and ``batch_commands()``.
4. Diagnose and automatically recover from hardware faults with ``diagnose()``
   and ``recover()``.
5. Connect the composed robot to ``RealWorldEnv`` and multi-node placement.

Internal details of hardware sessions, resource ownership, and worker placement
are covered in :doc:`Robotics Architecture <robotics_architecture>`.

Connect, Read, and Control a Robot
----------------------------------

A caller uses a robot in four steps: build and inspect the declaration, select
any parts needed for setup, connect and run the ``step`` loop, then disconnect.
The example below shows all four steps together:

.. code-block:: python

   from rlinf.robotics import Arm, Camera, build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # Safe to call before the robot is powered on or reachable.
   print(robot.describe())

   arm = robot.child("arm", Arm)
   cameras = robot.parts_of_type(Camera)

   robot.connect()
   try:
       if not arm.is_robot_up():
           raise RuntimeError("The arm is connected but not ready.")
       if not all(camera.is_ready() for camera in cameras.values()):
           raise RuntimeError("A camera is not delivering frames.")

       observation = robot.get_observation()
       tcp_pose = observation["arm"]["tcp_pose"]
       gripper_state = observation["end_effector"]["state"]

       applied = robot.send_action(
           {
               "arm": {"tcp_pose": target},
               "end_effector": {"target": width},
           }
       )
   finally:
       robot.disconnect()

``build_robot()`` looks up the registered robot builder and returns an
unconnected ``Robot``. Building the object only records hardware parameters and
placement; it does not import a vendor SDK or open a device. Because
``describe()`` reads that declaration, you can inspect part paths, target
nodes, and connection ownership even when the hardware is offline.

Before connecting, setup code can look up specific parts. ``child("arm", Arm)``
returns the part at ``arm`` and verifies that it is an ``Arm`` (which also gives
your editor the static type ``Arm``). ``parts_of_type(Camera)`` walks the full
robot tree and returns all cameras keyed by their dotted path. Use ``child()``
when the task expects a fixed path name, and ``parts_of_type()`` when you want
all devices of a category regardless of how they are named.

``connect()`` opens each underlying ``Connection`` once and rolls back any
already-opened connections if a later one fails. After connecting, category
methods such as ``Arm.is_robot_up()`` and ``Camera.is_ready()`` verify that
hardware is ready, while ``clear_errors()`` and ``reset_joint()`` handle setup
and reset outside the per-step control loop.

Inside the control loop, ``get_observation()`` reads all parts once and returns
a nested dictionary keyed by part path. ``send_action()`` takes a matching
dictionary, dispatches only the branches present in that call, and returns the
values actually sent as ``applied``. Calling ``disconnect()`` inside
``finally`` closes all connections in reverse order even if a read or command
raises an error, and repeated ``disconnect()`` calls are safe.

Every robot follows this same call sequence. What differs from one robot to
another is the set of paths in the observation and action dictionaries.

Part Paths and Robot Composition
--------------------------------

The keys inside ``get_observation()`` and ``send_action()`` come directly from
how parts are named and connected when the robot is assembled. Adding or
grouping parts introduces new path prefixes without changing the read and
command methods.

Independent Connections vs. Shared Buses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The keyword name given to a part becomes its first path segment, and any child
part attached to it adds another segment. Comparing Franka and SO-101 shows why
some paths are flat and others are nested.

On a single-arm Franka, the arm controller and the Franka Hand use separate
network endpoints, so ``arm`` and ``end_effector`` sit at the same level:

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

The two different ``via`` IDs in ``describe()`` indicate two independent
``Connection`` objects.

On an SO-101 arm, the five arm joints and the gripper share a single serial
servo bus. Because the gripper is driven through the arm's ``Connection``, it
appears under ``arm`` as ``arm.end_effector`` and shares ``SO101Arm#1``:

.. code-block:: text

   SO101Robot
   └── arm               SO101Arm             node=0     via SO101Arm#1
       └── end_effector  MethodEndEffector    node=0     via SO101Arm#1

Observations and actions follow that same structure:

.. code-block:: python

   joint_position = observation["arm"]["arm_joint_position"]
   gripper_state = observation["arm"]["end_effector"]["state"]

   robot.send_action(
       {
           "arm": {
               "joint_position": joint_target,
               "end_effector": {"target": gripper_target},
           }
       }
   )

Compose RobotPart and PartGroup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When assembling a ``Robot``, any ``RobotPart`` can be passed directly as a
keyword argument—for example, ``Robot(base=base, arm=arm, end_effector=hand)``
creates the top-level paths ``base``, ``arm``, and ``end_effector``. If a
shared ``Connection`` drives multiple subsystems (such as a whole-body or
dual-arm controller) and has no single observation of its own, call
``session.part("left")`` to select a readable ``RobotPart`` and pass that to
``Robot``.

When several parts belong to one logical unit (such as a left arm plus its
gripper), wrap them in a ``PartGroup`` to add a path prefix like ``left`` or
``right``:

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

``Robot`` does not hardcode slots like ``arms`` or ``cameras``. A mobile base,
waist, head, or third arm is simply added under its own name while callers keep
using ``get_observation()`` and ``send_action()``. When several parts share one
hardware ``Connection``, however, sending commands safely requires smoothing
large target steps and coordinating multi-part updates on that shared channel.

Smooth Motion and Batch Multi-Part Commands
-------------------------------------------

High-frequency streaming controllers (typically 50–100 Hz) expect small,
continuous target updates. They can fault or drop out of streaming mode in
three common situations:

1. A command jumps too far in a single control tick.
2. A joint is physically held back by gravity or contact while its command
   target keeps advancing, building up a large gap that snaps the joint once
   released.
3. On a shared whole-body bus, calling ``left_arm`` and ``right_arm`` methods
   one after another starts two separate blocking trajectory streams back to
   back.

``Connection`` prevents these issues through ``MotionProfile``,
``JointGroupSpec``, and ``batch_commands()``.

Smooth Trajectories and Velocity Pulses with MotionProfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ``Connection`` holds a ``MotionProfile`` (configured at construction or at
runtime via ``configure_motion_profile()``, with ``control_rate_hz`` and
``pulse_rate_hz`` defaulting to ``50.0`` Hz) and optional ``JointGroupSpec``
entries for joint groups with different mechanical inertia, such as waist,
arms, and neck.

When a command arrives, ``MotionProfile`` and ``JointGroupSpec`` handle three
tasks:

- **Prevent target wind-up (``reanchor()``)**: if external resistance or a
  brief pause prevented a joint from reaching previous commands, the cached
  target angle can drift ahead of the measured angle. Whenever that gap exceeds
  ``max_drift``, ``reanchor()`` pulls the command starting point back toward
  the measured angle before computing the next step.
- **Interpolate large moves (``generate_trajectory()``)**: if a target change
  is within the per-tick ``max_step``, the connection sends it immediately with
  no extra delay. If the move is larger (such as ``reset_joint()`` or a
  waypoint jump), ``generate_trajectory()`` splits it into a smooth
  raised-cosine (:math:`C^1` smoothstep) trajectory with zero velocity at the
  start and end, followed by ``hold_steps`` holding frames at the target.
- **Stream continuous base pulses (``generate_velocity_pulse()``)**: mobile
  bases with hardware watchdogs stop abruptly if velocity commands do not
  arrive continuously. ``generate_velocity_pulse()`` streams a smooth ramp-up,
  cruise, and cosine ramp-down curve at ``pulse_rate_hz`` and finishes with
  explicit zero-velocity stop frames.

Batch Multi-Part Commands with batch_commands()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On a dual-arm or whole-body robot whose parts share one ``Connection``, calling
``left_arm.reset_joint(...)`` followed by ``right_arm.reset_joint(...)`` would
normally run two separate trajectories one after the other: the first moves the
left arm while locking the right arm at its old target, and the abrupt switch
to the second trajectory stream can cause the controller to exit streaming
mode.

Wrapping multiple part calls in ``with robot.batch_commands():`` (or
``connection.batch_commands()``) combines them into a single trajectory. Inside
the ``with`` block, each part only updates its slice of the connection's cached
target vector; when the block exits, ``_flush_command_batch()`` interpolates
and streams all updated joint groups together:

.. code-block:: python

   # Optionally adjust control rate or step limits at runtime.
   robot.child("left_arm").connection.configure_motion_profile(
       control_rate_hz=100.0,
       max_step=0.006,
   )

   # Combine left and right arm resets into one synchronized trajectory.
   with robot.batch_commands():
       robot.child("left_arm").reset_joint(left_home_qpos)
       robot.child("right_arm").reset_joint(right_home_qpos)

During ``step`` execution, ``robot.send_action({...})`` automatically enables
``batch_commands()`` across every ``Connection`` involved, so multi-part
``policy`` actions are always combined. You only need
``with robot.batch_commands():`` when calling methods like ``reset_joint()``
across multiple parts during initialization or reset.

Diagnose and Recover from Hardware Faults
-----------------------------------------

Motion smoothing prevents software-induced jumps, but physical collisions or
joint limits can still trigger hardware protection during a run, disabling
motor drives or dropping the controller out of streaming mode. You can call
``diagnose()`` and ``recover()`` explicitly on the ``Robot`` (or any part), or
leave automatic recovery enabled so ``Connection`` checks and recovers before
sending each command:

.. code-block:: python

   status = robot.diagnose()
   if not status.healthy:
       robot.recover()

Detect and Classify Faults Without Blocking Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Querying hardware health through an extra synchronous RPC or serial round-trip
before every 50–100 Hz command would slow down the control loop. Instead,
``Connection`` caches a ``HealthStatus`` in memory and updates it through two
non-blocking channels:

1. **Asynchronous status callbacks (``report_health()``)**: when a background
   listener (such as a ROS/RFrame status subscriber, UDP broadcast listener, or
   SocketCAN thread) receives an error code or mode change, it updates the
   cached ``HealthStatus`` via ``self.report_health(...)``.
2. **State-read inspection (``_inspect_state_health()``)**: for buses whose
   normal state response already includes an error bitmask alongside joint
   angles (such as RS-485/TTL serial servos or CANopen PDO frames),
   ``Connection.read_state()`` passes each fresh state from ``get_state()`` to
   ``_inspect_state_health(state)`` and updates the cached health without a
   second bus read.

Before sending each command, ``Connection`` checks this in-memory status with
zero communication overhead. When a fault is recorded, or when you call
``robot.diagnose()``, the returned ``HealthStatus`` classifies the state into
one of four ``FaultLevel`` values:

- ``FaultLevel.OK``: the controller is healthy and ready for commands.
- ``FaultLevel.MODE_DROP``: no hardware error is latched, but the controller
  exited its streaming control mode.
- ``FaultLevel.RECOVERABLE_FAULT``: a motion protection check (such as a joint
  limit, velocity limit, or self-collision check) triggered or an axis dropped
  its enable state, which can be cleared in software.
- ``FaultLevel.UNRECOVERABLE_FAULT``: an emergency stop, safety lock, or lost
  connection occurred, which cannot be cleared in software and requires human
  intervention.

Automatic recovery (``enable_fault_recovery``) is enabled by default and can be
configured in the hardware YAML or at runtime via
``robot.configure_fault_recovery(...)``:

.. code-block:: yaml

   hardware:
     type: R2
     config:
       enable_fault_recovery: true
       max_recovery_attempts: 3
       recovery_cooldown_window: 10.0

Recover in Place Without Position Jumps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a fault is detected, ``recover()`` runs only the steps needed for that
``FaultLevel``:

- ``FaultLevel.MODE_DROP``: switches the controller back to streaming mode
  without resetting the motor drives.
- ``FaultLevel.RECOVERABLE_FAULT``: clears the controller error code,
  re-enables any disabled joint axes, and restores streaming mode.
- ``FaultLevel.UNRECOVERABLE_FAULT``: raises ``RuntimeError`` immediately. For
  recoverable faults, if automatic recovery triggers more than
  ``max_recovery_attempts`` times within ``recovery_cooldown_window`` seconds,
  the circuit breaker also stops retrying and raises ``RuntimeError``.

To prevent the robot from jerking when drives re-enable, ``Connection``
overwrites its cached target joint angles with the current measured joint
angles before enabling axes or restoring streaming mode. Even if an upper-level
``policy`` kept stepping its target while the arm was stopped, the first
command after recovery holds the robot at its current pose.

During normal fault-free operation, this mechanism adds no delay. Only on the
``step`` where a fault actually occurs do the recovery calls pause the command
stream for roughly 50–300 ms while the robot stays in place, after which
subsequent commands continue normally.

Support Fault Recovery in a New Driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add fault detection and automatic recovery to a new driver (any
``Connection`` or ``ControllablePart`` subclass), report health through either
channel and implement ``clear_errors()``:

- **Channel 1 (async callback or background thread)**: call
  ``self.report_health(HealthStatus(...))`` when the listener receives a fault
  or mode change.
- **Channel 2 (state-read inspection)**: override
  ``_inspect_state_health(self, state) -> HealthStatus | None`` to check the
  error field in the state dict returned by ``get_state()``.
- **Recovery action**: implement ``clear_errors(self) -> bool`` to sync cached
  targets to measured angles, clear the drive fault, and restore control mode
  (and optionally override ``_recover_mode(self) -> str`` if mode drops can be
  restored without a full fault reset).

.. code-block:: python

   from rlinf.robotics import ControllablePart, FaultLevel, HealthStatus


   class MyArmDriver(ControllablePart):
       # Channel 2: inspect the error code included in each state read
       def _inspect_state_health(self, state: dict) -> HealthStatus:
           err = int(state.get("error_code", 0))
           if err != 0:
               return HealthStatus(
                   healthy=False,
                   level=FaultLevel.RECOVERABLE_FAULT,
                   error_code=err,
                   error_message=f"drive fault {err}",
               )
           return HealthStatus(healthy=True)

       # Or Channel 1: call self.report_health(...) from an async SDK callback
       def _on_sdk_error_callback(self, err_code: int, msg: str) -> None:
           self.report_health(
               HealthStatus(
                   healthy=(err_code == 0),
                   level=FaultLevel.RECOVERABLE_FAULT
                   if err_code
                   else FaultLevel.OK,
                   error_code=err_code,
                   error_message=msg,
               )
           )

       def clear_errors(self) -> bool:
           self._sync_cmd_to_actual()
           self._sdk.reset_faults()
           return True

With those methods defined, ``robot.diagnose()``, ``robot.recover()``,
pre-command automatic recovery, the cooldown circuit breaker, and
``robot.consume_recovery_event()`` work automatically.

Use the Robot in RealWorldEnv and Multi-Node Clusters
-----------------------------------------------------

Because part paths, motion smoothing, and fault recovery are handled inside
``Robot`` and ``Connection``, ``RealWorldEnv`` and the cluster scheduler can
drive any robot composition through the same interface.

Connect to RealWorldEnv and Handle Fault-Interrupted Episodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``RobotPart`` defines how to read a sensor or move an actuator; a
``RealWorldEnv`` subclass defines task rewards, termination conditions, episode
resets, and Gymnasium spaces:

- **Observations and actions**: each ``step`` calls ``robot.get_observation()``
  once to build both ``state`` and camera ``frames`` (reusing a single state
  snapshot across parts that share a ``Connection``), and calls
  ``robot.send_action()`` to send commands. When a ``policy`` expects flat
  vectors, ``LegacyObservationAdapter`` and ``VectorActionAdapter`` convert
  between flat arrays and nested robot dicts.
- **Episode setup**: part references fetched via ``child()`` or
  ``parts_of_type()`` before ``connect()`` are used during ``reset()`` to check
  readiness, clear errors, and move arms back to their rest pose.
- **Fault-interrupted episodes**: in teleoperation or standalone evaluation,
  control can continue right after a 50–300 ms automatic recovery. In RL data
  collection, however, that pause breaks the transition continuity of the
  current ``step``. At the end of each ``step``, ``RealWorldEnv`` calls
  ``robot.consume_recovery_event()``, records any fault details in
  ``info["robot_fault"]``, and when ``truncate_on_fault_recovery`` is enabled,
  marks the ``step`` with ``truncated=True`` so the rollout runner ends the
  interrupted ``episode`` and resets the robot.

Place Connections Across Processes and Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Part paths identify capabilities, not where the driver process runs. A camera
in the ``env`` process and an arm worker on another machine still appear in the
same ``robot.get_observation()`` dict because ``node_rank`` and ``worker_name``
are configured on each owning ``Connection`` rather than in the path.
Independent connections run concurrently, while parts sharing one
``Connection`` run in declaration order so the underlying SDK session is never
called concurrently.

Moving a part to another process or node only changes its ``Connection``
placement arguments; ``env`` and ``policy`` code stays unchanged.

Choose What to Read Next
------------------------

- :doc:`New Real-World Tasks <../extending/new_task>`: define a new Gymnasium
  task on supported robot hardware.
- :doc:`Adding a Robot <../extending/new_robot>`: implement and register a new
  sensor, actuator, or robot composition.
- :doc:`Robotics Architecture <robotics_architecture>`: see how ``parts``,
  ``children``, shared connections, lifecycle, and worker placement are
  implemented.
- :doc:`Teleoperation <../guides/teleoperation>`: combine operator input
  devices with a robot.
