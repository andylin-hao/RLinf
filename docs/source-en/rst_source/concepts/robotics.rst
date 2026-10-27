Robotics Interface
==================

This page explains how task and ``env`` code uses the robotics interface to
inspect, connect, read from, and control a physical robot. It follows the
workflow from single-robot calls to full system integration:

1. How to connect a single robot, read observations, and send control actions.
2. How to assemble multi-part robots and access each part by path.
3. How the connection layer smooths large moves and combines multi-part
   commands automatically.
4. How hardware faults are detected and recovered in place without position
   jumps.
5. How reinforcement learning environments and multi-node clusters use the
   composed robot.

Internal details of hardware sessions, resource ownership, and process
placement are covered in :doc:`Robotics Architecture <robotics_architecture>`.

Connect, Read, and Control a Robot
----------------------------------

Controlling a robot typically follows four steps: build and inspect the robot
structure, select any parts needed for setup, connect hardware and run the
``step`` loop, then disconnect. The example below shows all four steps
together:

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

Construction only records hardware parameters and target nodes; it does not
import a vendor SDK or open a device. ``build_robot()`` returns an unconnected
``Robot`` object, and ``describe()`` prints the part paths, target nodes, and
underlying connections even when the robot is powered off.

Before connecting, setup code can look up specific parts. When you know a fixed
path (such as ``arm``), ``child("arm", Arm)`` returns that part and verifies
its type. When you want all devices of a category regardless of how they are
named, ``parts_of_type(Camera)`` walks the full robot tree and returns a
dictionary keyed by path.

``connect()`` opens each underlying connection in order and closes any
already-opened connections if a later one fails. Once connected,
``is_robot_up()`` and ``is_ready()`` verify that the arm and cameras are ready,
while ``clear_errors()`` and ``reset_joint()`` handle error clearing and initial
pose reset before the ``step`` loop begins.

Inside the ``step`` loop, two calls handle data exchange: ``get_observation()``
reads all parts once into a nested dictionary keyed by path, and
``send_action()`` takes a matching dictionary, sends only the branches provided
in that call, and returns the applied values. Putting ``disconnect()`` inside
``finally`` guarantees that all connections close in reverse order even if a
read or command fails, and repeated calls are safe.

Every robot uses this same call sequence; robots differ only in the paths inside
their observation and action dictionaries.

Assemble a Robot and Access Its Parts
-------------------------------------

The paths used by ``get_observation()`` and ``send_action()`` come directly from
how parts are named and connected when the robot is assembled. Adding or
grouping parts only adds path levels without changing the read or command calls.

Independent Connections vs. Shared Buses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The name assigned to a part becomes its first path segment, and any sub-part
attached to the same communication bus adds another level below it. Comparing a
single-arm Franka with an SO-101 arm shows both layouts.

On a single-arm Franka, the arm controller and the Franka Hand use separate
network endpoints, so ``arm`` and ``end_effector`` sit at the same level:

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

The different ``via`` IDs in ``describe()`` show that each part owns its own
connection.

On an SO-101 arm, the five arm joints and the gripper share one serial servo
bus. Because the gripper communicates through the arm's connection, it sits
under ``arm`` as ``arm.end_effector`` and shares ``SO101Arm#1``:

.. code-block:: text

   SO101Robot
   └── arm               SO101Arm             node=0     via SO101Arm#1
       └── end_effector  MethodEndEffector    node=0     via SO101Arm#1

Observations and actions mirror that path hierarchy:

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

Assemble Multi-Part Robots
^^^^^^^^^^^^^^^^^^^^^^^^^^

When assembling a robot, each readable part (``RobotPart``) can be passed
directly to ``Robot`` as a keyword argument—for example,
``Robot(base=base, arm=arm, end_effector=hand)`` creates the top-level paths
``base``, ``arm``, and ``end_effector``. If a single shared connection
(``Connection``) drives multiple subsystems such as a dual-arm or whole-body
controller, call ``session.part("left")`` to select each part and pass it to
``Robot``.

When several parts form one logical unit (such as a left arm and its gripper),
``PartGroup`` groups them under a prefix such as ``left`` or ``right``:

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

The interface does not require fixed slots such as ``arms`` or ``cameras``. A
mobile base, waist, head, or third arm is simply added under a chosen name while
callers continue using ``get_observation()`` and ``send_action()``. When
multiple parts share one physical connection, however, keeping hardware motion
smooth and conflict-free requires the connection to coordinate trajectory
smoothing and multi-part updates.

Smooth Motion and Coordinate Multiple Parts
-------------------------------------------

High-frequency streaming controllers (typically 50–100 Hz) expect small,
continuous target updates. In practice, three situations commonly cause sudden
jerks or make a controller drop out of streaming mode:

1. A single command jumps further than a motor can safely move in one frame.
2. A joint is physically held back by gravity or contact while its target angle
   keeps advancing, building up a large gap that snaps the joint once released.
3. On a dual-arm or whole-body robot sharing one connection, calling left-arm
   and right-arm methods one after another sends two blocking whole-body
   trajectory streams back to back.

The connection layer handles all three cases automatically so task code does not
need manual interpolation loops.

Smooth Large Moves and Base Velocity Pulses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whenever a command is sent, the underlying connection applies three protections:

- **Cap accumulated target drift**: if external resistance or a brief pause
  prevents a joint from keeping up with previous commands, the cached target
  angle can drift ahead of the measured angle. When that gap exceeds a safety
  threshold, the connection pulls the interpolation start back toward the
  measured angle before computing the next step.
- **Interpolate large moves smoothly**: small per-step changes within the
  single-frame step limit are sent immediately with zero extra delay. Larger
  moves (such as ``reset_joint()`` or large waypoint jumps) are automatically
  split into a raised-cosine trajectory with zero velocity at both ends,
  followed by several holding frames so the motors settle cleanly.
- **Stream continuous base velocity pulses**: many mobile bases have a hardware
  watchdog that stops the base if velocity frames do not arrive continuously.
  The connection streams a smooth acceleration, cruise, and cosine-deceleration
  curve at the configured rate and ends with explicit zero-velocity frames.

By default, connections run streaming and pulse loops at ``50.0`` Hz (stored in
``MotionProfile``, with optional ``JointGroupSpec`` limits for joint groups of
different inertia such as waist, arms, and neck). To use a different control
rate or adjust step limits, pass them at initialization or call
``configure_motion_profile()`` at runtime.

Combine Multi-Part Commands on a Shared Connection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On a dual-arm or whole-body robot whose parts share one connection, calling
``left_arm.reset_joint(...)`` followed by ``right_arm.reset_joint(...)`` would
normally run two separate trajectories in sequence: the first moves the left arm
while locking the right arm at its old target, and the abrupt handover to the
second stream often causes the controller to fault or exit streaming mode.

Wrapping multiple part calls in ``with robot.batch_commands():`` combines them
into a single synchronized trajectory. Inside the ``with`` block, each part only
updates its own slice of the target joint angles; when the block exits, the
connection interpolates and streams all updated joints together:

.. code-block:: python

   # Optionally adjust control rate or per-step limits at runtime.
   robot.child("left_arm").connection.configure_motion_profile(
       control_rate_hz=100.0,
       max_step=0.006,
   )

   # Combine left and right arm resets into one synchronized trajectory.
   with robot.batch_commands():
       robot.child("left_arm").reset_joint(left_home_qpos)
       robot.child("right_arm").reset_joint(right_home_qpos)

During the normal ``step`` loop, ``robot.send_action({...})`` automatically
batches commands across every connection involved, so multi-part ``policy``
actions are always combined. You only need ``with robot.batch_commands():`` when
calling methods such as ``reset_joint()`` across multiple parts during setup or
reset.

Diagnose and Recover from Hardware Faults
-----------------------------------------

Motion smoothing prevents software-induced jumps, but physical collisions or
joint limits can still trigger hardware protection during a run, disabling motor
drives or dropping the controller out of streaming mode. You can call
``diagnose()`` to inspect health and ``recover()`` to clear faults on the
``Robot`` (or any part), or keep automatic recovery enabled so the connection
checks and recovers before sending each command:

.. code-block:: python

   status = robot.diagnose()
   if not status.healthy:
       robot.recover()

Detect Faults Without Blocking the Control Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Querying hardware health through an extra synchronous RPC or serial round-trip
before every 50–100 Hz command would stall the control loop. Instead, the
connection caches a health status (``HealthStatus``) in memory and updates it
through two non-blocking paths:

1. **Background status callbacks**: when a background listener (such as a
   ROS/RFrame status subscriber, UDP broadcast listener, or SocketCAN thread)
   receives an error code or mode change, it updates the cached health status in
   memory.
2. **Inspection during state reads**: for hardware whose normal joint-state
   packet already includes an error bitmask (such as serial servos or CANopen
   frames), the connection checks that field while reading observations without
   making a second bus request.

Before sending each command, the connection only reads this in-memory status,
adding zero communication overhead during normal operation. When a fault is
recorded or when you call ``robot.diagnose()``, the returned ``HealthStatus``
indicates the severity with ``FaultLevel``:

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

Recover by Severity Without Position Jumps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When recovery runs, ``recover()`` executes only the steps required for that
fault level:

- **Streaming mode drop (``FaultLevel.MODE_DROP``)**: switches the controller
  back to streaming mode without resetting the motor drives.
- **Recoverable drive fault (``FaultLevel.RECOVERABLE_FAULT``)**: clears the
  controller error code, re-enables any disabled joint axes, and restores
  streaming mode.
- **Unrecoverable or repeating faults**: raises ``RuntimeError`` immediately on
  ``FaultLevel.UNRECOVERABLE_FAULT``. For recoverable faults, if automatic
  recovery triggers more than ``max_recovery_attempts`` times within
  ``recovery_cooldown_window`` seconds, the circuit breaker also stops retrying
  and raises ``RuntimeError``.

To prevent the arm from jerking when drives re-enable, the connection overwrites
its cached target joint angles with the current measured joint angles before
enabling axes or restoring streaming mode. Even if an upper-level ``policy``
kept advancing its target while the robot was stopped, the first command after
recovery holds the robot at its current pose.

During normal operation this mechanism adds no delay; only on the ``step`` where
a fault is actively cleared do the recovery calls pause the command stream for
roughly 50–300 ms while the robot stays in place.

Add Fault Recovery to a New Driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To support fault detection and automatic recovery in a new driver (any
``Connection`` or ``ControllablePart`` subclass), report health through either
path and implement ``clear_errors()``:

- **Option 1 (background status callback)**: call
  ``self.report_health(HealthStatus(...))`` when the listener receives a fault
  or mode change.
- **Option 2 (inspection during state reads)**: override
  ``_inspect_state_health(self, state) -> HealthStatus | None`` to check the
  error field in the state dict returned by ``get_state()``.
- **Recovery action**: implement ``clear_errors(self) -> bool`` to sync cached
  targets to measured angles, clear the hardware error, and restore control mode
  (and optionally override ``_recover_mode(self) -> str`` if mode drops can be
  restored without a full fault reset).

.. code-block:: python

   from rlinf.robotics import ControllablePart, FaultLevel, HealthStatus


   class MyArmDriver(ControllablePart):
       # Option 2: inspect the error code included in each state read
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

       # Or Option 1: report health from an async SDK callback
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
pre-command automatic recovery, the cooldown circuit breaker, and recovery event
recording work automatically.

Use the Robot in Environments and Multi-Node Clusters
-----------------------------------------------------

Once part paths, motion smoothing, and fault recovery are handled by the robot
interface, reinforcement learning environments and the cluster scheduler can
drive any robot composition in a uniform way.

Connect to an Environment and Handle Interrupted Episodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Robot parts read sensors and drive actuators, while the task environment
(inheriting from ``RealWorldEnv``) defines rewards, termination conditions,
``episode`` resets, and Gymnasium spaces:

- **Observation and action conversion**: each ``step`` calls
  ``robot.get_observation()`` once to build ``state`` and camera ``frames``
  (reusing a single state snapshot across parts that share a connection), and
  calls ``robot.send_action()`` to send commands. When a ``policy`` expects flat
  vectors, ``LegacyObservationAdapter`` and ``VectorActionAdapter`` convert
  between flat arrays and nested robot dicts.
- **Episode initialization and reset**: part references fetched via ``child()``
  or ``parts_of_type()`` before ``connect()`` are used during ``reset()`` to
  check readiness, clear errors, and return arms to their initial pose.
- **Truncating fault-interrupted episodes**: in teleoperation or standalone
  evaluation, control can continue right after a 50–300 ms automatic recovery.
  In RL data collection, however, that pause breaks the transition continuity of
  the current ``step``. At the end of each ``step``, the ``env`` calls
  ``robot.consume_recovery_event()``, records any fault details in
  ``info["robot_fault"]``, and when ``truncate_on_fault_recovery`` is enabled,
  marks the ``step`` with ``truncated=True`` so the rollout runner ends the
  interrupted ``episode`` and resets the robot.

Run Parts Across Processes and Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Part paths describe what capabilities the robot has, not which process runs the
driver. A camera in the ``env`` process and an arm worker on another machine
still appear in the same ``robot.get_observation()`` dict because ``node_rank``
and ``worker_name`` are configured on each underlying connection. Independent
connections run concurrently, while parts sharing one connection run in
declaration order so the vendor SDK is never called concurrently.

Moving a part to another process or node only changes its connection placement
parameters; ``env`` and ``policy`` code stays unchanged.

Choose What to Read Next
------------------------

- :doc:`New Real-World Tasks <../extending/new_task>`: define a new Gymnasium
  task on supported robot hardware.
- :doc:`Adding a Robot <../extending/new_robot>`: implement and register a new
  sensor, actuator, or robot composition.
- :doc:`Robotics Architecture <robotics_architecture>`: see how ``parts``,
  ``children``, shared ``Connection`` objects, lifecycle, and worker placement
  are implemented.
- :doc:`Teleoperation <../guides/teleoperation>`: combine operator input
  devices with a robot.
