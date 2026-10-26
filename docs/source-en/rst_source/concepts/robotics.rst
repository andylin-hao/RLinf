Robotics Interface
==================

This page explains how task and environment code uses the robotics interface to
inspect, connect, read from, and control a physical robot. It follows one
complete call sequence, then shows how stable part paths support composition and
where an environment converts robot data for a policy. Hardware sessions,
resource ownership, and worker placement are covered in
:doc:`Robotics Architecture <robotics_architecture>`.

The robotics interface exposes an arm, end effector, camera, or mobile base
through a stable name and path. Observations and actions use those same paths,
so task code does not depend on a concrete driver. As long as the paths remain
unchanged, a backend or placement can change without changing the caller.

Use the Interface
-----------------

A normal caller uses the interface in four phases: construct and inspect the
robot, select any parts needed for setup, connect and exchange data, then release
the hardware. The following example keeps those phases visible in one place:

.. code-block:: python

   from rlinf.robotics import Arm, Camera, build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # Safe before the robot is powered or reachable.
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

``build_robot()`` resolves the registered robot type and returns an unconnected
``Robot``. Construction records hardware arguments and placement, but it does
not import a vendor SDK or open a device. ``describe()`` reads that declaration,
so it can report paths, nodes, and connection ownership before the hardware is
powered or reachable.

Setup code then selects the capabilities it needs. ``child("arm", Arm)`` returns
the part at ``arm`` and checks that it implements ``Arm``; the expected class is
also the static return type seen by an editor. ``parts_of_type(Camera)`` walks
the full composition and returns every matching camera keyed by its dotted path.
Use ``child()`` when a path is part of the task contract, and
``parts_of_type()`` when the task needs a category without depending on camera
names.

``connect()`` opens each owning connection once and rolls back connections
already opened if a later one fails. Once connected, category methods such as
``Arm.is_robot_up()`` and ``Camera.is_ready()`` check whether the devices can be
used; ``clear_errors()`` and ``reset_joint()`` are available for setup outside
the per-step action stream.

The step loop uses two calls. ``get_observation()`` reads the composed robot once
and returns a nested dictionary whose keys are the part paths.
``send_action()`` accepts the corresponding action branches, dispatches only the
branches present in this step, and returns the action each part actually sent as
``applied``. Finally, ``disconnect()`` closes the owning connections in reverse
order. Keeping it in ``finally`` guarantees cleanup after a failed read or
command, and repeated calls are safe.

The lifecycle is the same for every robot. What changes between robots is the
set and nesting of the paths inside the observation and action dictionaries.

Read the Interface Paths
------------------------

To interpret the dictionaries returned above, start with how a path is formed.
The name assigned during composition becomes a path segment; a part carried by
another part adds the next segment. Comparing Franka and SO-101 shows both
cases.

A single-arm Franka has ``arm`` and ``end_effector`` at the same level because
the arm and Franka Hand open separate endpoints:

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

The different ``via`` values show that each part owns a connection. The two
top-level names therefore address two independently managed parts.

SO-101 represents the other case. Its five arm joints and gripper use one servo
bus, so the end effector appears below the arm and both paths use the same
connection:

.. code-block:: text

   SO101Robot
   └── arm               SO101Arm             node=0     via SO101Arm#1
       └── end_effector  MethodEndEffector    node=0     via SO101Arm#1

The corresponding observation and action follow that structure:

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

The nested action is not a second API: it is the same ``send_action()`` call
with one more path segment. ``describe()`` exposes ``via`` to explain why the
segment is nested and which paths share a resource; task code uses the paths and
values. The complete description is diagnostic text, not a serialization format.

Compose Parts Without Changing the Interface
--------------------------------------------

The previous section showed the paths a caller receives. Composition creates
those paths by joining readable parts under names, without changing how the
caller later reads or commands them.

A ``RobotPart`` can enter a robot directly. For example,
``Robot(base=base, arm=arm, end_effector=hand)`` uses the three keyword names as
top-level paths. A bare ``Connection`` is different: it owns a shared hardware
session but has no observation of its own. Select one readable part from it with
``session.part("left")`` before passing the returned ``RobotPart`` to a robot.

A part may also carry readable children backed by the same connection. Those
children appear below it automatically, which is why the SO-101 gripper is
reached as ``arm.end_effector``. When several parts form a reusable assembly,
``PartGroup`` adds one more named level. A dual-arm robot can therefore place
``left`` and ``right`` groups above the same arm interface:

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

There are no fixed ``arms`` or ``cameras`` slots. A mobile base, lift, head, or
third arm enters under a stable name, while callers continue to use
``get_observation()`` and ``send_action()``. With the public structure settled,
an environment can treat the complete composition as one device boundary.

Use the Same Boundary in an Environment
---------------------------------------

An environment turns the named robot data into the observation and action format
expected by a policy. It should therefore enter through the same complete
interface rather than read a driver beside it: one
``robot.get_observation()`` supplies the state used to build both policy-facing
``state`` and ``frames``, and ``robot.send_action()`` is the only per-step path
back to hardware. A part and any children sharing its connection reuse one
underlying state snapshot during that read.

The typed part references selected before ``connect()`` remain useful for work
outside the step loop. Current real-world environments use ``Arm`` for readiness,
error recovery, and joint reset, and use ``parts_of_type(Camera)`` to find frame
sources. The robot still owns the placement and lifecycle of those cameras; the
environment only consumes their observations.

Existing policies that expect flat vectors use ``LegacyObservationAdapter`` and
``VectorActionAdapter`` at this boundary. The adapters translate representation;
the robot interface remains named and nested.

Diagnose and Recover from Hardware Faults
-----------------------------------------

During a real-world run, a controller may drop out of its streaming control
mode or trip a joint limit, velocity limit, or collision protection check,
leaving the drives disabled and the robot unresponsive. When that happens, two
things need to occur in sequence: first, the robot driver must clear the
hardware fault, re-enable the drives, and restore the control mode so the robot
can move again; second, during reinforcement learning, the environment needs to
know that the current step was interrupted so it can decide whether to end the
episode.

You can call ``diagnose()`` to inspect health and ``recover()`` to clear faults
directly on the ``Robot`` (or any part), or enable automatic recovery so the
underlying connection checks and recovers before sending each command:

.. code-block:: python

   status = robot.diagnose()
   if not status.healthy:
       robot.recover()

How Fault Detection Works
^^^^^^^^^^^^^^^^^^^^^^^^^

High-frequency joint or Cartesian control typically runs at 50–100 Hz. Querying
controller health through an extra synchronous RPC or serial round-trip before
every command would block the control loop. Instead, ``Connection`` separates
fault *reporting* from pre-command *checking* through an in-memory health latch
that supports both ROS and non-ROS hardware:

1. **Asynchronous callbacks or SDK background threads** (such as ROS/RFrame
   status topics, UDP state broadcasts, or a SocketCAN listener thread): when
   the background callback receives an error frame or mode change, it records a
   ``HealthStatus`` in memory via ``self.report_health(...)``.
2. **Observation frame piggybacking** (such as RS-485/TTL serial servos or
   CANopen PDO frames whose state-read response already includes an error
   bitmask alongside joint angles): when ``get_observation()`` reads a fresh
   hardware frame via ``get_state()``, ``Connection.read_state()``
   automatically passes that state to ``_inspect_state_health(state)`` and
   updates the in-memory health status without a second bus transaction.

Before sending each command, the framework only checks that cached in-memory
fault state. When the robot is healthy, the check is an in-memory read and adds
no RPC or bus latency to 50–100 Hz control.

When a fault is latched, or when you call ``robot.diagnose()`` explicitly, the
connection returns a ``HealthStatus`` that classifies the state into one of
four ``FaultLevel`` values:

- ``FaultLevel.OK``: the controller is healthy and responding to commands.
- ``FaultLevel.MODE_DROP``: no hardware error is latched, but the controller
  exited its streaming control mode.
- ``FaultLevel.RECOVERABLE_FAULT``: a motion protection check (such as a joint
  limit, velocity limit, or self-collision check) triggered or an axis dropped
  its enable state, which can be cleared in software.
- ``FaultLevel.UNRECOVERABLE_FAULT``: an emergency stop, safety lock, or lost
  connection occurred, which cannot be cleared in software and requires human
  intervention.

Because controller error codes, disabled axes, and mode drops are deterministic
hardware signals that do not produce false positives, automatic fault recovery
(``enable_fault_recovery``) is enabled by default and can be configured in the
hardware YAML or at runtime via ``robot.configure_fault_recovery(...)``:

.. code-block:: yaml

   hardware:
     type: R2
     config:
       enable_fault_recovery: true
       max_recovery_attempts: 3
       recovery_cooldown_window: 10.0

How Recovery Works and How It Affects Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a fault is detected, ``recover()`` selects the recovery steps according to
``FaultLevel`` rather than resetting the entire controller every time:

- ``FaultLevel.MODE_DROP``: switches the controller back to streaming mode
  without resetting the motor drives.
- ``FaultLevel.RECOVERABLE_FAULT``: clears the controller error code,
  re-enables any disabled joint axes, and switches back to streaming mode.
- ``FaultLevel.UNRECOVERABLE_FAULT``: raises ``RuntimeError`` immediately. For
  recoverable faults, if automatic recovery is triggered more than
  ``max_recovery_attempts`` times within ``recovery_cooldown_window`` seconds,
  the underlying cause is still present and the circuit breaker also stops
  retrying and raises ``RuntimeError``.

Recovery also has to prevent position jumps. While the robot is stopped by a
fault, an upper-level policy may keep stepping its target joint angles away
from the robot's actual position. Re-enabling the drives and immediately
sending that old target would cause the arm to jerk toward it. Before
re-enabling drives or switching back to streaming mode, the connection
overwrites its cached target joint angles with the robot's current measured
joint angles. The first command after recovery therefore holds the robot at its
current pose.

During fault-free operation, this mechanism adds no delay to the control loop.
Only on the step where a fault actually occurs do the synchronous recovery
calls (clearing errors, enabling axes, and switching modes) pause the command
stream for roughly 50–300 ms while the robot stays in place, after which
subsequent commands resume normally. In teleoperation or evaluation scripts,
execution continues without restarting the program. In reinforcement learning
data collection, however, a 50–300 ms pause breaks the transition continuity of
that step. At the end of each step, the environment checks whether an automatic
recovery occurred, records the fault details in ``info["robot_fault"]``, and
when ``truncate_on_fault_recovery`` is enabled, marks the step with
``truncated=True`` so the rollout runner ends the interrupted trajectory and
resets the robot before the next episode.

Integrating Fault Detection for a New Robot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add fault detection and automatic recovery to a new robot driver (any
``Connection`` or ``ControllablePart`` subclass), implement one of the two
fault-reporting patterns together with ``clear_errors()``:

- **Pattern 1 (asynchronous callback or background thread)**: call
  ``self.report_health(HealthStatus(...))`` whenever the background listener
  receives a fault or mode change.
- **Pattern 2 (observation frame piggybacking)**: override
  ``_inspect_state_health(self, state) -> HealthStatus | None`` to inspect the
  error field already present in the state returned by ``get_state()``.
- **Recovery action**: implement ``clear_errors(self) -> bool`` to sync cached
  command targets to measured joint angles, clear the drive fault, and restore
  control mode (and optionally override ``_recover_mode(self) -> str`` when
  mode drops can be restored without a full fault reset).

.. code-block:: python

   from rlinf.robotics import ControllablePart, FaultLevel, HealthStatus


   class MyArmDriver(ControllablePart):
       # Pattern 2: inspect the error bitmask piggybacked on each state read
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

       # Or Pattern 1: call self.report_health(...) from an async SDK callback
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

With those methods in place, ``robot.diagnose()``, ``robot.recover()``,
pre-command automatic recovery, the cooldown circuit breaker, and
``robot.consume_recovery_event()`` work automatically without additional
plumbing.

Keep Placement Out of Task Code
-------------------------------

A path identifies a capability, not the process that hosts it. A camera in the
environment process and an arm on another node still appear in the same
observation because placement is recorded on each owning connection, outside
the path. Independent connections are read and commanded concurrently;
branches sharing one connection run in declaration order so the vendor session
is not accessed concurrently.

Task and policy code consequently depends on names and values rather than Ray
actors, RPCs, serial ports, or vendor sessions. Moving a connection changes its
placement declaration, not the calls or data described on this page. One final
boundary remains: a robot knows how hardware behaves, but not what the rollout
is trying to achieve.

Keep Task Logic Separate
------------------------

A part defines how to sense or move hardware; a task defines why those readings
and motions matter. Reward, termination, task-specific reset behavior, and
Gymnasium spaces therefore belong to the real-world env class registered for
the task, which drives the hardware through the same composed robot calls and
owns its lifecycle.

The result is two independent contracts: robot paths remain stable across tasks
and placement, while a task can change its policy-facing schema without changing
a driver.

Choose What to Read Next
------------------------

Continue from the boundary you need to change:

- To add a task on supported physical hardware, continue with
  :doc:`New Real-World Tasks <../extending/new_task>`.
- To add one local sensor or actuator, continue with
  :doc:`Adding a Robot <../extending/new_robot>`.
- To understand ``parts`` versus ``children``, shared connections, lifecycle,
  and worker placement, read
  :doc:`Robotics Architecture <robotics_architecture>`.
- To combine operator devices, read
  :doc:`Teleoperation <../guides/teleoperation>`.
