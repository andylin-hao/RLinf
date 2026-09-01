Robotics Interface
==================

RLinf presents a robot as named parts with one observation and action
interface. An arm, end effector, camera, or mobile base occupies one path in
that interface. The same paths appear in observations and, for controllable
parts, in actions.

This page covers the interface used by tasks and environments. It starts with
an existing robot, then explains how paths follow hardware composition and why
the same code works for local and remotely placed parts. Hardware sessions,
ownership, and worker placement are covered separately in
:doc:`Robotics Architecture <robotics_architecture>`.

Use the Interface
-----------------

Build the robot, inspect it before opening hardware, and keep
``disconnect()`` in a ``finally`` block:

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

       robot.send_action(
           {
               "arm": {"tcp_pose": target},
               "end_effector": {"target": width},
           }
       )
   finally:
       robot.disconnect()

``get_observation()`` reads the named interface once and returns one nested
dictionary. ``send_action()`` accepts the matching action tree; it may contain
only the branches to command in this step.

Use ``child(name, ExpectedType)`` when setup or reset code needs a part's
standard methods. The expected type is checked immediately and is also the
return type seen by an editor. For example, every ``Arm`` provides
``is_robot_up()``, ``clear_errors()``, and ``reset_joint()`` regardless of its
backend or placement. ``parts_of_type(Camera)`` is useful when a task needs all
cameras but does not depend on their configured names.

Read the Interface Paths
------------------------

Part names define the public data contract. A single-arm Franka has ``arm``
and ``end_effector`` at the same level because the arm and Franka Hand open
separate endpoints:

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

The different ``via`` values show that each part owns a connection. Either can
be placed, recovered, or replaced without changing the other.

Some hardware is inseparable at the connection boundary. An SO-101 drives its
five arm joints and gripper through one servo bus, so the end effector appears
under the arm and both paths use the same connection:

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

``describe()`` exposes ``via`` for diagnosis; task code uses paths and values.
The complete text is not a serialization format and should not be parsed.

Compose Parts Without Changing the Interface
--------------------------------------------

A readable part enters a robot directly. For example,
``Robot(base=base, arm=arm, end_effector=hand)`` publishes the three keyword
names as paths. If a shared controller is only a ``Connection`` and cannot be
observed itself, select the readable value it backs first, as in
``session.part("left")``.

A part may carry other parts. Those parts appear below it automatically, which
is why the SO-101 gripper is reached as ``arm.end_effector``. A dual-arm robot
can add ``left`` and ``right`` groups above the same interface:

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

There are no fixed ``arms`` or ``cameras`` slots. A mobile base, lift, head, or
third arm can be composed under a stable name without adding another robot API.

Use the Same Boundary in an Environment
---------------------------------------

An environment should read and command the composed robot, not its vendor
drivers. One call to ``robot.get_observation()`` gives the environment a
consistent input for building its policy-facing ``state`` and ``frames``. A
part and anything sharing its connection also reuse one underlying state
snapshot during that read.

Direct part access remains appropriate for operations that are outside the
per-step action stream. Current real-world environments use typed ``Arm``
parts for readiness, error recovery, and joint reset; they find cameras through
``parts_of_type(Camera)`` while the robot retains camera placement and
lifecycle ownership.

Existing policies that expect flat vectors can use
``LegacyObservationAdapter`` and ``VectorActionAdapter`` at the environment
boundary. The robot interface itself remains named and nested.

Keep Placement Out of Task Code
-------------------------------

A path does not say where its part runs. A camera in the environment process
and an arm on another node still appear in one observation. Independent
connections are read and commanded concurrently; branches sharing one
connection run in declaration order so the vendor session is not accessed
concurrently.

Task and policy code therefore depends on names and values rather than Ray
actors, RPCs, serial ports, or vendor sessions. Moving a connection changes its
placement declaration, not the task interface.

Keep Task Logic Separate
------------------------

A part knows how to sense or move hardware. It does not decide whether a
rollout succeeded. Reward, termination, task-specific reset behavior, and
Gymnasium spaces belong to a ``RobotTask`` or a concrete real-world env;
``RobotTaskEnv`` is available when a task follows the generic interface.

This boundary lets one robot run several tasks without changing device code,
and lets a task retain its policy-facing schema when hardware moves between
nodes.

Choose What to Read Next
------------------------

- To add a task on supported physical hardware, continue with
  :doc:`New Real-World Tasks <../extending/new_task>`.
- To add one local sensor or actuator, continue with
  :doc:`Adding a Robot <../extending/new_robot>`.
- To understand ``parts`` versus ``children``, shared connections, lifecycle,
  and worker placement, read
  :doc:`Robotics Architecture <robotics_architecture>`.
- To combine operator devices, read
  :doc:`Teleoperation <../guides/teleoperation>`.
