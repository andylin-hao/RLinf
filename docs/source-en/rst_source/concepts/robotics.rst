Robot Composition
=================

A robot is a tree of named parts. Each part exposes observations and may also
accept actions.

That is all you need when writing a task or using an existing robot.
An arm, gripper, camera, or mobile base occupies one path in the tree. That path
appears in observations and, when the part is controllable, in actions.

This page walks through an existing Franka, shows how to read its tree, and
marks the boundary between robot and task code. You do not need to understand
placement workers or hardware sessions to follow it; those details are linked
at the end for readers who are extending the robotics layer.

Use an Existing Robot
---------------------

Build the robot from its registered type, look at its composition, and then
connect it. Keep ``disconnect()`` in a ``finally`` block so hardware is released
when a rollout or a debugging command fails:

.. code-block:: python

   from rlinf.robotics import build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # Safe before any hardware is opened.
   print(robot.describe())

   robot.connect()
   try:
       observation = robot.get_observation()
       tcp_pose = observation["arm"]["tcp_pose"]
       gripper_width = observation["end_effector"]["state"]

       robot.send_action(
           {
               "arm": {"tcp_pose": target},
               "end_effector": {"target": width},
           }
       )
   finally:
       robot.disconnect()

``get_observation()`` returns one nested dictionary. ``send_action()`` accepts
the matching action branches and returns what was applied. You may send an
action to only the parts you want to command.

Read the Tree
-------------

Part names are the public data contract. A single-arm Franka has an ``arm`` and
an ``end_effector`` side by side:

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

The shape of the tree follows how the hardware is wired, not a convention. A
Franka Hand answers on its own endpoint, so it is a part in its own right and
stands beside the arm; the two ``via`` values differ because each opens its own
connection. Read that as a promise: either can be opened, recovered, or placed
on another node without touching the other.

Where a device really is inseparable from its arm, the tree says so. A GimArm
drives its gripper over the same CAN bus as the joints, so the gripper appears
one level down and both rows name the same connection:

.. code-block:: text

   GimArmRobot
   └── arm               GimArm               node=0     via GimArm#1
       └── end_effector  MethodEndEffector    node=0     via GimArm#1

You do not need the ``via`` column to use a robot. It is there to make a
configuration mistake visible before the hardware moves.

Nesting also decides how a part is composed. ``FrankaROSArm`` is already a
readable ``RobotPart``, so ``Robot(arm=arm, end_effector=hand)`` composes the
two directly. A shared controller that is only a ``Connection`` is not readable
on its own; select one of the parts it backs with ``session.part("left")``
first.

A dual-arm robot groups each side, and the same two parts sit under it:

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

The tree can be nested as deeply as the hardware requires. There is no fixed
``arms`` or ``cameras`` slot, so a lift, head, or third arm does not need a new
robot interface.

Local and Remote Parts Look the Same
------------------------------------

The path does not say where a part runs. A camera attached to the current
machine and an arm controlled from another node still appear in one observation
tree. RLinf reads independent hardware connections concurrently and preserves
the declared order for parts that share a connection. On a Franka that is worth
real time: the arm and the hand answer separately, so their readings overlap
instead of queueing behind one another.

Task and policy code therefore works with names and values, not Ray actors,
RPCs, serial ports, or vendor sessions. Existing policies that expect flat
vectors can use ``LegacyObservationAdapter`` and ``VectorActionAdapter`` at the
environment boundary.

Keep Tasks Separate
-------------------

A part knows how to sense or move hardware. It does not decide whether a rollout
succeeded. Reset behavior, reward, termination, and Gymnasium spaces belong to
a ``RobotTask``; ``RobotTaskEnv`` joins that task to the robot.

This boundary lets the same robot run another task without changing its device
code, and lets a task keep the same policy-facing schema when hardware is placed
on different nodes.

Choose What to Read Next
------------------------

- To add a task on supported physical hardware, continue with
  :doc:`New Real-World Tasks <../extending/new_task>`.
- To add one local sensor or actuator, continue with
  :doc:`Adding a Robot <../extending/new_robot>`.
- To understand ``parts`` versus ``children``, shared connections, part groups,
  lifecycle, and worker placement, read
  :doc:`Robotics Architecture <robotics_architecture>`.
- To combine operator devices, read
  :doc:`Teleoperation <../guides/teleoperation>`.
