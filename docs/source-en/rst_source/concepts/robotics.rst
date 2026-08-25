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
       gripper_width = observation["arm"]["end_effector"]["state"]

       robot.send_action(
           {
               "arm": {
                   "tcp_pose": target,
                   "end_effector": {"target": width},
               }
           }
       )
   finally:
       robot.disconnect()

``get_observation()`` returns one nested dictionary. ``send_action()`` accepts
the matching action branches and returns what was applied. You may send an
action to only the parts you want to command.

Read the Tree
-------------

Part names are the public data contract. A single-arm Franka has one top-level
``arm``. Its end effector is mounted on that arm and therefore appears below
it:

.. code-block:: text

   FrankaRobot
   └── arm                 FrankaROSArm         node=1     via FrankaROSArm#1
       └── end_effector    MethodEndEffector    node=1     via FrankaROSArm#1

The two rows share a ``via`` value because one Franka connection owns both
parts. You do not need that detail to use the robot; it is there to make a
configuration mistake visible before the hardware moves.

This distinction explains when a part is passed directly to ``Robot`` and when
``part(name)`` is needed. ``FrankaROSArm`` is already a readable ``RobotPart``,
so ``Robot(arm=arm)`` composes it directly and brings along the end effector it
carries. A shared controller that is only a ``Connection`` is not readable;
select one of the parts it backs with ``session.part("left")`` before composing
it.

On a dual-arm robot, the same structure sits below ``left`` and ``right``:

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["arm"]["end_effector"]["state"]

The tree can be nested as deeply as the hardware requires. There is no fixed
``arms`` or ``cameras`` slot, so a lift, head, or third arm does not need a new
robot interface.

Local and Remote Parts Look the Same
------------------------------------

The path does not say where a part runs. A camera attached to the current
machine and an arm controlled from another node still appear in one observation
tree. RLinf reads independent hardware connections concurrently and preserves
the declared order for parts that share a connection.

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
