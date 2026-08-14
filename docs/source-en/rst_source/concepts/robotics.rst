Robotics Model
==============

Use RLinf's robotics model to add hardware or debug a real-world run. Learn what
a component *is* to the policy, how components form a robot, and where each
component runs.

The Core Idea
-------------

**Treat every physical component as a part.** An arm, gripper, camera, or mobile
base is a ``RobotPart``. Each part connects and reports an observation. A
controllable part also accepts an action. Do not add a separate "driver" layer.

Declare parts when hardware does not map one-to-one onto components. For
example, a coupled dual-arm controller can drive two arms, two grippers, and two
wrist cameras over one ROS connection. Let the part declare what it exposes
instead of adding an abstraction for "the thing that owns the connection":

.. code-block:: python

   @property
   def parts(self) -> dict[str, RobotPart]:
       return {
           "left": MethodArm(self, commands={"tcp_pose": "move_left_arm"}),
           "right": MethodArm(self, commands={"tcp_pose": "move_right_arm"}),
           "left_end_effector": MethodGripper(self, state_field="follow1_pos"),
       }

Treat "owns a connection" as a property of some parts, not as another type.

**Compose a robot from named parts.** **Place any part on a node.** Together with
the first rule, these three rules define the model.

The Abstractions
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Abstraction
     - What it is
   * - ``RobotPart``
     - Any physical component. It defines ``connect``, ``get_observation``,
       ``disconnect``, and ``reset``. Its ``observation_features`` describes the
       returned data.
   * - ``ControllablePart``
     - A part that also accepts commands through ``send_action`` and describes
       them with ``action_features``.
   * - ``Camera`` / ``EndEffector`` / ``MobileBase`` / ``LeggedBase``
     - Specific part types that composition and remote proxies can distinguish.
   * - ``parts``
     - The named parts belonging to a part. One mechanism for both directions of
       composition: hardware returns what its connection drives, a ``Group``
       returns what it was composed of, a leaf returns ``{}``.
   * - ``Group``
     - A part made of named parts. An arm, a torso, or a whole robot is the same
       construct with different names.
   * - ``Robot``
     - The outermost group. It knows its registered type, builds itself from a
       hardware config, and owns the connections it places.
   * - ``at()`` / ``PartSpec``
     - A declaration: a part class, its arguments, and the node to build it on.
   * - ``PartHandle``
     - A reference with the same interface whether the part runs locally or in a
       worker.
   * - ``MethodArm`` / ``MethodGripper`` / ``MethodCamera``
     - Views that turn methods such as ``open_gripper`` and ``get_camera(id)``
       into parts.

Composition, Not Robot Types
----------------------------

A robot is named parts, and nothing more. There are no arm, camera, or base
slots to fit hardware into, so a lift or a head needs no new concept -- just
another name:

.. code-block:: python

   one = FrankaRobot(arm=arm, gripper=gripper)
   two = FrankaRobot(left=Group(arm=l, gripper=lg), right=Group(arm=r, gripper=rg))
   lifted = FrankaRobot(left=..., right=..., lift=lift, head=head_camera)

The observation and the action mirror the composition exactly. Names become
paths, at any depth:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Path
     - Meaning
   * - ``<name>``
     - A part of the robot, by the name you composed it under.
   * - ``<group>.<name>``
     - A part of a group, nested as deeply as the composition goes.

Let ``Robot`` reset, read, and command arms in parallel when they use independent
connections. A two-arm observation then costs one round trip, not two.

Placement Is a Property of Parts
--------------------------------

Declare where a part runs with ``at()``. Nobody calls a placement function:
:meth:`Robot.connect` builds every declaration on its node.

.. code-block:: python

   robot = FrankaRobot(
       left=FrankaROSArm.at("10.0.0.1", node_rank=1).part("arm"),
       scene=RealSenseCamera.at(info, node_rank=3),
   )
   robot.connect()

A declaration is inert. Nothing touches hardware until ``connect``, which places
each distinct declaration exactly once, publishes its handle as
``robot.handles[<name>]``, and tears down whatever it already placed if a later
part fails. ``disconnect`` releases them.

Declare a shared connection once and refer to its parts. One connection backing
two arms and two cameras is opened once, not four times:

.. code-block:: python

   hardware = Turtle2Hardware.at(50, camera_ids, node_rank=0)
   robot = Turtle2Robot(
       left=Group(
           arm=hardware.part("left"), gripper=hardware.part("left_end_effector")
       ),
       right=Group(
           arm=hardware.part("right"), gripper=hardware.part("right_end_effector")
       ),
       wrist_1=hardware.part("wrist_1"),
   )

Placement applies to every part, not only arms. A robot owns its cameras and
places them like anything else, so a camera runs on the machine it is plugged
into while the policy runs elsewhere. Give it a node with
``declare_cameras({name: info}, node_rank=...)`` and the robot opens it on
``connect`` and closes it on ``disconnect``. ``spawn()`` is the eager form
underneath; reach for it only outside a robot, such as in a bench script.

Do not write a worker class for each hardware device. RLinf synthesizes one from
the part class (``type(name, (Worker, PartCls), ...)``). ``WorkerGroup`` then
binds every public method as an RPC. Call methods outside the part interface
through the handle. The call shape stays the same locally and remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

Read :doc:`Placement <placement>` to map workers onto nodes and GPUs.

Compose Every Part Kind
-----------------------

An arm, its end effector, and its cameras are separate parts. The robot's
``build`` composes them by name, and each carries its own ``node_rank``:

.. code-block:: python

   arm = FrankaROSArm.at(robot_ip, node_rank=1)
   robot = FrankaRobot(
       arm=arm.part("arm"),
       gripper=RobotiqGripper.at(port="/dev/ttyUSB0", node_rank=2),
       wrist=RealSenseCamera.at(info, node_rank=3),
   )

A Robotiq gripper is a serial device of its own and a camera holds its own USB
link, so neither has to sit on the arm's machine. Take the end effector from
``arm.part("end_effector")`` only when it genuinely rides the arm's connection,
as a Franka hand does.

Composing is all a robot's ``build`` does. There is no per-part config class to
write: a camera is ``declare_cameras({name: info}, node_rank=...)``, an arm is
``at(...)`` with its arguments, and the robot's own ``RobotConfig`` -- the
hardware YAML schema it already needs for discovery -- supplies the fields.

Lifecycle
---------

Four steps, in order.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Step
     - What happens
   * - Declare
     - ``at()`` records a part class, its arguments, and its node. Nothing is
       built and no hardware is touched.
   * - Connect
     - ``Robot.connect`` builds each distinct declaration on its node, connects
       every part, and publishes handles as ``robot.handles[<name>]``.
   * - Use
     - ``get_observation`` and ``send_action`` fan out across independent
       connections in parallel.
   * - Disconnect
     - ``Robot.disconnect`` disconnects the parts, then releases the connections
       behind them.

Building a robot does not connect it. ``Robot.build`` composes declarations and
returns; call ``connect`` before you read or command anything. Until you do,
``is_connected`` is ``False`` and the slots still hold declarations.

If a part fails while connecting, everything already placed or connected is torn
down and the slots go back to their declarations, so you can fix the cause and
call ``connect`` again. ``disconnect`` restores them too, so a robot can be
connected, disconnected, and connected again.

An end effector may be declared as its own part only when the arm does not
already open one. The Franka arms build their gripper on their own connection
during ``connect``, so declaring a Robotiq gripper alongside would open the same
serial port twice; ``compose_arms`` rejects that rather than letting it fail on
hardware.


The Boundary
------------

Keep Ray, Gymnasium, and ``rlinf.scheduler`` out of parts. Importing a part must
not load the scheduler into the process. This boundary lets the bench scripts in
``toolkits/realworld_check`` run on a machine without a cluster.

Only ``rlinf/robotics/placement.py`` crosses this boundary. ``spawn`` imports it
lazily. The scheduler never imports robotics.
``tests/unit_tests/test_robotics_boundaries.py`` enforces both directions.

Where the Code Lives
--------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Contents
   * - ``parts/base.py``
     - The part taxonomy: ``RobotPart``, ``ControllablePart``, ``Camera``,
       ``EndEffector``, ``Arm``, ``MobileBase``, ``LeggedBase``.
   * - ``parts/arms/``
     - Arm hardware and the state dataclass for each family.
   * - ``parts/cameras/``
     - RealSense, ZED, Lumos.
   * - ``parts/end_effectors/``
     - ``grippers/`` and ``hands/``.
   * - ``parts/teleop/``
     - Leader arms and input devices: GELLO, glove, keyboard, Pico, and
       spacemouse.
   * - ``parts/transports/``
     - Shared transports such as ROS. They are not parts; they carry messages for
       a part.
   * - ``robots/``
     - One module per robot, containing its config, discovery, and builder.
   * - ``specs.py``
     - ``PartSpec`` and ``SubpartRef``: a declared part, and a reference into
       one of its parts.
   * - ``placement.py``
     - ``PartHandle`` and the synthesized worker. This is the only scheduler
       import.
   * - ``views.py``
     - The ``Method*`` views.
   * - ``robot.py``, ``discovery.py``, ``adapters.py``, ``config.py``
     - Composition, registration, legacy policy adapters, and environment
       variable config.

Tasks Stay Out of Hardware
--------------------------

Keep task logic out of hardware code. A part knows how to move and what it
senses, but not what counts as success. Put reset behavior, reward, termination,
and Gymnasium spaces in a ``RobotTask``. Combine it with a ``Robot`` through
``RobotTaskEnv``. Use ``LegacyObservationAdapter`` and ``VectorActionAdapter``
to translate the composed interface into the flat vectors and ``state``/``frames``
observations expected by an existing policy. Hardware code never needs the
policy schema.

Next
----

- :doc:`Adding a Robot <../extending/new_robot>`: follow the step-by-step guide.
- :doc:`Placement <placement>`: learn how workers map onto nodes and GPUs.
