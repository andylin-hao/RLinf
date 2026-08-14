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

Declare subparts when hardware does not map one-to-one onto components. For
example, a coupled dual-arm controller can drive two arms, two grippers, and two
wrist cameras over one ROS connection. Let the part declare what it exposes
instead of adding an abstraction for "the thing that owns the connection":

.. code-block:: python

   def subparts(self) -> dict[str, RobotPart]:
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
   * - ``subparts()``
     - The named components exposed by a part. Leaf parts return ``{}``.
   * - ``Arm``
     - A part that combines a manipulator, an optional end effector, and wrist
       cameras.
   * - ``Robot``
     - A composition of named arms, robot-level cameras, extra parts, and its
       owned handles.
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

Set the arm count through the size of a mapping. Use the same class for
single-arm and dual-arm robots:

.. code-block:: python

   single = FrankaRobot.single_arm(Arm(arm, gripper))
   dual = FrankaRobot.dual_arm(Arm(left, left_gripper), Arm(right, right_gripper))

Use composition to define the data shape seen by the policy. Names become paths:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Path
     - Meaning
   * - ``arms.<name>.state``
     - That arm's manipulator observation.
   * - ``arms.<name>.arm``
     - That arm's manipulator action.
   * - ``arms.<name>.end_effector``
     - Its end-effector observation and action.
   * - ``cameras.<name>`` / ``parts.<name>``
     - Robot-level cameras and extra components.

Let ``Robot`` reset, read, and command arms in parallel when they use independent
connections. A two-arm observation then costs one round trip, not two.

Placement Is a Property of Parts
--------------------------------

Declare where a part runs with ``at()``. Nobody calls a placement function:
:meth:`Robot.connect` builds every declaration on its node.

.. code-block:: python

   robot = FrankaRobot(
       arms={"left": Arm(FrankaROSArm.at("10.0.0.1", node_rank=1))},
       cameras={"scene": RealSenseCamera.at(info, node_rank=3)},
   )
   robot.connect()

A declaration is inert. Nothing touches hardware until ``connect``, which places
each distinct declaration exactly once, publishes its handle as
``robot.handles[<name>]``, and tears down whatever it already placed if a later
part fails. ``disconnect`` releases them.

Declare a shared connection once and refer to its subparts. One connection
backing two arms and two cameras is opened once, not four times:

.. code-block:: python

   hardware = Turtle2Hardware.at(50, camera_ids, node_rank=0)
   robot = Turtle2Robot.dual_arm(
       Arm(hardware.subpart("left"), hardware.subpart("left_end_effector")),
       Arm(hardware.subpart("right"), hardware.subpart("right_end_effector")),
       cameras={"wrist_1": hardware.subpart("wrist_1")},
   )

Placement applies to every part, not only arms. Run a camera on the machine it
is plugged into while the policy runs elsewhere. ``spawn()`` is the eager form
underneath; reach for it only outside a robot, such as in a bench script.

Do not write a worker class for each hardware device. RLinf synthesizes one from
the part class (``type(name, (Worker, PartCls), ...)``). ``WorkerGroup`` then
binds every public method as an RPC. Call methods outside the part interface
through the handle. The call shape stays the same locally and remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

Read :doc:`Placement <placement>` to map workers onto nodes and GPUs.

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
down before the error reaches you, so there is no half-built robot to clean up.

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
     - ``PartSpec`` and ``SubpartRef``: a declared part and a reference into it.
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
