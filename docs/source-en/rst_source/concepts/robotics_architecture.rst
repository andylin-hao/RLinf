Robotics Architecture
=====================

The introductory :doc:`robotics` page treats a robot as a tree of named parts.
This page follows that tree down to hardware sessions and across machine
boundaries. Read it when you are adding a device, sharing one connection between
several parts, or debugging placement and cleanup.

Start from the Public Model
---------------------------

Suppose a controller opens one ROS session for an arm and its gripper. A task
should still see two parts:

.. code-block:: text

   robot
   ├── arm
   └── end_effector

The distinction matters because the two views answer different questions:

- The **robot tree** says what the policy can observe or command.
- The **hardware session** says what must be opened, placed, and released once.

RLinf keeps those questions separate with ``children`` and ``exports``. Most of
the architecture follows from that one choice.

The Core Types
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Type
     - Role
   * - ``Endpoint``
     - Something RLinf opens on a machine and later closes. It owns a location
       and a lifecycle, but it is not necessarily observable.
   * - ``RobotPart``
     - An endpoint that exposes ``get_observation()`` and an
       ``observation_features`` contract.
   * - ``ControllablePart``
     - A ``RobotPart`` that also exposes ``send_action()`` and an
       ``action_features`` contract.
   * - ``Connection``
     - An endpoint that owns one hardware session for several parts. It is not
       itself a part, so it cannot appear in a robot's observation tree.
   * - ``Group``
     - A part composed from named ``children``. A group may represent one arm
       assembly, a torso, or the complete robot.
   * - ``Robot``
     - The outermost ``Group``. It also owns placement, registration, and the
       handles created during ``connect()``.
   * - ``PartSpec``
     - A deferred declaration produced by a part class's ``at()`` method. It
       records that class, constructor arguments, a node, and an optional
       worker name.
   * - ``PartHandle``
     - A uniform reference to a local part or a part hosted in a worker.

The specific ``Camera``, ``EndEffector``, ``MobileBase``, and ``LeggedBase``
types preserve a device category when composition or a remote proxy needs it.

Keep ``exports`` and ``children`` Distinct
------------------------------------------

Use ``exports`` to describe what one hardware session makes available. For
example, an arm connection can export the arm itself and a gripper view:

.. code-block:: python

   class ExampleArm(ControllablePart):
       @property
       def exports(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(
                   self, state_field="gripper_position"
               ),
           }

Export names belong to that connection. They are not automatically the names a
robot must publish. Composition makes that choice explicitly:

.. code-block:: python

   connection = ExampleArm.at("10.0.0.2", node_rank=1)
   robot = Robot(
       arm=connection.export("arm"),
       end_effector=connection.export("end_effector"),
   )

Now ``robot.children`` contains ``arm`` and ``end_effector``. The connection
has no ``children``, and the group exports nothing. This makes ownership hard
to misread: ``exports`` belongs to the hardware session; ``children`` belongs
to the named robot tree. Composition is the only place they meet.

The same rule handles a coupled controller without pretending the controller is
an arm:

.. code-block:: python

   connection = Turtle2Connection.at(50, camera_ids, node_rank=0)
   robot = Turtle2Robot(
       left=Group(
           arm=connection.export("left"),
           end_effector=connection.export("left_end_effector"),
       ),
       right=Group(
           arm=connection.export("right"),
           end_effector=connection.export("right_end_effector"),
       ),
       wrist=connection.export("wrist_1"),
   )

One declaration backs every reference, so the controller is opened once.

Declare Placement, Then Connect
-------------------------------

``at()`` records where a part should be built; it does not construct the part or
touch hardware:

.. code-block:: python

   arm_connection = ExampleArm.at("10.0.0.2", node_rank=1)
   robot = Robot(
       arm=arm_connection.export("arm"),
       end_effector=arm_connection.export("end_effector"),
       scene=RealSenseCamera.at(camera_info, node_rank=3),
   )

   print(robot.describe())
   robot.connect()
   try:
       observation = robot.get_observation()
   finally:
       robot.disconnect()

During ``connect()``, the robot resolves each distinct ``PartSpec`` once. A
local spec becomes a ``LocalPartHandle``. A remote spec is hosted in a scheduler
worker and becomes a ``RemotePartHandle``. Both expose the same public methods,
so callers do not branch on placement.

Placement also preserves resource ownership. If three robot paths refer to one
spec, the robot creates one handle and releases it once. Parts backed by
different handles may run concurrently; parts backed by the same handle are
called in their declared order because vendor sessions are rarely safe for
concurrent access.

Use ``spawn()`` only when no ``Robot`` owns the endpoint, such as a standalone
bench script. It places immediately and hands lifecycle management to the
caller. Inside a robot, prefer ``at()`` so connection rollback and cleanup stay
automatic.

Inspect a Declaration Before Opening Hardware
---------------------------------------------

``Robot.describe()`` reads the declaration snapshot rather than the live proxy.
As a result, node and ownership information remains stable before, during, and
after a connection:

.. code-block:: text

   FrankaRobot
   ├── arm           declared      node=1     via FrankaROSArm#1
   └── end_effector  declared      node=1     via FrankaROSArm#1

Rows sharing ``via`` share one endpoint. A direct part can report its category
before connection. A reference into an unopened ``Connection`` is shown as
``declared`` because that connection does not know which concrete parts it will
export until it opens. The description does not invent a category from the
connection itself.

At present, ``describe()`` focuses on topology, placement, and ownership. It
does not print observation or action feature schemas. In particular, an
unopened ``Connection`` cannot describe the concrete parts it will export. Use
the conformance checks in :doc:`../extending/new_robot` to validate those
schemas after opening the connection with a fake SDK or the real device.

Follow the Lifecycle
--------------------

The lifecycle has four stages:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage
     - What happens
   * - Declare
     - ``at()`` records construction and placement. No SDK is imported and no
       hardware is opened.
   * - Connect
     - The robot places each declaration, opens its endpoint, resolves exported
       parts, and publishes handles in ``robot.handles``.
   * - Use
     - The robot reads, resets, and commands the named tree, concurrently where
       resource ownership allows it.
   * - Disconnect
     - Parts release the exact device returned by ``_open()``, handles shut
       down, and the robot restores its declarations.

More precisely, ``_open()`` returns the vendor object and ``_release(device)``
receives that same object. Passing it explicitly keeps cleanup independent of
when ``_device`` is cleared.

Connection is all-or-nothing. If a later endpoint fails, ``Robot.connect()``
tears down everything it already placed or opened and restores the declaration
tree. After fixing the hardware, you can call ``connect()`` on the same robot
again. ``disconnect()`` is idempotent and returns the robot to the same
reconnectable state.

Reach Device-Specific Methods Through a Handle
----------------------------------------------

Placement creates the worker surface from the part class, so you do not write a
second worker class for every device. Public methods outside the standard part
contract remain callable through the handle:

.. code-block:: python

   robot.handles["arm"].is_robot_up().wait()[0]
   robot.handles["arm"].reset_joint(home_qpos).wait()

The expression is the same for local and remote parts. Keep task code on the
standard observation and action tree; use a handle for setup, diagnostics, or a
vendor operation that genuinely has no canonical part method.

Preserve the Import Boundary
----------------------------

Part modules must not import Ray, Gymnasium, or ``rlinf.scheduler``. A hardware
machine should be able to import and test its driver without loading the
cluster. ``rlinf/robotics/placement/handles.py`` is the one bridge: placement
imports it lazily when a remote part is requested.

The dependency also stays one-way. The scheduler does not import robotics.
``tests/unit_tests/test_robotics.py`` checks both directions.

Find the Implementation
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Path
     - Contents
   * - ``robotics/parts/base.py``
     - ``Endpoint``, part categories, ``Group``, and ``Connection``.
   * - ``robotics/parts/arms/``
     - Arm and coupled-controller implementations.
   * - ``robotics/parts/cameras/``
     - Camera lifecycle and RealSense, ZED, and Lumos implementations.
   * - ``robotics/parts/end_effectors/``
     - Grippers and dexterous hands.
   * - ``robotics/parts/views.py``
     - ``MethodArm``, ``MethodGripper``, and ``MethodCamera`` views over shared
       vendor sessions.
   * - ``robotics/placement/``
     - Deferred specs, local and remote handles, and worker placement.
   * - ``robotics/robot.py``
     - The outer composition, declaration snapshot, description, and lifecycle.
   * - ``robotics/discovery/``
     - Robot registration, discovery, and configuration lookup.

Next
----

- :doc:`Adding a Robot <../extending/new_robot>` applies these pieces in order.
- :doc:`Placement <placement>` explains how scheduler resources map onto nodes
  and GPUs.
- :doc:`Teleoperation <../guides/teleoperation>` composes operator devices with
  bindings on the environment side.
