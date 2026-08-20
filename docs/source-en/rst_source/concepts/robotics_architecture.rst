Robotics Architecture
=====================

The introductory :doc:`robotics` page treats a robot as a tree of named parts.
This page follows that tree down to hardware connections and across machine
boundaries. Read it when you are adding a device, sharing one connection between
several parts, or debugging placement and cleanup.

Start from the Public Model
---------------------------

Start with the common one-to-one case: one hardware connection represents one
logical part. A mobile base can enter the robot tree directly:

.. code-block:: python

   base = ExampleMobileBase(
       "tcp://mobile-base:7000",
       node_rank=0,
       worker_name="ExampleMobileBase-0-0",
   )
   robot = Robot(base=base)

Constructing the base creates an actual but unconnected ``MobileBase``. It
stores the device settings, while the ``Connection`` metaclass records
``node_rank`` and ``worker_name`` for placement. No SDK is imported and no
hardware is opened until ``robot.connect()``. Because the object is already the
logical part a policy should see, passing it as ``base=base`` adds it directly
to the public tree. The argument name ``base`` becomes its public path.

One hardware session may instead back several logical parts. Suppose a
controller opens one ROS session for an arm and its gripper. A task should still
see two parts:

.. code-block:: text

   robot
   ├── arm
   └── end_effector

The two views answer different questions:

- The **robot tree** says what the policy can observe or command.
- The **hardware connection** says what must be opened on one node and released
  once.

The code keeps those views in separate mappings:

- A ``PartGroup`` or ``Robot`` stores its public tree in ``children``. Each key
  becomes an observation and action path, such as ``left.arm``. Tasks, policies,
  and datasets use these names.
- A ``Connection`` lists the logical parts backed by one hardware session in
  ``parts``. These names belong to the driver and do not become robot paths by
  themselves.

Composition joins the mappings. ``connection.part("arm")`` records a choice from
the connection, and ``Robot(arm=...)`` gives that choice the public name ``arm``.
The choice remains unresolved until ``Robot.connect()`` opens the connection and
can inspect the parts it backs.

Choose the form that matches what you are composing:

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Value
     - Composition
     - Result
   * - One readable part, such as a mobile base or camera
     - ``Robot(base=base)``
     - The part enters the tree under ``base``.
   * - One session that backs several parts
     - ``Robot(arm=connection.part("arm"))``
     - The selected part enters the tree under ``arm``.
   * - An existing subtree
     - ``Robot(left=PartGroup(...))``
     - The group and its named children enter under ``left``.

The result of ``part(name)`` is an internal deferred choice, not another public
type a robot author needs to construct or annotate. ``PartGroup`` accepts that
choice, a ``RobotPart``, or another ``PartGroup``. It rejects a bare
``Connection`` that cannot be read and reports which keyword is invalid.

Some objects are both a connection and a readable part. An arm session, for
example, may expose the arm's own observation while also backing an end
effector. Passing such an object directly is valid; if it backs several parts,
it resolves to a ``PartGroup`` under that keyword. Use ``part(name)`` when those
parts should appear as separate siblings or under different public names.

The Core Types
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Type
     - Role
   * - ``Connection``
     - One link to hardware. It records where it runs, opens the vendor session,
       and releases it. A bare connection need not be observable.
   * - ``RobotPart``
     - A readable ``Connection`` with ``get_observation()`` and an
       ``observation_features`` contract.
   * - ``ControllablePart``
     - A ``RobotPart`` that also exposes ``send_action()`` and an
       ``action_features`` contract.
   * - ``PartGroup``
     - A readable, controllable subtree composed from named ``children``. It may
       represent an arm assembly, a torso, or another nested unit.
   * - ``Robot``
     - The outermost ``PartGroup``. It also owns placement, registration, the
       declaration snapshot, and handles created during ``connect()``.
   * - ``PartHandle``
     - A uniform reference to a local connection or one hosted in a worker.

``Camera`` and ``EndEffector`` register specialized remote-proxy categories
because they add methods to the standard part interface. ``MobileBase`` adds no
extra methods, so a hosted base uses the normal controllable-part proxy. It is
still a useful category for local code and backend registration: wheeled and
legged bases both inherit ``MobileBase``, while their observation and action
contracts describe the locomotion interface.

Select an Implementation from Configuration
-------------------------------------------

Suppose a camera config says ``camera_type: zed``. The robot builder should not
need a switch statement that imports every camera driver. Instead, each driver
registers the names that configs use, and the device family resolves the name:

.. code-block:: python

   @BaseCamera.register("example")
   class ExampleCamera(BaseCamera):
       ...


   camera_cls = BaseCamera.backend(camera_info.camera_type)
   camera = camera_cls(camera_info, node_rank=2)

``Connection.register()`` and ``backend()`` are inherited by device families
such as ``BaseCamera`` and ``MobileBase``. Backend names are case-insensitive,
and registering the same name for two classes is an error. A family may add an
``of()`` or ``declare()`` helper when its config has a standard shape;
``BaseCamera.of()`` uses ``CameraInfo.camera_type`` and
``BaseCamera.declare()`` returns cameras ready to add to a robot.

A driver that supports hardware enumeration can also declare its vendor module
in ``SDK`` and implement ``discover()``. The shared discovery code then reports
a missing SDK clearly and validates configured camera identifiers on the node
that owns them. Vendor imports still belong in ``_open()`` or ``discover()``,
not at module import time.

Three registries appear in the robotics code, but they name different things:

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - What is named
     - Public API
     - Used for
   * - One device backend
     - ``BaseCamera.register()`` and ``BaseCamera.backend()``
     - Selecting a driver such as ``realsense`` or ``zed`` from a device config.
   * - One complete robot type
     - ``Robot.register_type()`` and ``Robot.of_type()``
     - Selecting a named robot tree and its ``RobotConfig``; registration also
       supplies the standard discovery flow unless a custom class is passed.
   * - One remote proxy category
     - ``register_kind()``
     - Rebuilding a hosted camera or end effector with the correct interface.
       Framework category authors use this; ordinary driver authors do not.

Connect a Shared Hardware Session to the Robot Tree
---------------------------------------------------

Define the ``parts`` mapping when one session backs several policy-facing
parts. For example, this arm is readable itself and also presents its gripper as
a separate view:

.. code-block:: python

   class ExampleArm(ControllablePart):
       @property
       def parts(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(
                   self, state_field="gripper_position"
               ),
           }

The keys ``arm`` and ``end_effector`` are names local to the driver. Choose the
public paths when you compose the robot:

.. code-block:: python

   connection = ExampleArm(
       "10.0.0.2",
       node_rank=1,
       worker_name="ExampleArm-0-0",
   )
   robot = Robot(
       arm=connection.part("arm"),
       end_effector=connection.part("end_effector"),
   )

The keyword arguments become ``robot.children``, so the robot publishes
``arm`` and ``end_effector``. A bare ``Connection`` has no ``children`` because
it composes nothing. A ``PartGroup`` inherits an empty ``parts`` mapping because
its components are already in ``children``. The call to ``connection.part(...)``
is where the two naming systems meet.

If reading the shared session itself has no useful meaning, subclass
``Connection`` rather than ``RobotPart``. A coupled Turtle2 controller follows
that form:

.. code-block:: python

   connection = Turtle2Connection(
       50,
       tuple(camera_ids),
       node_rank=0,
       worker_name="Turtle2Connection-0-0",
   )
   robot = Turtle2Robot(
       left=PartGroup(
           arm=connection.part("left"),
           end_effector=connection.part("left_end_effector"),
       ),
       right=PartGroup(
           arm=connection.part("right"),
           end_effector=connection.part("right_end_effector"),
       ),
       wrist=connection.part("wrist_1"),
   )

Every choice points back to the same connection object, so the controller opens
once and is released once.

Choose Placement Before Opening Hardware
----------------------------------------

Pass placement alongside the hardware constructor arguments. Construction is
inert; ``Robot.connect()`` decides whether to open the existing object locally
or rebuild it in a worker on the selected node:

.. code-block:: python

   arm_connection = ExampleArm(
       "10.0.0.2",
       node_rank=1,
       worker_name="ExampleArm-0-0",
   )
   robot = Robot(
       arm=arm_connection.part("arm"),
       end_effector=arm_connection.part("end_effector"),
       scene=RealSenseCamera(camera_info, node_rank=3),
   )

   print(robot.describe())
   robot.connect()
   try:
       observation = robot.get_observation()
   finally:
       robot.disconnect()

During ``connect()``, the robot opens each distinct ``Connection`` once. A
local connection gets a ``LocalPartHandle``. A remote connection is rebuilt in
a scheduler worker and gets a ``RemotePartHandle``. Both present the same part
surface, so task code does not branch on placement.

Identity preserves resource ownership. Several robot paths that originate from
one connection share one handle and one cleanup operation. Parts on different
handles may run concurrently; parts sharing a handle run in declaration order
because vendor sessions are rarely safe for concurrent access.

Use ``spawn()`` only outside a robot, such as in a bench script that owns the
returned handle. Inside a robot, construct the unconnected part normally and
let ``Robot.connect()`` own startup, rollback, and cleanup.

Inspect the Composition Before Connecting
-----------------------------------------

``Robot.describe()`` reads the composition snapshot rather than a live proxy.
Node and ownership information therefore remains stable before, during, and
after a connection:

.. code-block:: text

   FrankaRobot
   ├── arm           declared      node=1     via FrankaROSArm#1
   └── end_effector  declared      node=1     via FrankaROSArm#1

Rows sharing ``via`` share one ``Connection``. A directly composed part can
report its category before connection. A choice from an unopened connection is
shown as ``declared`` because the concrete backed part is not known until that
connection opens; ``describe()`` does not substitute the connection's own kind.

At present, ``describe()`` focuses on topology, placement, and ownership. It
does not print observation or action feature schemas. Use the conformance checks
in :doc:`../extending/new_robot` to validate those schemas after opening the
connection with a mock SDK or the real device.

Follow the Lifecycle
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage
     - What happens
   * - Compose
     - Constructors record hardware settings and placement. They do not import
       the vendor SDK or open hardware.
   * - Connect
     - The robot opens each distinct connection, resolves selected parts, and
       publishes handles in ``robot.handles``.
   * - Use
     - The robot reads, resets, and commands the named tree, concurrently where
       resource ownership allows it.
   * - Disconnect
     - Connections release the exact device returned by ``_open()``, handles
       shut down, and the robot restores its original composition.

``_open()`` returns the vendor object, and ``_release(device)`` receives that
same object. Cleanup should release the argument rather than look it up again on
``self``.

Connection is all-or-nothing. If a later connection fails, ``Robot.connect()``
tears down everything it already opened and restores the composition. After
fixing the hardware, you can call ``connect()`` on the same robot again.
``disconnect()`` is idempotent and returns the robot to the same reconnectable
state.

Reach Device-Specific Methods Through a Handle
----------------------------------------------

Placement creates the worker surface from the connection class, so you do not
write another worker class for every device. Public methods outside the standard
part contract remain callable through the handle:

.. code-block:: python

   robot.handles["arm"].is_robot_up().wait()[0]
   robot.handles["arm"].reset_joint(home_qpos).wait()

The expression is the same for local and remote parts. Keep task code on the
standard observation and action tree; use a handle for setup, diagnostics, or a
vendor operation that has no canonical part method.

Preserve the Import Boundary
----------------------------

Part modules must not import ``rlinf.scheduler`` or Gymnasium.
``rlinf/robotics/placement/handles.py`` is the bridge, loaded lazily by
``Connection.place()`` when placement is needed.

The point is the direction of the dependency. The scheduler is a general
framework and robotics is one extension of it, so the scheduler never imports
this package: it imports the hardware-policy modules a config names, then calls
the discovery classes those modules registered. Gymnasium sits on the other
side, in the env layer that consumes a robot. Only the composition layer --
placement, discovery, the robot builders -- imports either back, which is what
lets a driver be read and tested as hardware code.

Ray is not on that list. It is a base dependency of RLinf, so every machine
running it already has Ray and forbidding the name buys nothing. Nor is this a
promise that importing a part loads nothing: a part may use ``rlinf.utils``
helpers such as ``get_logger``, and those reach further.
``tests/unit_tests/test_robotics.py`` checks both directions.

Find the Implementation
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Path
     - Contents
   * - ``robotics/parts/base.py``
     - ``Connection``, ``RobotPart``, ``ControllablePart``, ``PartGroup``, and
       the composition checks and driver registry.
   * - ``robotics/parts/arms/``
     - Arm and coupled-controller implementations.
   * - ``robotics/parts/cameras/``
     - Camera lifecycle and RealSense, ZED, and Lumos implementations.
   * - ``robotics/parts/end_effectors/``
     - Grippers and dexterous hands.
   * - ``robotics/parts/mobility/``
     - The ``MobileBase`` category and mobile-platform drivers.
   * - ``robotics/parts/views.py``
     - ``MethodArm``, ``MethodGripper``, and ``MethodCamera`` views over shared
       vendor sessions.
   * - ``robotics/placement/``
     - Connection resolution, local and remote handles, and worker placement.
   * - ``robotics/robot.py``
     - The outer composition, declaration snapshot, description, and lifecycle.
   * - ``robotics/discovery/``
     - Robot type registration, standard hardware enumeration, environment
       variable completion, and configuration lookup.

Next
----

- :doc:`Adding a Robot <../extending/new_robot>` applies these pieces in order.
- :doc:`Placement <placement>` explains how scheduler resources map onto nodes
  and GPUs.
- :doc:`Teleoperation <../guides/teleoperation>` composes operator devices with
  bindings on the environment side.
