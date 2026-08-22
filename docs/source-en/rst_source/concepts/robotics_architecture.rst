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
   └── arm
       └── end_effector

The two views answer different questions:

- The **robot tree** says what the policy can observe or command.
- The **hardware connection** says what must be opened on one node and released
  once.

The code keeps those views in separate mappings. The distinction matters even
when the same names appear in both:

- A ``PartGroup`` or ``Robot`` stores its public tree in ``children``. Each key
  becomes an observation and action path, such as ``left.arm``. Tasks, policies,
  and datasets use these names.
- A ``Connection`` lists the logical parts backed by one hardware session in
  ``parts``. For a readable part, the mapping contains the parts mounted on it;
  for a bare shared session, it contains the parts that can be selected with
  ``part(name)``. These names belong to the driver and do not become robot
  paths by themselves.

Composition joins the mappings. ``Robot(arm=connection)`` names a part, and
what rides on that part comes with it, one level down: an arm's gripper is at
``arm.end_effector`` because that is where the gripper is. These public paths,
their placement, and their ownership are available before ``connect()``, which
is why a composition can be inspected on a machine with no robot attached.

``connection.part(name)`` picks one part out of a link that is not a part
itself, such as a session driving two arms. That is the only case that needs
it.

Choose the form that matches what you are composing:

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Value
     - Composition
     - Result
   * - A part with nothing riding it, such as a camera
     - ``Robot(wrist=camera)``
     - The part enters the tree under ``wrist``.
   * - A part that carries others, such as an arm with a gripper
     - ``Robot(arm=connection)``
     - The arm enters under ``arm``, its gripper under ``arm.end_effector``.
   * - A link that is not a part, such as a two-arm session
     - ``Robot(left=session.part("left"))``
     - The named part enters the tree under ``left``.
   * - An existing subtree
     - ``Robot(left=PartGroup(...))``
     - The group and its named children enter under ``left``.

``part(name)`` returns a ``RobotPart`` -- there is no intermediate type for a
robot author to construct or annotate. ``PartGroup`` accepts a ``RobotPart`` or
another ``PartGroup``, and rejects a bare ``Connection`` that cannot be read,
naming the keyword that is wrong.

``children`` is one question with one answer, whatever it is asked of. For a
part it is what rides on it; for a ``PartGroup`` it is what the group was
composed of. Walking the tree -- to describe it, to find every camera, to read
it -- therefore never asks which kind of thing it is holding.

That is what lets a robot name an arm and stop there:

.. code-block:: python

   class ExampleRobot(Robot):
       @classmethod
       def build_arms(cls, **config):
           return {"arm": ExampleArm(config["robot_ip"], node_rank=config["node_rank"])}

The gripper is composed because the arm carries it. Naming it here as well
would put it *beside* the arm rather than on it, and would be a second list to
keep in step: an arm that decides at run time whether a gripper is fitted, or
that grows a part later, would be composed without it and nothing would report
the omission.

The mapping a driver returns from ``parts`` therefore says what rides on it and
never itself. Listing itself is refused, because a part does not ride itself and
the tree would have no bottom.

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
     - The outermost ``PartGroup``. It also owns registration and knows which
       node each of its connections runs on.

There is no separate type for a part running elsewhere. A connection given a
``node_rank`` is rebuilt in a worker on that node, and the object you already
hold becomes a view of it: the same object, a synthesized subclass, and every
public call now travelling. ``isinstance`` continues to match the original
driver and its device category. A category such as ``Camera`` or ``MobileBase``
needs nothing registered for placement, because the view is derived from the
driver class itself.

Select an Implementation from Configuration
-------------------------------------------

Suppose a camera config says ``camera_type: zed``. The robot builder should not
need a switch statement that imports every camera driver. Instead, each driver
registers the names that configs use, and the device family resolves the name:

.. code-block:: python

   @Camera.register("example")
   class ExampleCamera(BaseCamera):
       ...


   camera_cls = Camera.backend(camera_info.camera_type)
   camera = camera_cls(camera_info, node_rank=2)

``Connection.register()`` and ``backend()`` are inherited by every device
family, and the registry belongs to the *category* -- ``Camera``, ``Arm``,
``EndEffector`` -- because a config names a kind of device rather than a base
class. Backend names are case-insensitive, and registering the same name for
two classes is an error.

A family adds a builder when its config has a standard shape. ``Camera.of()``
takes a ``CameraInfo`` and reads the backend off it; ``EndEffector.of()`` takes
a name and whatever the arm fitting it can offer; ``Arm.declare()`` maps a
robot's arm settings onto one backend's constructor. In each case the mapping
lives on the driver, next to the constructor it serves, so a new backend
arrives without editing anything that selects one.

Arms are where this matters most, because two of them drive the same hardware.
A Franka is reached through libfranka or through ROS, so both register on
``Arm`` and a robot names one:

.. code-block:: python

   class FrankaRobot(Robot):
       BACKEND = "franka_ros"


   class DualFrankaRobot(FrankaRobot):
       BACKEND = "franky"

Naming the backend is the whole of the swap. Each backend maps the standard arm
settings onto its own constructor in its own ``declare()``, next to the
constructor it serves, so the robot does not know that one stack wants a ROS
package and the other a gripper port. A setting a backend cannot honour is
refused rather than dropped, because the alternative is an arm running with an
end effector the config did not ask for.

A driver that supports hardware enumeration can also declare its vendor module
in ``SDK`` and implement ``discover()``. The shared discovery code then reports
a missing SDK clearly and validates configured camera identifiers on the node
that owns them. Vendor imports still belong in ``_open()`` or ``discover()``,
not at module import time.

Two registries appear in the robotics package, and they name different things:

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - What is named
     - Public API
     - Used for
   * - One device backend
     - ``Camera.register()`` / ``Arm.register()`` and ``backend()``
     - Selecting a driver such as ``realsense`` or ``franky`` from a config.
   * - One complete robot type
     - ``Robot.register_type()`` and ``Robot.of_type()``
     - Selecting a named robot tree and its ``RobotConfig``; registration also
       supplies the standard discovery flow unless a custom class is passed.

Registration associates the robot class, config class, discovery class, and
builder; it does not convert a ``RobotConfig`` instance into builder arguments.
``Robot.of_type()`` and ``build_robot()`` forward the keyword arguments they
receive directly to ``build()``. A registered robot should therefore give its
builder an explicit, documented signature, and the environment that receives a
``RobotInfo`` should perform any required translation in one visible place.

The environment layer uses the same registration style for teleoperation, but
keeps a separate ``TeleopBackend`` registry. A teleop name selects a device and
the binding that gives its reading meaning for an environment; it does not
select a robot component. Keeping this registry under
``rlinf/envs/real/wrappers/teleop`` prevents Gymnasium configuration from
leaking into the robotics package.

Connect a Shared Hardware Session to the Robot Tree
---------------------------------------------------

Define the ``parts`` mapping to say what rides on a connection. This arm is
readable itself and also carries its gripper, so it lists the gripper and not
itself:

.. code-block:: python

   class ExampleArm(ControllablePart):
       @property
       def parts(self) -> dict[str, RobotPart]:
           return {
               "end_effector": MethodEndEffector(
                   self, state_field="gripper_position"
               ),
           }

``end_effector`` is a name local to the driver. Composing the arm publishes it
one level below whatever the robot calls the arm:

.. code-block:: python

   connection = ExampleArm(
       "10.0.0.2",
       node_rank=1,
       worker_name="ExampleArm-0-0",
   )
   robot = Robot(arm=connection)

   robot.children                     # {"arm": <ExampleArm>}
   robot.child("arm").children        # {"end_effector": <MethodEndEffector>}
   robot.get_observation()["arm"]     # the arm's fields, plus "end_effector"

The keyword arguments become ``robot.children``; the driver's own names appear
beneath the part that carries them. A bare ``Connection`` answers no
``children`` at all, because it has no place in the tree -- its parts are picked
one at a time with ``part(name)``. A ``PartGroup`` has an empty ``parts``
mapping, because nothing rides a group.

Selecting a part with ``part(name)`` also tells a connection-backed view which
connection opens it, so the view declares no lifecycle of its own: no
``_open()`` and no ``connect()`` override. Use ``parts`` for these borrowed
views. A device with its own link, such as a wrist camera on USB, should be
composed explicitly as another child of the robot or of an assembly
``PartGroup``. That explicit form gives the camera its own owner and ensures
``Robot.connect()`` opens it on the node it named.

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
       arm=arm_connection,
       scene=RealSenseCamera(camera_info, node_rank=3),
   )

   print(robot.describe())
   robot.connect()
   try:
       observation = robot.get_observation()
   finally:
       robot.disconnect()

During ``connect()``, the robot opens each distinct ``Connection`` once. With no
``node_rank`` that happens in this process. With one, the connection is rebuilt
inside a scheduler worker on that node and the object in the tree takes on a
synthesized subclass of its own class whose public methods and properties
forward to the worker. The object identity is preserved, although its concrete
class changes while it is connected. Task code therefore never branches on
placement, and ``isinstance`` keeps answering what it did before.

Identity is what preserves resource ownership. Every part answers ``owner``
with the connection opened on its behalf -- itself for an arm holding its own
link, the shared session for a view riding one -- and the robot connects owners
rather than parts. Parts on different connections may run concurrently; parts
sharing one run in declaration order, because vendor sessions are rarely safe
for concurrent access.

Inspect the Composition Before Connecting
-----------------------------------------

``Robot.describe()`` reads the composed tree, so node and ownership information
is available before any hardware is opened:

.. code-block:: text

   FrankaRobot
   └── arm                 FrankaROSArm         node=1     via FrankaROSArm#1
       └── end_effector    MethodEndEffector    node=1     via FrankaROSArm#1

Rows sharing ``via`` share one ``Connection``. After connecting, a placed part
uses a synthesized class name such as RemoteFrankaROSArm. Its path,
``node``, and ownership stay the same, but the complete output string is not a
stable serialization format; use it as a diagnostic rather than storing or
parsing it.

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
     - The robot opens each distinct connection once, on the node that
       connection named. A placed one is rebuilt there and the local object
       becomes a view of it.
   * - Use
     - The robot reads, resets, and commands the named tree, concurrently where
       resource ownership allows it.
   * - Disconnect
     - Connections release the exact device returned by ``_open()``. A placed
       one closes on its node before its worker is stopped, and becomes an
       ordinary unopened object again.

``_open()`` returns the vendor object, and ``_release(device)`` receives that
same object. Cleanup should release the argument rather than look it up again on
``self``.

Implement those two, never ``connect()`` and ``disconnect()``. The public pair
decides *where* a device runs, so a part that overrode them would opt itself
out of ever being placed -- and a thread started after ``super().connect()``
would start on the machine holding the part rather than the one holding the
device. A device category that wraps its drivers has ``_opened()`` and
``_closing()`` for that: ``BaseCamera`` starts and stops its capture loop
there, beside the camera wherever it ended up.

Robot startup rolls back the connections that completed successfully if a
later connection fails. A driver's ``_open()`` must still release anything it
acquired before raising, because no completed connection exists for the robot
to close in that case. After fixing the hardware, you can call ``connect()`` on
the same robot again. ``disconnect()`` is idempotent and returns successfully
closed connections to a reconnectable state.

Reach Device-Specific Methods on the Part Itself
------------------------------------------------

Both halves of placement are derived from the driver class, so a method outside
the standard part contract travels for the same reason the driver has it. Ask a
part which connection it rides, then call the method:

.. code-block:: python

   controller = robot.child("arm").owner
   controller.is_robot_up()
   controller.reset_joint(home_qpos)

The expression is the same whether that arm is on this bench or on another
node, and there is no result to unwrap. Keep task code on the standard
observation and action tree; reach for the connection for setup, diagnostics,
or a vendor operation with no canonical part method.

Preserve the Import Boundary
----------------------------

Part modules must not import ``rlinf.scheduler`` or Gymnasium.
``rlinf/robotics/placement/handles.py`` is the bridge, loaded lazily by
``Connection.connect()`` and only for a connection that named a node.

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
     - The ``Arm`` category and ``BaseArm``, then the backends that register on
       them: Franky, Franka ROS, GimArm, and the coupled controllers.
   * - ``robotics/parts/cameras/``
     - Camera lifecycle and RealSense, ZED, and Lumos implementations.
   * - ``robotics/parts/end_effectors/``
     - Grippers and dexterous hands.
   * - ``robotics/parts/mobility/``
     - The ``MobileBase`` category and mobile-platform drivers.
   * - ``robotics/parts/views.py``
     - ``MethodArm``, ``MethodEndEffector``, and ``MethodCamera`` views over shared
       vendor sessions.
   * - ``robotics/placement/``
     - The worker that hosts a connection, and the view a placed connection
       becomes. Both are synthesized from the driver class.
   * - ``robotics/robot.py``
     - The outer composition, description, and lifecycle.
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
