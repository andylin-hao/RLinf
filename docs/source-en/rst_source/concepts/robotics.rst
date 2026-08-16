Robotics Model
==============

Start with a robot's named parts rather than a fixed list of device slots.
``Robot.connect`` can then place those parts on the nodes beside their hardware,
while ``Robot`` presents one observation and action tree to the caller. We'll
build a Franka first, then trace that example back to the underlying classes and
lifecycle.

Assemble a Robot
----------------

Let's build a Franka from the ground up. Its class describes which arms and
cameras it carries; connection and placement happen later, when the caller has
a complete robot to start.

.. code-block:: python

   class FrankaRobot(Robot):
       ROBOT_TYPE = "Franka"       # the name build_robot() looks up
       BACKEND = "franka_ros"      # which arm implementation declare_arm creates

       @classmethod
       def build_arms(cls, *, robot_ip, node_rank, **config) -> dict[str, RobotPart]:
           # declare_arm returns one part: the arm at robot_ip, to be built on
           # node_rank, carrying whatever end effector its connection exposes.
           # It only records that description -- no hardware is touched yet.
           return {"arm": cls.declare_arm(robot_ip, node_rank=node_rank, name="arm")}

       @classmethod
       def build_cameras(cls, cameras=None, *, node_rank=None) -> dict[str, RobotPart]:
           # Same contract, for a different kind of part.
           return declare_cameras(cameras, node_rank=node_rank)


   FrankaRobot.register(FrankaConfig, FrankaDiscovery)

Each ``build_*`` method returns a ``{name: part}`` mapping for one kind of
hardware. The inherited ``build`` only has to merge those mappings, so Franka
variants can change one category without copying the whole builder:

.. code-block:: python

   @classmethod
   def build(cls, *, cameras=None, camera_node_rank=None, **config) -> "FrankaRobot":
       return cls(
           **cls.build_arms(**config),
           **cls.build_cameras(cameras, node_rank=camera_node_rank),
       )

The mapping keys become the robot's part names. Once assembly is complete,
``connect`` builds each part on its requested node; ``disconnect`` releases the
same resources.

Use a Robot
-----------

Now that the Franka is declared, the caller can use it without knowing how the
parts were assembled. Build the registered type, connect it once, then read its
state and send commands through the same interface used by other robots:

.. code-block:: python

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)
   robot.connect()

   observation = robot.get_observation()
   observation["arm"]["arm"]["tcp_pose"]            # Cartesian pose
   observation["arm"]["end_effector"]["state"]      # gripper width

   robot.send_action(
       {"arm": {"arm": {"tcp_pose": target}, "end_effector": {"target": width}}}
   )

   robot.reset()
   robot.disconnect()

Observations and actions are nested dictionaries whose keys come from the robot
composition. A dual-arm robot uses the same calls, with ``left`` and ``right``
occupying separate branches:

.. code-block:: python

   observation["left"]["arm"]["arm_joint_position"]
   observation["right"]["end_effector"]["state"]

Nothing in the calling code says where a part runs. An arm declared on another
node still reports its reading at the same path as a local arm. If the two arms
use independent connections, ``Robot.get_observation`` reads them concurrently,
so the caller waits for one round trip rather than two.

You can inspect the tree before connecting any hardware through
``observation_features`` and ``action_features``. ``RobotTaskEnv`` uses those
descriptions to construct Gymnasium spaces. If a policy expects flat vectors,
we'll return to the adapter boundary in `Tasks Stay Out of Hardware`_.

One Arm or Many Is the Same Code
--------------------------------

With that calling convention in place, moving from one arm to two changes only
the composition. ``DualFrankaRobot`` inherits the single-arm class and replaces
the mapping returned by ``build_arms``:

.. code-block:: python

   class DualFrankaRobot(FrankaRobot):
       ROBOT_TYPE = "DualFranka"
       BACKEND = "franky"

       @classmethod
       def build_arms(cls, *, left_robot_ip, right_robot_ip,
                      left_node_rank, right_node_rank, **config):
           return {
               "left": cls.declare_arm(left_robot_ip, node_rank=left_node_rank,
                                       name="left"),
               "right": cls.declare_arm(right_robot_ip, node_rank=right_node_rank,
                                        name="right"),
           }

``DualFrankaRobot`` still uses the inherited ``declare_arm``, ``build_cameras``,
and ``build`` methods, as well as the same placement and lifecycle code. A
third arm would be another entry in this mapping rather than another control
path.

Switching the Control Backend Is One Line
-----------------------------------------

Franka ships with two control stacks, and the robot class selects one through
``BACKEND``. ``declare_arm`` reads that attribute whenever it creates an arm,
whether the class declares one arm or several:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``BACKEND``
     - Arm part it declares
   * - ``"franka_ros"``
     - ``FrankaROSArm`` -- Cartesian impedance control over ROS.
   * - ``"franky"``
     - ``FrankyArm`` -- joint and Cartesian control over libfranka.

A third backend needs an arm implementation and an entry in
``FRANKA_BACKENDS``. Robot classes continue to call ``declare_arm`` rather than
naming the concrete arm class themselves.

An Arm Arrives Whole
--------------------

The gripper raises a related composition question: it shares the Franka arm's
connection, so it belongs to the arm. When the arm joins a robot, the gripper is
already present as its ``end_effector`` part:

.. code-block:: python

   robot = FrankaRobot.build(robot_ip="10.0.0.1", node_rank=1, ...)

   robot.parts                  # {"arm": ...}
   robot.part("arm").parts      # {"arm": ..., "end_effector": ...}

``build`` never mentions the gripper. The arm describes the components available
through its connection, while the robot chooses only its top-level hardware.

Any Part Runs Anywhere
----------------------

We'll get to the full placement lifecycle below. For now, the important point is
that ``node_rank`` belongs to a part declaration, so an arm and a camera can run
on different nodes:

.. code-block:: python

   robot = FrankaRobot(
       arm=FrankaROSArm.at(robot_ip, node_rank=1),      # on the arm's NUC
       wrist=RealSenseCamera.at(info, node_rank=3),     # where it is plugged in
   )

``Robot`` reads independent connections in parallel. Parts on the same
connection keep their declared order, because vendor SDKs often reject
concurrent calls over one link; the Franka class does not need to schedule those
calls itself.

The Core Idea
-------------

The examples above follow one rule: **treat every physical component as a
part.** An arm, gripper, camera, or mobile base is a ``RobotPart``. Each part
knows how to connect and report an observation, and a controllable part also
accepts an action. The device-facing behavior lives on that object rather than
behind a second "driver" abstraction.

One connection does not always correspond to one physical component. Suppose a
coupled controller drives two arms, two grippers, and two wrist cameras through
one ROS connection. The connected part lists each exposed component through
``parts``:

.. code-block:: python

   @property
   def parts(self) -> dict[str, RobotPart]:
       return {
           "left": MethodArm(self, commands={"tcp_pose": "move_left_arm"}),
           "right": MethodArm(self, commands={"tcp_pose": "move_right_arm"}),
           "left_end_effector": MethodGripper(self, state_field="follow1_pos"),
       }

Connection ownership remains an implementation detail of the part.

The two remaining rules are **compose a robot from named parts** and **place any
part on a node**. They account for the nesting and ``node_rank`` declarations in
the Franka examples.

The Abstractions
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Abstraction
     - What it is
   * - ``RobotPart``
     - Base class for a physical component. It defines ``connect``,
       ``get_observation``, ``disconnect``, and ``reset``. Its
       ``observation_features`` property describes the returned data.
   * - ``ControllablePart``
     - A part with a ``send_action`` method and an ``action_features``
       description of the accepted commands.
   * - ``Camera`` / ``EndEffector`` / ``MobileBase`` / ``LeggedBase``
     - Specific part types used when a composition or remote proxy must retain
       the device category.
   * - ``parts``
     - Named components exposed by a part. Connected hardware lists everything
       driven through that connection; a ``Group`` lists its members; a leaf
       returns ``{}``.
   * - ``Group``
     - A part composed from other named parts, whether they form an arm, a
       torso, or an entire robot.
   * - ``Robot``
     - The outermost ``Group``. It has a registered type, builds from the
       hardware config, and owns connections created during placement.
   * - ``at()`` / ``PartSpec``
     - An inert declaration of a part class, its constructor arguments, and its
       target node.
   * - ``PartHandle``
     - A reference with the same call interface for a local part and a part
       hosted by a worker.
   * - ``MethodArm`` / ``MethodGripper`` / ``MethodCamera``
     - Views that present methods such as ``open_gripper`` and
       ``get_camera(id)`` through the part interface.

Composition, Not Robot Types
----------------------------

Now let's apply those abstractions without assuming a particular robot shape.
A robot has named parts rather than fixed slots for arms, cameras, and bases, so
a lift or head joins the composition under its own name:

.. code-block:: python

   one = FrankaRobot(arm=arm, gripper=gripper)
   two = FrankaRobot(left=Group(arm=l, gripper=lg), right=Group(arm=r, gripper=rg))
   lifted = FrankaRobot(left=..., right=..., lift=lift, head=head_camera)

The observation and action dictionaries follow the same tree. Each part name
adds one path segment, including names nested inside a ``Group``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Path
     - Meaning
   * - ``<name>``
     - A part of the robot, by the name you composed it under.
   * - ``<group>.<name>``
     - A part of a group, nested as deeply as the composition goes.

If the arms have independent connections, ``Robot`` resets, reads, and commands
them concurrently. The policy still reads the same nested dictionary; it does
not coordinate the two requests.

Placement Is a Property of Parts
--------------------------------

Composition tells us what the robot contains; placement tells
:meth:`Robot.connect` where to create each part. Call ``at()`` with the target
node while declaring the part, then connect the assembled robot:

.. code-block:: python

   robot = FrankaRobot(
       left=FrankaROSArm.at("10.0.0.1", node_rank=1),
       scene=RealSenseCamera.at(info, node_rank=3),
   )
   robot.connect()

The calls to ``at()`` above only record declarations; they neither create a
worker nor open hardware. ``Robot.connect`` places each distinct declaration
once and publishes the resulting handle as ``robot.handles[<name>]``. If a later
part fails to start, it tears down the earlier ones. During a normal shutdown,
``disconnect`` follows the same cleanup path.

Shared hardware needs one more step. Declare its connection once, then refer to
the components exposed through ``parts``. In this example, one declaration
backs both arms and the cameras:

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

Cameras use the same placement path as arms. A camera may stay on the machine
where it is plugged in while the policy runs elsewhere; pass its node to
``declare_cameras({name: info}, node_rank=...)``. ``Robot.connect`` opens the
camera and ``Robot.disconnect`` closes it. The lower-level ``spawn()`` function
performs placement immediately and is intended for cases such as bench scripts,
where no ``Robot`` owns the lifecycle.

A placed part does not require a hand-written worker class. Placement synthesizes
one from the part class (``type(name, (Worker, PartCls), ...)``), and
``WorkerGroup`` binds the public methods as RPCs. A vendor-specific method that
falls outside the standard part interface remains callable through the handle,
with the same expression for a local or remote part::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

For the worker-to-node and worker-to-GPU rules, continue with
:doc:`Placement <placement>`.

Compose Every Part Kind
-----------------------

Placement does not change the composition tree. Start with the robot's top-level
parts, then let each connected part contribute its own components. When an arm
and gripper share a connection, adding the arm also adds its
``end_effector``:

.. code-block:: python

   robot = FrankaRobot(
       arm=FrankaROSArm.at(robot_ip, node_rank=1),
       wrist=RealSenseCamera.at(info, node_rank=3),
   )

   robot.part("arm").parts     # {"arm": ..., "end_effector": ...}

Use ``part(...)`` when one hardware declaration exposes several peer
components. For example, a coupled controller may drive two arms without being
an arm itself; the robot selects each arm from that declaration.

For robot classes, organize the builder by part kind. ``build`` combines the
``build_*`` mappings, while a variant with another arm count can replace only
``build_arms``:

.. code-block:: python

   class FrankaRobot(Robot):
       @classmethod
       def build_arms(cls, **config) -> dict[str, RobotPart]:
           return {"arm": cls.declare_arm(...)}

       @classmethod
       def build(cls, **config) -> "FrankaRobot":
           return cls(**cls.build_arms(**config), **cls.build_cameras(...))


   class DualFrankaRobot(FrankaRobot):
       @classmethod
       def build_arms(cls, **config) -> dict[str, RobotPart]:
           return {"left": ..., "right": ...}      # the only difference

You do not need a separate config class for each part. Declare cameras with
``declare_cameras({name: info}, node_rank=...)`` and arms with ``at(...)`` plus
their constructor arguments. Both read their fields from the robot's existing
``RobotConfig``, the same config used by hardware discovery.

Lifecycle
---------

At this point, the distinction between assembly and connection matters. A robot
moves through four phases, and no hardware is opened during the first one:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Step
     - What happens
   * - Declare
     - ``at()`` records a part class, its arguments, and its node. It does not
       build anything or touch hardware.
   * - Connect
     - ``Robot.connect`` builds each distinct declaration on its node, connects
       the parts, and publishes their handles as ``robot.handles[<name>]``.
   * - Use
     - ``get_observation`` and ``send_action`` run across independent
       connections in parallel.
   * - Disconnect
     - ``Robot.disconnect`` disconnects the parts before releasing their
       underlying connections.

After ``Robot.build`` returns, the robot still contains declarations rather than
live parts, and ``is_connected`` remains ``False``. Call ``connect`` before
reading observations or sending commands.

If one part fails during connection, ``Robot.connect`` tears down every part it
has already placed or connected and restores the declarations. After correcting
the hardware problem, you can call ``connect`` again. A normal ``disconnect``
returns the robot to that same state, ready for another connection attempt.


The Boundary
------------

There is one import boundary to preserve when implementing these classes. Part
modules must not import Ray, Gymnasium, or ``rlinf.scheduler``; otherwise, merely
loading a device driver also loads cluster dependencies. The bench scripts in
``toolkits/realworld_check`` rely on that separation when they run directly on a
hardware machine without a cluster.

``rlinf/robotics/placement/handles.py`` is the sole module allowed to cross it,
and ``spawn`` imports it lazily when placement is requested. Imports do not run
in the other direction: the scheduler never imports robotics.
``tests/unit_tests/test_robotics.py`` checks both rules.

Where the Code Lives
--------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Contents
   * - ``parts/base.py``
     - Core part types: ``RobotPart``, ``ControllablePart``, ``Camera``,
       ``EndEffector``, ``Group``, ``MobileBase``, ``LeggedBase``.
   * - ``parts/arms/``
     - Arm implementations and each hardware family's state dataclass.
   * - ``parts/cameras/``
     - RealSense, ZED, Lumos.
   * - ``parts/end_effectors/``
     - ``grippers/`` and ``hands/``.
   * - ``parts/transports/``
     - Shared transports such as ROS. A transport carries a part's messages but
       is not a part itself.
   * - ``robots/``
     - One module per robot, containing its config, discovery logic, and
       builder.
   * - ``parts/views.py``
     - The ``Method*`` views, which present a vendor SDK's methods as parts.
   * - ``placement/``
     - ``specs.py`` declares where a part runs; ``handles.py`` builds it there
       and is the only module that imports the scheduler.
   * - ``discovery/``
     - ``registry.py`` maps a robot type to its config, discovery, and builder;
       ``autoconfig.py`` fills that config from the environment.
   * - ``robot.py``, ``adapters.py``
     - Composition, and the adapters for policies that expect flat vectors.

An Operator Is Another Set of Parts
-----------------------------------

Operator hardware follows the same part model as robot hardware. A leader arm
reports encoder state, a glove reports finger angles, and a spacemouse reports a
twist. Each device connects, produces readings, disconnects, and lives on the
machine that owns its physical connection. RLinf models each one as a
``RobotPart`` and applies the usual lifecycle and placement:

.. code-block:: python

   leader = TeleopLeaderArm.at("/dev/ttyUSB0", node_rank=1)   # on the NUC
   mouse = SpaceMouse()                                       # here

The device reports a raw reading; a binding interprets that reading for a
particular robot. The binding declares the named action parts it ``PRODUCES`` and
returns values for those parts:

.. code-block:: python

   class SpaceMouseBinding(TeleopBinding):
       PRODUCES = ("arm", "end_effector")

       def action(self, reading, context):
           return {"arm": reading["twist"], "end_effector": self._grip(reading)}

The spacemouse part itself makes no assumption about a Cartesian arm. Pairing it
with another binding gives the same hardware a different action mapping.

Devices Compose the Way Parts Do
--------------------------------

A binding preserves part names in its output, and ``TeleopGroup`` merges those
outputs by name. In a dexterous-hand rig, a spacemouse contributes the arm
action while a glove contributes the hand action:

.. code-block:: yaml

   teleop: [spacemouse, glove]

.. code-block:: python

   parts, driving, _ = group.action(context)
   # {"arm": array(6), "hand": array(6)}

No action-vector slices are involved. If a binding offers a part that the robot
does not have, ``TeleopGroup`` leaves that part unfilled. On a robot with a hand
instead of a gripper, the spacemouse still drives the arm.

``TeleopGroup`` checks compatibility while it is built. Two devices cannot
claim the same part, and every device must match at least one part on the robot.

Devices in one rig may also coordinate through context. In the dexterous-hand
setup, the spacemouse binding publishes whether its second button is held. The
glove binding reads that flag and drives the hand only while the button remains
down:

.. code-block:: python

   def publish(self, reading):
       return {"hand_driving": bool(reading["buttons"][1])}

Repeated part types need one further distinction. Both leaders on a two-armed
robot produce an arm action, so ``drives`` selects the branch each one fills. No
other configuration field names a robot part:

.. code-block:: yaml

   teleop:
     - {gello_joint: {port: /dev/left,  drives: left}}
     - {gello_joint: {port: /dev/right, drives: right}}

Tasks Stay Out of Hardware
--------------------------

The last boundary separates hardware behavior from task semantics. A part knows
how to move and what it can sense; it does not decide whether an episode has
succeeded. Put reset behavior, reward, termination, and Gymnasium spaces in a
``RobotTask``, then pair the task with a ``Robot`` through ``RobotTaskEnv``. For
an existing policy that consumes flat vectors and ``state``/``frames``
observations, add ``LegacyObservationAdapter`` and ``VectorActionAdapter`` at
that boundary. The robot can then serve another task without changing its
device code.

Next
----

- :doc:`Adding a Robot <../extending/new_robot>`: follow the step-by-step guide.
- :doc:`Placement <placement>`: learn how workers map onto nodes and GPUs.
