Robotics Model
==============

Build a robot as a set of named parts, and RLinf can place those parts, read
independent connections in parallel, and clean everything up afterward. Start
with the Franka examples below to see the payoff, then use the rest of the page
to understand the model behind them.

Assemble a Robot
----------------

For each robot, describe the parts it carries. The shared robotics layer handles
their lifecycle and placement, so the robot definition stays focused on its
hardware:

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

Every ``build_*`` method answers the same question -- *which parts of this kind
does the robot carry, and under what names* -- by returning a
``{name: part}`` mapping. The inherited ``build`` is then just the merge, which
is why a new Franka variant never rewrites it:

.. code-block:: python

   @classmethod
   def build(cls, *, cameras=None, camera_node_rank=None, **config) -> "FrankaRobot":
       return cls(
           **cls.build_arms(**config),
           **cls.build_cameras(cameras, node_rank=camera_node_rank),
       )

Those names become the robot's parts. From there, ``connect`` builds each part
on the node it asked for and ``disconnect`` releases it. Both live in the common
layer, so no robot reimplements them.

Use a Robot
-----------

Driving a robot is five calls, and they are the same five whatever it is made
of. Build it from its registered type, connect once, then read and command it:

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

Observations and actions are nested dictionaries keyed by the names you composed
the robot with, so the data has the same shape as the hardware. Reading a
dual-arm robot uses the same call; only the names differ:

.. code-block:: python

   observation["left"]["arm"]["arm_joint_position"]
   observation["right"]["end_effector"]["state"]

Notice what the calling code does not mention: where anything runs. An arm
declared on another node reports its reading in the same place as a local one,
and parts on independent connections are read in parallel, so a two-arm
observation costs one round trip instead of two.

To learn that tree without touching hardware, read ``observation_features`` and
``action_features``. They describe the same nesting and are what ``RobotTaskEnv``
turns into Gymnasium spaces; see `Tasks Stay Out of Hardware`_ for how a policy
that expects flat vectors plugs in.

One Arm or Many Is the Same Code
--------------------------------

The dual-arm Franka does not need a second implementation of the robotics
machinery. It inherits the single-arm version and changes only the list returned
by ``build_arms``:

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

It inherits ``declare_arm``, ``build_cameras``, ``build``, placement, parallel
reads, and teardown. Adding a third arm would mean adding a third entry, not
inventing another control path. This keeps the common behavior in one place,
regardless of how many arms the hardware has.

Switching the Control Backend Is One Line
-----------------------------------------

Set ``BACKEND`` to choose the arm implementation created by a declaration.
Because every robot variant goes through ``declare_arm``, the same backend works
whether the robot has one arm or six:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``BACKEND``
     - Arm part it declares
   * - ``"franka_ros"``
     - ``FrankaROSArm`` -- Cartesian impedance control over ROS.
   * - ``"franky"``
     - ``FrankyArm`` -- joint and Cartesian control over libfranka.

To add a third backend, implement one arm part and add one entry to
``FRANKA_BACKENDS``. The robot classes do not change because they never name a
concrete arm class; that decision belongs to the backend mapping.

An Arm Arrives Whole
--------------------

The Franka gripper uses the arm's connection, so the arm owns it. Once you add
the arm to a robot, the gripper comes along as its ``end_effector`` part:

.. code-block:: python

   robot = FrankaRobot.build(robot_ip="10.0.0.1", node_rank=1, ...)

   robot.parts                  # {"arm": ...}
   robot.part("arm").parts      # {"arm": ..., "end_effector": ...}

Notice that ``build`` never mentions the gripper. The arm describes what its
connection exposes, while the robot only decides which top-level hardware it
carries. That split prevents connection details from leaking into every robot
definition.

Any Part Runs Anywhere
----------------------

Placement follows the device, not its category. Pass ``node_rank`` to any part:

.. code-block:: python

   robot = FrankaRobot(
       arm=FrankaROSArm.at(robot_ip, node_rank=1),      # on the arm's NUC
       wrist=RealSenseCamera.at(info, node_rank=3),     # where it is plugged in
   )

RLinf reads parts on independent connections in parallel. Parts that share a
connection keep their declared order, since vendor SDKs rarely allow concurrent
calls through the same link. You get concurrency where it is safe without
having to coordinate it in each robot class.

The Core Idea
-------------

**Treat every physical component as a part.** An arm, gripper, camera, or mobile
base is a ``RobotPart``. Every part knows how to connect and report an
observation; a controllable part also accepts an action. There is no separate
"driver" layer because the part already represents the physical capability the
rest of the system needs.

Hardware connections do not always map one-to-one to physical components. A
coupled dual-arm controller, for example, may drive two arms, two grippers, and
two wrist cameras through one ROS connection. In that case, let the connected
part declare everything it exposes instead of creating another abstraction for
"the thing that owns the connection":

.. code-block:: python

   @property
   def parts(self) -> dict[str, RobotPart]:
       return {
           "left": MethodArm(self, commands={"tcp_pose": "move_left_arm"}),
           "right": MethodArm(self, commands={"tcp_pose": "move_right_arm"}),
           "left_end_effector": MethodGripper(self, state_field="follow1_pos"),
       }

Connection ownership is a detail of some parts, not a new category in the type
system.

**Compose a robot from named parts.** **Place any part on a node.** Together with
the first rule, these choices keep composition, hardware access, and placement
independent without adding more layers.

The Abstractions
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Abstraction
     - What it is
   * - ``RobotPart``
     - The base for any physical component. It defines ``connect``,
       ``get_observation``, ``disconnect``, and ``reset``;
       ``observation_features`` describes the data it returns.
   * - ``ControllablePart``
     - A part that accepts commands through ``send_action`` and describes those
       commands with ``action_features``.
   * - ``Camera`` / ``EndEffector`` / ``MobileBase`` / ``LeggedBase``
     - More specific part types, used when composition or a remote proxy needs
       to distinguish the device category.
   * - ``parts``
     - The named parts exposed by a part. Hardware uses it to describe everything
       driven by one connection, a ``Group`` returns what you composed into it,
       and a leaf returns ``{}``.
   * - ``Group``
     - A part composed of other named parts. The same construct can represent an
       arm, a torso, or an entire robot.
   * - ``Robot``
     - The outermost group. It knows its registered type, builds itself from the
       hardware config, and owns the connections created during placement.
   * - ``at()`` / ``PartSpec``
     - An inert declaration containing a part class, its arguments, and the node
       where it should be built.
   * - ``PartHandle``
     - A reference that keeps the same interface whether the part runs locally
       or inside a worker.
   * - ``MethodArm`` / ``MethodGripper`` / ``MethodCamera``
     - Views that expose methods such as ``open_gripper`` and ``get_camera(id)``
       through the part interface.

Composition, Not Robot Types
----------------------------

A robot is a collection of named parts rather than a fixed set of arm, camera,
or base slots. That is why an unusual device such as a lift or a head does not
need a new framework concept; you give it a name and compose it like any other
part:

.. code-block:: python

   one = FrankaRobot(arm=arm, gripper=gripper)
   two = FrankaRobot(left=Group(arm=l, gripper=lg), right=Group(arm=r, gripper=rg))
   lifted = FrankaRobot(left=..., right=..., lift=lift, head=head_camera)

Observations and actions follow the same tree as the hardware. Each name becomes
a path segment, no matter how deeply you nest the groups:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Path
     - Meaning
   * - ``<name>``
     - A part of the robot, by the name you composed it under.
   * - ``<group>.<name>``
     - A part of a group, nested as deeply as the composition goes.

When arms use independent connections, ``Robot`` can reset, read, and command
them in parallel. A two-arm observation therefore takes one round trip instead
of two, without exposing that scheduling detail to the policy.

Placement Is a Property of Parts
--------------------------------

Use ``at()`` to record where a part should run. You do not call a separate
placement function; :meth:`Robot.connect` finds each declaration and builds it
on the requested node.

.. code-block:: python

   robot = FrankaRobot(
       left=FrankaROSArm.at("10.0.0.1", node_rank=1),
       scene=RealSenseCamera.at(info, node_rank=3),
   )
   robot.connect()

A declaration is intentionally inert, so assembling a robot has no hardware
side effects. When you call ``connect``, RLinf places each distinct declaration
once and publishes its handle as ``robot.handles[<name>]``. If a later part
fails, it tears down anything already placed; ``disconnect`` performs the same
cleanup during a normal shutdown.

For shared hardware, declare the connection once and refer to the parts it
exposes. A connection backing two arms and two cameras is then opened once
rather than four times:

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

Placement applies to every part, not just arms. For example, the robot can keep a
camera on the machine where it is physically plugged in while the policy runs
elsewhere. Assign its node with
``declare_cameras({name: info}, node_rank=...)``; the robot opens the camera on
``connect`` and closes it on ``disconnect``. Underneath this flow, ``spawn()``
does the eager placement. Use it directly only when there is no robot managing
the lifecycle, such as in a bench script.

You do not need a worker class for every hardware device. RLinf synthesizes one
from the part class (``type(name, (Worker, PartCls), ...)``), and ``WorkerGroup``
binds its public methods as RPCs. If you need a method outside the standard part
interface, call it through the handle; the call has the same shape locally and
remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

Read :doc:`Placement <placement>` to map workers onto nodes and GPUs.

Compose Every Part Kind
-----------------------

Think of a robot as a tree. You compose its top-level parts, and each of those
parts contributes the components exposed by its own connection. An arm with a
gripper therefore arrives as a complete unit; the robot never has to name that
gripper separately:

.. code-block:: python

   robot = FrankaRobot(
       arm=FrankaROSArm.at(robot_ip, node_rank=1),
       wrist=RealSenseCamera.at(info, node_rank=3),
   )

   robot.part("arm").parts     # {"arm": ..., "end_effector": ...}

Reach into a declaration only when one hardware connection exposes several
peer components. A coupled controller that drives two arms is not itself an
arm, so select the component you need with ``part(...)``.

Organize the builder by part kind. ``build`` combines the results of the
``build_*`` methods, which lets a robot with a different arm count override
``build_arms`` while inheriting the rest of the construction logic:

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

This design does not require a config class for every part. Declare a camera with
``declare_cameras({name: info}, node_rank=...)`` and an arm with ``at(...)`` plus
its constructor arguments. The fields come from the robot's existing
``RobotConfig``, which already defines the hardware YAML used during discovery.

Lifecycle
---------

The lifecycle has four explicit phases. Keeping declaration separate from
connection makes robot construction predictable and gives failures a clean
rollback point.

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

Building a robot only assembles declarations. After ``Robot.build`` returns,
call ``connect`` before reading observations or sending commands. Until then,
``is_connected`` remains ``False`` and the robot's slots still contain the
declarations rather than live parts.

If one part fails during connection, RLinf tears down everything it has already
placed or connected and restores the declarations. You can correct the problem
and call ``connect`` again. A normal ``disconnect`` also restores that state, so
the same robot object can connect, disconnect, and reconnect safely.


The Boundary
------------

Keep Ray, Gymnasium, and ``rlinf.scheduler`` out of part implementations.
Importing a device driver should not pull the scheduler into the process. This
boundary is what lets the bench scripts in ``toolkits/realworld_check`` talk to
hardware on a machine that is not running a cluster.

Only ``rlinf/robotics/placement.py`` crosses the boundary, and ``spawn`` imports
it lazily when placement is actually needed. The scheduler never imports
robotics. ``tests/unit_tests/test_robotics_boundaries.py`` checks both sides of
this contract.

Where the Code Lives
--------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Contents
   * - ``parts/base.py``
     - The core part types: ``RobotPart``, ``ControllablePart``, ``Camera``,
       ``EndEffector``, ``Group``, ``MobileBase``, ``LeggedBase``.
   * - ``parts/arms/``
     - Arm implementations and the state dataclass for each hardware family.
   * - ``parts/cameras/``
     - RealSense, ZED, Lumos.
   * - ``parts/end_effectors/``
     - ``grippers/`` and ``hands/``.
   * - ``parts/teleop/``
     - Leader arms and input devices such as GELLO, glove, keyboard, Pico, and
       spacemouse.
   * - ``parts/transports/``
     - Shared transports such as ROS. A transport carries messages for a part
       but is not itself a part.
   * - ``robots/``
     - One module per robot, with its config, discovery logic, and builder.
   * - ``specs.py``
     - ``PartSpec`` and ``SubpartRef``: the declaration for a part and a
       reference to one of the components it exposes.
   * - ``placement.py``
     - ``PartHandle`` and the synthesized worker. This is the only module that
       imports the scheduler.
   * - ``views.py``
     - The ``Method*`` views.
   * - ``robot.py``, ``discovery.py``, ``adapters.py``, ``config.py``
     - Robot composition, registration, legacy policy adapters, and environment
       variable configuration.

Tasks Stay Out of Hardware
--------------------------

A part should know how to move and what it can sense, but not what the task calls
success. Put reset behavior, reward, termination, and Gymnasium spaces in a
``RobotTask``, then combine it with a ``Robot`` through ``RobotTaskEnv``. If an
existing policy expects flat vectors and ``state``/``frames`` observations, use
``LegacyObservationAdapter`` and ``VectorActionAdapter`` at that boundary. This
keeps hardware code independent of the policy schema and lets you reuse the same
robot across tasks.

Next
----

- :doc:`Adding a Robot <../extending/new_robot>`: follow the step-by-step guide.
- :doc:`Placement <placement>`: learn how workers map onto nodes and GPUs.
