Adding a Robot
==============

To add a physical robot, describe its devices as parts and compose them into a
``Robot``. The steps on this page take that class through registration and into
the cluster config, while keeping the vendor SDK out of task and placement code.

If the part model is new to you, start with
:doc:`Robotics Model <../concepts/robotics>`. We'll use its three main ideas
throughout this guide: a physical component is a ``RobotPart``; one connection
may expose several components through ``parts``; and a ``Robot`` assigns stable
names to the resulting tree. The concept page also maps the
``rlinf/robotics`` package and explains how ``spawn()`` places a part on another
node.

Implement a Part
----------------

Begin with one device and choose the narrowest interface that describes it.
Inherit ``RobotPart`` for an observation-only device, or ``ControllablePart``
when the device also accepts commands.

Every part answers the same three questions: ``_open`` reaches the hardware,
``get_observation`` reads it, and ``_release`` lets it go. Connecting and
disconnecting are handled for you, so a part is written by saying what its
hardware is. Keep the vendor SDK import inside ``_open``: other nodes can then
import the part module without installing the SDK, while the node that opens
the connection still loads it normally.

.. code-block:: python

   import numpy as np

   from rlinf.robotics import ControllablePart


   class ExampleArm(ControllablePart):
       def __init__(self, endpoint: str):
           self.endpoint = endpoint

       def _open(self):
           from example_robot_sdk import Client

           return Client(self.endpoint)

       def _release(self) -> None:
           self._device.close()

       @property
       def observation_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       @property
       def action_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       def reset(self) -> None:
           self._device.move_home()

       def get_observation(self) -> dict[str, np.ndarray]:
           return {"joint_position": self._device.get_joint_position()}

       def send_action(
           self, action: dict[str, np.ndarray]
       ) -> dict[str, np.ndarray]:
           if set(action) != {"joint_position"}:
               raise KeyError("Expected only 'joint_position'.")
           self._device.move_joints(action["joint_position"])
           return action

Whatever ``_open`` returns is available as ``self._device`` and decides
``is_connected``. Opening there rather than in ``__init__`` is what lets a part
be declared on one machine and built on another. A part whose lifecycle is more
than opening a device -- an arm that must home before it is usable -- may
override ``connect`` and ``disconnect`` instead.

When the device fits ``Camera``, ``EndEffector``, ``MobileBase``, or
``LeggedBase``, inherit that specific interface instead. Compositions and remote
proxies can then retain its device category.

Expose Several Components on One Connection
-------------------------------------------

Next, account for the connection boundary. A socket, CAN bus, or ROS node may
drive several physical components; list them all through ``parts`` so callers
can address each one even though the device connection opens only once. For an
arm, the convention is to expose the part itself under ``"arm"``.

.. code-block:: python

   from rlinf.robotics import MethodGripper, RobotPart


   class ExampleArm(ControllablePart):
       ...

       @property
       def parts(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(self, state_field="gripper_position"),
           }

Some SDKs expose those capabilities as methods such as ``open_gripper``,
``move_left_arm``, and ``get_camera(id)``. Wrap them with ``MethodGripper``,
``MethodArm``, or ``MethodCamera`` and define the view beside the adapted
methods. The robot composition then sees ordinary parts rather than vendor
method names.

Compose the Robot
-----------------

A robot is named parts and nothing else. Constructed directly, that is the whole
of it:

.. code-block:: python

   from rlinf.robotics import Robot


   class Bench(Robot):
       ROBOT_TYPE = "Bench"


   robot = Bench(arm=ExampleArm.at("10.0.0.2", node_rank=1))
   robot.connect()

There is no hardware config and no discovery in that. Those exist so a robot can
be composed from its type name alone, which a config file needs and a script
does not. The rest of this page adds them.


Once the individual devices are represented, compose the robot and choose each
part name as if it were a public API field. The names become canonical
observation and action paths; changing one later also changes the schema used by
policies and datasets.

.. code-block:: python

   from rlinf.robotics import Group, Robot


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"


   robot = ExampleRobot(
       left=ExampleArm("tcp://left-arm:5000"),
       right=ExampleArm("tcp://right-arm:5000"),
   )
   robot.connect()
   observation = robot.get_observation()
   robot.send_action(
       {
           "arms": {
               "left": {"arm": {"joint_position": left_target}},
               "right": {"arm": {"joint_position": right_target}},
           }
       }
   )

In this layout, the left arm's canonical observation path is
``arms.left.state.joint_position``, and its action path is ``left.arm``.
End-effector actions use ``arms.<name>.end_effector``; robot-level cameras use
``cameras.<name>``; other components use ``parts.<name>``. Keep these names
stable across the environment, policy, and dataset code.

``Robot`` resets, reads, and commands arms on independent connections in
parallel. A two-arm observation therefore waits for one round trip, and the
robot subclass does not contain its own concurrency code.

Place Parts on Nodes
--------------------

Now we can decide where each part runs. Call ``at()`` with a node, then place the
returned declaration wherever you would otherwise put a local part.
``Robot.connect`` builds it on that node as part of the normal connection
sequence.

.. code-block:: python

   from rlinf.robotics import Group, Robot

   robot = Robot(arm=ExampleArm.at("tcp://left-arm:5000", node_rank=0))
   robot.connect()

The call to ``at()`` records an ``ExampleArm`` declaration for node 0. No part is
built until ``connect`` runs; at that point, the handle appears under
``robot.handles[<name>]`` and remains there until ``disconnect`` tears it down.
Compose end effectors and cameras explicitly, because this example arm contains
only the parts supplied to it.

The declaration is not arm-specific. For example, a camera can remain on the
machine where it is physically connected::

   scene=RealSenseCamera.at(info, node_rank=2)

If one connection backs several components, declare that connection once and
select the exposed parts from it::

   connection = ExampleConnection.at(node_rank=0)
   Group(arm=connection.part("left"), gripper=connection.part("left_end_effector"))

Underneath this flow, ``spawn()`` performs placement immediately. Call it
directly only outside a robot, for example in a bench script that also takes
care of the handle's lifecycle.

There is no separate worker class to write for the part. Placement synthesizes
one from its class, and ``WorkerGroup`` binds public methods as RPCs. A method
outside the standard part interface remains available through the handle, using
the same expression locally and remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

Describe and Build the Robot
----------------------------

With the parts in place, describe the robot's hardware inputs in a
``RobotConfig`` dataclass and implement ``build()`` to assemble them. Connection
addresses and placement belong here. Reset poses, rewards, and episode horizons
remain in the task config because they describe an episode rather than hardware
discovery.

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import Group, RobotConfig


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str = ""
       right_endpoint: str = ""


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"

       @classmethod
       def build(cls, *, config: ExampleRobotConfig) -> "ExampleRobot":
           return cls(
               **{
                   side: ExampleArm.at(
                       endpoint,
                       node_rank=config.node_rank,
                       name=f"ExampleArm-{side}",
                   )
                   for side, endpoint in (
                       ("left", config.left_endpoint),
                       ("right", config.right_endpoint),
                   )
               }
           )

A single-arm variant can reuse this builder shape and return one entry instead
of two. Startup must remain all-or-nothing: if a later part fails, disconnect
the handles already placed before propagating the error. Otherwise, callers
could receive a partial schema whose contents depend on the failure order.

.. warning::

   ``build()`` only composes declarations; it does not touch hardware. Call
   ``connect()`` on the result before reading observations or sending commands,
   then call ``disconnect()`` during teardown. Environments make these calls in
   their hardware setup.

Register the Robot
------------------

After defining the config, builder, and discovery logic, register them from the
robot's own module. Registration happens locally rather than through an edit to
a central table.

.. code-block:: python

   from typing import Optional

   from rlinf.robotics import RobotDiscovery, RobotInfo
   from rlinf.scheduler.hardware import HardwareConfig, HardwareResource


   class ExampleRobotDiscovery(RobotDiscovery):
       HW_TYPE = ExampleRobot.ROBOT_TYPE

       @classmethod
       def enumerate(
           cls,
           node_rank: int,
           configs: Optional[list[HardwareConfig]] = None,
       ) -> Optional[HardwareResource]:
           matching = [
               config
               for config in configs or []
               if isinstance(config, ExampleRobotConfig)
               and config.node_rank == node_rank
           ]
           if not matching:
               return None
           return HardwareResource(
               type=cls.HW_TYPE,
               infos=[
                   RobotInfo(type=cls.HW_TYPE, model=cls.HW_TYPE, config=config)
                   for config in matching
               ],
           )


   ExampleRobot.register(ExampleRobotConfig, ExampleRobotDiscovery)

Put the registration call at the end of the module, after the config and
discovery classes exist. It associates the robot class with its config,
discovery logic, and ``build`` method. Callers can then use
``build_robot("ExampleRobot", ...)`` without importing the concrete class.

If the new hardware is a variant of an existing robot, subclass that robot
instead. For example, ``DualFrankaRobot`` changes ``build_arms`` and ``BACKEND``
on ``FrankaRobot`` but inherits ``build`` and the lifecycle methods.

Import the registration module before constructing ``Cluster``; hardware
discovery cannot identify the robot until that import has run. The registered
hardware policy modules are also passed to node probes, so every node's
configured Python environment must be able to import this one.

Configure the Cluster
---------------------

The final construction input comes from the existing
``cluster.node_groups.hardware`` schema. Its entries are parsed by the
registered config class and passed to the registered builder. Put endpoints and
node ranks in this YAML rather than embedding a particular deployment in
Python:

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement: {}
     node_groups:
       - label: example_robot
         node_ranks: 0
         hardware:
           type: ExampleRobot
           configs:
             - node_rank: 0
               left_endpoint: tcp://left-arm:5000
               right_endpoint: tcp://right-arm:5000

Keep Tasks and Compatibility Separate
-------------------------------------

At this stage, the robot knows how to find and operate its hardware, but it
should not define the task. Put reset behavior, reward, success, truncation, and
Gymnasium spaces in a ``RobotTask`` or the real-world environment, then combine
the task and ``Robot`` through ``RobotTaskEnv``. When an existing policy expects
flat action vectors and ``state``/``frames`` observations, use
``LegacyObservationAdapter`` and ``VectorActionAdapter`` at that boundary.

.. warning::

   When introducing the canonical interface, keep every existing Gym ID, action
   dimension, observation key, camera name, and dataset field unchanged. Use an
   adapter and a regression test to preserve compatibility for trained policies
   and existing datasets.

Test the Integration
--------------------

Most of the integration is testable before the vendor SDK or robot is
available. Cover the part contract, composition paths, handle lifecycle,
discovery registration, and the schema policies expect:

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py tests/unit_tests/test_real_env.py

These exercise the scheduler import boundary, single-arm and dual-arm
composition, the task and robot split, and the policy-facing schema of every
built-in real-world environment. None of it requires physical hardware.

Run It Against Faked SDKs
~~~~~~~~~~~~~~~~~~~~~~~~~

A part imports its vendor SDK when it opens, never at import time, so a fake in
``sys.modules`` is enough to run the real part classes with nothing on the
other end of the cable. ``tests/robot_mocks`` holds one fake per SDK.

Walk your robot's composition against them:

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MyRobot --mock \
       --arg robot_ip=10.0.0.1 --arg node_rank=0

It reports what the robot is made of, which connection backs each part and
where it was placed, then reads every part and disconnects. It fails when a
part observes something it never declared, when a connection ends up in the
tree, or when anything still claims to be connected afterwards.

Add ``--remote`` to host the parts in scheduler workers instead of this
process. That is what catches a part that cannot be placed at all -- a method
whose name collides with the worker's own, or state that does not survive the
process boundary.

A whole training run works the same way. ``run.sh`` installs the fakes when the
config name contains ``mock``:

.. code-block:: bash

   bash tests/e2e_tests/embodied/run.sh realworld_mock_sac_cnn

Every shipped robot has one, so the composition, the wrapper stack, the
observation space and the runner all run as they would on a bench:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Robot
     - Config
   * - Franka
     - ``realworld_mock_sac_cnn``
   * - Dual Franka
     - ``realworld_dual_franka_mock_sac_cnn``
   * - GimArm
     - ``gim_arm_mock_sac_cnn``
   * - Turtle2
     - ``realworld_xsquare_turtle2_mock_sac_cnn``
   * - DOSW1
     - ``dosw1_mock_sac_mlp_pick``

Run It Against the Robot
~~~~~~~~~~~~~~~~~~~~~~~~

What is left needs the hardware: timing, calibration, and whatever the device
does that its documentation does not. Once it is powered and reachable, drop
``--mock`` and the same check runs against it:

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MyRobot \
       --arg robot_ip=10.0.0.1 --arg node_rank=1
