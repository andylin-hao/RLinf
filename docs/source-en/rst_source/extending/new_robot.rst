Adding a Robot
==============

Add a physical robot without tying its device SDK to cluster placement or task
logic. You will model the hardware as parts, compose those parts into a
``Robot``, register the result, and reference it from the cluster config. Once
that boundary is in place, RLinf can host each part near its device, read
independent connections in parallel, and present one coherent interface to the
policy.

Read :doc:`Robotics Model <../concepts/robotics>` first for the reasoning behind
this design. In short, every physical component is a ``RobotPart``; hardware
that drives several components lists them through ``parts``. A ``Robot`` gives
those parts stable names, while ``spawn()`` can place any of them on another
node. That page also maps the ``rlinf/robotics`` package when you need to find
the implementation.

Implement a Part
----------------

Choose the narrowest part interface that matches the device. Inherit
``RobotPart`` when it only produces observations, or ``ControllablePart`` when
it also accepts commands. Import the vendor SDK inside ``connect()`` so the
module remains importable on nodes that do not have that SDK installed. Only the
node that opens the hardware connection needs the dependency.

.. code-block:: python

   import numpy as np

   from rlinf.robotics import ControllablePart


   class ExampleArm(ControllablePart):
       def __init__(self, endpoint: str):
           self.endpoint = endpoint
           self._client = None

       @property
       def is_connected(self) -> bool:
           return self._client is not None

       @property
       def observation_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       @property
       def action_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       def connect(self) -> None:
           from example_robot_sdk import Client

           self._client = Client(self.endpoint)

       def reset(self) -> None:
           self._client.move_home()

       def get_observation(self) -> dict[str, np.ndarray]:
           return {"joint_position": self._client.get_joint_position()}

       def send_action(
           self, action: dict[str, np.ndarray]
       ) -> dict[str, np.ndarray]:
           if set(action) != {"joint_position"}:
               raise KeyError("Expected only 'joint_position'.")
           self._client.move_joints(action["joint_position"])
           return action

       def disconnect(self) -> None:
           if self._client is not None:
               self._client.close()
               self._client = None

If the device already fits ``Camera``, ``EndEffector``, ``MobileBase``, or
``LeggedBase``, inherit that more specific interface. Doing so preserves the
device category when the part is composed or accessed remotely.

Expose Several Components on One Connection
-------------------------------------------

When one socket, CAN bus, or ROS node drives several physical components,
describe all of them through ``parts``. The connection is opened once, while
the rest of the system can still address each capability separately. By
convention, expose the part itself under the ``"arm"`` entry.

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

Use ``MethodGripper``, ``MethodArm``, and ``MethodCamera`` when an SDK exposes
capabilities as named methods such as ``open_gripper``, ``move_left_arm``, and
``get_camera(id)``. These views turn the methods into ordinary parts, which
keeps the composition layer independent of the vendor API. Define each view in
Python beside the methods it adapts, where that mapping is easiest to maintain.

Compose the Robot
-----------------

Choose part names as carefully as you would choose public API fields. They
become the canonical observation and action paths, so keep them stable once
policies and datasets depend on them.

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

In this layout, ``arms.left.state.joint_position`` is the canonical observation
path for the left arm, and ``left.arm`` is its action path. End-effector actions
use ``arms.<name>.end_effector``. Robot-level cameras appear under
``cameras.<name>``, while additional components use ``parts.<name>``. Following
these conventions keeps the policy-facing schema consistent across robots.

Because ``Robot`` knows which arms use independent connections, it can reset,
read, and command them in parallel. Reading a two-arm observation therefore
takes one round trip rather than two, without putting concurrency code in the
robot implementation.

Place Parts on Nodes
--------------------

Use ``at()`` to declare the node for a part, then compose that declaration in
the same place you would put a local part. When ``Robot.connect`` runs, it builds
the part on that node. There is no separate placement call to coordinate.

.. code-block:: python

   from rlinf.robotics import Group, Robot

   robot = Robot(arm=ExampleArm.at("tcp://left-arm:5000", node_rank=0))
   robot.connect()

What this does: 1) records an ``ExampleArm`` declaration for node 0; 2) builds
and connects that arm when the robot connects. Compose an end effector or
cameras explicitly, because an arm contains only the parts you give it. During
startup, ``connect`` publishes each handle as ``robot.handles[<name>]``;
``disconnect`` releases those handles during teardown.

The same declaration works for every part, not just arms. A camera, for example,
can stay on the machine where it is plugged in::

   scene=RealSenseCamera.at(info, node_rank=2)

If one connection backs several components, declare it once and refer to the
parts it exposes. This ensures that the underlying device is opened only once::

   hardware = ExampleHardware.at(node_rank=0)
   Group(arm=hardware.part("left"), gripper=hardware.part("left_end_effector"))

Underneath the declaration flow, ``spawn()`` performs placement immediately.
Use it directly only outside a robot, such as in a bench script where you also
manage the handle's lifecycle.

You do not need to write a worker class for each robot. RLinf synthesizes one
from the part class, and ``WorkerGroup`` binds its public methods as RPCs. Methods
outside the standard part interface remain available through the handle, with
the same call shape locally and remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

Describe and Build the Robot
----------------------------

Put hardware connections and placement in a ``RobotConfig`` dataclass, then
implement ``build()`` on the robot class to compose a ``Robot`` from those
fields. Keep reset poses, rewards, and episode horizons in the task config; they
describe how you use the robot, not how to find its hardware.

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

The same builder can cover a single-arm variant by returning one entry instead
of two. Treat startup as an all-or-nothing operation: if a later part fails,
disconnect any handles already placed before propagating the error. Returning a
partial robot would make the policy-facing schema depend on startup order.

.. warning::

   ``build()`` only composes declarations; it does not touch hardware. Call
   ``connect()`` on the result before reading observations or sending commands,
   then call ``disconnect()`` during teardown. Environments make these calls in
   their hardware setup.

Register the Robot
------------------

Register the robot from its own module once the config, discovery logic, and
builder are defined. Keeping this information together means a new robot does
not require edits to a central registry.

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

Place the call at the end of the module, after the config and discovery classes
exist. Registration ties the robot class to its config, discovery logic, and
``build`` method. You can then call ``build_robot("ExampleRobot", ...)`` to
compose the robot by name instead of importing its class at every call site.

Subclass an existing robot when the hardware shares most of its construction.
``DualFrankaRobot`` extends ``FrankaRobot`` and only changes ``build_arms`` and
``BACKEND``; it inherits ``build`` along with the rest of the lifecycle.

Import the registration module before constructing ``Cluster`` so the robot is
known when hardware discovery begins. RLinf passes registered hardware policy
modules to the node probes, so every node's configured Python environment must
be able to import the module.

Configure the Cluster
---------------------

Describe the hardware with the existing ``cluster.node_groups.hardware``
schema. The registered config class parses each entry, and the registered
builder turns it into the robot composition. This keeps deployment details in
YAML instead of hard-coding endpoints or node ranks in Python.

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

Put reset behavior, reward, success, truncation, and Gymnasium spaces in a
``RobotTask`` or the real-world environment. Combine that task with a ``Robot``
through ``RobotTaskEnv``. If an existing policy expects flat action vectors and
``state``/``frames`` observations, adapt the boundary with
``LegacyObservationAdapter`` and ``VectorActionAdapter`` rather than teaching
the hardware about a particular policy schema.

.. warning::

   When introducing the canonical interface, keep every existing Gym ID, action
   dimension, observation key, camera name, and dataset field unchanged. Use an
   adapter and a regression test to preserve compatibility for trained policies
   and existing datasets.

Test the Integration
--------------------

Most integration behavior can be tested without a vendor SDK or physical
hardware. Cover the part contract, composition paths, handle lifecycle,
discovery registration, and the exact schema expected by legacy policies.

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py \
     tests/unit_tests/test_robotics_boundaries.py \
     tests/unit_tests/test_robot_task_env.py \
     tests/unit_tests/test_realworld_robotics_compatibility.py

What this does: verifies the scheduler boundary, single-arm and dual-arm
composition, separation between tasks and robots, and the policy-facing schema
of every built-in real-world environment. None of these tests requires physical
hardware.
