Adding a Robot
==============

Add a physical robot without coupling its device SDK, its placement on the
cluster, and its task logic. You implement parts, compose them into a ``Robot``,
register it, and point a cluster config at it. Everything else — hosting parts on
the right machines, reading them in parallel, exposing them to a policy — you get
from the layer.

Before you start, read :doc:`Robotics Model <../concepts/robotics>` for the
design this guide applies: every physical component is a ``RobotPart``, hardware
that drives several components declares them with ``subparts()``, a ``Robot`` is
a named composition, and any part can be placed on a node with ``spawn()``. That
page also maps the ``rlinf/robotics`` package.

Implement a Part
----------------

Inherit ``RobotPart`` for observation-only devices and ``ControllablePart`` for
anything that accepts commands. Import the vendor SDK inside ``connect()`` so
every node can import the module without installing that SDK.

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

Use ``Camera``, ``EndEffector``, ``MobileBase``, or ``LeggedBase`` when a more
specific interface applies.

Expose Several Components on One Connection
-------------------------------------------

When one socket, CAN bus, or ROS node drives more than one component, declare
them with ``subparts()``. The part itself is conventionally the ``"arm"`` entry.

.. code-block:: python

   from rlinf.robotics import MethodGripper, RobotPart


   class ExampleArm(ControllablePart):
       ...

       def subparts(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(self, state_field="gripper_position"),
           }

``MethodGripper``, ``MethodArm``, and ``MethodCamera`` adapt hardware that speaks
in named methods (``open_gripper``, ``move_left_arm``, ``get_camera(id)``) into
parts, so composition sees one uniform interface. Declare them in Python next to
the methods they wrap.

Compose the Robot
-----------------

Put every manipulator inside an ``Arm``. Arm names become canonical observation
and action paths, so keep them stable.

.. code-block:: python

   from rlinf.robotics import Arm, Robot


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"


   robot = ExampleRobot.dual_arm(
       left_arm=Arm(ExampleArm("tcp://left-arm:5000")),
       right_arm=Arm(ExampleArm("tcp://right-arm:5000")),
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

The canonical observation path for the left manipulator is
``arms.left.state.joint_position``; its action path is ``arms.left.arm``.
End-effector actions use ``arms.<name>.end_effector``. Robot-level cameras use
``cameras.<name>``, and extra components use ``parts.<name>``.

``Robot`` resets, reads, and commands independent arms in parallel, so a two-arm
observation costs one round trip rather than two.

Place Parts on Nodes
--------------------

``RobotPart.spawn`` is the only placement call, and every part has it. Without
``node_rank`` the part is built in this process; with one it is hosted in a
scheduler worker on that node. Both return a handle with the same API, so callers
never branch on placement. This is not limited to arms — a camera can run on the
machine it is plugged into while the policy runs elsewhere.

.. code-block:: python

   from rlinf.robotics import Arm, Robot

   handle = ExampleArm.spawn(
       "tcp://left-arm:5000",
       node_rank=0,
       name="ExampleArm-0",
   )
   robot = Robot.single_arm(
       Arm(handle.subpart("arm"), handle.subpart("end_effector")),
       handles={"arm": handle},
   )
   robot.connect()

What this does: 1) constructs ``ExampleArm`` on node 0 and connects it,
2) returns proxies for its subparts, 3) composes them into a robot that owns the
handle. Pass owned handles as ``handles=``; ``Robot.disconnect`` releases them
after every part is disconnected.

There is no per-robot worker class to write. RLinf synthesizes one from the part
class, so ``WorkerGroup`` binds every public method as an RPC. Methods outside the
part interface stay reachable through the handle, with the same call shape
locally and remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

.. warning::

   ``WorkerGroup`` reserves ``launch``, ``execute_on``, ``from_group_name``, and
   ``WorkerRank``. A part with a public method of one of those names cannot be
   hosted; rename it.

Describe and Build the Robot
----------------------------

Put connections and placement in a ``RobotConfig`` dataclass, and give it a
builder that turns those fields into a composed ``Robot``. Keep reset poses,
rewards, and episode horizons in the task config instead.

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import Arm, RobotConfig


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str = ""
       right_endpoint: str = ""


   def build_example_robot(config: ExampleRobotConfig) -> ExampleRobot:
       handles = {
           side: ExampleArm.spawn(
               endpoint,
               node_rank=config.node_rank,
               name=f"ExampleArm-{side}",
           )
           for side, endpoint in (
               ("left", config.left_endpoint),
               ("right", config.right_endpoint),
           )
       }
       return ExampleRobot.dual_arm(
           Arm(handles["left"].subpart("arm")),
           Arm(handles["right"].subpart("arm")),
           handles=handles,
       )

A single-arm variant is the same builder returning
``ExampleRobot.single_arm(...)``. If a later part fails to come up, disconnect
the handles already placed before letting the error propagate, so a partial robot
is never returned.

Register the Robot
------------------

Register the config, composition, discovery, and builder in one call from the
robot's own module. Nothing central needs editing.

.. code-block:: python

   from typing import Optional

   from rlinf.robotics import RobotDiscovery, RobotInfo, register_robot
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


   register_robot(
       ExampleRobotConfig, ExampleRobot, build=build_example_robot
   )(ExampleRobotDiscovery)

Place this call at the end of the module so it can name the builder. Once
registered, ``build_robot("ExampleRobot", ...)`` composes the robot by name,
without importing its builder directly.

Import the registration module before constructing ``Cluster``. RLinf propagates
registered hardware policy modules to node probes, so the module must be
importable in each node's configured Python environment.

Configure the Cluster
---------------------

Keep the existing ``cluster.node_groups.hardware`` schema. The registered config
class parses each item, and the registered builder composes the robot.

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

Implement reset, reward, success, truncation, and Gymnasium spaces in a
``RobotTask`` or the real-world environment. Use ``RobotTaskEnv`` to combine a
task with a ``Robot``. Use ``LegacyObservationAdapter`` and
``VectorActionAdapter`` when an existing policy expects flat action vectors and
``state``/``frames`` observations.

.. warning::

   Do not change an existing Gym ID, action dimension, observation key, camera
   name, or dataset field while introducing the canonical interface. Add an
   adapter and a regression test instead.

Test the Integration
--------------------

Test parts without vendor SDKs, composition paths, handle lifecycle, discovery
registration, and the exact legacy policy schema.

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py \
     tests/unit_tests/test_robotics_boundaries.py \
     tests/unit_tests/test_robot_task_env.py \
     tests/unit_tests/test_realworld_robotics_compatibility.py

What this does: it verifies the scheduler boundary, single-arm and dual-arm
composition, task/robot separation, and the policy-facing schema of every
built-in real-world environment, none of which requires physical hardware.
