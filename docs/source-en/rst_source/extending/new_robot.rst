Adding a Robot
==============

Add a physical robot while keeping its device SDK, cluster placement, and task
logic separate. Implement its parts, compose them into a ``Robot``, register it,
and point a cluster config at it. RLinf then hosts the parts on the correct
machines, reads them in parallel, and exposes them to a policy.

Read :doc:`Robotics Model <../concepts/robotics>` before you start. It explains
the design used here. Every physical component is a ``RobotPart``. Hardware that
drives several components declares them with ``subparts()``. A ``Robot`` is a
named composition, and ``spawn()`` places any part on a node. The page also maps
the ``rlinf/robotics`` package.

Implement a Part
----------------

Inherit ``RobotPart`` for an observation-only device. Inherit
``ControllablePart`` for a device that accepts commands. Import the vendor SDK
inside ``connect()``. This keeps the module importable on nodes without that SDK.

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

Use ``Camera``, ``EndEffector``, ``MobileBase``, or ``LeggedBase`` when the
device matches a more specific interface.

Expose Several Components on One Connection
-------------------------------------------

Declare components with ``subparts()`` when one socket, CAN bus, or ROS node
drives more than one of them. By convention, use the part itself as the ``"arm"``
entry.

.. code-block:: python

   from rlinf.robotics import MethodGripper, RobotPart


   class ExampleArm(ControllablePart):
       ...

       def subparts(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(self, state_field="gripper_position"),
           }

Use ``MethodGripper``, ``MethodArm``, and ``MethodCamera`` to adapt hardware with
named methods such as ``open_gripper``, ``move_left_arm``, and
``get_camera(id)``. They expose those methods as parts, so composition sees one
uniform interface. Declare the views in Python next to the methods they wrap.

Compose the Robot
-----------------

Wrap every manipulator in an ``Arm``. Keep arm names stable because they become
canonical observation and action paths.

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

Use ``arms.left.state.joint_position`` as the canonical observation path for the
left manipulator. Its action path is ``arms.left.arm``. Use
``arms.<name>.end_effector`` for end-effector actions. Use ``cameras.<name>`` for
robot-level cameras and ``parts.<name>`` for extra components.

Let ``Robot`` reset, read, and command independent arms in parallel. A two-arm
observation then costs one round trip rather than two.

Place Parts on Nodes
--------------------

Declare where a part runs with ``at()``. Compose the declaration exactly where
you would compose a part. ``Robot.connect`` builds it on its node; you never
call a placement function.

.. code-block:: python

   from rlinf.robotics import Arm, Robot

   robot = Robot.single_arm(Arm(ExampleArm.at("tcp://left-arm:5000", node_rank=0)))
   robot.connect()

What this does: 1) declares ``ExampleArm`` for node 0, 2) builds and connects it
when the robot connects. Compose an end effector or cameras explicitly; an
arm takes only what you give it. ``connect`` publishes each handle as
``robot.handles[<name>]`` and ``disconnect`` releases them.

Declaring works for every part, not only arms. A camera can run on the machine
it is plugged into::

   cameras={"scene": RealSenseCamera.at(info, node_rank=2)}

When one connection backs several components, declare it once and refer to its
subparts, so it is opened once::

   hardware = ExampleHardware.at(node_rank=0)
   Arm(hardware.subpart("left"), hardware.subpart("left_end_effector"))

``spawn()`` is the eager form underneath. Use it only outside a robot, such as in
a bench script, where you manage the handle yourself.

There is no per-robot worker class to write. RLinf synthesizes one from the part
class, so ``WorkerGroup`` binds every public method as an RPC. Methods outside the
part interface stay reachable through the handle, with the same call shape
locally and remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

Describe and Build the Robot
----------------------------

Store connections and placement in a ``RobotConfig`` dataclass. Implement
``build()`` on your robot class to compose a ``Robot`` from those fields. Keep
reset poses, rewards, and episode horizons in the task config.

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import Arm, RobotConfig


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str = ""
       right_endpoint: str = ""


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"

       @classmethod
       def build(cls, *, config: ExampleRobotConfig) -> "ExampleRobot":
           arms = {
               side: Arm(
                   ExampleArm.at(
                       endpoint,
                       node_rank=config.node_rank,
                       name=f"ExampleArm-{side}",
                   )
               )
               for side, endpoint in (
                   ("left", config.left_endpoint),
                   ("right", config.right_endpoint),
               )
           }
           return cls(arms=arms)

Build a single-arm variant with the same builder, but return
``ExampleRobot.single_arm(...)``. If a later part fails to start, disconnect the
handles that are already placed before propagating the error. Never return a
partial robot.

.. warning::

   ``build()`` composes declarations; it does not connect. Call ``connect()`` on
   the result before reading or commanding anything, and ``disconnect()`` when
   you are done. Environments do this in their hardware setup.

Register the Robot
------------------

Register the config, composition, discovery, and builder in one call from the
robot's module. Do not edit a central registry.

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

Place this call at the end of the module, once the config and discovery classes
exist. It registers the class, its config, its discovery, and its ``build``
together. After registration, call ``build_robot("ExampleRobot", ...)`` to
compose the robot by name without importing the class directly.

Subclass a robot to reuse its construction. ``DualFrankaRobot`` extends
``FrankaRobot``, inherits ``compose_arms`` unchanged, and overrides only
``BACKEND`` and ``build``.

Import the registration module before you construct ``Cluster``. RLinf
propagates registered hardware policy modules to node probes. Make the module
importable in the configured Python environment on every node.

Configure the Cluster
---------------------

Use the existing ``cluster.node_groups.hardware`` schema. The registered config
class parses each item. The registered builder then composes the robot.

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
``RobotTask`` or the real-world environment. Combine a task with a ``Robot``
through ``RobotTaskEnv``. Use ``LegacyObservationAdapter`` and
``VectorActionAdapter`` when an existing policy expects flat action vectors and
``state``/``frames`` observations.

.. warning::

   Keep every existing Gym ID, action dimension, observation key, camera name,
   and dataset field unchanged when you introduce the canonical interface. Add
   an adapter and a regression test instead.

Test the Integration
--------------------

Test parts without vendor SDKs. Also test composition paths, handle lifecycle,
discovery registration, and the exact legacy policy schema.

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py \
     tests/unit_tests/test_robotics_boundaries.py \
     tests/unit_tests/test_robot_task_env.py \
     tests/unit_tests/test_realworld_robotics_compatibility.py

What this does: verifies the scheduler boundary, single-arm and dual-arm
composition, task and robot separation, and the policy-facing schema of every
built-in real-world environment. These tests do not require physical hardware.
