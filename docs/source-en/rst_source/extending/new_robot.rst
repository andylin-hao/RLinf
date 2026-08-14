Adding a Robot
==============

Add a physical robot without coupling its device SDK, scheduler placement, and
task logic. Implement pure parts, describe the physical layout, compose a
``Robot``, and adapt its canonical interface to the existing policy schema.

Architecture
------------

Keep each layer focused on one responsibility.

.. list-table::
   :header-rows: 1

   * - Layer
     - Responsibility
   * - ``RobotPart`` and ``ControllablePart``
     - Own one device connection and expose canonical observations and actions.
   * - ``Arm``
     - Compose one arm driver, an optional ``EndEffector``, and named wrist cameras.
   * - ``Robot``
     - Compose named arms, robot-level cameras, and optional parts such as a base.
   * - ``RobotSpec``
     - Describe physical arms, cameras, end effectors, connections, and node ranks.
   * - ``RobotDiscovery``
     - Translate scheduler hardware configuration into generic hardware resources.
   * - ``RobotRuntime``
     - Coordinate local and remote parts, including parallel multi-arm operations.
   * - ``RobotTask`` and ``RobotTaskEnv``
     - Own reset, reward, termination, Gymnasium spaces, and policy compatibility.

The dependency direction is strict: drivers do not import Ray, Gymnasium, or
``rlinf.scheduler``. Runtime hosts may import scheduler APIs. Environments consume
the composed runtime and keep task semantics out of device drivers.

Implement a Pure Driver
-----------------------

Inherit from ``RobotPart`` for observation-only devices. Inherit from
``ControllablePart`` for arms, bases, and other devices that accept commands.
Import an optional vendor SDK inside ``connect()`` so every node can import the
module without installing that SDK.

.. code-block:: python

   import numpy as np

   from rlinf.robotics import ControllablePart


   class ExampleArmDriver(ControllablePart):
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
specific interface applies. Built-in implementations live under
``rlinf/robotics/cameras``, ``drivers``, ``end_effectors``, ``grippers``, and
``hands``.

Compose the Robot
-----------------

Put every arm driver inside an ``Arm``. Use stable arm names because they become
canonical observation and action paths. Single-arm and dual-arm robots use the
same structure.

.. code-block:: python

   from rlinf.robotics import Arm, Robot, RobotRuntime


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"


   robot = ExampleRobot.dual_arm(
       left_arm=Arm(ExampleArmDriver("tcp://left-arm:5000")),
       right_arm=Arm(ExampleArmDriver("tcp://right-arm:5000")),
   )
   runtime = RobotRuntime(robot)
   runtime.connect()
   observation = runtime.get_observation()
   runtime.send_action(
       {
           "arms": {
               "left": {
                   "arm": {"joint_position": left_target},
               },
               "right": {
                   "arm": {"joint_position": right_target},
               },
           }
       }
   )

The canonical observation path for the left driver is
``arms.left.state.joint_position``. End-effector actions use
``arms.<name>.end_effector``. Robot-level cameras use ``cameras.<name>``;
additional components use ``parts.<name>``.

Describe Physical Hardware
--------------------------

Make ``RobotConfig.to_spec()`` the translation boundary from an existing flat
YAML schema to the canonical physical layout. Put connections and placement in
``RobotSpec``. Keep reset poses, rewards, and episode horizons in the task config.

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import ArmSpec, RobotConfig, RobotSpec


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str
       right_endpoint: str

       def to_spec(self) -> RobotSpec:
           return RobotSpec(
               robot_type=ExampleRobot.ROBOT_TYPE,
               node_rank=self.node_rank,
               arms=(
                   ArmSpec(
                       name="left",
                       driver="example",
                       node_rank=self.node_rank,
                       connection={"endpoint": self.left_endpoint},
                   ),
                   ArmSpec(
                       name="right",
                       driver="example",
                       node_rank=self.node_rank,
                       connection={"endpoint": self.right_endpoint},
                   ),
               ),
           )

Use ``CameraSpec`` for robot-level or wrist cameras, ``EndEffectorSpec`` for a
tool attached to an arm, and ``PartSpec`` for a base, head, lift, or another
optional component.

Register Discovery
------------------

Keep scheduler discovery separate from ``Robot``. Implement
``RobotDiscovery.enumerate()`` and register the discovery class, physical config,
and composition class together.

.. code-block:: python

   from typing import Optional

   from rlinf.robotics import (
       RobotDiscovery,
       RobotInfo,
       register_robot,
   )
   from rlinf.scheduler.hardware import (
       HardwareConfig,
       HardwareResource,
   )


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
                   RobotInfo(
                       type=cls.HW_TYPE,
                       model=cls.HW_TYPE,
                       config=config,
                   )
                   for config in matching
               ],
           )


   register_robot(ExampleRobotConfig, ExampleRobot)(ExampleRobotDiscovery)

Import the registration module before constructing ``Cluster``. RLinf propagates
registered hardware policy modules to node probes, so the module must be
importable in each node's configured Python environment.

Configure Physical Hardware
---------------------------

Keep the existing ``cluster.node_groups.hardware`` schema. The registered config
class parses each item and ``to_spec()`` supplies the canonical layout.

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

Host a Driver Remotely
----------------------

Use ``ArmRuntime`` to construct and connect a pure arm driver in an RLinf
``Worker``. Wrap the one-worker group with ``RemoteControllablePart`` before
composing it into a local ``Robot``.

.. code-block:: python

   from rlinf.robotics import (
       Arm,
       ArmRuntime,
       RemoteControllablePart,
       Robot,
       RobotRuntime,
   )
   from rlinf.scheduler import NodePlacementStrategy

   group = ArmRuntime.create_group(
       ExampleArmDriver,
       {"endpoint": "tcp://left-arm:5000"},
   ).launch(
       cluster=cluster,
       placement_strategy=NodePlacementStrategy(node_ranks=[0]),
       name="ExampleArmRuntime",
   )
   remote_driver = RemoteControllablePart(group)
   runtime = RobotRuntime(Robot.single_arm(Arm(remote_driver)))
   runtime.connect()

``RobotRuntime`` applies independent arm resets, observations, and actions in
parallel. Built-in hardware uses the same boundary through
``launch_franka_runtime``, ``launch_dual_franka_runtime``,
``launch_gim_arm_runtime``, ``launch_turtle2_runtime``, and
``build_dosw1_runtime``.

Keep Tasks and Compatibility Separate
-------------------------------------

Implement reset, reward, success, truncation, and Gymnasium spaces in a
``RobotTask`` or the real-world environment. Use ``RobotTaskEnv`` to combine a
task with a ``RobotRuntime``. Use ``LegacyObservationAdapter`` and
``VectorActionAdapter`` when an existing policy expects flat action vectors and
``state``/``frames`` observations.

.. warning::

   Do not change an existing Gym ID, action dimension, observation key, camera
   name, or dataset field while introducing the canonical interface. Add an
   adapter and regression test instead.

Test the Integration
--------------------

Test pure drivers without vendor SDKs, composition paths, remote runtime
lifecycle, physical spec translation, discovery registration, and the exact
legacy policy schema.

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py \
     tests/unit_tests/test_robotics_boundaries.py \
     tests/unit_tests/test_robot_task_env.py \
     tests/unit_tests/test_realworld_robotics_compatibility.py

What this does: it verifies the scheduler boundary, nested single-arm and
dual-arm composition, task/runtime separation, and compatibility of all built-in
real-world environments without requiring physical hardware.
