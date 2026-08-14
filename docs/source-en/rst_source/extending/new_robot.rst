Adding a Robot
==============

Add a physical robot by implementing reusable parts, composing them into a
``Robot``, and registering its discovery configuration. Keep task logic in a
real-world environment rather than in the robot driver.

Architecture
------------

Use each layer for one responsibility:

.. list-table::
   :header-rows: 1

   * - Layer
     - Responsibility
   * - ``RobotPart``
     - Connect to one observable device and return its observations.
   * - ``ControllablePart``
     - Add actions to a part such as an arm, gripper, or mobile base.
   * - ``Robot``
     - Compose named parts and expose namespaced observations and actions.
   * - ``RobotConfig`` and ``RobotInfo``
     - Describe physical connectivity and scheduler discovery results.
   * - Real-world environment
     - Define resets, rewards, termination, and policy-facing spaces.
   * - ``PartRuntime``
     - Optionally host a part on another node through an RLinf ``Worker``.

Do not import Ray, Gymnasium, or scheduler APIs in a part implementation. This
keeps the same part usable locally, in a composed robot, or in a
``PartRuntime``.

Implement a Part
----------------

Inherit from ``RobotPart`` for observation-only devices such as cameras.
Inherit from ``ControllablePart`` for devices that also accept actions.

.. code-block:: python

   import numpy as np

   from rlinf.robotics import ControllablePart


   class ExampleArm(ControllablePart):
       def __init__(self, endpoint: str):
           self.endpoint = endpoint
           self._connected = False

       @property
       def is_connected(self) -> bool:
           return self._connected

       @property
       def observation_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       @property
       def action_features(self) -> dict:
           return {"joint_target": {"shape": (6,), "dtype": "float32"}}

       def connect(self) -> None:
           self._connected = True

       def get_observation(self) -> dict[str, np.ndarray]:
           return {"joint_position": np.zeros(6, dtype=np.float32)}

       def send_action(
           self, action: dict[str, np.ndarray]
       ) -> dict[str, np.ndarray]:
           # Send action["joint_target"] through the vendor SDK here.
           return action

       def disconnect(self) -> None:
           self._connected = False

Keep vendor SDK imports inside ``connect()`` or the constructor when only robot
nodes install that SDK.

Compose the Robot
-----------------

Give every part a stable name. The names become the top-level observation and
action keys, so the same structure supports one arm, two arms, cameras, hands,
wheels, or legs.

.. code-block:: python

   from rlinf.robotics import Robot

   robot = Robot(
       parts={
           "left_arm": ExampleArm("tcp://left-arm:5000"),
           "right_arm": ExampleArm("tcp://right-arm:5000"),
       }
   )
   robot.connect()
   observation = robot.get_observation()
   robot.send_action(
       {
           "left_arm": {"joint_target": left_target},
           "right_arm": {"joint_target": right_target},
       }
   )

Register Discovery
------------------

Register one ``RobotConfig`` and discovery policy for each hardware type. The
scheduler consumes only the generic ``HardwareResource`` result; robot-specific
validation stays in this module.

.. code-block:: python

   from dataclasses import dataclass
   from typing import Optional

   from rlinf.robotics import Robot, RobotConfig, RobotInfo
   from rlinf.scheduler.hardware import (
       HardwareConfig,
       HardwareInfo,
       HardwareResource,
   )


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       endpoint: str


   @Robot.register_robot(ExampleRobotConfig)
   class ExampleRobot(Robot):
       HW_TYPE = "ExampleRobot"

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

           infos: list[HardwareInfo] = [
               RobotInfo(
                   type=cls.HW_TYPE,
                   model=cls.HW_TYPE,
                   config=config,
               )
               for config in matching
           ]
           return HardwareResource(type=cls.HW_TYPE, infos=infos)

Import the integration module before constructing ``Cluster``. Registration is
an import-time operation, and the module must be importable in the Python
environment on every cluster node.

.. code-block:: python

   import my_project.example_robot  # Registers ExampleRobot.

   from rlinf.scheduler import Cluster

   cluster = Cluster(cluster_cfg=cfg.cluster)

Configure Physical Hardware
---------------------------

Put physical connectivity under ``cluster.node_groups.hardware``. Keep task
parameters such as reset poses, rewards, and episode horizons under ``env``.

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
               endpoint: tcp://robot-arm:5000

Host a Part Remotely
--------------------

Use ``PartRuntime`` when a device must run on a different node. It is an RLinf
``Worker``, so placement and lifecycle follow the same scheduler APIs as other
RLinf components.

.. code-block:: python

   from rlinf.robotics import PartRuntime
   from rlinf.scheduler import NodePlacementStrategy

   arm_runtime = PartRuntime.create_group(
       ExampleArm,
       {"endpoint": "tcp://robot-arm:5000"},
   ).launch(
       cluster=cluster,
       placement_strategy=NodePlacementStrategy(node_ranks=[0]),
       name="ExampleArmRuntime",
   )
   arm_runtime.initialize().wait()

Test the Integration
--------------------

Test parts without hardware by using a fake SDK or loopback transport. Cover
connection cleanup, observation keys, action dispatch, config parsing, and a
discovery result for each supported layout.

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py

What this does: it runs the built-in composition and registration contract tests
without requiring a full training run. Add equivalent unit tests for your part,
then add a hardware-gated end-to-end test when validation requires the physical
device or its vendor SDK.
