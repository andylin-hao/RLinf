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
     - Expose canonical observations and actions for one robot component.
   * - ``Driver``
     - Own one device connection and declare the parts it backs. The unit of placement.
   * - ``Arm``
     - Compose one arm driver, an optional ``EndEffector``, and named wrist cameras.
   * - ``Robot``
     - Compose named arms, robot-level cameras, and optional parts such as a base.
   * - ``RobotSpec``
     - Describe physical arms, cameras, end effectors, connections, and node ranks.
   * - ``RobotDiscovery``
     - Translate scheduler hardware configuration into generic hardware resources.
   * - ``DriverHandle``
     - Access a driver identically whether it runs locally or in a worker.
   * - ``RobotTask`` and ``RobotTaskEnv``
     - Own reset, reward, termination, Gymnasium spaces, and policy compatibility.

The dependency direction is strict. Drivers, parts, and cameras never import
Ray, Gymnasium, or ``rlinf.scheduler``; importing a driver must not pull the
scheduler into the process. Exactly one module bridges the two,
``rlinf/robotics/drivers/worker.py``, and ``Driver.spawn`` imports it lazily.
Environments consume the composed ``Robot`` and keep task semantics out of
device drivers. ``tests/unit_tests/test_robotics_boundaries.py`` enforces this.

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

   from rlinf.robotics import Arm, Robot


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"


   robot = ExampleRobot.dual_arm(
       left_arm=Arm(ExampleArmDriver("tcp://left-arm:5000")),
       right_arm=Arm(ExampleArmDriver("tcp://right-arm:5000")),
   )
   robot.connect()
   observation = robot.get_observation()
   robot.send_action(
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

Place a Driver on a Node
------------------------

``Driver.spawn`` is the only placement call. Without ``node_rank`` the driver is
built in this process; with one it is hosted in a scheduler worker on that node.
Both return a handle with the same API, so callers never branch on placement.

.. code-block:: python

   from rlinf.robotics import Arm, Robot

   handle = ExampleArmDriver.spawn(
       endpoint="tcp://left-arm:5000",
       node_rank=0,
       name="ExampleArmDriver-0",
   )
   robot = Robot.single_arm(
       Arm(handle.part("arm"), handle.part("end_effector")),
       drivers={"arm": handle},
   )
   robot.connect()

There is no per-robot worker class to write. RLinf synthesizes one from the
driver, so ``WorkerGroup`` binds every public driver method as an RPC. Methods
outside the part interface stay reachable through the handle, with the same call
shape locally and remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

Pass the handles a robot owns as ``drivers=``; ``Robot.disconnect`` releases them
after every part is disconnected. ``Robot`` applies independent arm resets,
observations, and actions in parallel. Built-in hardware uses this same path
through ``build_franka_robot``, ``build_dual_franka_robot``,
``build_gim_arm_robot``, ``build_turtle2_robot``, and ``build_dosw1_robot``.

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
   adapter and regression test instead.

Test the Integration
--------------------

Test pure drivers without vendor SDKs, composition paths, remote handle
lifecycle, physical spec translation, discovery registration, and the exact
legacy policy schema.

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py \
     tests/unit_tests/test_robotics_boundaries.py \
     tests/unit_tests/test_robot_task_env.py \
     tests/unit_tests/test_realworld_robotics_compatibility.py

What this does: it verifies the scheduler boundary, nested single-arm and
dual-arm composition, task/robot separation, and compatibility of all built-in
real-world environments without requiring physical hardware.
