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
       Hardware presenting several components over one connection declares
       them with ``subparts()``. Any part can be placed with ``spawn()``.
   * - ``Arm``
     - Compose one arm driver, an optional ``EndEffector``, and named wrist cameras.
   * - ``Robot``
     - Compose named arms, robot-level cameras, and optional parts such as a base.
   * - ``RobotDiscovery``
     - Translate scheduler hardware configuration into generic hardware resources.
   * - ``PartHandle``
     - Access a driver identically whether it runs locally or in a worker.
   * - ``RobotTask`` and ``RobotTaskEnv``
     - Own reset, reward, termination, Gymnasium spaces, and policy compatibility.

The dependency direction is strict. Parts never import Ray, Gymnasium, or
``rlinf.scheduler``; importing one must not pull the scheduler into the
process. Exactly one module bridges the two, ``rlinf/robotics/placement.py``,
and ``RobotPart.spawn`` imports it lazily. Environments consume the composed
``Robot`` and keep task semantics out of hardware code.
``tests/unit_tests/test_robotics_boundaries.py`` enforces this.

Implement a Part
----------------

Inherit from ``RobotPart`` for observation-only devices. Inherit from
``ControllablePart`` for arms, bases, and other devices that accept commands.
Import an optional vendor SDK inside ``connect()`` so every node can import the
module without installing that SDK.

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
specific interface applies. Built-in implementations live under
``rlinf/robotics/parts``, grouped by category: ``parts/arms``,
``parts/cameras``, ``parts/end_effectors/grippers``,
``parts/end_effectors/hands``, ``parts/teleop``, and ``parts/transports``.

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
       left_arm=Arm(ExampleArm("tcp://left-arm:5000")),
       right_arm=Arm(ExampleArm("tcp://right-arm:5000")),
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

Put connections and placement in a ``RobotConfig`` dataclass, and give it a
builder that turns those fields into a composed ``Robot``. Keep reset poses,
rewards, and episode horizons in the task config instead.

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import Arm, Robot, RobotConfig


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str
       right_endpoint: str


   def build_example_robot(config: ExampleRobotConfig) -> ExampleRobot:
       handles = {
           side: ExampleArm.spawn(
               endpoint=endpoint,
               node_rank=config.node_rank,
               name=f"ExampleArm-{side}",
           )
           for side, endpoint in (
               ("left", config.left_endpoint),
               ("right", config.right_endpoint),
           )
       }
       return ExampleRobot.dual_arm(
           Arm(handles["left"].part("arm")),
           Arm(handles["right"].part("arm")),
           handles=handles,
       )

Arm count is composition, not a robot type: a single-arm variant is the same
builder returning ``ExampleRobot.single_arm(...)``.

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

Place a Part on a Node
----------------------

``RobotPart.spawn`` is the only placement call, and every part has it. Without
``node_rank`` the part is built in this process; with one it is hosted in a
scheduler worker on that node. Both return a handle with the same API, so
callers never branch on placement. This is not limited to arms: a camera can run
on the machine it is plugged into while the policy runs elsewhere.

.. code-block:: python

   from rlinf.robotics import Arm, Robot

   handle = ExampleArm.spawn(
       endpoint="tcp://left-arm:5000",
       node_rank=0,
       name="ExampleArm-0",
   )
   robot = Robot.single_arm(
       Arm(handle.part("arm"), handle.part("end_effector")),
       handles={"arm": handle},
   )
   robot.connect()

There is no per-robot worker class to write. RLinf synthesizes one from the
part, so ``WorkerGroup`` binds every public method as an RPC. Methods outside
the part interface stay reachable through the handle, with the same call shape
locally and remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

Pass the handles a robot owns as ``handles=``; ``Robot.disconnect`` releases them
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

Test parts without vendor SDKs, composition paths, remote handle
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
