Adding a Robot
==============

This guide adds a mobile base, composes it with RLinf's existing Franka arm, and
drives the resulting mobile manipulator through a real-world Gymnasium
environment. The example starts with the base in one process. Placement and
hardware discovery come only after its observations and actions work locally.

Before continuing, read :doc:`Robot Composition <../concepts/robotics>`. If RLinf
already knows how to connect the hardware and you only need a new reward, reset,
or success condition, follow :doc:`New Real-World Tasks <new_task>` instead.

1. Implement the Mobile Base Locally
------------------------------------

A mobile base is one controllable robot part: it reports its pose and accepts a
velocity command. Inherit ``MobileBase`` so the class states that role directly.
Keep the vendor SDK import inside ``_open()``; nodes that only import the module do
not need the SDK installed.

.. code-block:: python

   import numpy as np

   from rlinf.robotics import MobileBase


   @MobileBase.register("example")
   class ExampleMobileBase(MobileBase):
       def __init__(self, endpoint: str):
           self.endpoint = endpoint

       def _open(self):
           from example_mobile_sdk import Client

           return Client(self.endpoint)

       def _release(self, device) -> None:
           try:
               device.stop()
           finally:
               device.close()

       @property
       def observation_features(self) -> dict:
           return {"pose": {"shape": (3,), "dtype": "float32"}}

       @property
       def action_features(self) -> dict:
           return {"velocity": {"shape": (2,), "dtype": "float32"}}

       def reset(self) -> None:
           self._device.stop()

       def get_observation(self) -> dict[str, np.ndarray]:
           pose = np.asarray(self._device.get_pose(), dtype=np.float32)
           return {"pose": pose}

       def send_action(
           self, action: dict[str, np.ndarray]
       ) -> dict[str, np.ndarray]:
           if set(action) != {"velocity"}:
               raise KeyError("Expected only 'velocity'.")
           velocity = np.asarray(action["velocity"], dtype=np.float32)
           if velocity.shape != (2,):
               raise ValueError(f"Expected velocity shape (2,), got {velocity.shape}.")
           self._device.set_velocity(
               linear=float(velocity[0]),
               angular=float(velocity[1]),
           )
           return {"velocity": velocity}

The decorator gives this driver the backend name ``example``. Code that already
has the concrete class can still instantiate it directly; a
configuration-driven builder can instead resolve it with
``MobileBase.backend("example")``. This registry names interchangeable device
drivers, not complete robot types. The robot type is registered later with
``register_type()``.

Here ``pose`` is ``[x, y, yaw]`` and ``velocity`` is
``[linear_velocity, angular_velocity]``. Use names and units that match the
canonical interface you want tasks and datasets to retain; do not expose a
vendor method name as a policy field.

Connect the base directly before composing it with anything else:

.. code-block:: python

   base = ExampleMobileBase("tcp://mobile-base:7000")
   base.connect()
   try:
       print(base.get_observation())
       base.send_action(
           {"velocity": np.array([0.1, 0.0], dtype=np.float32)}
       )
   finally:
       base.disconnect()

Whatever ``_open()`` returns becomes ``self._device``. Cleanup receives that
same object as ``_release(device)``, so stop and release the argument rather
than reading it back from ``self``. ``connect()`` and ``disconnect()`` are
idempotent in the base class; preserve that property if the device needs a
custom lifecycle.

.. warning::

   A mobile base can keep moving after the Python process loses contact. Make
   the hardware controller enforce command timeouts and velocity limits, and
   send a stop command from ``_release`` as a final software safeguard.

2. Compose It with an Existing Arm
----------------------------------

The base is useful on its own, but it should not require a new arm driver. Build
a mobile manipulator by placing the base beside parts backed by the existing
Franka connection:

.. code-block:: python

   from rlinf.robotics import FrankaRobot, Robot


   class MobileManipulator(Robot):
       ROBOT_TYPE = "MobileManipulator"


   base = ExampleMobileBase(
       "tcp://mobile-base:7000",
       node_rank=0,
       worker_name="ExampleMobileBase-0-0",
   )
   arm_connection = FrankaRobot.declare_arm(
       "10.0.0.2",
       node_rank=0,
       name="FrankaArm-0-0",
   )
   robot = MobileManipulator(
       base=base,
       arm=arm_connection,
   )

Both arguments are composed the same way, because both are parts. Constructing
the ``MobileBase`` subclass creates an unconnected ``RobotPart``, and the Franka
arm is one too. The argument name becomes each part's public path in
``robot.children``.

The difference is what they carry. The base carries nothing. The Franka arm
carries its end effector on the same hardware session, so composing the arm
composes the gripper with it, at ``arm.end_effector``. The robot does not name
it and does not need to know it is there -- which is what lets an arm that
decides at run time whether a gripper is fitted work without a robot edit.

Name a part one at a time only when the robot's names differ from the driver's,
or when a link is not a part at all and you have to pick from it: a two-arm
session is composed with ``session.part("left")``.

``PartGroup`` checks this boundary as soon as the robot is composed. It accepts
a ``RobotPart`` or another ``PartGroup``. A bare ``Connection`` that is not a
readable part is rejected with an error that names the invalid keyword.

The existing arm builder can shorten the same composition when its standard
names are appropriate:

.. code-block:: python

   arm_parts = FrankaRobot.build_arms(
       robot_ip="10.0.0.2",
       node_rank=0,
       worker_rank=0,
       env_idx=0,
   )
   robot = MobileManipulator(base=base, **arm_parts)

Replacing ``FrankaRobot.build_arms`` with another robot family's part builder,
or wrapping several arm parts in a ``PartGroup``, does not change the mobile base.
The composition defines the robot; no base-specific arm slot or mobile-arm
subclass is required by the framework.

Check those names and their resource ownership before opening hardware:

.. code-block:: text

   >>> print(robot.describe())
   MobileManipulator
   ├── base                ExampleMobileBase    node=0     via ExampleMobileBase#1
   └── arm                 FrankaROSArm         node=0     via FrankaROSArm#2
       └── end_effector    MethodEndEffector    node=0     via FrankaROSArm#2

The arm and end effector share one ``via`` because they use one Franka
connection. The base has its own connection. The paths, nodes, and ownership
remain visible after ``connect()``; a remotely placed connection then uses its
synthesized class name, such as RemoteFrankaROSArm.

Once connected, observations and actions use the names from the composition:

.. code-block:: python

   robot.connect()
   try:
       observation = robot.get_observation()
       base_pose = observation["base"]["pose"]
       arm_pose = observation["arm"]["tcp_pose"]

       # A partial action moves only the base.
       robot.send_action(
           {"base": {"velocity": np.array([0.1, 0.0], dtype=np.float32)}}
       )

       # A task may also command the base and existing arm together.
       robot.send_action(
           {
               "base": {"velocity": base_velocity},
               "arm": {
                   "tcp_pose": arm_target,
                   "end_effector": {"target": gripper_target},
               },
           }
       )
   finally:
       robot.disconnect()

``PartGroup.send_action`` accepts a partial tree, so a navigation task need not
send hold commands for the arm. When an action contains both connections,
RLinf can dispatch them in parallel; the arm and end effector remain ordered
because they share one connection.

3. Use the Robot in a Real-World Environment
--------------------------------------------

Hardware code says how to move the base. Task code decides where to move it,
when an episode succeeds, and which subset of the robot a policy controls. The
following ``RobotTask`` exposes only the base even though the robot also carries
an arm:

.. code-block:: python

   import gymnasium as gym

   from rlinf.envs.real import RobotTask, RobotTaskEnv


   class DriveToTarget(RobotTask):
       def __init__(self, target_xy: np.ndarray):
           self.target_xy = np.asarray(target_xy, dtype=np.float32)

       @property
       def description(self) -> str:
           return "drive the mobile manipulator to the target"

       @property
       def observation_space(self) -> gym.Space:
           return gym.spaces.Dict(
               {
                   "base": gym.spaces.Dict(
                       {
                           "pose": gym.spaces.Box(
                               -np.inf, np.inf, shape=(3,), dtype=np.float32
                           )
                       }
                   )
               }
           )

       @property
       def action_space(self) -> gym.Space:
           return gym.spaces.Dict(
               {
                   "base": gym.spaces.Dict(
                       {
                           "velocity": gym.spaces.Box(
                               low=np.array([-0.5, -1.0], dtype=np.float32),
                               high=np.array([0.5, 1.0], dtype=np.float32),
                           )
                       }
                   )
               }
           )

       @staticmethod
       def observe(robot: Robot) -> dict:
           return {"base": robot.get_observation()["base"]}

       def reset(self, robot: Robot, *, seed=None, options=None):
           del seed, options
           robot.reset()
           return self.observe(robot), {}

       def step(self, robot: Robot, action: dict):
           robot.send_action(action)
           observation = self.observe(robot)
           distance = float(
               np.linalg.norm(observation["base"]["pose"][:2] - self.target_xy)
           )
           reached = distance < 0.05
           return observation, float(reached), reached, False, {"distance": distance}


   env = RobotTaskEnv(robot, DriveToTarget(np.array([1.0, 0.0])))
   try:
       observation, info = env.reset()
       observation, reward, terminated, truncated, info = env.step(
           {"base": {"velocity": np.array([0.1, 0.0], dtype=np.float32)}}
       )
   finally:
       env.close()

``RobotTaskEnv`` connects the composed robot when the environment is created
and disconnects it in ``close()``. A manipulation task can expand the spaces
and actions with ``arm`` and ``arm.end_effector``; the base driver and robot
composition stay unchanged.

To launch the task through RLinf's distributed ``RealWorldEnv``, register the
environment with Gymnasium and set ``env_type: real`` plus its Gym ID in the env
YAML. The current rollout interface expects policy-facing ``state`` and
``frames`` observations, so add ``LegacyObservationAdapter`` and
``VectorActionAdapter`` when the policy uses that representation. Follow
:doc:`New Real-World Tasks <new_task>` for registration, YAML, wrappers, and
compatibility checks.

4. Place the Same Composition on Hardware Nodes
------------------------------------------------

Placement changes where each connection opens, not how tasks address it. For
example, keep the base controller on node 0 and the Franka controller on node 1:

.. code-block:: python

   base = ExampleMobileBase(
       "tcp://mobile-base:7000",
       node_rank=0,
       worker_name="ExampleMobileBase-0-0",
   )
   arm_connection = FrankaRobot.declare_arm(
       "10.0.0.2",
       node_rank=1,
       name="FrankaArm-0-0",
   )
   robot = MobileManipulator(
       base=base,
       arm=arm_connection,
   )

Constructing these objects records their arguments, ``node_rank``, and
``worker_name`` but does not import either vendor SDK or open hardware. The
metaclass on ``Connection`` consumes the placement keywords before the driver's
``__init__`` runs, so a new driver only declares hardware-specific parameters.
``robot.connect()`` opens each distinct connection once, on the node that
connection named. A connection bound for another node is rebuilt there and the
object in the tree becomes a view of it, so the robot holds the same parts
either way and nothing in your code branches on placement.

If a later connection fails, the robot closes the connections that completed
before it. A driver must still clean up resources acquired inside a failing
``_open()`` call, because that connection never completed and cannot be rolled
back by the robot.

5. Build the Composition from Configuration
-------------------------------------------

Once the bench composition is correct, describe its hardware inputs with a
``RobotConfig`` dataclass and implement ``build()``. Reuse the existing arm
declaration rather than copying its SDK or lifecycle code:

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import MobileBase, RobotConfig


   @dataclass
   class MobileManipulatorConfig(RobotConfig):
       base_backend: str = "example"
       base_endpoint: str | None = None
       arm_ip: str | None = None
       controller_node_rank: int | None = None


   class MobileManipulator(Robot):
       ROBOT_TYPE = "MobileManipulator"

       @classmethod
       def build(
           cls,
           *,
           base_backend: str = "example",
           base_endpoint: str | None,
           arm_ip: str | None,
           node_rank: int,
           controller_node_rank: int | None = None,
           worker_rank: int = 0,
           env_idx: int = 0,
       ) -> "MobileManipulator":
           if not base_endpoint:
               raise ValueError("MobileManipulator requires base_endpoint.")
           base_cls = MobileBase.backend(base_backend)
           base = base_cls(
               base_endpoint,
               node_rank=node_rank,
               worker_name=f"{base_cls.__name__}-{worker_rank}-{env_idx}",
           )
           arm_node_rank = (
               node_rank
               if controller_node_rank is None
               else controller_node_rank
           )
           arm_connection = FrankaRobot.declare_arm(
               arm_ip,
               node_rank=arm_node_rank,
               name=f"FrankaArm-{worker_rank}-{env_idx}",
           )
           return cls(base=base, arm=arm_connection)

``MobileBase.backend()`` resolves the driver name declared in the hardware
config. The builder then composes unconnected parts and selections from shared
connections; it does not open either device. Connection addresses, backend
selection, and placement belong here. Targets, rewards, reset poses, and
episode horizons remain in the task config.

Keep the signature explicit. ``Robot.of_type()`` and ``build_robot()`` forward
their keyword arguments directly to ``build()``; registration does not unpack a
``RobotConfig`` instance or discard fields the builder does not recognize. An
unexpected config key should therefore raise at this boundary instead of being
absorbed by ``**kwargs`` and silently ignored.

.. warning::

   Call ``connect()`` before reading observations or sending commands, and
   ``disconnect()`` during teardown. ``RobotTaskEnv`` performs both lifecycle
   operations when it owns the robot.

6. Register the Robot Type
--------------------------

Most robots do not need a discovery class of their own. Register the robot and
its config at the end of the module; ``register_type()`` creates the standard
discovery class and associates it with ``build()``:

.. code-block:: python

   MobileManipulator.register_type(MobileManipulatorConfig)

The standard discovery flow selects configs assigned to the current node,
fills unset fields from same-named uppercase environment variables, and returns
one hardware record per config. Camera fields, when present, use the shared
camera discovery and validation path. Pass a custom ``RobotDiscovery`` subclass
as the second argument only when the robot genuinely needs a different
enumeration procedure.

``Connection.register()`` and ``Robot.register_type()`` serve different
registries: the first names one device driver, while the second names the whole
robot composition. After the robot type is registered, callers can use either
``Robot.of_type("MobileManipulator", ...)`` or the convenience function
``build_robot("MobileManipulator", ...)``. Both calls still require the
builder's keyword arguments; registration does not turn a hardware config into
those arguments automatically.

For an in-tree implementation, place the module under
``rlinf/robotics/robots/`` and import it from that package's ``__init__.py``.
Importing ``rlinf.robotics.robots`` will then perform the registration before a
``Cluster`` or the bench checker looks up the type. An external integration
must import its registration module explicitly in its entry point instead.
Registered robot modules are also imported by node probes, so every node's
configured Python environment must be able to import the module.

7. Configure the Cluster
------------------------

The cluster describes physical resources under
``cluster.node_groups.hardware``. Each entry is parsed with the registered
config class and becomes a ``RobotInfo`` during hardware discovery. Put
endpoints and node ranks in this YAML rather than embedding a deployment in
Python:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement: {}
     node_groups:
       - label: mobile_manipulator
         node_ranks: 0
         hardware:
           type: MobileManipulator
           configs:
             - node_rank: 0
               base_backend: example
               base_endpoint: tcp://mobile-base:7000
               arm_ip: 10.0.0.2
               controller_node_rank: 1
       - label: arm_controller
         node_ranks: 1

Here ``node_rank`` identifies the node that owns the configured robot resource;
``controller_node_rank`` places the reused Franka connection on its controller node.
The env config selects the Gym ID separately, so the same hardware composition
can serve navigation, mobile manipulation, or data-collection tasks.

The environment receives that ``RobotInfo`` and calls the registered builder
explicitly. This is the visible boundary where scheduler metadata such as the
env worker rank is added:

.. code-block:: python

   hardware = robot_info.config
   robot = build_robot(
       "MobileManipulator",
       base_backend=hardware.base_backend,
       base_endpoint=hardware.base_endpoint,
       arm_ip=hardware.arm_ip,
       node_rank=hardware.node_rank,
       controller_node_rank=hardware.controller_node_rank,
       worker_rank=worker_info.rank,
       env_idx=env_idx,
   )

Keeping this call explicit prevents the hardware registry from becoming an
implicit adapter between unrelated config shapes. If several environments use
the same robot, place the translation in shared setup code rather than copying
it into each task.

8. Test the Integration
-----------------------

Before running a contract, add a minimal fake for the example SDK under
``tests/robot_mocks/`` and include it in ``sdk_modules()``. The fake needs
only the device-side methods used above: ``get_pose()``, ``set_velocity()``,
``stop()``, and ``close()``. Registering it in one place makes the same fake
available to unit tests, the bench checker, and remote mock workers.

Then add a test under ``tests/unit_tests/`` and point the contract classes at
what you wrote. The contracts state the promises the rest of the framework
relies on:

.. code-block:: python

   from robot_contracts import PartContract, RobotContract
   from robot_mocks import mocked_sdks


   def test_mobile_base_conforms():
       assert MobileBase.backend("example") is ExampleMobileBase
       with mocked_sdks():
           PartContract(
               lambda: ExampleMobileBase("tcp://mobile-base:7000"),
               action={"velocity": np.zeros(2, dtype=np.float32)},
           ).assert_kept()


   def test_mobile_manipulator_conforms():
       with mocked_sdks():
           RobotContract(
               lambda: MobileManipulator.build(
                   base_endpoint="tcp://mobile-base:7000",
                   arm_ip="10.0.0.2",
                   node_rank=0,
                   controller_node_rank=0,
               )
           ).assert_kept()

They live in ``tests/robot_contracts``, beside the fake SDKs in
``tests/robot_mocks``, because they check RLinf rather than being part of it.

The contracts connect, read, disconnect, repeat the lifecycle, and disconnect
once more. ``PartContract`` compares one part's observation names and shapes
with its declaration. When it receives the velocity sample, it also checks that
the action is accepted and an unknown field is refused. ``RobotContract`` checks
that the robot can be described before connecting, validates the composed
top-level leaves, and injects a failure during ``connect()`` to verify rollback.

The contract currently treats a ``RobotPart`` that carries riders as one leaf;
it does not separately walk those riders. Add direct composition assertions for
the paths and owners introduced by this robot:

.. code-block:: python

   robot = MobileManipulator.build(
       base_endpoint="tcp://mobile-base:7000",
       arm_ip="10.0.0.2",
       node_rank=0,
       controller_node_rank=0,
   )
   assert set(robot.named_parts) == {"base", "arm", "arm.end_effector"}
   end_effector = robot.child("arm").child("end_effector")
   assert end_effector.owner is robot.child("arm").owner
   assert len(robot.owners()) == 2

Add ``ConnectionContract`` only when the new SDK session backs several parts.
It checks the session lifecycle and the observations returned by its ``parts``
mapping. Also assert that ``connection.part(name).owner is connection`` for each
selected part; the contract does not currently make that selection itself. This
example adds one leaf ``MobileBase`` and reuses the already-tested Franka
connection, so there is no new shared session to test.

These checks cover failures the package has previously encountered. A contract
failure names the broken promise rather than only the assertion, and lists all
failures at once:

.. code-block:: text

   ConformanceError: ExampleMobileBase does not keep 2:
     - ExampleMobileBase: reconnecting raised RuntimeError: threads can only be started
       once; stall recovery closes a connection and opens it again
     - ExampleMobileBase observes pose with shape (4,), declares (3,)

The rest of the integration is testable the same way. Cover the part contract,
composition paths, connection lifecycle, discovery registration, and the schema
policies expect:

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py tests/unit_tests/test_conformance.py \
       tests/unit_tests/test_real_env.py

These exercise the scheduler import boundary, part composition, the task and
robot split, and the policy-facing schema of every built-in real-world
environment. None of it requires physical hardware.

Run It Against Faked SDKs
~~~~~~~~~~~~~~~~~~~~~~~~~

A part imports its vendor SDK when it opens, never at import time, so a fake in
``sys.modules`` is enough to run the real part classes with nothing on the
other end of the cable. ``tests/robot_mocks`` holds one fake per SDK.

Walk your robot's composition against them:

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MobileManipulator --mock \
       --arg base_endpoint=tcp://mobile-base:7000 \
       --arg arm_ip=10.0.0.2 --arg node_rank=0 --arg controller_node_rank=0

It reports what the robot is made of, which connection backs each part and
where it was placed, then reads every part and disconnects. It fails when a
part observes something it never declared, when a value comes back a different
shape from the one declared, when a connection ends up in the tree, or when
anything still claims to be connected afterwards.

The shape check is particularly important: an env builds its observation space
from what a part declares, so a value one number wider reaches a policy as data
rather than as an error.

Add ``--remote`` to preserve the ``node_rank`` declarations while using the
mock SDKs. Connections that declare a node then run in scheduler workers;
connections without a ``node_rank`` still open in the current process. This is
what catches a part that cannot be placed at all -- for example, a method whose
name collides with the worker's own, or state that does not survive the process
boundary.

A whole training run works the same way. ``run.sh`` installs the fakes when the
config name contains ``mock``. Turtle2 is the closest shipped example of a
mobile manipulator and provides a working template for the new config:

.. code-block:: bash

   bash tests/e2e_tests/embodied/run.sh realworld_xsquare_turtle2_mock_sac_cnn

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

The remaining checks require hardware: timing, calibration, and vendor behavior
not covered by the SDK documentation. Once the robot is powered and reachable,
drop ``--mock`` and run the same check against it:

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MobileManipulator \
       --arg base_endpoint=tcp://mobile-base:7000 \
       --arg arm_ip=10.0.0.2 --arg node_rank=0 --arg controller_node_rank=1
