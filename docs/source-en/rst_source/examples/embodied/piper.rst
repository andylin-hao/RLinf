AgileX Piper Setup and Control
==============================

.. figure:: https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/asserts/pictures/piper_urdf_zero.png
   :align: center
   :width: 70%

   Piper arm and gripper model. Image: AgileX Robotics, piper_ros.

Connect an AgileX Piper arm to RLinf, read its joints and gripper, and send a
command through the robotics interface. Start with a local CAN connection, add
a camera if your application needs images, then use the task-extension guide to
build your own real-world task. The supplied reach environment and mock SAC run
exercise the integration; they are not a validated task or training recipe.

Overview
--------

Piper support covers the arm and its attached AgxGripper through the
``pyAgxArm`` SDK. The current test configuration uses an MLP policy to check the
training path without physical hardware.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP policy (integration test)

   .. grid-item-card:: Algorithms
      :text-align: center

      SAC (integration test)

   .. grid-item-card:: Tasks
      :text-align: center

      User-defined; reach scaffold

   .. grid-item-card:: Hardware
      :text-align: center

      Piper · CAN · AgxGripper

| **You'll do:** install dependencies → configure CAN → read feedback → test joints and gripper → add a task.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Piper hardware · Linux controller · CAN adapter.

Tasks
~~~~~

Use the existing files to inspect the hardware-to-env contract before defining
the behavior you want the robot to learn.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Purpose
     - Entry Point
     - Scope
   * - Local hardware use
     - ``PiperRobot`` with ``PiperArm``
     - Read state and command the arm and gripper.
   * - Environment scaffold
     - ``PiperReachEnv-v1``; ``env/piper_reach.yaml``
     - Illustrates joint targets, reset, and reward wiring.
   * - Integration test
     - ``piper_mock_sac_mlp_reach``
     - Runs SAC with mock SDKs; does not establish physical task performance.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The robot returns nested dictionaries. The reach scaffold converts these into
the flat state and action arrays used by its test policy.

.. list-table::
   :header-rows: 1
   :widths: 23 77

   * - Field
     - Contract
   * - Observation
     - ``arm.arm_joint_position``: six radians; ``arm.tcp_pose``:
       ``[x, y, z, qx, qy, qz, qw]`` in the base frame, with metres for position;
       ``arm.end_effector.state``: one gripper opening in ``[0, 1]``.
       The scaffold's flat state has 14 values with a gripper.
   * - Action
     - ``arm.joint_position``: six absolute joint targets in radians;
       ``arm.end_effector.target``: one opening, ``0`` closed and ``1`` open.
       The scaffold's flat action has seven values; targets are not deltas.
   * - Images
     - Optional cameras. The scaffold centre-crops and resizes each image to
       ``(128, 128, 3)`` under ``frames.wrist_1``, ``frames.wrist_2``, etc.
       Without cameras it omits ``frames``; the mock MLP ignores images.
   * - Reward
     - The scaffold uses joint distance for dense reward or a per-joint
       tolerance for sparse reward. Define reward and success for your own task.
   * - Prompt
     - The scaffold reports ``reach a joint configuration``.

Hardware Setup
--------------

Prepare one controller machine and its attached arm before installing the software.
The local hardware checks run on the controller; a GPU is needed only for the
optional mock SAC integration test.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Requirement
   * - Arm and gripper
     - AgileX Piper with its power supply; AgxGripper if fitted.
   * - Controller machine
     - Linux computer with a USB-to-CAN adapter in SocketCAN mode and a CAN cable to the arm.
   * - Camera (optional)
     - Intel RealSense camera connected to the controller for image checks.
   * - GPU (optional)
     - One supported GPU for the mock SAC test, separate from the local hardware check.

Mount the arm on a stable surface and leave space for small joint movements.
Connect the power supply and CAN cable according to the manufacturer's wiring
instructions before powering it on.

Installation
------------

Install the robot dependencies on the controller machine using the custom-environment
workflow below. Run the checks from the same checkout and activated environment.

Robot Controller Node
~~~~~~~~~~~~~~~~~~~~~

Clone RLinf on the machine connected to the robot, then install its environment.

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Install the robot dependencies from the repository root:

.. code-block:: bash

   # Mainland China users can add --use-mirror for faster downloads.
   bash requirements/install.sh embodied --env piper
   source .venv/bin/activate
   export REPO_PATH="$PWD"

Run subsequent commands from this checkout with the environment activated.
The installer includes the pinned ``pyAgxArm`` driver used by ``PiperArm``; see
:doc:`Installation </rst_source/start/installation>` for platform prerequisites.
No model checkpoint is needed for the interface examples or the mock MLP test.

Prepare the CAN Connection
^^^^^^^^^^^^^^^^^^^^^^^^^^

Find the SocketCAN interface with ``ip -brief link``. A USB-to-CAN adapter in
SocketCAN mode normally appears as ``can0``; replace that name below if yours
differs. Bring this interface up at 1 Mbit/s before opening the robot:

.. code-block:: bash

   sudo ip link set can0 down
   sudo ip link set can0 type can bitrate 1000000
   sudo ip link set can0 up
   ip -details link show can0

The output should show an enabled interface and ``bitrate 1000000``. With the
arm powered on, run ``candump -n 5 can0`` to inspect five feedback frames. If it
waits without receiving frames, press Ctrl+C and check the arm's power, CAN
wiring, selected adapter, and bitrate before continuing. The SDK does not
configure the CAN interface. Consult the
`pyAgxArm documentation <https://github.com/agilexrobotics/pyAgxArm>`_ for adapter
setup and supported hardware.

Run It
------

With CAN configured, use the arm in the calling Python process first. A
``PiperArm`` declaration records the device settings without opening it, and
``PiperRobot`` names that arm ``arm``. Its gripper shares the CAN connection and
appears at ``arm.end_effector``.

Testing the Setup
~~~~~~~~~~~~~~~~~

First check the toolkit without hardware:

.. code-block:: bash

   python -m toolkits.realworld_check.test_piper_controller --mock --read-only

This prints six measured joint angles, a seven-value tool pose, and one gripper
opening, then exits. Mock values only verify that the software path works.
Repeat the check on the connected arm:

.. code-block:: bash

   python -m toolkits.realworld_check.test_piper_controller \
     --channel can0 --read-only

The command enables the motors and reads feedback without sending a position
target or resetting the arm. The detected firmware profile appears in the log.
For an AgxGripper with a 0.1 m stroke, add ``--gripper-max-width 0.1``; for an arm
without a gripper, add ``--no-gripper``.

To test a small movement, omit ``--read-only`` and enter the commands below one
at a time, waiting for the arm after each movement:

.. code-block:: bash

   python -m toolkits.realworld_check.test_piper_controller \
     --channel can0 --speed-percent 10

.. code-block:: text

   where
   joint 1 2
   where
   joint 1 -2
   grip 0.5
   where
   quit

``joint 1 2`` moves joint 1 by two degrees from its measured position. The
script accepts joints 1 through 6 and increments within five degrees; the arm
driver also clips targets to its travel. ``grip 0.5`` requests half the fitted
gripper's stroke. ``where`` reads feedback, so repeat it after the gripper has
settled. ``quit``, EOF, or Ctrl+C closes the CAN session and leaves the motors
holding position. Add ``--mock`` to rehearse the same commands without hardware.

.. warning::

   Connection enables the motors even with ``--read-only``. Clear the workspace
   before connecting, and keep hands away during joint and gripper commands.
   Closing the test is not an emergency stop and does not cut motor power.

Use the Robot Interface
~~~~~~~~~~~~~~~~~~~~~~~

To use the same interface in your own Python program, run the following code.
The example reads the current joint positions and gripper opening, then sends
those values back as targets. ``connect()`` opens the connection and enables the
arm; ``get_observation()`` returns the nested state used to construct the action.

.. code-block:: python

   from rlinf.robotics import PiperRobot
   from rlinf.robotics.parts.arms.piper import PiperArm
   from rlinf.utils.logging import get_logger

   logger = get_logger()
   robot = PiperRobot(arm=PiperArm.declare("can0", speed_percent=10))
   robot.connect()
   try:
       reading = robot.get_observation()["arm"]
       logger.info("Joint positions: %s", reading["arm_joint_position"])
       robot.send_action(
           {
               "arm": {
                   "joint_position": reading["arm_joint_position"],
                   "end_effector": {"target": reading["end_effector"]["state"]},
               }
           }
       )
   finally:
       robot.disconnect()

``send_action()`` dispatches the nested targets and returns a dictionary of
dispatched action values, not a new measurement. Read again to measure the
result. ``disconnect()`` releases the robot's connections, including the shared
gripper connection. Omitting ``node_rank`` keeps this example local and does not
require a Ray cluster.

Set ``gripper_max_width`` to the fitted gripper's full stroke in metres; it
defaults to ``0.07``. For an arm without a gripper, declare
``with_gripper=False`` and omit the end-effector action and observation access.
That variant has six action values and 13 state values in the scaffold.

Add a Camera
~~~~~~~~~~~~

To read images alongside the arm state, replace the robot declaration above
with this composition. ``CameraInfo`` describes a physical camera;
``Camera.declare()`` returns the named, unconnected camera parts to include in
the robot. The robot installer includes RealSense support on x86 Linux.

Select the RealSense driver with ``Camera.backend("realsense")`` and list
attached serial numbers with ``discover()``:

.. code-block:: python

   from rlinf.robotics import Camera

   logger.info("RealSense serials: %s", Camera.backend("realsense").discover())

Use one of the reported serial numbers as ``CAMERA_SERIAL`` in the following
code, then connect and read an image:

.. code-block:: python

   from rlinf.robotics import Camera, CameraInfo

   robot = PiperRobot(
       arm=PiperArm.declare("can0", speed_percent=10),
       **Camera.declare(
           {"wrist_1": CameraInfo(name="wrist_1", serial_number="CAMERA_SERIAL")}
       ),
   )
   robot.connect()
   try:
       image = robot.get_observation()["wrist_1"]["frame"]
       logger.info("Camera image shape: %s", image.shape)
   finally:
       robot.disconnect()

The default camera backend is RealSense. The image uses the camera's configured
resolution; resizing to the policy input belongs to the environment. The robot
opens and closes the camera along with the arm. See
:doc:`Robotics Interface </rst_source/concepts/robotics>` for other compositions.

Check the Integration Without Hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the interface is clear, the existing mock run can check the scheduler,
environment, rollout, and actor path on a machine with one supported GPU:

.. code-block:: bash

   bash tests/e2e_tests/embodied/run.sh piper_mock_sac_mlp_reach \
     runner.logger.log_path="$REPO_PATH/logs/piper_mock"

The launcher enables mock SDKs because the config name contains ``mock``. It
uses the real env implementation with ``is_dummy: False``, two fake cameras,
and an MLP with ``action_dim: 7`` and ``obs_dim: 14``. This is a short software
integration check; it neither moves a physical arm nor validates a task.

New Real-World Tasks
--------------------

After verifying your hardware, define the task's observations, reset procedure,
reward, and success condition using
:doc:`New Real-World Tasks </rst_source/extending/new_task>`. Piper currently has
hardware support and a reach scaffold, but no validated real-world task recipe.

Start from ``PiperEnv`` and ``PiperEnvConfig`` in
``rlinf/envs/real/piper/base.py``. The adjacent ``reach.py`` shows how to build a
task config from ``override_cfg`` and pass it into the base constructor; do not
copy the Franka-specific ``CONFIG_CLS`` shortcut from the guide. Register the
new class in ``rlinf/envs/real/piper/__init__.py`` under ``TASKS``, then copy
``examples/embodiment/config/env/piper_reach.yaml`` and select your Gymnasium ID
with ``init_params.id``.

Keep hardware settings in the cluster's ``hardware`` block with ``type: Piper``
and task settings in ``env.train.override_cfg``. The existing mock config shows
how to place ``env`` in the node group that carries the robot. For a controller
on another machine, follow :doc:`Multi-Node Training </rst_source/guides/multi_node>`
and :doc:`Robotics Architecture </rst_source/concepts/robotics_architecture>`.
Piper currently has no registered teleoperation device for its joint layout;
keep ``teleop: none`` unless you add a compatible device.

Visualization and Results
-------------------------

The mock run writes TensorBoard logs under ``logs/piper_mock``. Use them to
inspect software execution, not to measure physical task success. There are no
validated task results or pretrained task weights for this example. Once your
task is implemented, configure logging and video using
:doc:`Training Metrics </rst_source/reference/metrics>` and
:doc:`Logging </rst_source/guides/logger>`.
