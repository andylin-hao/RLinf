Teleoperation
=============

Set up the devices an operator drives, so a person can take over from the policy
mid-rollout: to collect demonstrations, to recover from a failure, or to run
DAgger. This guide covers choosing devices in configuration, running several
together, checking one before a robot is involved, and putting a device on the
machine it is plugged into.

For the model behind it, see :doc:`../concepts/robotics`.

Choose a Device
---------------

One key names the device:

.. code-block:: yaml

   env:
     eval:
       teleop_device: spacemouse

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - Device
     - What the operator does
     - Extra config
   * - ``spacemouse``
     - Pushes a 6-DoF puck for the arm; buttons latch the gripper.
     - none
   * - ``gello``
     - Poses a leader arm; the follower matches its pose.
     - ``gello_port``
   * - ``gello_joint``
     - Poses a leader arm, matched joint for joint.
     - ``left_gello_port`` / ``right_gello_port``
   * - ``pico``
     - Holds a VR controller; the grip says when it is driving.
     - ``pico:`` block
   * - ``none``
     - Nothing takes over; the policy runs alone.
     - none

Not every robot accepts every device. A dual-arm Franka has no single-arm
Cartesian path, so naming ``spacemouse`` there is refused rather than ignored.
Each env declares what it accepts, and the error lists them.

Run Several Together
--------------------

A device fills the parts it can drive, so several devices fill different ones.
Write a list instead of a name:

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

That is the dexterous-hand setup: the puck drives the arm, the glove drives the
hand, and holding the spacemouse's second button is what puts the glove in
control. Let go and the hand stays where you posed it.

On a robot with two of the same thing, say which branch each device drives:

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

A device that ends up driving nothing is an error when the rig is built, not a
surprise once the robot is moving. So is two devices claiming the same part.

Check a Device First
--------------------

Every device can be read on its own, with no robot and no cluster:

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

Use this when a leader arm reads zero or a spacemouse does nothing: it separates
a wiring or permissions problem from a configuration one. The bench scripts in
``toolkits/realworld_check`` cover the same ground for a whole robot.

Put a Device Where It Is Plugged In
-----------------------------------

A teleop device is a part, so it takes a node like any other. Give it a
``node_rank`` when the operator's hardware hangs off a different machine than the
one running the policy:

.. code-block:: python

   leader = TeleopLeaderArm.at("/dev/ttyUSB0", node_rank=1)

Devices on independent connections are read in parallel. One device listed under
two parts is opened once, so a spacemouse driving both an arm and a gripper does
not fight itself for the HID handle.

When the Rate Is the Problem
----------------------------

A leader arm tracks badly if the follower only hears from it at the policy's
step rate. Turn on direct streaming and a thread pushes joint targets to the
controllers at roughly 1 kHz, while ``env.step`` keeps reading state and stops
forwarding motion:

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         teleop_direct_stream: true

Leave it off unless tracking is visibly laggy. With it on, ``env.step`` no longer
dispatches joint targets, so a misconfigured rig stops moving rather than moving
badly.

Retired Spellings
-----------------

``use_spacemouse``, ``use_gello``, ``use_gello_joint``, and ``use_pico`` still
work and warn. A config carrying both a retired flag and ``teleop_device`` is an
error when the two disagree, because choosing either one silently would hand
somebody the wrong device with a robot already moving.

Next
----

- :doc:`Robotics Model <../concepts/robotics>`: devices, bindings, and groups.
- :doc:`Real-World Environment Model <../concepts/realworld_envs>`: where teleop
  sits in the wrapper stack.
- :doc:`Data Collection <data_collection>`: recording what an operator drove.
