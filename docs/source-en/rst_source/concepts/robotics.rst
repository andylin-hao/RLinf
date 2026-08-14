Robotics Model
==============

Understand how RLinf models a physical robot before you add hardware or debug a
real-world run. The layer answers three questions: what a component *is* to the
policy, how components compose into a robot, and where each one runs.

The Core Idea
-------------

**Everything physical is a part.** An arm, a gripper, a camera, and a mobile base
are all ``RobotPart``: they connect, report an observation, and — if controllable
— accept an action. There is no separate "driver" concept sitting underneath.

That matters because hardware rarely maps one-to-one onto components. A coupled
dual-arm controller drives two arms, two grippers, and two wrist cameras over a
single ROS connection. Rather than inventing a second abstraction for "the thing
that owns the connection", such a part declares what it exposes:

.. code-block:: python

   def subparts(self) -> dict[str, RobotPart]:
       return {
           "left": MethodArm(self, commands={"tcp_pose": "move_left_arm"}),
           "right": MethodArm(self, commands={"tcp_pose": "move_right_arm"}),
           "left_end_effector": MethodGripper(self, state_field="follow1_pos"),
       }

"Owns a connection" is a property some parts have, not a kind of thing.

**A robot is a named composition of parts**, and **any part can be placed on a
node**. Those three sentences are the whole model.

The Abstractions
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Abstraction
     - What it is
   * - ``RobotPart``
     - Anything physical: ``connect``, ``get_observation``, ``disconnect``,
       ``reset``, plus ``observation_features`` describing what it returns.
   * - ``ControllablePart``
     - A part that also takes commands: ``send_action`` and ``action_features``.
   * - ``Camera`` / ``EndEffector`` / ``MobileBase`` / ``LeggedBase``
     - Narrower kinds, so composition and remote proxies can tell them apart.
   * - ``subparts()``
     - The named components one part exposes. Leaves return ``{}``.
   * - ``Arm``
     - A part composing a manipulator, an optional end effector, and wrist cameras.
   * - ``Robot``
     - Named arms, robot-level cameras, and extra parts, with the handles it owns.
   * - ``PartHandle``
     - A reference to a part, identical whether it runs here or in a worker.
   * - ``MethodArm`` / ``MethodGripper`` / ``MethodCamera``
     - Views that turn a method surface (``open_gripper``, ``get_camera(id)``)
       into parts.

Composition, Not Robot Types
----------------------------

Arm count is the size of a mapping. A single-arm and a dual-arm robot are the
same class:

.. code-block:: python

   single = FrankaRobot.single_arm(Arm(arm, gripper))
   dual = FrankaRobot.dual_arm(Arm(left, left_gripper), Arm(right, right_gripper))

Composition also fixes the shape of what the policy sees. Names become paths:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Path
     - Meaning
   * - ``arms.<name>.state``
     - That arm's manipulator observation.
   * - ``arms.<name>.arm``
     - That arm's manipulator action.
   * - ``arms.<name>.end_effector``
     - Its end-effector observation and action.
   * - ``cameras.<name>`` / ``parts.<name>``
     - Robot-level cameras and extra components.

Because arms sit on independent connections, ``Robot`` resets, reads, and
commands them in parallel: a two-arm observation costs one round trip, not two.

Placement Is a Property of Parts
--------------------------------

``RobotPart.spawn`` is the only placement call.

.. code-block:: python

   local = RealSenseCamera.spawn(camera_info)                    # here
   remote = RealSenseCamera.spawn(camera_info, node_rank=2)      # on node 2

Both return a ``PartHandle`` with the same API, so callers never branch on
placement. This is not limited to arms — a camera can run on the machine it is
plugged into while the policy runs elsewhere.

There is no per-hardware worker class. RLinf synthesizes one from the part class
(``type(name, (Worker, PartCls), ...)``), so ``WorkerGroup`` binds every public
method as an RPC. Methods outside the part interface stay reachable through the
handle, with the same call shape locally and remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

See :doc:`Placement <placement>` for how workers map onto nodes and GPUs.

The Boundary
------------

Parts never import Ray, Gymnasium, or ``rlinf.scheduler``. Importing a part must
not pull the scheduler into the process, which is what lets the bench scripts in
``toolkits/realworld_check`` run on a machine with no cluster at all.

Exactly one module crosses the line — ``rlinf/robotics/placement.py`` — and
``spawn`` imports it lazily. The scheduler, in turn, never imports robotics.
``tests/unit_tests/test_robotics_boundaries.py`` enforces both directions.

Where the Code Lives
--------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Contents
   * - ``parts/base.py``
     - The taxonomy: ``RobotPart``, ``ControllablePart``, ``Camera``,
       ``EndEffector``, ``Arm``, ``MobileBase``, ``LeggedBase``.
   * - ``parts/arms/``
     - Arm hardware and each family's state dataclass.
   * - ``parts/cameras/``
     - RealSense, ZED, Lumos.
   * - ``parts/end_effectors/``
     - ``grippers/`` and ``hands/``.
   * - ``parts/teleop/``
     - Leader arms and input devices: GELLO, glove, keyboard, Pico, spacemouse.
   * - ``parts/transports/``
     - Shared transports such as ROS. Not parts — they carry messages for one.
   * - ``robots/``
     - One module per robot: its config, discovery, and builder.
   * - ``placement.py``
     - ``PartHandle`` and the synthesized worker. The only scheduler import.
   * - ``views.py``
     - The ``Method*`` views.
   * - ``robot.py``, ``discovery.py``, ``adapters.py``, ``config.py``
     - Composition, registration, legacy policy adapters, env-var config.

Tasks Stay Out of Hardware
--------------------------

A part knows how to move and what it senses. It does not know what counts as
success. Reset behavior, reward, termination, and Gymnasium spaces belong to a
``RobotTask``, combined with a ``Robot`` by ``RobotTaskEnv``.
``LegacyObservationAdapter`` and ``VectorActionAdapter`` translate the composed
interface into the flat vectors and ``state``/``frames`` observations an existing
policy expects, so hardware code never learns the policy's schema.

Next
----

- :doc:`Adding a Robot <../extending/new_robot>` — the step-by-step how-to.
- :doc:`Placement <placement>` — how workers map onto nodes and GPUs.
