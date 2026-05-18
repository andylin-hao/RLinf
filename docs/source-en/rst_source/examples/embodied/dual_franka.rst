Real-World Dual-Franka: GELLO Collection, π₀.₅ SFT, Deployment
================================================================

End-to-end guide for the **dual-arm Franka** real-world setup in RLinf:
two-node bring-up, 1 kHz GELLO joint-space dual-arm data collection,
π₀.₅ SFT in a 20-D rot6d action space, and foot-pedal-controlled
deployment back to hardware.

Read first:

* :doc:`franka` — single-arm Franka basics, Ray cluster bring-up,
  RealSense + SpaceMouse data collection path. Read it through
  before this page if you are not already familiar with
  ``FrankaController`` / ``FCI`` / ``RLINF_NODE_RANK``.
* :doc:`franka_gello` — GELLO hardware install, Dynamixel SDK,
  ``gello-teleop`` package, USB-FTDI permissions.


Hardware topology
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Node
     - Role
     - Hardware on this node
   * - **node 0** (head)
     - Ray head; env worker; left ``FrankyController``;
       deployment-time actor / rollout; all cameras and GELLO capture
     - 1× GPU (e.g. RTX 4090, only used at SFT and deployment);
       left Franka FR3 wired to a dedicated NIC into the FCI port;
       left Robotiq 2F-85 (USB-RS485 Modbus);
       **both GELLO** Dynamixel chains (USB-FTDI);
       **all three cameras** — base RealSense D435i (third-person) +
       left-wrist Lumos USB-3 + right-wrist Lumos USB-3;
       PCsensor 3-key foot pedal
   * - **node 1** (worker)
     - Ray worker; runs only the right ``FrankyController``
     - Optional GPU (not required for inference);
       right Franka FR3 wired to its own NIC into the FCI port;
       right Robotiq 2F-85

.. note::

   FCI IPs and NIC names depend on your network — fill them into
   the Hardware YAML below.

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Camera slot
     - Backend
     - Use
   * - ``base_0_rgb``
     - RealSense D435i
     - Third-person view, shared by both arms
   * - ``left_wrist_0_rgb``
     - Lumos USB 3 (XVisio vSLAM)
     - Left wrist; serves as π₀.₅'s main ``image``
   * - ``right_wrist_0_rgb``
     - Lumos USB 3 (XVisio vSLAM)
     - Right wrist

The foot pedal (PCsensor 3-key FootSwitch) must be plugged into
node 0. Keycodes ``a`` / ``b`` / ``c`` are burned into firmware
once with the vendor's Windows tool.


Install (run on each node)
--------------------------

1. Check the Franka firmware version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the Franka Desk web UI (typically ``http://<robot_ip>/desk``),
click the ``SETTINGS`` tab and read the version number next to
``Control`` under ``DashBoard``, as shown below. Note this firmware
version — later steps use it.

.. raw:: html

  <div style="flex: 1; text-align: center;">
      <img src="https://github.com/RLinf/misc/blob/main/pic/franka_firmware.png?raw=true" style="width: 60%;"/>
  </div>

Then look up the matching libfranka version in Franka's official
`compatibility matrix
<https://frankarobotics.github.io/docs/compatibility.html>`_ — the
"RLinf + franky" section below will need it.

2. PREEMPT_RT kernel and rtprio limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Boot a PREEMPT_RT kernel per `Setting up the real-time kernel
<https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html>`_
(validated on ``5.15.133-rt69``). Verify:

.. code-block:: bash

   uname -a | grep -o PREEMPT_RT

Drop the following into ``/etc/security/limits.d/99-realtime.conf``
and log out + back in:

.. code-block:: text

   *  -  rtprio    99
   *  -  memlock   unlimited

Log out and back in to let PAM re-read the limits; ``ulimit -r``
must then return ``99`` (or ``unlimited``) and ``ulimit -l`` must
return ``unlimited``. Without these,
``FrankyController.__init__`` logs ``SCHED_FIFO denied`` /
``mlockall failed`` and falls back to default scheduling — the
controller still runs, but RT jitter returns.

.. note::

   These limits are checked at startup by
   ``_apply_rt_hardening()`` in
   ``rlinf/envs/realworld/franka/franky_controller.py``. If
   ``SCHED_FIFO`` is denied or ``mlockall`` fails, the controller
   continues in best-effort mode and emits a warning rather than
   aborting; see the warning text for the exact remediation.

3. Per-boot RT tuning
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done'
   sudo sysctl -w kernel.sched_rt_runtime_us=-1
   sudo ethtool -C eno1 rx-usecs 0 tx-usecs 0   # replace eno1 with your NIC

4. RLinf + franky
~~~~~~~~~~~~~~~~~

Export ``LIBFRANKA_VERSION`` to the libfranka version determined in
"1. Check the Franka firmware version", then run the installer:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   # Set this to the libfranka version determined in "1. Check the
   # Franka firmware version".
   export LIBFRANKA_VERSION=0.15.0       # or 0.19.0, ...

   # One command does it all: system deps (rt-tests, ethtool, eigen,
   # pinocchio, ... — install.sh invokes franky_install.sh internally,
   # which needs sudo) + RLinf Python deps + the franky-control wheel
   # matching LIBFRANKA_VERSION.  Non-root users get a sudo password
   # prompt mid-install.
   bash requirements/install.sh embodied --env franka-franky --use-mirror
   source .venv/bin/activate

The ``--env franka-franky`` target pins the franky path — it pulls
the ``franky-control`` wheel from the
``Brunch-Life/franky`` fork's ``wheels-libfranka-<LIBFRANKA_VERSION>``
release, picking the wheel for the active Python ABI (cp39..cp314,
x86_64 manylinux_2_28, **libfranka is bundled inside the wheel**), and
**skips** the legacy ``serl_franka_controllers`` ROS / catkin build
used by :doc:`franka`. ``--use-mirror`` is for mainland China users
(switches PyPI / GitHub / HuggingFace mirrors).

.. note::

   ``requirements/install.sh embodied --env franka-franky`` is a
   **one-command install**: uv venv → invokes ``franky_install.sh``
   for system deps (``rt-tests``, ``ethtool``, ``cmake``,
   ``libeigen3-dev``, ``libpoco-dev``, ``libfmt-dev``, pinocchio, ...)
   → pulls the libfranka-matched ``franky-control`` wheel. **No need
   to run** ``franky_install.sh`` standalone.

.. warning::

   **Avoid libfranka 0.18.0 specifically.** Franka's official 0.18.0
   release notes flag a regression in the impedance / Cartesian
   control path.

5. GELLO (env-worker node)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both GELLO USB-FTDI cables plug into the env-worker node (**node 0**
in the shipped placement) and stay there during data collection.
``DualGelloJointIntervention`` opens both serial ports from inside
the env-worker process and reads them at ~1 kHz — routing through
the LAN to a GELLO physically wired to node 1 would blow the
real-time budget, drop samples, and cause tracker reference jumps.

For the actual install commands (``gello`` + ``gello-teleop`` +
USB-FTDI permission, with the rationale for why only the
``DynamixelSDK`` submodule is initialised), see :doc:`franka_gello`.
Run those commands on **node 0 only**, in the same venv as RLinf —
``DualGelloJointIntervention`` imports both packages in-process when
the env wrapper stack is built.

6. Foot pedal
~~~~~~~~~~~~~

4. GELLO (node 0 only)
~~~~~~~~~~~~~~~~~~~~~~

Both GELLO USB-FTDI cables plug into node 0; routing over the LAN
breaks the 1 kHz real-time budget.

Install ``gello`` + ``gello-teleop`` + USB-FTDI permissions into
the same venv on node 0 per :doc:`franka_gello`.

5. Foot pedal (node 0 only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the vendor's Windows tool once to burn keycodes ``a`` / ``b`` /
``c`` into the PCsensor FootSwitch firmware (persists across boots).

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd       # expect: usb-PCsensor_FootSwitch-event-kbd → ../eventXX
   sudo chmod 666 /dev/input/eventXX
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX   # must be set before `ray start`


Hardware checks
---------------

Single-test each device before starting Ray.

Cameras
~~~~~~~

.. code-block:: bash

   rs-enumerate-devices | grep -E "Name|Serial|USB Type"   # RealSense
   ls /dev/v4l/by-id/                                       # both Lumos nodes
   lsusb -t                                                 # expect 5000M; 480M means USB-2 fallback

GELLO (find port + verify joints)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each GELLO maps to ``/dev/serial/by-id/usb-FTDI_..._<unique_id>-if00-port0``.
Identify left vs. right by plug-pull:

.. code-block:: bash

   # plug left only → list; then plug right → the new entry is right
   ls /dev/serial/by-id/ | grep -i ftdi

Put the two by-id paths into ``env.eval.left_gello_port`` /
``right_gello_port`` of
``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``.

Live joint readout:

.. code-block:: bash

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   python -m gello_teleop.gello_expert \
       --port /dev/serial/by-id/usb-FTDI_..._<LEFT_ID>-if00-port0

If values stall or jump by ±2π, run ``calibrate`` in the next
section. Repeat for the right arm.

Each arm in isolation
~~~~~~~~~~~~~~~~~~~~~

On each node, run against its own arm:

.. code-block:: bash

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   FRANKA_ROBOT_IP=172.16.0.2 \
   FRANKA_GRIPPER_TYPE=robotiq \
   FRANKA_GRIPPER_PORT=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<id>-if00-port0 \
       python toolkits/realworld_check/test_franky_controller.py

Key REPL commands: ``getjoint`` / ``home`` / ``hold 30`` (listen
for hum at rest) / ``stream 4 0.001 500`` (1 kHz preemption stress
test) / ``open`` / ``close``.

Do not start Ray until both arms pass.


GELLO calibration
-----------------

Each GELLO unit needs to be calibrated once (re-calibrate after
replacing a motor). Calibration results are identical against either
arm, so **calibrate both units against the Franka attached directly
to node 0 (the left arm).**

For each GELLO, run "calibrate → align-sequential verify" once.
After the first unit passes both steps, point ``GELLO_PORT`` at the
second unit's by-id path and run the same two steps again.

1. **Calibrate**:

   .. code-block:: bash

      export PYTHONPATH=$PWD:${PYTHONPATH:-}
      export GELLO_PORT=/dev/serial/by-id/usb-FTDI_..._<ID>-if00-port0
      python toolkits/realworld_check/test_gello.py calibrate

   The script moves the robot to two known poses (``POSE_A`` =
   Franka home, ``POSE_B`` = π/4 multiples), prompts you to physically
   match the GELLO leader to each pose, then solves
   ``joint_signs`` and ``joint_offsets`` from the difference. Output
   is a paste-ready ``DynamixelRobotConfig`` block to drop into
   ``gello_software/gello/agents/gello_agent.py``::

       "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_<id>-if00-port0":
           DynamixelRobotConfig(
               joint_ids=(1, 2, 3, 4, 5, 6, 7),
               joint_offsets=(...),
               joint_signs=(...),
               gripper_config=(8, ..., ...),
           ),

   python toolkits/realworld_check/test_gello.py calibrate

2. **Verify with align-sequential**: immediately after pasting, run
   align-sequential through J1 → J7 to confirm every joint settles
   inside ±0.10 rad cleanly. If a joint never converges or the residual
   is larger than expected, go back to step 1 and re-calibrate.

**Align** (run when the leader and arm pose disagree):

      export PYTHONPATH=$PWD:${PYTHONPATH:-}
      python toolkits/realworld_check/test_gello.py align-sequential

   python toolkits/realworld_check/test_gello.py align-sequential

Both scripts auto-discover the local Robotiq port by globbing
``/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_*-if00-port0`` (on node 0
that resolves to the left arm's Robotiq), so no manual config needed.


Hardware YAML
-------------

Hardware config lives in
``examples/embodiment/config/env/realworld_dual_franka_joint.yaml``
(collection) and
``examples/embodiment/config/env/realworld_dual_franka_rot6d.yaml``
(rot6d deployment). Per-host placeholders:

* ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP`` — FCI IP of each arm
  (e.g. ``172.16.0.2``).
* ``BASE_CAMERA_SERIAL`` — base camera serial (RealSense reports
  it via ``rs.context().devices``; otherwise use the SDK serial
  matching ``base_camera_type``).
* ``LEFT_CAMERA_SERIAL`` / ``RIGHT_CAMERA_SERIAL`` — wrist camera
  serials (Lumos: the
  ``/dev/v4l/by-id/usb-XVisio_..._video-index0`` path; otherwise
  the SDK serial matching ``*_camera_type``).
* ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION`` —
  Robotiq 2F-85 RS-485 port, always
  ``/dev/serial/by-id/usb-FTDI_..._<id>-if00-port0``. **Never use**
  ``/dev/ttyUSB*`` (renumbered on reboot or hot-plug).
* ``LEFT_GELLO_PORT`` / ``RIGHT_GELLO_PORT`` — GELLO leader
  ``/dev/serial/by-id`` paths (both plug into the env-worker node,
  i.e. ``node_rank: 0``).
* ``ee_pose_limit_min`` / ``ee_pose_limit_max`` in the override
  block — tune to your workspace safety box; row 0 is left, row 1
  is right; each row is ``[x, y, z, roll, pitch, yaw]``.

``left_controller_node_rank`` / ``right_controller_node_rank``
(default ``0`` / ``1``, one arm per node) and ``node_rank``
(env worker + cameras) usually do not need to change.


Ray cluster bring-up
--------------------

Ray snapshots the exported environment at ``ray start`` time;
anything not exported is invisible to workers. Export first, then
start Ray.

.. code-block:: bash

   # node 0 (head)
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

   ray stop --force
   ray start --head --port=6379 --node-ip-address=<HEAD_IP>

.. code-block:: bash

   # node 1 (worker)
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=1
   ray stop --force
   ray start --address=<HEAD_IP>:6379 --node-ip-address=<WORKER_IP>

On node 0 verify both nodes are ALIVE with ``ray status``.

.. warning::

   The two nodes are independent checkouts. After any code change
   on node 0, run
   ``rsync -av --delete RLinf/ <node1>:/path/to/RLinf/`` and
   restart Ray on node 1. Otherwise expect worker ImportErrors or
   inconsistent behavior.


Data collection (GELLO joint-space)
-----------------------------------

env = ``DualFrankaJointEnv-v1`` with ``teleop_direct_stream: true``,
which spawns a 1 kHz daemon that streams GELLO readings straight to
the ``FrankyController`` actors. ``env.step`` runs at 10 Hz and only
reads state + grabs frames — it does not send motion. As a result
the dataset captures real 1 kHz operator motion instead of motion
clipped to a 100 ms grid.

Configuration
~~~~~~~~~~~~~

``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``,
fields you typically touch:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Field
     - Meaning
   * - ``runner.num_data_episodes``
     - Total target (accumulated across resumed sessions).
   * - ``env.eval.left_gello_port`` / ``right_gello_port``
     - Override per session if swapping GELLO units.
   * - ``env.eval.override_cfg.task_description``
     - Prompt written into each frame's ``task`` field.
   * - ``env.eval.override_cfg.joint_action_mode``
     - ``absolute`` for collection.
   * - ``env.eval.override_cfg.teleop_direct_stream``
     - Must be ``true``.
   * - ``data_collection.save_dir``
     - Dataset root; point multiple sessions at the same dir to
       accumulate.
   * - ``data_collection.resume``
     - ``true`` resumes from existing ``id_*`` shards.

Launch
~~~~~~

After Ray is up, open two terminals (both on node 0):

.. code-block:: bash

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   bash examples/embodiment/collect_data.sh \
        realworld_collect_data_gello_joint_dual_franka 2>&1 | tee logs/collect.log

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   python toolkits/realworld_check/collect_monitor.py logs/collect.log

The monitor exists because Ray's log monitor buffers stdout and
breaks tqdm's in-place refresh; it tails the log on its own and
renders a clean progress bar with pedal events and the latest
reward.

Per-episode workflow
~~~~~~~~~~~~~~~~~~~~

After ``align-sequential`` reports ``ALL JOINTS ALIGNED``:

1. (pre) reset skips home; the arm stays at the operator's current
   pose.
2. Step on ``a`` — start recording at frame 0.
3. Demonstrate. The arm tracks GELLO at 1 kHz; cameras grab at
   10 Hz.
4. Step on ``b`` — ``segment_id`` +1 (1 s debounce); marks
   approach / grasp / ... boundaries.
5. Step on ``c`` — success: ``reward=1.0``, writes the LeRobot
   shard.
6. Step on ``a`` mid-recording — abort: buffer dropped, return to
   pre; the arm does not home.

Output format
~~~~~~~~~~~~~

LeRobot v2.1, one shard per session at
``<save_dir>/rank_0/id_{N}/``. ``meta/info.json`` has
``state=[68]`` and ``actions=[16]`` (joint) or ``[20]`` (rot6d).

Key per-frame fields:

* ``state`` —
  ``[L_grip, R_grip, joint_position(14), joint_velocity(14),
  tcp_force(6), tcp_pose(14), tcp_torque(6), tcp_vel(12)]`` = 68
* ``actions`` — joint mode:
  ``[L_jpos(7), L_grip, R_jpos(7), R_grip]``
* ``image`` — ``left_wrist_0_rgb`` (main image)
* ``extra_view_image-0`` / ``-1`` — **order is locked** to
  ``(base_0_rgb, right_wrist_0_rgb)``; renaming triggers an
  assertion.
* ``is_success`` — ``True`` for the whole episode iff the episode
  ended by stepping on ``c``.
* ``segment_id`` — uint8, incremented when ``b`` is pressed.

Resume
~~~~~~

``data_collection.resume: true`` with the same ``save_dir``
re-runs: existing ``id_*`` shards are scanned, the new session
writes into a fresh ``id_{N}``. ``num_data_episodes`` is the
accumulated cross-session target.


Backfill rot6d and norm_stats
-----------------------------

Collection produces 16-D joint actions and 68-D state; π₀.₅ SFT
needs 20-D rot6d (``[xyz(3) + rot6d(6) + grip(1)] × 2``). Backfill
offline first, then compute norm_stats.

``<repo_id>`` is the dataset path relative to ``HF_LEROBOT_HOME``;
``joint_v1`` / ``rot6d_v1`` are version subdirs.

.. code-block:: bash

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   python toolkits/dual_franka/backfill_rot6d.py \
       --src $HF_LEROBOT_HOME/<repo_id>/joint_v1 \
       --dst $HF_LEROBOT_HOME/<repo_id>/rot6d_v1

The script rewrites the first 20 state slots into rot6d (sliced
from the existing tcp_pose columns, no FK), widens actions to 20-D
with xyz/rot6d taken from the **next frame's** tcp_pose as the
current target (gripper slots unchanged), and updates the schema.
Re-backfilling an already-converted dataset errors out.

.. code-block:: bash

   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_dualfranka_rot6d \
       --repo-id <repo_id>/rot6d_v1

Writes
``<openpi_assets_dirs>/pi05_dualfranka_rot6d/<repo_id>/norm_stats.json``.
**Compute after backfill**, otherwise the stats reflect absolute
targets rather than body-frame deltas.


SFT (π₀.₅, rot6d_v1)
--------------------

SFT runs on a remote GPU training cluster, not on node 0 / node 1.
Push the rot6d dataset to the cluster, train there, pull ckpt +
norm_stats back to node 0 for deployment.

1. Push dataset (node 0):

   .. code-block:: bash

      rsync -av $HF_LEROBOT_HOME/<repo_id>/rot6d_v1/ \
          <train>:$HF_LEROBOT_HOME/<repo_id>/rot6d_v1/

2. Install + train (on ``<train>``). Set
   ``runner.logger.wandb_entity`` and
   ``cluster.component_placement`` in
   ``examples/sft/config/realworld_sft_openpi_dual_franka_rot6d.yaml``
   first.

   .. code-block:: bash

      # --env is required by install.sh; not actually used.
      bash requirements/install.sh embodied --model openpi --env maniskill_libero --use-mirror
      source .venv/bin/activate

      export PYTHONPATH=$PWD:${PYTHONPATH:-}
      export HF_LEROBOT_HOME=/path/to/lerobot_root
      export DUAL_FRANKA_DATA_ROOT=$HF_LEROBOT_HOME/<repo_id>/rot6d_v1
      export PI05_BASE_CKPT=/path/to/pi05/torch

      python toolkits/lerobot/calculate_norm_stats.py \
          --config-name pi05_dualfranka_rot6d \
          --repo-id <repo_id>/rot6d_v1
      mkdir -p $PI05_BASE_CKPT/<repo_id>
      mv <openpi_assets_dirs>/pi05_dualfranka_rot6d/<repo_id>/norm_stats.json \
         $PI05_BASE_CKPT/<repo_id>/norm_stats.json

      bash examples/sft/run_vla_sft.sh realworld_sft_openpi_dual_franka_rot6d

   Ckpts land at
   ``<log_path>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``.

3. Pull ckpt + norm_stats back to node 0:

   .. code-block:: bash

      CKPT=<train_log>/checkpoints/global_step_<N>
      mkdir -p $CKPT/actor/model_state_dict $CKPT/<repo_id>
      rsync -av <train>:$CKPT/actor/model_state_dict/full_weights.pt \
          $CKPT/actor/model_state_dict/full_weights.pt
      rsync -av <train>:$PI05_BASE_CKPT/<repo_id>/norm_stats.json \
          $CKPT/<repo_id>/norm_stats.json

   Deployment reads
   ``$CKPT/actor/model_state_dict/full_weights.pt`` and
   ``$CKPT/<repo_id>/norm_stats.json``.

Real-world deployment
---------------------

Same Ray cluster as collection; different entry script and config.

Install
~~~~~~~

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env franka-franky --use-mirror
   source .venv/bin/activate

Configuration
~~~~~~~~~~~~~

``examples/embodiment/config/realworld_eval_dual_franka.yaml``.
Placeholders are flagged with ``# Replace:``. Most-edited fields:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Set to
   * - ``rollout.model.model_path``
     - ``<sft_log>/checkpoints/global_step_<N>/`` — must contain
       ``actor/model_state_dict/full_weights.pt`` and
       ``<data_config.repo_id>/norm_stats.json`` (see
       "ckpt / norm_stats lock-step" below).
   * - ``actor.model.openpi_data.repo_id``
     - Passed as ``data_kwargs`` to ``get_openpi_config``, overrides
       ``data_config.repo_id``; this is also the key used to find
       ``norm_stats.json`` at deployment. Must match
       ``calculate_norm_stats.py --repo-id``.
   * - ``env.eval.override_cfg.task_description``
     - Same as the SFT training prompt.
   * - ``env.eval.override_cfg.target_ee_pose``
     - Aligned with the collection workspace.

Hardware ``configs`` are identical to the collection YAML — same
IPs, camera serials, gripper ports. Wrappers are mounted by
``env.eval.use_*`` flags, so the only differences between
collection and deployment YAMLs are:

* ``use_gello_joint: false`` (collection: ``true``)
* ``keyboard_reward_wrapper: eval_control`` (collection:
  ``start_end``)

Launch
~~~~~~

.. code-block:: bash

   # node 0: sync code to node 1 first
   rsync -av --delete --exclude=results --exclude='.venv*' --exclude=.git \
       --exclude=__pycache__ --exclude='*.pyc' --exclude=wandb \
       ./ <node1>:/path/to/RLinf/

.. code-block:: bash

   # node 0 (head)
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

   ray stop --force
   ray start --head --port=6379 --node-ip-address=<HEAD_IP>

.. code-block:: bash

   # node 1 (worker)
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=1
   ray stop --force
   ray start --address=<HEAD_IP>:6379 --node-ip-address=<WORKER_IP>

.. code-block:: bash

   # node 0
   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka

   # Hydra override example:
   #   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka \
   #        rollout.model.model_path=/sft/global_step_5000 \
   #        env.eval.override_cfg.task_description="pour water"

Per-episode deployment workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``KeyboardEvalControlWrapper`` switches the foot-pedal wrapper
into autonomous inference mode:

1. After ``env.reset()`` both arms hold the reset pose.
   ``env.step()`` is intercepted into **idle** — calls are not
   forwarded to the inner env (the impedance controller keeps the
   last reset target, arm stays still), but the wrapper still
   returns the most recent observation so the policy's chunked
   rollout loop spins without issuing joint commands.
2. Step on ``a`` — wrapper flips to **running**. The next
   ``env.step`` starts forwarding policy outputs.
3. Step on ``c`` — success: ``terminated=True``, ``reward=1.0``,
   ``info["eval_result"]="success"``. The wrapper immediately
   calls ``env.reset()`` to home the arms, then returns to idle to
   wait for the next ``a``. This is the key to continuous pedal
   operation even when the eval ``env_worker`` is
   ``auto_reset=False``.
4. Step on ``b`` — failure: same flow as ``c`` but ``reward=0.0``
   and ``info["eval_result"]="failure"``.
5. During running the wrapper forces ``terminated`` /
   ``truncated`` to False unless the pedal fires; the env's own
   ``max_episode_steps`` does not cut the policy off. Set
   ``max_episode_steps`` large (the shipped YAML uses ``10000``)
   so the pedal owns the boundary.

ckpt / norm_stats lock-step
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rollout reads
``<rollout.model.model_path>/<actor.model.openpi_data.repo_id>/norm_stats.json``.
``repo_id`` must match the value used at SFT time with
``calculate_norm_stats.py --repo-id``; otherwise it falls back and
emits the warning
``"norm_stats fallback: ... verify they match training or
inference will be wrong"``. A mismatch does not crash but collapses
the policy to a fixed action.

Preflight checks:

.. code-block:: bash

   find <model_path> -maxdepth 3 -name norm_stats.json
   ls <model_path>/actor/model_state_dict/full_weights.pt


Troubleshooting
---------------

**GELLO daemon does not start**
   Power-cycle the GELLO, replug the FTDI, and verify with
   ``python -m gello_teleop.gello_expert --port /dev/...`` that
   both sides stream Dynamixel readings.

**Ray worker dies silently on import**
   In the shell that ran ``ray start``, check
   ``which python && python -c "import franky, gello, gello_teleop"``
   to confirm the venv and installed packages match. Worker
   tracebacks live in ``/tmp/ray/session_latest/logs/worker-*.err``.

**One arm hangs on reset**
   On the controller node run ``ping -c 100 <robot_ip>``; if there
   is packet loss, power-cycle the arm.

**``move_joints`` errors immediately after boot**
   Release the white E-stop → open the Desk page at
   ``http://<robot_ip>/desk/`` → click *Activate FCI* → wait for
   joint LEDs to turn from white to blue → launch.

**GELLO daemon races with env reset**
   Hold the GELLO leader on its stand during reset; wait for
   ``KeyboardStartEndWrapper`` to report reset complete before
   operating again.

**Foot pedal "Permission denied"**
   ``sudo chmod 666 /dev/input/eventXX``; for persistence add a
   udev rule
   (``KERNEL=="event*", SUBSYSTEM=="input",
   ATTRS{name}=="PCsensor FootSwitch", MODE="0666"``).

**RealSense drops to USB 2.x**
   Swap the cable; plug into a blue USB-3 port on the motherboard;
   ``lsusb -t`` should show ``5000M``, not ``480M``.

**Lumos fails on first cold boot**
   Replug the USB cable.

**Deployment stuck in idle**
   Verify ``RLINF_KEYBOARD_DEVICE`` points at the right
   ``/dev/input/eventXX`` and ``chmod 666`` still applies; then
   step on ``a``.

**Deployment tracking jitter**
   Lower ``RLINF_CART_K_R``, raise ``RLINF_CART_GAINS_TC``, tighten
   ``RLINF_CART_MAX_STEP_RAD``. If that is not enough, shorten the
   policy chunk length.

**Deployment cannot find ``norm_stats.json``**
   Copy
   ``<openpi_assets_dirs>/<repo_id>/`` written by
   ``calculate_norm_stats.py`` into ``<model_path>/<repo_id>/``;
   grep for the ``"norm_stats fallback"`` warning to confirm
   whether the fallback path was taken.

**collect_monitor shows no progress**
   Make sure the launcher pipes through ``2>&1 | tee
   logs/collect.log``. If the env worker is on a different node,
   pass ``--source=worker``.

**Controller emits ``sched_setaffinity failed`` warning at startup**
   Use a 6+ core host, or run
   ``sudo setcap cap_sys_nice=eip $(which python)`` against the
   venv interpreter.

**Both arms move on reset but only one tracks GELLO afterwards**
   For each GELLO run
   ``python toolkits/realworld_check/test_gello.py align-check``
   to verify both keep streaming, then restart.
