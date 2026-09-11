# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Gymnasium env that runs one task on one robot through one control.

:class:`TaskEnv` is the only env a task needs: a composed
:class:`~rlinf.robotics.Robot`, a :class:`~rlinf.envs.real.tasks.Task`, and a
:class:`~rlinf.envs.real.control.Control` go in, and it owns the step loop,
episode bookkeeping, and the robot's lifecycle. The robot is passed in as it
is; nothing robot-specific lives here.

:class:`RegisteredTaskEnv` builds those three from the ``override_cfg`` and
hardware a run is given, so a Gymnasium id can name one. Each robot package
subclasses it once with the robot's control and observation layout, and each
task id is then one more line on top.
"""

import copy
import dataclasses
import queue
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.real.control import Applied, Control
from rlinf.envs.real.tasks import Parts, ResetContext, Task, bind
from rlinf.envs.real.tasks.requirements import combine
from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.envs.real.utils.frames import policy_frame
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.envs.real.utils.video import VideoPlayer
from rlinf.robotics import Arm, Robot
from rlinf.robotics.actions import ActionPart
from rlinf.robotics.discovery import RobotConfig, RobotDiscovery, RobotInfo
from rlinf.robotics.parts.cameras import CameraInfo
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger


@dataclass
class TaskEnvConfig:
    """How an episode runs, whatever the task and robot."""

    is_dummy: bool = False
    """Run without hardware, sampling observations from the space."""

    step_frequency: float = 10.0
    """Control rate in Hz. A step sleeps for the remainder of its period."""

    max_num_steps: int = 100
    """Steps before an episode is truncated."""

    reward_scale: float = 1.0
    """Multiplier on the reward, applied after the termination check."""

    enable_camera_player: bool = True
    """Show the cameras in a viewer window."""


@dataclass(frozen=True)
class StateField:
    """One entry of the policy's ``state``, and where the robot reports it.

    Attributes:
        key: Name in the observation. The policy sees state keys in sorted
            order, so a key is part of the vector's layout.
        field: Name the part reports it under.
        shape: Shape in the observation.
        role: The role whose part reports it.
        end_effector: Read it from the end effector the role's part carries.
        low: Lower bound of the space.
        high: Upper bound of the space.
    """

    key: str
    field: str
    shape: tuple[int, ...]
    role: str = "arm"
    end_effector: bool = False
    low: float = -np.inf
    high: float = np.inf


@dataclass(frozen=True)
class ObservationSpec:
    """The observation a policy reads: state fields, then camera frames."""

    state: tuple[StateField, ...]
    cameras: tuple[CameraInfo, ...] = ()
    frame_size: tuple[int, int] = (128, 128)


class TaskEnv(gym.Env):
    """Run ``task`` on ``robot``, driven through ``control``.

    Construction checks that the robot has what the task and control need,
    before connecting it, then connects it, waits for its arms, and homes it.
    A dummy env takes no robot and samples its observation space instead.

    Args:
        robot: The composed robot, connected or not; ``None`` when dummy.
        task: What the episode is for.
        control: How a policy's action reaches the robot.
        observation: What the policy reads.
        config: How episodes run.
    """

    metadata = {"render_modes": []}

    #: Teleoperation devices whose action layout fits this env.
    TELEOP: ClassVar[tuple[str, ...]] = ()

    #: Device used when a run names none.
    TELEOP_DEFAULT: ClassVar[str] = "none"

    #: Whole-robot reads tried before a stalled camera ends the step.
    READ_ATTEMPTS: ClassVar[int] = 3

    #: Seconds between those reads, for a camera to come back.
    STALL_WAIT_S: ClassVar[float] = 5.0

    def __init__(
        self,
        robot: Optional[Robot],
        task: Task,
        control: Control,
        *,
        observation: ObservationSpec,
        config: Optional[TaskEnvConfig] = None,
    ) -> None:
        self.config = config if config is not None else TaskEnvConfig()
        if robot is None and not self.config.is_dummy:
            raise ValueError("A TaskEnv needs a robot unless it is dummy.")
        self._logger = get_logger()
        self.robot = robot
        self.task = task
        self.control = control
        self.observation = observation
        task.validate(control.dof())

        self.action_space = control.action_space()
        self.observation_space = self._observation_space()
        # A wrapper may rewrite the space it is handed in place; a dummy env
        # samples this private copy so its observations keep the raw layout.
        self._sample_space = copy.deepcopy(self.observation_space)
        self._num_steps = 0
        self._hold = 0
        self._reading: Optional[Mapping[str, Any]] = None
        self.parts: Optional[Parts] = None
        self.camera_player: Optional[VideoPlayer] = None
        if self.config.is_dummy:
            return

        self.parts = bind(
            robot,
            combine(task.requirements(), control.requirements()),
            owner=type(task).__name__,
        )
        if not robot.is_connected:
            robot.connect()
        self._wait_for_arms()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)
        task.home(self.parts, self._context())

    # Protocol the wrapper stack and teleop read off the unwrapped env.

    @property
    def ACTION_WRAPPERS(self) -> tuple[str, ...]:  # noqa: N802 - wrapper protocol
        """Action wrappers that fit the control's action layout."""
        return self.control.ACTION_WRAPPERS

    @property
    def TRANSFORMS(self) -> tuple[str, ...]:  # noqa: N802 - wrapper protocol
        """Observation and action transforms that fit it."""
        return self.control.TRANSFORMS

    @property
    def task_description(self) -> str:
        """Language instruction for policies that condition on one."""
        return self.task.description

    @property
    def num_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._num_steps

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Named slices of the action, for teleop and action wrappers."""
        return self.control.action_parts()

    def get_joint_positions(self) -> np.ndarray:
        """Joints of each driven arm as ``(arms, dof)``, zeros before a read."""
        rows = []
        for role, dof in self.control.dof().items():
            joints = None
            if self._reading is not None and self.parts is not None:
                joints = (
                    self.parts.read(self._reading).arm(role).get("arm_joint_position")
                )
            rows.append(np.zeros(dof or 0) if joints is None else np.asarray(joints))
        return np.stack(rows).astype(float)

    # Gymnasium API.

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Put the robot and scene back and return the first observation."""
        super().reset(seed=seed)
        # A dummy env samples its space instead of reading, so seeding the
        # space is what makes such a run reproducible.
        seed_sampled_spaces(seed, self._sample_space)
        self._num_steps = 0
        self._hold = 0
        self.control.reset()
        if self.config.is_dummy:
            return self._sample_space.sample(), {}
        self.task.reset(self.parts, self._context(options))
        return self._observe(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one action, wait out the control period, and score the result."""
        start = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        applied = Applied()
        if not self.config.is_dummy:
            applied = self.control.apply(
                self.parts, action, self.parts.read(self._reading or {})
            )
        self._num_steps += 1
        time.sleep(max(0.0, 1.0 / self.config.step_frequency - (time.time() - start)))

        if self.config.is_dummy:
            observation, reward = self._sample_space.sample(), 0.0
        else:
            observation = self._observe()
            reward = self._score(applied)
        terminated = reward >= 1.0 and self._hold >= self.task.config.success_hold_steps
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward * self.config.reward_scale, terminated, truncated, {}

    def close(self) -> None:
        """Stop the viewer and disconnect the robot."""
        if self.camera_player is not None:
            self.camera_player.stop()
            self.camera_player = None
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    # Internals.

    def _observation_space(self) -> gym.spaces.Dict:
        spec = self.observation
        state = gym.spaces.Dict(
            {
                field.key: gym.spaces.Box(field.low, field.high, shape=field.shape)
                for field in spec.state
            }
        )
        spaces: dict[str, gym.Space] = {"state": state}
        # Gymnasium's env checker rejects an empty Dict space, so a robot with
        # no camera reports no 'frames' key at all rather than an empty one.
        if spec.cameras:
            spaces["frames"] = gym.spaces.Dict(
                {
                    camera.name: gym.spaces.Box(
                        0, 255, shape=(*spec.frame_size, 3), dtype=np.uint8
                    )
                    for camera in spec.cameras
                }
            )
        return gym.spaces.Dict(spaces)

    def _context(self, options: Optional[Mapping[str, Any]] = None) -> ResetContext:
        return ResetContext(
            rng=self.np_random, control=self.control, options=options or {}
        )

    def _wait_for_arms(self) -> None:
        arms = [
            part
            for part in map(self.parts.part, self.parts.roles)
            if isinstance(part, Arm)
        ]
        start = warned = time.time()
        while not all(arm.is_robot_up() for arm in arms):
            time.sleep(0.5)
            if time.time() - warned > 30:
                warned = time.time()
                self._logger.warning(
                    "Waited %.0fs for the arm to report ready.", warned - start
                )

    def _read(self) -> Mapping[str, Any]:
        """Read the whole robot once, giving a stalled camera time to return."""
        for attempt in range(1, self.READ_ATTEMPTS):
            try:
                return self.robot.get_observation()
            except queue.Empty:
                self._logger.warning(
                    "A camera stopped producing frames; reading again in %.0fs "
                    "(attempt %d of %d).",
                    self.STALL_WAIT_S,
                    attempt + 1,
                    self.READ_ATTEMPTS,
                )
                time.sleep(self.STALL_WAIT_S)
        return self.robot.get_observation()

    def _observe(self) -> dict[str, Any]:
        """Read the robot once and build the policy's observation from it."""
        self._reading = self._read()
        view = self.parts.read(self._reading)
        state = {}
        for field in self.observation.state:
            if field.end_effector:
                source = view.end_effector(field.role)
            else:
                source = view.part(field.role)
            state[field.key] = np.array(source[field.field], dtype=np.float32)
        observation: dict[str, Any] = {"state": state}
        if self.observation.cameras:
            frames, display = {}, {}
            for camera in self.observation.cameras:
                frame, cropped = policy_frame(
                    self._reading[camera.name]["frame"],
                    self.observation.frame_size,
                    camera.crop_region,
                )
                frames[camera.name] = frame
                display[camera.name] = frame[..., ::-1]
                display[f"{camera.name}_full"] = cropped
            if self.camera_player is not None:
                self.camera_player.put_frame(display)
            observation["frames"] = frames
        return observation

    def _score(self, applied: Applied) -> float:
        """The task's reward, less any gripper penalty, and the success streak."""
        evaluation = self.task.evaluate(self.parts.read(self._reading), applied)
        self._hold = self._hold + 1 if evaluation.in_zone else 0
        reward = evaluation.reward
        config = self.task.config
        if (
            config.enable_gripper_penalty
            and applied.ee_effective
            and not applied.is_hand
        ):
            reward -= config.gripper_penalty
        return reward


def split_overrides(
    settings: Mapping[str, Any], owners: Sequence[type], *, owner: str
) -> tuple[Any, ...]:
    """Build one config per owner from a flat ``override_cfg``.

    Every key belongs to exactly one owner's dataclass. A key no owner takes is
    refused, as a dataclass refuses one, so a misspelled or misplaced setting
    fails here instead of being ignored.

    Args:
        settings: The run's ``override_cfg``, with defaults merged in.
        owners: Config dataclasses, each built from the keys it declares.
        owner: Who is being configured, for error messages.

    Raises:
        TypeError: If two owners declare one key, or no owner declares one.
    """
    claimed: dict[str, type] = {}
    for config_cls in owners:
        for field in dataclasses.fields(config_cls):
            if not field.init:
                continue
            if field.name in claimed:
                raise TypeError(
                    f"{owner}: {field.name!r} is declared by both "
                    f"{claimed[field.name].__name__} and {config_cls.__name__}."
                )
            claimed[field.name] = config_cls
    unknown = sorted(set(settings) - set(claimed))
    if unknown:
        raise TypeError(
            f"{owner}.__init__() got unexpected keyword arguments {unknown}."
        )
    return tuple(
        config_cls(
            **{
                key: value
                for key, value in settings.items()
                if claimed[key] is config_cls
            }
        )
        for config_cls in owners
    )


class RegisteredTaskEnv(TaskEnv):
    """A :class:`TaskEnv` built from a run's config, as a Gymnasium id needs.

    A robot package subclasses this once, naming its robot and control and
    describing what its policy observes. A task id is then a subclass that
    names the task and any defaults it differs by::

        class PiperReachEnv(PiperEnv):
            TASK = JointReach

    These classes only configure; ``step``, ``reset`` and the observation
    belong to :class:`TaskEnv` for every robot alike.
    """

    #: The robot class, composed from its registered hardware config.
    ROBOT: ClassVar[type[Robot]]

    #: The control class; its ``CONFIG`` takes the run's control settings.
    CONTROL: ClassVar[type[Control]]

    #: The task this id runs.
    TASK: ClassVar[type[Task]]

    #: Settings a run gets unless it overrides them. Merged down the class
    #: hierarchy, so a task id adds to its robot preset's rather than
    #: replacing them.
    DEFAULTS: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        override_cfg: Optional[Mapping[str, Any]] = None,
        worker_info: Optional[WorkerInfo] = None,
        robot_info: "Optional[RobotInfo[Any]]" = None,
        env_idx: int = 0,
    ) -> None:
        cls = type(self)
        settings = {**cls.defaults(), **(override_cfg or {})}
        config, control_config, task_config = split_overrides(
            settings,
            (TaskEnvConfig, cls.CONTROL.CONFIG, cls.TASK.CONFIG),
            owner=cls.__name__,
        )
        self.robot_info = robot_info
        self.env_idx = env_idx
        self.hardware = get_hardware_config(
            cls.robot_config(), robot_info, is_dummy=config.is_dummy
        )
        cameras = tuple(cls.camera_infos(self.hardware))
        robot = None
        if not config.is_dummy:
            robot = cls.ROBOT.from_config(
                self.hardware,
                cameras={camera.name: camera for camera in cameras},
                env_idx=env_idx,
                node_rank=worker_info.cluster_node_rank if worker_info else 0,
                worker_rank=worker_info.rank if worker_info else 0,
            )
        super().__init__(
            robot,
            cls.TASK(task_config),
            cls.make_control(self.hardware, control_config),
            observation=cls.make_observation(self.hardware, cameras),
            config=config,
        )

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        """Defaults merged from the robot preset down to this task id."""
        merged: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("DEFAULTS", {}))
        return merged

    @classmethod
    def robot_config(cls) -> type[RobotConfig]:
        """The hardware config class registered for :attr:`ROBOT`."""
        return RobotDiscovery.registry[cls.ROBOT.ROBOT_TYPE].config_cls

    @classmethod
    def camera_infos(cls, hardware: RobotConfig) -> Iterable[CameraInfo]:
        """Name the hardware's cameras ``wrist_1``, ``wrist_2``, and so on."""
        camera_type = getattr(hardware, "camera_type", None) or "realsense"
        serials = getattr(hardware, "camera_serials", None) or []
        for index, serial in enumerate(serials):
            yield CameraInfo(
                name=f"wrist_{index + 1}",
                serial_number=serial,
                camera_type=camera_type,
            )

    @classmethod
    def make_control(cls, hardware: RobotConfig, config: Any) -> Control:
        """The control this robot is driven with, from its hardware and settings."""
        raise NotImplementedError(f"{cls.__name__} does not define make_control().")

    @classmethod
    def make_observation(
        cls, hardware: RobotConfig, cameras: tuple[CameraInfo, ...]
    ) -> ObservationSpec:
        """What this robot's policy observes."""
        raise NotImplementedError(f"{cls.__name__} does not define make_observation().")
