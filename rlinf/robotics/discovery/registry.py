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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Optional, TypeVar

from rlinf.scheduler.hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)

from ..robot import Robot

RobotConfigType = TypeVar("RobotConfigType", bound="RobotConfig", covariant=True)


@dataclass
class RobotConfig(HardwareConfig):
    """Base physical configuration for a registered robot.

    What varies between robots at enumeration time is what their *config*
    says, so the two hooks below live here rather than on a discovery class.
    Most robots set neither.

    How many robots a node carries is not one of them: it follows from the
    fields themselves, since a field holding one value per robot is a scalar
    and a list field belongs to a single robot. See
    :class:`~rlinf.robotics.discovery.RobotAutoConfig`.
    """

    #: Whether a node carrying this robot must have at least one camera.
    REQUIRES_CAMERA: ClassVar[bool] = False

    def model(self, robot_type: str) -> str:
        """The model to report, when the type name is not specific enough.

        An arm that comes in two reaches is one robot type and two models, and
        the config is what knows which one this is.
        """
        return robot_type


@dataclass
class RobotInfo(HardwareInfo, Generic[RobotConfigType]):
    """Scheduler resource information for a registered robot."""

    config: RobotConfigType


@dataclass(frozen=True)
class RobotRegistration:
    """Everything one registered robot type contributes."""

    robot_cls: type[Robot]
    config_cls: type[RobotConfig]
    discovery_cls: type["RobotDiscovery"]
    build: Optional[Callable[..., Robot]] = None
    """Builder that composes the robot's deferred part declarations."""


class RobotDiscovery(Hardware):
    """Scheduler-facing discovery policy kept separate from robot composition.

    Finding a robot on a node is the same procedure whatever the robot is:
    take the configs of this robot's type that name this node, fill in what the
    node's environment can answer, check what the parts report, and describe
    what came out. :meth:`enumerate` does that once, here.

    No robot needs a subclass of this. What differs between robots is what
    their config says -- :attr:`RobotConfig.REQUIRES_CAMERA` and
    :meth:`RobotConfig.model` -- and what their parts check for themselves.
    :meth:`~rlinf.robotics.robot.Robot.register_type` makes one of these per
    robot type when none is given.
    """

    registry: ClassVar[dict[str, RobotRegistration]] = {}

    @classmethod
    def config_cls(cls) -> type[RobotConfig]:
        """The config class registered for this robot type."""
        registration = cls.registry.get(cls.HW_TYPE)
        if registration is None:
            raise KeyError(
                f"{cls.__name__} is not registered under {cls.HW_TYPE!r}; "
                "register_robot() is what pairs a discovery with its config."
            )
        return registration.config_cls

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
    ) -> Optional[HardwareResource]:
        """Describe the robots of this type attached to one node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: Every hardware config written for this node, of any type.

        Returns:
            What this node carries, or ``None`` when it carries none of this
            robot.
        """
        assert configs is not None, (
            f"{cls.HW_TYPE} hardware requires explicit configurations; "
            "enumeration reads them rather than probing for the robot."
        )
        # Imported here because autoconfig needs RobotConfig from this module.
        from .autoconfig import RobotAutoConfig

        config_cls = cls.config_cls()
        mine = [
            config
            for config in configs
            if isinstance(config, config_cls) and config.node_rank == node_rank
        ]
        # Fill unset fields from same-named env vars, one value per robot when
        # a node carries several. With nothing written in YAML, the env vars
        # are also what say how many robots to create.
        mine = RobotAutoConfig.resolve(mine, config_cls=config_cls, node_rank=node_rank)
        if not mine:
            return None

        infos: list[HardwareInfo] = []
        for config in mine:
            cls.prepare(config, node_rank)
            infos.append(
                RobotInfo(
                    type=cls.HW_TYPE,
                    model=config.model(cls.HW_TYPE),
                    config=config,
                )
            )
            if not getattr(config, "disable_validate", False):
                cls.validate(config, node_rank)
        return HardwareResource(type=cls.HW_TYPE, infos=infos)

    @classmethod
    def prepare(cls, config: RobotConfig, node_rank: int) -> None:
        """Fill in what only the node holding the hardware can answer.

        A config that names no camera means "use whatever is plugged in", and
        this node is the only one that can say what that is. Runs before the
        config is described, and whatever ``disable_validate`` says, because
        filling a blank is not a check.
        """
        if getattr(config, "camera_serials", ...) is None:
            camera_type = getattr(config, "camera_type", None) or "realsense"
            config.camera_serials = sorted(cls.enumerate_cameras(camera_type))

    @classmethod
    def validate(cls, config: RobotConfig, node_rank: int) -> None:
        """Check that the hardware this config names is really here.

        Cameras are asked here because every robot has them and the answer
        comes from the camera driver. Anything else a robot needs checked
        belongs to the part that owns it -- an arm validates its own address,
        a CAN-bus arm its own interface -- so there is nothing per-robot left
        to write.

        Skipped when the config sets ``disable_validate``, which is how an
        offline run or a bench check against faked SDKs gets past it.
        """
        serials = getattr(config, "camera_serials", None)
        if not serials and not config.REQUIRES_CAMERA:
            return
        cls.validate_cameras(
            getattr(config, "camera_type", None) or "realsense",
            serials or (),
            node_rank,
            require_one=config.REQUIRES_CAMERA,
        )

    @classmethod
    def enumerate_cameras(cls, camera_type: str = "realsense") -> set[str]:
        """Which cameras of one backend are attached to this node.

        The answer comes from the camera driver, which is the only thing that
        knows the SDK call that gives it. A node without that SDK reports
        nothing rather than failing, so enumeration can run anywhere.
        """
        from ..parts.cameras import BaseCamera

        return BaseCamera.backend(camera_type).discover()

    @classmethod
    def validate_cameras(
        cls,
        camera_type: str,
        serials: "Iterable[str]",
        node_rank: int,
        attached: "Optional[set[str]]" = None,
        require_one: bool = True,
    ) -> None:
        """Check the SDK is installed and every named camera is really there.

        Enumeration happens on the node that holds the hardware, so this is
        where a typo in a serial number becomes an error naming the node --
        rather than a camera that opens to nothing several minutes into a run.

        Args:
            camera_type: The backend the config asked for.
            serials: The camera identifiers the config named.
            node_rank: The node being enumerated, for the error messages.
            attached: What :meth:`enumerate_cameras` already found, when the
                caller has it; enumerated here otherwise.
            require_one: Whether a robot with no camera at all is an error.
        """
        from ..parts.cameras import BaseCamera

        camera_cls = BaseCamera.backend(camera_type)
        camera_cls.require_sdk(f"node rank {node_rank}")

        present = camera_cls.discover() if attached is None else attached
        if require_one and not present:
            raise ValueError(
                f"No {camera_type} cameras are connected to node rank "
                f"{node_rank}, and this robot requires at least one."
            )
        missing = [serial for serial in serials or () if serial not in present]
        if missing:
            raise ValueError(
                f"Cameras {missing} are not connected to node rank {node_rank}. "
                f"Available {camera_type} cameras: {sorted(present)}."
            )


def build_robot(robot_type: str, **kwargs: Any) -> Robot:
    """Build a registered robot by type name.

    The spelling most call sites use. It is :meth:`Robot.of_type` under a
    module-level name, kept because composing a robot from a config string is
    the one thing a caller does without having a robot class in hand.
    """
    return Robot.of_type(robot_type, **kwargs)


def register_robot(
    config_cls: type[RobotConfig],
    robot_cls: type[Robot],
    build: Optional[Callable[..., Robot]] = None,
) -> Callable[[type[RobotDiscovery]], type[RobotDiscovery]]:
    """Register composition, config, discovery, and builder for one robot type.

    One call per robot, made from that robot's own module, so adding hardware
    does not edit a central table.
    """

    def decorator(discovery_cls: type[RobotDiscovery]) -> type[RobotDiscovery]:
        if not issubclass(discovery_cls, RobotDiscovery):
            raise TypeError(
                f"{discovery_cls.__name__} must inherit from RobotDiscovery."
            )
        if not issubclass(config_cls, RobotConfig):
            raise TypeError(f"{config_cls.__name__} must inherit from RobotConfig.")
        if not issubclass(robot_cls, Robot):
            raise TypeError(f"{robot_cls.__name__} must inherit from Robot.")

        robot_type = discovery_cls.HW_TYPE
        if not robot_type:
            raise ValueError(f"{discovery_cls.__name__}.HW_TYPE must be set.")
        if robot_cls.ROBOT_TYPE != robot_type:
            raise ValueError(
                f"{robot_cls.__name__}.ROBOT_TYPE must equal {robot_type!r}."
            )
        if (
            robot_type in RobotDiscovery.registry
            or robot_type in Hardware.hw_types
            or robot_type in NodeHardwareConfig._hardware_config_registry
        ):
            raise ValueError(f"Robot type {robot_type!r} is already registered.")

        Hardware.register()(discovery_cls)
        NodeHardwareConfig.register_hardware_config(robot_type)(config_cls)
        RobotDiscovery.registry[robot_type] = RobotRegistration(
            robot_cls=robot_cls,
            config_cls=config_cls,
            discovery_cls=discovery_cls,
            build=build,
        )
        return discovery_cls

    return decorator
