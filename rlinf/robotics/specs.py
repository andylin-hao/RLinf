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

"""Declaring a part and where it runs, without building it yet.

A part whose vendor SDK only exists on one machine cannot be constructed here
and moved there. So placement takes a *declaration* -- a class, its constructor
arguments, and a ``node_rank`` -- and builds the part on the target node.

``RobotPart.at(...)`` produces a :class:`PartSpec`. Compose specs exactly where
you would compose parts; :meth:`Robot.connect` resolves them. Nobody writes a
placement call.

    robot = FrankaRobot(
        arms={"left": Arm(FrankaROSArm.at("10.0.0.1", node_rank=1))},
        cameras={"scene": RealSenseCamera.at(info, node_rank=2)},
    )
    robot.connect()

One spec is placed once, however many times it is referenced, so a driver whose
single connection backs several components is not opened twice.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class PartSpec:
    """A part to build, and the node to build it on.

    Attributes:
        part_cls: The :class:`~.parts.base.RobotPart` subclass to construct.
        args: Positional arguments for its constructor.
        kwargs: Keyword arguments for its constructor.
        node_rank: Node to build it on. ``None`` builds it in this process.
        name: Worker-group name when placed remotely. Defaults to something
            derived from the class and node.
    """

    part_cls: type
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    node_rank: Optional[int] = None
    name: Optional[str] = None

    def subpart(self, name: str) -> "SubpartRef":
        """Refer to one subpart of this spec, resolved after placement.

        Use it when one connection backs several components::

            hardware = Turtle2Hardware.at(50, node_rank=0)
            left = Arm(hardware.subpart("left"), hardware.subpart("left_end_effector"))
        """
        return SubpartRef(self, name)

    def place(self) -> Any:
        """Build the part, here or on its node, and return its handle."""
        return self.part_cls.spawn(
            *self.args,
            node_rank=self.node_rank,
            name=self.name,
            **self.kwargs,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        where = "here" if self.node_rank is None else f"node {self.node_rank}"
        return f"<PartSpec {self.part_cls.__name__} on {where}>"


@dataclass(frozen=True)
class SubpartRef:
    """A reference to one subpart of a :class:`PartSpec`."""

    spec: PartSpec
    name: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SubpartRef {self.spec.part_cls.__name__}.{self.name}>"


class Placement:
    """Places each distinct spec once and resolves references against it.

    :meth:`Robot.connect` owns one of these. It keeps the handles so the robot
    can release them, and tears down what it already placed if a later part
    fails, so a half-built robot is never handed back.
    """

    def __init__(self) -> None:
        self._handles: dict[int, Any] = {}
        self._order: list[Any] = []

    @property
    def handles(self) -> list[Any]:
        """Handles for every spec placed so far, in placement order."""
        return list(self._order)

    def resolve(self, value: Any) -> Any:
        """Return a live part for a spec, a reference, or a part.

        Anything that is already a part passes through untouched, so composing
        local and placed parts together needs no special case.
        """
        if isinstance(value, SubpartRef):
            return self._handle_for(value.spec).subpart(value.name)
        if isinstance(value, PartSpec):
            handle = self._handle_for(value)
            subparts = handle.subparts
            # A leaf -- a camera, a gripper on its own port -- is its own part.
            if not subparts:
                return handle.part
            if len(subparts) == 1:
                return next(iter(subparts.values()))
            if "arm" in subparts:
                return subparts["arm"]
            raise ValueError(
                f"{value.part_cls.__name__} exposes {sorted(subparts)}; "
                "name the one you mean with .subpart(<name>)."
            )
        return value

    def resolve_handle(self, spec: PartSpec) -> Any:
        """Return the handle for a spec, placing it if this is the first use."""
        return self._handle_for(spec)

    def _handle_for(self, spec: PartSpec) -> Any:
        key = id(spec)
        handle = self._handles.get(key)
        if handle is None:
            handle = spec.place()
            self._handles[key] = handle
            self._order.append(handle)
        return handle

    def release(self) -> None:
        """Release every placed handle, newest first."""
        while self._order:
            handle = self._order.pop()
            handle.disconnect()
        self._handles.clear()


@dataclass
class PartConfig:
    """Configuration for one part, including the node it runs on.

    Subclass it per part kind and say which class to build and with what. The
    declaring, node defaulting, and naming are inherited, so a robot's
    ``build`` is composition and nothing else::

        @dataclass
        class CameraConfig(PartConfig):
            info: CameraInfo = None

            def part_cls(self):
                return camera_cls(self.info.camera_type)

            def part_args(self):
                return (self.info,)
    """

    node_rank: Optional[int] = None
    """Node to build this part on. ``None`` falls back to the robot's node."""

    def part_cls(self) -> type:
        """Return the part class to build. Subclasses implement this."""
        raise NotImplementedError(f"{type(self).__name__} must implement part_cls().")

    def part_args(self) -> tuple[Any, ...]:
        """Positional arguments for the part's constructor."""
        return ()

    def part_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the part's constructor."""
        return {}

    def declare(
        self,
        *,
        default_node_rank: Optional[int] = None,
        name: Optional[str] = None,
    ) -> PartSpec:
        """Declare this part, defaulting its node to the robot's."""
        node_rank = self.node_rank if self.node_rank is not None else default_node_rank
        return self.part_cls().at(
            *self.part_args(),
            node_rank=node_rank,
            name=name,
            **self.part_kwargs(),
        )


def declare_all(
    configs: "Mapping[str, PartConfig]",
    *,
    default_node_rank: Optional[int] = None,
    name: Optional[Callable[[str], str]] = None,
) -> dict[str, PartSpec]:
    """Declare a mapping of part configs, keeping their names.

    Works for arms, cameras, end effectors, or anything else: the part kind is
    the config's business, not this function's.
    """
    return {
        key: config.declare(
            default_node_rank=default_node_rank,
            name=name(key) if name is not None else None,
        )
        for key, config in configs.items()
    }
