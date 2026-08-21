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

"""A robot: the outermost group of parts, and what places them."""

from typing import Any, ClassVar, Optional, TypeVar

from .parts.base import PartGroup, RobotPart

RobotPartType = TypeVar("RobotPartType", bound=RobotPart)


class Robot(PartGroup):
    """A named group of parts that owns where they run.

    A robot is a :class:`~rlinf.robotics.parts.base.PartGroup` like any other,
    with three additions: it knows its registered type, it builds itself from a
    hardware config, and :meth:`connect` opens every connection it was composed
    from -- on the machine each one belongs to::

        arm = FrankaROSArm(robot_ip, node_rank=1)
        robot = FrankaRobot(
            arm=arm.part("arm"),
            end_effector=arm.part("end_effector"),
            wrist=RealSenseCamera(info, node_rank=3),
        )
        robot.connect()

    There are no arm, camera, or base slots. Names carry the meaning, so a robot
    with a lift, a head, or a third arm needs no new concept.
    """

    ROBOT_TYPE: ClassVar[str] = ""

    @classmethod
    def build(cls, **kwargs: Any) -> "Robot":
        """Compose this robot from its hardware config.

        Only robots reached through the registry need this: it is what
        :meth:`register_type` hands to the registry, so :meth:`of_type` can
        compose a robot from its type name alone. A robot you construct
        yourself does not need it -- name the parts and connect::

            class Bench(Robot):
                ROBOT_TYPE = "Bench"


            connection = MyArm(port, node_rank=1)
            robot = Bench(
                arm=connection.part("arm"),
                gripper=connection.part("end_effector"),
                eye=MyCamera(info),
            )
            robot.connect()

        Composing does not connect.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement build(), which only robots "
            "composed from the registry by type name need. Construct "
            f"{cls.__name__}(part=..., ...) directly instead."
        )

    @classmethod
    def register_type(
        cls, config_cls: type, discovery_cls: Optional[type] = None
    ) -> type:
        """Register this robot's config, discovery, and builder in one call::

            FrankaRobot.register_type(FrankaConfig)

        Discovering a robot on a node is the same procedure for every robot,
        so ``discovery_cls`` is optional and one is made here when it is left
        out. It is named after this robot and takes its ``HW_TYPE`` from
        :attr:`ROBOT_TYPE`, which is the only thing a plain discovery class
        ever said. Pass one only to override
        :meth:`~rlinf.robotics.discovery.RobotDiscovery.enumerate` itself.

        Named for what it registers -- a robot *type* with the scheduler --
        because :meth:`~rlinf.robotics.parts.base.Connection.register` already
        means something else on every connection: putting a driver in a
        category's registry.
        """
        from .discovery import RobotDiscovery, register_robot

        if discovery_cls is None:
            discovery_cls = type(
                f"{cls.__name__}Discovery",
                (RobotDiscovery,),
                {
                    "HW_TYPE": cls.ROBOT_TYPE,
                    "__module__": cls.__module__,
                    "__doc__": f"Find {cls.ROBOT_TYPE} robots on a node.",
                },
            )
        elif not getattr(discovery_cls, "HW_TYPE", ""):
            discovery_cls.HW_TYPE = cls.ROBOT_TYPE

        return register_robot(config_cls, cls, build=cls.build)(discovery_cls)

    @classmethod
    def of_type(cls, robot_type: str, **kwargs: Any) -> "Robot":
        """Compose a registered robot by type name.

        What a config file names is a string, so this is the door from a
        deployment's YAML into a composed robot::

            robot = Robot.of_type("Franka", robot_ip="10.0.0.1", node_rank=1)

        The type has to have been registered, which happens when its module is
        imported. Importing :mod:`rlinf.robotics.robots` registers every robot
        that ships.
        """
        from .discovery import RobotDiscovery

        registration = RobotDiscovery.registry.get(robot_type)
        if registration is None:
            raise KeyError(
                f"Unknown robot type {robot_type!r}. "
                f"Registered: {sorted(RobotDiscovery.registry)}."
            )
        if registration.build is None:
            raise NotImplementedError(
                f"Robot type {robot_type!r} registered no builder."
            )
        return registration.build(**kwargs)

    def parts_of_type(self, part_type: type[RobotPartType]) -> dict[str, RobotPartType]:
        """Return every part implementing ``part_type``, by its dotted path."""
        matches: dict[str, RobotPartType] = {}

        def walk(group: PartGroup, prefix: str) -> None:
            for name, part in group.children.items():
                path = f"{prefix}{name}"
                if isinstance(part, part_type):
                    matches[path] = part
                if isinstance(part, PartGroup):
                    walk(part, f"{path}.")

        walk(self, "")
        return matches

    def describe(self) -> str:
        """What this robot is made of, where it runs, and what backs each part.

        Readable at any point, and most useful before anything is opened: a
        composed part already says which node it will run on and which
        connection it rides, so a composition can be checked without a robot
        present. The tree holds the same objects either side of ``connect``, so
        the answer does not change.

        Parts sharing a ``via`` share one connection, and are therefore opened
        once and commanded in their declared order rather than concurrently.
        """
        lines = [type(self).__name__]
        rows: list[tuple[str, RobotPart]] = []

        def walk(group: PartGroup, prefix: str) -> None:
            for name, part in group.children.items():
                path = f"{prefix}{name}"
                if isinstance(part, PartGroup):
                    walk(part, f"{path}.")
                else:
                    rows.append((path, part))

        walk(self, "")

        # Parts riding one connection share it; number the connections in the
        # order they appear so the grouping is visible at a glance.
        origins: dict[int, str] = {}
        for _, part in rows:
            owner = part.owner
            if id(owner) not in origins:
                origins[id(owner)] = f"{type(owner).__name__}#{len(origins) + 1}"

        width = max((len(path) for path, _ in rows), default=0)
        for index, (path, part) in enumerate(rows):
            branch = "└──" if index == len(rows) - 1 else "├──"
            owner = part.owner
            where = "here" if owner.node_rank is None else str(owner.node_rank)
            lines.append(
                f"{branch} {path:<{width}}  {type(part).__name__:<20} "
                f"node={where:<5} via {origins[id(owner)]}"
            )
        return "\n".join(lines)

    @property
    def named_parts(self) -> dict[str, RobotPart]:
        """Every part keyed by its dotted path."""
        return self.parts_of_type(RobotPart)
