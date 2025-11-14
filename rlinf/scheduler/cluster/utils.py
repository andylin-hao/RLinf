# Copyright 2025 The RLinf Authors.
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


from typing import Callable, Optional, Protocol


class DataclassProtocol(Protocol):
    """Protocol for dataclasses to enable type checking."""

    __dataclass_fields__: dict
    __dataclass_params__: dict
    __post_init__: Optional[Callable]


def parse_rank_config(rank_config: str, available_ranks: list[int]) -> list[int]:
    """Parse a rank configuration string into a list of ranks.

    Args:
        rank_config (str): The rank configuration string, e.g., "0-3,5,7-9" or "all".
        available_ranks (list[int]): The list of available ranks.

    Returns:
        list[int]: The list of ranks.
    """
    ranks = set()
    available_ranks = sorted(available_ranks)
    if rank_config.lower() == "all":
        ranks = set(available_ranks)
    else:
        # If the GPU placement is a single number
        # Omegaconf will parse it as an integer instead of a string
        rank_config = str(rank_config)
        # First split by comma
        rank_ranges = rank_config.split(",")
        for rank_range in rank_ranges:
            rank_range = rank_range.strip()
            if rank_range == "":
                continue
            # Then split by hyphen to get the start and end of the range
            rank_range = rank_range.split("-")
            try:
                if len(rank_range) == 1:
                    start_rank = int(rank_range[0])
                    end_rank = start_rank
                elif len(rank_range) == 2:
                    start_rank = int(rank_range[0])
                    end_rank = int(rank_range[1])
                else:
                    raise ValueError
            except (ValueError, IndexError):
                raise ValueError(
                    f'Invalid rank format {rank_config}, expected format: "a,b,c-d" or "all"'
                )
            assert end_rank >= start_rank, (
                f"Start rank {start_rank} must be less than or equal to end rank {end_rank} in rank config {rank_config}."
            )
            assert available_ranks[0] <= start_rank <= available_ranks[-1], (
                f"Start rank {start_rank} in rank config {rank_config} must be within the available ranks {available_ranks}."
            )
            assert available_ranks[0] <= end_rank <= available_ranks[-1], (
                f"End rank {end_rank} in rank config {rank_config} must be within the available ranks {available_ranks}."
            )
            ranks.update(range(start_rank, end_rank + 1))
    return sorted(ranks)


def dataclass_arg_check(dataclass: DataclassProtocol, kwargs: dict):
    """Check if the kwargs contain only valid fields for the given dataclass.

    Args:
        dataclass (DataclassProtocol): The dataclass to check against.
        kwargs (dict): The keyword arguments to check.
    """
    args = set(kwargs.keys())
    valid_args = set(dataclass.__dataclass_fields__.keys())

    missing_args = valid_args - args
    unknown_args = args - valid_args

    missing_required_args = []
    for missing_arg in missing_args:
        field_info = dataclass.__dataclass_fields__[missing_arg]
        if (
            field_info.default is field_info.default_factory
            and field_info.default_factory is field_info.default_factory
        ):
            missing_required_args.append(missing_arg)

    return missing_required_args, unknown_args, valid_args
