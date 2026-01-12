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

"""CLI entry for RewardVerifier using OmegaConf YAML configuration."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from toolkits.reward_alignment import RewardVerifier, load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Reward alignment & verification")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("reward_alignment.yaml"),
        help="Path to OmegaConf YAML config",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and launch RewardVerifier."""
    args = parse_args()
    cfg = load_config(args.config)
    print(f"[config] loaded from {args.config}")
    if Path(args.config).exists():
        print(OmegaConf.to_yaml(OmegaConf.load(args.config)))
    verifier = RewardVerifier(cfg)
    verifier.run()


if __name__ == "__main__":
    main()
