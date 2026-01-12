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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "reward_alignment"


@dataclass
class RepoConfig:
    """Repository sync configuration."""

    path: Path = REPO_ROOT
    pull: bool = True
    remote: str = "origin"
    branch: str = "main"


@dataclass
class BaselineConfig:
    """Baseline comparison settings."""

    file: Path = REPO_ROOT / "reward_baselines.json"
    rtol: float = 0.05
    atol: float = 0.01


@dataclass
class RunDefaults:
    """Defaults applied to all suites/cases unless overridden."""

    mode: str = "verify"  # "baseline" or "verify"
    interval_minutes: float = 0.0
    runner: Optional[str] = None
    model_path: Optional[str] = None
    max_epochs: int = 1
    log_root: Path = DEFAULT_LOG_ROOT
    python_exe: Optional[str] = None
    venv: Optional[str] = None
    pre_run: list[str] = field(default_factory=list)
    skip: list[str] = field(default_factory=list)
    extra_overrides: list[str] = field(default_factory=list)
    discovery_roots: list[str] = field(
        default_factory=lambda: [
            "examples/embodiment/config",
            "examples/sft/config",
            "examples/reasoning/config/math",
            "examples/reasoning/config/vqa",
            "examples/coding_online_rl/config",
            "examples/searchr1/config",
        ]
    )


@dataclass
class ExampleConfig:
    name: str
    config_path: str
    runner: Optional[str] = None
    model_path: Optional[str] = None
    max_epochs: Optional[int] = None
    python_exe: Optional[str] = None
    venv: Optional[str] = None
    pre_run: list[str] = field(default_factory=list)
    skip: bool = False
    extra_overrides: list[str] = field(default_factory=list)


@dataclass
class TestCase:
    """Single config to run."""

    name: str
    runner: Optional[str] = None
    model_path: Optional[str] = None
    max_epochs: Optional[int] = None
    venv: Optional[str] = None
    pre_run: list[str] = field(default_factory=list)
    skip: bool = False
    extra_overrides: list[str] = field(default_factory=list)


@dataclass
class TestSuite:
    """Group of related tests sharing defaults (model/env/venv)."""

    name: str
    runner: Optional[str] = None
    model_path: Optional[str] = None
    max_epochs: Optional[int] = None
    venv: Optional[str] = None
    pre_run: list[str] = field(default_factory=list)
    skip: bool = False
    extra_overrides: list[str] = field(default_factory=list)
    cases: list[TestCase] = field(default_factory=list)


@dataclass
class RewardVerifyConfig:
    """Top-level configuration for reward verification runs."""

    repo: RepoConfig = field(default_factory=RepoConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    defaults: RunDefaults = field(default_factory=RunDefaults)
    tests: list[TestSuite] = field(default_factory=list)


def _coerce_path(value) -> Path:
    """Convert incoming config value to Path."""
    return value if isinstance(value, Path) else Path(str(value))


def load_config(path: Path) -> RewardVerifyConfig:
    """Load and normalize YAML config into dataclasses."""
    raw = OmegaConf.load(path)
    cfg = OmegaConf.to_container(raw, resolve=True)

    repo_dict = cfg.get("repo", {})
    baseline_dict = cfg.get("baseline", {})
    defaults_dict = cfg.get("defaults", {})
    tests_list = cfg.get("tests", [])

    repo = RepoConfig(
        path=_coerce_path(repo_dict.get("path", RepoConfig.path)),
        pull=repo_dict.get("pull", True),
        remote=repo_dict.get("remote", "origin"),
        branch=repo_dict.get("branch", "main"),
    )

    baseline = BaselineConfig(
        file=_coerce_path(baseline_dict.get("file", BaselineConfig.file)),
        rtol=baseline_dict.get("rtol", 0.05),
        atol=baseline_dict.get("atol", 0.01),
    )

    defaults = RunDefaults(
        mode=defaults_dict.get("mode", "verify"),
        interval_minutes=defaults_dict.get("interval_minutes", 0.0),
        runner=defaults_dict.get("runner"),
        model_path=defaults_dict.get("model_path"),
        max_epochs=defaults_dict.get("max_epochs", 1),
        log_root=_coerce_path(defaults_dict.get("log_root", DEFAULT_LOG_ROOT)),
        python_exe=defaults_dict.get("python_exe"),
        venv=defaults_dict.get("venv"),
        pre_run=defaults_dict.get("pre_run", []),
        skip=defaults_dict.get("skip", []),
        extra_overrides=defaults_dict.get("extra_overrides", []),
        discovery_roots=defaults_dict.get(
            "discovery_roots", RunDefaults().discovery_roots
        ),
    )

    tests: list[TestSuite] = []
    for suite in tests_list:
        cases_data = suite.get("cases", [])
        cases: list[TestCase] = []
        for c in cases_data:
            cases.append(
                TestCase(
                    name=c["name"],
                    runner=c.get("runner"),
                    model_path=c.get("model_path"),
                    max_epochs=c.get("max_epochs"),
                    venv=c.get("venv"),
                    pre_run=c.get("pre_run", []),
                    skip=c.get("skip", False),
                    extra_overrides=c.get("extra_overrides", []),
                )
            )
        tests.append(
            TestSuite(
                name=suite["name"],
                runner=suite.get("runner"),
                model_path=suite.get("model_path"),
                max_epochs=suite.get("max_epochs"),
                venv=suite.get("venv"),
                pre_run=suite.get("pre_run", []),
                skip=suite.get("skip", False),
                extra_overrides=suite.get("extra_overrides", []),
                cases=cases,
            )
        )

    return RewardVerifyConfig(repo=repo, baseline=baseline, defaults=defaults, tests=tests)
