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

import datetime
import os
from pathlib import Path
from typing import Any

from .config import ExampleConfig, RunDefaults
from .utils import CMDRunner, read_metrics


class ExampleRunner:
    """Run a single example config with merged overrides and collect metrics."""
    def __init__(self, repo_path: Path, defaults: RunDefaults):
        self.repo_path = repo_path
        self.defaults = defaults

    def _build_env(self) -> dict[str, str]:
        """Build minimal environment variables for a run."""
        env = os.environ.copy()
        env.setdefault("REPO_PATH", str(self.repo_path))
        env.setdefault("PYTHONPATH", f"{self.repo_path}:{env.get('PYTHONPATH', '')}")
        return env

    def _pre_run_cmds(self, example: ExampleConfig) -> list[str]:
        """Commands to run prior to launching the job."""
        return list(example.pre_run)

    def _resolve_log_dir(self, example: ExampleConfig) -> Path:
        """Log directory for this run."""
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return self.defaults.log_root / Path(example.name).stem / timestamp

    def _build_overrides(self, log_dir: Path, example: ExampleConfig) -> list[str]:
        """Compose Hydra-style overrides from config fields."""
        overrides: list[str] = []
        mapping = (
            ("runner.logger.log_path", str(log_dir)),
            ("runner.max_epochs", example.max_epochs),
            ("actor.model.model_path", example.model_path),
            ("rollout.model.model_path", example.model_path),
        )
        for key, value in mapping:
            if value is None:
                continue
            overrides.append(f"\'{key}={value}\'")

        if getattr(example, "extra_overrides", None):
            overrides.extend(example.extra_overrides)

        return overrides

    def run(self, example: ExampleConfig) -> dict[str, Any]:
        """Execute one example and return collected metrics."""
        log_dir = self._resolve_log_dir(example)
        log_dir.mkdir(parents=True, exist_ok=True)

        config_path = Path(example.config_path)
        if not config_path.is_absolute():
            config_path = (self.repo_path / config_path).resolve()
        env = self._build_env()
        cmd_runner = CMDRunner(self.repo_path, env=env)

        pre_runs = self._pre_run_cmds(example)
        for cmd in pre_runs:
            result = cmd_runner.run_cmd(cmd)
            if result.returncode != 0:
                cmd_runner.close()
                raise RuntimeError(f"Pre-run command failed: {cmd}\n{result.stdout}")

        cfg_name = config_path.stem

        overrides = self._build_overrides(log_dir, example)

        runner = example.runner
        if runner is None:
            raise ValueError("runner must be specified in defaults, suite, or case")
        runner_path = Path(runner)
        if not runner_path.is_absolute():
            runner_path = (self.repo_path / runner_path).resolve()

        if runner_path.suffix != ".sh":
            raise ValueError(f"runner must be a .sh script, got {runner_path}")
        assert example.venv is not None, "venv must be specified in defaults, suite, or case"
        activate = Path(example.venv) / "bin" / "activate"
        pre = f"source {activate}"
        cmd = f"{pre} && bash {runner_path} {cfg_name} " + " ".join(overrides)

        result = cmd_runner.run_cmd(cmd)
        cmd_runner.close()
        if result.returncode != 0:
            raise RuntimeError(f"Run failed for {example.name}: {result.stdout}")

        metrics = read_metrics(log_dir / "tensorboard")

        return {
            "config": str(config_path),
            "runner": str(runner_path),
            "log_dir": str(log_dir),
            "metrics": metrics,
        }
