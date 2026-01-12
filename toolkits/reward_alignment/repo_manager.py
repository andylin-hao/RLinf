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

from pathlib import Path

from .config import RepoConfig
from .utils import run_cmd


class RepoManager:
    """Lightweight git operations used by reward verification."""
    def __init__(self, cfg: RepoConfig):
        self.cfg = cfg

    @property
    def path(self) -> Path:
        return self.cfg.path

    def current_commit(self) -> str:
        """Return current HEAD commit SHA, or 'unknown' if git fails."""
        result = run_cmd(["git", "rev-parse", "HEAD"], cwd=self.path, capture=True)
        if result.returncode != 0:
            return "unknown"
        return result.stdout.strip()

    def pull_main(self) -> None:
        """Optional fast-forward pull of main."""
        if not self.cfg.pull:
            return
        run_cmd(["git", "fetch", self.cfg.remote, self.cfg.branch], cwd=self.path)
        branch_name = run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=self.path, capture=True
        ).stdout.strip()
        if branch_name == self.cfg.branch:
            run_cmd(
                ["git", "merge", "--ff-only", f"{self.cfg.remote}/{self.cfg.branch}"],
                cwd=self.path,
            )
