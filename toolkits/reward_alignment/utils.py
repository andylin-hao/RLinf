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

import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REWARD_TAG_CANDIDATES = [
    "env/reward",
    "env/return",
    "rollout/rewards",
    "train/rewards",
    "reward",
]
SUCCESS_TAG_CANDIDATES = [
    "env/success_once",
    "eval/success_once",
    "env/success_at_end",
]


def run_cmd(
    cmd: list[str] | str,
    cwd: Path,
    env: Optional[dict[str, str]] = None,
    capture: bool = False,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and return the completed process."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=capture,
        check=False,
        shell=shell,
    )
    return result


class CMDRunner:
    """Maintain a persistent bash session to run multiple commands with shared env."""

    def __init__(self, cwd: Path, env: Optional[dict[str, str]] = None):
        self.cwd = Path(cwd)
        self.env = env
        self._proc = subprocess.Popen(
            ["bash"],
            cwd=str(self.cwd),
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def run_cmd(self, cmd: str) -> subprocess.CompletedProcess[str]:
        """Run a command inside the persistent bash session, streaming stdout."""
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("bash session is not available")
        marker = f"__CMD_EXIT__{uuid.uuid4().hex}__"
        # send command and sentinel
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.write(f"echo {marker}$?\n")
        self._proc.stdin.flush()

        stdout_lines: list[str] = []
        returncode: int | None = None
        while True:
            line = self._proc.stdout.readline()
            if line == "" and self._proc.poll() is not None:
                break
            if line.startswith(marker):
                try:
                    returncode = int(line.replace(marker, "").strip())
                except ValueError:
                    returncode = -1
                break
            # stream to live stdout
            print(line, end="", flush=True)
            stdout_lines.append(line)

        stdout = "".join(stdout_lines)
        return subprocess.CompletedProcess(
            args=cmd, returncode=returncode if returncode is not None else -1, stdout=stdout, stderr=""
        )

    def close(self):
        """Terminate the bash session."""
        if self._proc.poll() is None:
            if self._proc.stdin:
                try:
                    self._proc.stdin.write("exit\n")
                    self._proc.stdin.flush()
                except Exception:
                    pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=1)
            except Exception:
                pass

    def __del__(self):
        self.close()


def _select_tag(
    tags: Iterable[str], candidates: list[str], substring: str
) -> Optional[str]:
    """Pick the first matching tag from candidates, or fallback by substring."""
    for tag in candidates:
        if tag in tags:
            return tag
    for tag in tags:
        if substring in tag.lower():
            return tag
    return None


def read_metrics(tb_dir: Path) -> dict[str, dict[str, Any]]:
    """Read reward/success metrics from TensorBoard event files."""
    event_files = sorted(
        tb_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime
    )
    if not event_files:
        return {}

    ea = EventAccumulator(str(event_files[-1]))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    reward_tag = _select_tag(tags, REWARD_TAG_CANDIDATES, "reward")
    success_tag = _select_tag(tags, SUCCESS_TAG_CANDIDATES, "success_once")

    metrics: dict[str, dict[str, Any]] = {}
    if reward_tag:
        values = ea.Scalars(reward_tag)
        if values:
            metrics["reward"] = {
                "tag": reward_tag,
                "value": values[-1].value,
                "step": values[-1].step,
            }
    if success_tag:
        values = ea.Scalars(success_tag)
        if values:
            metrics["success_once"] = {
                "tag": success_tag,
                "value": values[-1].value,
                "step": values[-1].step,
            }
    return metrics
