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

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from rlinf.utils.embodied_runtime import EmbodiedRuntimeCLI


def _result(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload).encode())


def test_cli_discovers_mounted_binary(monkeypatch, tmp_path):
    executable = tmp_path / "camctr"
    executable.touch()
    monkeypatch.setenv("RLINF_EMBODIED_PATH", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)

    assert EmbodiedRuntimeCLI.find_executable("camctr") == str(executable)


def test_cli_discovers_default_runtime_mount(monkeypatch):
    monkeypatch.delenv("RLINF_EMBODIED_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr(
        "pathlib.Path.is_file", lambda path: str(path) == "/opt/rlinf/bin/rosctr"
    )

    assert EmbodiedRuntimeCLI.find_executable("rosctr") == "/opt/rlinf/bin/rosctr"


def test_cli_resolves_runtime_inventory(monkeypatch):
    run = MagicMock(
        side_effect=[
            _result(
                {
                    "cameras": [
                        {
                            "cameraId": "video0",
                            "name": "wrist",
                            "serialNumber": "cam-123",
                        }
                    ]
                }
            ),
            _result(
                {
                    "robots": [
                        {
                            "robotId": "franka-0",
                            "params": {"robot_ip": "172.16.0.2"},
                        }
                    ]
                }
            ),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    cli = EmbodiedRuntimeCLI("camctr", executable="/opt/rlinf/bin/camctr")

    assert cli.resolve_camera_id("cam-123") == "video0"
    cli.tool = "rosctr"
    assert cli.resolve_robot_id("172.16.0.2") == "franka-0"


def test_cli_does_not_treat_broken_controller_as_uninstalled(monkeypatch, tmp_path):
    executable = tmp_path / "rosctr"
    executable.touch()
    monkeypatch.setenv("RLINF_EMBODIED_ROS_CLI", str(executable))
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(side_effect=subprocess.TimeoutExpired("rosctr", 5)),
    )

    assert EmbodiedRuntimeCLI.is_installed("rosctr")
    assert not EmbodiedRuntimeCLI.is_available("rosctr")


def test_cli_respects_disabled_runtime_manager(monkeypatch, tmp_path):
    executable = tmp_path / "camctr"
    executable.touch()
    monkeypatch.setenv("RLINF_EMBODIED_PATH", str(tmp_path))
    monkeypatch.setenv("RLINF_EMBODIED_RUNTIME_ENABLED", "1")
    monkeypatch.delenv("RLINF_EMBODIED_CAMERA_ENABLED", raising=False)

    assert EmbodiedRuntimeCLI.is_installed("camctr")
    assert not EmbodiedRuntimeCLI.is_enabled("camctr")

    monkeypatch.setenv("RLINF_EMBODIED_CAMERA_ENABLED", "1")
    assert EmbodiedRuntimeCLI.is_enabled("camctr")


def test_enabled_runtime_reports_missing_cli(monkeypatch):
    monkeypatch.setenv("RLINF_EMBODIED_RUNTIME_ENABLED", "1")
    monkeypatch.setenv("RLINF_EMBODIED_ROS_ENABLED", "1")
    monkeypatch.setattr("shutil.which", lambda _: None)

    assert EmbodiedRuntimeCLI.is_enabled("rosctr")
    with pytest.raises(FileNotFoundError, match="rosctr"):
        EmbodiedRuntimeCLI("rosctr")


def test_cli_rejects_ambiguous_robot(monkeypatch):
    monkeypatch.setattr(
        EmbodiedRuntimeCLI,
        "list_robots",
        lambda _: [{"robotId": "left"}, {"robotId": "right"}],
    )
    cli = EmbodiedRuntimeCLI("rosctr", executable="rosctr")

    with pytest.raises(ValueError, match="Could not select"):
        cli.resolve_robot_id()


def test_cli_does_not_fall_back_on_robot_ip_mismatch(monkeypatch):
    monkeypatch.setattr(
        EmbodiedRuntimeCLI,
        "list_robots",
        lambda _: [{"robotId": "franka-0", "params": {"robot_ip": "172.16.0.2"}}],
    )
    cli = EmbodiedRuntimeCLI("rosctr", executable="rosctr")

    with pytest.raises(ValueError, match="172.16.0.3"):
        cli.resolve_robot_id("172.16.0.3")


def test_cli_rejects_ambiguous_camera_name(monkeypatch):
    monkeypatch.setattr(
        EmbodiedRuntimeCLI,
        "list_cameras",
        lambda _: [
            {"cameraId": "video0", "name": "wrist"},
            {"cameraId": "video1", "name": "wrist"},
        ],
    )
    cli = EmbodiedRuntimeCLI("camctr", executable="camctr")

    with pytest.raises(ValueError, match="ambiguous"):
        cli.resolve_camera_id("wrist")


def test_cli_reports_timeout(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(side_effect=subprocess.TimeoutExpired("camctr", 10)),
    )
    cli = EmbodiedRuntimeCLI("camctr", executable="camctr")

    with pytest.raises(RuntimeError, match="timed out"):
        cli.run("list")


def test_cli_reports_command_failure_without_stderr(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(side_effect=subprocess.CalledProcessError(2, "camctr")),
    )
    cli = EmbodiedRuntimeCLI("camctr", executable="camctr")

    with pytest.raises(RuntimeError, match="failed: 2"):
        cli.run("list")


def test_cli_reports_invalid_json(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout=b"not-json")),
    )
    cli = EmbodiedRuntimeCLI("camctr", executable="camctr")

    with pytest.raises(RuntimeError, match="invalid JSON"):
        cli.run_json("list")
