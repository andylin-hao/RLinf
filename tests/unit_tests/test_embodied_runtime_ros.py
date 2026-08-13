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

import importlib
import sys
import types
from unittest.mock import MagicMock


def _load_ros_controller(monkeypatch):
    rospy = types.ModuleType("rospy")
    setattr(rospy, "init_node", MagicMock())
    setattr(rospy, "Publisher", MagicMock())
    setattr(rospy, "Subscriber", MagicMock())
    setattr(rospy, "Message", object)
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    module = importlib.import_module("rlinf.envs.realworld.common.ros.ros_controller")
    return importlib.reload(module), rospy


def test_ros_controller_connects_through_runtime(monkeypatch):
    module, rospy = _load_ros_controller(monkeypatch)
    runtime = MagicMock()
    runtime.resolve_robot_id.return_value = "franka-0"
    runtime.run_json.side_effect = [
        {"rosMasterUri": "http://10.0.0.2:11311", "state": "running"},
        {"rosMasterUri": "http://10.0.0.2:11311"},
    ]
    runtime_cli = MagicMock(return_value=runtime)
    runtime_cli.is_enabled.return_value = True
    monkeypatch.setattr(module, "EmbodiedRuntimeCLI", runtime_cli)
    popen = MagicMock()
    monkeypatch.setattr(module.psutil, "Popen", popen)

    controller = module.ROSController(robot_ip="172.16.0.2")
    controller.start_runtime_mode(
        "impedance", {"robot_ip": "172.16.0.2", "load_gripper": "true"}
    )

    popen.assert_not_called()
    rospy.init_node.assert_called_once()
    runtime.run_json.assert_called_with(
        "start",
        "franka-0",
        "impedance",
        "--arg",
        "robot_ip=172.16.0.2",
        "--arg",
        "load_gripper=true",
    )


def test_ros_controller_falls_back_without_runtime(monkeypatch):
    module, _ = _load_ros_controller(monkeypatch)
    monkeypatch.setattr(module.EmbodiedRuntimeCLI, "is_enabled", lambda _: False)
    monkeypatch.setattr(module.psutil, "process_iter", lambda: [])
    popen = MagicMock()
    monkeypatch.setattr(module.psutil, "Popen", popen)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    module.ROSController()

    popen.assert_called_once()
    assert popen.call_args.args[0] == ["roscore"]
