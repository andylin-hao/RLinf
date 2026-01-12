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
import math
import time
from pathlib import Path
from typing import Any

from .config import ExampleConfig, RewardVerifyConfig
from .example_runner import ExampleRunner
from .repo_manager import RepoManager


def load_baselines(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"entries": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[warn] baseline file {path} is not valid JSON; ignoring its contents")
        return {"entries": []}


def save_baselines(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_baselines(path).get("entries", [])
    lookup = {(e["config"], e.get("commit")): e for e in existing}
    for entry in entries:
        lookup[(entry["config"], entry.get("commit"))] = entry
    payload = {"entries": list(lookup.values())}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def find_baseline(
    baselines: dict[str, Any], config: str, commit: str | None
) -> dict[str, Any] | None:
    entries = baselines.get("entries", [])
    filtered = [e for e in entries if e.get("config") == config]
    if commit:
        filtered = [e for e in filtered if e.get("commit") == commit]
    if not filtered:
        return None
    return filtered[-1]


def compare_metrics(
    run_metrics: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    rtol: float,
    atol: float,
):
    result = {"matches": True, "details": {}}
    base_metrics = baseline.get("metrics", {})
    for key in ["reward", "success_once"]:
        if key not in base_metrics:
            continue
        if key not in run_metrics:
            result["matches"] = False
            result["details"][key] = {
                "status": "missing",
                "baseline": base_metrics[key],
                "current": None,
            }
            continue
        current_value = run_metrics[key]["value"]
        base_value = base_metrics[key]["value"]
        matches = math.isclose(current_value, base_value, rel_tol=rtol, abs_tol=atol)
        result["matches"] = result["matches"] and matches
        result["details"][key] = {
            "status": "ok" if matches else "mismatch",
            "baseline": base_metrics[key],
            "current": run_metrics[key],
            "diff": current_value - base_value,
        }
    return result


class RewardVerifier:
    """Coordinate repo sync, config expansion, run execution, and baseline checks."""

    def __init__(self, cfg: RewardVerifyConfig):
        self.cfg = cfg
        self.repo_manager = RepoManager(cfg.repo)
        self.example_runner = ExampleRunner(cfg.repo.path, cfg.defaults)

    def _discover_yaml_paths(self) -> set[Path]:
        """Return all YAML files under configured discovery roots."""
        roots = [
            Path(self.repo_manager.path, p) for p in self.cfg.defaults.discovery_roots
        ]
        found: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.yaml"):
                found.add(path.resolve())
        return found

    def _known_paths(self) -> set[Path]:
        """Return YAMLs that are either explicitly tested or skipped."""
        known = set()
        for suite in self.cfg.tests:
            for case in suite.cases:
                resolved = self._resolve_config_path(case.name)
                if resolved:
                    known.add(resolved)
        for sk in self.cfg.defaults.skip:
            known.add(Path(self.repo_manager.path, sk).resolve())
        return known

    def _resolve_config_path(self, config_name: str) -> Path | None:
        """Resolve a config name to a YAML path within discovery roots."""
        roots = [Path(self.repo_manager.path, p) for p in self.cfg.defaults.discovery_roots]
        candidates = [config_name]
        if not config_name.endswith(".yaml"):
            candidates.append(f"{config_name}.yaml")
        for root in roots:
            for cand in candidates:
                candidate = root / cand
                if candidate.exists():
                    return candidate.resolve()
        return None

    def warn_on_new_yamls(self) -> list[Path]:
        """Log YAMLs that are not covered by tests or skip lists."""
        discovered = self._discover_yaml_paths()
        known = self._known_paths()
        new_entries = sorted(discovered - known)
        if new_entries:
            print("[warn] new YAMLs detected (not in tests or skip):")
            for p in new_entries:
                try:
                    rel = p.relative_to(self.repo_manager.path)
                except ValueError:
                    rel = p
                print(f"  - {rel}")
        return new_entries

    def _expand_tests(self) -> list[ExampleConfig]:
        """Flatten suites/cases into executable ExampleConfig objects."""
        expanded: list[ExampleConfig] = []
        for suite in self.cfg.tests:
            for case in suite.cases:
                # merge defaults -> suite -> case
                cfg_path = self._resolve_config_path(case.name)
                if cfg_path is None:
                    raise FileNotFoundError(
                        f"Config '{case.name}' not found in discovery_roots {self.cfg.defaults.discovery_roots}"
                    )
                expanded.append(
                    ExampleConfig(
                        name=f"{suite.name}::{case.name}",
                        config_path=str(cfg_path),
                        runner=case.runner or suite.runner or self.cfg.defaults.runner,
                        model_path=case.model_path or suite.model_path or self.cfg.defaults.model_path,
                        max_epochs=case.max_epochs or suite.max_epochs or self.cfg.defaults.max_epochs,
                        python_exe=self.cfg.defaults.python_exe,
                        venv=case.venv or suite.venv or self.cfg.defaults.venv,
                        pre_run=(self.cfg.defaults.pre_run or [])
                        + (suite.pre_run or [])
                        + (case.pre_run or []),
                        skip=suite.skip or case.skip,
                        extra_overrides=(self.cfg.defaults.extra_overrides or [])
                        + (suite.extra_overrides or [])
                        + (case.extra_overrides or []),
                    )
                )
        return expanded

    def _run_examples(self, commit: str) -> list[dict[str, Any]]:
        """Run all expanded examples and attach commit metadata."""
        results = []
        skip_set = {Path(s).resolve() for s in self.cfg.defaults.skip}
        for example in self._expand_tests():
            cfg_path = Path(example.config_path).resolve()
            if example.skip or cfg_path in skip_set:
                continue
            entry = self.example_runner.run(example)
            entry["commit"] = commit
            results.append(entry)
        return results

    def run_once(self):
        """Execute one verification/baseline pass across all configured tests."""
        self.repo_manager.pull_main()
        commit = self.repo_manager.current_commit()

        self.warn_on_new_yamls()

        entries = self._run_examples(commit)

        if self.cfg.defaults.mode == "baseline":
            save_baselines(self.cfg.baseline.file, entries)
            print(
                f"[baseline] wrote {len(entries)} entries to {self.cfg.baseline.file}"
            )
            return

        baselines = load_baselines(self.cfg.baseline.file)
        failures = []
        for entry in entries:
            baseline = find_baseline(
                baselines, entry["config"], self.cfg.baseline.commit
            )
            if not baseline:
                failures.append((entry["config"], "no baseline"))
                print(f"[warn] no baseline found for {entry['config']}")
                continue
            cmp_result = compare_metrics(
                entry["metrics"],
                baseline,
                self.cfg.baseline.rtol,
                self.cfg.baseline.atol,
            )
            status = "PASS" if cmp_result["matches"] else "FAIL"
            print(f"[verify] {entry['config']} -> {status}")
            for key, detail in cmp_result["details"].items():
                cur = detail["current"]["value"] if detail.get("current") else None
                base = detail["baseline"]["value"]
                print(
                    f"  {key}: current={cur}, baseline={base}, status={detail['status']}"
                )
            if not cmp_result["matches"]:
                failures.append((entry["config"], "mismatch"))

        if failures:
            raise SystemExit(f"Verification failed: {failures}")

    def run(self):
        """Run once or loop at an interval based on configuration."""
        if self.cfg.defaults.interval_minutes <= 0:
            self.run_once()
            return
        while True:
            self.run_once()
            sleep_s = self.cfg.defaults.interval_minutes * 60
            print(f"[loop] sleeping {sleep_s} seconds before next pull")
            time.sleep(sleep_s)
