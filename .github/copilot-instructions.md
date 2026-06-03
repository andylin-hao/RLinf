# Copilot code review instructions for RLinf

When reviewing a pull request in this repository, apply the following rules. Use
the agent skills in `.github/skills/` when they are relevant to the changes.

## 1. Contribution and code-style compliance

Check that the PR conforms to the code-style requirements in `CONTRIBUTING.md`.
In particular:

- **Python style**: follows the Google Python Style Guide; stay consistent with
  surrounding code.
- **Docstrings**: every public class and method has a Google-style docstring.
- **Type hints**: all functions/methods type-hint their parameters, and the
  return type when it cannot be statically inferred.
- **Error handling**: assertions and exceptions carry clear, meaningful
  messages (never empty, never just restating `x != y`); inputs and states are
  validated early (e.g. before division or indexing).
- **Logging, not print**: use logging instead of `print`. Inside a `Worker`,
  use `self.log_info` / `self.log_warning` / `self.log_error`; elsewhere use
  `get_logger()` from `rlinf.utils.logging`.
- **Configuration YAML**: values are static (no computation or dynamic values in
  YAML); code does not overwrite user-settable config fields; cross-field
  references are avoided.
- **Tests and docs**: user-facing changes are accompanied by tests and
  documentation.
- **Commits / PR title**: follow Conventional Commits
  (`<type>(<scope>): <description>`); commits are `Signed-off-by`; the PR
  description fills in the Description and Checklist sections.

Flag any deviation from the above.

## 2. General PR review — `review-pr` skill

Follow the `review-pr` agent skill in `.github/skills/review-pr/` to review the
PR. Apply its workflow and the checklist in
`.github/skills/review-pr/reference.md`, verifying compliance with
`CONTRIBUTING.md`.

## 3. Documentation changes — `docs-check` skill

If the PR changes documentation (e.g. files under `docs/`, `README*`, or `*.rst`
/ `*.md`), follow the `docs-check` agent skill in `.github/skills/docs-check/`.
Cross-check docs against the code and against each other, including
English ↔ Chinese (EN/ZH) parity, and validate commands, config keys, and
model/env names. See `.github/skills/docs-check/reference.md`.

## 4. Install-script changes — `install-check` skill

If the PR changes the install logic (e.g. `requirements/install.sh`,
`requirements/embodied/`, or `docker/Dockerfile`), follow the `install-check`
agent skill in `.github/skills/install-check/`. Verify the conventions it
describes: reuse of common utilities, system deps kept in `sys_deps.sh`, pinned
or forked git deps, no ad-hoc `pyproject.toml`/core-dep hacks, and a matching
Dockerfile build stage for every new model/env.
