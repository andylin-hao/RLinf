# RLinf Docs Style Guide

The contract for the docs refactor (see [refactor-proposal.md](refactor-proposal.md)).
Every new or edited page in `docs/source-en` and `docs/source-zh` follows this.

## Voice

- **Second person, imperative.** "You'll fine-tune…", "Run the script", "Set `cluster.num_nodes`".
  Never "RLinf provides a comprehensive guide to launching and managing…".
- **Outcome first.** Open every page/section with what the reader gets, then how.
- **No throat-clearing.** Cut "This section provides a comprehensive guide to … within the RLinf framework, focusing on…". Start with the verb or the result.
- **Short sentences.** One idea each. Prefer lists to run-on paragraphs.

## Example / recipe page requirements

Every benchmark (env) or model example page must:

1. **Open with a figure + intro.** Lead with the upstream benchmark/model figure
   (credited) and one paragraph on what it is and how RLinf uses it — like the
   [LeRobot benchmark pages](https://huggingface.co/docs/lerobot/en/libero).
2. **Put the benchmark facts inside Overview as tables.** Under the card grid, add two
   H3 subsections — `Tasks` (always a `list-table`, never a bullet list) and
   `Observation and Action` (a `list-table` of observation/action/reward/prompt). There is
   no separate "Tasks and Environment" section.
3. **Overview = 4 aligned cards.** Use `.. grid:: 2 4 4 4` (see below). Cards must
   align within each gallery subsection, with the exact same card titles and order in
   every page in that subsection. On an **env** page the **Models** card lists *every*
   model supported on that env and the **Algorithms** card lists *every* algorithm.
4. **No "Env type" card** — it carries too little information; put the env-type string in
   prose or the overview table instead.
5. **No generic "Algorithm" section** and **no boilerplate VLA intro** ("This section
   provides a comprehensive guide…", "Visual Understanding / Language Comprehension…").
   Algorithm definitions live in Reference, not on every recipe page.
6. **Cards or tables, not bullet walls.** Replace bullet lists of specs/metrics/perturbations
   with cards or `list-table`s.
7. **Don't explain metrics per page.** Link to the shared
   :doc:`Training metrics <tutorials/configuration/metrics>` page; keep only the page-specific
   "watch `env/success_once`" pointer and the results table.
8. **Name the card-grid section "Overview".** On a single-recipe page the card grid lives
   under an `Overview` heading right after the intro. On a multi-recipe page (e.g. LIBERO),
   there's no page-level `Overview`; instead each recipe family is its own section with a
   **descriptive, parallel** name (e.g. `Standard LIBERO Suites` / `LIBERO-Pro & LIBERO-Plus
   Suites`) and its own card grid. Don't repeat the H1 in a subtitle, and give any `:ref:`
   that points at a renamed section explicit link text so it still reads right.

## Navigation labels

Toctree entry captions (what shows in the left "Section Navigation") must be the **bare
name** — no "Benchmark", "Benchmarks", "Models", "World Model", "Simulation Platform", "RL
with …", "Training", "评测平台", "仿真平台", "模型" prefixes/suffixes. Use an explicit caption
in the toctree: ``LIBERO <embodied/libero>``, ``MLP <embodied/mlp>``,
``π₀ / π₀.₅ <embodied/pi0>``. The page **H1 title** may stay descriptive (e.g. "RL with
LIBERO Benchmarks"); only the nav caption is shortened. This applies to **every** gallery —
simulators, models, SFT, algorithms, and real-world.

**Top-level gallery category captions** (in `examples/index.rst`) are a single word — the
category, not a sentence:

| Index page | Nav caption (EN) | Nav caption (ZH) |
|---|---|---|
| `simulators_index` | Simulators | 模拟器 |
| `real_world_index` | Robots | 真机 |
| `vla_wam_index` | Models | 模型 |
| `sft_index` | SFT | SFT |
| `methods_index` | Algorithms | 算法 |
| `agentic/index` | Agents | 智能体 |
| `system/index` | Systems | 系统 |

The category index pages keep their descriptive H1 (e.g. "Algorithms for Embodiment"); only
the `examples/index.rst` toctree caption is the one-word form.

**Category ownership:** Place a page by the reader's starting point. Simulators /
benchmarks go in `simulators_index`; physical hardware goes in `real_world_index`; model
families and policy classes go in `vla_wam_index` (Models), including lightweight policies
such as ``MLP``; training recipes / algorithms go in `methods_index`; SFT-only workflows go
in `sft_index`. Do not duplicate the same page in multiple gallery indexes.

**Robots / Franka hierarchy:** Franka belongs under the Robots gallery, not as a top-level
Examples category. The Robots toctree links ``Franka <embodied/franka>`` so clicking
**Franka** opens the base Real-World RL page. The Franka page owns the nested Franka
variant toctree (``Reward Model``, ``ZED + Robotiq``, ``GELLO``, ``Dual-Arm``,
``Dexterous Hand``, ``Pi0 SFT``, ``HG-DAgger``). Keep these nav captions short and stable;
do not mirror longer page H1 titles into the nav.

## Page anatomy (recipe / example pages)

```rst
RL with <Name> Benchmarks            ← descriptive H1; nav caption is just "<Name>"
=========================

.. figure:: <upstream figure URL>
   :align: center
   :width: 90%

   <caption with image credit>

<One paragraph: what the benchmark/model is and how RLinf uses it.>

Overview                             ← cards + the benchmark facts (no "Tasks and Environment" title)
--------

<one-line outcome>.

.. grid:: 2 4 4 4                    ← 4 aligned cards (a 12-col grid aligns cleanly only
   :gutter: 2                          for 1/2/3/4/6 — avoid 5; push overflow to prose)

   .. grid-item-card:: Models        ← list EVERY model supported on this env
      :text-align: center
      <list>
   .. grid-item-card:: Algorithms    ← list EVERY algorithm supported
   .. grid-item-card:: Tasks
   .. grid-item-card:: Hardware

| **You'll do:** install → download model → launch → watch ``<metric>``.
| **Prerequisites:** :doc:`Installation <…>` · <other prereqs>.

Tasks                                ← H3, always a TABLE (never a bullet list)
~~~~~
.. list-table::   (benchmark suites: Suite · config id · Tasks · Focus;
                   multi-task envs: Category · Task · Description)

Observation and Action               ← H3
~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::   (Observation · Action · Reward · Task prompt)

Installation              → .. include:: _setup_common.rst + recipe-specific tag / --env
Download the Model        → recipe-specific download + .. include:: _model_path.rst
Run It                    → command + "What this command does" + "Configure further" admonition
Visualization and Results → TensorBoard / video / logger + link to Training metrics; results as a TABLE
```

## Reuse

- **Link, don't inline** reference material (full config tables, the complete metrics
  list, placement theory). Each page does one job and links to the canonical Reference.
- **Shared partials** live as underscore-prefixed files (`_setup_common.rst`,
  `_model_path.rst`). They are excluded from the build (`exclude_patterns = ["**/_*.rst"]`)
  and pulled in with `.. include:: _name.rst`. Substitutions don't work inside code
  blocks, so partials hold only the *identical* prose/code; recipe-specific tokens stay
  on the page.

## Images and media

- **Verify every image/media URL resolves (HTTP 200) before committing.** Broken images
  are a recurring problem — check the figure, every `<img>`/`<source>` in `raw:: html`
  blocks, and result images. Quick scan:

  ```bash
  grep -rhoE '(\.\. (figure|image):: |src=")https?://[^ "<>]+' source-en source-zh \
    | sed -E 's/^\.\. (figure|image):: //; s/^src="//' \
    | grep -iE '\.(png|jpg|jpeg|gif|svg|mp4|webm)$' | sort -u \
    | while read -r u; do echo "$(curl -s -o /dev/null -w '%{http_code}' -L "$u")  $u"; done \
    | grep -v '^200 '
  ```

- **Use a direct host URL, not a redirecting one.** Prefer
  `https://raw.githubusercontent.com/<org>/<repo>/<branch>/<path>` (or the gh-pages site).
  Avoid `https://github.com/<org>/<repo>/raw/...` — it 301/302-redirects through
  `text/html` responses that browsers don't reliably render as an `<img>`.
- **Watch for repo renames.** A 301 on the `raw` path means the org/repo moved (e.g.
  `haosulab/ManiSkill` → `mani-skill/ManiSkill`); point at the current name directly.
- **Confirm the exact path.** RLinf assets live in `RLinf/misc` under `pic/` *and*
  subfolders (e.g. `pic/rlinf-vla/…`, `pic/release_0.2/…`) — a wrong subfolder 404s.
- **Prefer a static image over a large animated GIF** for page figures (GIFs must fully
  download before they display, so heavy ones look broken).
- **Make robot figure captions specific to the page.** Do not use generic captions such as
  "Robot setup used by this RLinf recipe." State what the image represents for that
  recipe, for example "GELLO joint-level teleoperation device used to collect Franka
  demonstrations." For RLinf-owned robot images, omit image-credit suffixes.

## Headings & admonitions

- **Title Case for all headings**, consistent: `Run It`, `Download the Model`,
  `Visualization and Results` (lowercase only articles/short prepositions/conjunctions:
  a, an, the, and, or, of, to, in, on, with, vs).
- **Standard section names** (use these exact names so pages match):
  `Overview` (the card grid + the `Tasks` and `Observation and Action` subsections —
  there is no separate "Tasks and Environment" section) · `Installation` ·
  `Download the Model` (and `Download the Assets` if needed) · `Run It` ·
  `Visualization and Results`.
- **Align pages within each gallery subsection when applicable.** Simulator / benchmark
  pages use `Overview` → `Tasks` → `Observation and Action` → `Installation` → optional
  download sections → `Run It` → `Visualization and Results`. Model pages use the same
  overview table pattern, including lightweight policies such as ``MLP``. Within one
  subsection, overview cards and tables must use the same fields. Card schemas are:
  Simulators and Robots use `Models`, `Algorithms`, `Tasks`, `Hardware`; Models use
  `Environments`, `Algorithms`, `Tasks`, `Hardware`; Algorithms use `Algorithm`,
  `Models`, `Environments / Data`, `Training`; SFT uses `Models`, `Methods`, `Data`,
  `Hardware` (translated in ZH). Models pages use `Tasks` columns `Environment`,
  `Task / Suite`, `Config / Weights`, `Focus`, and `Observation and Action` rows
  `Observation`, `Action`, `Reward`, `Prompt` (translated in ZH, technical row names
  unchanged). Omit `Download the Model` only when there is no checkpoint to download.
  Algorithm pages use `Overview` with cards and a task/config table, then a method-specific
  `How <Method> Works` / `Pipeline` section before setup and commands. SFT pages use
  `Overview`, then dataset/model preparation sections as needed, followed by `Installation`,
  `Run It`, and `Visualization and Results` where applicable. Robots pages may keep
  hardware/safety workflow sections, but still start with `Overview`, `Tasks`, and
  `Observation and Action`.
- **Overview** uses a `sphinx-design` **card grid** (`.. grid:: 2 4 4 4` + `grid-item-card`),
  not a `tip` admonition — see the page anatomy above.
- `note` = side info · `warning` = footguns (OOM, `MUJOCO_GL`, `RLINF_NODE_RANK` ordering,
  multi-node gotchas). Put footguns in a `warning`, not prose.

## EN ↔ ZH parity

- Land every change in **both** trees in the same pass.
- **Code identifiers are sacred** — never translate config keys, env-type strings, CLI
  flags, script names, or model names in ZH.
- Headings are translated; internal links use stable `:doc:` / `:ref:` (no hardcoded
  ReadTheDocs URLs).
- **ZH:** don't put `**bold**` directly between CJK characters — docutils won't render it.

## Gate

After each change: `sphinx-build` both trees (`/opt/venv/docs/bin/sphinx-build -b html source-en /tmp/build-en` and `source-zh`) with **zero new warnings**, and run the `docs-check` skill for EN/ZH parity.
