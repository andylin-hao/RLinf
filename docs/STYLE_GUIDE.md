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
2. **Promote the benchmark facts to the top as tables.** Move observation/action/reward,
   task suites, and available-task lists into `list-table`s near the top — not buried in
   a per-recipe "Environment" subsection.
3. **At a glance = 4 aligned cards.** Use `.. grid:: 2 4 4 4` (see below). On an **env**
   page the **Models** card lists *every* model supported on that env and the **Algorithms**
   card lists *every* algorithm; on a **model** page they list every env / algorithm the
   model supports. Cards must align across env and model pages (same 4-card set).
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
8. **Don't duplicate the page title in the first subtitle.** When the first recipe
   section's name would just repeat the H1 (e.g. page "RL with LIBERO Benchmarks" →
   section "LIBERO Benchmark"), title that section **"At a Glance"** instead. Named
   variants keep their own subtitle (e.g. "LIBERO-Pro & LIBERO-Plus Benchmark"). Give any
   `:ref:` that pointed at the renamed section explicit link text so it still reads right.

## Navigation labels

Toctree entry captions (what shows in the left "Section Navigation") must be the **bare
name** — no "Benchmark", "Benchmarks", "Models", "World Model", "Simulation Platform",
"评测平台", "仿真平台", "模型" suffixes. Use an explicit caption: ``LIBERO <embodied/libero>``,
``ManiSkill <embodied/maniskill>``, ``π₀ / π₀.₅ <embodied/pi0>``. The page **H1 title** may
stay descriptive (e.g. "RL with LIBERO Benchmarks"); only the nav caption is shortened.

## Page anatomy (recipe / example pages)

```rst
RL with <Name> Benchmarks            ← descriptive H1; nav caption is just "<Name>"
=========================

.. figure:: <upstream figure URL>
   :align: center
   :width: 90%

   <caption with image credit>

<One paragraph: what the benchmark/model is and how RLinf uses it.>

Task suites                          ← promote benchmark facts to the top, as tables
-----------
.. list-table::   (Suite · config id · Tasks · Focus)

Observation and action
----------------------
.. list-table::   (Observation · Action · Reward · Task prompt)

<Recipe section>                     ← e.g. "LIBERO Benchmark"
----------------

**At a glance** — <one-line outcome>.

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

Install              → .. include:: _setup_common.rst + recipe-specific tag / --env
Download the model   → recipe-specific download + .. include:: _model_path.rst
Run it               → command + "What this command does" + "Configure further" admonition
Visualization and results → TensorBoard / video / logger + link to Training metrics; results as a TABLE
```

## Reuse

- **Link, don't inline** reference material (full config tables, the complete metrics
  list, placement theory). Each page does one job and links to the canonical Reference.
- **Shared partials** live as underscore-prefixed files (`_setup_common.rst`,
  `_model_path.rst`). They are excluded from the build (`exclude_patterns = ["**/_*.rst"]`)
  and pulled in with `.. include:: _name.rst`. Substitutions don't work inside code
  blocks, so partials hold only the *identical* prose/code; recipe-specific tokens stay
  on the page.

## Headings & admonitions

- **Sentence case**, consistent: `Run it`, not `Running the Script`.
- **At a glance** uses a `sphinx-design` **card grid** (`.. grid:: 2 4 4 4` + `grid-item-card`),
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
