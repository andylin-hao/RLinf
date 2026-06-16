# RLinf Documentation Refactor — Proposal (RFC)

**Status:** Approved (direction) — §10 questions resolved · **Scope:** `docs/source-en` + `docs/source-zh` · **Author:** docs refactor
**Goal:** bring the RLinf docs to the clarity and professionalism of the
[LeRobot](https://huggingface.co/docs/lerobot/en/index) and
[Ray](https://docs.ray.io/en/latest/index.html) docs, refactoring from
high-level structure down to low-level wording.

This is a planning document. **No production doc files change until this is
approved.** It is intentionally outside the Sphinx build (`docs/source-en` /
`docs/source-zh` are the only built trees).

---

## 1. Decisions locked for this refactor

| Decision | Choice |
|---|---|
| Voice / tone | **Second-person imperative** ("You'll train…", "Run this"), outcome-first, short sentences — the LeRobot/Ray style. |
| Languages | **EN and ZH together, every change.** No phase lands in one language only; parity is checked each step. |
| Process | **This proposal first**, then execute phase by phase against it. |

**Resolved (§10):** (1) nav axes accepted, **Examples placed before Concepts**;
(2) **accept URL breakage** — no redirects; (3) keep the full feature/benchmark
pitch but **rewrite it for clarity** on a dedicated *Why RLinf* page; (4) Phase 3
covers **all example pages**; (5) **ship a Cheat Sheet** (contents in Appendix A).

---

## 2. Diagnosis — where RLinf differs from LeRobot / Ray

**Root cause:** RLinf docs are *feature-first and exhaustive*; the references are
*task-first and oriented*. Concretely, by level:

1. **Information architecture.** `Tutorials` is a grab-bag of five doc *types*
   (reference `configuration/`, theory `rlalg/`, how-to `usage/`+`embodied/`+`agentic/`,
   contributor `extend/`, changelog `release`). `Examples/embodied` overlaps with
   `Tutorials/embodied`. There is no Concepts destination in the nav (the
   Worker/WorkerGroup/Channel model is buried atop `tutorials/index.rst`), no cheat
   sheet, and Blog/Publications sit as heavyweight top-level captions.

2. **Landing page.** Today's landing ([index.rst](source-en/index.rst)) is a vendor
   pitch ("RLinf is unique with… is fast with…") with jargon, no CTAs, no code, no
   use-case routing. Ray's equivalent space is one value sentence + 3 buttons + use-case
   code cards.

3. **Section intros.** `start/index.rst` mixes a quickstart with a "SOTA Reproduction"
   marketing block and *describes pages* instead of guiding action. `tutorials/index.rst`
   opens with three dense architecture paragraphs (reference material) before any task.

4. **Page design.** Each example page is a fixed 11-section monolith
   (Environment → Algorithm → Dependency Installation → Clone → Install → Assets →
   Model Download → Running the Script → Visualization → Results → Config Details),
   repeats clone/install/model-download across ~20 pages, opens with a 28-word
   throat-clearing sentence, and inlines the full config + every metric on every page.
   LeRobot pages open with "You'll learn: 1/2/3", annotate each command
   ("Let's explain the command: 1… 2…"), pair CLI + API, and link out to deeper guides.

5. **Wording.** Third-person feature-marketing, run-on sentences, inconsistent heading
   casing, throat-clearing intros — vs. the references' second-person, imperative,
   benefit-first, short prose.

---

## 3. Target information architecture (Phase 1)

Regroup the top nav into clean, single-purpose axes. Left = today, right = proposed.

Final top-nav order: **Get Started · Examples · Concepts · Guides · Reference ·
Extending · Resources** (Examples before Concepts, per §10 Q1).

```
TODAY                         →   PROPOSED
─────────────────────────────────────────────────────────────
Quickstart (start/)           →   Get Started      (install · quickstarts · cheat sheet)
Examples/ (galleries)         →   Examples         (the 5 galleries, promoted & unchanged in content)
                              →   Concepts         (NEW: M2Flow, Worker/WorkerGroup/Channel,
                                                    execution modes, placement — the mental model)
Tutorials/ (8 mixed subdirs)  →   Guides           (how-to: usage/, configuration/ how-tos,
                                                    accelerators/, advance/ operational)
                              →   Reference        (config keys, CLI/scripts, APIs/, algorithm specs rlalg/)
                              →   Extending        (extend/: new env/model/algorithm)
Blog/ , Publications/         →   Resources        (blog, publications, release notes, FAQ, community,
                                                    Why RLinf)
APIs/                         →   (folded into Reference)
FAQ                           →   (folded into Resources)
```

### Where each existing directory lands

| Current path | New home | Notes |
|---|---|---|
| `start/installation,vla,llm,distribute,vla-eval,llm-eval` | **Get Started** | Split "quickstart" (vla/llm) from "evaluation" (vla-eval/llm-eval); add a **Cheat Sheet** page. |
| `tutorials/index` architecture prose | **Concepts** (new landing) | Becomes a real Concepts page, not a section preamble. |
| `tutorials/usage/` (worker, channel, placement, execution_modes, flow, multi_node, convertor) | **Concepts** (flow/channel/worker/placement) + **Guides** (multi_node, convertor) | Conceptual pages explain the model; operational pages are how-tos. |
| `tutorials/configuration/` (basic, embodiment, agentic, hetero, logger, resume) | **Guides** (the how-to ones) + **Reference** (the key tables) | Separate "how to configure X" from "the list of keys". |
| `tutorials/rlalg/` (ppo, grpo, dapo, sac, …) | **Reference → Algorithms** | Algorithm specs are reference, not tutorials. |
| `tutorials/accelerators/` | **Guides → Accelerators** | Already a good standalone group (from the reorg). |
| `tutorials/advance/` (5D, lora, profile, cluster, dynamic_scheduling, …) | **Guides → Advanced** | Operational deep-dives. |
| `tutorials/extend/` (new_env, new_model_*, mbridge, weight_syncer) | **Extending** | Contributor-facing; own top-level axis like Ray's "Developer guides". |
| `tutorials/embodied/`, `tutorials/agentic/` | merge into **Concepts/Guides** per page | Removes the `tutorials/embodied` vs `examples/embodied` ambiguity. |
| `examples/*_index` (5 galleries) | **Examples** | Promote to first-class; content unchanged. |
| `apis/` | **Reference → API** | Unchanged content, regrouped. |
| `blog/`, `publications/`, `release` | **Resources** | Lower visual weight. |

> The mapping above is the *contract* to review. Phase 1 only moves files and
> rewrites toctrees/captions + cross-links (EN+ZH); page bodies are untouched in
> Phase 1.

---

## 4. Landing page redesign (Phase 2)

Replace the feature wall with a task router. Sketch:

```
RLinf — Scalable RL post-training for foundation models & embodied agents
[ Get Started ]   [ Install ]   [ Examples ]

Choose your path
┌─ Embodied RL ──────────┬─ Agentic / Reasoning RL ─┐
│ Fine-tune a VLA on      │ GRPO on math/agent tasks │
│ LIBERO/ManiSkill with   │ with Qwen/DeepSeek       │
│ PPO/GRPO  → snippet     │           → snippet      │
├─ Bring your own ───────┼─ Scale to a cluster ─────┤
│ Add a model / env /     │ Collocated · disaggregated│
│ algorithm   → Extending │ · hybrid   → Concepts    │
└─────────────────────────┴──────────────────────────┘

Why RLinf  (3 short benefit lines + "Learn more →" to the Why-RLinf page)
```

The current "unique / fast / flexible" bullets move to a dedicated **Why RLinf**
page linked from the landing — kept, but below the fold.

---

## 5. Section intros (Phase 2)

- **Get Started landing:** install → **one** copy-paste hello-world run → "What's next"
  (3 links). Drop the SOTA-reproduction block here (move to **Why RLinf** / a
  Benchmarks results page).
- **Concepts landing:** the execution-flow figure + Worker/WorkerGroup/Channel prose
  from today's `tutorials/index.rst`, rewritten in second person, with a diagram-first,
  short-paragraph layout.
- **Guides / Reference / Extending landings:** lead with a one-line purpose + a routed
  list ("Pick this when…"), not prose — mirror the reorg's gallery-intro style.

---

## 6. Recipe page template (Phase 3)

`libero.rst` is the **reference implementation** of this template (EN + ZH). Every
example/recipe page must follow it. Hard requirements:

1. **Open with a figure + intro** — the upstream benchmark/model figure (credited) and one
   paragraph on what it is and how RLinf uses it.
2. **Promote benchmark facts to the top as tables** — observation/action/reward, task
   suites, available tasks become `list-table`s, not a buried "Environment" subsection.
3. **At a glance = 4 aligned cards** (`.. grid:: 2 4 4 4`). On **env** pages the **Models**
   card lists *every* supported model and **Algorithms** lists *every* supported algorithm
   (read from the configs); on **model** pages they list every supported env/algorithm.
   Same 4-card set everywhere so cards align across pages.
4. **Remove** the "Env type" card, the generic "Algorithm" section, and the boilerplate
   VLA intro ("This section provides a comprehensive guide…", "Visual Understanding…").
5. **Cards or tables, not bullet walls** (specs, metrics, perturbation dimensions).
6. **Metrics live on one page** — link to :doc:`Training metrics <…>`; never re-explain
   metrics per recipe. Keep only "watch `env/success_once`" + the results table.
7. **Nav captions are bare names** — `LIBERO`, `ManiSkill`, `π₀ / π₀.₅` (no
   "Benchmark/Models/Platform/评测平台/模型" suffixes); the H1 may stay descriptive.

```rst
RL with <Name> Benchmarks            ← descriptive H1; nav caption is just "<Name>"
=========================

.. figure:: <upstream figure>        ← credited
<One paragraph: what it is + how RLinf uses it.>

Task suites / Observation and action ← promoted to top as .. list-table::

<Recipe section>
----------------
**At a glance** — <one-line outcome>

.. grid:: 2 4 4 4                    ← Models · Algorithms · Tasks · Hardware (all aligned)
   .. grid-item-card:: Models        ← lists EVERY supported model on this env
   .. grid-item-card:: Algorithms    ← lists EVERY supported algorithm
   .. grid-item-card:: Tasks
   .. grid-item-card:: Hardware

| **You'll do:** install → download model → launch → watch ``<metric>``
| **Prerequisites:** :doc:`Installation <…>`

Install              → .. include:: _setup_common.rst + recipe-specific tag/--env
Download the model   → recipe-specific download + .. include:: _model_path.rst
Run it               → command + "What this command does" + "Configure further" admonition
Visualization and results → TensorBoard/video/logger + link to Training metrics; results TABLE
```

Reference material that was copy-pasted per page (full placement examples, the entire
metrics list, config-key explanations) is **extracted once** into Reference pages and
linked — e.g. the new `tutorials/configuration/metrics.rst` already does this for metrics.
Shared includes live as underscore-prefixed partials (`exclude_patterns = ["**/_*.rst"]`).

---

## 7. Voice & style guide (Phase 0 — the contract everything conforms to)

- **Second person, imperative.** "You'll fine-tune…", "Run the script", "Set `…`".
  Not "RLinf provides a comprehensive guide to…".
- **Outcome first.** Open every page/section with what the reader gets, then how.
- **No throat-clearing.** Cut "This section provides a comprehensive guide to launching
  and managing the … within the RLinf framework, focusing on…".
- **Annotate commands.** After any non-trivial command, add "What this does: 1… 2…".
- **Headings: sentence case**, consistent (`Run it`, not `Running the Script`).
- **Admonitions:** `tip` = At-a-glance; `note` = side info; `warning` = footguns
  (OOM, env vars like `MUJOCO_GL`, multi-node ordering). Don't bury footguns in prose.
- **Link, don't inline** reference material; each page does one job.
- **Code identifiers are sacred in both languages** — never translate config keys,
  env-type strings, CLI flags, or model names in ZH.
- **ZH specifics:** avoid `**bold**` sandwiched directly between CJK characters
  (docutils won't render it; the existing ZH docs already avoid this).

---

## 8. EN ↔ ZH parity policy

Every change lands in both `source-en` and `source-zh` in the same pass. After each
phase, run the `docs-check` skill (structure + EN/ZH parity) and a full
`sphinx-build` of both trees (`/opt/venv/docs`) — **zero new warnings** is the gate.
Section headings are translated; code tokens are identical; internal links use
stable `:doc:`/`:ref:` (no hardcoded ReadTheDocs URLs).

---

## 9. Phased rollout

| Phase | Deliverable | Blast radius | Gate |
|---|---|---|---|
| **0. Conventions** | Style guide adopted; `_setup_common.rst` + `_model_path.rst` partials; **`sphinx-design`** added for At-a-glance card grids; `libero` pilot converted (done) | small | build clean |
| **1. IA / nav** | Move dirs per §3 mapping; rewrite all toctrees/captions + cross-links (EN+ZH); add Concepts/Reference/Resources/Extending landings + Cheat Sheet stub | large (link churn) | build clean, no orphans, docs-check parity |
| **2. Landing + intros** | New landing task-router; Why-RLinf page; slim Get Started; Concepts page from old tutorials prose | medium | build clean |
| **3. Page design** | Apply §6 template to all recipe pages; extract Config/Metrics Reference pages | large (per-page) | build clean; pilot template locked on `libero` first |
| **4. Wording pass** | Voice/imperative/heading-case sweep across EN+ZH | medium | build clean; spot review |

Each phase is independently shippable and reviewable.

---

## 10. Open questions — RESOLVED

1. **Nav axis names** — ✅ accepted; **Examples moved before Concepts**:
   `Get Started · Examples · Concepts · Guides · Reference · Extending · Resources`.
2. **URL stability** — ✅ **accept breakage**, no redirects (docs are versioned).
3. **"Why RLinf"** — ✅ **keep** the full feature/benchmark pitch, but **rewrite it for
   clarity and readability** on a dedicated *Why RLinf* page under Resources, linked from
   the landing.
4. **Scope of Phase 3** — ✅ **all example pages** (~42), embodied + agentic + system.
5. **Cheat Sheet** — ✅ ship it; contents in **Appendix A**.

---

## Appendix A — Cheat Sheet contents (Get Started → Cheat Sheet)

One page, copy-paste oriented, EN+ZH. Confirmed inventory (all verified against the
repo):

**Install**
- `bash requirements/install.sh embodied --model <model> --env <env>` (add `--use-mirror` in mainland China; `--venv <name>`, `--python <ver>` optional)
- Docker: `docker run … rlinf/rlinf:<tag>` + `source switch_env <model>`

**Run (entry scripts)**
- Embodied train: `bash examples/embodiment/run_embodiment.sh <config>`
- Embodied eval: `bash examples/embodiment/eval_embodiment.sh <config>`
- Async embodied: `examples/embodiment/run_async.sh` · Real-world: `run_realworld.sh` / `run_realworld_eval.sh`
- Reasoning GRPO (math): `examples/reasoning/run_main_grpo_math.sh` · (VQA): `run_main_grpo_vqa.sh`
- Offline RL: `run_offline_rl.sh` · Placement autotune: `run_placement_autotune.sh`

**Key env vars**
- `MUJOCO_GL=egl` — rendering backend (footgun: headless EGL vs OSMesa)
- `ROBOT_PLATFORM` — embodied platform select
- `LIBERO_TYPE=pro|plus` — LIBERO suite switch
- `RLINF_NODE_RANK=<0..N-1>` — set **before** `ray start` on each node
- `RLINF_COMM_NET_DEVICES` — NIC selection for multi-node
- `HF_ENDPOINT=https://hf-mirror.com` — HF mirror for mainland China

**Most-tuned config keys** (link to Reference for the full table)
- Placement: `cluster.num_nodes`, `cluster.component_placement`, `rollout.pipeline_stage_num`
- OOM: `env.total_num_envs`, `*.micro_batch_size`, `rollout.gpu_memory_utilization`, `*.enable_offload`, `actor.gradient_checkpointing`
- Algorithm: `algorithm.adv_type`, `algorithm.loss_type`, `algorithm.group_size`
- Checkpoint/resume: `runner.save_interval`, `runner.resume_dir`
- Logging: `runner.logger.logger_backends` (`tensorboard` / `wandb` / `swanlab`)

---

*Next step:* execute **Phase 0** (adopt §7 style guide; add the `_model_path.rst`
partial used by the recipe template), then open **Phase 1 (IA)** as its own reviewable
change implementing the §3 mapping in EN+ZH.
