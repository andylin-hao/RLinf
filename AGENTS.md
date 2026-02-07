# AGENTS.md

Brief for AI coding agents working on RLinf. For full contribution flow, code style, and PR process see [CONTRIBUTING.md](CONTRIBUTING.md).

**Quick orientation:** RLinf is a distributed RL stack (embodied + reasoning + agent). It uses **Ray** for process management and **Hydra** for config. Single-machine runs use `cluster.num_nodes: 1`; multi-node needs Ray started on every node with `RLINF_NODE_RANK` set *before* `ray start`. Pre-commit runs Ruff (lint + format) and commit-check; use Google-style docstrings and type hints. All user-facing changes need tests and docs. If something is unclear, add a `TODO(agent)` and note the limitation.

---

## Code structure

- **`.cursor/`** – Rules and skills: `rules/agents-md.mdc`, `skills/add-install-docker-ci-e2e`, `skills/add-example-doc-model-env`, `skills/review-pr`.
- **`rlinf/`** – Main package:
  - `agents/` – Agent logic (reasoning, tools).
  - `algorithms/` – Advantages, losses, registry, rewards (math, code, searchr1, vqa).
  - `config.py` – Hydra config, `SupportedModel`, `SupportedEnvType`, validation.
  - `data/` – Datasets for embodied, reasoning, agent.
  - `envs/` – ManiSkill, LIBERO, IsaacLab, CALVIN, MetaWorld, Behavior, RoboCasa, FrankaSim, RealWorld, RoboTwin, Habitat, OpenSora world model; `get_env_cls()` in `envs/__init__.py`.
  - `hybrid_engines/` – SGLang/vLLM rollout integration.
  - `models/` – Embodiment (OpenVLA, OpenVLA-OFT, OpenPI, GR00T, MLP/CNN/Flow/CMA) and reasoning wiring.
  - `runners/` – Embodied (sync/async), reasoning, coding_online_rl, agent, SFT, eval.
  - `scheduler/` – Cluster, Worker, WorkerGroup, channel, manager, placement, dynamic_scheduler.
  - `utils/` – Logging, placement, data iter, distributed, checkpoint, resharding.
  - `workers/` – Actor (FSDP/Megatron), rollout (HF/server), env (sync/async), reward, replay buffer.
- **`examples/`** – Entrypoints and YAML: embodiment, reasoning, coding_online_rl, searchr1, sft, multiturn_demo, wideseek_r1.
- **`tests/`** – `unit_tests/`, `e2e_tests/` (embodied, agent, reasoning), scheduler tests; e2e configs under `e2e_tests/embodied/*.yaml`.
- **`requirements/`** – `install.sh` (targets: embodied, reason, docs; `--model`, `--env`), optional deps in subdirs.
- **`docker/`** – Dockerfile and build targets per model/env.
- **`ray_utils/`** – `start_ray.sh` (multi-node head/worker), `check_ray.sh`, `realworld/setup_before_ray.sh`.
- **`toolkits/`** – Checkpoint convertors, verifiers, eval scripts, replay buffer, auto-placement.
- **`docs/`** – Sphinx RST (EN/ZH): start, tutorials, examples, APIs, FAQ.

---

## How RLinf runs

You launch one entry script (e.g. `train_embodied_agent.py`, `train_async.py`). It builds a **Cluster** (Ray must already be up), figures **component placement** (actor, rollout, env, reward, agent), and starts **Worker** groups. A **Runner** drives the loop: rollout → reward → advantage → actor update (and any inference/engine lifecycle). Cluster config lives in YAML under `cluster:`: `num_nodes`, `component_placement`, `node_groups` (labels, node_ranks, env_configs, optional hardware e.g. Franka). Placement (e.g. `HybridComponentPlacement`, `ModelParallelComponentPlacement`) maps components to node groups and hardware ranks. Workers are Ray remote actors with `MASTER_*`, `RANK`, etc.; they can `send`/`recv` across groups. Training backends: FSDP or Megatron. Rollout: SGLang or vLLM. Runners pick loss/advantage from config (PPO, GRPO, SAC, etc.).

---

## Single-node and multi-node

**Single machine:** Install via Docker or `bash requirements/install.sh embodied --model <model> --env <env>` (set `REPO_PATH` and any asset paths). Ray may auto-start; or run `ray start --head`. Use a config with `cluster.num_nodes: 1` (e.g. from `examples/embodiment/config/`). Launch with `bash examples/embodiment/run_embodiment.sh <config_name>` or `python examples/embodiment/train_embodied_agent.py --config-name <config_name>`, and set env vars the example needs (e.g. `MUJOCO_GL=egl`, `ROBOT_PLATFORM`).

**Multiple machines:** On each node, *before* `ray start`: set `export RLINF_NODE_RANK=<0..N-1>` (unique) and optionally `RLINF_COMM_NET_DEVICES`. Head: `ray start --head --port=6379 --node-ip-address=<head_ip>`. Workers: `ray start --address=<head_ip>:6379`. You can use `ray_utils/start_ray.sh`. Set `cluster.num_nodes` to the total; optionally use `node_groups` and `component_placement` (see `rlinf/scheduler/cluster/config.py` and the [heterogeneous cluster tutorial](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/hetero.html)). Run the entry script *only on the head*; it attaches to the existing Ray cluster and schedules workers by placement.

---

## Configuration guides

**Placement and throughput**

Component placement is set under `cluster.component_placement` in YAML. It controls which GPUs (or other hardware) run actor, rollout, env, reward, and agent. For higher throughput:

- **Collocated:** Put actor, inference, and rollout on the same GPUs (`actor,inference,rollout: 0-7` or `all`) so they share resources; use when GPU memory fits all components or when you rely on offload/reload to swap them.
- **Disaggregated:** Put actor on one set of GPUs and rollout on another so they run in parallel; reduces contention but can leave GPUs idle if one phase dominates.
- **Hybrid:** Mix collocated and disaggregated per component (e.g. rollout + env on one group, actor on another). Use the node-group form when you have multiple node types: set `node_groups` with labels (e.g. `a800`, `4090`), then in `component_placement` reference `node_group: a800` and `placement: 0-63` (hardware ranks within that group). More GPUs for rollout/env usually increase sample throughput; more for actor increase training throughput. See the [placement tutorial](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/placement.html) and [execution modes](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/mode/index.html) (collocated, disaggregated, hybrid).

**OOM and memory-related knobs**

- **Env:** Reduce `env.train.total_num_envs` and `env.eval.total_num_envs`; reduce `env.train.group_size` / `env.eval.group_size` if each group holds many envs.
- **Rollout:** Reduce batch size or sequence length if the rollout worker OOMs. For SGLang/vLLM, lower `rollout.static_mem_fraction` (or equivalent) so the server reserves less GPU memory; ensure weights are released before reload after actor updates (see FAQ). Set `rollout.enable_offload: True` if the rollout worker is configured to offload when idle.
- **Actor:** Reduce `actor.micro_batch_size` and/or `actor.global_batch_size`; increase gradient accumulation steps if you keep `global_batch_size` fixed. Enable **gradient checkpointing** to trade compute for memory: in YAML set `actor.model.gradient_checkpointing: True` (or under `actor.fsdp_config.gradient_checkpointing` for FSDP, depending on config structure). Enable **actor offload** to move params/grads (and optionally optimizer state) to CPU when not training: `actor.enable_offload: True`; the runner will offload/onload around update steps. See example configs (e.g. `examples/embodiment/config/libero_spatial_ppo_dexbotic_pi0.yaml`) for `enable_offload`, `micro_batch_size`, and `gradient_checkpointing`.

**Multiple nodes and heterogeneous clusters**

- **Node count and identity:** Set `cluster.num_nodes` to the total number of nodes. Each node must have a unique **node rank** (0 to N−1). Node rank is **not** set in YAML: it is fixed by the **environment variable `RLINF_NODE_RANK`** on the process that runs `ray start` on that node. Ray captures the environment at `ray start` time; any env vars (including `RLINF_NODE_RANK` and `RLINF_COMM_NET_DEVICES`) must be exported **before** calling `ray start` on that node, or they will not be visible to workers later. Worker processes inherit the env that Ray had when it started.
- **YAML cluster config:** Under `cluster` you can optionally set `node_groups` and `component_placement`. **`node_groups`** define named groups (e.g. `a800`, `4090`, `franka`) and assign node ranks to each via `node_ranks` (e.g. `0-7`). For hetero setups you can attach **`env_configs`** to a node group: each entry has `node_ranks`, and optionally **`env_vars`** (list of one-key dicts) and **`python_interpreter_path`**. These YAML-specified env vars and interpreter path are applied by the scheduler when it **allocates** workers to nodes (e.g. in `Cluster.allocate` / worker launch). They override or extend the env that the node had at `ray start`: precedence is (1) default env on the node at ray start, (2) env vars set on the head between ray start and RLinf init, (3) `env_vars` from the cluster YAML for that node group. Use `env_configs` to set things like `GLOO_SOCKET_IFNAME` or per-group Python interpreters without changing the script that starts Ray. For hardware that is not auto-detected (e.g. robots), add a **`hardware`** block to the node group with the appropriate `type` and `configs`. See [heterogeneous cluster](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/hetero.html) and `rlinf/scheduler/cluster/config.py` for the full schema.

---

## Metrics, checkpoints, and evaluation

**Metrics**

Runners log metrics via `MetricLogger`, which can send them to one or more backends. Configure backends in YAML with `runner.logger.logger_backends` (e.g. `["tensorboard", "wandb", "swanlab"]`). Typical metric namespaces:

- **`train/`** – Training loss and algorithm-specific metrics (e.g. actor loss, critic loss, clip fraction, KL) from the actor worker after each update.
- **`eval/`** – Evaluation metrics when validation runs: e.g. `eval/success_once`, `eval/success_at_end`, `eval/return`, `eval/reward`, `eval/episode_len` for embodied tasks; reasoning/agent tasks may log different eval keys.
- **`env/`** – Aggregated env-side metrics (e.g. from rollout episodes) such as returns or success counts.
- **`rollout/`** – Rollout-phase metrics (e.g. from advantage/return computation).
- **`time/`** – Timing (e.g. `time/env/...`, `time/rollout/...`, `time/actor/...`) for profiling.

Logs and backend-specific directories (e.g. `tensorboard/`, `wandb/`, `swanlab/`) are written under `runner.logger.log_path`. For full setup (TensorBoard, W&B, SwanLab), see the [training visualization / logger tutorial](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/logger.html).

**Saving and resuming checkpoints**

- **Saving:** Checkpoints are written every `runner.save_interval` steps. The path pattern is `<log_path>/<experiment_name>/checkpoints/global_step_<N>/`. For **Megatron**, checkpoints live under `output_dir/experiment_name/checkpoints/` and include sharded model/optimizer/RNG and optionally `data/data.pt` (dataloader state). For **FSDP/FSDP2**, they live under `log_path/experiment_name/checkpoints/` and use DCP (`.distcp` files) plus optional full weights.
- **Resuming:** Set `runner.resume_dir` to the checkpoint directory (e.g. `.../checkpoints/global_step_50`). Relaunch the same training command (Ray and entry script). The runner loads the actor checkpoint from `resume_dir/actor`, restores the global step from the directory name, and (for reasoning/agent) may load dataloader state from `resume_dir/data/data.pt` if present. Some runners support `resume_dir: auto` to pick the latest checkpoint under the experiment. Full layout and step-by-step resume: [Checkpoint resume tutorial](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/resume.html).

**Evaluation**

- **During training:** Set `runner.val_check_interval`; every that many steps the runner can run validation (e.g. call `evaluate()`), log `eval/*` metrics, and optionally save a checkpoint (when `save_interval` is hit).
- **Standalone embodied (VLA):** Use the evaluation script and an eval config: `bash examples/embodiment/eval_embodiment.sh <config_name>`. The config must include `env.eval` and usually `runner.only_eval: True`. Set `rollout.model.model_path` to the model to evaluate and, for a specific checkpoint, `runner.ckpt_path` (e.g. a `.pt` file; convert from distributed format if needed). Control evaluation size with `env.eval.total_num_envs`, `algorithm.eval_rollout_epoch`, and `env.eval.auto_reset`; optional video via `env.eval.video_cfg.save_video`. Results (e.g. `eval/success_once`, `eval/return`) are printed and written to logs. See [VLA evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla-eval.html).
- **Standalone reasoning (LLM):** Use the [LLM evaluation tutorial](https://rlinf.readthedocs.io/en/latest/rst_source/start/llm-eval.html): convert Megatron checkpoints to HuggingFace if needed, then use the LLMEvalKit and instructions there.

---

## When things go wrong

For debugging (breakpoints, rendering/EGL, network, NCCL/CUDA, timeouts), see the [FAQ](https://rlinf.readthedocs.io/en/latest/rst_source/faq.html) in Further reading.

---

## Key ideas and plugging in

**Config** (`rlinf/config.py`): `build_config` / `validate_cfg` produce the full DictConfig. New model or env types go into `SupportedModel` / `SupportedEnvType` and validation.

**Cluster and placement:** `ClusterConfig` and strategies in `rlinf/scheduler/placement/`, `rlinf/utils/placement.py`. Placement controls where actor/rollout/env run (one node vs many, GPU vs CPU, heterogeneous).

**Algorithms:** Advantage and loss functions are registered in `rlinf/algorithms/` (registry + decorators); rewards are registered in `rlinf/algorithms/rewards/`. Config keys `algorithm.adv_type` and `algorithm.loss_type` select them. See [Extending RLinf: algorithms, models, envs](#extending-rlinf-algorithms-models-envs) for step-by-step instructions.

**Models (embodied):** Register in `SupportedModel` in `config.py`, implement under `rlinf/models/embodiment/<name>/` (e.g. `BasePolicy`), wire in config and workers. Use add-install-docker-ci-e2e for install/Docker/CI. Details in the extension section below.

**Environments:** Register in `SupportedEnvType` and `get_env_cls()` in `rlinf/envs/__init__.py`, implement under `rlinf/envs/<name>/`. Use add-install-docker-ci-e2e and add-example-doc-model-env for install and docs. Details below.

**Workers:** Subclass `Worker`, implement `initialize` and your API, launch with `create_group(...).launch(...)`. Use `self.log_info` / `log_warning` / `log_error`; no print.

**Runners:** They own the training loop. New task type = new runner + entry script that builds Cluster, placement, worker groups, and calls the runner.

---

## Extending RLinf: algorithms, models, envs

### New algorithms (advantage, loss, reward)

**Advantage function**

- Implement a function that takes the same keyword args as existing ones (e.g. `rewards`, `values`, `dones`, `gamma`, `loss_mask`, …) and returns `(advantages, returns)`. See `rlinf/algorithms/advantages.py` (e.g. `compute_gae_advantages_and_returns`) for signatures.
- Register it: `from rlinf.algorithms.registry import register_advantage` then `@register_advantage("my_adv")` on your function. The name is case-normalized to lowercase.
- In config YAML set `algorithm.adv_type: my_adv`. Actor workers call `calculate_adv_and_returns(adv_type=...)` which dispatches via `get_adv_and_returns(name)`.
- For non-GAE styles (e.g. GRPO, Reinforce++), `rlinf/algorithms/utils.py` may need to compute scores first; check how `adv_type` is used in `calculate_adv_and_returns` and in the actor worker.

**Policy loss**

- Implement a function that accepts the kwargs passed by the actor (e.g. `logprobs`, `old_logprobs`, `advantages`, `clip_ratio_low`, `clip_ratio_high`, `loss_mask`, …) and returns `(loss_tensor, metrics_dict)`. See `rlinf/algorithms/losses.py` (e.g. `compute_ppo_actor_loss`, `compute_ppo_actor_critic_loss`, `compute_grpo_actor_loss_fn`).
- Register: `from rlinf.algorithms.registry import register_policy_loss` then `@register_policy_loss("my_loss")`.
- In config set `algorithm.loss_type: my_loss`. For PPO-style actor+critic you need a critic and value loss; the unified entry is `policy_loss(loss_type=..., **kwargs)` in `registry.py`. Add validation in `rlinf/config.py` if your loss has special requirements (e.g. `validate_cfg` already checks `loss_type == "actor_critic"` for value head).

**Reward**

- Add a reward class (e.g. under `rlinf/algorithms/rewards/<domain>/`) that matches the interface expected by the reward worker (e.g. callable or class with a clear contract for prompt/completions/ids).
- In `rlinf/algorithms/rewards/__init__.py`: import the class, then `register_reward("my_reward", MyRewardClass)`. The registry is `reward_registry`; lookup via `get_reward_class(name)`.
- Wire the reward name in config and in the runner/reward worker so the correct class is instantiated and used. For reasoning/agent tasks the config path may be under `reward.path` or similar.

### New embodied model

- **Registration:** In `rlinf/config.py`, add a new value to the `SupportedModel` enum: `MY_MODEL = ("my_model", "embodied")`. Use `get_supported_model(model_type)` in validation so `model.model_type: my_model` is accepted.
- **Implementation:** Create a package under `rlinf/models/embodiment/my_model/`. For policies that fit the embodied actor interface, inherit from `rlinf.models.embodiment.base_policy.BasePolicy` and implement `default_forward` and `predict_action_batch`; add other forward types (e.g. `sac_forward`, `crossq_forward`) if the algorithm needs them. For HuggingFace-based VLAs, follow the pattern in the docs: register config and processor in `rlinf/models/__init__.py` (`get_model_config_and_processor`), then implement an action model that wraps generation and optional value head.
- **Config and workers:** Ensure `build_config` / default configs provide the right `model.model_type`, checkpoint paths, and any model-specific options. Actor and rollout workers already branch on `cfg.actor.model.model_type` / `cfg.rollout.model.model_type`; add branches or a factory so your model is instantiated and used. For FSDP+HuggingFace, see the [new model (FSDP) tutorial](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_fsdp.html); for Megatron there is a separate [new model (Megatron) tutorial](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_megatron.html).
- **Install and CI:** If the model needs extra deps or a dedicated venv, add it to `requirements/install.sh` (e.g. `SUPPORTED_MODELS`, and an `install_my_model()` or branch in the model switch). For Docker and e2e: use the skill `.cursor/skills/add-install-docker-ci-e2e` (install script, Dockerfile stage, CI job, e2e config under `tests/e2e_tests/embodied/`).

### New environment

- **Registration:** In `rlinf/envs/__init__.py`, add a member to `SupportedEnvType`: e.g. `MY_ENV = "my_env"`. In `get_env_cls(env_type, env_cfg=None, ...)` add an `elif env_type == SupportedEnvType.MY_ENV:` branch that imports your env class and returns it (lazy import to avoid loading heavy deps at import time). If the env needs a task id (like IsaacLab), use `env_cfg` and document the expected shape.
- **Implementation:** Create `rlinf/envs/my_env/` with at least one module defining a gym-style env (e.g. `gymnasium.Env`): `reset`, `step`, and the usual attributes (`observation_space`, `action_space`). Follow the [new environment tutorial](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_env.html) for the expected structure (e.g. vectorized `num_envs`, `group_size`, `ret_device`). If your env uses custom action formatting, add a branch in `rlinf/envs/action_utils.py` in `prepare_actions(env_type, ...)` so rollout/workers pass correctly shaped actions.
- **Config:** Set `env.train.env_type` and `env.eval.env_type` to the string value of your enum (e.g. `my_env`). Add any env-specific defaults or validation in `rlinf/config.py` (e.g. `validate_cfg` already has env-specific checks for ManiSkill, Behavior, etc.; add similar ones if needed).
- **Install and docs:** For install/Docker/CI, use `.cursor/skills/add-install-docker-ci-e2e` (add env to `SUPPORTED_ENVS`, install logic, e2e config). For example docs and RST, use `.cursor/skills/add-example-doc-model-env`.

---

## Style and contributing

Google Python style; Ruff for lint/format; docstrings and type hints on public APIs. Logging: `rlinf.utils.logging.get_logger()` or Workers’ `self.log_*`. Config YAML: static values only; no computed fields; don’t overwrite user-facing fields in code. Commits: [Conventional Commits](https://www.conventionalcommits.org/), ~72-char subject, imperative; every commit `Signed-off-by:` (e.g. `git commit -s`). PRs: same title format, fill template, link issues; for perf-sensitive changes include test results. New behavior needs tests (unit or e2e); if e2e needs GPUs/hardware, document and skip appropriately in CI. Full details: [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Further reading

- [Docs (EN)](https://rlinf.readthedocs.io/en/latest/) · [中文](https://rlinf.readthedocs.io/zh-cn/latest/)
- [Installation](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html) · [VLA quickstart](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html)
- [Example gallery](https://rlinf.readthedocs.io/en/latest/rst_source/examples/index.html) · configs in `examples/embodiment/config/`, `examples/reasoning/`, etc.
- Tutorials: [placement / cluster / YAML](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/index.html), [hybrid / disaggregated](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/mode/index.html), [heterogeneous cluster](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/hetero.html), [extend (new env/model)](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/index.html), [RL algorithms](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/rlalg/index.html), [logger (metrics)](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/logger.html), [checkpoint resume](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/resume.html)
- Evaluation: [VLA evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla-eval.html) · [LLM evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/start/llm-eval.html)
- [APIs](https://rlinf.readthedocs.io/en/latest/rst_source/apis/index.html) (actor, channel, cluster, placement, worker, env, data, …) · [FAQ](https://rlinf.readthedocs.io/en/latest/rst_source/faq.html)
