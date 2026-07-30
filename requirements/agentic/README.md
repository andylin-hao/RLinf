# Agentic engine requirements

`sglang` and `vllm` are not pyproject extras: `uv sync` builds one universal lock
across every extra, so a single engine pin that only resolves on one CUDA line
(sglang >= 0.5.11 requires `cuda-python >= 13`) would make even an embodied
install unsatisfiable. Each supported engine build gets a file here instead, and
the set of files *is* the set of supported versions.

## Layout

    <engine>_<version>_<cu12|cu13>.txt   one supported engine build
    <engine>_<version>_common.txt        deps shared by that version's CUDA lines

The CUDA line is picked from the driver by `install.sh` (`agentic_cuda_line`).

## One venv per engine

A venv holds exactly one engine, chosen with `--engine`:

    bash requirements/install.sh agentic --engine sglang
    bash requirements/install.sh agentic --venv .venv-vllm --engine vllm

sglang and vLLM pin the same kernel libraries to different versions
(`nvidia-cutlass-dsl`, `flashinfer-python`, `tilelang`, `tokenspeed-mla`), and
sometimes a different torch, so in a shared venv whichever is installed second
downgrades the other's kernels — silently, until a kernel actually runs. Keeping
them apart is also why each file can simply follow its own engine's declared
pins. The one exception, `install.sh docs`, installs both because autodoc has to
import them and never launches a kernel.

## How a file is installed

`install.sh` installs the dependency lines normally and then the engine wheel
itself with `--no-deps`, taking the wheel from the `# engine:` header:

    # engine: sglang==0.5.12.post1

`--no-deps` is what makes a CUDA 12 build of the sglang >= 0.5.11 line possible:
the wheel's metadata pins `cuda-python >= 13`, `nvidia-cutlass-dsl[cu13]` and
`sglang-kernel`/`sgl-deep-gemm` wheels that link `libcudart.so.13`. Upstream
publishes cu12 builds of all of those:

| what | where |
| --- | --- |
| `sglang-kernel`, `sgl-deep-gemm` | <https://docs.sglang.ai/whl/cu129/> |
| `vllm` | <https://wheels.vllm.ai/0.23.0/cu129/> |
| `nvidia-cutlass-dsl`, `flashinfer-python` | PyPI, via the `cu12` extra instead of `cu13` |

The two index-hosted wheels are pinned by URL rather than through an extra
index, so resolution cannot drift onto a nightly and does not depend on uv's
index-strategy. Compare THUDM/slime's `build_conda.sh`, which reaches the same
place by installing sglang normally and then force-reinstalling the cu12 wheels
over the cu13 ones; going through `--no-deps` avoids the repair step, and with
it the risk slime documents of cu13 `nvidia-*` wheels overwriting their cu12
counterparts inside the shared `site-packages/nvidia/*` directories.

It also keeps the engines from dragging the rest of the venv backwards. vllm
0.8.5 requires `opentelemetry-sdk < 1.27`, whose `opentelemetry-proto` requires
`protobuf < 5`, which breaks `tensorboard` (its generated `event_pb2` needs
protoc >= 5.27), while `ray >= 2.48` requires `opentelemetry-sdk >= 1.30`. vllm
imports opentelemetry lazily and runs fine without it, so the otel packages are
simply not listed.

## Regenerating a file

The dependency lines are the engine's own `requires_dist` minus the torch family
(`pyproject.toml` owns torch/torchvision/torchaudio/torchcodec) with the CUDA 13
pins swapped for their CUDA 12 counterparts. After bumping a version, check the
list still covers what upstream declares — a dependency dropped here only shows
up as an ImportError at rollout time:

    python - <<'EOF'
    import json, re, urllib.request, pathlib
    PKG, VER = "sglang", "0.5.12.post1"
    FILES = ["sglang_0.5.12.post1_common.txt", "sglang_0.5.12.post1_cu12.txt"]
    EXTRAS = {"tracing", "http2", "all"}          # the extras install.sh wants
    norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()

    have = set()
    for f in FILES:
        for line in pathlib.Path(f).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-r ", "--")):
                have.add(norm(re.split(r"[<>=!~;\[ @]", line, 1)[0]))

    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{PKG}/{VER}/json"))
    need = set()
    for r in meta["info"]["requires_dist"]:
        m = re.search(r'extra\s*==\s*"([^"]+)"', r)
        if m and m.group(1) not in EXTRAS:
            continue
        need.add(norm(re.split(r"[<>=!~;\[ @]", r.strip(), 1)[0]))

    torch_family = {"torch", "torchvision", "torchaudio", "torchcodec", norm(PKG)}
    print("missing:", sorted(need - have - torch_family))
    EOF
