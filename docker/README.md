## Building Docker Images

RLinf provides a unified Dockerfile for both the math reasoning image and the various embodied images. Use the `BUILD_TARGET` build argument to select which image to build:

- `reason` — math reasoning image
- `embodied-<env>` or `embodied-<env>-<model>` — embodied image for a specific environment (and optionally a specific model when multiple model flavors exist for the same env)

To build the Docker image, run the following command **in the RLinf root directory**:

```shell
export BUILD_TARGET=reason # or one of the embodied-* targets listed below
docker build -f docker/Dockerfile --build-arg BUILD_TARGET=$BUILD_TARGET -t rlinf:$BUILD_TARGET .
```

### Available `BUILD_TARGET` values

| `BUILD_TARGET` | Environment | Installed venvs (`/opt/venv/<name>`) |
| --- | --- | --- |
| `reason` | (math reasoning, no simulator) | `reason` |
| `embodied-maniskill_libero` | ManiSkill / LIBERO | `openvla`, `openvla-oft`, `openpi`, `gr00t`, `gr00t_n1d6`, `dexbotic`, `starvla`, `abot_m0` |
| `embodied-behavior-openvlaoft` | BEHAVIOR | `openvla-oft` |
| `embodied-behavior-openpi` | BEHAVIOR | `openpi` |
| `embodied-metaworld` | MetaWorld | `openvla-oft`, `openpi` |
| `embodied-calvin` | CALVIN | `openvla-oft`, `openpi` |
| `embodied-robocasa` | RoboCasa | `openpi` |
| `embodied-isaaclab` | IsaacLab | `gr00t`, `openpi` |
| `embodied-franka` | Franka (Ubuntu 20.04 + ROS Noetic, GPU-agnostic) | `franka-0.10.0`, `franka-0.13.3`, `franka-0.14.1`, `franka-0.15.0`, `franka-0.18.0`, `franka-0.19.0` |
| `embodied-robotwin` | RoboTwin | `openvla-oft`, `openpi`, `lingbotvla` |
| `embodied-opensora` | OpenSora | `openvla-oft` |
| `embodied-wan` | WAN | `openvla-oft` |
| `embodied-frankasim` | FrankaSim | `openvla` |
| `embodied-embodichain` | EmbodiChain | `embodichain` |
| `embodied-libero` | LIBERO | `openvla`, `openvla-oft` |
| `embodied-liberopro` | LIBERO-Pro | `openvla-oft` |
| `embodied-liberoplus` | LIBERO-Plus | `openvla-oft` |
| `embodied-roboverse` | RoboVerse | `openpi` |
| `embodied-polaris` | Polaris | `openpi` |
| `embodied-dreamzero` | (SFT only, no simulator) | `dreamzero` |
| `embodied-dreamzero-libero` | LIBERO | `dreamzero-libero` |

### Additional build arguments

- `PLATFORM` (default `nvidia`) — hardware platform: `nvidia` (CUDA), `amd` (ROCm), or `ascend` (CANN). Selects the base image and is also recorded as `RLINF_PLATFORM` in the final image. The `embodied-franka` target ignores `PLATFORM` and always uses a plain `ubuntu:20.04` base.
- Per-platform runtime versions: `CUDA_VER`, `ROCM_VER`, `ROCM_ARCHS`, `CANN_VER`, `UBUNTU_VER`. Override any of these to bump versions without changing the rest of the build. For a fully custom base, set `NVIDIA_BASE_IMAGE`, `AMD_BASE_IMAGE`, or `ASCEND_BASE_IMAGE` directly.
- `NO_MIRROR` — set to `1` to skip the USTC apt/pypi mirror rewrites (recommended outside of mainland China).

Example with non-default args:

```shell
docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-metaworld \
    --build-arg PLATFORM=nvidia \
    --build-arg CUDA_VER=12.4.1 \
    --build-arg NO_MIRROR=1 \
    -t rlinf:embodied-metaworld .
```

# Using the Docker Image

The built Docker image contains one or more Python virtual environments (venvs) under `/opt/venv/`. Which venvs are present, and which one is activated by default in new shells, depends on the `BUILD_TARGET` — see the table above.

To switch between venvs, use the built-in `switch_env` script:

```shell
source switch_env <env_name> # e.g., source switch_env openvla-oft, source switch_env openpi, etc.
```