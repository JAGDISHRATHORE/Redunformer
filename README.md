# Redunformer

Cross-granularity redundancy analysis for large language models (SoSe 2026).

**Group:** Weight-level redundancy  
**Branch:** `Weight`

## Quick start

Requires **NVIDIA GPU** with Docker GPU support (`--gpus all`) or Podman on mlsp (`--device nvidia.com/gpu=all`).

**Driver note:** the container ships `torch 2.12+cu130`. Host NVIDIA driver must support **CUDA 13.x** (e.g. driver **610+** on Windows). Older drivers fail with `NVIDIA driver too old`.

**Windows:** install [GNU Make](https://strawberryperl.com/) (e.g. Strawberry Perl → `C:\Strawberry\c\bin` on PATH) and Docker Desktop with GPU support.

**Validated locally (Weight branch):** `make verify` + `make smoke` on RTX 3060 Laptop — see `reports/weight_level/week1-2-baseline.md`.

### 1. Build the container

```bash
make build
# on mlsp student pool:
make CONTAINER=podman build
```

### 2. Verify setup (fast, no model download)

```bash
make verify
```

Checks imports and configs inside the GPU container. Downloads nothing from Hugging Face.

### 3. Run baseline evaluations (GPU)

```bash
make smoke              # gpt2, limit=0.01 by default
make smoke LIMIT=0.05
make qwen-small         # Qwen3-1.7B
make qwen               # Qwen3-4B
```

On mlsp:

```bash
make CONTAINER=podman verify
make CONTAINER=podman qwen
```

### 4. Run without Make

```bash
docker run --rm --gpus all \
  -v "$(pwd)/experiments:/app/experiments" \
  -v redunformer_hf_cache:/root/.cache/huggingface \
  redunformer \
  python scripts/run_baseline.py --config configs/models/gpt2.yaml --limit 0.05
```

## Model configs

Model choice is a YAML file under `configs/models/`:

| Config | Model | Use case |
|--------|-------|----------|
| `gpt2.yaml` | `gpt2` | Smoke test |
| `qwen3-1.7b.yaml` | `Qwen/Qwen3-1.7B-Instruct` | Light GPU baseline |
| `qwen3-4b.yaml` | `Qwen/Qwen3-4B-Instruct` | Main baseline |

Change `pretrained`, `tasks`, `seed`, and `dtype` in the YAML. Weights are downloaded from Hugging Face on first run and cached in the `redunformer_hf_cache` volume.

Recommended: **8GB+ VRAM** for Qwen3-4B. RTX 3060 Laptop (6 GB) — try `make qwen-small` first.

## Weeks 1–2 checklist

| Done | Task |
|------|------|
| yes | Repo scaffold, Docker + uv, lm-eval wiring |
| yes | `make verify`, `make smoke` (gpt2 on GPU) |
| no | `make qwen-small` or `make qwen` (main baseline) |
| no | Note in `reports/weight_level/week1-2-baseline.md` with Qwen metrics |

Details and smoke results: `reports/weight_level/week1-2-baseline.md`.

## Project layout

```text
configs/models/     model + eval configs
src/redundancy/     shared library (models, eval, plotting)
scripts/            CLI entrypoints
experiments/        JSON results (gitignored)
reports/weight_level/  group notes and deliverables
```

## Reproducibility

Each run writes:

- `experiments/baseline/<model>.json` from lm-eval
- `experiments/baseline/<model>.meta.json` with config path, command, seed, tasks, timestamp

## Portable setup

On a GPU machine (local 3060+ or mlsp pool):

1. `git clone` + `cd Redunformer` + `git checkout Weight`
2. `make build`
3. `make verify`
4. `make qwen` (or `make smoke` first)

Edit code on any machine; run `make` targets only where NVIDIA + container GPU support works.

On mlsp: `make CONTAINER=podman ...`, SSH with your pool username, RTPT, mlstudentpool Mattermost channel.

## Development with uv (optional, outside Docker)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python scripts/run_baseline.py --config configs/models/gpt2.yaml --limit 0.05
```
