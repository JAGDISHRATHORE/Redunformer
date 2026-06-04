# Redunformer

Cross-granularity redundancy analysis for large language models (SoSe 2026).

**Group:** Weight-level redundancy  
**Branch:** `Weight`

## Quick start

Requires **NVIDIA GPU** with Docker GPU support (`--gpus all`) or Podman on mlsp (`--device nvidia.com/gpu=all`)

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

Recommended: **8GB+ VRAM** for Qwen3-4B (e.g. RTX 3060 or better).

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
