# Baseline eval runs — 2026-06-06

Local GPU: RTX 3060 Laptop (6 GB), driver 610.47, Docker + `make`.

Raw JSON outputs stay in `experiments/baseline/` (gitignored). Each run also writes a `.meta.json` with the full command and seed.

## Smoke: gpt2

- **Config:** `configs/models/gpt2.yaml`
- **Command:** `make smoke` (limit=0.01)
- **Model:** `gpt2`, float32, seed=42
- **Task:** hellaswag

| Metric   | Value  | Stderr |
|----------|--------|--------|
| acc      | 0.3564 | 0.0479 |
| acc_norm | 0.4356 | 0.0496 |

- **Output:** `experiments/baseline/gpt2.json`
- **Meta:** `experiments/baseline/gpt2.meta.json`
- **Timestamp:** 2026-06-06T03:09:33Z

## Baseline: Qwen3-1.7B

- **Config:** `configs/models/qwen3-1.7b.yaml`
- **Command:** `make qwen-small`
- **Model:** `Qwen/Qwen3-1.7B`, bfloat16, seed=42, full eval (no limit)
- **Tasks:** hellaswag, piqa
- **Runtime:** ~47 min, batch_size auto → 32

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.4612 | 0.6038   | 10042 |
| piqa      | 0.7258 | 0.7203   | 1838  |

- **Output:** `experiments/baseline/Qwen_Qwen3-1.7B.json`
- **Meta:** `experiments/baseline/Qwen_Qwen3-1.7B.meta.json`
- **Timestamp:** 2026-06-06T04:09:32Z

## Config fix

Official Hugging Face model IDs do not use the `-Instruct` suffix:

- `Qwen/Qwen3-1.7B-Instruct` → `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B-Instruct` → `Qwen/Qwen3-4B`

## Main baseline: Qwen3-4B (mlsp4)

- **Machine:** mlsp4, RTX 2080 Ti (11 GB VRAM), AMD Ryzen 7 3800X, 64 GB RAM
- **Container:** podman (`make CONTAINER=podman qwen`); Dockerfile patched to `docker.io/nvidia/cuda:...` for podman short-name resolution
- **Config:** `configs/models/qwen3-4b.yaml`
- **Command:** `make CONTAINER=podman qwen`
- **Model:** `Qwen/Qwen3-4B`, bfloat16, seed=42, full eval (no limit)
- **Tasks:** hellaswag, piqa, arc_easy
- **Runtime:** ~87 min, batch_size auto → 9

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

Reference baseline before pruning. 0-shot multiple choice via lm-eval log-likelihood.

- **Output:** `experiments/baseline/Qwen_Qwen3-4B.json` (local copy from mlsp run)
- **Meta:** `experiments/baseline/Qwen_Qwen3-4B.meta.json`
- **Timestamp:** 2026-06-06T23:50:54Z

### 1.7B vs 4B (same tasks where comparable)

| Task      | 1.7B acc_norm | 4B acc_norm |
|-----------|---------------|-------------|
| hellaswag | 0.6038        | **0.6844**  |
| piqa      | 0.7203        | **0.7476**  |

## Weeks 1–2 status

All baseline deliverables complete: pipeline, smoke (gpt2), interim (1.7B), **main baseline (4B on mlsp4)**.
