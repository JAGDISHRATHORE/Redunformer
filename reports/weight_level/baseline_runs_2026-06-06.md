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

## Still pending

- `make qwen` (Qwen3-4B, hellaswag + piqa + arc_easy)
