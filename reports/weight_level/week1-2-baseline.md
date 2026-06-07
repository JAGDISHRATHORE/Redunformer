# Weeks 1–2: Baseline pipeline (Weight group)

**Branch:** `Weight`  
**Status:** complete  
**Detailed run log:** [baseline_runs_2026-06-06.md](baseline_runs_2026-06-06.md)

## Goal (seminar Weeks 1–2)

- Shared baseline pipeline in repo (Docker + uv + lm-eval)
- Load ≥1 HF causal LM and ≥1 benchmark dataset
- Run baseline evaluation, save structured JSON + short note

---

## Deliverable checklist

| PDF requirement | Status |
|-----------------|--------|
| Clone repo, branch `Weight` | Done |
| Python env with uv (`pyproject.toml`, `uv.lock`, Dockerfile) | Done |
| Load HF causal LM | Done — gpt2, Qwen3-1.7B, Qwen3-4B |
| Load dataset(s) | Done — hellaswag, piqa, arc_easy (via lm-eval) |
| Baseline eval (lm-eval-harness) | Done |
| JSON + `.meta.json` (command, seed) | Done (local, gitignored) |
| Short note with model, tasks, command, metrics | Done (this file + `baseline_runs_2026-06-06.md`) |

---

## Pipeline status

| Step | Command | Machine | Status |
|------|---------|---------|--------|
| Docker image | `make build` | RTX 3060 Laptop | Done |
| Env check | `make verify` | RTX 3060 Laptop | Done |
| Smoke | `make smoke` | RTX 3060 Laptop | Done |
| Interim baseline | `make qwen-small` | RTX 3060 Laptop | Done |
| **Main baseline** | `make CONTAINER=podman qwen` | **mlsp4** | Done |

---

## Main baseline (reference for pruning)

**Model:** `Qwen/Qwen3-4B` (open-weight Qwen3, no `-Instruct` suffix on HF)  
**Why this model:** seminar model list includes Qwen3; 4B is our group main baseline; weight-level pruning (magnitude, later SparseGPT) will use the same checkpoint.

**Eval:** lm-evaluation-harness, 0-shot multiple choice (log-likelihood), seed=42, bfloat16.

**Command (mlsp4):**

```bash
make CONTAINER=podman qwen
# equivalent: python scripts/run_baseline.py --config configs/models/qwen3-4b.yaml
```

**Tasks:** hellaswag, piqa, arc_easy

| Task | acc | acc_norm |
|------|-----|----------|
| hellaswag | 0.5214 | **0.6844** |
| piqa | 0.7492 | 0.7476 |
| arc_easy | 0.8060 | 0.7849 |

Full tables, runtime, and comparison with 1.7B: [baseline_runs_2026-06-06.md](baseline_runs_2026-06-06.md).

**Artifacts (local, gitignored):**

- `experiments/baseline/Qwen_Qwen3-4B.json`
- `experiments/baseline/Qwen_Qwen3-4B.meta.json`

---

## Other runs (sanity / interim)

| Run | Command | Purpose |
|-----|---------|---------|
| gpt2 smoke | `make smoke` | Pipeline sanity check (hellaswag 1%) |
| Qwen3-1.7B | `make qwen-small` | Interim baseline on 6 GB laptop VRAM |

Metrics for both: [baseline_runs_2026-06-06.md](baseline_runs_2026-06-06.md).

---

## Infrastructure notes

- **Local laptop:** Windows, RTX 3060 6 GB, Docker, driver 610.47+ (CUDA 13.x for container torch).
- **mlsp4:** RTX 2080 Ti 11 GB, podman. Dockerfile on server uses `docker.io/nvidia/cuda:...` for podman registry resolution.
- **mlsp access:** SSH to mlsp2/4/7 from TU network (`130.83.*`); VPN blocked from Wohnheim — use campus WiFi or eduroam.
- **Reproducibility:** each run logs model, tasks, seed, full command in `.meta.json`.

---

## Reproducibility checklist

- [x] model id (`Qwen/Qwen3-4B`)
- [x] tasks (hellaswag, piqa, arc_easy)
- [x] seed (`42`)
- [x] full eval command in `.meta.json`
- [x] baseline metrics recorded in report
