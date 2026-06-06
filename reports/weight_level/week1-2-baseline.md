# Weeks 1–2: Baseline pipeline (Weight group)

**Branch:** `Weight`  
**Status:** pipeline validated on GPU; Qwen baseline pending

## Goal (seminar Weeks 1–2)

- Shared baseline pipeline in repo (Docker + uv + lm-eval)
- Load ≥1 HF causal LM and ≥1 benchmark dataset
- Save structured JSON results + short note with model, tasks, command, metrics

---

## Pipeline status

| Step | Command | Status | Notes |
|------|---------|--------|-------|
| Clone + checkout | `git checkout Weight` | Done | |
| Docker image | `make build` | Done | ~12 min, image ~29 GB |
| Env check | `make verify` | Done | imports + 3 YAML configs, no HF download |
| Smoke (gpt2) | `make smoke` | Done | GPU `cuda:0`, RTX 3060 Laptop |
| Qwen baseline | `make qwen-small` / `make qwen` | **TODO** | main model for group report |
| Note (this file) | — | In progress | fill Qwen results after run |

---

## Validated run: gpt2 smoke

**Machine:** Windows, RTX 3060 Laptop, Docker Desktop  
**Driver:** NVIDIA 610.47 (CUDA UMD 13.3) — required for `torch 2.12+cu130` in container  
**Earlier failure:** driver 566.36 → `NVIDIA driver too old (12070)` until update

**Command:**

```bash
make smoke
# equivalent: python scripts/run_baseline.py --config configs/models/gpt2.yaml --limit 0.01
```

**Config:** `configs/models/gpt2.yaml`  
- Model: `gpt2`  
- Task: `hellaswag`  
- Seed: `42`  
- dtype: `float32`  
- limit: `0.01` (Makefile default; yaml default is `0.05`)

**Results (hellaswag, 1% subset):**

| Metric | Value |
|--------|-------|
| acc | 0.3564 |
| acc_norm | 0.4356 |

**Artifacts (local, gitignored):**

- `experiments/baseline/gpt2.json`
- `experiments/baseline/gpt2.meta.json`

CPU-only smoke was also run manually before driver fix — same acc (~0.356), confirms pipeline logic.

---

## TODO before Weeks 1–2 is complete

1. **`make qwen-small`** (Qwen3-1.7B) or **`make qwen`** (Qwen3-4B) on GPU  
   - Record all task metrics from JSON  
   - Copy command from `.meta.json`
2. **Update this note** with Qwen model choice justification (why Qwen3, size vs VRAM)
3. **Optional:** announce working pipeline in group Mattermost channel (supervisor suggestion)
4. **Optional:** PR from `Weight` if group wants review (not required by PDF)

No SSH / mlsp required for closing Weeks 1–2 if local GPU run succeeds.

---

## Model choice (draft)

**Planned main baseline:** `Qwen/Qwen3-4B-Instruct` (or `1.7B` if 6 GB VRAM tight on laptop)

- Open-weight, fits seminar model list (Qwen3)
- Same pipeline as gpt2 smoke — only YAML + `make` target changes
- Weight-level pruning experiments (SparseGPT) will target this model later

---

## Reproducibility checklist

Each eval run must log (auto in `.meta.json`):

- [x] model id (`gpt2`)
- [x] tasks (`hellaswag`)
- [x] seed (`42`)
- [x] full eval command
- [ ] Qwen run — pending
