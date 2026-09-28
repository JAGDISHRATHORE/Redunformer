# Experiment 3 – Llama-3.2-3B-Instruct (second model)

August 2026. All Qwen3-4B results were from one model, so we checked the most important ones on a second model.

## Setup

- Model: `meta-llama/Llama-3.2-3B-Instruct` (needs a Hugging Face token)
- Same code as Experiment 2, only `--model` changed. Llama uses the same layer names
  (`self_attn.q_proj`, `mlp.up_proj`, ...), so nothing had to be changed in the code.
- Tiles 32x32, WikiText-2 full test set, 64 calibration windows, uniform pruning
- Dense perplexity **10.45** (from `wholemodel_magnitude_p0.json`, a 0% run)
- Dense accuracy HellaSwag 0.716, PIQA 0.768, ARC-Easy 0.709

## Results

**Whole model, perplexity at 5%** (`results/whole_model_sweep/`)

| Method | Llama ppl @ 5% | Qwen3-4B ppl @ 5% |
|---|---|---|
| magnitude | 21.9 | 3448.9 |
| random (3 seeds) | 17.5 / 18.1 / 15.6 | 22.9 / 20.2 / 48.0 |
| wanda | 16.6 | 28.3 |
| sparsegpt | 14.8 | 28.0 |
| sparsegpt_recon | 12.8 | 15.6 |
| wanda_recon | 12.6 | 15.4 |
| random_recon (3 seeds) | 13.0 / 13.0 / 12.9 | 14.3 (seed 1) |

sparsegpt_recon on Llama: 21.2 (10%), 104.9 (20%), 359.6 (30%).

**Downstream, retained ability with sparsegpt_recon** (`results/downstream/`)

| Sparsity | 1% | 2% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|---|
| Llama | 99% | 97% | 89% | 69% | 20% | 10% |
| Qwen3-4B | 100% | 96% | 90% | 79% | 43% | 23% |

At 5%: `wanda_recon` 91%, `random_recon` 86% / 77% / 86% (3 seeds).

**1x1 unstructured** (`results/tile_size_1x1/`)
Perplexity 10.56 / 11.00 / 11.97 / 14.73 at 20/30/40/50%. Retained ability 96% at 40% and 86% at 50%
(32x32 is already down to 20% at 20% sparsity).

## What carries over and what not

Same as Qwen:
- The ~5% limit for 32x32 tiles (89% vs 90% retained).
- Repair is what makes it work (all `*_recon` methods are close to dense at 5%).
- 1x1 keeps much more than 32x32, so the structured-pruning cost is there too.

Different from Qwen:
- Magnitude pruning is only the worst method, not a total collapse (21.9 vs 3449).
- Llama drops faster above 5% (20% retained at 20% sparsity vs 43% for Qwen).
- `random_recon` has more variance between seeds on downstream.

We only have 1 seed for most settings and 3 for the random ones, so the differences should be checked with more
seeds before relying on them. We did not run screening, Policy B, depth concentration or iterative calibration on
Llama.

## Files

- `results/whole_model_sweep/` – perplexity runs (`run_tile_pruning.py --whole-model`)
- `results/downstream/` – lm-eval runs (`run_downstream_eval.py`)
- `results/tile_size_1x1/` – 1x1 runs (`run_unstructured_pruning.py`)
- `results/smoke_tests/` – two short test runs (32-64 examples per task) we did first to check the setup
- `figures/` – `f1_qwen_vs_llama.png` and `f10_llama_1x1_vs_32.png`, made by `analysis/plot_llama_vs_qwen.py`

The Qwen numbers in `plot_llama_vs_qwen.py` are typed in from Experiment 2 (retained ability of `sparsegpt_recon`).

## How to run (examples)

```bash
uv run python scripts/run_tile_pruning.py --model meta-llama/Llama-3.2-3B-Instruct --method sparsegpt_recon --prune-ratio 0.05 --whole-model --calib-samples 64 --experiment-dir experiments/experiment_3_llama3_2_3b/results/whole_model_sweep
uv run python scripts/run_downstream_eval.py --model meta-llama/Llama-3.2-3B-Instruct --dense --output experiments/experiment_3_llama3_2_3b/results/downstream/downstream_dense.json
uv run python scripts/run_downstream_eval.py --model meta-llama/Llama-3.2-3B-Instruct --method sparsegpt_recon --prune-ratio 0.05 --output experiments/experiment_3_llama3_2_3b/results/downstream/downstream_sparsegpt_recon_p5_uniform.json
uv run python scripts/run_unstructured_pruning.py --model meta-llama/Llama-3.2-3B-Instruct --method unstructured_sparsegpt --prune-ratio 0.5 --calib-samples 64 --output experiments/experiment_3_llama3_2_3b/results/tile_size_1x1/downstream/us_sparsegpt_recon_p50_T1.json
python experiments/experiment_3_llama3_2_3b/analysis/plot_llama_vs_qwen.py
```
