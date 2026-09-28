# Experiment 3 – Llama-3.2-3B-Instruct (2nd model)

## What we did here

So all of our main results earlier came only from one model, Qwen3-4 But we wanted to test if they are true in general or if the results only hold for one model. That's why we took a second model to extend this lab, and we took it from a different family, Llama-3.2-3B-Instruct. We repeated the most important tests on it and followed the same idea as before. We cut the weight tables into 32x32 blocks (tiles), deleted some of them, sometimes also repaired the remaining weights to compare if repair works or not, and then checked how much the model got damaged. We looked at perplexity, which tells us how well the model predicts text, and at real multiple-choice tasks, which tell us how much of the model's actual ability is left. Then we compared everything with the Qwen results to see what stays the same and what is different.

## Setup

- Model: `meta-llama/Llama-3.2-3B-Instruct` (needs a Hugging Face token)
- We used the same code as for Qwen3-4B, only with a different `--model`. Llama uses the same layer names (`self_attn.q_proj`, `mlp.up_proj`, ...), so we didn't have to change anything in the code.
- Tiles 32x32, WikiText-2 full test set, 64 calibration pieces, same % of pruning everywhere
- Dense perplexity **10.45** (taken from `wholemodel_magnitude_p0.json`, which is a 0% run)
- Dense accuracy: HellaSwag 0.716, PIQA 0.768, ARC-Easy 0.709

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

`sparsegpt_recon` on Llama at higher sparsity: 21.2 (10%), 104.9 (20%), 359.6 (30%).

**Real tasks, retained ability with `sparsegpt_recon`** (`results/downstream/`)

| Sparsity | 1% | 2% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|---|
| Llama | 99% | 97% | 89% | 69% | 20% | 10% |
| Qwen3-4B | 100% | 96% | 90% | 79% | 43% | 23% |

At 5%: `wanda_recon` kept 91%, and `random_recon` kept 86% / 77% / 86% (3 seeds).

**1x1 (single weights)** (`results/tile_size_1x1/`)
Perplexity 10.56 / 11.00 / 11.97 / 14.73 at 20/30/40/50%. Retained ability 96% at 40% and 86% at 50%, while with 32x32 tiles it is already down to 20% at 20% sparsity.

## What is the same and what is different

Same as Qwen:
- The ~5% limit for 32x32 tiles is there again (89% vs 90% retained).
- Repair is what makes it work, all the `*_recon` methods stay close to the dense model at 5%.
- 1x1 keeps much more than 32x32, so deleting in blocks costs a lot here too.

Different from the Qwen:
- Magnitude pruning is only the worst method, it doesn't completely break the model (21.9 vs 3449).
- Llama drops faster after 5% (20% retained at 20% sparsity, Qwen still had 43%).
- `random_recon` changes more between seeds on the real tasks.

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
