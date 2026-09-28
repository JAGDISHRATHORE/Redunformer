# Experiment 2 – Qwen3-4B (main study)

Mid July to August 2026. This is the main part of the project.

## Setup

- Model: `Qwen/Qwen3-4B` (36 layers, bf16), 7 linear matrices per layer
- Tiles: **32x32** → 98,560 tiles per layer, 3,548,160 in total
- Perplexity: WikiText-2 test. Dense = **13.22** on the full test set, **13.559** on the first 20% (we used the
  20% subset for the fast per-layer screening runs)
- Downstream: HellaSwag, PIQA, ARC-Easy with lm-eval (acc_norm). Dense = 0.6836 / 0.7492 / 0.7828
- We report "retained ability" = (acc − chance) / (acc_dense − chance), averaged over the 3 tasks.
  Chance is 25% / 50% / 25%.
- Calibration for Wanda and SparseGPT: random 512-token windows from the WikiText-2 train split (64 or 128 windows)

Tile selection methods:

| Method | How tiles are picked | Repair |
|---|---|---|
| `random` | random (5 seeds) | no |
| `magnitude` | smallest tile norm | no |
| `wanda` | smallest mean of \|W\| x activation norm | no |
| `sparsegpt` | smallest output error when the tile is masked | no |
| `sparsegpt_recon` | smallest error after reconstruction | yes (SparseGPT) |
| `wanda_recon` | Wanda selection | yes |
| `random_recon` | random selection (same tiles as `random`) | yes |

## Results folders

`results/` is numbered in the order we did things. Figures for each part are in `figures/` with the same number.

**00_baselines** – dense perplexity and the first lm-eval run from week 2.

**01_per_matrix_screening** – layers 0, 9, 18, 27, 35. Each matrix pruned alone at 10/20/40%, all 5 methods.
Most single matrices take 20% with almost no change in perplexity. Plots: `figures/01_per_matrix_screening/`,
made by `analysis/plot_per_matrix_screening.py`. `analysis/classify_reconstruction_benefit.py` splits the cells
into "redundant", "fixable by repair" and "essential".

**02_whole_layer_screening** – all 7 matrices of a layer pruned together. The damage of a full layer is smaller
than the sum of the single-matrix damages (sub-additive), but across layers it adds up.

**03_cluster_zoom_in** – layers 15-21 and 29-35 for `o_proj`, `up_proj` and `k_proj`. `o_proj` is safe at every
depth (layer 35: −0.43 perplexity at 40% Wanda). `up_proj` is fine until layers 34-35 and then explodes (+4.61).
So robustness depends on depth and matrix type together.

**04_layer_budget_wanda** – give a whole layer one budget and let the Wanda scores decide how it is split between
the 7 matrices (instead of the same % everywhere). It only helped in some layers.

**05_whole_model_sweep** – all 36 layers pruned at once, 1% to 70%, all methods. Also Policy B (sensitivity map
from the screening, same total number of tiles). Whole-model perplexity at 5%:

| Method | ppl @ 5% |
|---|---|
| magnitude | 3448.9 |
| random (seeds 1/2/3) | 22.9 / 20.2 / 48.0 |
| wanda | 28.3 |
| sparsegpt | 28.0 |
| sparsegpt_recon | 15.6 |
| wanda_recon | 15.4 |
| random_recon (seed 1) | 14.3 |

sparsegpt_recon over sparsity: 18.8 (10%), 31.0 (20%), 61.6 (30%), 137.5 (50%), 480.7 (70%).

**06_depth_concentration** – same total budget (20% of the model) but put into fewer layers.
N=32: 21.7, N=24: 22.3, N=16: 29.4, N=12: 47.0, N=8: 1185. More concentrated is always worse.
(The old 64x64 run showed a best point at N=24, which did not come back at 32x32.)

**07_oproj_targeting** – Wanda on `o_proj` only, in layers 17-21, 32, 34, 35. Perplexity went **below dense**:
12.21 at 20% and 11.45 at 40%. But HellaSwag went down (0.6824 and 0.6671 vs 0.6836). So you can make perplexity
look better while the model gets worse.

**08_downstream** – lm-eval on the pruned models. Retained ability with `sparsegpt_recon` (uniform):

| Sparsity | 1% | 2% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|---|
| retained | 100% | 96% | 90% | 79% | 43% | 23% |

This is where the "~5%" comes from. Other results:
- `wanda_recon`: 99% (1%), 91% (5%), 75% (10%), 52% (20%)
- `random_recon` at 5%: 84-88% over 5 seeds, 67% at 10%, 29% at 20%. It has the best perplexity but the worst
  ability, so perplexity even gets the order of the repair methods wrong.
- Policy B at 20%: perplexity 23.6 vs 31.0 for uniform, but retained ability only 48% vs 43%. The perplexity gain
  is much bigger than the real gain, because the map was made from perplexity.

**09_iterative_calibration** – prune in 4 steps and re-collect the calibration statistics after each step.
Retained 89.8% / 78.8% / 42.4% at 5/10/20% vs one-shot 90.0% / 78.7% / 43.3%. No improvement.

**10_tile_size_and_shape** – is the 5% because of the model or because of the 32x32 blocks?
- 1x1 (unstructured) with SparseGPT repair: perplexity 13.48 / 13.73 / 14.51 / 15.69 at 20/30/40/50%
  (32x32: 30.95 / 61.56 / 76.30 / 137.52). Retained ability 100% / 100% / 98% / 91% / 82% at 10-50%.
- Purity probe: the weights that could be removed are scattered. Almost none of them form clean blocks of 2x2 or
  bigger.
- Strip probe (1xN, Nx1): the best strips (2x1, 1x2) only cover 0.3-0.4% of the weights as clean strips. Strips do
  not really help.

So the 5% is the cost of block structure, not the real redundancy of the model. But 1x1 sparsity gives no speed-up
on normal hardware, so for usable block pruning 5% is still the limit.

**legacy_tile64** – runs from 15-17 July that used 64x64 by mistake (depth, o_proj, cluster scans). Kept for
reference only, not used for the results above. `analysis/legacy_tile64/` has the two scripts that plot them.

## Scripts

- Shared runners in `scripts/` (repo root): `run_tile_pruning.py`, `run_downstream_eval.py`,
  `run_unstructured_pruning.py`, `run_dense_baseline.py`
- Only for this experiment (`scripts/` here): `run_layer_budget_wanda.py`, `run_iterative_calibration.py`,
  `probe_tile_purity.py`, `probe_strip_purity.py`, `tile_size_sweep_queue.sh` (queue for the tile-size runs;
  we only ended up running the probe and the 1x1 part)
- Plots (`analysis/`): one script per figure group. `make_findings_figures.py` makes F1-F8 in `figures/findings/`,
  `make_figure_f10_structured_tax.py` makes F10. None of the plot scripts need a GPU.

Example commands:

```bash
uv run python scripts/run_tile_pruning.py --method sparsegpt_recon --prune-ratio 0.05 --whole-model --experiment-dir experiments/new_runs/wholemodel
uv run python scripts/run_tile_pruning.py --method sparsegpt_recon --prune-ratio 0.2 --whole-model --policy sensitivity --experiment-dir experiments/new_runs/wholemodel
uv run python scripts/run_downstream_eval.py --method wanda_recon --prune-ratio 0.05
uv run python experiments/experiment_2_qwen3_4b/scripts/run_iterative_calibration.py --prune-ratio 0.2 --steps 4
python experiments/experiment_2_qwen3_4b/analysis/make_findings_figures.py
```

## Limitations

- One calibration seed, and it is not saved in the JSON files.
- Some whole-model runs used 64 calibration windows and some 128.
- Only multiple-choice tasks, no generation tests.
- One run per setting except the random methods.
