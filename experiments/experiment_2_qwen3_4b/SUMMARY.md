# Experiment 2 – Qwen3-4B (main study)

## What we did based to the meeting notes

So basically, a language model like Qwen3-4B is a huge collection of numbers, and these numbers are stored in big tables. We wanted to know how much of these numbers we can delete before the model stops working properly. We didn't delete single numbers. We cut the tables into small 32x32 blocks, called tiles, and deleted whole tiles, because only this kind of block deleting can actually make a model faster on normal hardware. We tried different ways to pick which tiles to delete: simple ones like random or smallest tiles first, and smarter ones like Wanda and SparseGPT that look at how the model really uses the weights. We also tried a "repair" step, where after deleting we adjust the remaining weights a bit so the model can make up for what we removed. To check the damage we looked at two things: perplexity and real multiple-choice tasks, which will tell us how much of the model's actual ability is left apart from only perplexity.

## Setup

- Model: `Qwen/Qwen3-4B` (36 layers, bf16), 7 linear matrices in every layer
- Tiles: **32x32**, so 98,560 tiles per layer and 3,548,160 in total
- Perplexity: WikiText-2 test set. Dense model = **13.22** on the full test set and **13.559** on the first 20%. We used the 20% part for the fast screening runs.
- Real tasks: HellaSwag, PIQA and ARC-Easy with lm-eval (acc_norm). Dense model = 0.6836 / 0.7492 / 0.7828
- "Retained ability" = (acc − chance) / (acc_dense − chance), averaged over the 3 tasks. Chance is 25% / 50% / 25%. So 100% means as good as the original model and 0% means just guessing.
- Calibration for Wanda and SparseGPT: random 512-token pieces from the WikiText-2 train split (64 or 128 pieces)

These are the ways we picked the tiles:

| Method | How tiles are picked | Repair |
|---|---|---|
| `random` | randomly (5 seeds) | no |
| `magnitude` | smallest tiles first | no |
| `wanda` | smallest weight x activation score | no |
| `sparsegpt` | smallest change in the output when the tile is removed | no |
| `sparsegpt_recon` | smallest error after repair | yes (SparseGPT) |
| `wanda_recon` | picked like Wanda | yes |
| `random_recon` | picked randomly (same tiles as `random`) | yes |

## Results

The `results/` folder is numbered in the order we did things, and the plots for each part are in `figures/` with the same number.

**00_baselines** – dense perplexity and our first lm-eval run from week 2.

**01_per_matrix_screening** – layers 0, 9, 18, 27 and 35. We pruned every matrix alone at 10/20/40% with all 5 methods. Most single matrices can lose 20% of their tiles and perplexity almost doesn't change. The plots are made by `analysis/plot_per_matrix_screening.py`, and `analysis/classify_reconstruction_benefit.py` sorts the cases into "redundant", "fixable by repair" and "essential".

**02_whole_layer_screening** – here we pruned all 7 matrices of one layer together. Inside one layer the damage is smaller than the sum of the single matrices, but across many layers it adds up.

**03_cluster_zoom_in** – layers 15-21 and 29-35 for `o_proj`, `up_proj` and `k_proj`. `o_proj` was safe at every depth (layer 35: −0.43 perplexity at 40% Wanda). `up_proj` was fine until layers 34-35, then it exploded (+4.61). So how robust a part is depends on the layer and the matrix type together, not on one of them alone.

**04_layer_budget_wanda** – we gave a whole layer one budget and let the Wanda scores decide how to split it between the 7 matrices, instead of taking the same % everywhere. This helped only in some layers.

**05_whole_model_sweep** – all 36 layers pruned at the same time, from 1% to 70%, with all methods. We also tried Policy B, which uses the screening results to prune sensitive parts less and robust parts more, with the same total number of tiles. Perplexity of the whole model at 5%:

| Method | ppl @ 5% |
|---|---|
| magnitude | 3448.9 |
| random (seeds 1/2/3) | 22.9 / 20.2 / 48.0 |
| wanda | 28.3 |
| sparsegpt | 28.0 |
| sparsegpt_recon | 15.6 |
| wanda_recon | 15.4 |
| random_recon (seed 1) | 14.3 |

So repair is what really makes the difference, and picking the smallest tiles (magnitude) completely breaks the model. `sparsegpt_recon` at higher sparsity: 18.8 (10%), 31.0 (20%), 61.6 (30%), 137.5 (50%), 480.7 (70%).

**06_depth_concentration** – same total budget (20% of the model), but put into fewer layers. N=32: 21.7, N=24: 22.3, N=16: 29.4, N=12: 47.0, N=8: 1185. The more we concentrated it, the worse it got. The old 64x64 run showed a best point at N=24, but that didn't come back at 32x32.

**07_oproj_targeting** – Wanda only on `o_proj`, in layers 17-21, 32, 34 and 35. Here perplexity went **below the dense model**: 12.21 at 20% and 11.45 at 40%. But HellaSwag went down (0.6824 and 0.6671 vs 0.6836). So you can make perplexity look better while the model actually gets worse.

**08_downstream** – lm-eval on the pruned models. Retained ability with `sparsegpt_recon` (same % everywhere):

| Sparsity | 1% | 2% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|---|
| retained | 100% | 96% | 90% | 79% | 43% | 23% |

This is where our "~5%" comes from. Some other results:
- `wanda_recon`: 99% (1%), 91% (5%), 75% (10%), 52% (20%)
- `random_recon` at 5%: 84-88% over 5 seeds, 67% at 10% and 29% at 20%. It had the best perplexity but the worst real ability, so perplexity even ranks the repair methods in the wrong order.
- Policy B at 20%: perplexity 23.6 vs 31.0 for the uniform version, but retained ability only 48% vs 43%. The perplexity gain looks much bigger than the real gain, because the map for Policy B was made from perplexity itself.

**09_iterative_calibration** – we pruned in 4 small steps and collected the calibration data again after each step. Retained ability was 89.8% / 78.8% / 42.4% at 5/10/20%, and one-shot gave 90.0% / 78.7% / 43.3%. So no improvement.

**10_tile_size_and_shape** – here we wanted to know if the 5% limit comes from the model itself or from the 32x32 blocks.
- 1x1 (single weights) with SparseGPT repair: perplexity 13.48 / 13.73 / 14.51 / 15.69 at 20/30/40/50%, while 32x32 gives 30.95 / 61.56 / 76.30 / 137.52. Retained ability 100% / 100% / 98% / 91% / 82% at 10-50%.
- Purity probe: the weights that could be removed are scattered all over. Almost none of them form clean blocks of 2x2 or bigger.
- Strip probe (1xN and Nx1): even the best strips (2x1, 1x2) only cover 0.3-0.4% of the weights as clean strips, so strips don't really help.

So the 5% is the price we pay for deleting in blocks, not the real redundancy of the model. But deleting single weights doesn't make the model faster on normal hardware, so for block pruning that is actually useful, 5% is still the limit.

**legacy_tile64** – runs from 15-17 July that used 64x64 tiles by mistake (depth, o_proj and cluster scans). We kept them only for reference, and they are not used in the results above. `analysis/legacy_tile64/` has the two scripts that plot them.

## Scripts

- Shared runners in `scripts/` at the repo root: `run_tile_pruning.py`, `run_downstream_eval.py`, `run_unstructured_pruning.py`, `run_dense_baseline.py`
- Only for this experiment (`scripts/` in this folder): `run_layer_budget_wanda.py`, `run_iterative_calibration.py`, `probe_tile_purity.py`, `probe_strip_purity.py` and `tile_size_sweep_queue.sh`. That last one is the queue for the tile-size runs; in the end we only ran the probe and the 1x1 part.
- Plots (`analysis/`): one script for each group of figures. `make_findings_figures.py` makes F1-F8 in `figures/findings/`, and `make_figure_f10_structured_tax.py` makes F10. None of the plot scripts need a GPU.

Example commands:

```bash
uv run python scripts/run_tile_pruning.py --method sparsegpt_recon --prune-ratio 0.05 --whole-model --experiment-dir experiments/new_runs/wholemodel
uv run python scripts/run_tile_pruning.py --method sparsegpt_recon --prune-ratio 0.2 --whole-model --policy sensitivity --experiment-dir experiments/new_runs/wholemodel
uv run python scripts/run_downstream_eval.py --method wanda_recon --prune-ratio 0.05
uv run python experiments/experiment_2_qwen3_4b/scripts/run_iterative_calibration.py --prune-ratio 0.2 --steps 4
python experiments/experiment_2_qwen3_4b/analysis/make_findings_figures.py
```
