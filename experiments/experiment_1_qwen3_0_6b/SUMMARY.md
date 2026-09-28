# Experiment 1 – Qwen3-0.6B (first tile pruning tests)

June to early July 2026.

## Setup

- Model: `Qwen/Qwen3-0.6B` (28 layers)
- Data: WikiText-2 (`wikitext-2-raw-v1`), test split, perplexity on the full test set
- Dense perplexity: **20.0445**
- Tiles: **64x64** (later we found this was a wrong default, the main experiments use 32x32)
- Methods: lowest magnitude (Frobenius norm of the tile), highest magnitude, random
- No calibration data and no downstream tasks yet

We started with the small model because it runs fast and we wanted to see if tile pruning does anything at all.

## What we ran

**1. Layer 0 up_proj sweep** (`results/01_layer0_up_proj_sweep/`)
Pruned 5%, 10%, 20% of the tiles of `model.layers.0.mlp.up_proj`.

| | 5% | 10% | 20% |
|---|---|---|---|
| lowest magnitude | 20.27 | 20.48 | 20.77 |
| highest magnitude | 20.08 | 20.24 | 20.55 |
| random (seed 42) | 20.07 | 20.16 | 20.31 |

Random seeds 1, 2, 3 at 20%: 20.66, 20.60, 20.42. Already here random was not worse than magnitude.

**2. Layer study, layers 0 / 12 / 27** (`results/02_layer_study_L0_L12_L27/`)
Each of the 7 matrices pruned on its own at 20% (magnitude), plus random with seeds 42, 123, 456, 789.
Layer 0 `up_proj` magnitude is the file `up_proj_magnitude_p20.json` in the sweep folder.

Perplexity with magnitude pruning (dense = 20.04):

| Matrix | Layer 0 | Layer 12 | Layer 27 |
|---|---|---|---|
| gate_proj | 20.47 | 20.52 | 28.18 |
| up_proj | 20.77 | 20.51 | 26.02 |
| down_proj | 21.12 | 20.17 | 28.35 |
| q_proj | 26.53 | 21.06 | 27.67 |
| k_proj | 24.70 | 23.13 | 21.04 |
| v_proj | 31.03 | 19.92 | 19.45 |
| o_proj | 20.68 | 20.04 | 20.14 |

What we saw: the damage depends on the layer and on the matrix. Layer 12 was almost free to prune, early attention
(q, k, v in layer 0) and late MLP (layer 27) were sensitive. Random pruning (mean of seeds 123, 456, 789) was better than
magnitude in 16 of the 21 cells.

**3. Full scan of all 28 layers** (`results/03_full_scan_all_layers/`)
All 28 layers x 7 matrices, magnitude vs random (seed 42), 20% each, one matrix at a time.
Random gave lower perplexity than magnitude in **144 of 196** cells. The worst magnitude cells were layer 5
`q_proj` (40.9) and layer 13 `k_proj` (40.5). Plots are in `results/03_full_scan_all_layers/plots/`.

## What we took from it

- Tile redundancy exists, but it is not the same everywhere.
- Magnitude is a bad way to pick tiles. This became Finding 8 in Experiment 2.
- Perplexity alone was not enough and the model was small, so we moved to Qwen3-4B and added Wanda, SparseGPT and
  downstream tasks.

## Notes

- These files were deleted from the repo during a cleanup on 18 July and restored from git history
  (commits `68896e4` and `e3875b2`). The numbers are exactly the ones in our report (Tables 1-3).
- The full-scan plots were redrawn with the correct baseline line (20.0445). Before, they showed the Qwen3-4B
  baseline by mistake.
- The full scan was a separate run from the layer study, so the numbers can differ a little in the 3rd decimal.

## How to rerun

The shared runner still supports this setup:

```bash
uv run python scripts/run_dense_baseline.py --model Qwen/Qwen3-0.6B
uv run python scripts/run_tile_pruning.py --model Qwen/Qwen3-0.6B --tile-size 64 --prune-ratio 0.2 --method magnitude --all-layers --all-matrices --baseline-ppl 20.0445 --experiment-dir experiments/new_runs/qwen06_full_scan
uv run python scripts/run_tile_pruning.py --model Qwen/Qwen3-0.6B --tile-size 64 --prune-ratio 0.2 --method random --seeds 42 --all-layers --all-matrices --baseline-ppl 20.0445 --experiment-dir experiments/new_runs/qwen06_full_scan
```

`scripts/inspect_weight_shapes.py` prints the shape of every weight matrix of the model (we used it to see how
many tiles each matrix has).
