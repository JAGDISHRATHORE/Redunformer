# Redunformer – Tile-level redundancy in LLMs

Practical Course in Artificial Intelligence, Summer Semester 2026, TU Darmstadt
Group 4: Sebastian Fiebiger, Jagdish Dattarsingh Rathore, Ayush Pratap Singh, Venkata Anusha Vangavolu
Supervisor: Haoyi Yang

## What we did

Our question was simple: how much of a language model can you delete in blocks (tiles) before it stops working?
A tile is a square block of a weight matrix (32x32 in the main experiments). Pruning a tile means setting the whole
block to zero. We only pruned the seven linear layers inside each transformer block (`q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, `down_proj`).

The project ran from June to August 2026 in three stages:

| | Model | Tile size | What it was for |
|---|---|---|---|
| Experiment 1 | Qwen3-0.6B | 64x64 | first try: magnitude vs random pruning, which layers and matrices are sensitive |
| Experiment 2 | Qwen3-4B | 32x32 | main study: Wanda, SparseGPT, repair, downstream tasks, budgets, tile size |
| Experiment 3 | Llama-3.2-3B-Instruct | 32x32 | check if the Qwen3-4B results also hold on a second model |

Every experiment folder has its own `SUMMARY.md` with the details.

## Main results

- Only about **5% of the 32x32 tiles** of Qwen3-4B can be removed while keeping ~90% of the ability on HellaSwag, PIQA
  and ARC-Easy. Perplexity alone made it look like 40-70% was possible.
- **Repair matters more than selection.** After removing tiles, SparseGPT reconstruction of the remaining weights
  brings whole-model perplexity at 5% from 20-48 (no repair) down to 14-16. How we choose the tiles matters much less.
- **Perplexity can be misleading.** Pruning `o_proj` on a few chosen layers made WikiText perplexity better than the
  dense model (12.21 vs 13.22), while HellaSwag got slightly worse.
- **Magnitude pruning is worse than random** at tile level (Qwen3-4B, 5%: 3449 vs ~23 perplexity).
- **The 5% is a price for block structure.** The same repair with single weights (1x1) keeps ~100% ability up to 20%
  sparsity and ~82% at 50%. The redundancy is there, but it is spread out and not in blocks.
- **Llama-3.2-3B shows the same ~5% limit** (89% ability kept at 5%), but it drops faster after that, and magnitude
  pruning is not a disaster there.

We first wrote the code for 64x64 tiles by mistake (a wrong default in `run_pruning.py`). We fixed it on 17 July and
re-ran the affected Qwen3-4B runs at 32x32. The old 64x64 runs are kept in
`experiments/experiment_2_qwen3_4b/results/legacy_tile64/`.

## Folder structure

```
src/redundancy/        shared code (model loading, data, perplexity, tile scores, repair, policies)
scripts/               shared runners, used for Qwen3-4B and Llama (and Qwen3-0.6B with --model)
experiments/
  experiment_1_qwen3_0_6b/     results, plots, SUMMARY.md
  experiment_2_qwen3_4b/       results, figures, analysis scripts, extra runners, SUMMARY.md
  experiment_3_llama3_2_3b/    results, figures, analysis script, SUMMARY.md
pyproject.toml, uv.lock
```

Inside each experiment: `results/` has the JSON files, `figures/` the plots, `analysis/` the plotting scripts and
`scripts/` the runners that only this experiment used. Some runners also save quick plots next to their JSON
files in a `plots/` folder.

## Setup

We used one RTX 4080 Super (16 GB) for everything.

```bash
uv sync
```

Models and datasets are downloaded from Hugging Face on the first run. Llama-3.2 needs a Hugging Face token with
access to `meta-llama/Llama-3.2-3B-Instruct`. All commands are run from the repository root.

## How to run

```bash
# dense perplexity
uv run python scripts/run_dense_baseline.py --model Qwen/Qwen3-4B

# prune one matrix type at a time on some layers (screening)
uv run python scripts/run_tile_pruning.py --method wanda --prune-ratio 0.2 --layers 0 9 18 27 35 --all-matrices --eval-frac 0.2 --experiment-dir experiments/new_runs/screen

# prune the whole model and measure perplexity
uv run python scripts/run_tile_pruning.py --method sparsegpt_recon --prune-ratio 0.05 --whole-model --experiment-dir experiments/new_runs/wholemodel

# downstream accuracy (HellaSwag, PIQA, ARC-Easy) with lm-eval
uv run python scripts/run_downstream_eval.py --dense
uv run python scripts/run_downstream_eval.py --method sparsegpt_recon --prune-ratio 0.05

# 1x1 (unstructured) pruning for comparison
uv run python scripts/run_unstructured_pruning.py --method unstructured_sparsegpt --prune-ratio 0.5 --eval-ppl

# redraw all Qwen3-4B figures from the saved JSON (no GPU needed)
python experiments/experiment_2_qwen3_4b/analysis/make_findings_figures.py
```

Methods for `--method`: `magnitude`, `magnitude_high`, `random`, `wanda`, `sparsegpt`, `sparsegpt_recon`,
`wanda_recon`, `random_recon`. The `*_recon` methods remove the tiles and then repair the rest of the row with
SparseGPT. New runs go to `experiments/new_runs/`, which is not tracked by git.

## How the result files are named

The name tells you what was run:

| Pattern | Meaning |
|---|---|
| `layer{L}_{method}_p{P}.json` | one matrix type at a time on layer L, P% of its tiles pruned (one entry per matrix inside) |
| `layer{L}_{method}_p{P}_seed{S}.json` | same, random methods with seed S |
| `layer{L}_wholelayer_{method}_p{P}.json` | all 7 matrices of layer L pruned together |
| `wholemodel_{method}_p{P}.json` | all layers pruned, uniform P% everywhere |
| `wholemodel_{method}_p{P}_sensitivity.json` | same budget, but spread by the sensitivity map (Policy B) |
| `downstream_{method}_p{P}_{policy}.json` | lm-eval accuracy of that pruned model |
| `us_{method}_p{P}_T1.json` | 1x1 unstructured pruning |
| `{method}_p0.10/` folders | the sparsity level of the files inside (10%, 20%, 40%) |

The tile size is not in the file name, but every JSON has a `tile_size` field.

## Results in the report

The results in our report were produced with the code at commit `f500f91` (branch `Jagdish-Tile`). After that we
only moved files, renamed variables and removed comments. We checked that the code does exactly the same thing and
that the plots come out the same.
