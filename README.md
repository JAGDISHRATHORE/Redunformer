# Redunformer

Cross-granularity redundancy analysis for large language models (SoSe 2026).

**Group:** Tile-level redundancy  
**Branch:** `Tile`

## Setup

Use `uv` to manage the environment and dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python scripts/run_baseline.py --model gpt2 --dataset Salesforce/wikitext --subset wikitext-2-raw-v1
```

## Baseline Results (Week 1–2)

### Models

| Model | Parameters | dtype |
|-------|-----------|-------|
| `gpt2` | 117M | float32 |
| `Qwen/Qwen3-4B` | 4B | bfloat16 |

### Datasets

- **WikiText-2** (`Salesforce/wikitext`, `wikitext-2-raw-v1`, test split) — for perplexity
- **HellaSwag**, **PIQA**, **ARC Easy** — via lm-evaluation-harness

### Evaluation commands

```bash
# Custom perplexity script
uv run python scripts/run_baseline.py --model gpt2 --dataset Salesforce/wikitext --subset wikitext-2-raw-v1 --output experiments/baseline_gpt2.json

# lm-evaluation-harness (requires lm_eval CLI)
lm_eval --model hf --model_args pretrained=Qwen/Qwen3-4B,dtype=bfloat16,device_map=auto --tasks hellaswag,piqa,arc_easy --batch_size auto --seed 42
```

### Results

**Perplexity (WikiText-2, custom script):**

| Model | Perplexity |
|-------|-----------|
| GPT-2 (117M) | 24.38 |

**Accuracy (lm-evaluation-harness):**

| Model | HellaSwag (norm) | PIQA (norm) | ARC Easy (norm) |
|-------|-----------------|-------------|-----------------|
| GPT-2 (117M) | 39.6% | — | — |
| Qwen3-4B | 68.4% | 74.9% | 78.3% |

Lower perplexity = better. Higher accuracy = better. Full results saved in `experiments/`.

## Project layout

```text
src/redundancy/     shared library (models, eval, data, hooks, plotting)
scripts/            CLI entrypoints
experiments/        JSON results
```
