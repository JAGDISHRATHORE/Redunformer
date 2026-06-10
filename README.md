# Redunformer

**Project Seminar: Redundancy in Large Language Models — SS 2026**
**Group 1: Block / Layer-level Redundancy**

## Setup

```bash
# Requires Python >= 3.12 and uv
uv sync
```

## Running the Baseline

```bash
# Perplexity only (fast)
uv run python scripts/run_baseline.py --skip-lm-eval

# Full baseline (perplexity + HellaSwag via lm-eval-harness)
uv run python scripts/run_baseline.py --config configs/baseline.json
```

Results are saved as JSON in `experiments/`.

## Structure

```
├── configs/baseline.json        # Default evaluation config
├── experiments/                  # Result JSONs
├── scripts/run_baseline.py      # Main entry point
└── src/
    ├── redundancy/
    │   ├── models.py            # Load GPT-2 + auto device selection
    │   ├── data.py              # Load WikiText-2 + sliding-window tokenisation
    │   └── eval.py              # Perplexity computation + lm-eval wrapper
    └── utils/                   # Helper functions (config, saving, printing)
```

## Model & Dataset

- **Model**: GPT-2 (124M params, 12 layers)
- **Dataset**: WikiText-2 (raw, test split)
- **Metrics**: Perplexity (sliding-window) + HellaSwag (lm-eval-harness)
