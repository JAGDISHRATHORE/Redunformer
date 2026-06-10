# Codebase Documentation

## Overview

This project evaluates **GPT-2** on **WikiText-2** to establish a baseline for block/layer-level redundancy experiments. The pipeline loads the model, tokenises the dataset with a sliding window, computes perplexity, and optionally runs [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) benchmarks.

---

## Pipeline Flow

```
run_baseline.py
  │
  ├─ utils.build_config()           → merge JSON config + CLI args
  ├─ models.load_model()            → GPT-2 model + tokenizer on best device
  ├─ data.load_wikitext()           → WikiText-2 test split
  ├─ data.prepare_encodings()       → sliding-window token chunks
  ├─ eval.compute_perplexity()      → perplexity score
  ├─ eval.run_lm_eval()             → (optional) HellaSwag accuracy
  ├─ utils.build_result_dict()      → structured results dict
  ├─ utils.save_results()           → experiments/baseline_gpt2_<timestamp>.json
  └─ utils.print_summary()          → final summary to stdout
```

---

## Module Reference

### `src/redundancy/models.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_device` | `(device: str \| None) → torch.device` | Picks the best available device. Priority: user-specified → CUDA → MPS → CPU. |
| `load_model` | `(model_name, device, dtype) → (model, tokenizer)` | Downloads the model and tokenizer from HuggingFace, moves to device, sets eval mode. Defaults to `gpt2` with `float16` on GPU/MPS and `float32` on CPU. |

**Where is GPT-2 stored?** The model weights are **not** kept in this repository. When `load_model("gpt2")` runs for the first time, HuggingFace Transformers downloads the weights (~525 MB) from the Hub and caches them at `~/.cache/huggingface/hub/models--gpt2/`. Every subsequent run loads directly from that cache — no re-download needed. This keeps the git repo lightweight.

---

### `src/redundancy/data.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_wikitext` | `(split) → Dataset` | Loads `Salesforce/wikitext` (`wikitext-2-raw-v1`) from HuggingFace Hub. |
| `prepare_encodings` | `(dataset, tokenizer, max_length, stride) → Tensor` | Joins all text, tokenises it into one long sequence, then slices it into overlapping windows of `max_length` tokens with a step of `stride`. Returns a `(n_windows, max_length)` tensor. |

**Why sliding windows?** GPT-2 has a fixed context of 1024 tokens. To evaluate on a longer text, we slide a window across the full token sequence. Overlapping regions give the model context, but we only score the *non-overlapping* tokens to avoid counting any token twice.

---

### `src/redundancy/eval.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_perplexity` | `(model, input_ids, device, stride) → float` | Iterates over all sliding windows, runs a forward pass on each, computes cross-entropy loss on the non-overlapping portion, and returns `exp(avg_loss)`. Prints progress every 50 windows. |
| `run_lm_eval` | `(model_name, tasks, device, batch_size) → dict` | Calls `lm_eval.simple_evaluate` with the `"hf"` backend. Returns the full results dict. Used for standardised benchmarks like HellaSwag. |

**Perplexity scoring detail:** For window `i > 0`, only the last `stride` tokens are scored (the earlier tokens are overlap from the previous window). For window `i = 0`, all tokens are scored. This follows the [HuggingFace perplexity guide](https://huggingface.co/docs/transformers/perplexity).

---

### `src/utils/__init__.py`

| Function | Description |
|----------|-------------|
| `load_json_config` | Loads a JSON config file and returns a dict. |
| `build_config` | Merges a JSON config file with CLI argument overrides. Hardcodes `model_name="gpt2"` and `dataset="wikitext-2-raw-v1"`. |
| `build_result_dict` | Builds a structured results dict with perplexity, config, device, and system info (platform, Python version, PyTorch version). |
| `save_results` | Writes the results dict to `experiments/baseline_gpt2_<timestamp>.json`. |
| `print_summary` | Prints a final summary block with model name, perplexity, and lm-eval scores. |

---

### `scripts/run_baseline.py`

| Function | Description |
|----------|-------------|
| `parse_args` | Defines CLI arguments: `--config`, `--max-length`, `--stride`, `--seed`, `--device`, `--skip-lm-eval`, `--output-dir`, `--batch-size`. |
| `main` | Orchestrates the full pipeline: parse args → load model → load data → compute perplexity → (optional) lm-eval → save → print summary. |


---

## Configuration

`configs/baseline.json` contains default parameters:

| Key | Default | Description |
|-----|---------|-------------|
| `model_name` | `"gpt2"` | HuggingFace model identifier |
| `dataset` | `"wikitext-2-raw-v1"` | WikiText-2 raw subset |
| `max_length` | `1024` | Context window size (tokens) |
| `stride` | `512` | Sliding window step size |
| `lm_eval_tasks` | `["hellaswag"]` | Benchmark tasks for lm-eval-harness |
| `batch_size` | `4` | Batch size for lm-eval |
| `seed` | `42` | Random seed |

---

## Output Format

Each run produces a JSON file in `experiments/`:

```json
{
  "timestamp": "2026-06-10T09:01:17+00:00",
  "model_name": "gpt2",
  "dataset": "wikitext-2-raw-v1",
  "max_length": 1024,
  "stride": 512,
  "seed": 42,
  "device": "mps",
  "system": {
    "platform": "macOS-...",
    "python": "3.13.5",
    "torch": "2.12.0"
  },
  "perplexity": 25.18,
  "lm_eval": null
}
```

When `--skip-lm-eval` is not set, `lm_eval` contains per-task metrics (e.g. `acc_norm` for HellaSwag).
