"""Shared utility functions."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch


def load_json_config(path: str) -> dict:
    """Load a JSON config file."""
    with open(path) as f:
        return json.load(f)


def build_config(args):
    """Merge JSON config with CLI overrides."""
    cfg = {}
    if args.config:
        cfg = load_json_config(args.config)

    return {
        "model_name":    "gpt2",
        "dataset":       "wikitext-2-raw-v1",
        "max_length":    args.max_length   or cfg.get("max_length", 1024),
        "stride":        args.stride       or cfg.get("stride", 512),
        "batch_size":    args.batch_size   or cfg.get("batch_size", 4),
        "seed":          args.seed if args.seed is not None else cfg.get("seed", 42),
        "device":        args.device       or cfg.get("device"),
        "lm_eval_tasks": cfg.get("lm_eval_tasks", ["hellaswag"]),
        "skip_lm_eval":  args.skip_lm_eval or cfg.get("skip_lm_eval", False),
        "output_dir":    args.output_dir,
    }


def build_result_dict(cfg, ppl, device):
    """Build a structured results dict with system info."""
    return {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "model_name": cfg["model_name"],
        "dataset":    cfg["dataset"],
        "max_length": cfg["max_length"],
        "stride":     cfg["stride"],
        "seed":       cfg["seed"],
        "device":     str(device),
        "system": {
            "platform": platform.platform(),
            "python":   platform.python_version(),
            "torch":    torch.__version__,
        },
        "perplexity": round(ppl, 4),
        "lm_eval":    None,
    }


def save_results(results: dict, output_dir: str, model_name: str):
    """Save results as a timestamped JSON file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"baseline_{model_name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {path}")
    return path


def print_summary(cfg, ppl, lm_eval_results=None):
    """Print a final summary block."""
    print("\n" + "=" * 50)
    print(f"  Model:       {cfg['model_name']}")
    print(f"  Perplexity:  {ppl:.2f}")
    if lm_eval_results:
        for task, metrics in lm_eval_results.items():
            acc = metrics.get("acc_norm,none", metrics.get("acc,none", "N/A"))
            print(f"  {task}: {acc}")
    print("=" * 50)
