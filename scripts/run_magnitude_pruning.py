#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redundancy.pruning.magnitude import apply_magnitude_pruning
from redundancy.eval import run_baseline_eval


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate magnitude pruning.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pruning_ratio", type=float, required=True)
    parser.add_argument("--limit", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["pretrained"],
        torch_dtype=cfg.get("dtype", "auto"),
        device_map=cfg.get("device_map", "auto")
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained"])
    
    apply_magnitude_pruning(model, args.pruning_ratio)
    
    tmp_model_dir = ROOT / "experiments" / "tmp_pruned_model"
    model.save_pretrained(tmp_model_dir)
    tokenizer.save_pretrained(tmp_model_dir)
    
    del model
    del tokenizer
    torch.cuda.empty_cache()

    cfg["pretrained"] = str(tmp_model_dir)
    cfg["output_dir"] = f"experiments/pruning_magnitude_{args.pruning_ratio}"
    
    output_path = run_baseline_eval(cfg, config_path=args.config, limit=args.limit)
    print(f"Results successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
