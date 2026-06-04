#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redundancy.eval import run_baseline_eval


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline lm-eval for a model config.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config, e.g. configs/models/gpt2.yaml",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional sample limit override for quick smoke tests.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_path = run_baseline_eval(cfg, config_path=args.config, limit=args.limit)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
