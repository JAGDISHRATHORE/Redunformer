#!/usr/bin/env python3

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redundancy.models import load_model_and_tokenizer
from redundancy.pruning.random_pruning import random_prune_model, count_parameters

MODEL_NAME = sys.argv[1]

print("Loading model...")

model, tokenizer = load_model_and_tokenizer(
    MODEL_NAME,
    dtype="bfloat16",
)

print(f"Parameters: {count_parameters(model):,}")

random_prune_model(
    model,
    sparsity=0.20,
)

model_name_safe = MODEL_NAME.split("/")[-1].lower()
save_path = f"experiments/pruned/{model_name_safe}-random20"

print(f"Saving to {save_path}")

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Done.")
