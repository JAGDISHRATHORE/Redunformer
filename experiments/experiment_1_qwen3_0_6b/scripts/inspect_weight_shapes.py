import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from redundancy.models import load_model

model, tokenizer = load_model("Qwen/Qwen3-0.6B")

for name, param in model.named_parameters():
    if len(param.shape) == 2:
        rows, cols = param.shape
        print(f"{rows*cols:>10} | {name:60} | {param.shape}")
