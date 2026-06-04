from __future__ import annotations

from datasets import load_dataset


def load_hf_dataset(name: str, config: str | None = None, split: str = "train"):
    if config:
        return load_dataset(name, config, split=split)
    return load_dataset(name, split=split)
