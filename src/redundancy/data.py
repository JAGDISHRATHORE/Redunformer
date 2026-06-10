"""Dataset loading and tokenisation for perplexity evaluation."""

import torch
from datasets import load_dataset


def load_wikitext(split: str = "test"):
    """Load WikiText-2 (raw) from HuggingFace."""
    print(f"Loading WikiText-2 ({split}) ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    print(f"  {len(ds)} examples loaded.")
    return ds

def prepare_encodings(dataset, tokenizer, max_length: int = 1024, stride: int = 512):
    """Concatenate all text and build sliding-window chunks for perplexity."""
    full_text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids  # (1, total_tokens)
    total_tokens = input_ids.size(1)
    print(f"  {total_tokens} tokens | stride={stride} | ctx={max_length}")

    # Build overlapping windows
    chunks = []
    for start in range(0, total_tokens - max_length + 1, stride):
        chunks.append(input_ids[:, start : start + max_length])

    if not chunks:
        chunks.append(input_ids)

    stacked = torch.cat(chunks, dim=0)  # (n_windows, max_length)
    print(f"  {stacked.size(0)} windows.")
    return stacked
