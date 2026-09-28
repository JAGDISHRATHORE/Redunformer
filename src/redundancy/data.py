import random

from datasets import load_dataset


def load_wikitext(ds_name: str = "Salesforce/wikitext", subset: str = "wikitext-2-raw-v1", split: str = "test"):
    print(f"Loading dataset {ds_name} ({subset}), split: {split}...")
    if ds_name == "wikitext":
        ds_name = "Salesforce/wikitext"

    dataset = load_dataset(ds_name, subset, split=split)
    return dataset


def get_calib_data(tokenizer, n_calib: int = 128, seqlen: int = 512, seed: int = 0,
                   ds_name: str = "Salesforce/wikitext", subset: str = "wikitext-2-raw-v1"):
    print(f"Loading calibration data: {n_calib} x {seqlen} tokens from {ds_name} ({subset}) train...")
    if ds_name == "wikitext":
        ds_name = "Salesforce/wikitext"

    data = load_dataset(ds_name, subset, split="train")
    text = "\n\n".join(t for t in data["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids
    total = ids.size(1)

    if total <= seqlen:
        raise ValueError(f"Calibration corpus ({total} tokens) shorter than seqlen ({seqlen}).")

    rng = random.Random(seed)
    windows = []
    for _ in range(n_calib):
        start = rng.randint(0, total - seqlen - 1)
        windows.append(ids[:, start:start + seqlen])

    return windows
