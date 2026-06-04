from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str | None) -> torch.dtype | str:
    if not name or name == "auto":
        return "auto"
    try:
        return DTYPE_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def load_model_and_tokenizer(
    pretrained: str,
    *,
    dtype: str | None = "auto",
    device_map: str | None = "auto",
):
    torch_dtype = resolve_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model, tokenizer
