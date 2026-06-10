"""Model loading utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device(device: str | None = None) -> torch.device:
    """Auto-select device: user-specified > CUDA > MPS > CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_name: str = "gpt2",
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace causal LM and its tokenizer onto the best device."""
    dev = get_device(device)
    if dtype is None:
        dtype = torch.float32 if dev.type == "cpu" else torch.float16

    print(f"Loading '{model_name}' on {dev} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(dev).eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = getattr(model.config, "n_layer", "?")
    print(f"  {n_params / 1e6:.1f}M params | {n_layers} layers | vocab {model.config.vocab_size}")

    return model, tokenizer
