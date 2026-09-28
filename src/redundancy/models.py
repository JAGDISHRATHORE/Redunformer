import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(name: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading tokenizer {name}...")
    tokenizer = AutoTokenizer.from_pretrained(name)

    print(f"Loading model {name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True
    )

    return model, tokenizer
