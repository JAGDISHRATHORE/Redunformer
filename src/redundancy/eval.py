"""Evaluation: perplexity and lm-evaluation-harness wrapper."""

import torch
from torch.nn import CrossEntropyLoss


@torch.no_grad()
def compute_perplexity(model, input_ids: torch.Tensor, device, stride: int = 512) -> float:
    """Sliding-window perplexity. Only scores non-overlapping tokens to avoid double-counting."""
    max_length = input_ids.size(1)
    n_windows = input_ids.size(0)
    loss_fn = CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_tokens = 0

    for i in range(n_windows):
        ids = input_ids[i : i + 1].to(device)
        logits = model(ids, labels=ids).logits

        # Shift: predict token t+1 from token t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()

        # Only score the non-overlapping portion (avoid double-counting)
        start = 0 if i == 0 else max_length - stride - 1
        losses = loss_fn(
            shift_logits[:, start:, :].reshape(-1, shift_logits.size(-1)),
            shift_labels[:, start:].reshape(-1),
        )

        total_loss += losses.sum().item()
        total_tokens += losses.numel()

        if (i + 1) % 50 == 0 or (i + 1) == n_windows:
            ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
            print(f"  [{i + 1}/{n_windows}] running ppl = {ppl:.2f}")

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def run_lm_eval(model_name: str, tasks: list[str], device: str | None = None, batch_size: int | str = "auto"):
    """Thin wrapper around lm-evaluation-harness."""
    import lm_eval

    print(f"Running lm-eval: model='{model_name}', tasks={tasks} ...")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name}",
        tasks=tasks,
        device=device,
        batch_size=batch_size,
    )

    for task, metrics in results.get("results", {}).items():
        for k, v in metrics.items():
            if not k.startswith("alias") and isinstance(v, float):
                print(f"  {task} / {k} = {v:.4f}")

    return results
