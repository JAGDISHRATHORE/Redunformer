import torch
import torch.nn as nn

def apply_magnitude_pruning(model: nn.Module, pruning_ratio: float) -> None:
    model.eval()
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                if hasattr(module, "weight") and module.weight is not None:
                    weights = module.weight.data
                    
                    abs_weights = torch.abs(weights)
                    threshold = torch.quantile(abs_weights, pruning_ratio)
                    mask = (abs_weights >= threshold).float()
                    
                    module.weight.data.mul_(mask)
