import torch


def flatten(x):
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    return x


class GramHook:
    def __init__(self, module, name):
        self.name = name
        n_in = module.weight.shape[1]
        device = module.weight.device
        self.H = torch.zeros(n_in, n_in, dtype=torch.float32, device=device)
        self.n_tokens = 0
        self._handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        x = flatten(args[0].detach()).to(torch.float32)
        self.H += x.t() @ x
        self.n_tokens += x.shape[0]

    def col_norms(self):
        return torch.sqrt(torch.clamp(torch.diag(self.H), min=0.0))

    def remove(self):
        self._handle.remove()


@torch.no_grad()
def collect_grams(model, mods, calib, device=None):
    if device is None:
        device = model.device

    collectors = {name: GramHook(mod, name) for name, mod in mods.items()}
    try:
        for input_ids in calib:
            model(input_ids.to(device))
    finally:
        for c in collectors.values():
            c.remove()

    return collectors
