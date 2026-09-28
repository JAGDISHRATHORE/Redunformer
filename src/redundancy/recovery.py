import torch


def damped_inv(H, damp):
    cols = H.shape[0]
    diag_mean = torch.diag(H).mean()
    Hd = H.clone()
    idx = torch.arange(cols, device=H.device)
    Hd[idx, idx] += damp * diag_mean
    return torch.linalg.inv(Hd)


def prune_and_repair(weight, hessian, tile_size, prune_ratio, damp=1e-2, eps=1e-8):
    Wf = weight.detach().float()
    H = hessian.detach().float().to(Wf.device)
    rows, cols = Wf.shape

    Hinv = damped_inv(H, damp)
    denom = (Wf @ H * Wf).sum().item() + eps

    schur = {}
    for c in range(0, cols, tile_size):
        if c + tile_size <= cols:
            schur[c] = torch.linalg.inv(Hinv[c:c + tile_size, c:c + tile_size])

    scored = []
    for r in range(0, rows, tile_size):
        for c in range(0, cols, tile_size):
            if r + tile_size <= rows and c + tile_size <= cols:
                Wt = Wf[r:r + tile_size, c:c + tile_size]
                num = (Wt @ schur[c] * Wt).sum().item()
                scored.append((max(num, 0.0) / denom, r, c))

    scored.sort(key=lambda x: x[0])
    n_prune = int(len(scored) * prune_ratio)
    pruned = scored[:n_prune]

    cols_per_row = {}
    for _, r, c in pruned:
        cols_per_row.setdefault(r, []).extend(range(c, c + tile_size))

    with torch.no_grad():
        for r, pcols in cols_per_row.items():
            idx = torch.tensor(sorted(pcols), device=Wf.device)
            S = torch.linalg.inv(Hinv[idx][:, idx])
            A = Hinv[:, idx] @ S
            block = weight[r:r + tile_size, idx].float()
            weight[r:r + tile_size, :] -= (block @ A.t()).to(weight.dtype)

    return len(scored), n_prune
