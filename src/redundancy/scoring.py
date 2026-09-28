import torch


def tile_positions(rows, cols, tile_size):
    pos = []
    for r in range(0, rows, tile_size):
        for c in range(0, cols, tile_size):
            if r + tile_size <= rows and c + tile_size <= cols:
                pos.append((r, c))
    return pos


def wanda_scores(weight, col_norms, tile_size):
    W = weight.detach().float()
    cn = col_norms.detach().float().to(W.device)
    rows, cols = W.shape

    scored = []
    for r, c in tile_positions(rows, cols, tile_size):
        tile = W[r:r + tile_size, c:c + tile_size].abs()
        wt = tile * cn[c:c + tile_size].unsqueeze(0)
        scored.append((wt.mean().item(), r, c))

    return scored


def sgpt_mask_err(weight, hessian, tile_size, eps=1e-8):
    W = weight.detach().float()
    H = hessian.detach().float().to(W.device)
    rows, cols = W.shape

    denom = (W @ H * W).sum().item() + eps

    scored = []
    for r, c in tile_positions(rows, cols, tile_size):
        Wt = W[r:r + tile_size, c:c + tile_size]
        Hcc = H[c:c + tile_size, c:c + tile_size]
        num = (Wt @ Hcc * Wt).sum().item()
        scored.append((num / denom, r, c))

    return scored


def sgpt_recon_err(weight, hessian, tile_size, eps=1e-8, damp=1e-2):
    W = weight.detach().float()
    H = hessian.detach().float().to(W.device)
    rows, cols = W.shape

    diag_mean = torch.diag(H).mean()
    Hdamp = H.clone()
    idx = torch.arange(cols, device=W.device)
    Hdamp[idx, idx] += damp * diag_mean
    Hinv = torch.linalg.inv(Hdamp)

    denom = (W @ H * W).sum().item() + eps

    schur = {}
    for c in range(0, cols, tile_size):
        if c + tile_size <= cols:
            schur[c] = torch.linalg.inv(Hinv[c:c + tile_size, c:c + tile_size])

    scored = []
    for r, c in tile_positions(rows, cols, tile_size):
        Wt = W[r:r + tile_size, c:c + tile_size]
        num = (Wt @ schur[c] * Wt).sum().item()
        scored.append((max(num, 0.0) / denom, r, c))

    return scored
