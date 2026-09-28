import random as _random

import torch

from .recovery import damped_inv
from .scoring import wanda_scores, tile_positions


def norms_from_H(hessian):
    return torch.sqrt(torch.diag(hessian.detach().float()).clamp_min(0))


def repair_tiles(weight, hessian, tiles_out, tile_size, damp=1e-2):
    H = hessian.detach().float().to(weight.device)
    Hinv = damped_inv(H, damp)

    cols_per_row = {}
    for r, c in tiles_out:
        cols_per_row.setdefault(r, []).extend(range(c, c + tile_size))

    with torch.no_grad():
        for r, pcols in cols_per_row.items():
            idx = torch.tensor(sorted(set(pcols)), device=weight.device)
            S = torch.linalg.inv(Hinv[idx][:, idx])
            A = Hinv[:, idx] @ S
            block = weight[r:r + tile_size, idx].float()
            weight[r:r + tile_size, :] -= (block @ A.t()).to(weight.dtype)


def random_recon(weight, tile_size, prune_ratio, hessian, seed, damp=1e-2):
    rows, cols = weight.shape
    tiles = tile_positions(rows, cols, tile_size)
    rng = _random.Random(seed)
    rng.shuffle(tiles)
    n_prune = int(len(tiles) * prune_ratio)
    pruned = tiles[:n_prune]

    repair_tiles(weight, hessian, pruned, tile_size, damp)

    return len(tiles), n_prune


def wanda_recon(weight, tile_size, prune_ratio, hessian, damp=1e-2):
    scored = wanda_scores(weight, norms_from_H(hessian), tile_size)
    scored.sort(key=lambda x: x[0])
    n_prune = int(len(scored) * prune_ratio)
    pruned = [(r, c) for _, r, c in scored[:n_prune]]

    repair_tiles(weight, hessian, pruned, tile_size, damp)

    return len(scored), n_prune
