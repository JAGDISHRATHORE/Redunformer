import numpy as np


def pool_layer_budget(scores, budget, mode="mean"):
    pooled = []
    for m, scored in scores.items():
        vals = np.array([s for s, _, _ in scored], dtype=np.float64)
        if mode == "mean":
            denom = vals.mean()
            norm = vals / denom if denom > 0 else vals
        elif mode == "raw":
            norm = vals
        else:
            raise ValueError(f"unknown mode {mode!r}")
        for (nv, (_, r, c)) in zip(norm, scored):
            pooled.append((float(nv), m, r, c))

    total = len(pooled)
    n_prune = int(total * budget)
    pooled.sort(key=lambda x: x[0])
    chosen = pooled[:n_prune]

    pruned = {m: [] for m in scores}
    for _, m, r, c in chosen:
        pruned[m].append((r, c))

    info = {
        "mode": mode,
        "budget_ratio": budget,
        "layer_total_tiles": total,
        "layer_pruned_tiles": n_prune,
        "achieved_ratio": n_prune / total if total else 0.0,
        "per_matrix": {
            m: {
                "tiles": len(scores[m]),
                "pruned": len(pruned[m]),
                "ratio": len(pruned[m]) / len(scores[m])
                if scores[m] else 0.0,
            }
            for m in scores
        },
    }
    return pruned, info
