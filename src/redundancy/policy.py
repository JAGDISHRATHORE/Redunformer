import json
import glob
import os

SENSITIVE, MODERATE, ROBUST = "sensitive", "moderate", "robust"


def load_sens(screen_dir, method="sparsegpt_recon", ref_ratio="0.20", dense_ppl=13.559):
    sens = {}
    for f in glob.glob(os.path.join(screen_dir, f"{method}_p{ref_ratio}", "layer*.json")):
        d = json.load(open(f))
        for r in d["results"]:
            sens[(d["layer"], r["matrix"])] = r["perplexity"] - dense_ppl
    if not sens:
        raise ValueError(f"No screening data found for {method} p{ref_ratio} in {screen_dir}")
    return sens


def classify(sens, hi_thr=0.30, lo_thr=0.05):
    out = {}
    for k, v in sens.items():
        if v >= hi_thr:
            out[k] = SENSITIVE
        elif v >= lo_thr:
            out[k] = MODERATE
        else:
            out[k] = ROBUST
    return out


def nearest_layer(layer, measured):
    return min(measured, key=lambda m: abs(m - layer))


def fill_all_layers(classes, layer_list, matrices, measured):
    full = {}
    for layer in layer_list:
        src = nearest_layer(layer, measured)
        for m in matrices:
            full[(layer, m)] = classes.get((src, m), ROBUST)
    return full


def allocate_budget(cls_all, tiles, target,
                    mults=None, max_ratio=0.95, tol=1e-6):
    if mults is None:
        mults = {SENSITIVE: 0.30, MODERATE: 1.00, ROBUST: 1.60}

    n_total = sum(tiles.values())
    budget = target * n_total

    def removed(scale):
        return sum(min(mults[cls_all[k]] * scale, max_ratio) * n for k, n in tiles.items())

    lo, hi = 0.0, max_ratio / min(mults.values()) + 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if removed(mid) < budget:
            lo = mid
        else:
            hi = mid
        if abs(removed(mid) - budget) <= tol * n_total:
            break
    scale = (lo + hi) / 2

    ratios = {k: min(mults[cls_all[k]] * scale, max_ratio) for k in tiles}
    achieved = removed(scale) / n_total
    by_class = {}
    for c in (SENSITIVE, MODERATE, ROBUST):
        ks = [k for k in tiles if cls_all[k] == c]
        if ks:
            by_class[c] = {
                "ratio": min(mults[c] * scale, max_ratio),
                "matrices": len(ks),
                "tiles": sum(tiles[k] for k in ks),
            }
    return ratios, {"target": target, "achieved": achieved, "scale": scale, "per_class": by_class}
