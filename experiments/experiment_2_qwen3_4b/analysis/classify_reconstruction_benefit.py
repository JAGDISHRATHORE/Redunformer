import json
import glob
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCREEN_DIR = "experiments/experiment_2_qwen3_4b/results/01_per_matrix_screening"
OUT_DIR = "experiments/experiment_2_qwen3_4b/figures/01_per_matrix_screening"
os.makedirs(OUT_DIR, exist_ok=True)

DENSE = 13.559
RATIO = "0.20"
LAYERS = [0, 9, 18, 27, 35]
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

FIX_FRAC = 0.60
CCOL = {"A": "#009E73", "B": "#0072B2", "C": "#D55E00"}
CNAME = {"A": "A · naturally redundant", "B": "B · compensatable", "C": "C · essential"}
INK, MUTED = "#1a1a1a", "#6b6b6b"


def load_screen(method, ratio=RATIO):
    out = {}
    for f in glob.glob(os.path.join(SCREEN_DIR, f"{method}_p{ratio}", "layer*.json")):
        d = json.load(open(f))
        if d["layer"] not in LAYERS:
            continue
        for r in d["results"]:
            out[(d["layer"], r["matrix"])] = r["perplexity"] - DENSE
    return out


def noise():
    cells = defaultdict(list)
    for f in glob.glob(os.path.join(SCREEN_DIR, f"random_p{RATIO}", "layer*_seed*.json")):
        d = json.load(open(f))
        if d["layer"] not in LAYERS:
            continue
        for r in d["results"]:
            cells[(d["layer"], r["matrix"])].append(r["perplexity"] - DENSE)
    spreads = [np.std(v, ddof=1) for v in cells.values() if len(v) > 1]
    if not spreads:
        return None, 0
    return float(np.median(spreads)), len(spreads)


def classify(mask_d, recon_d, thresh):
    out = {}
    for k, m in mask_d.items():
        if k not in recon_d:
            continue
        r = recon_d[k]
        if m < thresh:
            out[k] = ("A", m, r, None)
            continue
        repaired = (m - r) / m if m > 0 else 0.0
        out[k] = ("B" if repaired >= FIX_FRAC else "C", m, r, repaired)
    return out


def main():
    mask_d = load_screen("sparsegpt")
    recon_d = load_screen("sparsegpt_recon")
    if not mask_d or not recon_d:
        print("missing mask-only or reconstructed screening data")
        return

    sigma, n = noise()
    if sigma is None:
        print("no random control found; cannot set a principled threshold")
        return
    thresh = 2 * sigma
    print(f"random control: median per-cell sigma over seeds = {sigma:.4f} ppl (from {n} cells)")
    print(f"'meaningful damage' bar = 2 sigma = {thresh:.4f} ppl  [not hand-picked]")
    print(f"'compensatable' = repair removes >= {FIX_FRAC:.0%} of the damage\n")

    classes = classify(mask_d, recon_d, thresh)
    counts = defaultdict(int)
    for c, *_ in classes.values():
        counts[c] += 1
    total = sum(counts.values())
    print(f"{'class':26} {'cells':>6}  share")
    for c in "ABC":
        print(f"  {CNAME[c]:24} {counts[c]:>6}  {counts[c] / total * 100:5.1f}%")
    print()

    print("Category C — repair cannot fix these (the real bottlenecks):")
    for k, (c, m, r, rep) in sorted(classes.items(), key=lambda x: -x[1][1]):
        if c == "C":
            print(f"  layer {k[0]:2} {k[1]:10}  mask +{m:6.2f} -> recon +{r:6.2f}  (repaired {rep:5.1%})")
    print()
    print("Category B — repair does the work (biggest rescues):")
    b = [(k, v) for k, v in classes.items() if v[0] == "B"]
    for k, (c, m, r, rep) in sorted(b, key=lambda x: -x[1][1])[:6]:
        print(f"  layer {k[0]:2} {k[1]:10}  mask +{m:6.2f} -> recon +{r:6.2f}  (repaired {rep:5.1%})")

    fig, ax = plt.subplots(figsize=(9.5, 7.4))
    lim = max(max(mask_d.values()), 0.5) * 1.35

    ax.plot([0, lim], [0, lim], "--", color=MUTED, linewidth=1.3, alpha=0.85, zorder=2)
    ax.annotate("ABOVE the line = repair made it WORSE", xy=(lim * 0.30, lim * 0.40), fontsize=9.5,
                color="#D55E00", style="italic", fontweight="bold", rotation=38)
    ax.annotate("below the line = repair helped", xy=(lim * 0.60, lim * 0.30), fontsize=9,
                color="#0072B2", style="italic", rotation=38)
    ax.axvspan(-lim * 0.03, thresh, color=CCOL["A"], alpha=0.07, linewidth=0)
    ax.annotate(f"A: damage below the random control's\nnoise floor (2σ = {thresh:.3f})",
                xy=(thresh * 1.3, lim * 0.93), fontsize=8.5, color=CCOL["A"])

    for k, (c, m, r, rep) in classes.items():
        ax.scatter([m], [r], s=64, color=CCOL[c], alpha=0.85, edgecolors="white", linewidths=0.6,
                   zorder=4)
        if c == "C" and m > 0.3:
            ax.annotate(f"L{k[0]} {k[1].replace('_proj', '')}  ({rep:+.0%})", xy=(m, r), xytext=(8, -4),
                        textcoords="offset points", fontsize=8.2, color=CCOL[c],
                        fontweight="bold")

    for c in "ABC":
        ax.scatter([], [], s=64, color=CCOL[c], label=f"{CNAME[c]}  ({counts[c]})")

    ax.set_xlim(-lim * 0.03, lim)
    ax.set_ylim(-lim * 0.03, lim)
    ax.set_xlabel("damage with NO repair — ΔPPL, mask-only (eq. 22)", fontsize=10.5)
    ax.set_ylabel("damage AFTER repair — ΔPPL, reconstructed (eq. 23)", fontsize=10.5)
    fig.suptitle("Reconstruction-benefit classification — where does repair do its work?",
                 fontsize=14.5, fontweight="bold", color=INK, y=0.98)
    ax.set_title("each point = one layer×matrix at 20% tile sparsity · far below the diagonal = repair rescued it",
                 fontsize=9.5, color=MUTED, pad=8)
    ax.grid(alpha=0.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "reconstruction_classes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("\nsaved", out)


if __name__ == "__main__":
    main()
