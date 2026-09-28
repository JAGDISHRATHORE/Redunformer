import json
import glob
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEPTH_DIR = "experiments/experiment_2_qwen3_4b/results/legacy_tile64/depth"
OUT_DIR = os.path.join(DEPTH_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

DENSE_PPL = 13.22
PPL_ALL36 = 30.95
INK, MUTED = "#1a1a1a", "#6b6b6b"
ACCENT, GOOD, BAD = "#0072B2", "#009E73", "#D55E00"


def load():
    rows = []
    for f in glob.glob(os.path.join(DEPTH_DIR, "wholemodel_*.json")):
        d = json.load(open(f))
        ppl = d.get("perplexity")
        if ppl is None and d.get("results"):
            ppl = d["results"][0].get("perplexity")
        per_layer = d.get("per_layer") or []
        n = len({e.get("layer") for e in per_layer if e.get("layer") is not None}) or None
        if n and ppl:
            rows.append((n, float(d["prune_ratio"]), ppl))
    return sorted(rows)


def main():
    rows = load()
    if not rows:
        print("no depth results yet")
        return
    print(f"{'layers':>7} {'local':>7} {'ppl':>10}   vs uniform-36")
    for n, r, p in rows:
        print(f"{n:>7} {r:>7.3f} {p:>10.2f}   {PPL_ALL36 / p:>5.2f}x")

    ns = [r[0] for r in rows]
    ratios = [r[1] for r in rows]
    ppls = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6.4))

    ax.axhline(PPL_ALL36, color=BAD, linestyle="--", linewidth=1.6)
    ax.annotate(f"all 36 layers @ 20%  →  ppl {PPL_ALL36}\n(the reference: same tiles, spread thin)",
                xy=(min(ns) - 0.4, PPL_ALL36 * 1.04), fontsize=9, color=BAD, fontweight="bold")
    ax.axhline(DENSE_PPL, color=MUTED, linestyle=":", linewidth=1.3)
    ax.annotate(f"dense {DENSE_PPL}", xy=(max(ns) + 2.6, DENSE_PPL * 1.06), fontsize=8.5, color=MUTED)

    ax.plot(ns, ppls, "-o", color=ACCENT, linewidth=2.4, markersize=9, zorder=5)
    for n, r, p in rows:
        better = p < PPL_ALL36
        ax.annotate(f"{int(r * 100)}% local\n{p:,.1f}", xy=(n, p),
                    xytext=(10, -20 if better else 10),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5,
                    color=GOOD if better else BAD, fontweight="bold", zorder=6)

    ax.axhspan(0, PPL_ALL36, color=GOOD, alpha=0.05, linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel("number of layers absorbing the budget  (fewer = more concentrated →)", fontsize=11)
    ax.set_ylabel("WikiText-2 perplexity (log scale)", fontsize=11)
    fig.suptitle("Depth concentration — identical tile budget, distributed differently",
                 fontsize=15, fontweight="bold", color=INK, y=0.99)
    ax.set_title("every point removes exactly 709,632 tiles (7.2 layer-equivalents = 20% of the model) · "
                 "budget drawn only from layers 0-31",
                 fontsize=9.5, color=MUTED, pad=8)
    ax.grid(alpha=0.25, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(max(ns) + 3.5, min(ns) - 4.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "depth_concentration.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
