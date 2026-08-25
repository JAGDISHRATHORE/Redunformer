"""Figures for the overnight results: #4 repair-variant downstream ladder and
#2 iterative-vs-one-shot. Data-only artifact (no prose); reads the JSON on disk."""
import json, glob, os, sys, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CH = {"hellaswag": 0.25, "piqa": 0.5, "arc_easy": 0.25}
DS = "experiments/downstream"


def acc(d, t):
    v = d["results"][t]
    return v.get("acc_norm,none", v.get("acc,none"))


dense = json.load(open(f"{DS}/downstream_dense.json"))


def ravg(d):
    return 100 * sum((acc(d, t) - CH[t]) / (acc(dense, t) - CH[t]) for t in CH) / 3


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


doses = [5, 10, 20]

# ---- #4: repair-variant downstream ladder ----------------------------------
def rr_med(dose):
    vs = [ravg(json.load(open(f))) for f in
          sorted(glob.glob(f"{DS}/downstream_random_recon_p{dose}_uniform_seed*.json"))]
    return st.median(vs), min(vs), max(vs)

series = {}
series["random_recon (random select)"] = [rr_med(d)[0] for d in doses]
series["wanda_recon (Wanda select)"] = [ravg(load(f"{DS}/downstream_wanda_recon_p{d}_uniform.json")) for d in doses]
series["sparsegpt_recon (SGPT select)"] = [ravg(load(f"{DS}/downstream_sparsegpt_recon_p{d}_uniform.json")) for d in doses]

os.makedirs(f"{DS}/plots", exist_ok=True)
plt.figure(figsize=(7.5, 5), dpi=150)
markers = {"random_recon (random select)": ("o", "#d1495b"),
           "wanda_recon (Wanda select)": ("s", "#2e86ab"),
           "sparsegpt_recon (SGPT select)": ("^", "#3c896d")}
for name, ys in series.items():
    mk, col = markers[name]
    plt.plot(doses, ys, marker=mk, color=col, linewidth=2, label=name)
# random_recon seed band at each dose
lo = [rr_med(d)[1] for d in doses]; hi = [rr_med(d)[2] for d in doses]
plt.fill_between(doses, lo, hi, color="#d1495b", alpha=0.15)
plt.axhline(90, color="grey", ls="--", lw=1, alpha=0.7)
plt.text(20.3, 90.5, "90% (capability-preserving)", fontsize=8, color="grey", ha="right")
plt.gca().invert_xaxis() if False else None
plt.xlabel("structured sparsity (%)")
plt.ylabel("retained above-chance ability (%)")
plt.title("Repair variants: real capability vs sparsity (tile-32)\n"
          "random_recon has the BEST perplexity but the WORST capability at every dose")
plt.xticks(doses)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
p4 = f"{DS}/plots/f_repair_downstream_ladder.png"
plt.savefig(p4, bbox_inches="tight")
plt.close()
print("saved", p4)

# ---- #2: iterative K=4 vs one-shot -----------------------------------------
oneshot = {d: ravg(load(f"{DS}/downstream_sparsegpt_recon_p{d}_uniform.json")) for d in doses}
iter_k4 = {d: ravg(load(f"experiments/iterative/downstream_iter_sgr_K4_p{d}.json")) for d in doses}
os.makedirs("experiments/iterative/plots", exist_ok=True)
plt.figure(figsize=(7.5, 5), dpi=150)
plt.plot(doses, [oneshot[d] for d in doses], marker="o", color="#2e86ab", lw=2,
         label="one-shot sparsegpt_recon (K=1)")
plt.plot(doses, [iter_k4[d] for d in doses], marker="s", color="#e07a5f", lw=2, ls="--",
         label="iterative dose-split (K=4)")
for d in doses:
    plt.annotate(f"{iter_k4[d]-oneshot[d]:+.0f}", (d, iter_k4[d]), textcoords="offset points",
                 xytext=(0, -14), ha="center", fontsize=8, color="#e07a5f")
plt.xlabel("structured sparsity (%)")
plt.ylabel("retained above-chance ability (%)")
plt.title("Iterative (LS-only) calibration does NOT raise the ceiling\n"
          "K=4 dose-split ≈ one-shot at every dose (Δ in orange)")
plt.xticks(doses)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
p2 = "experiments/iterative/plots/iter_vs_oneshot.png"
plt.savefig(p2, bbox_inches="tight")
plt.close()
print("saved", p2)
