"""Qwen vs Llama generality figures (#1). Data artifact; reads experiments/llama + known Qwen numbers."""
import json, glob, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CH={"hellaswag":0.25,"piqa":0.5,"arc_easy":0.25}
L="experiments/llama"
def acc(d,t):
    v=d["results"][t]; return v.get("acc_norm,none",v.get("acc,none"))
dense=json.load(open(f"{L}/downstream/downstream_dense.json"))
def ravg(d): return 100*sum((acc(d,t)-CH[t])/(acc(dense,t)-CH[t]) for t in CH)/3
def ld(p):
    try: return json.load(open(p))
    except: return None

doses=[1,2,5,10,20,30]
qwen_f1={1:100,2:96,5:90,10:79,20:43,30:23}
llama_f1={n:ravg(ld(f"{L}/downstream/downstream_sparsegpt_recon_p{n}_uniform.json")) for n in doses}

os.makedirs(f"{L}/plots",exist_ok=True)
# --- F1 ladder: Qwen vs Llama ---
plt.figure(figsize=(7.5,5),dpi=150)
plt.plot(doses,[qwen_f1[n] for n in doses],marker="o",color="#3c896d",lw=2,label="Qwen3-4B")
plt.plot(doses,[llama_f1[n] for n in doses],marker="s",color="#d1495b",lw=2,label="Llama-3.2-3B-Instruct")
plt.axhline(90,color="grey",ls="--",lw=1,alpha=.7); plt.axvline(5,color="grey",ls=":",lw=1,alpha=.7)
plt.text(5.3,15,"5% ceiling",fontsize=8,color="grey")
plt.xlabel("structured sparsity (%)"); plt.ylabel("retained above-chance ability (%)")
plt.title("Structured-pruning ceiling generalizes (n=2)\n~5% ceiling holds on both; Llama collapses steeper above it")
plt.xticks(doses); plt.grid(True,alpha=.3); plt.legend()
plt.tight_layout(); p1=f"{L}/plots/f1_qwen_vs_llama.png"; plt.savefig(p1,bbox_inches="tight"); plt.close()
print("saved",p1)

# --- F10 Llama: 1x1 vs 32x32 retained (where downstream exists) ---
plt.figure(figsize=(7.5,5),dpi=150)
d32={n:llama_f1[n] for n in [5,10,20,30]}
plt.plot(list(d32),list(d32.values()),marker="s",color="#d1495b",lw=2,label="32x32 (structured)")
one={40:ravg(ld(f"{L}/tilesize/downstream/us_sgpt_p40_T1.json")),
     50:ravg(ld(f"{L}/tilesize/downstream/us_sgpt_p50_T1.json"))}
plt.plot(list(one),list(one.values()),marker="^",color="#2e86ab",lw=2,label="1x1 (unstructured)")
plt.axhline(90,color="grey",ls="--",lw=1,alpha=.7)
plt.xlabel("sparsity (%)"); plt.ylabel("retained above-chance ability (%)")
plt.title("Structured-pruning tax generalizes (Llama)\n1x1 keeps ~86% at 50% where 32x32 is gone by 20%")
plt.grid(True,alpha=.3); plt.legend()
plt.tight_layout(); p2=f"{L}/plots/f10_llama_1x1_vs_32.png"; plt.savefig(p2,bbox_inches="tight"); plt.close()
print("saved",p2)
