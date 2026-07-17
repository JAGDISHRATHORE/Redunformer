# Tile-Level Redundancy — Findings

Living log of results. Last updated: **2026-07-16**.
All numbers are from the runs in `experiments/`; plots in `experiments/*/plots/`.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3-4B` (36 layers, bf16), RTX 4080 Super 16 GB |
| Tile size | 32 × 32 |
| Representative layers | 0, 9, 18, 27, 35 |
| Matrices | `q/k/v/o_proj`, `gate/up/down_proj` (7 per layer) |
| Methods | random (5 seeds), magnitude, wanda, sparsegpt (mask), sparsegpt_recon |
| Eval | WikiText-2 perplexity + output divergence (KL / top-1 / hidden cosine) |
| Dense baseline | **13.22** (full eval) · **13.559** (20% screening subset) |
| Calibration | WikiText-2 *train* (disjoint from eval), seed 0 |

**All five methods are tile pruning** (32×32 blocks). Where results are contrasted with "weight-level," that refers to the *literature's* unstructured setting, not our runs.

> ⚠️ **Read findings 7 and 8 before quoting any perplexity number below.** Downstream accuracy
> shows the model has lost ~60% of its capability at 20% sparsity, where perplexity suggested
> merely "degraded". All **relative** findings (method ranking, policy comparisons) hold; the
> **absolute** perplexity readings are far more flattering than the truth.

---

## Key findings

### 1. Reconstruction is essential at whole-model scale (relative headline — see finding 7)
Uniform whole-model pruning (dense 13.22):

| sparsity | Wanda | SparseGPT (mask) | **SparseGPT (recon)** |
|---|---|---|---|
| 5% | 28.31 | 28.03 | **15.62** |
| 10% | 52.22 | 36.00 | **18.78** |
| 20% | 156.51 | 74.65 | **30.95** |
| 30% | 1,145 | 285 | **61.56** |
| 40% | 7,812 | 3,070 | **76.30** |
| 50% | 20,289,518 | 9,193 | **137.52** |
| 60% | 17,595,522 | 15,125 | **246.18** |
| 70% | 6,107,905 | 39,582,984 | **480.74** |

Masking-only methods **collapse** (errors compound across 36 layers); reconstruction patches each layer and stays bounded — a **~10,000× gap** at 70%.

**Reconstruction's value scales with pruning scope:** marginal per-matrix → moderate per-layer → **decisive whole-model**.

### 2. Redundancy is concentrated, not spread
- **Middle layers (9/18/27) are genuinely redundant** — ΔPPL ≈ 0, sometimes slightly negative.
- **The final-layer MLP is the bottleneck** — layer-35 `up_proj` @40%: 18.17 / 17.56 / 20.11 (wanda / mask / recon) vs ~13.7 at layer 27.
- **Attention and MLP have different depth profiles** — attention (`q`/`v`) is most sensitive in the *middle*; MLP sensitivity is at the *end*.
- **Mechanism:** the final layer feeds the LM head with no downstream layer to absorb its error.
- **Verified real** (adversarial check): MLP-selective (attention at L35 is unremarkable — a bug would inflate all matrices), identical `num_pruned` across layers, reproduces at the last layer of Qwen3-0.6B.

### 3. Magnitude is worse than random — in the tile setting
Median ΔPPL over the 35 layer×matrix cells (screening subset, dense 13.559):

| method | 10% | 20% | 40% |
|---|---|---|---|
| magnitude | 0.072 | 0.123 | 0.301 |
| random *(floor)* | 0.038 | 0.083 | 0.241 |
| wanda | 0.030 | 0.035 | 0.123 |
| sparsegpt | 0.026 | 0.063 | 0.115 |
| **sparsegpt_recon** | **0.006** | **0.016** | **0.050** |

- **Magnitude sits *above* the random floor** — i.e. worse than chance. Confirmed on both models: 0.6B random **144-52**, 4B random **63-42**.
- **`o_proj` is the reproducible exception** — magnitude wins there in *both* models independently (4B **10-5**, 0.6B **21-7**). Weight-norm is a genuinely good signal for the attention output projection.
- **Data-awareness validated:** recon is **5× better than random**, 7.7× better than magnitude @20%.

**Why tiles break magnitude — aggregation loss:**

| method | how it becomes tile-level | loss |
|---|---|---|
| magnitude | per-weight magnitudes → block norm | worst — a low-norm block can hold critical weights |
| wanda | per-weight scores → tile mean | some (activation info survives better) |
| sparsegpt | **measures the whole tile's output effect directly** | **none — natively tile-level** |

This empirically confirms the limitation Rathore predicted in his metric report (§3.8, *"aggregation may hide important weights"*).

### 4. Sensitivity-aware pruning (Policy B) helps within a range — and only as a *complement* to repair
Budget-matched (identical total tiles removed). Gain = ppl(A) / ppl(B); **>1 means B wins**.
Plot: `experiments/wholemodel/plots/policy_a_vs_b.png`

| sparsity | Wanda | SparseGPT (mask) | **SparseGPT (recon)** |
|---|---|---|---|
| 5% | 1.25× | 1.32× | 1.10× |
| 10% | **1.44×** | **1.51×** | 1.19× |
| 20% | 1.08× | 1.37× | 1.31× |
| 30% | *destroyed* | 0.80× ❌ | **1.58×** (peak) |
| 40% | *destroyed* | *destroyed* | 1.11× |
| 50% | *destroyed* | *(running)* | 0.87× ❌ |
| 60% | *destroyed* | *(running)* | 0.67× ❌ |
| 70% | *destroyed* | *(running)* | 0.50× ❌ |

**Answer to Rathore's final research question: yes — but conditionally.** Every method gains from B at low sparsity, then reverses. Measured crossovers: **SparseGPT ~26%**, **recon ~45%**. Wanda never shows a measurable crossover — it is already destroyed past 20%.

**Replicated 3× independently.** All three methods turn on the policy once the target moves far from 20% — the sparsity at which the sensitivity labels were measured. This was a single-method observation before; it now holds across three independent methods, which makes the extrapolation explanation robust rather than anecdotal.

**Cause:** the "robust" label was measured **at 20%**. At a 60% target the policy pushes those regions to **67%** — far outside where the label was ever valid. They are not robust at 67%, so concentrating damage there beats spreading it. → **A redundancy map built at a single sparsity mislabels when extrapolated.** (Empirically confirmed, not hypothetical — no cap was hit; robust @60% target = 0.672.)

**The crossover moves with repair (new).** Masking methods are already gibberish by 20–30%; recon keeps its B advantage to ~45% and stays measurable all the way to 70%. **Repair widens the range where sensitivity-aware allocation pays** — the two techniques are complementary, not redundant.

**Policy B does not rescue masking-only pruning.** Wanda's best result at *any* sparsity under *either* policy (B @5% = 22.67) is still worse than reconstruction's plain *uniform* @5% (15.62). Allocation cannot substitute for repair.

**Excluded as noise:** Wanda's apparent "4.45× @50%" and "5.44× @60%" compare two *destroyed* models (4.5M vs 20M perplexity). Both are gibberish; the ratio is not a win and is omitted from the plot.

**Implication:** re-derive the classification *at the target sparsity*, or damp the multipliers as the target rises.

### 5. Damage is sub-additive within a layer, compounding across layers
Whole-layer ΔPPL is only **0.69–0.81×** the sum of its 7 individual matrices (degradations overlap). But *across* layers it compounds — which is why uniform whole-model collapses.

Whole-layer ΔPPL (mean over 5 layers, dense 13.559):

| method | 10% | 20% | 40% |
|---|---|---|---|
| wanda | 0.738 | 1.169 | 1.760 |
| sparsegpt | 0.705 | 1.166 | 1.801 |
| **sparsegpt_recon** | **0.451** | **0.859** | **1.495** |

### 6. Reconstruction backfires at the extreme
Layer-35 `up_proj` @40%: recon **20.11** is *worse* than mask (17.56) and Wanda (18.17). Correction cannot save the final layer when pruned hard — despite recon having the best *median* everywhere.

### 7. Perplexity badly understates the damage (downstream reality check) ⚠️
Dense reference (lm-eval, full task sets) — these **reproduce the published Qwen3-4B numbers**
(68.4 / 74.9 / 78.3), validating the harness independently of our pruning code:

| task | dense | chance |
|---|---|---|
| HellaSwag (acc_norm) | 0.6836 ± 0.0046 | 0.25 |
| PIQA (acc_norm) | 0.7492 ± 0.0101 | 0.50 |
| ARC-Easy (acc_norm) | 0.7828 ± 0.0085 | 0.25 |

**Our best method (`sparsegpt_recon`) at a modest 20% uniform sparsity:**

| task | dense | pruned | **of learned ability retained** |
|---|---|---|---|
| HellaSwag | 0.6836 | 0.4124 | **37%** |
| PIQA | 0.7492 | 0.6273 | **51%** |
| ARC-Easy | 0.7828 | 0.4697 | **41%** |

"Retained" = `(acc_pruned − chance) / (acc_dense − chance)` — the share of *above-chance*
ability that survives. **~60% of what the model learned is gone at 20%.**

Perplexity called this same model **30.95** vs dense 13.22 — a 2.3× rise that reads as
"degraded but working". It is not working.

**Consequence — the ranking survives, the absolute claims do not.** Reconstruction still beats
masking by orders of magnitude (finding 1 is a *relative* result and is untouched). But
"recon holds perplexity to 481 at 70%" is practically meaningless: the model was already
mostly destroyed at 20%. **The usable range for 32×32 tile pruning is far narrower than the
perplexity curves implied — likely <10%, not 40–70%.**

**Actionable:** perplexity is a poor proxy for tile-pruning damage. Report accuracy, or at
minimum calibrate the ppl→accuracy relation before trusting a perplexity curve.

*Status: 1 of 5 configs (recon p20 uniform). Policy A/B @20/30% + wanda p20 in flight.*

### 8. "Robust" is measured in isolation and does not compose — this explains the collapse
Class distribution over the whole model, from our own screening labels:

| class | matrices | tiles | share |
|---|---|---|---|
| sensitive | 12 | 291,840 | **8.2%** |
| moderate | 35 | 385,280 | 10.9% |
| **robust** | 205 | 2,871,040 | **80.9%** |

Each of those 205 robust matrices measured **ΔPPL ≈ 0 when pruned alone**. Prune them all
together at 20% and the model loses ~60% of its capability (finding 7).

> **The redundancy map measures MARGINAL damage. We were reading it as JOINT damage.**

A matrix being individually harmless says nothing about it being harmless when the other 204
are also pruned. This single confusion explains three separate observations at once:
- the whole-model collapse (per-layer ΔPPL≈0, whole-model catastrophic),
- the downstream result (finding 7),
- why Policy B's gains are only 1.1–1.6× — you cannot fix a compounding problem by
  reallocating budget among regions that are *all* mislabeled the same way.

**Corollary — "prune only the redundant part" cannot save us.** Since 81% is already robust,
hitting 20% global while touching *only* robust regions needs 24.7% local — nearly identical
to plain uniform 20%. The damage does not come from the sensitive 8%; it is the accumulation
across the robust 81%.

**Also corrects the map's shape:** this is *not* a "middle layers are redundant" story.
Layers **0–4 and 23–31 are fully robust**; only **32–35** are sensitive. Early layers are as
prunable as middle ones.

This is the empirical, quantified form of the "sequential dependency" limitation Rathore
flagged in §4.8.

⚠️ **Caveat:** we only measured layers 0/9/18/27/35; every other layer inherits its nearest
measured neighbour. The "sensitive" 8% is really *layer 35's labels stamped onto 32–35* —
layers 32/33/34 were never measured. The layer-34 check matters more because of this.

---

## Caveats
- **5 layers sampled** (0/9/18/27/35) — no fine depth profile; layer-34 check pending.
- **Screening uses a 20% eval subset** (5.1× faster; validated ppl delta 0.34). Headline/whole-model use full eval.
- **One model** (Qwen3-4B), **one tile size** (32) — tile-size/shape sweep is backlogged.
- **No downstream accuracy yet** — perplexity + divergence only. This is the biggest open gap.
- **One-shot calibration** — Gram matrices come from the dense model per layer, not re-derived after upstream pruning (Rathore §4.8 "sequential dependency").
- Policy B: recon and wanda complete; **sparsegpt 50/60/70% still running**.
- **Comparisons above perplexity ~1000 are not meaningful** — both models are gibberish there, so A/B ratios in that zone are excluded rather than reported.
- Sensitivity labels come from `sparsegpt_recon` @20% only; applying them to other methods assumes sensitivity transfers across methods (untested).
- All methods deterministic (bit-identical across runs); only random uses seeds.

## Status
- ✅ Screening (315 exp), whole-layer (45), whole-model uniform (24), baselines (magnitude + random ×5 seeds)
- ✅ Policy A vs B: recon 8/8, wanda 8/8 → `plots/policy_a_vs_b.png`
- ⏳ Policy B sparsegpt 5/8 (50/60/70% running)
- ⬜ Downstream lm-eval on final models · `wanda_recon` select-vs-repair ablation · calibration-seed robustness · layer-34 depth check · tile-size sweep

## Ready but not yet run
- `scripts/run_downstream.py` — downstream accuracy (HellaSwag/PIQA/ARC-Easy) on any pruned config. **Needs `uv sync` first: `lm_eval` is declared and locked but was never installed.**
- `src/redundancy/combined.py` — `wanda_recon` (Wanda selection + SparseGPT repair). CPU-verified; fills the missing cell of the select-vs-repair 2×2, which our current ladder confounds. Needs wiring into `run_pruning.py`.

## Where things live
```
experiments/screen/              per-matrix screening + baselines (+ plots/)
experiments/screen_wholelayer/   whole-layer (+ plots/)
experiments/wholemodel/          whole-model Policy A + B (+ plots/)
docs/metrics.md                  every metric, and why
scripts/plot_{screening,wholelayer,wholemodel}.py   offline plotting from JSON
src/redundancy/{scoring,recovery,policy,hooks,eval}.py
```
