# Tile-Level Redundancy in LLMs — Findings (Qwen3-4B)

**One line:** You can delete about **5% of the model** (as 32×32 weight-tiles) before its real-task ability starts to break — *far* less than perplexity implies — and the thing that makes pruning work is **repairing the surviving weights, not cleverly choosing which tiles to cut.**

*Model: Qwen/Qwen3-4B. Tile size: 32×32 (project standard, Rathore §2.4). Dense WikiText perplexity = 13.22. Downstream ability measured on HellaSwag / PIQA / ARC-Easy via lm-eval, normalised to above-chance range: retained = (acc − chance) / (acc_dense − chance).*

---

## How to read this doc

Every finding below is written as **What we test → Why we ran it → Result → Reliability**. The *Reliability* tag is the single most important label:

- **downstream-anchored** — judged on real task accuracy. Trust these most.
- **control-relative** — a comparison against a matched control (e.g. random tiles) that shares the same biases, so the *ranking* is trustworthy even if absolute numbers aren't.
- **perplexity-absolute** — rests on raw perplexity; directionally useful but carries the caveat that perplexity misleads (that's Finding 2).

> **Data note:** all headline numbers are at the corrected **32×32** tile size. A prior bug ran some controls at 64×64; those were re-run (commit `a1be30e`). A few findings still rest on archived 64×64 "shape evidence" — flagged explicitly where they occur.

---

## The four things worth saying in the presentation

1. **~5% is the real redundancy budget** (downstream-anchored) — perplexity suggested 40–70%, and it was wrong.
2. **Perplexity is not just a loose proxy — it is *gameable*.** You can push perplexity *below dense* while the model gets measurably worse on real tasks.
3. **Repair is the whole method.** Which tiles you delete barely matters; reconstructing the surviving weights is what preserves ability. Calibrated tile-selection loses to a coin flip at the usable operating point.
4. **Redundancy is not a per-part property.** Pieces that are safe alone are not safe together, and safety is the *interaction* of depth × matrix-type, not either alone.

---

## Reliability at a glance

| Finding | Rests on | Status |
|---|---|---|
| 1 — ~5% redundancy ladder | **downstream-anchored** | ✅ holds |
| 2 — perplexity is nonlinear | **downstream-anchored** | ✅ holds |
| 2b — perplexity is *gameable* | downstream drop tile-32; ppl-gaming tile-64 | ⚠️ holds, but "one model" pairing is cross-tile (see caveats) |
| 3 — Policy B games its own metric | **downstream-anchored** | ✅ holds |
| 4 / 4e — repair ≫ selection | control-relative (same-tile ablation) | ✅ holds-stronger (clean tile-32 paired ablation) |
| 4b — repair backfires at final MLP | control-relative | ✅ holds |
| 4c / 4d — selection loses to random | control-relative (corrected tile-32 floor) | ✅ holds-stronger |
| 5 — isolation does not compose | **downstream-anchored** | ✅ holds |
| 6 / 6b — reallocation optimum | control-relative; depth runs tile-64 | ⚠️ shape holds; N=24 optimum unconfirmed at tile-32 |
| 7 — depth × matrix interaction | control-relative; **tile-64, perplexity-only** | ⚠️ holds as marginal claim (see caveats) |
| 8 — magnitude < random | control-relative (vs random control) | ✅ holds |
| 9 — damage sub-additive in-layer, compounds across | perplexity-absolute | ⚠️ direction holds, carries caveat |

---

# Findings

## 1 — About 5% of tiles are removable before real ability breaks ⭐ (downstream-anchored)

**What we test.** How much of the model we can delete in 32×32 tiles before it fails on actual reasoning tasks — not just on a text-prediction score. We prune 1/2/5/10/20/30% with the best method (Wanda-select + SparseGPT-repair) and measure HellaSwag / PIQA / ARC-Easy.

**Why we ran it.** Our perplexity curves implied 40–70% was removable. Perplexity is only a proxy; we wanted the honest number judged on abilities a person cares about.

**Result.** ~5% of tiles can be removed while keeping **~90%** of above-chance ability — strikingly, exactly 90/90/90 across all three tasks. Past that it degrades steadily. The honest nuance: 5% is not a cliff — 10% is still a *usable-but-degraded* ~79%; the real collapse is between 10% and 20% (79% → 43%).

**Numbers (retained above-chance ability, HellaSwag/PIQA/ARC-Easy; tile-32):** 1% → 100/103/98; 2% → 97/98/93; **5% → 90/90/90 (avg 90%)**; 10% → 73/81/82 (avg 79%); 20% → 37/51/41 (avg 43%); 30% → 17/29/24 (avg 23%). Paired WikiText ppl: 5%=15.62, 10%=18.78, 20%=30.95, 30%=61.56 (all tile-32).

**Figure:** `figures/f1_redundancy_ladder.png` — retained ability vs sparsity, three tasks, 90% line + 5% marker.

---

## 2 — Perplexity is a *nonlinear* proxy, dangerous exactly where you'd rely on it ⭐ (downstream-anchored)

**What we test.** Whether WikiText perplexity actually tracks real ability across the sparsity range.

**Why we ran it.** Perplexity is the fast, cheap number we'd instinctively use to rank methods. If it flatters an aggressive setting, any perplexity-only headline is unsafe.

**Result.** Perplexity never points the *wrong* way, but the mapping to capability is brutally nonlinear. At 5% ppl reads 1.18× dense and the model keeps ~90% of ability; at 20% ppl reads only 2.34× ("about twice as bad") yet **over half the model's ability is gone** (43% retained); at 30%, 4.66× leaves 23%. So perplexity is fine in the usable regime (≤5%) and dangerously flattering exactly where you'd use it to justify aggressive pruning.

**Numbers (tile-32):** ppl-ratio → retained-avg: 1.18×→90%, 1.42×→79%, 2.34×→43%, 4.66×→23%.

**Figure:** `figures/f2_perplexity_scissors.png` (shared with 2b).

---

## 2b — Perplexity is *adversarially gameable*: push it below dense while the model gets worse ⭐⭐ (mixed tile — see caveat)

**What we test.** Whether you can deliberately make perplexity look *better* while real ability drops — by masking o_proj only in the layers where it helps in isolation.

**Why we ran it.** This is the sharpest possible statement of Finding 2, and the reason we ran ~11 downstream evals: if perplexity can be *driven the opposite direction* from capability, the field's default yardstick is not just loose but exploitable.

**Result.** Restricting Wanda o_proj masking to the 8 layers where o_proj improved in isolation (17–21, 32, 34, 35) drives WikiText perplexity **below dense** — 12.24 at 20% dose (−0.98) and **11.50 at 40% (−1.72, i.e. 13% "better" than dense)** — while real ability goes flat then down: at 40% HellaSwag drops a genuine **−3.6σ**, ARC-Easy −1.5σ, PIQA flat. Two controls make it airtight: blanket o_proj across all 36 layers never beats dense (14.61 at 20%), and SparseGPT **repair erases the perplexity win entirely** (13.29 / +0.07) — because repair reconstructs the dense output. The "gain" exists *only because you didn't reconstruct*: a textbook proxy exploit.

**⚠️ Caveat (most attackable claim in the set):** the perplexity-gaming numbers are archived **tile-64**; the downstream drop is **tile-32** (`run_downstream.py` re-prunes fresh at tile-32 and logs no perplexity). So the "two metrics move in opposite directions *in one network*" table is a **cross-tile composite**. Each half is solid within its own tile size; before presenting it as one model, we should run one tile-32 perplexity measurement on the exact gamed model. *(This is scheduled — see Pending.)*

**Figure:** `figures/f2_perplexity_scissors.png` — capability-vs-perplexity-ratio scatter; the o_proj points sit left of the dense line ("looks improved") yet at/below dense ability.

---

## 3 — "Smart" layer budgeting (Policy B) games the metric it was built from ⭐⭐ (downstream-anchored)

**What we test.** Whether a sensitivity-aware budget that protects fragile regions (Policy B) beats plain uniform pruning (Policy A) at the same compression — *and* whether its perplexity win shows up on real tasks.

**Why we ran it.** The sensitivity map was built *from perplexity*, so Policy B risks just flattering the number it was tuned on. Downstream accuracy at matched budgets separates a genuine win from a mirage.

**Result.** Both halves hold. Policy B genuinely edges uniform on perplexity by ~1.1–1.6× (peaking 1.58× at 30%), but that **badly overstates the capability win**: at its 1.58× peak it buys essentially **zero** real ability (+3.1/−0.6/−0.4 pp = noise). It only wins for real at lower sparsity (20%: +5.2/+1.3/+7.1 pp). At *matched perplexity* it delivers ~10–18 pp **less** capability than uniform on all 6 measurements. Confirmed in metadata: it protects the perplexity-flagged "sensitive" class (pruned 0.042) and hammers the "robust" class (0.224) — and because that map came from perplexity (dominated by final layers feeding the LM head), it flatters the metric it was fit to.

**Numbers (tile-32, sparsegpt_recon):** ppl gain 1.10/1.19/1.31/1.58/1.11× at 5/10/20/30/40%; downstream Δ(B−A) = +5.2/+1.3/+7.1 pp at 20%, +3.1/−0.6/−0.4 pp at 30%.

**Figure:** `figures/f3_policyB_divergence.png` — perplexity-gain bars vs capability-gain line; they diverge at 30%.

---

## 4 — Repair is everything; *which tiles you pick* barely matters ⭐ (control-relative)

**What we test.** Whether the model's recovery comes from *selecting* good tiles to delete or from *repairing* (least-squares reconstruction of) the surviving weights after deletion.

**Why we ran it.** These are the two levers. Knowing which one carries the method tells us where to spend effort — and it's the opposite of the field's instinct (which obsesses over selection).

**Result (now proven with the full tile-32 paired ablation).** Repair carries the entire method — and *which tiles you pick contributes nothing*. At 5% whole-model, **repairing after deleting *random* tiles (`random_recon`, 14.64) is the best result of all** — better than repairing after "smart" Wanda selection (`wanda_recon` 15.38) or SparseGPT selection (`sparsegpt_recon` 15.62), and all three land near dense (13.22). Selection *without* repair (Wanda 28.31, SparseGPT 28.03) is beaten by a plain coin flip (random 22.87). The causal proof is the same-tile pair: `random` and `random_recon` prune the **identical** tiles at the same seed, so adding repair alone drops perplexity **22.87 → 14.64**. Repair is the whole lever; calibrated selection not only fails to beat random, random-select-then-repair actually *edges out* calibrated-select-then-repair.

**Numbers (tile-32, 5% whole-model, dense 13.22):** repair — random_recon 14.64 (median of 5 seeds, 14.2–14.9), wanda_recon 15.38, sparsegpt_recon 15.62; no-repair — random 22.87; selection-only — wanda 28.31, sparsegpt 28.03; magnitude 3449.

**Figure:** `figures/f4_repair_vs_selection.png` — repair methods (near dense) vs selection-only (2× worse) vs no-repair random, at 5%.

## 4b — Repair backfires exactly where it's needed most (final-layer MLP) (control-relative)

**Result.** At the deepest layer (L35), adding *more* calibration makes things *worse*, and the globally-best method — `sparsegpt_recon` — is the **single worst** choice there (dPPL 5.39 at 40%, vs random's 2.38). Repairing against a stale/ill-conditioned final-layer signal actively harms. This is the one place the "always repair" rule inverts.

## 4c / 4d — At the usable operating point, tile-selection loses to a coin flip ⭐⭐ (control-relative, corrected tile-32)

**What we test.** Whether data-aware tile scores (Wanda, SparseGPT) beat deleting *random* tiles — at 5% whole-model, and layer by layer.

**Why we ran it.** A calibrated selection rule only earns its complexity if it beats chance. The earlier controls ran at the wrong tile size (64); we re-ran random + magnitude at tile-32 to give the claim a valid floor.

**Result (sharper after correction).** At **5% whole-model** — the setting where the pruned model still works — Wanda and SparseGPT each beat only **1 of 5** random seeds and lose to the random median (22.87). Per layer: **L0** data-aware is essential (random blows up at 40%); **middle layers** ~indistinguishable from random until 40%; **L35** selection **backfires** — all 9 calibrated method×sparsity cells are worse than random (z = +3.5 to +6.6), and damage grows monotonically with how much calibration a method uses.

**⚠️ Scope note (important for the talk):** "selection loses to random" is a claim about the **5% operating point**. Above 5% (10–20%), selection *does* start beating random — but the model is already past the capability cliff there (random median 221 ppl at 10%, 178k at 20%), so it's a race between broken models. State it as: *at the only sparsity where capability is preserved, selection is worthless; above it, everything without repair is dead anyway.*

**Numbers (tile-32):** 5% whole-model — random seeds 18.0/20.2/22.9/26.7/48.0 (median 22.9); Wanda 28.3, SparseGPT 28.0 (each beat 1/5); magnitude 3,449 (0/5); sparsegpt_recon 15.6 (5/5). L35@40% dPPL: random 2.38 (best) < magnitude 3.20 < sparsegpt 4.53 < wanda 4.80 < sparsegpt_recon 5.39 (worst).

**Figure:** `figures/f4_L35_backfire.png` — L35 dPPL by method ordered by calibration amount, random floor shaded.

---

## 5 — "Robust in isolation" does not compose: marginal safety ≠ joint safety ⭐ (downstream-anchored)

**What we test.** Whether a piece being prunable *on its own* tells you it's safe to prune once *many* pieces go at once.

**Why we ran it.** Our redundancy map is built one matrix at a time. The entire strategy leans on trusting that map jointly — this checks whether that assumption holds.

**Result.** It doesn't. One at a time, most of the model is harmless: at 20% sparsity 83% of screened matrices (with repair) barely move perplexity, and even a whole single layer's 7 matrices together stay near dense (except final-layer MLP). Yet pruning the **whole model** at 20% destroys most real ability (37/51/41% retained, ~60% gone). The map measures **marginal** damage; we were reading it as **joint**. The non-composition is an across-layer effect: 36 individually-fine hits compound. This is the quantified sequential-dependency limit.

**Numbers (tile-32):** isolated @20% — 83% of matrices ΔPPL<0.12; whole-layer joint @20% — L0/9/18/27 ≤ +0.4 over dense, L35 = 17.16; whole-model joint @20% — ppl 30.95, ~43% ability retained.

**Figure:** `figures/f5_scope_escalation.png` — perplexity vs pruning scope (matrix → layer → whole model), whole-model bar annotated "only 43% ability retained."

---

## 6 / 6b — Reallocation has a sweet spot, and only pays where structure exists (control-relative)

**What we test.** (6) Given a fixed tile budget, is it better to spread cuts thinly across many layers or pack them densely into fewer? (6b) When a layer gets one budget, does letting its 7 matrices share it unevenly beat forcing uniform sparsity?

**Why we ran it.** Two forces pull opposite ways: damage compounds across layers (→ concentrate) but grows super-linearly within a layer past ~30–45% (→ spread). If both are real there's an optimum; and clever sharing should help only where a layer has uneven sensitivity to exploit.

**Result.** Concentration traces a clean **inverted-U**: quality peaks at **N=24 layers** (30% local sparsity, ppl 24.17), degrades gently toward spreading (N=32 → 24.94) and steeply toward over-concentration (collapsing 61× by N=8, ppl 1890). Layer-wide budget matching (6b) is a coin flip overall (7/15) but not noise: it wins big at L35 (up to 2.75×, real structure) and loses at L0 (0.61×, uniformly robust so reshuffling only concentrates damage). Reallocation helps **only where structure exists**.

**⚠️ Caveat:** the depth runs (Finding 6) are archived **tile-64**. The inverted-U *shape* is defensible; the absolute multipliers and the **N=24 optimum are not yet confirmed at tile-32** (the vs-uniform column mixes a tile-32 denominator). Treat N=24 as "holds pending a tile-32 re-run." 6b is clean tile-32.

**Figure:** `figures/f6_inverted_u.png` — perplexity vs concentration (log y), minimum marked at N=24; inset of 6b per-layer win-ratios.

---

## 7 — Robustness is the depth × matrix-type *interaction*, not either alone ⭐⭐ (control-relative, tile-64)

**What we test.** Whether prunability is set by depth, by matrix type, or specifically by the *combination* — pruning o_proj (attention-out) vs up_proj (MLP) across a mid band (15–21) and final layers (29–35).

**Why we ran it.** If it were purely "late layers fragile" or "MLP fragile," a one-line rule would suffice. If it's the pairing, every 1-D rule is wrong in principle and per-(layer, matrix) budgeting is justified.

**Result.** The **combination wins in all 6 cells** — the interaction term exceeds both main effects every time. o_proj is safe to prune at *every* depth (even lowers perplexity at the final layer); up_proj is safe everywhere *except* the last one or two layers, where it explodes (+4.70 dPPL at L35 vs +0.06 at L29). So "late layers sensitive" and "MLP sensitive" are each false alone — only *the final layers' MLP* is. Replicates across Wanda (mask) and sparsegpt_recon (repair), so it's a model property, not a method artifact.

**⚠️ Caveat + reconciliation:** this is **tile-64, perplexity-only, no downstream anchor** — and it's a *marginal* (one-matrix-at-a-time) claim. Finding 2b shows that o_proj-driven perplexity *below dense* can hide real capability loss when done *jointly* across many layers. These are reconcilable: Finding 7 is marginal, 2b is 8-layer joint. **Present 7 as a marginal-sensitivity map, not a deployment recommendation.** (Also: the "8 o_proj-improving layers" in 2b come from the 14-layer cluster scan; the "32/36 improving" in 7 from the full-model scan — different scans, name both.)

**Figure:** `figures/f7_depth_matrix_interaction.png` — dPPL vs depth, o_proj vs up_proj lines fanning apart only at L34–35.

---

## 8 — Magnitude pruning is *worse than random* — and we know why (control-relative)

**What we test.** When deleting 32×32 tiles, is picking the smallest-magnitude tiles (the textbook default) better or worse than random?

**Why we ran it.** Magnitude is the obvious rule. If it can't even beat random block selection, tile pruning genuinely needs a smarter signal and naive intuition is a trap.

**Result.** At 5% (the usable point) magnitude **wipes the model out** (ppl 3449, ~260× dense) while random stays roughly functional (median 22.9, ~1.7×) — magnitude is **~150× worse than a coin flip**. Reproduces on the 0.6B model across 196 layer×matrix cells (random beats magnitude 144–52; the one consistent exception is o_proj). Mechanism (unifies with F9): per cell magnitude is only *slightly* worse than random, but those tiny deficits **compound across 36 layers** into the 150× model-scale gap ("aggregation loss").

**Figure:** `figures/f8_magnitude_vs_random.png` — ppl (log) vs sparsity, magnitude vs random-median-with-band, 5% point annotated.

## 9 — Damage is sub-additive within a layer, compounding across layers (perplexity-absolute)

**Result.** Within a layer, damage overlaps (whole-layer ≈ 0.5–0.8× the sum of its 7 matrices for the controls) → sub-additive. Across layers it super-compounds (whole-model damage runs 17× to thousands× the sum of per-layer damage). **Caveats:** the exact 0.5–0.8 band is one slice — sub-additivity is method-dependent (data-aware methods sit near 1.0, ~additive), L0 is a super-additive exception, and this is perplexity-absolute. F8 (a magnitude-vs-random comparison at identical settings) is robust to that caveat; F9 carries it.

---

# Two things we are careful about (for the talk)

1. **"Usable operating point" means 5%.** Findings 4c/4d/8 say selection/magnitude lose "at the only usable point" — that point is 5%, where capability is preserved. Finding 1 correctly notes 10% is still usable-but-degraded (~79%). Above 5%, selection/magnitude comparisons flip or become races between broken models. We apply the 5% definition consistently and say so.
2. **The o_proj / perplexity story is marginal, not a recommendation.** Finding 7's "o_proj is safe everywhere" is a *one-at-a-time, perplexity-only, tile-64* observation; Finding 2b shows the joint, downstream-anchored reality (gaming). The two agree once you separate marginal from joint scope.

---

# Pending experiments

## ★ The tile-size experiment — is 5% real, or an artifact of the 32×32 tile? (flagship)

**The question.** Every finding above is at 32×32, which is *large*. Unstructured pruning (1×1) recovers ~50% on LLMs; we get ~5%. So the ~5% ceiling — and "selection ≈ random" and "magnitude < random" — may be **artifacts of coarse granularity**, not real properties of Qwen3-4B. This experiment settles it. *(Designed by a 6-lens planning council.)*

**The design.** Treat tile size **T as the redundancy's *correlation length*** and measure it three ways that must agree:
1. **Purity probe** (Seb's front-end — *predicts*): a weight-space analysis that measures whether the removable 5% is spatially *clustered* (at some scale S) or *diffuse*. Mark the bottom-5% of weights by importance (Wanda + SparseGPT-saliency maps) and, for each T, measure the fraction of near-pure tiles vs a permutation null. Diffuse → smaller tiles always better, no sweet spot. Clustered at S → optimal tile ≈ S. Runs in minutes; sets the sweep priority.
2. **Perplexity/KL screen** (*previews*): fast proxy sweep, clearly labelled "not a capability claim."
3. **Downstream retained-ability ladder** (*decides*): the citable curve.

**What's swept:** new square sizes **{16, 8, 4}** + **1×1** (unstructured ceiling, separate vectorized code) — **32×32 is the existing reference, not re-run.** The model's matrix dims (2560/4096/1024/9728) divide every T exactly, so "% weights zeroed" = "% tiles zeroed" with **zero edge effects** (this is why 24 was ruled out — non-integer). Whole-model uniform, `sparsegpt_recon`, exact rungs {5,10,20}%, 5% first (the only point where capability is preserved).

**Pre-registered outcomes:**
- **If coarse-granularity artifact:** as T shrinks, three findings move *together* — the ~5% ceiling rises, selection regains power over random, and magnitude<random vanishes — and all three crossovers coincide with the probe's correlation length S. → headline becomes *"redundancy is granularity-gated,"* and we ship the structured-pruning tax curve.
- **If real:** probe is diffuse and the downstream ladder stays flat across 32/16/8/4. → *"~5% is a real property of Qwen3-4B, not a tiling artifact"* — we ship that null.

**Staged delivery (what you get first):** the purity-probe verdict + a perplexity/KL screen at T16/T8 (T4 last) is a **complete first-pass answer**; the rigorous downstream ladder is launched and runs multi-day. One GPU job at a time, behind the recon refix.

**Later (noted, not run):** **non-square tile shapes** (1×N column strips) — the purity probe reports row-vs-column anisotropy, so it tells us *in advance* whether shape is worth chasing.

## Other pending items
- **Close the 2b cross-tile gap:** one tile-32 perplexity run on the exact gamed o_proj model, so "opposite directions in one network" is literally one model (the most attackable claim → cheapest fix).
- **Confirm Finding 6's N=24 optimum at tile-32** (currently tile-64 shape evidence).
- **Recommended-method capability data:** wanda_recon / random_recon downstream ladder (finalising at tile-32 now).
- **Second model (Llama-3.2-3B)** — the generality gate; every capability finding is n=1 model.
- **Iterative / sequential calibration** — the one shot at genuinely raising the ~5% ceiling.

---

# Where things live

- Active data: `experiments/{downstream, wholemodel, screen, screen_wholelayer, layer_budget}/`
- Archived (tile-64 shape evidence / closed threads): `experiments/archive/{full_scan, depth, oproj*, screen_cluster*}/`
- Figures: `figures/`
- Raw working log with full history: `FINDINGS.md`
