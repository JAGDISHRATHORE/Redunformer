# Tile-Level Redundancy in LLMs — Findings

Living log. Last updated: **2026-07-17** (after the overnight downstream + ablation marathon).
Numbers come from the runs in `experiments/`; plots in `experiments/*/plots/`.

## The answer, in one line

> **At 32×32 tiles, ~5% of Qwen3-4B is genuinely redundant** — removable while keeping 90% of
> the model's learned ability. Not the 40–70% our perplexity curves implied.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3-4B` (36 layers, bf16), RTX 4080 Super 16 GB |
| Tile size | 32 × 32 (never varied — see "next marathon") |
| Screened layers | 0, 9, 18, 27, **32, 33, 34**, 35 |
| Matrices | `q/k/v/o_proj`, `gate/up/down_proj` (7 per layer, 98,560 tiles each, 3,548,160 total) |
| Selection methods | random, magnitude, wanda, sparsegpt (eq-22/23) |
| Repair | SparseGPT reconstruction (eq-23), exact per output-row-block |
| Eval | WikiText-2 perplexity · output divergence (KL) · **lm-eval accuracy (HellaSwag/PIQA/ARC-Easy)** |
| Dense reference | ppl **13.22** · acc **0.6836 / 0.7492 / 0.7828** (reproduces published Qwen3-4B) |

**All methods are tile pruning.** Contrasts with "weight-level" refer to the *literature's*
unstructured setting, not our runs.

---

# Headline findings

## 1. Tile redundancy is ~5% — measured on real tasks ⭐
`sparsegpt_recon`, uniform. **Retained = (acc − chance) / (acc_dense − chance)** — the share of
*above-chance* ability kept. A broken model still scores chance by guessing, so raw accuracy
flatters it.

| sparsity | HellaSwag | PIQA | ARC-Easy | perplexity |
|---|---|---|---|---|
| 1% | 100% | 103% | 98% | — |
| 2% | 97% | 98% | 93% | — |
| **5%** | **90%** | **90%** | **90%** | 15.62 |
| 10% | 73% | 81% | 82% | 18.78 |
| 20% | 37% | 51% | 41% | 30.95 |
| 30% | 17% | 29% | 24% | 61.56 |

**5% is the sweet spot** (90% retained, strikingly consistent across all three tasks); 10% is the
aggressive edge (~79%); past that it collapses. This is the project's core answer — modest, but
measured and defensible.

## 2. Perplexity is a *nonlinear* proxy — and dangerous exactly where you'd rely on it ⭐
Same configs as above:

| perplexity | vs dense | ability retained |
|---|---|---|
| 15.62 | 1.18× | ~90% (proportionate) |
| 18.78 | 1.42× | ~79% |
| **30.95** | **2.34×** | **~43%** ← "only 2× worse" = more than half destroyed |
| 61.56 | 4.66× | ~20% |

Perplexity is monotonic with capability — it doesn't point the wrong way. The failure is that
the mapping is **brutally nonlinear**: a 2.3× perplexity rise *reads* as mild degradation and
*means* the model is mostly gone. It is a fine proxy in the usable regime (≤5%) and misleading
precisely where you would use it to judge an aggressive method.

## 3. Policy B games the metric it was built from ⭐⭐
Two separate results, both damning for reading perplexity as capability.

**(a) At high damage, a perplexity gain buys nothing.** Policy B vs A, same tile budget:

| sparsity | ppl gain | HellaSwag | PIQA | ARC-Easy |
|---|---|---|---|---|
| 20% | 1.31× | **+6pp** | +1pp | **+7pp** | ← real capability |
| **30%** | **1.58× (its peak)** | +3pp | **−1pp** | **0pp** | ← **nothing** |

Policy B's headline number — its 1.58× peak — **buys no capability at all**. Both models are
already near the guessing floor at 30%; improving a broken model's perplexity does not un-break
it. The point where a perplexity-optimizing method looks *most* impressive is the point where its
gain is most likely worthless. **We would have made that 1.58× the paper's headline.**

**(b) At *matched perplexity*, Policy B delivers 10–16pp LESS capability than uniform.**
Interpolating the recon+uniform curve to Policy B's exact perplexity:

| config | uniform at same ppl | Policy B | gap |
|---|---|---|---|
| p20 sens (ppl 23.6) | 56.5 / 67.3 / 63.3 | 43 / 52 / 48 | **−13.5 / −15.3 / −15.3 pp** |
| p30 sens (ppl 39.0) | 30.2 / 43.6 / 35.3 | 20 / 28 / 24 | **−10.2 / −15.6 / −11.3 pp** |

**6/6 measurements, same direction, large.** At 20% Policy B improves perplexity 1.31×; an honest
gain that size implies HellaSwag 37→56.5. It delivers 43. **Policy B buys ~31% of the capability
its perplexity advertises.**

**Why — and it is slightly circular:** our sensitivity map was *derived from perplexity*. We
measured which matrices hurt perplexity, then built a policy protecting exactly those. Perplexity
is dominated by the final layers (they feed the LM head); downstream tasks depend on the whole
computation. So the policy protects what perplexity cares about and flatters the metric it was
fit to. **Policy B's advantage is partly an artifact of how its map was built.**

Both things remain true: at a fixed sparsity budget B *is* genuinely better (43 vs 37). Its
perplexity number simply overstates that by ~3×.

→ **Next marathon:** build the sensitivity map from *downstream accuracy* instead of perplexity
and re-run A/B. If the gap closes, this is confirmed as a metric artifact.

## 4. Repair is everything; *which tiles you pick* barely matters ⭐
Controlled ablation — the repair path is **bit-identical** between `wanda_recon` and
`sparsegpt_recon` (CPU-verified), so selection is the only variable:

| sparsity | wanda | **wanda_recon** | sparsegpt_recon | repair adds | eq-23 selection adds |
|---|---|---|---|---|---|
| 5% | 28.31 | **15.59** | 15.62 | 1.82× | 1.00× |
| 10% | 52.22 | 19.81 | **18.78** | 2.64× | 1.05× |
| 20% | 156.51 | **25.95** | 30.95 | 6.03× | 0.84× |
| 30% | 1,145 | **38.45** | 61.56 | 29.78× | 0.62× |
| 40% | 7,812 | **74.42** | 76.30 | **104.97×** | 0.98× |

- **Repair dominates, monotonically**: 1.82× → **104.97×** as damage grows.
- **Selection is ~neutral**: 0.62–1.05×, no trend.
- **`wanda_recon` wins 4/5 sparsities** and is *cheaper* (no per-tile Schur complements).

**This corrects our own headline.** "SparseGPT reconstruction is best" was never about SparseGPT's
*selection* — it was about repair. **Recommended method: `wanda_recon`** — not because Wanda picks
better, but because selection barely matters and Wanda's is cheap.

Capability confirms it at 20%: `wanda` retains 14/12/16%, `sparsegpt_recon` 37/51/41% — repair is
worth ~3× in real ability, not just perplexity.

## 5. "Robust" is measured in isolation and does not compose ⭐
Per our labels, **~81% of the model is individually "robust"** (ΔPPL ≈ 0 when pruned alone). Prune
all of it together at 20% and ~60% of the model's ability is gone.

> **The redundancy map measures MARGINAL damage. We were reading it as JOINT damage.**

A matrix being harmless alone says nothing about it being harmless when the other 204 are also
pruned. This single confusion explains the whole-model collapse, the capability numbers, *and*
why Policy B only buys 1.1–1.6× — you cannot fix a compounding problem by reallocating budget
among regions that are all mislabelled the same way. It is the empirical, quantified form of the
"sequential dependency" limitation Rathore flagged in §4.8.

## 6. Reallocation has an optimum — and Policy B + depth are ONE mechanism ⭐
Depth concentration, **identical 20% budget** (every layer holds the same tile count, so
N × local = 36 × 0.20 = 7.2 layer-equivalents is exact arithmetic):

| layers | local | perplexity | vs uniform-36 (30.95) |
|---|---|---|---|
| 32 | 22.5% | 24.94 | 1.24× |
| **24** | **30%** | **24.17** | **1.28× ← peak** |
| 16 | 45% | 30.42 | 1.02× |
| 12 | 60% | 54.51 | 0.57× |
| **8** | **90%** | **1,890** | **0.02× (61× worse)** |

A clean inverted-U. **Two competing forces explain both this and Policy B:**

- damage **compounds across layers** → concentrate
- damage is **super-linear within a layer** past ~30–45% → spread

The optimum balances them. Policy B reverses at high sparsity for exactly the same reason: it
pushes "robust" regions to 67% local, straight into the super-linear regime. **Two findings that
looked independent are one mechanism.**

Note: N=32 (prune only the robust zone, sensitive tail untouched) gives 1.24× — "prune only the
redundant part" works, but modestly. 24.17 is still ~1.8× dense: concentration optimizes *within*
the broken regime, it does not unlock a new one.

## 7. The bottleneck is layer 35 specifically — not "the final layers"
`up_proj` ΔPPL @20% (newly measured 32/33/34):

| layer | 32 | 33 | 34 | 35 |
|---|---|---|---|---|
| ΔPPL | 0.38 | 0.19 | 1.01 | **3.37** |

**Layers 32–33 are robust; 34 ramps; 35 spikes.** Our policy stamped layer 35's labels onto 32–35
by nearest-neighbour, so **two of three were mislabelled** — the truly sensitive region is ~2–4%
of the model, not the 8% we assumed. Attention at these layers is unremarkable; the effect is
MLP-selective, which is also why it isn't a bug (verified: identical `num_pruned`, reproduces at
the last layer of Qwen3-0.6B).

**Mechanism:** layer 35 feeds the LM head with no downstream layer left to absorb its error.

**Consequence: Policy B is handicapped by its own labels** — it spends 0.30× multipliers
protecting layers 32–33, which never needed it. A corrected Policy B should beat everything
measured here. (Top of the next-marathon list.)

## 8. Magnitude is worse than random — and we know why
Median ΔPPL over 35 layer×matrix cells (screening subset, dense 13.559):

| method | 10% | 20% | 40% |
|---|---|---|---|
| magnitude | 0.072 | 0.123 | 0.301 |
| random *(floor)* | 0.038 | 0.083 | 0.241 |
| wanda | 0.030 | 0.035 | 0.123 |
| sparsegpt | 0.026 | 0.063 | 0.115 |
| **sparsegpt_recon** | **0.006** | **0.016** | **0.050** |

Magnitude sits **above the random floor** — worse than chance. Replicated independently on two
model sizes (0.6B random 144-52, 4B random 63-42). `o_proj` is the reproducible exception
(magnitude wins in *both* models: 4B 10-5, 0.6B 21-7).

**Why: aggregation loss.** How you turn a per-weight score into a tile score matters more than
which score you start from — magnitude (weights → block norm) is worst, wanda (weights → tile
mean) better, sparsegpt (measures the tile's actual output effect) is tile-native. This
empirically confirms Rathore's §3.8 prediction ("aggregation may hide important weights").

## 9. Damage is sub-additive within a layer, compounding across layers
Whole-layer ΔPPL is only **0.69–0.81×** the sum of its 7 individual matrices — degradations
overlap. Across layers it compounds, which is why uniform whole-model pruning collapses.

---

# Caveats
- **One model** (Qwen3-4B), **one tile size** (32×32), **one calibration seed**.
- Downstream measured for `sparsegpt_recon` (+ `wanda` @20%). **`wanda_recon` — our new
  recommended method — has no capability data yet.**
- Screening uses a 20% eval subset (5.1× faster); whole-model/downstream use full eval.
- Layers 1–8, 10–17, 19–26, 28–31 still inherit labels from their nearest measured neighbour.
- **One-shot calibration**: Gram matrices come from the dense model per layer, never re-derived
  after upstream pruning (Rathore §4.8).
- Comparisons above ppl ~1000 are meaningless (both models gibberish) and are excluded, not
  reported — this caught a fake "694× win" for Policy B at 70%.
- All methods deterministic (bit-identical reruns); only `random` uses seeds.

# Next testing marathon — flagged experiments
Ordered by expected value.

1. **Capability-derived sensitivity map** ⭐ — rebuild the map from *downstream accuracy* rather
   than perplexity, then re-run A/B. Finding #3(b) shows Policy B flatters the metric it was fit
   to; this tests that directly and, if it works, produces a policy that optimizes the thing we
   actually care about. **Highest value, and the most interesting.**
2. **Corrected Policy B** — relabel using the now-measured layers 32/33/34 and re-run the A/B
   sweep. It has been protecting robust layers (32–33) for free. Cheap, and should beat every
   policy result we have. Combines naturally with (1).
3. **Tile size 8 / 16 vs 32** ⭐ — the big one. Coarse structure is the prime suspect for why only
   ~5% is redundant. Finding #1 turns this from a nice-to-have into a sharp hypothesis: if
   redundancy is real but the 32×32 grid is too coarse to find it, 8×8 should recover
   substantially more. Needs a re-screen at the new tile sizes.
4. **`wanda_recon` downstream** — our recommended method, no capability data. Run the same
   1/2/5/10/20% ladder to confirm the perplexity win is real ability. Note finding #3(b): a
   perplexity win is not evidence of a capability win until measured.
5. **Depth optimum at *usable* sparsity** — the inverted-U (peak N=24) was measured at a 20%
   budget, i.e. inside the broken regime. Redo at 5%, where the model still works, and the
   optimum may sit elsewhere.
6. **Iterative / sequential calibration** — re-derive H after upstream layers are pruned. Directly
   attacks finding #5 (marginal ≠ joint) and Rathore §4.8. The most scientifically interesting
   of these.
7. **Calibration-seed robustness** — never tested; cheap insurance against a seed artifact.
8. **Second model** (Qwen3-0.6B) for the downstream story — the perplexity findings replicate
   across sizes; the capability findings are single-model.
9. **Fill the depth profile** — measure the remaining unmeasured layers, or at least confirm the
   nearest-neighbour assumption holds somewhere in the middle.

# Where things live
```
experiments/screen/              per-matrix screening + baselines (+ plots/)
experiments/screen_wholelayer/   whole-layer (+ plots/)
experiments/wholemodel/          whole-model: uniform, Policy B, wanda_recon (+ plots/)
experiments/depth/               depth concentration (+ plots/)
experiments/downstream/          lm-eval accuracy (+ plots/)
docs/metrics.md                  every metric, and why
scripts/plot_{screening,wholelayer,wholemodel,policy,depth,downstream}.py
src/redundancy/{scoring,recovery,combined,policy,hooks,eval}.py
```
