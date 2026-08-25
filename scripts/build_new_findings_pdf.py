"""Build a standalone PDF addendum for the new experiments (#1-#4).

Separate from FINDINGS_clean (that doc is untouched). ASCII-only text so the
fpdf2 core fonts render cleanly. Reads nothing but embeds the already-generated
result figures. Output: FINDINGS_new_experiments.pdf in the repo root.
"""
import os
from fpdf import FPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = {
    "f4":  "experiments/downstream/plots/f_repair_downstream_ladder.png",
    "f2":  "experiments/iterative/plots/iter_vs_oneshot.png",
    "f3":  "experiments/tilesize/probe_strips/harvestable_by_orientation.png",
    "f1a": "experiments/llama/plots/f1_qwen_vs_llama.png",
    "f1b": "experiments/llama/plots/f10_llama_1x1_vs_32.png",
}

INK = (33, 37, 41)
MUTE = (110, 116, 122)
ACC = (61, 137, 109)


class PDF(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTE)
        self.cell(0, 8, f"New-experiments addendum  -  draft for review  -  page {self.page_no()}",
                  align="C")


pdf = PDF(format="A4")
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(20, 18, 20)
CW = 170  # content width mm


def h1(t):
    pdf.ln(2); pdf.set_font("Helvetica", "B", 14); pdf.set_text_color(*INK)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(CW, 7, t, new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)


def h2(t):
    pdf.ln(1); pdf.set_font("Helvetica", "B", 10.5); pdf.set_text_color(*ACC)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(CW, 5.5, t, new_x="LMARGIN", new_y="NEXT")


def body(t, style="", size=9.7, color=INK):
    pdf.set_font("Helvetica", style, size); pdf.set_text_color(*color)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(CW, 5.0, t, new_x="LMARGIN", new_y="NEXT"); pdf.ln(0.6)


def bullets(items):
    pdf.set_font("Helvetica", "", 9.7); pdf.set_text_color(*INK)
    for it in items:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(CW, 5.0, "-  " + it, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.6)


def table(headers, rows, widths):
    pdf.set_font("Helvetica", "B", 8.8); pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(*ACC)
    pdf.set_x(pdf.l_margin)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6.2, h, border=0, align="C", fill=True, new_x="RIGHT", new_y="TOP")
    pdf.ln(6.2)
    pdf.set_font("Helvetica", "", 8.8); pdf.set_text_color(*INK)
    for i, row in enumerate(rows):
        pdf.set_fill_color(*( (245, 247, 248) if i % 2 == 0 else (255, 255, 255)))
        pdf.set_x(pdf.l_margin)
        for c, w in zip(row, widths):
            al = "L" if c is row[0] else "C"
            pdf.cell(w, 5.8, str(c), border=0, align=al, fill=True, new_x="RIGHT", new_y="TOP")
        pdf.ln(5.8)
    pdf.ln(2)


def figure(key, w=132):
    p = os.path.join(ROOT, FIG[key])
    if not os.path.exists(p):
        return
    if pdf.get_y() + w * 0.68 > 275:
        pdf.add_page()
    x = 20 + (CW - w) / 2
    pdf.image(p, x=x, w=w)
    pdf.ln(2)


# ---------------------------------------------------------------- title page
pdf.add_page()
pdf.ln(6)
pdf.set_font("Helvetica", "B", 19); pdf.set_text_color(*INK)
pdf.multi_cell(CW, 9, "New Experiments - Findings Addendum", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 11); pdf.set_text_color(*MUTE)
pdf.multi_cell(CW, 6, "Tile-level redundancy in LLMs. Qwen3-4B (instruct) and a second model, "
                      "Llama-3.2-3B-Instruct. Tile size 32x32. Downstream measured on "
                      "HellaSwag / PIQA / ARC-Easy; retained = (acc - chance) / (acc_dense - chance).",
                      new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
body("Standalone addendum to FINDINGS_clean, which is left unchanged. Covers four experiments run "
     "to close the report's open next-steps: (A) the recommended-method downstream ladder, "
     "(B) iterative calibration, (C) non-square strips, and (D) the second-model generality gate. "
     "Draft for review; numbers are first-pass where noted.", size=9.7)
pdf.ln(1)

# ---------------------------------------------------------------- Finding A
h1("A. Recommended-method downstream ladder: perplexity inverts real capability")
h2("What we test")
body("Whether the near-dense perplexity ordering among the three repair variants "
     "(random_recon, wanda_recon, sparsegpt_recon) is mirrored by real downstream capability, "
     "across the capability-relevant band (5 / 10 / 20 percent structured sparsity).")
h2("Why we ran it")
body("The report noted this downstream data was unfinished. If methods that all look near-dense "
     "on perplexity disagree on real tasks, ranking repair methods by perplexity is unsafe.")
h2("Result")
body("They disagree - perplexity inverts capability. random_recon has the best (lowest) whole-model "
     "perplexity at every dose, but the worst retained ability at every dose. And given repair, "
     "calibrated selection beats random selection by a margin that GROWS with sparsity.")
table(["variant (repair)", "p5", "p10", "p20", "whole-model ppl"],
      [["random_recon", "85", "67", "29", "14.6 / 16.5 / 24.4"],
       ["wanda_recon", "91", "75", "52", "15.4 / 18.4 / 24.0"],
       ["sparsegpt_recon", "90", "79", "43", "15.6 / 18.8 / 30.9"]],
      [46, 20, 20, 20, 64])
figure("f4")
h2("Reliability")
body("Downstream-anchored (full task sets). random_recon is 5 seeds at p5, single median seed at "
     "p10/p20; the calibrated variants are deterministic. This QUALIFIES Finding 4's 'selection "
     "contributes nothing' - that claim is perplexity-scoped; on real capability, selection matters, "
     "and the gap widens from ~6 pp at p5 to ~14-23 pp at p20.")

# ---------------------------------------------------------------- Finding B
pdf.add_page()
h1("B. Iterative (least-squares-only) calibration does not raise the ceiling")
h2("What we test")
body("Whether splitting the target sparsity into K increments and re-collecting the calibration "
     "signal on the progressively-pruned model between increments (iterative / sequential "
     "calibration) lifts the ~5 percent structured ceiling above one-shot pruning.")
h2("Result")
body("It does not. A K=1 run reproduces one-shot exactly (30.95 ppl at 20 percent, which validates "
     "the implementation). K=4 dose-splitting is statistically indistinguishable from one-shot:")
table(["dose", "iterative K=4", "one-shot", "delta"],
      [["p5", "90%", "90%", "0"], ["p10", "79%", "79%", "0"], ["p20", "42%", "43%", "-1"]],
      [30, 47, 47, 46])
figure("f2")
h2("Reliability")
body("Downstream-anchored; the K=1 == one-shot anchor confirms correctness. Mechanism: the one-shot "
     "pipeline ALREADY re-collects per-layer calibration on the progressively-pruned model, and the "
     "repair is already exact for its mask, so least-squares-only iteration adds no new information. "
     "Raising the ceiling would require gradient information (light fine-tuning between cuts) - a "
     "larger-scope direction. This closes the LS-only iterative fork with a clean negative result.")

# ---------------------------------------------------------------- Finding C
pdf.add_page()
h1("C. Non-square strips do not rescue redundancy; the '1xN' orientation is backwards")
h2("What we test")
body("Whether non-square 1xN (wide) or Nx1 (tall) weight strips harvest more of the diffuse "
     "redundancy than square tiles of equal area, and which orientation the row anisotropy favors.")
h2("Result")
body("Strips do not rescue usable redundancy. At the 5 percent redundant set (purity >= 0.9), the "
     "best strip harvests less than 0.5 percent of weights, versus ~5 percent at 1x1 and 0.00 percent "
     "at square 32x32 - the redundancy is diffuse even at the two-element scale. Of the two "
     "orientations, Nx1 (tall, spanning output neurons) marginally beats 1xN (wide) - the OPPOSITE of "
     "the '1xN row strips' the report proposed.")
table(["orientation", "best shape", "harvestable"],
      [["Nx1 (tall)", "2x1", "0.44%"], ["1xN (wide)", "1x2", "0.29%"], ["square", "32x32", "0.00%"]],
      [56, 57, 57])
figure("f3")
h2("Reliability")
body("Weight-space probe (mask-only ceiling), CPU-validated on planted structure. Caveat: this run's "
     "aggregate anisotropy came out near-isotropic (A ~ 1.05), not reproducing the original probe's "
     "gate_proj-driven A ~ 0.24; the harvestable measurement (the direct arbiter) is unaffected, but "
     "the anisotropy metric should be reconciled before it is cited. Practical verdict: do not pursue "
     "non-square strips.")

# ---------------------------------------------------------------- Finding D
pdf.add_page()
h1("D. Second model (Llama-3.2-3B-Instruct): what transfers and what does not")
h2("What we test")
body("Whether the capability-critical findings replicate on a second architecture. Model: "
     "Llama-3.2-3B-Instruct, matched to Qwen3-4B (also an instruct model), same tile-32 and same "
     "WikiText + HellaSwag/PIQA/ARC-Easy setup. Minimum set: F1 (structured ladder), F4 (repair vs "
     "selection), F10 (structured tax). Llama dense: acc 0.716/0.768/0.709, ppl 10.45 (Qwen 13.22).")
h2("Result - structured ladder (F1), retained ability")
table(["model", "p1", "p2", "p5", "p10", "p20", "p30"],
      [["Llama-3.2-3B-Inst", "98", "97", "89", "69", "20", "10"],
       ["Qwen3-4B", "100", "96", "90", "79", "43", "23"]],
      [50, 20, 20, 20, 20, 20, 20])
figure("f1a")

pdf.add_page()
h2("Transferable - core findings generalize across both architectures")
bullets([
    "The ~5 percent structured ceiling: Llama retains 89 percent at 5 percent vs Qwen 90 percent.",
    "Repair is the dominant lever: repair variants sit near dense on both models.",
    "The structured-pruning tax: Llama 1x1 keeps 86 percent at 50 percent while 32x32 is gone by 20 "
    "percent (1x1 ppl 10.6/11.0/12.0/14.7 at p20/30/40/50 vs 32x32 ppl 105 at p20) - even stronger "
    "than on Qwen.",
    "Calibrated selection beats random selection on downstream given repair (~5 pp on both): Llama "
    "wanda_recon 91 > sparsegpt_recon 89 > random_recon 86.",
])
figure("f1b")
h2("Not transferable - Qwen3-4B-specific (scope these to Qwen)")
bullets([
    "Magnitude pruning does NOT catastrophize Llama: 21.9 ppl at 5 percent (functional) vs Qwen 3449 "
    "(destroyed).",
    "Calibrated selection does NOT lose to random on Llama - it beats random even without repair "
    "(no-repair ppl at 5 percent: sparsegpt 14.8 < wanda 16.6 < random 17.5 < magnitude 21.9), the "
    "opposite of Qwen.",
    "The perplexity-inversion (Finding A) does NOT occur on Llama - random_recon is worst on BOTH "
    "perplexity and capability, so the two axes agree.",
    "Llama collapses more steeply above the ceiling: 20 percent retained at p20 vs Qwen's 43 percent.",
])
h2("Reliability")
body("Downstream-anchored on Llama; first-pass (3 seeds for F4, minimum doses). The DIVERGENT claims "
     "above deserve 2-3 more seeds before publication. Base-vs-instruct is matched (both instruct), so "
     "the divergences are an architecture-or-training-recipe effect, not a base/instruct artifact.")

# ---------------------------------------------------------------- closing
pdf.ln(2)
h1("Reading these together")
body("The robust, load-bearing claims - a ~5 percent structured ceiling, repair as the dominant "
     "lever, and a large but diffuse redundancy that only 1x1 can reach (the structured tax) - now "
     "hold across two architectures (n=2). Several of the more surprising Qwen behaviours - magnitude "
     "catastrophe, calibrated selection losing to a coin flip, and perplexity moving opposite to "
     "capability - are model-specific and do not generalize. Scoping them to Qwen makes the core "
     "contribution more credible, not less.")
h2("Open / not done")
bullets([
    "The FINDINGS_clean write-up itself (author's prose) - unchanged by this addendum.",
    "2-3 more seeds on Llama's divergent claims (magnitude, selection) before papering them.",
    "Reconcile the Finding C anisotropy metric.",
    "Optional generative / out-of-distribution validation (report next-step #5) - not run.",
])

out = os.path.join(ROOT, "FINDINGS_new_experiments.pdf")
pdf.output(out)
print("wrote", out)
