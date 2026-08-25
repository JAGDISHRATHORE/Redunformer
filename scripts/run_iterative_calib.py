"""
Iterative / sequential calibration (next-step #2) -- dose-split whole-model pruning.

One-shot `sparsegpt_recon` prunes each matrix to the FINAL ratio in a single pass,
repairing survivors by exact least-squares against a Gram collected on the model
as-it-then-stands (earlier layers already pruned). This module splits the target
sparsity into K increments and re-runs selection + repair each increment, with the
calibration Gram re-collected on the progressively-pruned model.

Scheme (repair-from-original, exact per mask)
---------------------------------------------
  increments r_1 < ... < r_K = target.  Persistent pruned-tile mask per (layer,matrix).
  For each increment, for each layer in order:
    - collect a fresh Gram H on the CURRENT (progressively-pruned) model,
    - score every tile of the ORIGINAL weight by the eq-23 reconstructed error
      under H, and grow the mask to int(r_k * n_tiles) by adding the lowest-error
      not-yet-pruned tiles,
    - set weight = exact-LS reconstruction of the ORIGINAL weight for the FULL
      cumulative mask under H  (redundancy.combined.reconstruct_given_tiles).

K=1 reproduces the existing one-shot whole-model `sparsegpt_recon` exactly (same
selection, same exact-per-mask repair) -- the validation anchor and the baseline
to beat.

Expected effect (stated honestly): the one-shot path ALREADY recalibrates per
layer on the progressively-pruned model, and the repair is already exact for its
mask, so LS-only iteration can only improve tile SELECTION and the Gram informing
repair -- it adds no gradient information. A large ceiling gain would instead need
light fine-tuning between increments. This experiment measures the LS-only ceiling.

CPU self-test (--selftest) validates the increment bookkeeping and the
repair-from-original mechanics on synthetic data; it does not touch the GPU.

Example
  python scripts/run_iterative_calib.py --prune-ratio 0.20 --steps 4 \
    --output experiments/iterative/downstream_iter_sgr_K4_p20.json
"""

import argparse
import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # scripts/
sys.path.insert(0, os.path.join(HERE, "..", "src"))        # src/

import numpy as np


# ----------------------------------------------------------------------------
# Increment bookkeeping (pure -- unit-tested)
# ----------------------------------------------------------------------------

def cumulative_targets(n_tiles, target_ratio, steps):
    """int() tile counts at each increment r_k = target*k/steps, k=1..steps.
    Uses int() (floor) so the final count == one-shot int(n_tiles*target)."""
    return [int(n_tiles * target_ratio * k / steps) for k in range(1, steps + 1)]


def incremental_select(scored, mask_set, target_count):
    """Pick the lowest-error tiles NOT already in mask_set to bring the mask up to
    target_count. scored: iterable of (err, r, c). Returns the list of new (r,c)."""
    need = target_count - len(mask_set)
    if need <= 0:
        return []
    out = []
    for err, r, c in sorted(scored, key=lambda x: x[0]):
        if (r, c) in mask_set:
            continue
        out.append((r, c))
        if len(out) == need:
            break
    return out


# ----------------------------------------------------------------------------
# Iterative whole-model pruning
# ----------------------------------------------------------------------------

def prune_whole_model_iterative(model, layers, calib_samples, target_ratio, steps,
                                tile_size, matrices, damp=1e-2, verbose=True):
    """Dose-split whole-model pruning. Returns (total_tiles, total_pruned, info)."""
    import torch
    from run_pruning import (build_target_name, get_target_weight,
                             collect_stats_for_targets, get_full_tiles)
    from redundancy.scoring import sparsegpt_reconstructed_errors
    from redundancy.combined import reconstruct_given_tiles

    dev = next(model.parameters()).device

    # Persist ORIGINAL weights (CPU) so every increment repairs from the same base;
    # the pruned model lives in the live weights. n_tiles is fixed per matrix.
    originals, n_tiles = {}, {}
    for L in layers:
        for m in matrices:
            name = build_target_name(L, m)
            w = get_target_weight(model, name)
            originals[(L, m)] = w.detach().to("cpu", copy=True)
            n_tiles[(L, m)] = len(get_full_tiles(w, tile_size))

    mask = {(L, m): set() for L in layers for m in matrices}
    grams_collected = 0

    for k in range(1, steps + 1):
        rk = target_ratio * k / steps
        if verbose:
            print(f"\n=== increment {k}/{steps}  cumulative ratio {rk:.4f} ===")
        for L in layers:
            names = [build_target_name(L, m) for m in matrices]
            # Fresh Gram on the CURRENT model (earlier layers already at r_k this pass).
            stats = collect_stats_for_targets(model, calib_samples, names, "sparsegpt_recon")
            grams_collected += 1
            for m in matrices:
                name = build_target_name(L, m)
                w = get_target_weight(model, name)
                H = stats[name]
                orig = originals[(L, m)].to(dev)
                tgt = int(n_tiles[(L, m)] * rk)

                # grow mask using eq-23 scores of the ORIGINAL weight under fresh H
                scored = sparsegpt_reconstructed_errors(orig, H, tile_size, damp=damp)
                add = incremental_select(scored, mask[(L, m)], tgt)
                mask[(L, m)].update(add)

                # exact-LS reconstruction of the ORIGINAL weight for the full mask
                with torch.no_grad():
                    w.copy_(orig)
                if mask[(L, m)]:
                    reconstruct_given_tiles(w, H, sorted(mask[(L, m)]), tile_size, damp=damp)
                del orig
            del stats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if verbose:
                pruned_so_far = sum(len(mask[(L, mm)]) for mm in matrices)
                print(f"  layer {L:2}: mask {pruned_so_far} tiles")

    total_tiles = sum(n_tiles.values())
    total_pruned = sum(len(s) for s in mask.values())
    del originals
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    info = {"steps": steps, "grams_collected": grams_collected,
            "increments": [target_ratio * k / steps for k in range(1, steps + 1)]}
    return total_tiles, total_pruned, info


# ----------------------------------------------------------------------------
# Main: prune iteratively, then downstream (or ppl) eval
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Iterative (dose-split) whole-model sparsegpt_recon.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--prune-ratio", type=float, default=0.20)
    p.add_argument("--steps", type=int, default=4, help="K dose increments (K=1 == one-shot).")
    p.add_argument("--tile-size", type=int, default=32)
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--matrices", type=str, nargs="+", default=None)
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--tasks", nargs="+", default=["hellaswag", "piqa", "arc_easy"])
    p.add_argument("--batch-size", default="8")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eval-ppl", action="store_true",
                   help="WikiText-2 perplexity instead of downstream (fast integration check).")
    p.add_argument("--eval-frac", type=float, default=1.0)
    p.add_argument("--output", default=None)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    import torch
    from run_pruning import MATRICES, get_num_layers, save_json
    from redundancy.models import load_model_and_tokenizer
    from redundancy.data import load_calibration_dataset

    matrices = args.matrices or list(MATRICES.keys())
    model, tokenizer = load_model_and_tokenizer(args.model)
    layers = args.layers if args.layers is not None else list(range(get_num_layers(model)))
    calib = load_calibration_dataset(tokenizer, n_samples=args.calib_samples, seqlen=args.calib_seqlen)

    print(f"Iterative prune: target {args.prune_ratio}  K={args.steps}  over {len(layers)} layers")
    tt, tp, info = prune_whole_model_iterative(
        model, layers, calib, args.prune_ratio, args.steps, args.tile_size, matrices)
    print(f"Pruned {tp}/{tt} = {tp/tt:.4f}   (grams collected: {info['grams_collected']})")

    del calib
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tag = f"iter_sgr_K{args.steps}_p{int(args.prune_ratio*100)}"
    out = {"tag": tag, "model": args.model, "method": "iterative_sparsegpt_recon",
           "prune_ratio": args.prune_ratio, "steps": args.steps, "tile_size": args.tile_size,
           "achieved_sparsity": tp / tt, "iter_info": info, "limit": args.limit}

    if args.eval_ppl:
        from redundancy.data import load_evaluation_dataset
        from redundancy.eval import evaluate_perplexity
        ds = load_evaluation_dataset()
        ppl = evaluate_perplexity(model, tokenizer, ds, eval_frac=args.eval_frac)
        out["perplexity"] = ppl
        path = args.output or os.path.join("experiments", "iterative", f"{tag}_ppl.json")
        save_json(path, out)
        print(f"WikiText ppl: {ppl:.4f}")
    else:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
        res = lm_eval.simple_evaluate(model=lm, tasks=args.tasks, limit=args.limit)
        out["results"] = res["results"]
        path = args.output or os.path.join("experiments", "iterative", f"downstream_{tag}.json")
        save_json(path, out)
        print("\n=== downstream (iterative) ===")
        for t, v in res["results"].items():
            print(f"  {t:12} {v.get('acc_norm,none', v.get('acc,none'))}")


# ----------------------------------------------------------------------------
# Self-test (CPU): increment bookkeeping + repair-from-original mechanics
# ----------------------------------------------------------------------------

def selftest():
    import torch
    from redundancy.combined import reconstruct_given_tiles
    from redundancy.scoring import sparsegpt_reconstructed_errors
    print("=" * 72)
    print("  SELF-TEST: iterative increment bookkeeping + repair-from-original (CPU)")
    print("=" * 72)
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        if not cond:
            fails += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    # ---- cumulative targets: monotonic, final == one-shot int() ----------
    n = 100
    ct = cumulative_targets(n, 0.20, 4)
    check("cumulative targets monotonic non-decreasing", all(ct[i] <= ct[i+1] for i in range(len(ct)-1)),
          f"{ct}")
    check("final increment == one-shot int(n*ratio)", ct[-1] == int(n * 0.20), f"{ct[-1]} vs {int(n*0.2)}")

    # ---- incremental_select: exact count, no dupes, lowest-error survivors ----
    scored = [(float(i), i, 0) for i in range(20)]        # err == row index; tile (i,0)
    mask = set()
    add1 = incremental_select(scored, mask, 5); mask.update(add1)
    check("step1 picks the 5 lowest-error tiles", set(add1) == {(i, 0) for i in range(5)}, f"{sorted(add1)}")
    add2 = incremental_select(scored, mask, 8); mask.update(add2)
    check("step2 adds exactly the next 3 (no re-pick)", set(add2) == {(5, 0), (6, 0), (7, 0)}, f"{sorted(add2)}")
    check("mask exact size after two steps", len(mask) == 8)
    check("no duplicates across steps", len(add1) + len(add2) == len(set(add1) | set(add2)))
    check("shrink request adds nothing", incremental_select(scored, mask, 3) == [])

    # ---- repair-from-original: masked tiles zeroed; path-independent ----
    torch.manual_seed(0)
    ts = 4
    rows, cols = 8, 12
    W = torch.randn(rows, cols).double()
    A = torch.randn(cols, cols).double()
    cov = A @ A.t() / cols + 0.1 * torch.eye(cols).double()
    X = torch.randn(400, cols).double() @ torch.linalg.cholesky(cov).t()
    H = X.t() @ X

    def recon_err(Wc):
        return ((X @ W.t() - X @ Wc.t()) ** 2).sum().item()

    # choose a couple of tiles as the final mask
    final_mask = [(0, 0), (4, 4), (0, 8)]
    # one-shot: repair original with full mask
    W_oneshot = W.clone()
    reconstruct_given_tiles(W_oneshot, H, final_mask, ts, damp=1e-2)
    # iterative: grow the SAME mask in two steps, each repairing from ORIGINAL
    W_iter = W.clone()
    reconstruct_given_tiles(W_iter, H, final_mask[:1], ts, damp=1e-2)      # step1 (discarded state)
    W_iter.copy_(W)                                                         # <- repair-from-ORIGINAL
    reconstruct_given_tiles(W_iter, H, final_mask, ts, damp=1e-2)           # step2 full mask
    rel = (W_oneshot - W_iter).norm().item() / (W_oneshot.norm().item() + 1e-12)
    check("repair-from-original is path-independent (iter == one-shot for same final mask)",
          rel < 1e-9, f"rel_diff={rel:.2e}")

    # masked tiles are zeroed (to the float32 precision of the production repair path,
    # which computes the correction in fp32 -- residual ~1e-6, not a hard zero; cf.
    # run_pruning.extract_pruned_mask using rel_tol=0.1 for exactly this reason).
    zeroed_ok = all(
        W_oneshot[r:r+ts, c:c+ts].abs().max().item() < 1e-4 * W[r:r+ts, c:c+ts].abs().max().item()
        for r, c in final_mask)
    check("masked tiles zeroed to fp32 precision (<1e-4 x original)", zeroed_ok)

    # repair reduces output error vs mask-only
    W_maskonly = W.clone()
    for r, c in final_mask:
        W_maskonly[r:r+ts, c:c+ts] = 0.0
    check("repair reduces reconstruction error vs mask-only",
          recon_err(W_oneshot) < recon_err(W_maskonly),
          f"repaired={recon_err(W_oneshot):.3e} mask-only={recon_err(W_maskonly):.3e}")

    # ---- K=1 selection == one-shot selection (lowest-error int(n*r) tiles) ----
    scored_real = sparsegpt_reconstructed_errors(W, H, ts, damp=1e-2)
    n_tiles = len(scored_real)
    tgt = int(n_tiles * 0.5)
    k1 = set(incremental_select(scored_real, set(), tgt))
    oneshot = set((r, c) for _, r, c in sorted(scored_real, key=lambda x: x[0])[:tgt])
    check("K=1 mask == one-shot lowest-error selection", k1 == oneshot, f"|k1|={len(k1)} |1shot|={len(oneshot)}")

    print("=" * 72)
    print(f"  SELF-TEST RESULT: {'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
    print("=" * 72)
    return fails == 0


if __name__ == "__main__":
    main()
