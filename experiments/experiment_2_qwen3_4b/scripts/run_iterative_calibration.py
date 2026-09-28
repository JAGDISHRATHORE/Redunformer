import argparse
import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np


def step_targets(n_tiles, target, steps):
    return [int(n_tiles * target * k / steps) for k in range(1, steps + 1)]


def select_more(scored, mask_set, n_target):
    need = n_target - len(mask_set)
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


def prune_iterative(model, layers, calib, target, steps,
                    tile_size, matrices, damp=1e-2, verbose=True):
    import torch
    from run_tile_pruning import (param_name, get_weight,
                                  calib_stats, full_tiles)
    from redundancy.scoring import sgpt_recon_err
    from redundancy.combined import repair_tiles

    dev = next(model.parameters()).device

    originals, n_tiles = {}, {}
    for L in layers:
        for m in matrices:
            name = param_name(L, m)
            w = get_weight(model, name)
            originals[(L, m)] = w.detach().to("cpu", copy=True)
            n_tiles[(L, m)] = len(full_tiles(w, tile_size))

    mask = {(L, m): set() for L in layers for m in matrices}
    n_grams = 0

    for k in range(1, steps + 1):
        rk = target * k / steps
        if verbose:
            print(f"\n=== increment {k}/{steps}  cumulative ratio {rk:.4f} ===")
        for L in layers:
            names = [param_name(L, m) for m in matrices]
            stats = calib_stats(model, calib, names, "sparsegpt_recon")
            n_grams += 1
            for m in matrices:
                name = param_name(L, m)
                w = get_weight(model, name)
                H = stats[name]
                orig = originals[(L, m)].to(dev)
                tgt = int(n_tiles[(L, m)] * rk)

                scored = sgpt_recon_err(orig, H, tile_size, damp=damp)
                add = select_more(scored, mask[(L, m)], tgt)
                mask[(L, m)].update(add)

                with torch.no_grad():
                    w.copy_(orig)
                if mask[(L, m)]:
                    repair_tiles(w, H, sorted(mask[(L, m)]), tile_size, damp=damp)
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
    info = {"steps": steps, "grams_collected": n_grams,
            "increments": [target * k / steps for k in range(1, steps + 1)]}
    return total_tiles, total_pruned, info


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
        sys.exit(0 if self_test() else 1)

    import torch
    from run_tile_pruning import PROJS, count_layers, save_json
    from redundancy.models import load_model
    from redundancy.data import get_calib_data

    matrices = args.matrices or list(PROJS.keys())
    model, tokenizer = load_model(args.model)
    layers = args.layers if args.layers is not None else list(range(count_layers(model)))
    calib = get_calib_data(tokenizer, n_calib=args.calib_samples, seqlen=args.calib_seqlen)

    print(f"Iterative prune: target {args.prune_ratio}  K={args.steps}  over {len(layers)} layers")
    tt, tp, info = prune_iterative(
        model, layers, calib, args.prune_ratio, args.steps, args.tile_size, matrices)
    print(f"Pruned {tp}/{tt} = {tp/tt:.4f}   (grams collected: {info['n_grams']})")

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
        from redundancy.data import load_wikitext
        from redundancy.eval import eval_ppl
        ds = load_wikitext()
        ppl = eval_ppl(model, tokenizer, ds, eval_frac=args.eval_frac)
        out["perplexity"] = ppl
        path = args.output or os.path.join("experiments/experiment_2_qwen3_4b", "results", "09_iterative_calibration", f"{tag}_ppl.json")
        save_json(path, out)
        print(f"WikiText ppl: {ppl:.4f}")
    else:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
        res = lm_eval.simple_evaluate(model=lm, tasks=args.tasks, limit=args.limit)
        out["results"] = res["results"]
        path = args.output or os.path.join("experiments/experiment_2_qwen3_4b", "results", "09_iterative_calibration", f"downstream_{tag}.json")
        save_json(path, out)
        print("\n=== downstream (iterative) ===")
        for t, v in res["results"].items():
            print(f"  {t:12} {v.get('acc_norm,none', v.get('acc,none'))}")


def self_test():
    import torch
    from redundancy.combined import repair_tiles
    from redundancy.scoring import sgpt_recon_err
    print("=" * 72)
    print("  SELF-TEST: iterative increment bookkeeping + repair-from-original (CPU)")
    print("=" * 72)
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        if not cond:
            fails += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    n = 100
    ct = step_targets(n, 0.20, 4)
    check("cumulative targets monotonic non-decreasing", all(ct[i] <= ct[i+1] for i in range(len(ct)-1)),
          f"{ct}")
    check("final increment == one-shot int(n*ratio)", ct[-1] == int(n * 0.20), f"{ct[-1]} vs {int(n*0.2)}")

    scored = [(float(i), i, 0) for i in range(20)]
    mask = set()
    add1 = select_more(scored, mask, 5); mask.update(add1)
    check("step1 picks the 5 lowest-error tiles", set(add1) == {(i, 0) for i in range(5)}, f"{sorted(add1)}")
    add2 = select_more(scored, mask, 8); mask.update(add2)
    check("step2 adds exactly the next 3 (no re-pick)", set(add2) == {(5, 0), (6, 0), (7, 0)}, f"{sorted(add2)}")
    check("mask exact size after two steps", len(mask) == 8)
    check("no duplicates across steps", len(add1) + len(add2) == len(set(add1) | set(add2)))
    check("shrink request adds nothing", select_more(scored, mask, 3) == [])

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

    final_mask = [(0, 0), (4, 4), (0, 8)]
    W_oneshot = W.clone()
    repair_tiles(W_oneshot, H, final_mask, ts, damp=1e-2)
    W_iter = W.clone()
    repair_tiles(W_iter, H, final_mask[:1], ts, damp=1e-2)
    W_iter.copy_(W)
    repair_tiles(W_iter, H, final_mask, ts, damp=1e-2)
    rel = (W_oneshot - W_iter).norm().item() / (W_oneshot.norm().item() + 1e-12)
    check("repair-from-original is path-independent (iter == one-shot for same final mask)",
          rel < 1e-9, f"rel_diff={rel:.2e}")

    zeroed_ok = all(
        W_oneshot[r:r+ts, c:c+ts].abs().max().item() < 1e-4 * W[r:r+ts, c:c+ts].abs().max().item()
        for r, c in final_mask)
    check("masked tiles zeroed to fp32 precision (<1e-4 x original)", zeroed_ok)

    W_mask = W.clone()
    for r, c in final_mask:
        W_mask[r:r+ts, c:c+ts] = 0.0
    check("repair reduces reconstruction error vs mask-only",
          recon_err(W_oneshot) < recon_err(W_mask),
          f"repaired={recon_err(W_oneshot):.3e} mask-only={recon_err(W_mask):.3e}")

    scored_real = sgpt_recon_err(W, H, ts, damp=1e-2)
    n_tiles = len(scored_real)
    tgt = int(n_tiles * 0.5)
    k1 = set(select_more(scored_real, set(), tgt))
    oneshot = set((r, c) for _, r, c in sorted(scored_real, key=lambda x: x[0])[:tgt])
    check("K=1 mask == one-shot lowest-error selection", k1 == oneshot, f"|k1|={len(k1)} |1shot|={len(oneshot)}")

    print("=" * 72)
    print(f"  SELF-TEST RESULT: {'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
    print("=" * 72)
    return fails == 0


if __name__ == "__main__":
    main()
