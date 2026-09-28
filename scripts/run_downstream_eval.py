import argparse
import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch

from run_tile_pruning import (
    PROJS, param_name, get_weight, do_prune,
    calib_stats, tile_counts, count_layers, save_json,
)
from redundancy.models import load_model
from redundancy.data import get_calib_data
from redundancy.policy import load_sens, classify, fill_all_layers, allocate_budget

TASKS = ["hellaswag", "piqa", "arc_easy"]
NEEDS_CALIB = ("wanda", "sparsegpt", "sparsegpt_recon", "wanda_recon", "random_recon")
METHODS = ["magnitude", "magnitude_high", "random", "wanda", "sparsegpt", "sparsegpt_recon",
           "wanda_recon", "random_recon"]


def prune_whole_model(model, args, layers, calib, ratio_map=None):
    projs = getattr(args, "matrices", None) or list(PROJS.keys())
    total_tiles = total_pruned = 0
    for layer in layers:
        stats = None
        if args.method in NEEDS_CALIB:
            names = [param_name(layer, m) for m in projs]
            stats = calib_stats(model, calib, names, args.method)

        for m in projs:
            target = param_name(layer, m)
            w = get_weight(model, target)
            stat = stats[target] if stats is not None else None
            ratio = ratio_map[(layer, m)] if ratio_map is not None else args.prune_ratio
            nt, npd = do_prune(w, args.method, args.tile_size, ratio, args.seed, stat=stat)
            total_tiles += nt
            total_pruned += npd

        del stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  layer {layer:2}: total pruned {total_pruned}")

    return total_tiles, total_pruned


def main():
    p = argparse.ArgumentParser(description="Downstream accuracy of a pruned whole model.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--dense", action="store_true", help="Evaluate the unpruned model (baseline).")
    p.add_argument("--method", default="sparsegpt_recon", choices=METHODS)
    p.add_argument("--prune-ratio", type=float, default=0.20)
    p.add_argument("--policy", choices=["uniform", "sensitivity"], default="uniform")
    p.add_argument("--tile-size", type=int, default=32)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Restrict pruning to these layers (default: all).")
    p.add_argument("--matrices", type=str, nargs="+", default=None, choices=list(PROJS.keys()),
                   help="Restrict pruning to these projection types (default: all seven).")
    p.add_argument("--seed", type=int, default=42, help="Only used by --method random.")
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--screen-dir", default="experiments/experiment_2_qwen3_4b/results/01_per_matrix_screening")
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--limit", type=int, default=None, help="Examples per task (for a quick smoke).")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    model, tokenizer = load_model(args.model)

    tag = "dense"
    prune_info = None
    if not args.dense:
        layers = args.layers if args.layers is not None else list(range(count_layers(model)))

        calib = None
        if args.method in NEEDS_CALIB:
            calib = get_calib_data(tokenizer, n_calib=args.calib_samples,
                                   seqlen=args.calib_seqlen)

        ratio_map, policy_info = None, None
        if args.policy == "sensitivity":
            sens = load_sens(args.screen_dir, method="sparsegpt_recon", ref_ratio="0.20")
            classes = classify(sens)
            measured = sorted({l for (l, _) in sens})
            full = fill_all_layers(classes, layers, list(PROJS.keys()), measured)
            tiles = tile_counts(model, layers, args.tile_size)
            ratio_map, policy_info = allocate_budget(full, tiles, args.prune_ratio)
            print(f"Policy B: target {policy_info['target']:.3f} -> achieved {policy_info['achieved']:.4f}")

        print(f"Pruning whole model: {args.method} @ {args.prune_ratio} ({args.policy} policy)")
        tt, tp = prune_whole_model(model, args, layers, calib, ratio_map)
        print(f"Pruned {tp}/{tt} tiles = {tp / tt:.4f}")
        prune_info = {"total_tiles": tt, "total_pruned": tp, "achieved_sparsity": tp / tt,
                      "policy_info": policy_info}
        tag = f"{args.method}_p{int(args.prune_ratio * 100)}_{args.policy}"

        del calib, ratio_map
        calib = ratio_map = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            print(f"GPU free before eval: {free / 1e9:.2f} / {total / 1e9:.2f} GB")

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    print(f"Running lm-eval on tasks: {args.tasks}")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    res = lm_eval.simple_evaluate(model=lm, tasks=args.tasks, limit=args.limit)

    out = {
        "tag": tag,
        "model": args.model,
        "dense": args.dense,
        "method": None if args.dense else args.method,
        "prune_ratio": None if args.dense else args.prune_ratio,
        "policy": None if args.dense else args.policy,
        "tile_size": args.tile_size,
        "limit": args.limit,
        "pruned": prune_info,
        "results": res["results"],
    }
    path = args.output or os.path.join("experiments/experiment_2_qwen3_4b", "results", "08_downstream", f"downstream_{tag}.json")
    save_json(path, out)

    print("\n=== downstream accuracy ===")
    for t, v in res["results"].items():
        acc = v.get("acc_norm,none", v.get("acc,none"))
        print(f"  {t:12} {acc}")


if __name__ == "__main__":
    main()
