import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch

from run_tile_pruning import (
    PROJS, param_name, get_weight, calib_stats,
    count_layers, save_json, pct_str, zero_tiles,
)
from redundancy.models import load_model
from redundancy.data import load_wikitext, get_calib_data
from redundancy.eval import eval_ppl
from redundancy.scoring import wanda_scores
from redundancy.layer_budget import pool_layer_budget


def main():
    p = argparse.ArgumentParser(description="Rathore W3: budget-matched layer-wide Wanda pruning.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 9, 18, 27, 35])
    p.add_argument("--prune-ratio", type=float, default=0.20,
                   help="Total layer budget as a fraction of the layer's tiles.")
    p.add_argument("--mode", choices=["mean", "raw"], default="mean",
                   help="Score normalisation within each matrix before pooling.")
    p.add_argument("--tile-size", type=int, default=32)
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--subset", default="wikitext-2-raw-v1")
    p.add_argument("--experiment-dir", default="experiments/experiment_2_qwen3_4b/results/04_layer_budget_wanda")
    args = p.parse_args()

    model, tokenizer = load_model(args.model)
    dataset = load_wikitext(args.dataset, args.subset, split="test")
    calib = get_calib_data(tokenizer, n_calib=args.calib_samples,
                           seqlen=args.calib_seqlen)

    layers = args.layers or list(range(count_layers(model)))
    ratio = pct_str(args.prune_ratio)

    for layer in layers:
        print(f"\n########## Layer {layer} — W3 budget-matched (budget {args.prune_ratio}) ##########")

        names = [param_name(layer, m) for m in PROJS.keys()]
        stats = calib_stats(model, calib, names, "wanda")

        weights = {m: get_weight(model, param_name(layer, m)) for m in PROJS.keys()}
        originals = {m: w.detach().clone() for m, w in weights.items()}

        scores = {
            m: wanda_scores(weights[m], stats[param_name(layer, m)], args.tile_size)
            for m in PROJS.keys()
        }
        pruned, info = pool_layer_budget(scores, args.prune_ratio, mode=args.mode)

        print(f"  layer tiles {info['layer_total_tiles']:,} · budget {info['layer_pruned_tiles']:,} "
              f"· achieved {info['achieved_ratio']:.4f}")
        print("  emergent allocation (uniform would be flat at "
              f"{args.prune_ratio:.0%}):")
        for m, d in sorted(info["per_matrix"].items(), key=lambda x: -x[1]["ratio"]):
            print(f"    {m:11} {d['pruned']:>6,}/{d['tiles']:<6,} -> {d['ratio']:6.1%}")

        with torch.no_grad():
            for m, tiles in pruned.items():
                if tiles:
                    zero_tiles(weights[m], tiles, args.tile_size)

        ppl = eval_ppl(model, tokenizer, dataset, eval_frac=args.eval_frac)
        print(f"  perplexity {ppl:.4f}")

        with torch.no_grad():
            for m in PROJS.keys():
                weights[m].copy_(originals[m])
        del stats, originals
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_json(os.path.join(args.experiment_dir, f"layer{layer}_w3_p{ratio}.json"), {
                  "model": args.model,
                  "dataset": args.dataset,
                  "subset": args.subset,
                  "scope": "layer_budget_matched",
                  "strategy": "rathore_W3",
                  "layer": layer,
                  "method": "wanda_layer_budget",
                  "mode": args.mode,
                  "tile_size": args.tile_size,
                  "prune_ratio": args.prune_ratio,
                  "calib_samples": args.calib_samples,
                  "calib_seqlen": args.calib_seqlen,
                  "eval_frac": args.eval_frac,
                  "perplexity": ppl,
                  "allocation": info,
        })


if __name__ == "__main__":
    main()
