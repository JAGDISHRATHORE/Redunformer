import argparse
import json
import os
import random
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from redundancy.models import load_model
from redundancy.data import load_wikitext, get_calib_data
from redundancy.eval import (
    eval_ppl,
    make_probe,
    dense_outputs,
    compare_to_dense,
)
from redundancy.hooks import collect_grams
from redundancy.scoring import wanda_scores, sgpt_mask_err
from redundancy.recovery import prune_and_repair
from redundancy.combined import wanda_recon, random_recon
from redundancy.policy import load_sens, classify, fill_all_layers, allocate_budget


SEEDED = ("random", "random_recon")

PROJS = {
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
}


def full_tiles(weight, tile_size):
    rows, cols = weight.shape
    tiles = []

    for r in range(0, rows, tile_size):
        for c in range(0, cols, tile_size):
            tile = weight[r:r + tile_size, c:c + tile_size]
            if tile.shape == (tile_size, tile_size):
                tiles.append((r, c))

    return tiles


def zero_tiles(weight, to_prune, tile_size):
    with torch.no_grad():
        for r, c in to_prune:
            weight[r:r + tile_size, c:c + tile_size] = 0


def get_mask(weight, W_orig, tile_size, rel_tol=0.1):
    with torch.no_grad():
        rows, cols = weight.shape
        nr, nc = rows // tile_size, cols // tile_size
        cut_r, cut_c = nr * tile_size, nc * tile_size
        wt = weight[:cut_r, :cut_c].contiguous().float().reshape(nr, tile_size, nc, tile_size)
        w0 = W_orig[:cut_r, :cut_c].contiguous().float().reshape(nr, tile_size, nc, tile_size)
        cur = wt.abs().amax(dim=3).amax(dim=1)
        orig = w0.abs().amax(dim=3).amax(dim=1)
        removed = (orig > 0) & (cur <= rel_tol * orig)
        idx = removed.nonzero(as_tuple=False)
    return [[int(i) * tile_size, int(j) * tile_size] for i, j in idx.tolist()]


def prune_mag(weight, tile_size, prune_ratio):
    tiles = full_tiles(weight, tile_size)
    scored = []

    for r, c in tiles:
        tile = weight[r:r + tile_size, c:c + tile_size]
        score = torch.norm(tile).item()
        scored.append((score, r, c))

    scored.sort(key=lambda x: x[0])

    n_prune = int(len(scored) * prune_ratio)
    to_prune = [(r, c) for _, r, c in scored[:n_prune]]

    zero_tiles(weight, to_prune, tile_size)

    return len(tiles), n_prune


def prune_mag_high(weight, tile_size, prune_ratio):
    tiles = full_tiles(weight, tile_size)
    scored = []

    for r, c in tiles:
        tile = weight[r:r + tile_size, c:c + tile_size]
        score = torch.norm(tile).item()
        scored.append((score, r, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    n_prune = int(len(scored) * prune_ratio)
    to_prune = [(r, c) for _, r, c in scored[:n_prune]]

    zero_tiles(weight, to_prune, tile_size)

    return len(tiles), n_prune


def prune_random(weight, tile_size, prune_ratio, seed):
    tiles = full_tiles(weight, tile_size)

    rng = random.Random(seed)
    rng.shuffle(tiles)

    n_prune = int(len(tiles) * prune_ratio)
    to_prune = tiles[:n_prune]

    zero_tiles(weight, to_prune, tile_size)

    return len(tiles), n_prune


def prune_lowest(weight, scored, tile_size, prune_ratio):
    scored = sorted(scored, key=lambda x: x[0])

    n_prune = int(len(scored) * prune_ratio)
    to_prune = [(r, c) for _, r, c in scored[:n_prune]]

    zero_tiles(weight, to_prune, tile_size)

    return len(scored), n_prune


def prune_wanda(weight, tile_size, prune_ratio, col_norms):
    scored = wanda_scores(weight, col_norms, tile_size)
    return prune_lowest(weight, scored, tile_size, prune_ratio)


def prune_sparsegpt(weight, tile_size, prune_ratio, hessian):
    scored = sgpt_mask_err(weight, hessian, tile_size)
    return prune_lowest(weight, scored, tile_size, prune_ratio)


def prune_sparsegpt_recon(weight, tile_size, prune_ratio, hessian):
    return prune_and_repair(weight, hessian, tile_size, prune_ratio)


def do_prune(weight, method, tile_size, prune_ratio, seed, stat=None):
    if method == "magnitude":
        return prune_mag(weight, tile_size, prune_ratio)
    if method == "magnitude_high":
        return prune_mag_high(weight, tile_size, prune_ratio)
    if method == "random":
        return prune_random(weight, tile_size, prune_ratio, seed)
    if method == "wanda":
        return prune_wanda(weight, tile_size, prune_ratio, stat)
    if method == "sparsegpt":
        return prune_sparsegpt(weight, tile_size, prune_ratio, stat)
    if method == "sparsegpt_recon":
        return prune_sparsegpt_recon(weight, tile_size, prune_ratio, stat)
    if method == "wanda_recon":
        return wanda_recon(weight, tile_size, prune_ratio, stat)
    if method == "random_recon":
        return random_recon(weight, tile_size, prune_ratio, stat, seed)

    raise ValueError(f"Unknown pruning method: {method}")


def get_weight(model, pname):
    for name, param in model.named_parameters():
        if name == pname:
            return param

    raise ValueError(f"Could not find target weight: {pname}")


def get_module(model, pname):
    mpath = pname[:-len(".weight")] if pname.endswith(".weight") else pname

    module = model
    for attr in mpath.split("."):
        module = module[int(attr)] if attr.isdigit() else getattr(module, attr)

    return module


def calib_stats(model, calib, pnames, method):
    mods = {name: get_module(model, name) for name in pnames}
    collectors = collect_grams(model, mods, calib)

    stats = {}
    for name, collector in collectors.items():
        if method == "wanda":
            stats[name] = collector.col_norms().detach().clone()
        else:
            stats[name] = collector.H

    return stats


def count_layers(model):
    ids = set()

    for name, _ in model.named_parameters():
        parts = name.split(".")
        if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers":
            if parts[2].isdigit():
                ids.add(int(parts[2]))

    if not ids:
        raise ValueError("Could not automatically detect model layers.")

    return max(ids) + 1


def param_name(layer, proj):
    component = PROJS[proj]
    return f"model.layers.{layer}.{component}.{proj}.weight"


def pct_str(prune_ratio):
    return str(int(prune_ratio * 100))


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to {path}")


def run_one_matrix(model, tokenizer, dataset, args, pname, seed=None, stat=None,
                   probe=None, ref=None):
    W = get_weight(model, pname)
    W_orig = W.detach().clone()

    print("\n=======================================")
    print(f"Pruning target matrix: {pname}")
    print(f"Matrix shape: {W.shape}")
    print(f"Tile size: {args.tile_size}")
    print(f"Prune ratio: {args.prune_ratio}")
    print(f"Method: {args.method}")
    print(f"Seed: {seed if args.method == 'random' else None}")
    print("=======================================")

    n_tiles, n_pruned = do_prune(
        W,
        args.method,
        args.tile_size,
        args.prune_ratio,
        seed,
        stat=stat,
    )

    print(f"Total full tiles: {n_tiles}")
    print(f"Pruned tiles: {n_pruned}")

    ppl = eval_ppl(model, tokenizer, dataset, eval_frac=args.eval_frac)

    print(f"Final Perplexity after pruning: {ppl:.4f}")

    div = None
    if probe is not None and ref is not None:
        div = compare_to_dense(model, probe, ref)
        print(
            f"Divergence vs dense: KL={div['kl_dense_pruned']:.4f} "
            f"top1={div['top1_agreement']:.3f} cos={div['hidden_cosine']:.4f}"
        )

    try:
        pruned_mask = get_mask(W, W_orig, args.tile_size)
    except Exception as exc:
        print(f"[warn] mask extraction failed: {exc}")
        pruned_mask = None

    with torch.no_grad():
        W.copy_(W_orig)

    return {
        "target_matrix": pname,
        "tile_size": args.tile_size,
        "prune_ratio": args.prune_ratio,
        "num_tiles": n_tiles,
        "num_pruned": n_pruned,
        "method": args.method,
        "seed": seed if args.method in SEEDED else None,
        "perplexity": ppl,
        "divergence": div,
        "pruned_mask": pruned_mask,
    }


def run_whole_layer(model, tokenizer, dataset, args, layer, seed=None,
                    layer_stats=None, probe=None, ref=None):
    targets = {m: param_name(layer, m) for m in PROJS.keys()}
    weights = {m: get_weight(model, t) for m, t in targets.items()}
    originals = {m: w.detach().clone() for m, w in weights.items()}

    print("\n=======================================")
    print(f"Whole-layer pruning: layer {layer}  method {args.method}  ratio {args.prune_ratio}")
    print("=======================================")

    proj_info = []
    for proj in PROJS.keys():
        stat = layer_stats[targets[proj]] if layer_stats is not None else None
        n_tiles, n_pruned = do_prune(
            weights[proj], args.method, args.tile_size, args.prune_ratio, seed, stat=stat,
        )
        try:
            mask = get_mask(weights[proj], originals[proj], args.tile_size)
        except Exception as exc:
            print(f"[warn] mask extraction failed ({proj}): {exc}")
            mask = None
        proj_info.append({
            "matrix": proj,
            "num_tiles": n_tiles,
            "num_pruned": n_pruned,
            "pruned_mask": mask,
        })

    ppl = eval_ppl(model, tokenizer, dataset, eval_frac=args.eval_frac)
    print(f"Whole-layer perplexity: {ppl:.4f}")

    div = None
    if probe is not None and ref is not None:
        div = compare_to_dense(model, probe, ref)
        print(
            f"Divergence vs dense: KL={div['kl_dense_pruned']:.4f} "
            f"top1={div['top1_agreement']:.3f} cos={div['hidden_cosine']:.4f}"
        )

    with torch.no_grad():
        for proj in PROJS.keys():
            weights[proj].copy_(originals[proj])

    return {
        "layer": layer,
        "method": args.method,
        "seed": seed if args.method in SEEDED else None,
        "tile_size": args.tile_size,
        "prune_ratio": args.prune_ratio,
        "scope": "whole_layer",
        "perplexity": ppl,
        "divergence": div,
        "matrices": proj_info,
    }


def tile_counts(model, layers, tile_size):
    tiles = {}
    for layer in layers:
        for proj in PROJS.keys():
            w = get_weight(model, param_name(layer, proj))
            rows, cols = w.shape
            tiles[(layer, proj)] = (rows // tile_size) * (cols // tile_size)
    return tiles


def run_whole_model(model, tokenizer, dataset, args, layers, seed=None,
                    calib=None, probe=None, ref=None,
                    ratio_map=None):
    need_calib = args.method in ("wanda", "sparsegpt", "sparsegpt_recon", "wanda_recon", "random_recon")
    total_tiles = 0
    total_pruned = 0
    per_layer = []

    projs = getattr(args, "matrices", None) or list(PROJS.keys())

    for layer in layers:
        layer_stats = None
        if need_calib:
            pnames = [param_name(layer, m) for m in projs]
            layer_stats = calib_stats(model, calib, pnames, args.method)

        layer_pruned = 0
        for proj in projs:
            pname = param_name(layer, proj)
            weight = get_weight(model, pname)
            stat = layer_stats[pname] if layer_stats is not None else None
            ratio = ratio_map[(layer, proj)] if ratio_map is not None else args.prune_ratio
            n_tiles, n_pruned = do_prune(
                weight, args.method, args.tile_size, ratio, seed, stat=stat,
            )
            total_tiles += n_tiles
            total_pruned += n_pruned
            layer_pruned += n_pruned

        per_layer.append({"layer": layer, "num_pruned": layer_pruned})
        print(f"  layer {layer:2}: pruned {layer_pruned} tiles (total {total_pruned})")

        del layer_stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Evaluating fully pruned whole model...")
    ppl = eval_ppl(model, tokenizer, dataset, eval_frac=args.eval_frac)
    print(f"Whole-model perplexity: {ppl:.4f}")

    div = None
    if probe is not None and ref is not None:
        div = compare_to_dense(model, probe, ref)
        print(
            f"Divergence vs dense: KL={div['kl_dense_pruned']:.4f} "
            f"top1={div['top1_agreement']:.3f} cos={div['hidden_cosine']:.4f}"
        )

    return {
        "method": args.method,
        "seed": seed if args.method in SEEDED else None,
        "tile_size": args.tile_size,
        "prune_ratio": args.prune_ratio,
        "policy": getattr(args, "policy", "uniform"),
        "scope": "whole_model",
        "num_layers": len(layers),
        "total_tiles": total_tiles,
        "total_pruned": total_pruned,
        "achieved_sparsity": (total_pruned / total_tiles) if total_tiles else None,
        "perplexity": ppl,
        "divergence": div,
        "per_layer": per_layer,
    }


def load_layer_jsons(exp_dir):
    data = []

    for fname in os.listdir(exp_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(exp_dir, fname)

        with open(path, "r") as f:
            item = json.load(f)

        if "results" in item and "layer" in item and "method" in item:
            data.append(item)

    return data


def plot_layer_trends(exp_dir, base_ppl):
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    data = load_layer_jsons(exp_dir)

    if not data:
        raise ValueError(f"No valid JSON files found in {exp_dir}")

    groups = defaultdict(list)

    for item in data:
        method = item["method"]
        seed = item.get("seed", None)

        if method == "random":
            key = f"random_seed{seed}"
        else:
            key = method

        groups[key].append(item)

    for key, summaries in groups.items():
        summaries = sorted(summaries, key=lambda x: x["layer"])
        layers = [s["layer"] for s in summaries]

        present = [m for m in PROJS.keys() if all(
                   any(r["matrix"] == m for r in s["results"]) for s in summaries)]
        vals_by_proj = {m: [] for m in present}

        for summary in summaries:
            rmap = {r["matrix"]: r["perplexity"] for r in summary["results"]}
            for matrix in present:
                vals_by_proj[matrix].append(rmap[matrix])

        if not vals_by_proj:
            continue

        plt.figure(figsize=(15, 7), dpi=200)

        for matrix, values in vals_by_proj.items():
            plt.plot(layers, values, marker="o", linewidth=1.8, label=matrix)

        plt.axhline(
            base_ppl,
            linestyle="--",
            linewidth=1.2,
            label=f"Baseline PPL = {base_ppl}",
        )

        plt.title(f"{key}: perplexity across all layers")
        plt.xlabel("Layer")
        plt.ylabel("Perplexity")
        plt.xticks(layers)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()

        out = os.path.join(plot_dir, f"{key}_all_layers_projection_trend.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {out}")

    means = {}
    n_proj = 0

    for key, summaries in groups.items():
        values = []

        for summary in summaries:
            layer = summary["layer"]
            n_proj = max(n_proj, len(summary["results"]))
            mean_ppl = sum(r["perplexity"] for r in summary["results"]) / len(summary["results"])
            values.append((layer, mean_ppl))

        means[key] = sorted(values, key=lambda x: x[0])

    plt.figure(figsize=(15, 7), dpi=200)

    for key, values in sorted(means.items()):
        layers = [x[0] for x in values]
        mean_ppl = [x[1] for x in values]
        plt.plot(layers, mean_ppl, marker="o", linewidth=2, label=key)

    plt.axhline(
        base_ppl,
        linestyle="--",
        linewidth=1.2,
        label=f"Baseline PPL = {base_ppl}",
    )

    title_methods = " vs ".join(sorted(means))
    plt.title(f"Mean layer sensitivity: {title_methods}")
    plt.xlabel("Layer")
    plt.ylabel(f"Mean perplexity across {n_proj} projection matri"
               f"{'x' if n_proj == 1 else 'ces'}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(plot_dir, "mean_layer_trend.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")

    mag_key = None
    rand_keys = []

    for key in groups.keys():
        if key == "magnitude":
            mag_key = key
        if key.startswith("random_seed"):
            rand_keys.append(key)

    if mag_key is not None and rand_keys:
        for rand_key in rand_keys:
            mag_map = {}
            rand_map = {}

            for summary in groups[mag_key]:
                layer = summary["layer"]
                for result in summary["results"]:
                    mag_map[(layer, result["matrix"])] = result["perplexity"]

            for summary in groups[rand_key]:
                layer = summary["layer"]
                for result in summary["results"]:
                    rand_map[(layer, result["matrix"])] = result["perplexity"]

            common = sorted(set(l for l, _ in mag_map.keys()) & set(l for l, _ in rand_map.keys()))

            for matrix in PROJS.keys():
                layers = []
                mag_vals = []
                rand_vals = []

                for layer in common:
                    k2 = (layer, matrix)
                    if k2 in mag_map and k2 in rand_map:
                        layers.append(layer)
                        mag_vals.append(mag_map[k2])
                        rand_vals.append(rand_map[k2])

                if not layers:
                    continue

                plt.figure(figsize=(12, 6), dpi=200)
                plt.plot(layers, mag_vals, marker="o", linewidth=2, label="Magnitude")
                plt.plot(layers, rand_vals, marker="o", linewidth=2, label=rand_key)

                plt.axhline(
                    base_ppl,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Baseline PPL = {base_ppl}",
                )

                plt.title(f"{matrix}: magnitude vs {rand_key}")
                plt.xlabel("Layer")
                plt.ylabel("Perplexity")
                plt.xticks(layers)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()

                out = os.path.join(
                    plot_dir,
                    f"{matrix}_magnitude_vs_{rand_key}_layer_trend.png",
                )
                plt.savefig(out, bbox_inches="tight")
                plt.close()
                print(f"Saved plot: {out}")

            dlayers = common
            diff = []

            for matrix in PROJS.keys():
                row = []
                for layer in dlayers:
                    k2 = (layer, matrix)
                    if k2 in mag_map and k2 in rand_map:
                        row.append(mag_map[k2] - rand_map[k2])
                    else:
                        row.append(float("nan"))
                diff.append(row)

            diff = np.array(diff)

            plt.figure(figsize=(16, 6), dpi=200)
            im = plt.imshow(diff, aspect="auto")

            plt.colorbar(im, label="Magnitude PPL - Random PPL")
            plt.yticks(range(len(PROJS)), list(PROJS.keys()))
            plt.xticks(range(len(dlayers)), dlayers)
            plt.xlabel("Layer")
            plt.ylabel("Projection matrix")
            plt.title(f"Difference heatmap: magnitude - {rand_key}")

            for i in range(diff.shape[0]):
                for j in range(diff.shape[1]):
                    val = diff[i, j]
                    if not np.isnan(val):
                        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

            plt.tight_layout()

            out = os.path.join(
                plot_dir,
                f"magnitude_minus_{rand_key}_difference_heatmap.png",
            )
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {out}")


def main():
    parser = argparse.ArgumentParser(description="Run tile-level pruning experiment.")

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")

    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--prune-ratio", type=float, default=0.20)

    parser.add_argument(
        "--method",
        type=str,
        choices=["magnitude", "magnitude_high", "random", "wanda", "sparsegpt", "sparsegpt_recon",
                 "wanda_recon", "random_recon"],
        default="magnitude",
    )

    parser.add_argument(
        "--calib-samples",
        type=int,
        default=128,
        help="Number of calibration windows for wanda/sparsegpt.",
    )

    parser.add_argument(
        "--calib-seqlen",
        type=int,
        default=512,
        help="Token length of each calibration window.",
    )

    parser.add_argument(
        "--eval-frac",
        type=float,
        default=1.0,
        help="Fraction of the eval corpus for perplexity (e.g. 0.2 for fast screening; 1.0 = full).",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds, e.g. --seeds 123 456 789",
    )

    parser.add_argument(
        "--target-name",
        type=str,
        default="model.layers.0.mlp.up_proj.weight",
        help="Single target matrix to prune when --all-matrices is not used.",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to run, e.g. --layers 0 12 27",
    )

    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Run every transformer layer detected in the model.",
    )

    parser.add_argument(
        "--all-matrices",
        action="store_true",
        help="Run all seven projection matrices for each layer.",
    )

    parser.add_argument(
        "--matrices",
        type=str,
        nargs="+",
        default=None,
        choices=list(PROJS.keys()),
        help="With --all-matrices, restrict the scan to these projection types "
             "(e.g. --matrices o_proj up_proj). Default: all seven. Used by the cluster "
             "analysis, which scans only the most robust and most sensitive matrix types.",
    )

    parser.add_argument(
        "--whole-layer",
        action="store_true",
        help="Prune all seven matrices of each layer simultaneously and evaluate once.",
    )

    parser.add_argument(
        "--whole-model",
        action="store_true",
        help="Prune every matrix across all layers at one uniform sparsity, evaluate once.",
    )

    parser.add_argument(
        "--policy",
        type=str,
        choices=["uniform", "sensitivity"],
        default="uniform",
        help="Whole-model budget policy: uniform (A), or sensitivity-aware budget-matched (B).",
    )

    parser.add_argument(
        "--screen-dir",
        type=str,
        default="experiments/experiment_2_qwen3_4b/results/01_per_matrix_screening",
        help="Screening results used to derive the sensitivity classes for --policy sensitivity.",
    )

    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments/new_runs",
        help="Directory where JSON files and plots will be saved.",
    )

    parser.add_argument(
        "--baseline-ppl",
        type=float,
        default=13.2181,
        help="Baseline perplexity shown as a reference line in plots.",
    )

    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only create plots from existing JSON files.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for single-matrix mode.",
    )

    args = parser.parse_args()

    if args.plot_only:
        plot_layer_trends(args.experiment_dir, args.baseline_ppl)
        return

    model, tokenizer = load_model(args.model)
    dataset = load_wikitext(args.dataset, args.subset, split="test")

    print("Building divergence probe and dense reference...")
    probe = make_probe(tokenizer, dataset)
    ref = dense_outputs(model, probe)

    need_calib = args.method in ("wanda", "sparsegpt", "sparsegpt_recon", "wanda_recon", "random_recon")
    calib = None
    if need_calib:
        calib = get_calib_data(
            tokenizer,
            n_calib=args.calib_samples,
            seqlen=args.calib_seqlen,
        )

    if args.all_layers:
        num_layers = count_layers(model)
        layers = list(range(num_layers))
        print(f"Detected {num_layers} layers: {layers}")
    elif args.layers is not None:
        layers = args.layers
    else:
        layers = None

    if args.whole_layer:
        if layers is None or len(layers) == 0:
            raise ValueError("Use --layers or --all-layers with --whole-layer")

        seeds = args.seeds if args.method in SEEDED else [None]

        for seed in seeds:
            for layer in layers:
                layer_stats = None
                if need_calib:
                    pnames = [param_name(layer, m) for m in PROJS.keys()]
                    print(f"Collecting {args.method} calibration stats for layer {layer}...")
                    layer_stats = calib_stats(model, calib, pnames, args.method)

                result = run_whole_layer(
                    model, tokenizer, dataset, args, layer, seed=seed,
                    layer_stats=layer_stats, probe=probe, ref=ref,
                )

                ratio = pct_str(args.prune_ratio)
                if args.method in SEEDED:
                    fname = f"layer{layer}_wholelayer_{args.method}_p{ratio}_seed{seed}.json"
                else:
                    fname = f"layer{layer}_wholelayer_{args.method}_p{ratio}.json"

                summary = {
                    "model": args.model,
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "calib_samples": args.calib_samples if need_calib else None,
                    "calib_seqlen": args.calib_seqlen if need_calib else None,
                    "eval_frac": args.eval_frac,
                    **result,
                }
                save_json(os.path.join(args.experiment_dir, fname), summary)

        return

    if args.whole_model:
        if layers is None:
            layers = list(range(count_layers(model)))
        print(f"Whole-model pruning over {len(layers)} layers: {layers}")

        ratio_map, policy_info = None, None
        if args.policy == "sensitivity":
            sens = load_sens(args.screen_dir, method="sparsegpt_recon", ref_ratio="0.20")
            classes = classify(sens)
            measured = sorted({l for (l, _) in sens})
            full = fill_all_layers(classes, layers, list(PROJS.keys()), measured)
            tiles = tile_counts(model, layers, args.tile_size)
            ratio_map, policy_info = allocate_budget(full, tiles, args.prune_ratio)
            print(f"Policy B (sensitivity-aware, budget-matched) — measured layers {measured}")
            print(f"  target sparsity {policy_info['target']:.3f}  ->  achieved {policy_info['achieved']:.4f}")
            for c, d in policy_info["per_class"].items():
                print(f"  {c:10} ratio {d['ratio']:.3f}   matrices {d['matrices']:3}   tiles {d['tiles']:,}")

        seed = args.seeds[0] if args.method in SEEDED else None
        result = run_whole_model(
            model, tokenizer, dataset, args, layers, seed=seed,
            calib=calib, probe=probe, ref=ref,
            ratio_map=ratio_map,
        )

        ratio = pct_str(args.prune_ratio)
        suffix = f"_seed{seed}" if args.method in SEEDED else ""
        pol = "" if args.policy == "uniform" else f"_{args.policy}"
        summary = {
            "model": args.model,
            "dataset": args.dataset,
            "subset": args.subset,
            "calib_samples": args.calib_samples if need_calib else None,
            "calib_seqlen": args.calib_seqlen if need_calib else None,
            "eval_frac": args.eval_frac,
            "policy_info": policy_info,
            **result,
        }
        save_json(os.path.join(args.experiment_dir, f"wholemodel_{args.method}_p{ratio}{pol}{suffix}.json"), summary)
        return

    if args.all_matrices:
        if layers is None or len(layers) == 0:
            raise ValueError("Use --layers or --all-layers with --all-matrices")

        seeds = args.seeds if args.method in SEEDED else [None]

        for seed in seeds:
            for layer in layers:
                res_list = []

                print("\n#######################################")
                print(f"Running Layer {layer}")
                print(f"Method: {args.method}")
                print(f"Seed: {seed if args.method == 'random' else None}")
                print("#######################################")

                projs = args.matrices or list(PROJS.keys())

                layer_stats = None
                if need_calib:
                    pnames = [param_name(layer, m) for m in projs]
                    print(f"Collecting {args.method} calibration stats for layer {layer}...")
                    layer_stats = calib_stats(
                        model, calib, pnames, args.method
                    )

                for proj in projs:
                    pname = param_name(layer, proj)
                    stat = layer_stats[pname] if layer_stats is not None else None

                    result = run_one_matrix(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        args=args,
                        pname=pname,
                        seed=seed,
                        stat=stat,
                        probe=probe,
                        ref=ref,
                    )

                    result["layer"] = layer
                    result["matrix"] = proj
                    res_list.append(result)

                ratio = pct_str(args.prune_ratio)

                if args.method in SEEDED:
                    fname = f"layer{layer}_{args.method}_p{ratio}_seed{seed}.json"
                else:
                    fname = f"layer{layer}_{args.method}_p{ratio}.json"

                out_path = os.path.join(args.experiment_dir, fname)

                layer_summary = {
                    "model": args.model,
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "layer": layer,
                    "method": args.method,
                    "seed": seed if args.method in SEEDED else None,
                    "tile_size": args.tile_size,
                    "prune_ratio": args.prune_ratio,
                    "calib_samples": args.calib_samples if need_calib else None,
                    "calib_seqlen": args.calib_seqlen if need_calib else None,
                    "eval_frac": args.eval_frac,
                    "results": res_list,
                }

                save_json(out_path, layer_summary)

        plot_layer_trends(args.experiment_dir, args.baseline_ppl)

    else:
        seed = args.seeds[0] if args.method in SEEDED else None

        stat = None
        if need_calib:
            print(f"Collecting {args.method} calibration stats for {args.target_name}...")
            stats = calib_stats(
                model, calib, [args.target_name], args.method
            )
            stat = stats[args.target_name]

        result = run_one_matrix(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            args=args,
            pname=args.target_name,
            seed=seed,
            stat=stat,
            probe=probe,
            ref=ref,
        )

        out_path = args.output
        if out_path is None:
            out_path = "experiments/new_runs/single_matrix_run.json"

        results = {
            "model": args.model,
            "dataset": args.dataset,
            "subset": args.subset,
            **result,
        }

        save_json(out_path, results)


if __name__ == "__main__":
    main()
