import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from redundancy.models import load_model_and_tokenizer
from redundancy.data import load_evaluation_dataset
from redundancy.eval import evaluate_perplexity


MATRICES = {
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
}


def get_full_tiles(weight, tile_size):
    rows, cols = weight.shape
    tiles = []

    for r in range(0, rows, tile_size):
        for c in range(0, cols, tile_size):
            tile = weight[r:r + tile_size, c:c + tile_size]
            if tile.shape == (tile_size, tile_size):
                tiles.append((r, c))

    return tiles


def zero_tiles(weight, tiles_to_prune, tile_size):
    with torch.no_grad():
        for r, c in tiles_to_prune:
            weight[r:r + tile_size, c:c + tile_size] = 0


def prune_lowest_magnitude(weight, tile_size, prune_ratio):
    tiles = get_full_tiles(weight, tile_size)
    scored_tiles = []

    for r, c in tiles:
        tile = weight[r:r + tile_size, c:c + tile_size]
        score = torch.norm(tile).item()
        scored_tiles.append((score, r, c))

    scored_tiles.sort(key=lambda x: x[0])

    num_prune = int(len(scored_tiles) * prune_ratio)
    tiles_to_prune = [(r, c) for _, r, c in scored_tiles[:num_prune]]

    zero_tiles(weight, tiles_to_prune, tile_size)

    return len(tiles), num_prune


def prune_highest_magnitude(weight, tile_size, prune_ratio):
    tiles = get_full_tiles(weight, tile_size)
    scored_tiles = []

    for r, c in tiles:
        tile = weight[r:r + tile_size, c:c + tile_size]
        score = torch.norm(tile).item()
        scored_tiles.append((score, r, c))

    scored_tiles.sort(key=lambda x: x[0], reverse=True)

    num_prune = int(len(scored_tiles) * prune_ratio)
    tiles_to_prune = [(r, c) for _, r, c in scored_tiles[:num_prune]]

    zero_tiles(weight, tiles_to_prune, tile_size)

    return len(tiles), num_prune


def prune_random(weight, tile_size, prune_ratio, seed):
    tiles = get_full_tiles(weight, tile_size)

    rng = random.Random(seed)
    rng.shuffle(tiles)

    num_prune = int(len(tiles) * prune_ratio)
    tiles_to_prune = tiles[:num_prune]

    zero_tiles(weight, tiles_to_prune, tile_size)

    return len(tiles), num_prune


def apply_pruning(weight, method, tile_size, prune_ratio, seed):
    if method == "magnitude":
        return prune_lowest_magnitude(weight, tile_size, prune_ratio)
    if method == "magnitude_high":
        return prune_highest_magnitude(weight, tile_size, prune_ratio)
    if method == "random":
        return prune_random(weight, tile_size, prune_ratio, seed)

    raise ValueError(f"Unknown pruning method: {method}")


def get_target_weight(model, target_name):
    for name, param in model.named_parameters():
        if name == target_name:
            return param

    raise ValueError(f"Could not find target weight: {target_name}")


def run_single_experiment(model, tokenizer, dataset, args, target_name, seed=None):
    target_weight = get_target_weight(model, target_name)
    original_weight = target_weight.detach().clone()

    print("\n=======================================")
    print(f"Pruning target matrix: {target_name}")
    print(f"Matrix shape: {target_weight.shape}")
    print(f"Tile size: {args.tile_size}")
    print(f"Prune ratio: {args.prune_ratio}")
    print(f"Method: {args.method}")
    print(f"Seed: {seed if args.method == 'random' else None}")
    print("=======================================")

    num_tiles, num_pruned = apply_pruning(
        target_weight,
        args.method,
        args.tile_size,
        args.prune_ratio,
        seed,
    )

    print(f"Total full tiles: {num_tiles}")
    print(f"Pruned tiles: {num_pruned}")
    ppl = evaluate_perplexity(model, tokenizer, dataset)

    print(f"Final Perplexity after pruning: {ppl:.4f}")

    with torch.no_grad():
        target_weight.copy_(original_weight)

    return {
        "target_matrix": target_name,
        "tile_size": args.tile_size,
        "prune_ratio": args.prune_ratio,
        "num_tiles": num_tiles,
        "num_pruned": num_pruned,
        "method": args.method,
        "seed": seed if args.method == "random" else None,
        "perplexity": ppl,
    }


def build_target_name(layer, matrix_name):
    component = MATRICES[matrix_name]
    return f"model.layers.{layer}.{component}.{matrix_name}.weight"


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nResults saved to {path}")


def method_short_name(method):
    return method


def ratio_short_name(prune_ratio):
    return str(int(prune_ratio * 100))


def main():
    parser = argparse.ArgumentParser(description="Run tile-level pruning experiment.")

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")

    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--prune-ratio", type=float, default=0.05)

    parser.add_argument(
        "--method",
        type=str,
        choices=["magnitude", "magnitude_high", "random"],
        default="magnitude",
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
        help="Layers to run when using --all-matrices, e.g. --layers 0 12 27",
    )

    parser.add_argument(
        "--all-matrices",
        action="store_true",
        help="Run all seven projection matrices for each provided layer.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for single-matrix mode.",
    )

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    dataset = load_evaluation_dataset(args.dataset, args.subset, split="test")

    if args.all_matrices:
        if args.layers is None or len(args.layers) == 0:
            raise ValueError(
                "When using --all-matrices, please provide --layers, e.g. --layers 0 12 27"
            )

        for seed in args.seeds:
            print("\n" + "=" * 70)
            print(f"Running experiments with RANDOM SEED = {seed}")
            print("=" * 70)

            for layer in args.layers:
                layer_results = []

                print("\n#######################################")
                print(f"Running Layer {layer}")
                print("#######################################")

                for matrix_name in MATRICES.keys():
                    target_name = build_target_name(layer, matrix_name)

                    result = run_single_experiment(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        args=args,
                        target_name=target_name,
                        seed=seed,
                    )

                    result["layer"] = layer
                    result["matrix"] = matrix_name
                    layer_results.append(result)

                output_path = (
                    f"experiments/"
                    f"layer{layer}_"
                    f"{method_short_name(args.method)}_"
                    f"{ratio_short_name(args.prune_ratio)}"
                    f"_seed{seed}.json"
                )

                layer_summary = {
                    "model": args.model,
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "layer": layer,
                    "method": args.method,
                    "seed": seed if args.method == "random" else None,
                    "tile_size": args.tile_size,
                    "prune_ratio": args.prune_ratio,
                    "results": layer_results,
                }

                save_json(output_path, layer_summary)

    else:
        seed = args.seeds[0]

        result = run_single_experiment(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            args=args,
            target_name=args.target_name,
            seed=seed,
        )

        output_path = args.output
        if output_path is None:
            output_path = "experiments/tile_pruning_results.json"

        results = {
            "model": args.model,
            "dataset": args.dataset,
            "subset": args.subset,
            **result,
        }

        save_json(output_path, results)


if __name__ == "__main__":
    main()