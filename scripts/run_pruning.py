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


def prune_random(weight, tile_size, prune_ratio, seed):
    tiles = get_full_tiles(weight, tile_size)

    rng = random.Random(seed)
    rng.shuffle(tiles)

    num_prune = int(len(tiles) * prune_ratio)
    tiles_to_prune = tiles[:num_prune]

    zero_tiles(weight, tiles_to_prune, tile_size)

    return len(tiles), num_prune


def main():
    parser = argparse.ArgumentParser(description="Run tile-level pruning experiment.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--prune-ratio", type=float, default=0.05)
    parser.add_argument("--method", type=str, choices=["magnitude", "random"], default="magnitude")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiments/tile_pruning_results.json")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    target_name = "model.layers.0.mlp.up_proj.weight"

    target_weight = None
    for name, param in model.named_parameters():
        if name == target_name:
            target_weight = param
            break

    if target_weight is None:
        raise ValueError(f"Could not find target weight: {target_name}")

    print(f"Pruning target matrix: {target_name}")
    print(f"Matrix shape: {target_weight.shape}")
    print(f"Tile size: {args.tile_size}")
    print(f"Prune ratio: {args.prune_ratio}")
    print(f"Method: {args.method}")

    if args.method == "magnitude":
        num_tiles, num_pruned = prune_lowest_magnitude(
            target_weight,
            args.tile_size,
            args.prune_ratio,
        )
    else:
        num_tiles, num_pruned = prune_random(
            target_weight,
            args.tile_size,
            args.prune_ratio,
            args.seed,
        )

    print(f"Total full tiles: {num_tiles}")
    print(f"Pruned tiles: {num_pruned}")

    dataset = load_evaluation_dataset(args.dataset, args.subset, split="test")
    ppl = evaluate_perplexity(model, tokenizer, dataset)

    print(f"\nFinal Perplexity after pruning: {ppl:.4f}")

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "subset": args.subset,
        "target_matrix": target_name,
        "tile_size": args.tile_size,
        "prune_ratio": args.prune_ratio,
        "num_tiles": num_tiles,
        "num_pruned": num_pruned,
        "method": args.method,
        "seed": args.seed if args.method == "random" else None,
        "perplexity": ppl,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()