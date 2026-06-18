from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import yaml

from redundancy.models import load_model_and_tokenizer


MODELS = {
    "1": ("Qwen/Qwen3-4B", "qwen3-4b"),
    "2": ("Qwen/Qwen3-1.7B", "qwen3-1.7b"),
    "3": ("gpt2", "gpt2"),
}


def choose_model():
    print("\nAvailable models:\n")

    print("1) Qwen3-4B")
    print("2) Qwen3-1.7B")
    print("3) GPT2")

    choice = input("\nSelect model: ").strip()

    if choice not in MODELS:
        raise ValueError("Invalid model selection")

    return MODELS[choice]


def discover_pruning_algorithms():
    pruning_dir = Path("src/redundancy/pruning")

    algorithms = []

    for file in sorted(pruning_dir.glob("*.py")):
        if file.name.startswith("__"):
            continue

        algorithms.append(file.stem)

    return algorithms


def choose_algorithm():
    algorithms = discover_pruning_algorithms()

    print("\nAvailable pruning algorithms:\n")

    for idx, algo in enumerate(algorithms, start=1):
        print(f"{idx}) {algo}")

    choice = int(input("\nSelect algorithm: "))

    return algorithms[choice - 1]


def load_pruning_function(algo_name):
    module = importlib.import_module(
        f"redundancy.pruning.{algo_name}"
    )

    for name in dir(module):
        if name.endswith("_prune_model"):
            return getattr(module, name)

    raise RuntimeError(
        f"No pruning function found in {algo_name}"
    )


def main():
    model_id, model_name = choose_model()

    algorithm = choose_algorithm()

    sparsity = float(
        input("\nSparsity (0-1): ").strip()
    )

    print("\nLoading model...\n")

    model, tokenizer = load_model_and_tokenizer(
        model_id,
        dtype="float32",
        device_map="cpu",
    )

    prune_fn = load_pruning_function(algorithm)

    print(
        f"\nRunning {algorithm} pruning...\n"
    )

    prune_fn(
        model=model,
        sparsity=sparsity,
    )

    percent = int(sparsity * 100)

    output_name = (
        f"{model_name}-{algorithm}{percent}"
    )

    output_dir = (
        f"experiments/pruned/{output_name}"
    )

    print(
        f"\nSaving model to {output_dir}\n"
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nModel saved.")

    config = {
        "name": output_name,
        "model": "hf",
        "pretrained": output_dir,
        "dtype": "bfloat16",
        "device_map": "auto",
        "tasks": [
            "hellaswag",
            "piqa",
            "arc_easy",
        ],
        "batch_size": "auto",
        "seed": 42,
        "output_dir": "experiments/baseline",
    }

    config_path = (
        f"configs/models/{output_name}.yaml"
    )

    Path("configs/models").mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
        )

    print(
        f"\nCreated config: {config_path}"
    )

    print(
        "\nStarting evaluation...\n"
    )

    subprocess.run(
        [
            "python",
            "scripts/run_baseline.py",
            "--config",
            config_path,
        ],
        check=True,
    )

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
