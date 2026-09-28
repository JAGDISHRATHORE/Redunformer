import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from redundancy.models import load_model
from redundancy.data import load_wikitext
from redundancy.eval import eval_ppl

def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation for an LLM.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="Hugging Face model name")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1", help="Dataset subset")
    parser.add_argument("--output", type=str, default="experiments/new_runs/dense_baseline.json", help="Output file path")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    dataset = load_wikitext(args.dataset, args.subset, split="test")

    ppl = eval_ppl(model, tokenizer, dataset)

    print(f"\nFinal Perplexity: {ppl:.4f}")

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "subset": args.subset,
        "perplexity": ppl
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
