import argparse

def main():
    parser = argparse.ArgumentParser(description="Run pruning / intervention.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    print("Pruning script placeholder.")

if __name__ == "__main__":
    main()
