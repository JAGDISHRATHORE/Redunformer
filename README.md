

## Setup

Use `uv` to manage the environment and dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv
source .venv/bin/activate
uv pip install -e .
python scripts/run_baseline.py --model Qwen/Qwen3-0.6B --dataset wikitext
```
