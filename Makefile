IMAGE ?= redunformer
CONTAINER ?= docker
PROJECT_DIR := $(CURDIR)
LIMIT ?= 0.01

ifeq ($(CONTAINER),podman)
GPU_FLAGS := --device nvidia.com/gpu=all
else
GPU_FLAGS := --gpus all
endif

RUN_FLAGS := --rm $(GPU_FLAGS) \
	-v $(PROJECT_DIR)/experiments:/app/experiments \
	-v redunformer_hf_cache:/root/.cache/huggingface

.PHONY: help build verify smoke smoke-fast qwen-small qwen shell

help:
	@echo "Targets (all eval runs use GPU):"
	@echo "  make build         Build container image"
	@echo "  make verify        Fast check (imports + configs, no model download)"
	@echo "  make smoke         gpt2 smoke test on GPU (limit=$(LIMIT))"
	@echo "  make smoke-fast    Alias for smoke"
	@echo "  make qwen-small    Qwen3-1.7B baseline on GPU"
	@echo "  make qwen          Qwen3-4B baseline on GPU"
	@echo "  make shell         Interactive shell in container with GPU"
	@echo ""
	@echo "Examples:"
	@echo "  make verify"
	@echo "  make smoke LIMIT=0.05"
	@echo "  make CONTAINER=podman qwen"

build:
	$(CONTAINER) build -t $(IMAGE) .

verify:
	$(CONTAINER) run $(RUN_FLAGS) $(IMAGE) python scripts/verify_setup.py

smoke smoke-fast:
	$(CONTAINER) run $(RUN_FLAGS) $(IMAGE) \
		python scripts/run_baseline.py --config configs/models/gpt2.yaml --limit $(LIMIT)

qwen-small:
	$(CONTAINER) run $(RUN_FLAGS) $(IMAGE) \
		python scripts/run_baseline.py --config configs/models/qwen3-1.7b.yaml

qwen:
	$(CONTAINER) run $(RUN_FLAGS) $(IMAGE) \
		python scripts/run_baseline.py --config configs/models/qwen3-4b.yaml

shell:
	$(CONTAINER) run -it $(RUN_FLAGS) --entrypoint bash $(IMAGE)
