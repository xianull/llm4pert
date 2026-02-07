#!/bin/bash
# DyGenePT Training Script
#
# Usage:
#   bash scripts/run_train.sh                              # 8-GPU DDP (default)
#   bash scripts/run_train.sh --config configs/custom.yaml # custom config
#   NGPU=1 bash scripts/run_train.sh                       # single GPU
#
# Prerequisites:
#   - Run precomputation first: bash scripts/run_precompute.sh
#   - scGPT checkpoint in checkpoints/scgpt/whole_human/
#   - Install dependencies: pip install -r requirements.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== DyGenePT Training ==="
echo "Project root: $PROJECT_DIR"

# Check scGPT checkpoint
if [ ! -f "checkpoints/scgpt/whole_human/best_model.pt" ]; then
    echo "ERROR: scGPT checkpoint not found at checkpoints/scgpt/whole_human/"
    echo "Please download from: https://github.com/bowang-lab/scGPT"
    exit 1
fi

# Check precomputed facets
if [ ! -f "data/gene_facet_embeddings.pt" ]; then
    echo "ERROR: Precomputed facet embeddings not found."
    echo "Please run: bash scripts/run_precompute.sh"
    exit 1
fi

# Number of GPUs (default: 8, override with NGPU=N)
NGPU=${NGPU:-8}

if [ "$NGPU" -gt 1 ]; then
    echo "Launching DDP training on $NGPU GPUs..."
    torchrun --standalone --nproc_per_node="$NGPU" \
        -m src.train \
        --config configs/default.yaml \
        "$@"
else
    echo "Launching single-GPU training..."
    python -m src.train \
        --config configs/default.yaml \
        "$@"
fi

echo "=== Training finished ==="
