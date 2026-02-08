#!/bin/bash
# DyGenePT Training Script (single-GPU / multi-GPU)
#
# Usage:
#   bash scripts/run_train.sh                           # auto-detect GPUs
#   bash scripts/run_train.sh --config configs/large.yaml  # custom config
#   NGPU=4 bash scripts/run_train.sh                    # force 4 GPUs
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

# Detect GPU count
if [ -z "$NGPU" ]; then
    if command -v nvidia-smi &>/dev/null; then
        NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        NGPU=0
    fi
fi

CONFIG="configs/default.yaml"

# Parse --config from arguments if provided
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Config: $CONFIG"
echo "GPUs:   $NGPU"
echo ""

if [ "$NGPU" -gt 1 ]; then
    echo "Launching DDP training with $NGPU GPUs..."
    torchrun --nproc_per_node="$NGPU" \
        -m src.train \
        --config "$CONFIG" \
        "${EXTRA_ARGS[@]}"
else
    echo "Launching single-device training..."
    python -m src.train \
        --config "$CONFIG" \
        "${EXTRA_ARGS[@]}"
fi

echo "=== Training finished ==="
