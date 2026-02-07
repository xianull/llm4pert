#!/bin/bash
# DyGenePT Cross-Validation Training Script
# For fair comparison with LangPert on Replogle K562/RPE1
#
# Usage:
#   bash scripts/run_cv.sh k562      # 5-fold CV on K562
#   bash scripts/run_cv.sh rpe1      # 5-fold CV on RPE1
#   bash scripts/run_cv.sh --config configs/custom.yaml  # custom config
#
# Prerequisites:
#   - Run precomputation first: bash scripts/run_precompute.sh
#   - scGPT checkpoint in checkpoints/scgpt/whole_human/
#   - Install dependencies: pip install -r requirements.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== DyGenePT Cross-Validation ==="
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

# Select config based on first argument
DATASET="${1:-k562}"
shift 2>/dev/null || true

case "$DATASET" in
    k562|K562)
        CONFIG="configs/replogle_k562.yaml"
        echo "Dataset: Replogle K562 Essential"
        ;;
    rpe1|RPE1)
        CONFIG="configs/replogle_rpe1.yaml"
        echo "Dataset: Replogle RPE1 Essential"
        ;;
    --config)
        CONFIG="$1"
        shift
        echo "Custom config: $CONFIG"
        ;;
    *)
        echo "Usage: $0 {k562|rpe1|--config <path>}"
        exit 1
        ;;
esac

echo "Config: $CONFIG"
echo ""

# Run cross-validation training
python -m src.train_cv \
    --config "$CONFIG" \
    "$@"

echo "=== Cross-validation finished ==="
