#!/bin/bash
# DyGenePT Precomputation Pipeline
# Run all three precomputation steps (or specify a single step)
#
# Usage:
#   bash scripts/run_precompute.sh              # run all steps
#   bash scripts/run_precompute.sh --step 1     # step 1 only: build gene text
#   bash scripts/run_precompute.sh --step 2     # step 2 only: LLM facet decomposition
#   bash scripts/run_precompute.sh --step 3     # step 3 only: BiomedBERT encoding
#
# Prerequisites:
#   - Set LLM_API_KEY environment variable for step 2
#   - Install dependencies: pip install -r requirements.txt
#   - GEARS dataset will be auto-downloaded on first run

set -e

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== DyGenePT Precomputation ==="
echo "Project root: $PROJECT_DIR"

# Check LLM_API_KEY
if [ -z "$LLM_API_KEY" ]; then
    echo "WARNING: LLM_API_KEY not set. Step 2 (facet decomposition) will fail."
fi

# Run precomputation
python -m src.precompute.run_precompute \
    --config configs/default.yaml \
    "$@"

echo "=== Precomputation finished ==="
