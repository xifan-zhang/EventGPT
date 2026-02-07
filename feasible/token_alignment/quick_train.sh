#!/bin/bash
# =============================================================================
# Quick Train - Run immediately with existing cached data
# =============================================================================
#
# MEMORY: ~500MB (very lightweight)
# TIME: ~10-15 minutes for 50 epochs
#
# This uses the TokenAdapter approach which works with just tokens (no hidden
# states needed). Expected improvement: 3% -> 15-25% acceptance.
#
# For higher accuracy (50%+), use the full EAGLE pipeline which requires
# extracting hidden states.
#
# =============================================================================

set -e

# Use existing cached data
CACHED_DIR="./feasible/token_alignment/cached_outputs_1s_train"
OUTPUT_DIR="./feasible/token_alignment/checkpoints_token_adapter"

echo "=============================================="
echo "Quick Train: Token Adapter"
echo "=============================================="
echo "Using cached data: $CACHED_DIR"
echo "Memory: ~500MB"
echo ""

# Check if cached data exists
if [ ! -f "$CACHED_DIR/draft_tokens.pt" ]; then
    echo "ERROR: Cached data not found at $CACHED_DIR"
    echo "Please run the full pipeline first to extract tokens."
    exit 1
fi

# Train
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ./feasible/token_alignment/token_adapter.py \
    --cached_dir "$CACHED_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --device cuda

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps for higher accuracy:"
echo "  1. Extract hidden states: ./run_eagle_pipeline.sh"
echo "  2. Train EAGLE fusion: train_eagle_fusion.py"
