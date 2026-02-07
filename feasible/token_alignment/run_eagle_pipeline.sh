#!/bin/bash
# =============================================================================
# EAGLE Fusion Training Pipeline for Token-Level Alignment
# =============================================================================
#
# This pipeline improves token-level acceptance from ~3.4% to target 50%+
# for effective speculative decoding speedup.
#
# MEMORY REQUIREMENTS (RTX 4090 24GB):
# ------------------------------------
# Stage 1 (Extraction): ~14GB peak (one model at a time)
#   - EventGPT 4-bit: ~8GB
#   - Video-LLaVA 4-bit: ~10GB
#   - Models loaded SEQUENTIALLY, not together
#
# Stage 2 (Training): ~1-2GB (uses cached features only)
#   - No large models loaded
#   - Pure PyTorch training on tensors
#   - Batch size 16-32 comfortable
#
# Stage 3 (Evaluation): ~1GB (no models needed)
#
# TOTAL TIME ESTIMATE (1000 samples):
# ------------------------------------
# - Extraction: ~2-3 hours (includes model switching)
# - Training: ~30 minutes (50 epochs)
# - Evaluation: ~5 minutes
#
# =============================================================================

set -e

# Configuration
DATASET_DIR="${DATASET_DIR:-./data/my_egpt_dsec_train/my_egpt_dsec_train_1s}"
OUTPUT_BASE="${OUTPUT_BASE:-./feasible/token_alignment}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
DEVICE="${DEVICE:-cuda}"

# Derived paths
CACHED_DIR="${OUTPUT_BASE}/cached_outputs_1s"
CHECKPOINT_DIR="${OUTPUT_BASE}/checkpoints_1s"

echo "=============================================="
echo "EAGLE Fusion Training Pipeline"
echo "=============================================="
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_BASE"
echo "Max samples: $MAX_SAMPLES"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""

# Check GPU memory
echo "Checking GPU memory..."
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | head -1
echo ""

# =============================================================================
# Stage 1: Feature Extraction
# =============================================================================
echo "=============================================="
echo "Stage 1: Feature Extraction"
echo "=============================================="
echo "This stage extracts hidden states from both models."
echo "Memory: ~14GB peak (models loaded one at a time)"
echo ""

if [ -d "$CACHED_DIR" ] && [ -f "$CACHED_DIR/draft_tokens.pt" ]; then
    echo "Cached features found at $CACHED_DIR"
    echo "Skipping extraction. Delete the directory to re-extract."
else
    echo "Extracting features..."
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ${OUTPUT_BASE}/extract_features.py \
        --dataset_dir "$DATASET_DIR" \
        --output_dir "$CACHED_DIR" \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens 50 \
        --extract_hidden_states \
        --device $DEVICE

    echo "Feature extraction complete!"
fi
echo ""

# =============================================================================
# Stage 2: EAGLE Fusion Training
# =============================================================================
echo "=============================================="
echo "Stage 2: EAGLE Fusion Training"
echo "=============================================="
echo "Training lightweight fusion module on cached features."
echo "Memory: ~1-2GB (no large models loaded)"
echo ""

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ${OUTPUT_BASE}/train_eagle_fusion.py \
    --cached_dir "$CACHED_DIR" \
    --output_dir "$CHECKPOINT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --hidden_dim 1024 \
    --num_fusion_layers 2 \
    --num_attention_heads 8 \
    --device $DEVICE \
    --early_stopping 10

echo "Training complete!"
echo ""

# =============================================================================
# Stage 3: Evaluation
# =============================================================================
echo "=============================================="
echo "Stage 3: Evaluation"
echo "=============================================="
echo "Evaluating speculative decoding performance."
echo ""

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ${OUTPUT_BASE}/evaluate_speculative.py \
    --model_path "${CHECKPOINT_DIR}/best_model.pt" \
    --cached_dir "$CACHED_DIR" \
    --output_path "${CHECKPOINT_DIR}/evaluation_results.json" \
    --device $DEVICE

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - Model: ${CHECKPOINT_DIR}/best_model.pt"
echo "  - Metrics: ${CHECKPOINT_DIR}/final_metrics.json"
echo "  - Evaluation: ${CHECKPOINT_DIR}/evaluation_results.json"
echo ""
echo "Next steps if acceptance < 50%:"
echo "  1. Train on more data (increase MAX_SAMPLES)"
echo "  2. Extract logits and enable --use_kl_loss"
echo "  3. Increase model capacity (--hidden_dim 2048)"
echo "  4. Fine-tune EventGPT's LM head directly"
