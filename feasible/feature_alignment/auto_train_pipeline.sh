#!/bin/bash
# Auto-run training pipeline: L1 → Eval → L2 → Eval → ...
# Created: 2026-02-06
# Run with: nohup bash feasible/feature_alignment/auto_train_pipeline.sh > feasible/feature_alignment/auto_train.log 2>&1 &

set -e  # Exit on error

# Configuration
CONDA_ENV="egpt"
TRAIN_DATA="/mnt/hdd/data/egpt/chunked_train_1s_4bit"
TEST_DATA="/mnt/hdd/data/egpt/hidden_states/chunked_test_1s_4bit"
CHECKPOINT_DIR="/home/ps/Documents/code/EventGPT/feasible/feature_alignment/tasks"
LOG_DIR="/home/ps/Documents/code/EventGPT/feasible/feature_alignment/logs"

# Create directories
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

# Activate conda
source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "=========================================="
echo "Auto Training Pipeline Started"
echo "Time: $(date)"
echo "=========================================="

# Function to wait for extraction to complete
wait_for_extraction() {
    echo "[$(date '+%H:%M:%S')] Waiting for test extraction to complete..."

    while true; do
        # Check if extraction process is still running
        if ! pgrep -f "extract_hidden_states.py.*test" > /dev/null; then
            echo "[$(date '+%H:%M:%S')] Extraction process not found, checking if complete..."

            # Check if index.json has all samples
            if [ -f "$TEST_DATA/index.json" ]; then
                TOTAL=$(python -c "import json; print(json.load(open('$TEST_DATA/index.json'))['total_samples'])")
                if [ "$TOTAL" -ge 10000 ]; then
                    echo "[$(date '+%H:%M:%S')] Extraction complete! Total samples: $TOTAL"
                    return 0
                fi
            fi

            echo "[$(date '+%H:%M:%S')] Extraction incomplete, waiting..."
        fi

        # Show progress
        if [ -f "$TEST_DATA/index.json" ]; then
            CURRENT=$(python -c "import json; print(json.load(open('$TEST_DATA/index.json'))['total_samples'])" 2>/dev/null || echo "0")
            echo "[$(date '+%H:%M:%S')] Progress: $CURRENT / 11000 samples"
        fi

        sleep 300  # Check every 5 minutes
    done
}

# Function to train adapter
train_adapter() {
    local LEVEL=$1
    local EPOCHS=$2
    local BATCH=$3

    echo ""
    echo "=========================================="
    echo "[$(date '+%H:%M:%S')] Training L${LEVEL} Adapter"
    echo "=========================================="

    python feasible/feature_alignment/train_hidden_adapter.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$TEST_DATA" \
        --adapter_level $LEVEL \
        --num_epochs $EPOCHS \
        --batch_size $BATCH \
        --output_dir "$CHECKPOINT_DIR/L${LEVEL}" \
        2>&1 | tee "$LOG_DIR/train_L${LEVEL}.log"

    echo "[$(date '+%H:%M:%S')] L${LEVEL} training complete!"
}

# Function to evaluate adapter
eval_adapter() {
    local LEVEL=$1

    echo ""
    echo "=========================================="
    echo "[$(date '+%H:%M:%S')] Evaluating L${LEVEL} Adapter"
    echo "=========================================="

    # Find the best checkpoint
    CKPT=$(ls -t "$CHECKPOINT_DIR/L${LEVEL}/"*"/best_model.pt" 2>/dev/null | head -1)

    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found for L${LEVEL}"
        return 1
    fi

    echo "Using checkpoint: $CKPT"

    python feasible/feature_alignment/measure_feature_acceptance.py \
        --checkpoint "$CKPT" \
        --test_data "$TEST_DATA" \
        --output_dir "$CHECKPOINT_DIR/L${LEVEL}/eval" \
        2>&1 | tee "$LOG_DIR/eval_L${LEVEL}.log"

    echo "[$(date '+%H:%M:%S')] L${LEVEL} evaluation complete!"
}

# ==================== MAIN PIPELINE ====================

# Step 0: Wait for extraction
wait_for_extraction

# Step 1: Train & Eval L1
train_adapter 1 50 64
eval_adapter 1

# Step 2: Train & Eval L2
train_adapter 2 30 64
eval_adapter 2

# Step 3: Train & Eval L3
train_adapter 3 20 32
eval_adapter 3

# Step 4: Train & Eval L4
train_adapter 4 15 16
eval_adapter 4

# Step 5: Train & Eval L5 (EAGLE)
train_adapter 5 10 16
eval_adapter 5

echo ""
echo "=========================================="
echo "ALL TRAINING COMPLETE!"
echo "Time: $(date)"
echo "=========================================="
echo ""
echo "Results saved in: $CHECKPOINT_DIR"
echo "Logs saved in: $LOG_DIR"
