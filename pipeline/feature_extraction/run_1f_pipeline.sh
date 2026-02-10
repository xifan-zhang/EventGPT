#!/bin/bash
# Chain: 1f train extraction → L4 retrain on 1f data
# Run after test extraction finishes
set -e

REPO=/home/ps/Documents/code/EventGPT
cd "$REPO"

DATA_5F=feasible/feature_alignment/data
DATA_1F=pipeline/feature_extraction/data

echo "=== Step 1: Extract 1f train hidden states (52K samples, ~12h) ==="
python pipeline/feature_extraction/extract_hidden_states_1f.py \
    --existing_chunks "$DATA_5F/chunked_train_1s_4bit" \
    --dataset_dir data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
    --split train --event_image_key event_image_1f --resume

echo ""
echo "=== Step 2: Retrain L4 on 1f data (300 epochs) ==="
python pipeline/adapter_train/train_hidden_adapter.py \
    --train_data "$DATA_1F/chunked_train_1s_4bit_1f" \
    --val_data "$DATA_1F/chunked_test_1s_4bit_1f" \
    --adapter_level 4 --num_epochs 300 --batch_size 64 \
    --early_stopping 50 \
    --output_dir pipeline/adapter_train/tasks/L4_1f

echo ""
echo "=== Done ==="
echo "Compare L4_1f vs L4 (5-frame) results"
