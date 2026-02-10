#!/bin/bash
# Full sequential pipeline: 5f eval/benchmark → 1f extract/train/eval/benchmark
# All tasks share one GPU — must run sequentially
set -e

REPO=/home/ps/Documents/code/EventGPT
cd "$REPO"

DATA_5F=feasible/feature_alignment/data
DATA_1F=pipeline/feature_extraction/data
L4_5F_CKPT=pipeline/adapter_train/tasks/starred/L4_20260207_203245/best_model.pt
LM_HEAD=$DATA_5F/vl_lm_head.pt

echo "============================================"
echo "  Step 1/6: Evaluate retrained L4 (5f)"
echo "============================================"
python pipeline/evaluation/measure_feature_acceptance.py \
    --checkpoint "$L4_5F_CKPT" \
    --test_data "$DATA_5F/chunked_test_1s_4bit" \
    --lm_head "$LM_HEAD" \
    --output_dir pipeline/evaluation/tasks/L4_5f_retrained

echo ""
echo "============================================"
echo "  Step 2/6: E2E benchmark retrained L4 (5f)"
echo "============================================"
python pipeline/benchmark_e2e/benchmark_e2e_wallclock.py \
    --max_samples 1100 --max_new_tokens 30 \
    --configs "vl_baseline,L4+VL" \
    --adapter_dir feasible/feature_alignment/tasks

echo ""
echo "============================================"
echo "  Step 3/6: Extract 1f train hidden states"
echo "============================================"
python pipeline/feature_extraction/extract_hidden_states_1f.py \
    --existing_chunks "$DATA_5F/chunked_train_1s_4bit" \
    --dataset_dir data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
    --split train --event_image_key event_image_1f --resume

echo ""
echo "============================================"
echo "  Step 4/6: Retrain L4 on 1f data"
echo "============================================"
python pipeline/adapter_train/train_hidden_adapter.py \
    --train_data "$DATA_1F/chunked_train_1s_4bit_1f" \
    --val_data "$DATA_1F/chunked_test_1s_4bit_1f" \
    --adapter_level 4 --num_epochs 300 --batch_size 64 \
    --early_stopping 50 \
    --output_dir pipeline/adapter_train/tasks/L4_1f

echo ""
echo "============================================"
echo "  Step 5/6: Evaluate L4_1f (1f test data)"
echo "============================================"
# Find the latest L4_1f checkpoint
L4_1F_CKPT=$(ls -td pipeline/adapter_train/tasks/L4_1f/L4_*/best_model.pt 2>/dev/null | head -1)
if [ -z "$L4_1F_CKPT" ]; then
    echo "ERROR: L4_1f checkpoint not found"
    exit 1
fi
echo "Using checkpoint: $L4_1F_CKPT"

python pipeline/evaluation/measure_feature_acceptance.py \
    --checkpoint "$L4_1F_CKPT" \
    --test_data "$DATA_1F/chunked_test_1s_4bit_1f" \
    --lm_head "$LM_HEAD" \
    --output_dir pipeline/evaluation/tasks/L4_1f

echo ""
echo "============================================"
echo "  Step 6/6: E2E benchmark L4_1f"
echo "============================================"
python pipeline/benchmark_e2e/benchmark_e2e_wallclock.py \
    --max_samples 1100 --max_new_tokens 30 \
    --configs "vl_baseline,L4+VL" \
    --adapter_dir pipeline/adapter_train/tasks/L4_1f

echo ""
echo "============================================"
echo "  DONE — Compare 5f vs 1f results"
echo "============================================"
