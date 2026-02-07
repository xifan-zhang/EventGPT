#!/bin/bash
# Run token-level evaluation on remaining adapters (L3-L5, B1, L5F)
# L1/L2 already completed

BASE=/home/ps/Documents/code/EventGPT/feasible/feature_alignment
TEST_DATA=$BASE/data/chunked_test_1s_4bit
LM_HEAD=$BASE/data/vl_lm_head.pt
SCRIPT=$BASE/measure_feature_acceptance.py

echo "=== Continuing evaluation (L3-L5, B1, L5F) ==="

for LEVEL in L3 L4; do
    CKPT=$(find $BASE/tasks/$LEVEL -name "best_model.pt" | head -1)
    echo "=== Evaluating $LEVEL ==="
    conda run -n egpt python $SCRIPT --checkpoint $CKPT --test_data $TEST_DATA --lm_head $LM_HEAD --batch_size 64
    echo "$LEVEL done."
done

CKPT=$(find $BASE/tasks/L5 -name "best_model.pt" | head -1)
echo "=== Evaluating L5 ==="
conda run -n egpt python $SCRIPT --checkpoint $CKPT --test_data $TEST_DATA --lm_head $LM_HEAD --batch_size 64
echo "L5 done."

CKPT=$(find $BASE/tasks/B1 -name "best_model.pt" | head -1)
echo "=== Evaluating B1 (vlm_only) ==="
conda run -n egpt python $SCRIPT --checkpoint $CKPT --test_data $TEST_DATA --lm_head $LM_HEAD --batch_size 64 --vlm_only
echo "B1 done."

CKPT=$(find $BASE/tasks/L5F -name "best_model.pt" | head -1)
echo "=== Evaluating L5F ==="
conda run -n egpt python $SCRIPT --checkpoint $CKPT --test_data $TEST_DATA --lm_head $LM_HEAD --batch_size 64
echo "L5F done."

echo "=== All remaining evaluations complete ==="
