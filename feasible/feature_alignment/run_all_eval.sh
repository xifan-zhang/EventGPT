#!/bin/bash
# Run token-level evaluation on all trained adapters
# Each adapter runs as a separate process to avoid OOM

BASE=/home/ps/Documents/code/EventGPT/feasible/feature_alignment
TEST_DATA=$BASE/data/chunked_test_1s_4bit
LM_HEAD=$BASE/data/vl_lm_head.pt
SCRIPT=$BASE/measure_feature_acceptance.py

echo "=== Starting evaluation pipeline ==="
echo "Test data: $TEST_DATA (1s test set, 11K samples, 10 questions each)"
echo "LM head: $LM_HEAD"
echo ""

# L1-L4: Standard adapters
for LEVEL in L1 L2 L3 L4; do
    CKPT=$(find $BASE/tasks/$LEVEL -name "best_model.pt" | head -1)
    echo "=============================================="
    echo "Evaluating $LEVEL: $CKPT"
    echo "=============================================="
    conda run -n egpt python $SCRIPT --checkpoint $CKPT --test_data $TEST_DATA --lm_head $LM_HEAD --batch_size 64
    echo ""
    echo "$LEVEL done."
    echo ""
done

# L5: EAGLE
CKPT=$(find $BASE/tasks/L5 -name "best_model.pt" | head -1)
echo "=============================================="
echo "Evaluating L5: $CKPT"
echo "=============================================="
conda run -n egpt python $SCRIPT --checkpoint $CKPT --test_data $TEST_DATA --lm_head $LM_HEAD --batch_size 64
echo ""
echo "L5 done."
echo ""

# B1: VLM-only
CKPT=$(find $BASE/tasks/B1 -name "best_model.pt" | head -1)
echo "=============================================="
echo "Evaluating B1 (vlm_only): $CKPT"
echo "=============================================="
conda run -n egpt python $SCRIPT --checkpoint $CKPT --test_data $TEST_DATA --lm_head $LM_HEAD --batch_size 64 --vlm_only
echo ""
echo "B1 done."
echo ""

# L5F: Fused EAGLE
CKPT=$(find $BASE/tasks/L5F -name "best_model.pt" | head -1)
echo "=============================================="
echo "Evaluating L5F: $CKPT"
echo "=============================================="
conda run -n egpt python $SCRIPT --checkpoint $CKPT --test_data $TEST_DATA --lm_head $LM_HEAD --batch_size 64
echo ""
echo "L5F done."
echo ""

echo "=== All evaluations complete ==="
