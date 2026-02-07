#!/bin/bash
# Two-Phase Pipeline: L1+L5F, L2+L5F, L3+L5F, L4+L5F + baselines (B1-only, L5F-only)
BASE=/home/ps/Documents/code/EventGPT/feasible/feature_alignment
TEST=$BASE/data/chunked_test_1s_4bit
LM=$BASE/data/vl_lm_head.pt
SCRIPT=$BASE/eval_two_phase.py
L5F_CKPT=$(find $BASE/tasks/L5F -name "best_model.pt" | head -1)
B1_CKPT=$(find $BASE/tasks/B1 -name "best_model.pt" | head -1)

# ---- Two-Phase: L1-L4 (prefill) + L5F (decode) ----
for LEVEL in L1 L2 L3 L4; do
    PREFILL_CKPT=$(find $BASE/tasks/$LEVEL -name "best_model.pt" | head -1)
    echo "=============================================="
    echo "Two-Phase: $LEVEL (prefill) + L5F (decode)"
    echo "=============================================="
    conda run -n egpt python $SCRIPT --prefill_checkpoint $PREFILL_CKPT --decode_checkpoint $L5F_CKPT --test_data $TEST --lm_head $LM --batch_size 64 --gamma_decode 5 --label ${LEVEL}_L5F
    echo ""
    echo "$LEVEL+L5F done."
    echo ""
done

# ---- Baseline: B1-only (decode only, no prefill hiding) ----
echo "=============================================="
echo "Baseline: B1-only (no prefill hiding)"
echo "=============================================="
conda run -n egpt python $SCRIPT --no_prefill --decode_checkpoint $B1_CKPT --decode_vlm_only --test_data $TEST --lm_head $LM --batch_size 64 --gamma_decode 5 --label B1_only
echo ""
echo "B1-only done."
echo ""

# ---- Baseline: L5F-only (decode only, no prefill hiding) ----
echo "=============================================="
echo "Baseline: L5F-only (no prefill hiding)"
echo "=============================================="
conda run -n egpt python $SCRIPT --no_prefill --decode_checkpoint $L5F_CKPT --test_data $TEST --lm_head $LM --batch_size 64 --gamma_decode 5 --label L5F_only
echo ""
echo "L5F-only done."
echo ""

echo "=== All two-phase + baseline evaluations complete ==="
