#!/bin/bash
# Two-Phase Pipeline: L1+L5F, L2+L5F, L3+L5F, L4+L5F + baselines (B1-only, L5F-only)

REPO=/home/ps/Documents/code/EventGPT
ADAPTER_DIR=$REPO/pipeline/adapter_train/tasks
DATA_DIR=$REPO/pipeline/feature_extraction/data
LM_HEAD=$DATA_DIR/vl_lm_head.pt
SCRIPT=$REPO/pipeline/evaluation/eval_two_phase.py

L5F_CKPT=$(find $ADAPTER_DIR/L5F -name "best_model.pt" | head -1)
B1_CKPT=$(find $ADAPTER_DIR/B1 -name "best_model.pt" | head -1)

cd $REPO

# ---- Two-Phase: L1-L4 (prefill) + L5F (decode) ----
for LEVEL in L1 L2 L3 L4; do
    PREFILL_CKPT=$(find $ADAPTER_DIR/$LEVEL -name "best_model.pt" | head -1)
    if [ -z "$PREFILL_CKPT" ] || [ -z "$L5F_CKPT" ]; then
        echo "Skipping $LEVEL+L5F: missing checkpoint"
        continue
    fi
    echo "=============================================="
    echo "Two-Phase: $LEVEL (prefill) + L5F (decode)"
    echo "=============================================="
    conda run -n egpt python $SCRIPT \
        --prefill_checkpoint $PREFILL_CKPT \
        --decode_checkpoint $L5F_CKPT \
        --test_data $DATA_DIR/chunked_test_1s_4bit \
        --lm_head $LM_HEAD \
        --batch_size 64 --gamma_decode 5 \
        --label ${LEVEL}_L5F
    echo "$LEVEL+L5F done."
    echo ""
done

# ---- Baseline: B1-only (decode only, no prefill hiding) ----
if [ -n "$B1_CKPT" ]; then
    echo "=============================================="
    echo "Baseline: B1-only (no prefill hiding)"
    echo "=============================================="
    conda run -n egpt python $SCRIPT \
        --no_prefill \
        --decode_checkpoint $B1_CKPT --decode_vlm_only \
        --test_data $DATA_DIR/chunked_test_1s_4bit \
        --lm_head $LM_HEAD \
        --batch_size 64 --gamma_decode 5 \
        --label B1_only
    echo "B1-only done."
    echo ""
fi

# ---- Baseline: L5F-only (decode only, no prefill hiding) ----
if [ -n "$L5F_CKPT" ]; then
    echo "=============================================="
    echo "Baseline: L5F-only (no prefill hiding)"
    echo "=============================================="
    conda run -n egpt python $SCRIPT \
        --no_prefill \
        --decode_checkpoint $L5F_CKPT \
        --test_data $DATA_DIR/chunked_test_1s_4bit \
        --lm_head $LM_HEAD \
        --batch_size 64 --gamma_decode 5 \
        --label L5F_only
    echo "L5F-only done."
    echo ""
fi

echo "=== All two-phase + baseline evaluations complete ==="
