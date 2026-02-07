#!/bin/bash
# Run token-level evaluation on all trained adapters
# Each adapter runs as a separate process to avoid OOM

REPO=/home/ps/Documents/code/EventGPT
ADAPTER_DIR=$REPO/pipeline/adapter_train/tasks
DATA_DIR=$REPO/pipeline/feature_extraction/data
LM_HEAD=$DATA_DIR/vl_lm_head.pt
SCRIPT=$REPO/pipeline/evaluation/measure_feature_acceptance.py
OUTPUT=$REPO/pipeline/evaluation/tasks

cd $REPO

echo "=== Starting evaluation pipeline ==="
echo "Adapter dir: $ADAPTER_DIR"
echo "Test data: $DATA_DIR/chunked_test_1s_4bit"
echo "Output: $OUTPUT"
echo ""

# L1-L4: Standard adapters
for LEVEL in L1 L2 L3 L4; do
    CKPT=$(find $ADAPTER_DIR/$LEVEL -name "best_model.pt" | head -1)
    if [ -z "$CKPT" ]; then
        echo "Skipping $LEVEL: no checkpoint found"
        continue
    fi
    echo "=============================================="
    echo "Evaluating $LEVEL: $CKPT"
    echo "=============================================="
    conda run -n egpt python $SCRIPT \
        --checkpoint $CKPT \
        --test_data $DATA_DIR/chunked_test_1s_4bit \
        --lm_head $LM_HEAD \
        --output_dir $OUTPUT/$LEVEL \
        --batch_size 64
    echo ""
    echo "$LEVEL done."
    echo ""
done

# L5: EAGLE
CKPT=$(find $ADAPTER_DIR/L5 -name "best_model.pt" | head -1)
if [ -n "$CKPT" ]; then
    echo "=============================================="
    echo "Evaluating L5: $CKPT"
    echo "=============================================="
    conda run -n egpt python $SCRIPT \
        --checkpoint $CKPT \
        --test_data $DATA_DIR/chunked_test_1s_4bit \
        --lm_head $LM_HEAD \
        --output_dir $OUTPUT/L5 \
        --batch_size 64
    echo "L5 done."
    echo ""
fi

# B1: VLM-only
CKPT=$(find $ADAPTER_DIR/B1 -name "best_model.pt" | head -1)
if [ -n "$CKPT" ]; then
    echo "=============================================="
    echo "Evaluating B1 (vlm_only): $CKPT"
    echo "=============================================="
    conda run -n egpt python $SCRIPT \
        --checkpoint $CKPT \
        --test_data $DATA_DIR/chunked_test_1s_4bit \
        --lm_head $LM_HEAD \
        --output_dir $OUTPUT/B1 \
        --batch_size 64 --vlm_only
    echo "B1 done."
    echo ""
fi

# L5F: Fused EAGLE
CKPT=$(find $ADAPTER_DIR/L5F -name "best_model.pt" | head -1)
if [ -n "$CKPT" ]; then
    echo "=============================================="
    echo "Evaluating L5F: $CKPT"
    echo "=============================================="
    conda run -n egpt python $SCRIPT \
        --checkpoint $CKPT \
        --test_data $DATA_DIR/chunked_test_1s_4bit \
        --lm_head $LM_HEAD \
        --output_dir $OUTPUT/L5F \
        --batch_size 64
    echo "L5F done."
    echo ""
fi

echo "=== All evaluations complete ==="
