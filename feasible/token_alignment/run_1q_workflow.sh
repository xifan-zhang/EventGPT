#!/bin/bash
# Extract with 1 question, train, and evaluate
set -e
cd /home/ps/Documents/code/EventGPT

echo "============================================================"
echo "Token Alignment: 1 Question Workflow"
echo "Started: $(date)"
echo "============================================================"

TRAIN_JSON="./feasible/token_alignment/train_tokens_1q.json"
TEST_JSON="./feasible/token_alignment/test_tokens_1q.json"

# Wait for extraction to complete if running
if pgrep -f "extract_tokens_train.py" > /dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] Waiting for extraction to complete..."
    while pgrep -f "extract_tokens_train.py" > /dev/null 2>&1; do
        progress=$(tail -1 /tmp/extract_1q.log 2>/dev/null | grep -oE '[0-9]+/[0-9]+' | tail -1)
        echo "[$(date '+%H:%M:%S')] Progress: $progress"
        sleep 60
    done
fi

# Check extraction result
if [ -f "$TRAIN_JSON" ]; then
    success=$(python3 -c "import json; d=json.load(open('$TRAIN_JSON')); print(d.get('videollava_success', len(d.get('results', []))))")
    echo "[$(date '+%H:%M:%S')] Extraction complete: $success pairs"
else
    echo "ERROR: Extraction file not found: $TRAIN_JSON"
    exit 1
fi

# Extract test tokens (1s test set with same question)
echo ""
echo "[$(date '+%H:%M:%S')] Extracting test tokens..."
python3 feasible/token_alignment/extract_tokens_train.py \
    --dataset_dir /home/ps/Documents/code/EventGPT/data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \
    --output_file "$TEST_JSON" \
    --single_question \
    --max_samples -1

test_success=$(python3 -c "import json; d=json.load(open('$TEST_JSON')); print(d.get('videollava_success', len(d.get('results', []))))")
echo "[$(date '+%H:%M:%S')] Test extraction: $test_success pairs"

# Train
echo ""
echo "[$(date '+%H:%M:%S')] Starting training..."
python3 feasible/token_alignment/train_and_evaluate.py \
    --train_benchmark "$TRAIN_JSON" \
    --test_benchmark "$TEST_JSON" \
    --task_name 1q \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --early_stopping 10

echo ""
echo "============================================================"
echo "Workflow Complete"
echo "Finished: $(date)"
echo "============================================================"

# Show results
LATEST=$(ls -td ./feasible/token_alignment/task/1q_* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo ""
    echo "Results from $LATEST:"
    cat "$LATEST/RESULTS.md" 2>/dev/null || echo "Results file not found"
fi
