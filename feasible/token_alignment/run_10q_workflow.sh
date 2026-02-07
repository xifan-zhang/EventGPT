#!/bin/bash
# Extract with 10 questions, train for 50 epochs (no early stopping), and evaluate
# Usage: nohup ./feasible/token_alignment/run_10q_workflow.sh > /tmp/workflow_10q.log 2>&1 &
set -e
cd /home/ps/Documents/code/EventGPT

echo "============================================================"
echo "Token Alignment: 10 Questions Workflow (50 epochs, no early stop)"
echo "Started: $(date)"
echo "============================================================"

TRAIN_JSON="./feasible/token_alignment/train_tokens_10q.json"
TEST_JSON="./feasible/token_alignment/test_tokens_10q.json"

# Wait for extraction to complete if running
if pgrep -f "extract_tokens_parallel.py.*max_questions 10" > /dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] Waiting for 10q extraction to complete..."
    while pgrep -f "extract_tokens_parallel.py.*max_questions 10" > /dev/null 2>&1; do
        progress=$(grep -oP 'Extracting:\s+\d+%\|[^|]+\|\s+\d+/52080' /tmp/extract_10q_parallel.log 2>/dev/null | tail -1)
        echo "[$(date '+%H:%M:%S')] $progress"
        sleep 120
    done
    echo "[$(date '+%H:%M:%S')] Extraction completed!"
fi

# Check extraction result
if [ -f "$TRAIN_JSON" ]; then
    success=$(python3 -c "import json; d=json.load(open('$TRAIN_JSON')); print(d.get('videollava_success', len(d.get('results', []))))")
    echo "[$(date '+%H:%M:%S')] Training extraction complete: $success pairs"
else
    echo "ERROR: Extraction file not found: $TRAIN_JSON"
    exit 1
fi

# Extract test tokens (10 questions on test set)
echo ""
echo "[$(date '+%H:%M:%S')] Extracting test tokens (10 questions)..."
python3 feasible/token_alignment/extract_tokens_parallel.py \
    --dataset_dir /home/ps/Documents/code/EventGPT/data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \
    --output_file "$TEST_JSON" \
    --max_questions 10 \
    --max_samples -1

test_success=$(python3 -c "import json; d=json.load(open('$TEST_JSON')); print(d.get('videollava_success', len(d.get('results', []))))")
echo "[$(date '+%H:%M:%S')] Test extraction: $test_success pairs"

# Train for full 50 epochs (no early stopping)
echo ""
echo "[$(date '+%H:%M:%S')] Starting training (50 epochs, no early stopping)..."
python3 feasible/token_alignment/train_and_evaluate.py \
    --train_benchmark "$TRAIN_JSON" \
    --test_benchmark "$TEST_JSON" \
    --task_name 10q \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --early_stopping 999

echo ""
echo "============================================================"
echo "Workflow Complete"
echo "Finished: $(date)"
echo "============================================================"

# Show results
LATEST=$(ls -td ./feasible/token_alignment/task/10q_* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo ""
    echo "Results from $LATEST:"
    cat "$LATEST/RESULTS.md" 2>/dev/null || echo "Results file not found"
fi
