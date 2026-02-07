#!/bin/bash
# Full Token Alignment Workflow: 1s dataset then 500ms dataset
# Usage: nohup ./run_full_workflow.sh > /tmp/full_workflow.log 2>&1 &

set -e
cd /home/ps/Documents/code/EventGPT

echo "============================================================"
echo "Token Alignment Full Workflow"
echo "Started: $(date)"
echo "============================================================"

# ============================================================
# PHASE 1: Complete 1s Dataset Workflow
# ============================================================

echo ""
echo "============================================================"
echo "PHASE 1: 1s Dataset"
echo "============================================================"

# Check if 1s extraction is still running
if pgrep -f "extract_tokens_train.py" > /dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] Waiting for 1s extraction to complete..."
    while pgrep -f "extract_tokens_train.py" > /dev/null 2>&1; do
        progress=$(tail -c 2000 /tmp/extract_train_tokens.log 2>/dev/null | grep -oE '(EventGPT|Video-LLaVA):[^EV]*' | tail -1)
        echo "[$(date '+%H:%M:%S')] $progress"
        sleep 60
    done
    echo "[$(date '+%H:%M:%S')] Extraction completed!"
fi

# Check 1s extraction results
TRAIN_1S_JSON="./feasible/token_alignment/train_tokens_full.json"
if [ -f "$TRAIN_1S_JSON" ]; then
    success_1s=$(python3 -c "import json; d=json.load(open('$TRAIN_1S_JSON')); print(d.get('videollava_success', len(d.get('results', []))))")
    echo "1s extraction: $success_1s successful samples"
else
    echo "ERROR: 1s extraction file not found: $TRAIN_1S_JSON"
    exit 1
fi

# Train on 1s dataset
if [ "$success_1s" -gt 0 ]; then
    echo ""
    echo "[$(date '+%H:%M:%S')] Starting 1s training..."

    # Auto-creates task folder: ./feasible/token_alignment/task/1s_YYYYMMDD_HHMMSS/
    python3 feasible/token_alignment/train_and_evaluate.py \
        --train_benchmark ./feasible/token_alignment/train_tokens_full.json \
        --test_benchmark ./feasible/benchmark_parallel_prefill/results/parallel_prefill_5stages_20260127_160820.json \
        --task_name 1s \
        --max_train_samples -1 \
        --num_epochs 50 \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --early_stopping 10

    echo "[$(date '+%H:%M:%S')] 1s training completed!"

    # Show results from latest 1s task folder
    LATEST_1S=$(ls -td ./feasible/token_alignment/task/1s_* 2>/dev/null | head -1)
    echo ""
    echo "1s Results (from $LATEST_1S):"
    cat "$LATEST_1S/RESULTS.md" 2>/dev/null || echo "Results file not found"
else
    echo "ERROR: No successful 1s samples"
    exit 1
fi

# ============================================================
# PHASE 2: 500ms Dataset Workflow
# ============================================================

echo ""
echo "============================================================"
echo "PHASE 2: 500ms Dataset"
echo "============================================================"

TRAIN_500MS_DIR="/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_train/my_egpt_dsec_train_500ms"
TEST_500MS_DIR="/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_test/my_egpt_dsec_seq_500ms"
TRAIN_500MS_JSON="./feasible/token_alignment/train_tokens_500ms.json"
TEST_500MS_JSON="./feasible/token_alignment/test_tokens_500ms.json"

# Step 2a: Extract 500ms training tokens
echo ""
echo "[$(date '+%H:%M:%S')] Extracting 500ms training tokens..."

python3 feasible/token_alignment/extract_tokens_train.py \
    --dataset_dir "$TRAIN_500MS_DIR" \
    --output_file "$TRAIN_500MS_JSON" \
    --max_samples -1 \
    --max_new_tokens 50

train_500ms_success=$(python3 -c "import json; d=json.load(open('$TRAIN_500MS_JSON')); print(d.get('videollava_success', len(d.get('results', []))))")
echo "[$(date '+%H:%M:%S')] 500ms train extraction: $train_500ms_success samples"

# Step 2b: Extract 500ms test tokens
echo ""
echo "[$(date '+%H:%M:%S')] Extracting 500ms test tokens..."

python3 feasible/token_alignment/extract_tokens_train.py \
    --dataset_dir "$TEST_500MS_DIR" \
    --output_file "$TEST_500MS_JSON" \
    --max_samples -1 \
    --max_new_tokens 50

test_500ms_success=$(python3 -c "import json; d=json.load(open('$TEST_500MS_JSON')); print(d.get('videollava_success', len(d.get('results', []))))")
echo "[$(date '+%H:%M:%S')] 500ms test extraction: $test_500ms_success samples"

# Step 2c: Train on 500ms dataset
if [ "$train_500ms_success" -gt 0 ] && [ "$test_500ms_success" -gt 0 ]; then
    echo ""
    echo "[$(date '+%H:%M:%S')] Starting 500ms training..."

    # Auto-creates task folder: ./feasible/token_alignment/task/500ms_YYYYMMDD_HHMMSS/
    python3 feasible/token_alignment/train_and_evaluate.py \
        --train_benchmark "$TRAIN_500MS_JSON" \
        --test_benchmark "$TEST_500MS_JSON" \
        --task_name 500ms \
        --max_train_samples -1 \
        --num_epochs 50 \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --early_stopping 10

    echo "[$(date '+%H:%M:%S')] 500ms training completed!"

    # Show results from latest 500ms task folder
    LATEST_500MS=$(ls -td ./feasible/token_alignment/task/500ms_* 2>/dev/null | head -1)
    echo ""
    echo "500ms Results (from $LATEST_500MS):"
    cat "$LATEST_500MS/RESULTS.md" 2>/dev/null || echo "Results file not found"
else
    echo "ERROR: 500ms extraction failed"
    exit 1
fi

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "============================================================"
echo "WORKFLOW COMPLETE"
echo "Finished: $(date)"
echo "============================================================"

echo ""
echo "Results Summary:"
echo "----------------"

# Find latest task folders
LATEST_1S=$(ls -td ./feasible/token_alignment/task/1s_* 2>/dev/null | head -1)
LATEST_500MS=$(ls -td ./feasible/token_alignment/task/500ms_* 2>/dev/null | head -1)

echo ""
echo "1s Dataset ($LATEST_1S):"
if [ -n "$LATEST_1S" ] && [ -f "$LATEST_1S/results.json" ]; then
    python3 -c "
import json
d = json.load(open('$LATEST_1S/results.json'))
train = d['final']['train']
test = d['final']['test']
bl_train = d['baseline']['train']['acceptance_rate']
bl_test = d['baseline']['test']['acceptance_rate']
print(f'  Train: {bl_train*100:.2f}% -> {train[\"acceptance_rate\"]*100:.2f}% (+{(train[\"acceptance_rate\"]-bl_train)*100:.2f}%)')
print(f'  Test:  {bl_test*100:.2f}% -> {test[\"acceptance_rate\"]*100:.2f}% (+{(test[\"acceptance_rate\"]-bl_test)*100:.2f}%)')
"
else
    echo "  Results not found"
fi

echo ""
echo "500ms Dataset ($LATEST_500MS):"
if [ -n "$LATEST_500MS" ] && [ -f "$LATEST_500MS/results.json" ]; then
    python3 -c "
import json
d = json.load(open('$LATEST_500MS/results.json'))
train = d['final']['train']
test = d['final']['test']
bl_train = d['baseline']['train']['acceptance_rate']
bl_test = d['baseline']['test']['acceptance_rate']
print(f'  Train: {bl_train*100:.2f}% -> {train[\"acceptance_rate\"]*100:.2f}% (+{(train[\"acceptance_rate\"]-bl_train)*100:.2f}%)')
print(f'  Test:  {bl_test*100:.2f}% -> {test[\"acceptance_rate\"]*100:.2f}% (+{(test[\"acceptance_rate\"]-bl_test)*100:.2f}%)')
"
else
    echo "  Results not found"
fi

echo ""
echo "Task folders:"
echo "  1s:    $LATEST_1S"
echo "  500ms: $LATEST_500MS"
echo ""
echo "All task folders:"
ls -la ./feasible/token_alignment/task/
