#!/bin/bash
# Run all remaining parallel prefill benchmarks with consistent settings
# Token-level acceptance rate: ON
# max_new_tokens: 50 (consistent with 1s dataset)
# Quantization: 4-bit NF4

set -e

cd /home/ps/Documents/code/EventGPT

DATASETS=(
    "my_egpt_dsec_seq_500ms:2220"
    "my_egpt_dsec_seq_2s:540"
    "my_egpt_dsec_seq_4s:260"
    "my_egpt_dsec_seq_5s:193"
    "my_egpt_dsec_seq_8s:117"
    "my_egpt_dsec_seq_10s:93"
    "my_egpt_dsec_seq_16s:23"
    "my_egpt_dsec_seq_20s:38"
)

echo "=========================================================================="
echo "RUNNING ALL REMAINING PARALLEL PREFILL BENCHMARKS"
echo "=========================================================================="
echo ""
echo "Settings:"
echo "  - max_new_tokens:        50 (consistent with 1s dataset)"
echo "  - Quantization:          4-bit NF4 + double quant"
echo "  - Acceptance Metric:     Token-level (target tokenizer)"
echo "  - Output Directory:      ./feasible/benchmark_parallel_prefill/results/"
echo ""

TOTAL_SAMPLES=0
for DATASET_INFO in "${DATASETS[@]}"; do
    DATASET="${DATASET_INFO%:*}"
    SAMPLES="${DATASET_INFO#*:}"
    TOTAL_SAMPLES=$((TOTAL_SAMPLES + SAMPLES))
done

echo "Datasets to process: ${#DATASETS[@]}"
echo "Total samples: $TOTAL_SAMPLES"
echo ""

START_TIME=$(date +%s)

for DATASET_INFO in "${DATASETS[@]}"; do
    DATASET="${DATASET_INFO%:*}"
    SAMPLES="${DATASET_INFO#*:}"
    DURATION="${DATASET#my_egpt_dsec_seq_}"

    echo "=========================================================================="
    echo "Dataset: $DURATION (${SAMPLES} samples)"
    echo "=========================================================================="
    echo "Command:"
    echo "  python feasible/benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py \\"
    echo "    --dataset_dir ./data/my_egpt_dsec_test/$DATASET \\"
    echo "    --max_samples -1 \\"
    echo "    --max_new_tokens 50"
    echo ""

    python feasible/benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py \
        --dataset_dir "./data/my_egpt_dsec_test/$DATASET" \
        --max_samples -1 \
        --max_new_tokens 50

    echo ""
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo "Elapsed time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"
    echo ""
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "=========================================================================="
echo "ALL BENCHMARKS COMPLETE"
echo "=========================================================================="
echo ""
echo "Summary:"
echo "  Total samples processed: $TOTAL_SAMPLES"
echo "  Total time elapsed: $((TOTAL_ELAPSED / 3600)) hours $((TOTAL_ELAPSED % 3600 / 60)) minutes"
echo "  Results saved to: ./feasible/benchmark_parallel_prefill/results/"
echo ""
echo "To analyze results, run:"
echo "  python /tmp/analyze_1s_results.py  # For latest benchmark"
echo ""
