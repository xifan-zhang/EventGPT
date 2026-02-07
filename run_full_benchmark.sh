#!/bin/bash
#
# Run full parallel prefill benchmark on 1s testing set (1100 samples)
# Both models at 4-bit quantization
#

set -e

# Configuration
DATASET_DIR="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s"
MAX_SAMPLES=1100
MAX_NEW_TOKENS=100
QUANTIZATION="4bit"
DEVICE="cuda"

# Print header
echo "=========================================="
echo "Full Benchmark - 1s Testing Set"
echo "=========================================="
echo "Dataset: $DATASET_DIR"
echo "Samples: $MAX_SAMPLES"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Quantization: $QUANTIZATION"
echo "Device: $DEVICE"
echo "=========================================="
echo ""

# Run benchmark
python feasible/benchmark_parallel_prefill/benchmark_parallel_quantized.py \
    --dataset_dir "$DATASET_DIR" \
    --max_samples $MAX_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --quantization $QUANTIZATION \
    --device $DEVICE

# Exit with status of benchmark
