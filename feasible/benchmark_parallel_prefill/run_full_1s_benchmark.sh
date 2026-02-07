#!/bin/bash
################################################################################
# Run 4-bit Parallel Prefill Benchmark on Full 1s Dataset
################################################################################
#
# This script runs the parallel prefill benchmark with acceptance rate measurement
# on the entire my_egpt_dsec_seq_1s dataset (1100 samples).
#
# OUTPUT:
#   - JSON results saved to: feasible/benchmark_parallel_prefill/results/
#   - Includes: timing metrics, acceptance rate, effective speedup
#
# USAGE:
#   ./run_full_1s_benchmark.sh
#
# ETA: ~2-3 hours for 1100 samples
#
# AUTHOR: Alice Zhang
# DATE: 2026-01-26
################################################################################

set -e

# Configuration
DATASET_DIR="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s"
QUANTIZATION="4bit"
MAX_SAMPLES=-1  # -1 means all samples
MAX_NEW_TOKENS=28
GAMMA=5

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================================================"
echo "  4-bit Parallel Prefill Benchmark - Full 1s Dataset"
echo "================================================================================"
echo "Dataset:        $DATASET_DIR"
echo "Quantization:   $QUANTIZATION"
echo "Max samples:    $MAX_SAMPLES (all)"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Gamma:          $GAMMA"
echo "Output dir:     $SCRIPT_DIR/results"
echo "================================================================================"
echo ""

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Run benchmark (with protobuf workaround)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python "$SCRIPT_DIR/benchmark_parallel_prefill_4bit.py" \
    --dataset_dir "$DATASET_DIR" \
    --max_samples $MAX_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --gamma $GAMMA \
    --quantization $QUANTIZATION \
    --output_dir "$SCRIPT_DIR/results"

echo ""
echo "================================================================================"
echo "  Benchmark Complete!"
echo "================================================================================"
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
