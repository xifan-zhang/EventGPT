#!/bin/bash
################################################################################
# Run 5-Stage Parallel Prefill Benchmark (4-bit)
################################################################################
#
# This script runs the comprehensive parallel prefill benchmark with:
# - All 5 stages timed separately for EventGPT and Video-LLaVA
# - Wall-clock time for parallel execution
# - Hidden tokens analysis (tokens generated during VL's slow prefill)
# - Full token output and text verification
# - 4-bit quantization for both 7B models
#
# OUTPUT:
#   - JSON results: results/parallel_prefill_5stages_{timestamp}.json
#   - Markdown report: results/parallel_prefill_5stages_{timestamp}.md
#
# USAGE:
#   ./run_5stages_benchmark.sh                    # Run all samples
#   ./run_5stages_benchmark.sh 10                 # Run 10 samples
#   ./run_5stages_benchmark.sh 50 100             # Run 50 samples, 100 tokens max
#
# AUTHOR: Alice Zhang
# DATE: 2026-01-27
################################################################################

set -e

# Configuration
DATASET_DIR="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s"
MAX_SAMPLES=${1:--1}  # First arg or -1 (all)
MAX_NEW_TOKENS=${2:-50}  # Second arg or 50

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================================================"
echo "  5-Stage Parallel Prefill Benchmark (4-bit Quantization)"
echo "================================================================================"
echo "Dataset:        $DATASET_DIR"
echo "Max samples:    $MAX_SAMPLES (-1 means all)"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Output dir:     $SCRIPT_DIR/results"
echo "================================================================================"
echo ""

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Run benchmark (with protobuf workaround for bitsandbytes)
cd "$(dirname "$SCRIPT_DIR")" && cd ..
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python "$SCRIPT_DIR/benchmark_parallel_prefill_5stages.py" \
    --dataset_dir "$DATASET_DIR" \
    --max_samples $MAX_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir "$SCRIPT_DIR/results"

echo ""
echo "================================================================================"
echo "  Benchmark Complete!"
echo "================================================================================"
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
