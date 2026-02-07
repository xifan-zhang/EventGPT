#!/bin/bash
################################################################################
# Run Parallel Prefill 5-Stage Benchmark (4-bit)
################################################################################
#
# This script runs the parallel prefill benchmark comparing EventGPT and
# Video-LLaVA with 5-stage timing breakdown.
#
# KEY FEATURES:
#   - All 5 stages timed separately (data load, preprocess, vision, prefill, decode)
#   - Wall-clock time for parallel execution
#   - Hidden tokens analysis (tokens generated during VL prefill overlap)
#   - Full token output and text from both models
#
# OUTPUT:
#   - JSON results: results/parallel_5stages_{timestamp}.json
#   - Markdown report: results/parallel_5stages_{timestamp}.md
#
# USAGE:
#   ./run_parallel_5stages.sh              # Run with defaults (50 samples)
#   ./run_parallel_5stages.sh 100          # Run with 100 samples
#   ./run_parallel_5stages.sh -1           # Run all samples
#
# AUTHOR: Alice Zhang
# DATE: 2026-01-27
################################################################################

set -e

# Configuration
DATASET_DIR="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s"
MAX_SAMPLES=${1:-50}  # Default 50 samples, or use first argument
MAX_NEW_TOKENS=50

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================================================"
echo "  Parallel Prefill 5-Stage Benchmark"
echo "================================================================================"
echo "Dataset:          $DATASET_DIR"
echo "Max samples:      $MAX_SAMPLES"
echo "Max new tokens:   $MAX_NEW_TOKENS"
echo "Output dir:       $SCRIPT_DIR/results"
echo "================================================================================"
echo ""

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Run benchmark (with protobuf workaround for some environments)
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
