#!/bin/bash
#
# Quick test run (10 samples) for verifying the benchmark works
#

set -e

echo "=========================================="
echo "Quick Benchmark Test (10 samples)"
echo "=========================================="

python feasible/benchmark_parallel_prefill/benchmark_parallel_quantized.py \
    --dataset_dir "./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s" \
    --max_samples 10 \
    --max_new_tokens 50 \
    --quantization 4bit \
    --device cuda
