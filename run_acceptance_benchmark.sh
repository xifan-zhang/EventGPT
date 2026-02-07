#!/bin/bash
#
# Run acceptance rate benchmark on full dataset
# Gamma = 5, 512 max_tokens, 1100 samples
#

set -e

# Configuration
DATASET_DIR="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s"
MAX_SAMPLES=1100
MAX_NEW_TOKENS=512
GAMMA=5
QUANTIZATION="4bit"
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MAX_SAMPLES=10
            MAX_NEW_TOKENS=100
            shift
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--gamma N]"
            exit 1
            ;;
    esac
done

# Print header
echo "=========================================="
echo "Acceptance Rate Benchmark - Full Dataset"
echo "=========================================="
echo "Dataset: $DATASET_DIR"
echo "Samples: $MAX_SAMPLES"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Gamma (batch size): $GAMMA"
echo "Quantization: $QUANTIZATION"
echo "=========================================="
echo ""

# Run benchmark
python feasible/benchmark_parallel_prefill/benchmark_acceptance_rate.py \
    --dataset_dir "$DATASET_DIR" \
    --max_samples $MAX_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --gamma $GAMMA \
    --quantization $QUANTIZATION \
    --device $DEVICE
