#!/bin/bash
#
# Run acceptance rate benchmark on all EventGPT datasets
# Supports both 1s and 500ms testing datasets
#

set -e

# Default configuration
QUANTIZATION="4bit"
DEVICE="cuda"
GAMMA=5
MAX_NEW_TOKENS=512

# Parse command line arguments
DATASET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --1s|--dataset-1s)
            DATASET="1s"
            shift
            ;;
        --500ms|--dataset-500ms)
            DATASET="500ms"
            shift
            ;;
        --all)
            DATASET="all"
            shift
            ;;
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
            echo "Usage: $0 [--1s|--500ms|--all] [--test] [--gamma N]"
            exit 1
            ;;
    esac
done

# Default to 1s dataset if not specified
if [ -z "$DATASET" ]; then
    DATASET="1s"
fi

# Function to run benchmark on a dataset
run_benchmark() {
    local DATASET_DIR=$1
    local DATASET_NAME=$2
    local MAX_SAMPLES=$3

    echo "=========================================="
    echo "Running Benchmark: $DATASET_NAME"
    echo "=========================================="
    echo "Dataset: $DATASET_DIR"
    echo "Samples: $MAX_SAMPLES"
    echo "Max new tokens: $MAX_NEW_TOKENS"
    echo "Gamma: $GAMMA"
    echo "=========================================="
    echo ""

    python feasible/benchmark_parallel_prefill/benchmark_acceptance_rate.py \
        --dataset_dir "$DATASET_DIR" \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --gamma $GAMMA \
        --quantization $QUANTIZATION \
        --device $DEVICE

    echo ""
    echo "✓ $DATASET_NAME benchmark completed"
    echo ""
}

# Get dataset info
if [ "$DATASET" = "1s" ] || [ "$DATASET" = "all" ]; then
    DATASET_1S="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s"
    if [ -d "$DATASET_1S" ]; then
        SAMPLES_1S=$(python -c "import json; f=open('$DATASET_1S/EventGPT_Instruction_Subset.json'); d=json.load(f); print(len(d))" 2>/dev/null || echo "1100")
    else
        echo "Warning: 1s dataset not found at $DATASET_1S"
        SAMPLES_1S=0
    fi
else
    SAMPLES_1S=0
fi

if [ "$DATASET" = "500ms" ] || [ "$DATASET" = "all" ]; then
    DATASET_500MS="./data/my_egpt_dsec_test/my_egpt_dsec_seq_500ms"
    if [ -d "$DATASET_500MS" ]; then
        SAMPLES_500MS=$(python -c "import json; f=open('$DATASET_500MS/EventGPT_Instruction_Subset.json'); d=json.load(f); print(len(d))" 2>/dev/null || echo "1100")
    else
        echo "Warning: 500ms dataset not found at $DATASET_500MS"
        SAMPLES_500MS=0
    fi
else
    SAMPLES_500MS=0
fi

# Run benchmarks
if [ "$DATASET" = "all" ]; then
    if [ $SAMPLES_1S -gt 0 ]; then
        run_benchmark "$DATASET_1S" "1-Second Dataset" $SAMPLES_1S
    fi
    if [ $SAMPLES_500MS -gt 0 ]; then
        run_benchmark "$DATASET_500MS" "500ms Dataset" $SAMPLES_500MS
    fi
elif [ "$DATASET" = "1s" ]; then
    if [ $SAMPLES_1S -gt 0 ]; then
        run_benchmark "$DATASET_1S" "1-Second Dataset" $SAMPLES_1S
    else
        echo "Error: 1s dataset not found"
        exit 1
    fi
elif [ "$DATASET" = "500ms" ]; then
    if [ $SAMPLES_500MS -gt 0 ]; then
        run_benchmark "$DATASET_500MS" "500ms Dataset" $SAMPLES_500MS
    else
        echo "Error: 500ms dataset not found"
        exit 1
    fi
fi

echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="
