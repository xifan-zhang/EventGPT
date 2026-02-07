#!/bin/bash
# =============================================================================
# Prefill-Only Benchmark Runner
# =============================================================================
# Runs the prefill-only benchmark that measures the benefit of parallel prefill
# without speculative decoding complexity.
#
# Usage:
#   ./run_prefill_only.sh [OPTIONS]
#
# Options:
#   --quick       Run quick test with 10 samples (default)
#   --full        Run full benchmark with all samples
#   --samples N   Run with N samples
#
# =============================================================================

set -e

cd "$(dirname "$0")/../.."

# Default values
MAX_SAMPLES=10
MAX_NEW_TOKENS=50
DATASET_DIR="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s"
OUTPUT_DIR="./feasible/egpt_prefill_only/results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MAX_SAMPLES=10
            shift
            ;;
        --full)
            MAX_SAMPLES=-1
            shift
            ;;
        --samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --dataset)
            DATASET_DIR="$2"
            shift 2
            ;;
        --tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "         PREFILL-ONLY BENCHMARK"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Dataset:         $DATASET_DIR"
echo "  Max samples:     $MAX_SAMPLES"
echo "  Max new tokens:  $MAX_NEW_TOKENS"
echo "  Output dir:      $OUTPUT_DIR"
echo ""
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# Run benchmark
python3 feasible/egpt_prefill_only/benchmark_prefill_only.py \
    --dataset_dir "$DATASET_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
