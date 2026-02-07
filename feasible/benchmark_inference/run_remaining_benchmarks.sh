#!/bin/bash
# Run remaining benchmarks for 8s, 10s, 16s, 20s durations (200 samples each)

set -e
cd /home/ps/Documents/code/EventGPT

source ~/anaconda3/etc/profile.d/conda.sh
conda activate egpt
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

LOG_DIR="/tmp"
DATA_DIR="/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_test"

echo "=========================================="
echo "Starting remaining benchmarks at $(date)"
echo "=========================================="

for duration in 8s 10s 16s 20s; do
    echo ""
    echo ">>> Starting $duration benchmark at $(date)"
    echo ""

    python feasible/benchmark_inference/benchmark_inference.py \
        --dataset_dir "$DATA_DIR/my_egpt_dsec_seq_$duration" \
        --output_json "$DATA_DIR/my_egpt_dsec_seq_$duration/benchmark_results.json" \
        --use_video_llava \
        --max_samples 200 \
        --device cuda 2>&1 | tee "$LOG_DIR/benchmark_${duration}.log"

    echo ""
    echo ">>> Completed $duration benchmark at $(date)"
    echo ""
done

echo "=========================================="
echo "All benchmarks completed at $(date)"
echo "=========================================="
