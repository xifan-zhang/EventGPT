#!/bin/bash
# Token Alignment Pipeline
# Waits for GPU to be free, runs benchmark to collect token data, then trains alignment

set -e
cd /home/ps/Documents/code/EventGPT

echo "============================================"
echo "Token Alignment Pipeline"
echo "============================================"

# Step 1: Wait for GPU to be free
echo "[Step 1] Waiting for GPU to be free..."
while true; do
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_PROCS" -eq "0" ]; then
        echo "GPU is free!"
        break
    fi
    echo "  GPU still in use ($GPU_PROCS processes). Waiting 30s..."
    sleep 30
done

# Step 2: Run benchmark on 1s training set to collect token data
echo ""
echo "[Step 2] Running benchmark on 1s training set..."
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python feasible/benchmark_inference/benchmark_inference.py \
    --dataset_dir ./data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
    --output_json ./feasible/benchmark_inference/benchmark_results_S1_train.json \
    --use_video_llava \
    --max_samples 200 \
    --device cuda

echo ""
echo "Benchmark complete. Checking results..."
python -c "import json; data=json.load(open('feasible/benchmark_inference/benchmark_results_S1_train.json')); print(f'Samples: {len(data)}, With tokens: {sum(1 for d in data if d.get(\"egpt_token_ids\") and d.get(\"llava_token_ids\"))}')"

# Step 3: Train token alignment
echo ""
echo "[Step 3] Training token alignment..."
python feasible/token_alignment/train_token_alignment_1s.py \
    --benchmark_json ./feasible/benchmark_inference/benchmark_results_S1_train.json \
    --output_dir ./feasible/token_alignment/checkpoints_1s \
    --num_epochs 20 \
    --batch_size 32 \
    --device cuda

echo ""
echo "============================================"
echo "Pipeline Complete!"
echo "============================================"
