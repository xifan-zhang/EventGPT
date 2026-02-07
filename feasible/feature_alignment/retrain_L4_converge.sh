#!/bin/bash
# Retrain L4 adapter until convergence (300 epochs)
# Previous training only ran 50 epochs and didn't converge
set -e

cd /home/ps/Documents/code/EventGPT

echo "=== L4 Adapter Retraining (300 epochs, proper convergence) ==="
echo "Previous: 50 epochs, best_val_loss=1.2458, accept@0.90=24.8%"
echo "Fix: T_max=300 (was 100), early_stopping=50 (was 10)"
echo ""

python feasible/feature_alignment/train_hidden_adapter.py \
    --train_data ./feasible/feature_alignment/data/chunked_train_1s_4bit \
    --val_data ./feasible/feature_alignment/data/chunked_test_1s_4bit \
    --adapter_level 4 \
    --num_epochs 300 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --early_stopping 50 \
    --output_dir ./feasible/feature_alignment/tasks/L4

echo ""
echo "=== L4 Retraining Complete ==="
