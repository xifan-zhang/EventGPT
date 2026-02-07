# Stage 2: Adapter Training

> Author: Alice Zhang
> Date: 2026-02-07

Train lightweight adapters to align EventGPT decoder hidden states to Video-LLaVA hidden state space, enabling cross-modal speculative decoding.

```bash
export REPO=/home/ps/Documents/code/EventGPT
cd $REPO
DATA=pipeline/feature_extraction/data
```

---

## Files

| File | Description |
|------|-------------|
| `hidden_adapter.py` | All adapter architectures (L1-L5, B1, L5F) |
| `train_hidden_adapter.py` | Training loop with memory-efficient chunked data loading |
| `auto_train_pipeline.sh` | Automated training of all levels (L1→L2→...→L5F) sequentially |
| `retrain_L4_converge.sh` | L4 retraining script (300 epochs, proper cosine schedule) |

---

## Adapter Architectures

| Level | Name | Params | Architecture | Key Feature |
|-------|------|--------|-------------|-------------|
| L1 | Bottleneck | 2.1M | `4096→256→4096 + residual` | Simple, fast |
| L2 | Multi-Layer | 6.3M | `3x (4096→256→4096)` | Stacked nonlinearity |
| L3 | Wide | 16M | `3x (4096→1024→4096)` | Larger bottleneck |
| L4 | Attention | 101M | `Transformer + FFN + residual` | Token dependencies |
| L5 | EAGLE | 103M | `CausalAttn + FFN, dual loss` | Align + predict next h |
| L5F | Fused EAGLE | 170M | `Gate(h_egpt, h_vl) + L5` | Both input streams |
| B1 | VLM-only | 103M | Same as L5, `--vlm_only` | Upper bound (no cross-modal gap) |

### Module Classes (`hidden_adapter.py`)

| Class | Levels | Description |
|-------|--------|-------------|
| `BottleneckAdapter` | L1 | Linear down → GELU → Linear up + residual |
| `MultiLayerBottleneckAdapter` | L2, L3 | N stacked bottleneck blocks |
| `AttentionAdapter` | L4 | Multi-head self-attention + FFN + residual |
| `EAGLEStyleAdapter` | L5, B1 | Causal attention + FFN, dual loss (alignment + next-h prediction) |
| `FusedEAGLEAdapter` | L5F | Gated fusion `Gate(h_egpt, h_vl)` → EAGLE head |

---

## Usage

### Train individual adapter

```bash
# L4 Attention adapter (300 epochs, ~10h)
python pipeline/adapter_train/train_hidden_adapter.py \
    --train_data $DATA/chunked_train_1s_4bit \
    --val_data   $DATA/chunked_test_1s_4bit \
    --adapter_level 4 --num_epochs 300 --batch_size 64 --early_stopping 50

# L5 EAGLE cross-modal (100 epochs, ~12h)
python pipeline/adapter_train/train_hidden_adapter.py \
    --train_data $DATA/chunked_train_1s_4bit \
    --val_data   $DATA/chunked_test_1s_4bit \
    --adapter_level 5 --num_epochs 100 --batch_size 64

# B1 VLM-only baseline (upper bound, no cross-modal gap)
python pipeline/adapter_train/train_hidden_adapter.py \
    --train_data $DATA/chunked_train_1s_4bit \
    --val_data   $DATA/chunked_test_1s_4bit \
    --adapter_level 5 --vlm_only --num_epochs 50 --batch_size 64
```

Default output: `pipeline/adapter_train/tasks/{L1,L2,...,B1}/`

### Train all levels automatically

```bash
bash pipeline/adapter_train/auto_train_pipeline.sh
```

### Retrain L4 with proper convergence

```bash
bash pipeline/adapter_train/retrain_L4_converge.sh
```

---

## CLI Reference (`train_hidden_adapter.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--train_data` | required | Chunked dir or single `.pt` file |
| `--val_data` | none | Chunked dir or `.pt` (splits 10% from train if omitted) |
| `--adapter_level` | 1 | 1=Bottleneck, 2=MultiLayer, 3=Wide, 4=Attention, 5=EAGLE, 6=FusedEAGLE |
| `--num_epochs` | 50 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--learning_rate` | 1e-3 | AdamW learning rate |
| `--early_stopping` | 10 | Patience (epochs without val improvement) |
| `--vlm_only` | off | B1 mode: VL→VL input (upper bound, no cross-modal gap) |
| `--output_dir` | `pipeline/adapter_train/tasks/` | Output base directory |

---

## Training Details

| Component | Setting |
|-----------|---------|
| Loss | `MSE(aligned, target) + 0.5 * CosineLoss(aligned, target)` |
| Optimizer | AdamW, weight_decay=0.01 |
| Scheduler | CosineAnnealingLR (T_max=num_epochs, eta_min=1e-6) |
| Gradient clipping | max_norm=1.0 |
| Memory | ChunkedTrainLoader: 1 chunk (~1.6GB) at a time with background prefetch |

---

## Output

```
pipeline/adapter_train/tasks/
├── L1/L1_YYYYMMDD_HHMMSS/
│   ├── best_model.pt           # Best validation loss checkpoint
│   ├── final_model.pt          # Last epoch checkpoint
│   ├── config.json             # Training configuration
│   ├── history.json            # Per-epoch: train_loss, val_loss, cos_sim, accept rates
│   └── training_curves.png     # Loss, cos_sim, acceptance rate plots
├── L4/L4_YYYYMMDD_HHMMSS/
├── B1/B1_YYYYMMDD_HHMMSS/
└── ...
```

---

## Results

| Level | Params | Val Loss | cos_sim | Accept@0.90 |
|-------|--------|----------|---------|-------------|
| L1 | 2.1M | 1.2798 | 0.777 | 21.9% |
| L2 | 6.3M | 1.2787 | 0.779 | 23.2% |
| L3 | 16M | 1.2499 | 0.790 | 24.9% |
| L4 | 101M | 1.2458 | 0.791 | 24.8% |
| L5 | 103M | 1.3413 | 0.759 | 11.2% |
| B1 | 103M | 0.6812 | 0.912 | 61.2% |
| L5F | 170M | 0.7282 | 0.896 | 66.2% |
