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
| `train_lora_adapter.py` | L6: LoRA finetune EventGPT decoder for VL alignment |
| `auto_train_pipeline.sh` | Automated training of all levels (L1→L2→...→L5F) sequentially |
| `retrain_L4_converge.sh` | L4 retraining script (300 epochs, proper cosine schedule) |
| `TEACHER_FORCED_VS_AR.md` | Comparison: teacher-forced forward pass vs AR generate() |

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
| **L6** | **LoRA** | **16.8M** | `QLoRA on decoder q,k,v,o` | **Modifies drafter model itself** |

### Module Classes (`hidden_adapter.py`)

| Class | Levels | Description |
|-------|--------|-------------|
| `BottleneckAdapter` | L1 | Linear down → GELU → Linear up + residual |
| `MultiLayerBottleneckAdapter` | L2, L3 | N stacked bottleneck blocks |
| `AttentionAdapter` | L4 | Multi-head self-attention + FFN + residual |
| `EAGLEStyleAdapter` | L5, B1 | Causal attention + FFN, dual loss (alignment + next-h prediction) |
| `FusedEAGLEAdapter` | L5F | Gated fusion `Gate(h_egpt, h_vl)` → EAGLE head |
| `LoRATrainer` (in `train_lora_adapter.py`) | L6 | QLoRA on decoder, teacher-forced forward, live model training |

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

Default output: `pipeline/adapter_train/tasks/{L1,L2,...,B1,L6}/`

### Train all levels automatically

```bash
bash pipeline/adapter_train/auto_train_pipeline.sh
```

### L6: LoRA finetune EventGPT decoder

```bash
# L6 QLoRA alignment (requires event images on disk, ~8-10 GB GPU)
python pipeline/adapter_train/train_lora_adapter.py \
    --existing_chunks $DATA/chunked_train_1s_4bit_1f \
    --dataset_dir data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
    --event_image_key event_image_1f \
    --lora_rank 16 --lora_alpha 32 \
    --num_epochs 20 --gradient_accumulation 16 --early_stopping 5
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

### L6 LoRA Training Details

| Component | Setting |
|-----------|---------|
| Base model | EventGPT-7B, 4-bit quantized (NF4 + double quant) |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` (all 32 decoder layers) |
| LoRA rank / alpha | 16 / 32 |
| Trainable params | 16.8M / 3.5B (0.48%) |
| Loss | `MSE(pred, vl_target) + 0.5 * CosineLoss(pred, vl_target)` |
| Forward pass | Teacher-forced (all tokens in one pass, causal mask) |
| Optimizer | AdamW, lr=1e-4, weight_decay=0.01 |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Gradient | Checkpointing enabled, accumulation=16 |
| Vision encoder | Fully frozen (no gradients) |

**Memory estimate (24GB 4090):**

| Component | Memory |
|-----------|--------|
| EventGPT-7B 4-bit | ~3.5 GB |
| LoRA adapters | ~67 MB |
| Optimizer states | ~134 MB |
| Activations + gradients (checkpointed) | ~4-6 GB |
| **Total** | **~8-10 GB** |

**L1-L5 vs L6 — Key Differences:**

| | L1-L5 (Post-hoc Adapter) | L6 (LoRA) |
|---|---|---|
| Training data | Pre-extracted `.pt` hidden states | Live event images from disk |
| Model in memory | No model needed (adapter only) | Full EventGPT in 4-bit |
| Forward pass | None (uses cached hidden states) | Teacher-forced through decoder |
| What changes | Separate adapter network | Decoder weights via LoRA |
| Inference | EGPT → adapter → aligned hidden | EGPT (with LoRA) → already aligned |
| Overhead at inference | Adapter forward pass (~1ms) | Zero (LoRA merged into weights) |

See [`TEACHER_FORCED_VS_AR.md`](TEACHER_FORCED_VS_AR.md) for a detailed comparison of teacher-forced forward pass vs autoregressive generation.

---

## CLI Reference (`train_lora_adapter.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--existing_chunks` | required | Chunked dir with VL hidden states |
| `--dataset_dir` | required | DSEC dataset with event images |
| `--questions_file` | `top50_questions.json` | Questions for prompting |
| `--event_image_key` | `event_image` | `event_image` or `event_image_1f` |
| `--lora_rank` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--learning_rate` | 1e-4 | AdamW learning rate |
| `--gradient_accumulation` | 16 | Effective batch = 1 * 16 = 16 |
| `--num_epochs` | 20 | Training epochs |
| `--early_stopping` | 5 | Patience (epochs without val improvement) |
| `--output_dir` | `tasks/L6` | Output directory |

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
├── L6/L6_YYYYMMDD_HHMMSS/
│   ├── best_model/             # PEFT LoRA adapter weights
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── final_model/            # Final LoRA adapter weights
│   ├── config.json
│   └── history.json
└── ...
```

---

## Results

### Offline Metrics (5f, initial training)

| Level | Params | Val Loss | cos_sim | Accept@0.90 |
|-------|--------|----------|---------|-------------|
| L1 | 2.1M | 1.2798 | 0.777 | 21.9% |
| L2 | 6.3M | 1.2787 | 0.779 | 23.2% |
| L3 | 16M | 1.2499 | 0.790 | 24.9% |
| L4 | 101M | 1.2458 | 0.791 | 24.8% |
| L5 | 103M | 1.3413 | 0.759 | 11.2% |
| B1 | 103M | 0.6812 | 0.912 | 61.2% |
| L5F | 170M | 0.7282 | 0.896 | 66.2% |

### L4 Retrained (300 epochs, cosine schedule)

| Setting | cos_sim | Accept@0.90 | E2E Speedup |
|---------|---------|-------------|-------------|
| L4 5f (retrained, 228ep) | 0.797 | 28.85% | 1.03x |
| L4 1f (retrained, 267ep) | 0.790 | 28.0% | 1.03x |

5f and 1f are essentially identical — single-frame EventGPT produces equivalent hidden states.

### L6 LoRA (in progress)

| Epoch | Train cos_sim | Val Accept@0.90 | Status |
|-------|---------------|-----------------|--------|
| 1 | 0.69 | 28.27% | Training... |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-10 | L6 LoRA: fix gradient checkpointing (`use_reentrant=False`), add I/O prefetching |
| 2026-02-10 | E2E benchmark: add `--event_image_key` for 1f images, fix flat adapter discovery |
| 2026-02-09 | 1f pipeline: extract 52K train + 11K test, retrain L4, eval, E2E benchmark |
| 2026-02-09 | Eval OOM fix: float16 streaming + early tensor freeing |
| 2026-02-08 | L4 retrained (300ep cosine schedule), starred checkpoint system |
| 2026-02-07 | L6 LoRA training script, teacher-forced forward pass |
| 2026-02-07 | Initial 4-stage pipeline with L1-L5F adapters |
