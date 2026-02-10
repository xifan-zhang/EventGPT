# Cross-Modal Speculative Decoding Pipeline

> Author: Alice Zhang
> Date: 2026-02-07

EventGPT (drafter) → Adapter → Video-LLaVA (verifier)

4-stage pipeline for cross-modal speculative decoding with prefill hiding.
All commands assume `$REPO` is the repo root:

```bash
export REPO=/home/ps/Documents/code/EventGPT
cd $REPO
```

---

## Overview

```
Stage 1: feature_extraction/     Stage 2: adapter_train/
  Extract paired hidden states     Train adapter (L1-L5, B1, L5F)
  from EventGPT + Video-LLaVA     on extracted hidden states

Stage 3: evaluation/             Stage 4: benchmark_e2e/
  Offline acceptance metrics       E2E wall-clock with real inference
  (cos_sim, accept rate, speedup)  on live models + actual GPU timing
```

---

## Data Flow

```
pipeline/feature_extraction/data/       → extracted hidden states + vl_lm_head.pt
pipeline/adapter_train/tasks/L4/...     → trained adapter checkpoints
pipeline/evaluation/tasks/L4/...        → acceptance metrics, plots
pipeline/benchmark_e2e/tasks/...        → E2E wall-clock results, JSON, plots
```

| Path | Description |
|------|-------------|
| `pipeline/feature_extraction/data/` | Extracted hidden states (chunked .pt files, ~100GB) |
| `pipeline/feature_extraction/data/vl_lm_head.pt` | VL LM head weights for token metrics |
| `pipeline/adapter_train/tasks/` | Trained adapter checkpoints + training curves |
| `pipeline/evaluation/tasks/` | Offline evaluation results |
| `pipeline/benchmark_e2e/tasks/` | E2E benchmark results |

### Input Datasets

| Path | Description |
|------|-------------|
| `$REPO/data/my_egpt_dsec_train/my_egpt_dsec_train_1s` | DSEC train (5,200 scenes) |
| `$REPO/data/my_egpt_dsec_test/my_egpt_dsec_seq_1s` | DSEC test (1,100 scenes) |

---

## Quick Start (Full Pipeline)

```bash
cd $REPO
DATA=pipeline/feature_extraction/data
TASKS=pipeline/adapter_train/tasks

# Stage 0: Extract LM head (one-time, ~2min)
python pipeline/feature_extraction/extract_vl_lm_head.py

# Stage 1: Extract hidden states (~24h train, ~5h test)
python pipeline/feature_extraction/extract_hidden_states.py --split train --chunked --quant 4bit
python pipeline/feature_extraction/extract_hidden_states.py --split test --chunked --quant 4bit

# Stage 2: Train L4 adapter (300 epochs, ~10h)
python pipeline/adapter_train/train_hidden_adapter.py \
    --train_data $DATA/chunked_train_1s_4bit \
    --val_data   $DATA/chunked_test_1s_4bit \
    --adapter_level 4 --num_epochs 300 --batch_size 64 --early_stopping 50

# Stage 3: Evaluate (~20min)
python pipeline/evaluation/measure_feature_acceptance.py \
    --checkpoint $TASKS/L4/L4_*/best_model.pt \
    --test_data  $DATA/chunked_test_1s_4bit \
    --lm_head    $DATA/vl_lm_head.pt

# Stage 4: E2E benchmark (~6h)
python pipeline/benchmark_e2e/benchmark_e2e_wallclock.py \
    --max_samples 1100 --max_new_tokens 30 \
    --configs "vl_baseline,L4+VL"
```

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
| B1 | VLM-only | 103M | Same as L5, `--vlm_only` | Upper bound (no gap) |
| L6 | LoRA | 16.8M | `QLoRA on decoder q,k,v,o` | Modifies drafter model itself |

---

## Results

### Offline Metrics (11K test samples)

| Level | Params | Val Loss | cos_sim | Accept@0.90 |
|-------|--------|----------|---------|-------------|
| L1 | 2.1M | 1.2798 | 0.777 | 21.9% |
| L2 | 6.3M | 1.2787 | 0.779 | 23.2% |
| L3 | 16M | 1.2499 | 0.790 | 24.9% |
| L4 | 101M | 1.2458 | 0.791 | 24.8% |
| L5 | 103M | 1.3413 | 0.759 | 11.2% |
| B1 | 103M | 0.6812 | 0.912 | 61.2% |
| L5F | 170M | 0.7282 | 0.896 | 66.2% |

B1 >> L5 confirms the cross-modal gap is the dominant bottleneck.

### E2E Wall-Clock (10,970 samples, L4 adapter)

**30-token generation (5f event images):**

| Config | Prefill (ms) | Decode (ms) | Total (ms) | Accept | Speedup |
|--------|-------------|------------|-----------|--------|---------|
| VL baseline | 317 | 419 | 736 | --- | 1.00x |
| L4+VL SD | 317 | 404 | 721 | 21.2% | 1.03x |

**512-token / EOS generation (1f event images):**

| Config | Prefill (ms) | Decode (ms) | Total (ms) | Accept | Speedup |
|--------|-------------|------------|-----------|--------|---------|
| VL baseline | 316 | 1086 | 1401 | --- | 1.00x |
| L4+VL SD | 316 | 1062 | 1378 | 23.7% | 1.03x |

### 5f vs 1f Comparison

1-frame EventGPT produces equivalent hidden states to 5-frame:

| Metric | 5f | 1f | Delta |
|--------|----|----|-------|
| cos_sim | 0.789 | 0.790 | +0.1% |
| Accept@0.90 | 27.8% | 28.0% | +0.2% |
| E2E Speedup | 1.03x | 1.03x | same |

### Speedup Analysis

EGPT generates ~22 tokens free during VL prefill, but VL only accepts ~5 of them.

| Component | Value |
|-----------|-------|
| Tokens saved per sample | 5.4 (4.4 accepted + 1 bonus) |
| Value of saved tokens | 5.4 x 14.0 ms = 75 ms |
| Verify batch cost | ~70 ms |
| **Net saving** | **~5 ms per sample** |

Root causes: (1) prefill is 43% of wall time, unchanged by SD; (2) 21% acceptance = 32% of samples slower with SD; (3) verify cost nearly cancels savings.

| Accept% | Speedup |
|---------|---------|
| 21% (current) | 1.01x |
| 50% | 1.14x |
| 70% | 1.25x |
| 100% | 1.47x |

---

## Directory Structure

```
pipeline/
├── README.md                              # This file
├── feature_extraction/                    # Stage 1
│   ├── README.md
│   ├── extract_vl_lm_head.py             # Extract VL LM head (one-time)
│   ├── extract_hidden_states.py           # Run both models, save paired hidden states
│   ├── monitor_extraction.sh              # Monitor extraction progress
│   └── data/                              # Output: hidden states + vl_lm_head.pt
├── adapter_train/                         # Stage 2
│   ├── README.md
│   ├── hidden_adapter.py                  # All adapter architectures
│   ├── train_hidden_adapter.py            # Training loop (L1-L5, B1, L5F)
│   ├── train_lora_adapter.py              # L6 LoRA training
│   ├── auto_train_pipeline.sh             # Train all levels automatically
│   ├── retrain_L4_converge.sh             # L4 retraining (300 epochs)
│   ├── TEACHER_FORCED_VS_AR.md            # Teacher-forced vs AR comparison
│   └── tasks/                             # Output: checkpoints + training curves
├── evaluation/                            # Stage 3
│   ├── README.md
│   ├── measure_feature_acceptance.py      # Acceptance metrics
│   ├── eval_two_phase.py                  # Two-phase pipeline eval
│   ├── run_all_eval.sh                    # Evaluate all adapters
│   ├── run_two_phase_eval.sh              # Evaluate all two-phase combos
│   └── tasks/                             # Output: metrics, plots
└── benchmark_e2e/                         # Stage 4
    ├── README.md
    ├── benchmark_e2e_wallclock.py          # Real inference benchmark
    └── tasks/                             # Output: JSON, summary, plots
```
