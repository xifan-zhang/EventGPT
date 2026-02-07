# Lightweight Adapter for EventGPT → Video-LLaVA Alignment

## Overview

This document explains how the lightweight adapter aligns EventGPT features to Video-LLaVA feature space, enabling cross-modal understanding and speculative decoding.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LIGHTWEIGHT ADAPTER PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

    Event Camera                                      RGB Video
         │                                                │
         ▼                                                ▼
┌─────────────────┐                            ┌─────────────────┐
│  Event Image    │                            │  Video Frame    │
│  (224×224×3)    │                            │  (224×224×3)    │
└────────┬────────┘                            └────────┬────────┘
         │                                              │
         │ Flatten                                      │ Flatten
         ▼                                              ▼
┌─────────────────┐                            ┌─────────────────┐
│  Event Features │                            │ Target Features │
│    (150,528)    │                            │    (150,528)    │
└────────┬────────┘                            └────────┬────────┘
         │                                              │
         │                                              │
         ▼                                              │
┌─────────────────────────────────────┐                │
│         LIGHTWEIGHT ADAPTER         │                │
│  ┌─────────────────────────────┐   │                │
│  │   Linear (150528 → 512)     │   │                │
│  │   LayerNorm + ReLU          │   │                │
│  │   Dropout (0.1)             │   │                │
│  │   Linear (512 → 512)        │   │                │
│  │   LayerNorm + ReLU          │   │                │
│  │   Linear (512 → 150528)     │   │                │
│  │   + Residual Connection     │   │                │
│  └─────────────────────────────┘   │                │
└────────────────┬────────────────────┘                │
                 │                                      │
                 ▼                                      ▼
         ┌───────────────────────────────────────────────┐
         │              COSINE SIMILARITY                │
         │         cos_sim(aligned, target)              │
         └───────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Acceptance Rate │
                    │   Estimation    │
                    └─────────────────┘
```

## Training Process

```
┌──────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                               │
└──────────────────────────────────────────────────────────────────┘

  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │   Epoch 1   │ ──▶ │   Epoch 2   │ ──▶ │   Epoch 3   │
  └─────────────┘     └─────────────┘     └─────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ Loss:   │         │ Loss:   │         │ Loss:   │
   │ ~0.44   │         │ ~0.05   │         │ ~0.04   │
   │ CosSim: │         │ CosSim: │         │ CosSim: │
   │ ~0.11   │         │ ~0.87   │         │ ~0.88   │
   └─────────┘         └─────────┘         └─────────┘

Loss Function:
  L = MSE_loss + 0.5 × (1 - cosine_similarity)
```

## Results (1s Duration Dataset)

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Samples | 5,208 |
| Train Samples | 4,166 |
| Val Samples | 1,042 |
| Feature Dimension | 150,528 |

### Alignment Improvement

```
BEFORE Alignment                    AFTER Alignment
─────────────────                   ────────────────
Mean Cosine Sim: 0.6477      ──▶    Mean Cosine Sim: 0.8828

                            Improvement: +0.2351 (+36.3%)
```

### Estimated Acceptance Rate

| Threshold | Before | After | Improvement |
|-----------|--------|-------|-------------|
| cos_sim > 0.5 | 92.3% | 99.9% | +7.6% |
| cos_sim > 0.7 | 30.9% | 96.2% | +65.3% |
| cos_sim > 0.8 | 2.3% | **86.8%** | +84.5% |
| cos_sim > 0.9 | 0.0% | 54.3% | +54.3% |
| cos_sim > 0.95 | 0.0% | 12.4% | +12.4% |

### Test Set Results (1,042 samples)

Evaluated on held-out validation set (20% of data, same random seed for reproducibility).

```
┌────────────────────────────────────────────────────────────────┐
│                    TEST SET PERFORMANCE                        │
└────────────────────────────────────────────────────────────────┘

  BEFORE                              AFTER
  ──────                              ─────
  Mean: 0.6461                        Mean: 0.8842
  Std:  0.1016                        Std:  0.0762
  Min:  0.275                         Min:  0.505
  Max:  0.861                         Max:  0.977

                    Improvement: +0.2381 (+36.8%)
```

#### Acceptance Rate (Test Set)

| Threshold | Before | After | Gain |
|-----------|--------|-------|------|
| > 0.5 | 91.7% | **100.0%** | +8.3% |
| > 0.6 | 73.7% | 99.5% | +25.8% |
| > 0.7 | 31.6% | 96.4% | +64.9% |
| > 0.8 | 2.2% | **85.7%** | +83.5% |
| > 0.85 | 0.2% | 77.5% | +77.4% |
| > 0.9 | 0.0% | 55.3% | +55.3% |
| > 0.95 | 0.0% | 13.1% | +13.1% |

#### Percentile Distribution (After Alignment)

```
Percentile    Cosine Similarity
──────────    ─────────────────
   P10             0.7645
   P25             0.8582
   P50 (Median)    0.9098
   P75             0.9382
   P90             0.9532
   P95             0.9599
   P99             0.9687
```

**Key Findings:**
- Model generalizes well (test ≈ train performance)
- No overfitting observed (std decreased from 0.10 → 0.08)
- Median sample achieves 0.91 cosine similarity
- 85.7% of test samples exceed 0.8 threshold

### Benchmark Results Files

| File | Description |
|------|-------------|
| `benchmark_adapter_full_S1.json` | Full benchmark with adapter (5,208 samples) |
| `benchmark_results_S1.json` | Model text outputs (EventGPT + LLaVA) |
| `alignment_metrics_S1.json` | Alignment metrics (test set only) |

**Full paths:**
```
feasible/benchmark_inference/benchmark_adapter_full_S1.json
feasible/benchmark_inference/benchmark_results_S1.json
feasible/benchmark_inference/alignment_metrics_S1.json
```

Run benchmarks:
```bash
# Full benchmark with adapter (recommended)
python feasible/benchmark_inference/benchmark_with_adapter_S1.py --device cuda

# Model text outputs
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python feasible/benchmark_inference/benchmark_inference.py \
    --dataset_dir ./data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
    --output_json ./feasible/benchmark_inference/benchmark_results_S1.json \
    --max_samples 10 --device cuda
```

### Full Benchmark Results (5,208 samples)

```
======================================================================
ACCEPTANCE RATE: EventGPT + Adapter vs Video-LLaVA
======================================================================

                        BEFORE          AFTER           GAIN
                        ------          -----           ----
Mean Cosine Sim:        0.6477          0.8828          +0.2351
Acceptance @ 0.8:       2.3%            86.8%           +84.5%
Acceptance @ 0.9:       0.0%            54.3%           +54.3%

Adapter Throughput: 48,249 samples/sec (0.02ms per sample)

CONCLUSION: ✓ Speculative decoding with EventGPT as draft is FEASIBLE
======================================================================
```

## Speculative Decoding Implications

```
┌─────────────────────────────────────────────────────────────────┐
│               SPECULATIVE DECODING WITH ADAPTER                 │
└─────────────────────────────────────────────────────────────────┘

                    Draft Model (EventGPT)
                            │
                            ▼
               ┌────────────────────────┐
               │    Event Features      │
               │  (fast, low latency)   │
               └───────────┬────────────┘
                           │
                           ▼
               ┌────────────────────────┐
               │   Lightweight Adapter  │◀── ~1ms overhead
               │   (MLP projection)     │
               └───────────┬────────────┘
                           │
                           ▼
               ┌────────────────────────┐
               │   Aligned Features     │
               │   (Video-LLaVA space)  │
               └───────────┬────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│ Target Model    │                 │   Verification  │
│ (Video-LLaVA)   │                 │   cos_sim > θ   │
└────────┬────────┘                 └────────┬────────┘
         │                                   │
         └───────────────┬───────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Accept/Reject      │
              │  Draft Tokens       │
              └─────────────────────┘

Expected acceptance rate at θ=0.8: ~87%
```

## Memory & Compute Requirements

| Component | Size/Time |
|-----------|-----------|
| Adapter Parameters | ~154M (MLP) |
| Checkpoint Size | ~1.8 GB |
| Training Memory | < 8 GB VRAM |
| Inference Latency | ~1 ms |
| Training Time (3 epochs) | ~2-3 minutes |

## Usage

### Training
```bash
python run_egpt_dsec_alignment.py \
    --strategy lightweight \
    --durations 1s \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-3
```

### Inference
```python
import torch
from feature_alignment import LightweightAlignmentModule

# Load trained adapter
checkpoint = torch.load('checkpoints/alignment_1s/lightweight_alignment.pt')
model = LightweightAlignmentModule(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Align features
with torch.no_grad():
    aligned_features = model(event_features)
```

## Limitations (Current Implementation)

1. **Raw Image Features**: Currently using flattened raw images (150,528 dim) instead of actual encoder features
2. **No Encoder Integration**: EventGPT and Video-LLaVA encoders not yet integrated
3. **High Dimensionality**: 150K dim features are expensive; proper encoder features (~768/1024 dim) would be much more efficient

## Next Steps

1. **Integrate Encoders**: Use EventGPT visual encoder + CLIP/Video-LLaVA encoder for proper feature extraction
2. **Reduce Dimensionality**: With proper encoders, features will be ~768→1024 dim (200x smaller)
3. **Fine-tune on Downstream**: Add KL divergence loss for speculative decoding task
4. **Benchmark Speed**: Measure actual speedup in speculative decoding pipeline

## References

- CEIA: Contrastive Event-Image Alignment
- Video-LLaVA: Learning United Visual Representation
- Speculative Decoding: Fast Inference from Transformers
