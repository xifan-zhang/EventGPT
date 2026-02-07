# Task 1: Hidden State Adapter Training (100 samples × 10 questions)

**Date:** 2026-01-30

## Configuration

| Parameter | Value |
|-----------|-------|
| Samples | 100 |
| Questions | 10 (top frequent) |
| Total pairs | 1000 |
| Hidden dim | 4096 |
| Bottleneck dim | 256 |
| Epochs | 50 |
| Batch size | 64 |

## Results

### Training Metrics (Final Epoch)
| Metric | Value |
|--------|-------|
| Train Loss | 1.38 |
| Val Loss | 1.42 |
| Cosine Similarity | 0.764 |

### Acceptance Rates
| Threshold | Rate |
|-----------|------|
| @0.80 | 39.5% |
| @0.85 | 24.2% |
| @0.90 | 19.5% |
| @0.95 | 15.0% |

### Consecutive Acceptance (threshold 0.90)
| Metric | Value |
|--------|-------|
| Mean tokens | **6.35** |
| Max tokens | 14 |

### Per-Position Acceptance (threshold 0.90)
| Position | Mean Cos | Accept Rate |
|----------|----------|-------------|
| 0 | 0.989 | **100%** |
| 1 | 0.985 | **100%** |
| 2 | 0.985 | **99.9%** |
| 3 | 0.901 | 69.2% |
| 4 | 0.901 | 65.3% |
| 5 | 0.887 | 55.6% |
| 6 | 0.879 | 50.4% |
| 7 | 0.839 | 42.3% |
| 8 | 0.825 | 26.4% |
| 9 | 0.782 | 18.2% |

## Comparison with Token-Level SD
| Method | Consecutive Accepts |
|--------|---------------------|
| Token-level | 0% |
| **Feature-level** | **6.35 tokens avg** |

## Speedup Estimate (γ=5)
| Mode | Speedup |
|------|---------|
| Token-level | 1.0x |
| **Feature consecutive** | **5.77x** |

## Files
- Checkpoint: `checkpoints/hidden_adapter/hidden_adapter_20260130_014220/best_model.pt`
- Hidden states: `hidden_states/hidden_states_100samples_10q.pt`
