# Prefill Acceleration Test Results

**Date:** 2026-01-29
**Samples:** 10
**Dataset:** my_egpt_dsec_seq_1s (test set)

## Summary

Using parallel prefill, EventGPT generates **26 free draft tokens** during Video-LLaVA's slower prefill phase. With the trained TokenAdapter (27.9% acceptance), **7 tokens are accepted**, saving **~100ms**.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 270.2 ms |
| **Free Draft Tokens** | 26 tokens |
| **Accepted Tokens** | 7 tokens |
| **Time Saved** | 101.1 ms |
| **Speedup** | 1.05x |

## Timing Comparison

### Per-Model Breakdown

| Model | Prefill | Decode | Tokens | Total |
|-------|---------|--------|--------|-------|
| **EventGPT** | 112.5 ms | 438.2 ms | 44 | 550.7 ms |
| **Video-LLaVA** | 382.8 ms | 722.0 ms | 50 | 1104.8 ms |
| **Speedup** | 3.4x | 1.6x | - | 2.0x |

### Token Generation Rate

| Model | Rate |
|-------|------|
| EventGPT | 99.7 tok/s (10.0 ms/tok) |
| Video-LLaVA | 69.3 tok/s (14.4 ms/tok) |

## Parallel Prefill Analysis

```
Timeline:

EventGPT:     |--Prefill (112ms)--|--------Decode (438ms)--------|
                                  |<-- 26 FREE tokens generated -->|

Video-LLaVA:  |----------Prefill (383ms)----------|----Decode (722ms)----|
                                                   ^
                                                   |
                                            Verify 26 drafts
                                            Accept 7 tokens
```

### Overlap Window Calculation

```
Overlap = VL_prefill - EGPT_prefill
        = 382.8ms - 112.5ms
        = 270.2ms

Free tokens = Overlap × EGPT_token_rate
            = 270.2ms × 99.7 tok/s
            = 26 tokens
```

## Speedup Analysis

### Approaches Compared

| Approach | Wall-Clock Time | Speedup | Notes |
|----------|-----------------|---------|-------|
| VL Baseline | 1104.8 ms | 1.00x | Video-LLaVA only |
| Parallel (no SD) | 1104.8 ms | 1.00x | No verification benefit |
| **Parallel + SD** | **1053.7 ms** | **1.05x** | With batch verification |

### Speedup Breakdown

```
Time saved = Accepted_tokens × VL_token_time
           = 7 × 14.4ms
           = 101.1ms

Parallel + SD time = VL_prefill + Verify + (VL_decode - Time_saved)
                   = 382.8ms + 50ms + (722ms - 101.1ms)
                   = 1053.7ms
```

## TokenAdapter Performance

Using the trained adapter from `1q_20260128_151847`:

| Metric | Value |
|--------|-------|
| Baseline Acceptance | 1.58% |
| **Model Acceptance** | **27.9%** |
| Top-5 Accuracy | 51.64% |
| Improvement | +26.32% |

### Impact on Speedup

| Acceptance Rate | Accepted Tokens | Time Saved | Speedup |
|-----------------|-----------------|------------|---------|
| 1.58% (baseline) | 0 | 0 ms | 1.00x |
| **27.9% (current)** | **7** | **101 ms** | **1.05x** |
| 50% (target) | 13 | 187 ms | 1.17x |
| 70% (optimistic) | 18 | 260 ms | 1.24x |

## Conclusions

1. **Parallel prefill works**: 26 draft tokens generated for FREE during VL prefill
2. **Acceptance rate is the bottleneck**: 27.9% → only 7 tokens accepted
3. **Modest speedup (1.05x)**: Need higher acceptance for meaningful gains
4. **Most time in decode**: 722ms decode vs 383ms prefill

## Recommendations

To improve speedup:

1. **Increase acceptance rate to 50%+**
   - Train on more data (currently 5200 samples)
   - Use EAGLE fusion with hidden states
   - Fine-tune EventGPT LM head

2. **Target longer outputs**
   - Prefill benefit increases with output length
   - 100 tokens: potential 1.15x speedup

3. **Consider EventGPT-only for latency-critical**
   - EventGPT is 2.0x faster overall
   - Quality trade-off acceptable for some use cases

## Configuration

```
Models: 4-bit quantized
  - EventGPT-7b
  - Video-LLaVA-7B-hf

TokenAdapter:
  - Path: task/starred/1q_20260128_151847/best_model.pt
  - Parameters: 45M
  - Training: 5200 samples, 50 epochs

Hardware:
  - GPU: RTX 4090 24GB
  - VRAM: ~4.3GB per model (4-bit)
```

## Related Files

- Token Alignment Results: `1q_20260128_151847/RESULTS.md`
- Training Curves: `1q_20260128_151847/training_curves.png`
- Implementation: `feasible/egpt_prefill_only/prefill_then_verify.py`

---

*Generated: 2026-01-29*
