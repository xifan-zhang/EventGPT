# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_1s

**Date:** 2026-01-27 16:40:40
**Samples:** 1100
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 360.5 ms |
| **Hidden Tokens** | 35.3 tokens |
| **Parallel Speedup** | 1.48x |
| **Acceptance Rate** | 3.4% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.8 ± 0.1 ms | 79.3 ± 15.7 ms | 44.05x |
| **Stage 2: Preprocessing** | 3.6 ± 0.1 ms | 62.3 ± 1.5 ms | 17.28x |
| **Stage 3: Vision Encoding** | 8.1 ± 3.5 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 83.2 ± 1.8 ms | 315.5 ± 1.1 ms | 3.79x |
| **Stage 5: LLM Decode** | 472.9 ± 43.9 ms | 724.0 ± 1.6 ms | 1.53x |
| **TOTAL** | 569.6 ms | 1181.1 ms | 2.07x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 96.7 ms | 637 tokens | 6590 tok/s |
| Video-LLaVA | 457.1 ms | 4124 tokens | 9021 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 47.0 | 99.4 | 472.9 ms |
| Video-LLaVA | 50.0 | 69.1 | 724.0 ms |

## GPU Memory Usage (4-bit Quantization)

| Model | Model Size | Peak Inference Memory |
|-------|------------|----------------------|
| EventGPT | 3905 MB | 8661 MB |
| Video-LLaVA | 4048 MB | 10183 MB |
| **Total (both models)** | 7953 MB | - |

## Parallel Execution Analysis

### Token Hiding Opportunity

```
Timeline (Parallel Execution):

  EventGPT:  |--Prefill--|---Decode (hidden)---|---Decode (visible)---|
                         ^                      ^
  Video-LLaVA: |--------Prefill (slow)---------|--------Decode--------|

              |<------ Overlap Window -------->|
                     (360 ms)
                     (35 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 96.7 ms |
| Video-LLaVA Prefill Complete | 457.1 ms |
| **Overlap Window** | 360.5 ms |
| **Hidden Tokens** | 35.3 tokens |
| Wall-Clock Time (parallel) | 1181.1 ms |
| Sequential Time (baseline) | 1750.7 ms |
| **Parallel Speedup** | 1.48x |

## Sample Output Verification

### Sample 0

**EventGPT** (34 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a curb.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a person standing on the side of the road, and a beautiful view of the mountains in the background. The car is moving along th...

---

### Sample 1

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a stone wall on the side of the road, and a beautiful view of the ocean in the background. The car is moving along the road, a...

---

### Sample 2

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a stone wall on the side of the road, and a beautiful view of the mountains in the background. The car is moving along the roa...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 4.73x faster than Video-LLaVA
2. **Overlap Window**: 360.5ms available for free draft token generation
3. **Hidden Tokens**: ~35 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 3.4% - drafts are partially aligned

## Implications for Speculative Decoding

With 35 hidden tokens and 3.4% acceptance rate:
- Expected accepted tokens per batch: 1.2
- Effective speedup potential: 2.20x

---

*Generated: 2026-01-27 16:40:40*
*Author: Alice Zhang*
