# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_1s

**Date:** 2026-01-27 01:31:50
**Samples:** 1100
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 364.0 ms |
| **Hidden Tokens** | 34.9 tokens |
| **Parallel Speedup** | 1.49x |
| **Acceptance Rate** | 36.3% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.8 ± 0.1 ms | 79.1 ± 15.6 ms | 43.86x |
| **Stage 2: Preprocessing** | 3.7 ± 0.1 ms | 63.2 ± 0.9 ms | 16.90x |
| **Stage 3: Vision Encoding** | 8.2 ± 3.6 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 84.3 ± 2.2 ms | 319.7 ± 4.9 ms | 3.79x |
| **Stage 5: LLM Decode** | 482.1 ± 44.7 ms | 732.6 ± 17.2 ms | 1.52x |
| **TOTAL** | 580.1 ms | 1194.6 ms | 2.06x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 98.0 ms | 637 tokens | 6498 tok/s |
| Video-LLaVA | 462.0 ms | 4124 tokens | 8926 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 47.0 | 97.6 | 482.1 ms |
| Video-LLaVA | 50.0 | 68.3 | 732.6 ms |

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
                     (364 ms)
                     (35 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 98.0 ms |
| Video-LLaVA Prefill Complete | 462.0 ms |
| **Overlap Window** | 364.0 ms |
| **Hidden Tokens** | 34.9 tokens |
| Wall-Clock Time (parallel) | 1194.6 ms |
| Sequential Time (baseline) | 1774.8 ms |
| **Parallel Speedup** | 1.49x |

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

1. **Prefill Speedup**: EventGPT prefill is 4.71x faster than Video-LLaVA
2. **Overlap Window**: 364.0ms available for free draft token generation
3. **Hidden Tokens**: ~35 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 36.3% - drafts are partially aligned

## Implications for Speculative Decoding

With 35 hidden tokens and 36.3% acceptance rate:
- Expected accepted tokens per batch: 12.7
- Effective speedup potential: 13.66x

---

*Generated: 2026-01-27 01:31:50*
*Author: Alice Zhang*
