# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_4s

**Date:** 2026-01-27 21:07:34
**Samples:** 260
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 516.3 ms |
| **Hidden Tokens** | 44.4 tokens |
| **Parallel Speedup** | 1.42x |
| **Acceptance Rate** | 3.7% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.7 ± 0.2 ms | 228.7 ± 54.3 ms | 135.95x |
| **Stage 2: Preprocessing** | 3.8 ± 1.1 ms | 63.7 ± 1.7 ms | 16.64x |
| **Stage 3: Vision Encoding** | 8.9 ± 7.3 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 85.2 ± 3.4 ms | 323.5 ± 12.1 ms | 3.80x |
| **Stage 5: LLM Decode** | 476.0 ± 55.4 ms | 744.9 ± 25.5 ms | 1.56x |
| **TOTAL** | 575.7 ms | 1360.9 ms | 2.36x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 99.6 ms | 637 tokens | 6393 tok/s |
| Video-LLaVA | 615.9 ms | 4124 tokens | 6695 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 45.8 | 96.5 | 476.0 ms |
| Video-LLaVA | 50.0 | 67.2 | 744.9 ms |

## GPU Memory Usage (4-bit Quantization)

| Model | Model Size | Peak Inference Memory |
|-------|------------|----------------------|
| EventGPT | 3905 MB | 8660 MB |
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
                     (516 ms)
                     (44 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 99.6 ms |
| Video-LLaVA Prefill Complete | 615.9 ms |
| **Overlap Window** | 516.3 ms |
| **Hidden Tokens** | 44.4 tokens |
| Wall-Clock Time (parallel) | 1360.9 ms |
| Sequential Time (baseline) | 1936.5 ms |
| **Parallel Speedup** | 1.42x |

## Sample Output Verification

### Sample 0

**EventGPT** (50 tokens):
> In the scene, there is a road with a car driving away from the viewer. The road has a white line marking its edge and is bordered by grassy areas. There are buildings on either side of the road, and a...

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a person standing on the side of the road, and a beautiful view of the mountains in the background. The car is moving along th...

---

### Sample 1

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a stone wall on the side of the road, and a beautiful view of the mountains in the background. The car is moving along the roa...

---

### Sample 2

**EventGPT** (33 tokens):
> In the scene, there is a car with its headlights on, positioned on a road. The road has a white line marking its edge.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, surrounded by a beautiful mountain landscape. The car is positioned in the middle of the road, and the road appears to be a tw...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 6.18x faster than Video-LLaVA
2. **Overlap Window**: 516.3ms available for free draft token generation
3. **Hidden Tokens**: ~44 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 3.7% - drafts are partially aligned

## Implications for Speculative Decoding

With 44 hidden tokens and 3.7% acceptance rate:
- Expected accepted tokens per batch: 1.7
- Effective speedup potential: 2.65x

---

*Generated: 2026-01-27 21:07:34*
*Author: Alice Zhang*
