# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_10s

**Date:** 2026-01-27 21:24:59
**Samples:** 93
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 804.0 ms |
| **Hidden Tokens** | 44.3 tokens |
| **Parallel Speedup** | 1.34x |
| **Acceptance Rate** | 3.1% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.6 ± 0.2 ms | 523.5 ± 134.8 ms | 335.09x |
| **Stage 2: Preprocessing** | 3.7 ± 0.1 ms | 62.6 ± 0.8 ms | 16.96x |
| **Stage 3: Vision Encoding** | 9.2 ± 11.3 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 83.8 ± 5.1 ms | 316.0 ± 3.4 ms | 3.77x |
| **Stage 5: LLM Decode** | 445.4 ± 61.7 ms | 721.2 ± 3.1 ms | 1.62x |
| **TOTAL** | 543.6 ms | 1623.4 ms | 2.99x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 98.2 ms | 637 tokens | 6487 tok/s |
| Video-LLaVA | 902.2 ms | 4124 tokens | 4571 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 44.3 | 99.6 | 445.4 ms |
| Video-LLaVA | 50.0 | 69.3 | 721.2 ms |

## GPU Memory Usage (4-bit Quantization)

| Model | Model Size | Peak Inference Memory |
|-------|------------|----------------------|
| EventGPT | 3905 MB | 8658 MB |
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
                     (804 ms)
                     (44 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 98.2 ms |
| Video-LLaVA Prefill Complete | 902.2 ms |
| **Overlap Window** | 804.0 ms |
| **Hidden Tokens** | 44.3 tokens |
| Wall-Clock Time (parallel) | 1623.4 ms |
| Sequential Time (baseline) | 2167.0 ms |
| **Parallel Speedup** | 1.34x |

## Sample Output Verification

### Sample 0

**EventGPT** (43 tokens):
> In the scene, there is a car with its headlights on, positioned on a road. The car is moving towards the viewer, and there are trees lining both sides of the road.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a mountain in the background, and a stone wall on the side of the road. The car is moving along the road, and the mountain and...

---

### Sample 1

**EventGPT** (46 tokens):
> In the scene, there is a car with its headlights on, positioned on a road. The car is moving towards the viewer, and there are trees and grassy areas on both sides of the road.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, surrounded by a beautiful mountain landscape. The car is positioned in the middle of the road, and the road appears to be a tw...

---

### Sample 2

**EventGPT** (50 tokens):
> In the scene, there is a person standing in front of a large, closed garage door. The person is wearing a suit and tie. The garage door is positioned centrally, and there is a clear sky above.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a mountain in the background, and a foggy atmosphere. The car is moving along the road, and the mountain in the distance adds ...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 9.19x faster than Video-LLaVA
2. **Overlap Window**: 804.0ms available for free draft token generation
3. **Hidden Tokens**: ~44 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 3.1% - drafts are partially aligned

## Implications for Speculative Decoding

With 44 hidden tokens and 3.1% acceptance rate:
- Expected accepted tokens per batch: 1.4
- Effective speedup potential: 2.36x

---

*Generated: 2026-01-27 21:24:59*
*Author: Alice Zhang*
