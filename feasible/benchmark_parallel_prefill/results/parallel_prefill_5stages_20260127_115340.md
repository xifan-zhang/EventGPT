# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_10s

**Date:** 2026-01-27 11:57:13
**Samples:** 93
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 797.9 ms |
| **Hidden Tokens** | 44.3 tokens |
| **Parallel Speedup** | 1.34x |
| **Acceptance Rate** | 33.8% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.6 ± 0.2 ms | 518.7 ± 133.1 ms | 329.77x |
| **Stage 2: Preprocessing** | 3.6 ± 0.1 ms | 61.5 ± 1.0 ms | 16.91x |
| **Stage 3: Vision Encoding** | 8.9 ± 10.8 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 83.4 ± 5.6 ms | 315.3 ± 2.8 ms | 3.78x |
| **Stage 5: LLM Decode** | 445.6 ± 61.2 ms | 721.0 ± 2.0 ms | 1.62x |
| **TOTAL** | 543.1 ms | 1616.5 ms | 2.98x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 97.5 ms | 637 tokens | 6530 tok/s |
| Video-LLaVA | 895.5 ms | 4124 tokens | 4605 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 44.3 | 99.6 | 445.6 ms |
| Video-LLaVA | 50.0 | 69.3 | 721.0 ms |

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
                     (798 ms)
                     (44 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 97.5 ms |
| Video-LLaVA Prefill Complete | 895.5 ms |
| **Overlap Window** | 797.9 ms |
| **Hidden Tokens** | 44.3 tokens |
| Wall-Clock Time (parallel) | 1616.5 ms |
| Sequential Time (baseline) | 2159.7 ms |
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

1. **Prefill Speedup**: EventGPT prefill is 9.18x faster than Video-LLaVA
2. **Overlap Window**: 797.9ms available for free draft token generation
3. **Hidden Tokens**: ~44 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 33.8% - drafts are partially aligned

## Implications for Speculative Decoding

With 44 hidden tokens and 33.8% acceptance rate:
- Expected accepted tokens per batch: 15.0
- Effective speedup potential: 15.97x

---

*Generated: 2026-01-27 11:57:13*
*Author: Alice Zhang*
