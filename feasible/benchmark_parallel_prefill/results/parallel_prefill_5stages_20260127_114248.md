# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_5s

**Date:** 2026-01-27 11:49:23
**Samples:** 193
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 553.0 ms |
| **Hidden Tokens** | 45.4 tokens |
| **Parallel Speedup** | 1.40x |
| **Acceptance Rate** | 34.8% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.6 ± 0.2 ms | 273.7 ± 74.3 ms | 166.45x |
| **Stage 2: Preprocessing** | 3.6 ± 0.2 ms | 61.2 ± 0.7 ms | 16.93x |
| **Stage 3: Vision Encoding** | 8.6 ± 7.7 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 83.1 ± 3.4 ms | 315.1 ± 2.1 ms | 3.79x |
| **Stage 5: LLM Decode** | 456.1 ± 54.3 ms | 720.9 ± 1.0 ms | 1.58x |
| **TOTAL** | 553.1 ms | 1370.9 ms | 2.48x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 97.0 ms | 637 tokens | 6570 tok/s |
| Video-LLaVA | 649.9 ms | 4124 tokens | 6345 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 45.7 | 100.2 | 456.1 ms |
| Video-LLaVA | 50.0 | 69.4 | 720.9 ms |

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
                     (553 ms)
                     (45 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 97.0 ms |
| Video-LLaVA Prefill Complete | 649.9 ms |
| **Overlap Window** | 553.0 ms |
| **Hidden Tokens** | 45.4 tokens |
| Wall-Clock Time (parallel) | 1370.9 ms |
| Sequential Time (baseline) | 1923.9 ms |
| **Parallel Speedup** | 1.40x |

## Sample Output Verification

### Sample 0

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, surrounded by trees and mountains. The car is positioned in the middle of the road, and the road appears to be a two-way stree...

---

### Sample 1

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a stone wall on the side of the road, and a mountain in the background. The car is moving along the road, and the stone wall a...

---

### Sample 2

**EventGPT** (46 tokens):
> In the scene, there is a car with its headlights on, positioned on a road. The car is moving towards the viewer, and there are trees and grassy areas on both sides of the road.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a mountain in the background, and a person standing on the side of the road. The car is moving along the road, and the person ...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 6.70x faster than Video-LLaVA
2. **Overlap Window**: 553.0ms available for free draft token generation
3. **Hidden Tokens**: ~45 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 34.8% - drafts are partially aligned

## Implications for Speculative Decoding

With 45 hidden tokens and 34.8% acceptance rate:
- Expected accepted tokens per batch: 15.8
- Effective speedup potential: 16.79x

---

*Generated: 2026-01-27 11:49:23*
*Author: Alice Zhang*
