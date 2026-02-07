# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_20s

**Date:** 2026-01-27 21:29:24
**Samples:** 38
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 1293.0 ms |
| **Hidden Tokens** | 42.9 tokens |
| **Parallel Speedup** | 1.26x |
| **Acceptance Rate** | 3.4% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.5 ± 0.4 ms | 1015.7 ± 253.0 ms | 664.14x |
| **Stage 2: Preprocessing** | 3.9 ± 0.4 ms | 62.5 ± 1.4 ms | 16.01x |
| **Stage 3: Vision Encoding** | 11.8 ± 17.9 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 92.6 ± 25.4 ms | 324.6 ± 45.4 ms | 3.51x |
| **Stage 5: LLM Decode** | 440.2 ± 84.2 ms | 767.4 ± 151.1 ms | 1.74x |
| **TOTAL** | 550.0 ms | 2170.2 ms | 3.95x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 109.8 ms | 637 tokens | 5802 tok/s |
| Video-LLaVA | 1402.8 ms | 4124 tokens | 2940 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 42.9 | 98.4 | 440.2 ms |
| Video-LLaVA | 50.0 | 66.7 | 767.4 ms |

## GPU Memory Usage (4-bit Quantization)

| Model | Model Size | Peak Inference Memory |
|-------|------------|----------------------|
| EventGPT | 3905 MB | 8657 MB |
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
                     (1293 ms)
                     (43 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 109.8 ms |
| Video-LLaVA Prefill Complete | 1402.8 ms |
| **Overlap Window** | 1293.0 ms |
| **Hidden Tokens** | 42.9 tokens |
| Wall-Clock Time (parallel) | 2170.2 ms |
| Sequential Time (baseline) | 2720.1 ms |
| **Parallel Speedup** | 1.26x |

## Sample Output Verification

### Sample 0

**EventGPT** (50 tokens):
> In the scene, there is a person standing in front of a large, closed garage door. The person is wearing a suit and tie. The garage door is positioned centrally, and there is a clear sky above.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a mountain in the background, and a foggy atmosphere. The car is moving along the road, and the mountain in the distance adds ...

---

### Sample 1

**EventGPT** (37 tokens):
> In the scene, there is a tennis ball positioned near a tennis racket. The tennis ball is located on the ground, and the racket is placed next to it.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, surrounded by trees and mountains. The car is moving at a moderate speed, and the road appears to be a two-way street. The car...

---

### Sample 2

**EventGPT** (32 tokens):
> In the scene, there is a person standing on a stage with a microphone in their mouth. The stage has a red curtain in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a curvy road, a mountain in the background, and a fence on the side of the road. The car is moving along the road, and the mountain in the dis...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 12.78x faster than Video-LLaVA
2. **Overlap Window**: 1293.0ms available for free draft token generation
3. **Hidden Tokens**: ~43 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 3.4% - drafts are partially aligned

## Implications for Speculative Decoding

With 43 hidden tokens and 3.4% acceptance rate:
- Expected accepted tokens per batch: 1.5
- Effective speedup potential: 2.46x

---

*Generated: 2026-01-27 21:29:24*
*Author: Alice Zhang*
