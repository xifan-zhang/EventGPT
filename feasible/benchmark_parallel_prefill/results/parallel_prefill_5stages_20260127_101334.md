# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_500ms

**Date:** 2026-01-27 11:17:56
**Samples:** 2220
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 335.0 ms |
| **Hidden Tokens** | 32.7 tokens |
| **Parallel Speedup** | 1.50x |
| **Acceptance Rate** | 35.9% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.8 ± 0.1 ms | 52.7 ± 8.3 ms | 28.69x |
| **Stage 2: Preprocessing** | 3.6 ± 0.1 ms | 62.3 ± 1.2 ms | 17.10x |
| **Stage 3: Vision Encoding** | 8.0 ± 2.6 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 83.5 ± 1.8 ms | 317.1 ± 4.2 ms | 3.80x |
| **Stage 5: LLM Decode** | 477.3 ± 43.8 ms | 724.8 ± 9.3 ms | 1.52x |
| **TOTAL** | 574.3 ms | 1156.8 ms | 2.01x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 97.0 ms | 637 tokens | 6566 tok/s |
| Video-LLaVA | 432.0 ms | 4124 tokens | 9546 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 47.4 | 99.3 | 477.3 ms |
| Video-LLaVA | 50.0 | 69.0 | 724.8 ms |

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
                     (335 ms)
                     (33 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 97.0 ms |
| Video-LLaVA Prefill Complete | 432.0 ms |
| **Overlap Window** | 335.0 ms |
| **Hidden Tokens** | 32.7 tokens |
| Wall-Clock Time (parallel) | 1156.8 ms |
| Sequential Time (baseline) | 1731.1 ms |
| **Parallel Speedup** | 1.50x |

## Sample Output Verification

### Sample 0

**EventGPT** (47 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a curb. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, surrounded by trees and mountains. The car is moving at a moderate speed, and the road appears to be a two-way street. The car...

---

### Sample 1

**EventGPT** (34 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a curb.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, surrounded by trees and mountains. The car is positioned in the middle of the road, and the road appears to be empty. The car ...

---

### Sample 2

**EventGPT** (47 tokens):
> In the scene, there is a car with its headlights on, positioned on a road. The road has a white line marking its edge. There are trees and grassy areas on both sides of the road.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a stone wall on the side of the road, and a beautiful view of the ocean in the background. The car is moving along the road, a...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 4.45x faster than Video-LLaVA
2. **Overlap Window**: 335.0ms available for free draft token generation
3. **Hidden Tokens**: ~33 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 35.9% - drafts are partially aligned

## Implications for Speculative Decoding

With 33 hidden tokens and 35.9% acceptance rate:
- Expected accepted tokens per batch: 11.8
- Effective speedup potential: 12.75x

---

*Generated: 2026-01-27 11:17:56*
*Author: Alice Zhang*
