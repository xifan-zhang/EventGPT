# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_8s

**Date:** 2026-01-27 21:20:10
**Samples:** 117
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 698.7 ms |
| **Hidden Tokens** | 44.8 tokens |
| **Parallel Speedup** | 1.36x |
| **Acceptance Rate** | 3.4% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.6 ± 0.1 ms | 415.8 ± 101.4 ms | 261.10x |
| **Stage 2: Preprocessing** | 3.7 ± 0.1 ms | 63.2 ± 0.9 ms | 16.94x |
| **Stage 3: Vision Encoding** | 7.9 ± 1.1 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 83.3 ± 0.4 ms | 316.3 ± 3.2 ms | 3.80x |
| **Stage 5: LLM Decode** | 452.7 ± 54.3 ms | 726.1 ± 2.9 ms | 1.60x |
| **TOTAL** | 549.2 ms | 1521.3 ms | 2.77x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 96.5 ms | 637 tokens | 6599 tok/s |
| Video-LLaVA | 795.2 ms | 4124 tokens | 5186 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 44.8 | 99.1 | 452.7 ms |
| Video-LLaVA | 50.0 | 68.9 | 726.1 ms |

## GPU Memory Usage (4-bit Quantization)

| Model | Model Size | Peak Inference Memory |
|-------|------------|----------------------|
| EventGPT | 3905 MB | 8659 MB |
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
                     (699 ms)
                     (45 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 96.5 ms |
| Video-LLaVA Prefill Complete | 795.2 ms |
| **Overlap Window** | 698.7 ms |
| **Hidden Tokens** | 44.8 tokens |
| Wall-Clock Time (parallel) | 1521.3 ms |
| Sequential Time (baseline) | 2070.6 ms |
| **Parallel Speedup** | 1.36x |

## Sample Output Verification

### Sample 1

**EventGPT** (46 tokens):
> In the scene, there is a car with its headlights on, positioned on a road. The car is moving towards the viewer, and there are trees and grassy areas on both sides of the road.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, surrounded by a beautiful mountain landscape. The car is positioned in the middle of the road, and the road appears to be a tw...

---

### Sample 2

**EventGPT** (46 tokens):
> In the scene, there is a car with its headlights on, positioned on a road. The car is moving towards the viewer, and there are trees and grassy areas on both sides of the road.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, surrounded by a beautiful mountain landscape. The car is positioned in the middle of the road, and the road appears to be a tw...

---

### Sample 3

**EventGPT** (44 tokens):
> In the scene, there is a car with its headlights on, positioned on a road. The car is moving towards the viewer, and there are no other vehicles or pedestrians visible.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, surrounded by a beautiful mountain landscape. The car is positioned in the middle of the road, and the road appears to be a tw...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 8.24x faster than Video-LLaVA
2. **Overlap Window**: 698.7ms available for free draft token generation
3. **Hidden Tokens**: ~45 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 3.4% - drafts are partially aligned

## Implications for Speculative Decoding

With 45 hidden tokens and 3.4% acceptance rate:
- Expected accepted tokens per batch: 1.5
- Effective speedup potential: 2.50x

---

*Generated: 2026-01-27 21:20:10*
*Author: Alice Zhang*
