# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_8s

**Date:** 2026-01-27 11:53:38
**Samples:** 117
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 692.0 ms |
| **Hidden Tokens** | 44.8 tokens |
| **Parallel Speedup** | 1.36x |
| **Acceptance Rate** | 34.5% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.6 ± 0.1 ms | 411.2 ± 100.1 ms | 261.26x |
| **Stage 2: Preprocessing** | 3.7 ± 0.1 ms | 61.9 ± 1.1 ms | 16.92x |
| **Stage 3: Vision Encoding** | 8.1 ± 1.5 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 83.0 ± 0.2 ms | 315.2 ± 2.7 ms | 3.80x |
| **Stage 5: LLM Decode** | 447.6 ± 53.7 ms | 721.9 ± 0.9 ms | 1.61x |
| **TOTAL** | 543.9 ms | 1510.3 ms | 2.78x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 96.4 ms | 637 tokens | 6610 tok/s |
| Video-LLaVA | 788.4 ms | 4124 tokens | 5231 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 44.8 | 100.2 | 447.6 ms |
| Video-LLaVA | 50.0 | 69.3 | 721.9 ms |

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
                     (692 ms)
                     (45 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 96.4 ms |
| Video-LLaVA Prefill Complete | 788.4 ms |
| **Overlap Window** | 692.0 ms |
| **Hidden Tokens** | 44.8 tokens |
| Wall-Clock Time (parallel) | 1510.3 ms |
| Sequential Time (baseline) | 2054.3 ms |
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

1. **Prefill Speedup**: EventGPT prefill is 8.18x faster than Video-LLaVA
2. **Overlap Window**: 692.0ms available for free draft token generation
3. **Hidden Tokens**: ~45 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 34.5% - drafts are partially aligned

## Implications for Speculative Decoding

With 45 hidden tokens and 34.5% acceptance rate:
- Expected accepted tokens per batch: 15.5
- Effective speedup potential: 16.46x

---

*Generated: 2026-01-27 11:53:38*
*Author: Alice Zhang*
