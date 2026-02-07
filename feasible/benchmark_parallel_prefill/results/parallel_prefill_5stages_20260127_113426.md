# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_4s

**Date:** 2026-01-27 11:42:46
**Samples:** 260
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 505.2 ms |
| **Hidden Tokens** | 44.8 tokens |
| **Parallel Speedup** | 1.42x |
| **Acceptance Rate** | 35.6% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.6 ± 0.2 ms | 224.9 ± 55.9 ms | 137.24x |
| **Stage 2: Preprocessing** | 3.6 ± 0.2 ms | 61.9 ± 1.1 ms | 17.17x |
| **Stage 3: Vision Encoding** | 8.4 ± 6.6 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 83.1 ± 2.9 ms | 315.1 ± 1.8 ms | 3.79x |
| **Stage 5: LLM Decode** | 456.2 ± 49.0 ms | 719.1 ± 1.7 ms | 1.58x |
| **TOTAL** | 553.0 ms | 1321.1 ms | 2.39x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 96.7 ms | 637 tokens | 6586 tok/s |
| Video-LLaVA | 601.9 ms | 4124 tokens | 6851 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 45.8 | 100.5 | 456.2 ms |
| Video-LLaVA | 50.0 | 69.5 | 719.1 ms |

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
                     (505 ms)
                     (45 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 96.7 ms |
| Video-LLaVA Prefill Complete | 601.9 ms |
| **Overlap Window** | 505.2 ms |
| **Hidden Tokens** | 44.8 tokens |
| Wall-Clock Time (parallel) | 1321.1 ms |
| Sequential Time (baseline) | 1874.0 ms |
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

1. **Prefill Speedup**: EventGPT prefill is 6.22x faster than Video-LLaVA
2. **Overlap Window**: 505.2ms available for free draft token generation
3. **Hidden Tokens**: ~45 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 35.6% - drafts are partially aligned

## Implications for Speculative Decoding

With 45 hidden tokens and 35.6% acceptance rate:
- Expected accepted tokens per batch: 15.9
- Effective speedup potential: 16.94x

---

*Generated: 2026-01-27 11:42:46*
*Author: Alice Zhang*
