# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_16s

**Date:** 2026-01-27 14:25:42
**Samples:** 23
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 994.3 ms |
| **Hidden Tokens** | 43.7 tokens |
| **Parallel Speedup** | 1.31x |
| **Acceptance Rate** | 33.3% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 15.2 ± 8.5 ms | 722.5 ± 104.5 ms | 47.52x |
| **Stage 2: Preprocessing** | 4.0 ± 0.6 ms | 62.9 ± 1.0 ms | 15.55x |
| **Stage 3: Vision Encoding** | 10.5 ± 3.6 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 85.6 ± 1.5 ms | 324.2 ± 2.9 ms | 3.79x |
| **Stage 5: LLM Decode** | 451.0 ± 53.7 ms | 743.9 ± 2.5 ms | 1.65x |
| **TOTAL** | 566.3 ms | 1853.5 ms | 3.27x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 115.3 ms | 637 tokens | 5525 tok/s |
| Video-LLaVA | 1109.6 ms | 4124 tokens | 3717 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 43.7 | 96.8 | 451.0 ms |
| Video-LLaVA | 50.0 | 67.2 | 743.9 ms |

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
                     (994 ms)
                     (44 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 115.3 ms |
| Video-LLaVA Prefill Complete | 1109.6 ms |
| **Overlap Window** | 994.3 ms |
| **Hidden Tokens** | 43.7 tokens |
| Wall-Clock Time (parallel) | 1853.5 ms |
| Sequential Time (baseline) | 2419.7 ms |
| **Parallel Speedup** | 1.31x |

## Sample Output Verification

### Sample 11

**EventGPT** (48 tokens):
> In the scene, there is a person standing in front of a building with a staircase leading up to it. The building has multiple windows and a door. There is a streetlight visible on the right side of the...

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a curvy road, a tunnel, and a mountain in the background. The car is moving along the road, and the tunnel appears to be a part of the roadway...

---

### Sample 12

**EventGPT** (50 tokens):
> In the scene, there is a person standing in front of a large, dome-shaped structure. The person is wearing a long-sleeved top and shorts. There is a fence in the foreground and some trees

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a bridge in the background, and a tunnel in the middle of the road. The car is moving along the road, and the tunnel appears t...

---

### Sample 18

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a snowy road, a beautiful sunset in the background, and a mountain range in the distance. The car is moving along the road, and the sunset cre...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 9.62x faster than Video-LLaVA
2. **Overlap Window**: 994.3ms available for free draft token generation
3. **Hidden Tokens**: ~44 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 33.3% - drafts are partially aligned

## Implications for Speculative Decoding

With 44 hidden tokens and 33.3% acceptance rate:
- Expected accepted tokens per batch: 14.5
- Effective speedup potential: 15.54x

---

*Generated: 2026-01-27 14:25:42*
*Author: Alice Zhang*
