# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_16s

**Date:** 2026-01-27 14:32:11
**Samples:** 23
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 1011.2 ms |
| **Hidden Tokens** | 43.7 tokens |
| **Parallel Speedup** | 1.30x |
| **Acceptance Rate** | 33.3% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.6 ± 0.4 ms | 722.4 ± 104.6 ms | 445.31x |
| **Stage 2: Preprocessing** | 3.9 ± 0.2 ms | 63.3 ± 0.7 ms | 16.43x |
| **Stage 3: Vision Encoding** | 8.7 ± 2.3 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 85.3 ± 2.1 ms | 325.0 ± 3.9 ms | 3.81x |
| **Stage 5: LLM Decode** | 451.4 ± 52.9 ms | 744.5 ± 3.2 ms | 1.65x |
| **TOTAL** | 550.9 ms | 1855.2 ms | 3.37x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 99.5 ms | 637 tokens | 6403 tok/s |
| Video-LLaVA | 1110.7 ms | 4124 tokens | 3713 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 43.7 | 96.7 | 451.4 ms |
| Video-LLaVA | 50.0 | 67.2 | 744.5 ms |

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
                     (1011 ms)
                     (44 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 99.5 ms |
| Video-LLaVA Prefill Complete | 1110.7 ms |
| **Overlap Window** | 1011.2 ms |
| **Hidden Tokens** | 43.7 tokens |
| Wall-Clock Time (parallel) | 1855.2 ms |
| Sequential Time (baseline) | 2406.0 ms |
| **Parallel Speedup** | 1.30x |

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

1. **Prefill Speedup**: EventGPT prefill is 11.16x faster than Video-LLaVA
2. **Overlap Window**: 1011.2ms available for free draft token generation
3. **Hidden Tokens**: ~44 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 33.3% - drafts are partially aligned

## Implications for Speculative Decoding

With 44 hidden tokens and 33.3% acceptance rate:
- Expected accepted tokens per batch: 14.5
- Effective speedup potential: 15.54x

---

*Generated: 2026-01-27 14:32:11*
*Author: Alice Zhang*
