# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_16s

**Date:** 2026-01-27 21:27:13
**Samples:** 23
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 994.9 ms |
| **Hidden Tokens** | 43.7 tokens |
| **Parallel Speedup** | 1.30x |
| **Acceptance Rate** | 3.2% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.6 ± 0.4 ms | 713.5 ± 108.2 ms | 440.13x |
| **Stage 2: Preprocessing** | 3.7 ± 0.1 ms | 61.7 ± 1.4 ms | 16.45x |
| **Stage 3: Vision Encoding** | 8.0 ± 0.6 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 83.6 ± 0.8 ms | 316.7 ± 1.8 ms | 3.79x |
| **Stage 5: LLM Decode** | 440.6 ± 52.0 ms | 727.3 ± 4.9 ms | 1.65x |
| **TOTAL** | 537.5 ms | 1819.1 ms | 3.38x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 96.9 ms | 637 tokens | 6573 tok/s |
| Video-LLaVA | 1091.8 ms | 4124 tokens | 3777 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 43.7 | 99.1 | 440.6 ms |
| Video-LLaVA | 50.0 | 68.7 | 727.3 ms |

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
                     (995 ms)
                     (44 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 96.9 ms |
| Video-LLaVA Prefill Complete | 1091.8 ms |
| **Overlap Window** | 994.9 ms |
| **Hidden Tokens** | 43.7 tokens |
| Wall-Clock Time (parallel) | 1819.1 ms |
| Sequential Time (baseline) | 2356.6 ms |
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

1. **Prefill Speedup**: EventGPT prefill is 11.27x faster than Video-LLaVA
2. **Overlap Window**: 994.9ms available for free draft token generation
3. **Hidden Tokens**: ~44 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 3.2% - drafts are partially aligned

## Implications for Speculative Decoding

With 44 hidden tokens and 3.2% acceptance rate:
- Expected accepted tokens per batch: 1.4
- Effective speedup potential: 2.41x

---

*Generated: 2026-01-27 21:27:13*
*Author: Alice Zhang*
