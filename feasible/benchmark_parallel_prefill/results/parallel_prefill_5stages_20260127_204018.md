# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_2s

**Date:** 2026-01-27 20:57:07
**Samples:** 540
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 412.3 ms |
| **Hidden Tokens** | 39.7 tokens |
| **Parallel Speedup** | 1.46x |
| **Acceptance Rate** | 3.4% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.7 ± 0.1 ms | 128.1 ± 29.6 ms | 74.09x |
| **Stage 2: Preprocessing** | 3.7 ± 0.1 ms | 61.8 ± 1.4 ms | 16.83x |
| **Stage 3: Vision Encoding** | 8.4 ± 5.0 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 84.7 ± 2.7 ms | 320.8 ± 4.5 ms | 3.79x |
| **Stage 5: LLM Decode** | 475.8 ± 42.6 ms | 733.0 ± 10.3 ms | 1.54x |
| **TOTAL** | 574.3 ms | 1243.7 ms | 2.17x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 98.5 ms | 637 tokens | 6470 tok/s |
| Video-LLaVA | 510.7 ms | 4124 tokens | 8075 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 46.8 | 98.3 | 475.8 ms |
| Video-LLaVA | 50.0 | 68.2 | 733.0 ms |

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
                     (412 ms)
                     (40 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 98.5 ms |
| Video-LLaVA Prefill Complete | 510.7 ms |
| **Overlap Window** | 412.3 ms |
| **Hidden Tokens** | 39.7 tokens |
| Wall-Clock Time (parallel) | 1243.7 ms |
| Sequential Time (baseline) | 1817.9 ms |
| **Parallel Speedup** | 1.46x |

## Sample Output Verification

### Sample 0

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a stone wall on the side of the road, and a beautiful view of the mountains in the background. The car is moving along the roa...

---

### Sample 1

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a stone wall on the side of the road, and a beautiful view of the mountains in the background. The car is moving along the roa...

---

### Sample 2

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a stone wall on the side of the road, and a tree in the foreground. The car is moving along the road, and the stone wall adds ...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 5.19x faster than Video-LLaVA
2. **Overlap Window**: 412.3ms available for free draft token generation
3. **Hidden Tokens**: ~40 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 3.4% - drafts are partially aligned

## Implications for Speculative Decoding

With 40 hidden tokens and 3.4% acceptance rate:
- Expected accepted tokens per batch: 1.4
- Effective speedup potential: 2.36x

---

*Generated: 2026-01-27 20:57:07*
*Author: Alice Zhang*
