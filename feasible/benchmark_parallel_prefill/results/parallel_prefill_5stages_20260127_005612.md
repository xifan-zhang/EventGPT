# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_1s

**Date:** 2026-01-27 00:56:40
**Samples:** 3
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 355.2 ms |
| **Hidden Tokens** | 33.3 tokens |
| **Parallel Speedup** | 1.47x |
| **Acceptance Rate** | 40.1% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 2.4 ± 1.0 ms | 108.2 ± 7.3 ms | 44.54x |
| **Stage 2: Preprocessing** | 4.4 ± 0.9 ms | 62.5 ± 0.3 ms | 14.36x |
| **Stage 3: Vision Encoding** | 44.6 ± 51.1 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 100.9 ± 21.0 ms | 336.8 ± 13.2 ms | 3.34x |
| **Stage 5: LLM Decode** | 441.7 ± 58.2 ms | 750.0 ± 4.2 ms | 1.70x |
| **TOTAL** | 594.0 ms | 1257.5 ms | 2.12x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 152.3 ms | 637 tokens | 4183 tok/s |
| Video-LLaVA | 507.5 ms | 4124 tokens | 8126 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 42.0 | 95.0 | 441.7 ms |
| Video-LLaVA | 50.0 | 66.7 | 750.0 ms |

## GPU Memory Usage (4-bit Quantization)

| Model | Model Size | Peak Inference Memory |
|-------|------------|----------------------|
| EventGPT | 3905 MB | 8656 MB |
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
                     (355 ms)
                     (33 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 152.3 ms |
| Video-LLaVA Prefill Complete | 507.5 ms |
| **Overlap Window** | 355.2 ms |
| **Hidden Tokens** | 33.3 tokens |
| Wall-Clock Time (parallel) | 1257.5 ms |
| Sequential Time (baseline) | 1851.5 ms |
| **Parallel Speedup** | 1.47x |

## Sample Output Verification

### Sample 0

**EventGPT** (34 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a curb.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a person standing on the side of the road, and a beautiful view of the mountains in the background. The car is moving along th...

---

### Sample 1

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a stone wall on the side of the road, and a beautiful view of the ocean in the background. The car is moving along the road, a...

---

### Sample 2

**EventGPT** (46 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a building. There is also a streetlight and some trees in the background.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a stone wall on the side of the road, and a beautiful view of the mountains in the background. The car is moving along the roa...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 3.33x faster than Video-LLaVA
2. **Overlap Window**: 355.2ms available for free draft token generation
3. **Hidden Tokens**: ~33 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 40.1% - drafts are partially aligned

## Implications for Speculative Decoding

With 33 hidden tokens and 40.1% acceptance rate:
- Expected accepted tokens per batch: 13.4
- Effective speedup potential: 14.35x

---

*Generated: 2026-01-27 00:56:40*
*Author: Alice Zhang*
