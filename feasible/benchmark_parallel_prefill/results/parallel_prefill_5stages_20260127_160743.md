# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_1s

**Date:** 2026-01-27 16:07:58
**Samples:** 1
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 266.1 ms |
| **Hidden Tokens** | 26.0 tokens |
| **Parallel Speedup** | 1.48x |
| **Acceptance Rate** | 2.0% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 3.9 ± 0.0 ms | 118.4 ± 0.0 ms | 30.52x |
| **Stage 2: Preprocessing** | 4.8 ± 0.0 ms | 61.9 ± 0.0 ms | 13.02x |
| **Stage 3: Vision Encoding** | 118.8 ± 0.0 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 132.3 ± 0.0 ms | 345.5 ± 0.0 ms | 2.61x |
| **Stage 5: LLM Decode** | 346.4 ± 0.0 ms | 729.5 ± 0.0 ms | 2.11x |
| **TOTAL** | 606.2 ms | 1255.3 ms | 2.07x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 259.8 ms | 637 tokens | 2452 tok/s |
| Video-LLaVA | 525.8 ms | 4124 tokens | 7843 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 34.0 | 98.2 | 346.4 ms |
| Video-LLaVA | 50.0 | 68.5 | 729.5 ms |

## GPU Memory Usage (4-bit Quantization)

| Model | Model Size | Peak Inference Memory |
|-------|------------|----------------------|
| EventGPT | 3905 MB | 8650 MB |
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
                     (266 ms)
                     (26 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 259.8 ms |
| Video-LLaVA Prefill Complete | 525.8 ms |
| **Overlap Window** | 266.1 ms |
| **Hidden Tokens** | 26.0 tokens |
| Wall-Clock Time (parallel) | 1255.3 ms |
| Sequential Time (baseline) | 1861.5 ms |
| **Parallel Speedup** | 1.48x |

## Sample Output Verification

### Sample 0

**EventGPT** (34 tokens):
> In the scene, there is a car parked on the side of a road. The car has a visible license plate and is positioned near a curb.

**Video-LLaVA** (50 tokens):
> The key elements in this scene include a car driving down a winding road, a person standing on the side of the road, and a beautiful view of the mountains in the background. The car is moving along th...

---


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 2.02x faster than Video-LLaVA
2. **Overlap Window**: 266.1ms available for free draft token generation
3. **Hidden Tokens**: ~26 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 2.0% - drafts are partially aligned

## Implications for Speculative Decoding

With 26 hidden tokens and 2.0% acceptance rate:
- Expected accepted tokens per batch: 0.5
- Effective speedup potential: 1.51x

---

*Generated: 2026-01-27 16:07:58*
*Author: Alice Zhang*
