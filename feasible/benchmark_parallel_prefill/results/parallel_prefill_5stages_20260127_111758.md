# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_2s

**Date:** 2026-01-27 11:34:25
**Samples:** 540
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 407.1 ms |
| **Hidden Tokens** | 39.4 tokens |
| **Parallel Speedup** | 1.47x |
| **Acceptance Rate** | 36.1% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 1.7 ± 0.1 ms | 127.4 ± 28.9 ms | 73.98x |
| **Stage 2: Preprocessing** | 3.6 ± 0.1 ms | 61.2 ± 1.0 ms | 17.02x |
| **Stage 3: Vision Encoding** | 8.1 ± 4.6 ms | 0.0 ± 0.0 ms | 0.00x |
| **Stage 4: LLM Prefill** | 83.0 ± 2.0 ms | 315.0 ± 1.3 ms | 3.79x |
| **Stage 5: LLM Decode** | 473.5 ± 42.7 ms | 721.5 ± 1.3 ms | 1.52x |
| **TOTAL** | 570.0 ms | 1225.1 ms | 2.15x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 96.5 ms | 637 tokens | 6602 tok/s |
| Video-LLaVA | 503.6 ms | 4124 tokens | 8190 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 46.8 | 98.8 | 473.5 ms |
| Video-LLaVA | 50.0 | 69.3 | 721.5 ms |

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
                     (407 ms)
                     (39 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 96.5 ms |
| Video-LLaVA Prefill Complete | 503.6 ms |
| **Overlap Window** | 407.1 ms |
| **Hidden Tokens** | 39.4 tokens |
| Wall-Clock Time (parallel) | 1225.1 ms |
| Sequential Time (baseline) | 1795.0 ms |
| **Parallel Speedup** | 1.47x |

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

1. **Prefill Speedup**: EventGPT prefill is 5.22x faster than Video-LLaVA
2. **Overlap Window**: 407.1ms available for free draft token generation
3. **Hidden Tokens**: ~39 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 36.1% - drafts are partially aligned

## Implications for Speculative Decoding

With 39 hidden tokens and 36.1% acceptance rate:
- Expected accepted tokens per batch: 14.2
- Effective speedup potential: 15.22x

---

*Generated: 2026-01-27 11:34:25*
*Author: Alice Zhang*
