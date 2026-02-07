# Parallel Prefill Benchmark Report: my_egpt_dsec_seq_1s

**Date:** 2026-01-27 00:49:59
**Samples:** 0
**Max New Tokens:** 50

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | 0.0 ms |
| **Hidden Tokens** | 0.0 tokens |
| **Parallel Speedup** | 0.00x |
| **Acceptance Rate** | 0.0% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 0.0 ± 0.0 ms | 0.0 ± 0.0 ms | 1.00x |
| **Stage 2: Preprocessing** | 0.0 ± 0.0 ms | 0.0 ± 0.0 ms | 1.00x |
| **Stage 3: Vision Encoding** | 0.0 ± 0.0 ms | 0.0 ± 0.0 ms | 1.00x |
| **Stage 4: LLM Prefill** | 0.0 ± 0.0 ms | 0.0 ± 0.0 ms | 1.00x |
| **Stage 5: LLM Decode** | 0.0 ± 0.0 ms | 0.0 ± 0.0 ms | 1.00x |
| **TOTAL** | 0.0 ms | 0.0 ms | 1.00x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | 0.0 ms | 0 tokens | 0 tok/s |
| Video-LLaVA | 0.0 ms | 0 tokens | 0 tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | 0.0 | 0.0 | 0.0 ms |
| Video-LLaVA | 0.0 | 0.0 | 0.0 ms |

## GPU Memory Usage (4-bit Quantization)

| Model | Model Size | Peak Inference Memory |
|-------|------------|----------------------|
| EventGPT | 3905 MB | 0 MB |
| Video-LLaVA | 3897 MB | 0 MB |
| **Total (both models)** | 7802 MB | - |

## Parallel Execution Analysis

### Token Hiding Opportunity

```
Timeline (Parallel Execution):

  EventGPT:  |--Prefill--|---Decode (hidden)---|---Decode (visible)---|
                         ^                      ^
  Video-LLaVA: |--------Prefill (slow)---------|--------Decode--------|

              |<------ Overlap Window -------->|
                     (0 ms)
                     (0 tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | 0.0 ms |
| Video-LLaVA Prefill Complete | 0.0 ms |
| **Overlap Window** | 0.0 ms |
| **Hidden Tokens** | 0.0 tokens |
| Wall-Clock Time (parallel) | 0.0 ms |
| Sequential Time (baseline) | 0.0 ms |
| **Parallel Speedup** | 0.00x |

## Sample Output Verification


## Key Findings

1. **Prefill Speedup**: EventGPT prefill is 1.00x faster than Video-LLaVA
2. **Overlap Window**: 0.0ms available for free draft token generation
3. **Hidden Tokens**: ~0 EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: 0.0% - drafts are partially aligned

## Implications for Speculative Decoding

With 0 hidden tokens and 0.0% acceptance rate:
- Expected accepted tokens per batch: 0.0
- Effective speedup potential: 1.00x

---

*Generated: 2026-01-27 00:49:59*
*Author: Alice Zhang*
