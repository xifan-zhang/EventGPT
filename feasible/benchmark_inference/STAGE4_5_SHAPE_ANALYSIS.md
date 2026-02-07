# Stage 4 & 5 Data Shape Analysis: EventGPT vs Video-LLaVA

**Date:** 2026-01-24 17:18:34

## Summary

This analysis investigates why EventGPT prefill (Stage 4) is not significantly faster than Video-LLaVA, despite EventGPT being designed for efficient event-based vision processing.

## Key Findings

### 1. Prefill Sequence Length Comparison

| Model | Vision Tokens | Text Tokens | Total Prefill Length |
|-------|--------------|-------------|---------------------|
| EventGPT | 582 (5 frames → pooled) | 60 | **641** |
| Video-LLaVA | 576 (1 image patches) | 21 | **597** |

**Finding:** EventGPT has **MORE** tokens to process in prefill (641 vs 597), not fewer!

### 2. Data Shapes at Each Stage

#### EventGPT
```
Stage 3 (Vision Encoding):
  - Input: 5 event frames × [3, 336, 336]
  - Per-frame features: [577, 4096]
  - After spatio-temporal pooling: [1, 582, 4096]

Stage 4 (Prefill):
  - inputs_embeds: [1, 641, 4096]
  - KV cache: 32 layers × [1, 32, 641, 128] = 320.50 MB
  - Prefill time: ~90-115 ms
  - Throughput: ~5500-7000 tokens/sec

Stage 5 (Decode):
  - Per-token input: [1, 1, 4096]
  - Decode throughput: ~53 tokens/sec
```

#### Video-LLaVA
```
Stage 3 (Vision Encoding):
  - Input: 1 image × [3, 336, 336]
  - Vision features: [1, 577, 1024]
  - Encoding time: ~5-33 ms (1 image only)

Stage 4 (Prefill):
  - Input tokens already expanded: [1, 597] (includes 576 image patches)
  - KV cache: 32 layers × [1, 32, 597, 128] = 298.50 MB
  - Prefill time: ~71-76 ms
  - Throughput: ~7800-8300 tokens/sec

Stage 5 (Decode):
  - Per-token input: [1, 1] (token IDs)
  - Decode throughput: ~50-54 tokens/sec
```

### 3. Why EventGPT Prefill is NOT Faster

| Factor | EventGPT | Video-LLaVA | Impact |
|--------|----------|-------------|--------|
| Prefill tokens | 641 | 597 | EventGPT has **7% more** tokens |
| LLM backbone | Vicuna-7B | LLaVA-7B | Same compute per token |
| Hidden dim | 4096 | 4096 | Identical |
| Attention heads | 32 | 32 | Identical |
| KV cache size | 320 MB | 298 MB | EventGPT needs more memory |

**Core insight:** The spatio-temporal pooling in EventGPT compresses 5 event frames (5×577 = 2885 potential tokens) down to only 582 tokens - a 5x compression! However, this is still slightly MORE than Video-LLaVA's 576 tokens for a single image.

### 4. Time Distribution Analysis

```
EventGPT (per sample):
  Stage 3 (Vision): ~50ms (5 frames)
  Stage 4 (Prefill): ~100ms
  Stage 5 (Decode): ~840ms (84% of total)
  ─────────────────
  Total: ~1000ms

Video-LLaVA (per sample):
  Stage 3 (Vision): ~20ms (1 frame)
  Stage 4 (Prefill): ~90ms
  Stage 5 (Decode): ~1740ms (95% of total)
  ─────────────────
  Total: ~1850ms
```

**Key observation:** EventGPT's 1.86x speedup comes primarily from **shorter decode phase** (generating fewer tokens), NOT from faster prefill.

## Why Decode Dominates

The decode phase (Stage 5) dominates because:

1. **Memory-bound operations**: Each decode step requires reading the entire KV cache
2. **Sequential nature**: Tokens must be generated one at a time (autoregressive)
3. **Token count**: ~50 tokens × ~18ms/token = ~900ms

Even a 2x prefill speedup would only yield:
- Current: 100ms prefill + 900ms decode = 1000ms
- Optimized: 50ms prefill + 900ms decode = 950ms
- Improvement: Only **5% total speedup**

## Conclusions

1. **EventGPT's prefill is NOT faster** because it processes similar token counts
2. **The real speedup** comes from EventGPT generating fewer/shorter responses
3. **Optimization priority** should be:
   - Decode acceleration (speculative decoding, quantization)
   - Reducing output token count
   - KV cache optimization

## Recommendations

To improve EventGPT inference speed:

1. **Speculative Decoding**: Use a smaller draft model to predict multiple tokens
2. **Token Reduction**: Further compress event features (current: 582 tokens)
3. **Quantization**: INT8/INT4 quantization for decode phase
4. **KV Cache Optimization**: PagedAttention, continuous batching
