# EventGPT (1 frame) vs Video-LLaVA (8 frames) Analysis

**Date:** 2026-01-24 17:41:22

## Experimental Setup

- **EventGPT**: 1 event frame (first frame from event image sequence)
- **Video-LLaVA**: 8 frames uniformly sampled from MP4 video (21 frames total)
- **Query**: "What are the key elements in this scene?"
- **Dataset**: DSEC 1-second sequences

## Results Summary

| Metric | EventGPT (1 frame) | Video-LLaVA (8 frames) | Ratio |
|--------|-------------------|------------------------|-------|
| **Prefill Length** | 636 tokens | 4,643 tokens | **7.3x** |
| **Prefill Time** | 66-127 ms | 567-572 ms | **4.5x faster** |
| **KV Cache Size** | 318 MB | 2,322 MB | **7.3x** |
| **Decode Throughput** | 54 tok/s | 33 tok/s | **1.6x faster** |

## Detailed Stage Breakdown

### EventGPT (1 Event Frame)

```
STAGE 3 (Vision Encoding, 1 frame):
  - Input: [3, 336, 336] event image
  - Output: [1, 577, 4096] features
  - Time: 5.9-101.3 ms

STAGE 4 (Prefill):
  - inputs_embeds: [1, 636, 4096]
  - PREFILL LENGTH: 636 tokens
  - KV cache: 32 layers × [1, 32, 636, 128]
  - KV cache size: 318.0 MB
  - Time: 65.9-126.7 ms
  - Throughput: 5,021-9,655 tokens/sec

STAGE 5 (Decode):
  - Time per 5 tokens: 92.5-93.0 ms
  - Throughput: 53.8-54.1 tokens/sec
```

### Video-LLaVA (8 Video Frames)

```
STAGE 2 (Preprocess 8 images):
  - Time: 76.4-83.6 ms

STAGE 3 (Vision Encoding, 8 frames):
  - Input: [8, 3, 336, 336] images
  - Output: [8, 577, 1024] features
  - Time: 29.3-55.0 ms

STAGE 4 (Prefill):
  - input_ids: [1, 4643] (already expanded)
  - PREFILL LENGTH: 4,643 tokens
  - KV cache: 32 layers × [1, 32, 4643, 128]
  - KV cache size: 2,321.5 MB
  - Time: 567.3-572.4 ms
  - Throughput: 8,112-8,185 tokens/sec

STAGE 5 (Decode):
  - Time per 5 tokens: 143.9-152.3 ms
  - Throughput: 32.8-34.8 tokens/sec
```

## Token Scaling Analysis

### Video-LLaVA: Linear Scaling

```
Frames → Tokens
   1   →    597 tokens (576 patches + 21 text)
   4   → ~2,325 tokens
   8   →  4,643 tokens (4,608 patches + 35 text)
  16   → ~9,251 tokens (projected)
  32   → ~18,467 tokens (projected)
```

Formula: `tokens = 576 × num_frames + ~35`

### EventGPT: Constant Scaling

```
Frames → Tokens
   1   →  636 tokens (577 vision + 59 text)
   5   →  641 tokens (582 vision + 59 text)
   8   → ~646 tokens (projected)
  16   → ~646 tokens (projected)
  32   → ~646 tokens (projected)
```

Formula: `tokens ≈ 577 + 5 × log2(num_frames) + 59` (approximate)

## Why This Matters

### 1. Prefill Compute Scales Quadratically

Self-attention is O(n²) in sequence length:
- EventGPT (636 tokens): 636² = 404,496 attention operations
- Video-LLaVA (4643 tokens): 4643² = 21,557,449 attention operations
- **53x more compute** for Video-LLaVA prefill

### 2. KV Cache Memory Scales Linearly

- EventGPT: 318 MB for 636 tokens
- Video-LLaVA: 2,322 MB for 4,643 tokens
- **7.3x more memory** limits batch size and max context

### 3. Decode Speed Affected by KV Cache Size

Each decode step attends to entire KV cache:
- EventGPT: 54 tok/s (smaller attention window)
- Video-LLaVA: 33 tok/s (larger attention window)
- **1.6x faster decode** for EventGPT

## Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVENTGPT ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│  N event frames → CLIP encoder → Feature Adaptor → Spatio-temporal │
│                                                    Pooling          │
│                                                        ↓            │
│                                                   ~577 tokens       │
│                                                   (constant)        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   VIDEO-LLAVA ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│  N video frames → CLIP encoder → Concatenate all patches           │
│                                        ↓                            │
│                                  576 × N tokens                     │
│                                  (linear scaling)                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Implications for Real-World Use

| Scenario | EventGPT | Video-LLaVA |
|----------|----------|-------------|
| 1 sec video (20 fps) | ~650 tokens | ~11,500 tokens |
| 5 sec video | ~650 tokens | ~57,600 tokens |
| 10 sec video | ~650 tokens | ~115,200 tokens |
| Max context (4K) | ✓ Works | ✗ Exceeds limit at ~7 frames |

## Conclusion

EventGPT's spatio-temporal compression provides:
1. **4.5x faster prefill** (fewer tokens to process)
2. **7.3x less memory** (smaller KV cache)
3. **1.6x faster decode** (smaller attention window)
4. **Constant scaling** regardless of input frame count

This architectural advantage becomes more pronounced with longer videos, where Video-LLaVA would exceed context limits while EventGPT maintains constant efficiency.
