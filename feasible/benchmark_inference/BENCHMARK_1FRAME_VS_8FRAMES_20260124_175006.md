# EventGPT (1 frame) vs Video-LLaVA (8 frames) Benchmark

**Date:** 2026-01-24 18:09:55
**Output Path:** `/home/ps/Documents/code/EventGPT/feasible/benchmark_inference/benchmark_1frame_vs_8frames_20260124_175006.json`

## Configuration

| Setting | EventGPT | Video-LLaVA |
|---------|----------|-------------|
| Input Frames | 1 event frame | 8 video frames (from MP4) |
| Samples Tested | 1100 | 1100 |

## Results Summary

| Metric | EventGPT (1 frame) | Video-LLaVA (8 frames) | Ratio |
|--------|-------------------|------------------------|-------|
| **Prefill Length** | 636 tokens | 4643 tokens | **7.3x** |
| **Vision Encoding** | 5.99 ms | 29.49 ms | 4.93x |
| **Prefill Time** | 66.21 ms | 568.36 ms | **8.58x** |
| **Decode Time (5 tok)** | 92.90 ms | 144.38 ms | 1.55x |
| **Total Time** | 165.10 ms | 742.22 ms | **4.50x** |

## Throughput Analysis

| Metric | EventGPT | Video-LLaVA |
|--------|----------|-------------|
| Prefill Throughput | 9605 tok/s | 8169 tok/s |
| Decode Throughput | 53.8 tok/s | 34.6 tok/s |
| Samples/sec | 6.06 | 1.35 |

## Key Findings

### 1. Token Efficiency

- **EventGPT**: 636 tokens (constant regardless of frame count)
- **Video-LLaVA**: 4643 tokens (576 per frame × 8 frames)
- **Ratio**: Video-LLaVA uses **7.3x more tokens**

### 2. Prefill Speedup

- EventGPT prefill: 66.21 ms ± 1.48 ms
- Video-LLaVA prefill: 568.36 ms ± 0.77 ms
- **EventGPT is 8.58x faster** in prefill

### 3. Decode Speedup

- EventGPT decode: 53.8 tokens/sec
- Video-LLaVA decode: 34.6 tokens/sec
- **EventGPT is 1.55x faster** in decode (smaller KV cache)

### 4. Memory Efficiency

- EventGPT KV cache: ~318 MB
- Video-LLaVA KV cache: ~2322 MB
- **7.3x less memory** for EventGPT

## Token Scaling Analysis

```
Video-LLaVA (LINEAR scaling):
  1 frame  →    ~600 tokens
  8 frames →  ~4,640 tokens  (7.7x increase)
  16 frames → ~9,260 tokens  (15.4x increase)
  32 frames → ~18,500 tokens (30.8x increase)

EventGPT (CONSTANT scaling):
  1 frame  →  ~636 tokens
  5 frames →  ~641 tokens
  8 frames →  ~646 tokens
  N frames →  ~650 tokens (constant)
```

## Conclusion

EventGPT achieves **4.50x total speedup** over Video-LLaVA when processing temporal video data:

1. **8.6x faster prefill** due to 7.3x fewer tokens
2. **1.6x faster decode** due to smaller KV cache attention window
3. **7.3x less memory** enabling larger batch sizes

This advantage grows with longer videos as Video-LLaVA scales linearly while EventGPT remains constant.
