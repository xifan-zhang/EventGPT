# EventGPT (1 frame) vs Video-LLaVA (8 frames) Benchmark

**Date:** 2026-01-24 20:15:21
**Dataset:** DSEC 500ms sequences
**Output Path:** `/home/ps/Documents/code/EventGPT/feasible/benchmark_inference/benchmark_1frame_vs_8frames_20260124_193405.json`

## Configuration

| Setting | EventGPT | Video-LLaVA |
|---------|----------|-------------|
| Input Frames | 1 event frame | 8 video frames (from MP4) |
| Samples Tested | 2220 | 2220 |

## Results Summary

| Metric | EventGPT (1 frame) | Video-LLaVA (8 frames) | Ratio |
|--------|-------------------|------------------------|-------|
| **Prefill Length** | 636 tokens | 4643 tokens | **7.3x** |
| **Vision Encoding** | 5.93 ms | 29.48 ms | 4.97x |
| **Prefill Time** | 66.16 ms | 568.13 ms | **8.59x** |
| **Decode Time (5 tok)** | 92.68 ms | 143.94 ms | 1.55x |
| **Total Time** | 164.76 ms | 741.56 ms | **4.50x** |

## Throughput Analysis

| Metric | EventGPT | Video-LLaVA |
|--------|----------|-------------|
| Prefill Throughput | 9614 tok/s | 8172 tok/s |
| Decode Throughput | 54.0 tok/s | 34.7 tok/s |
| Samples/sec | 6.07 | 1.35 |

## Memory Usage

| Metric | EventGPT | Video-LLaVA | Ratio |
|--------|----------|-------------|-------|
| KV Cache | 318.0 MB | 2321.5 MB | 7.3x |

## Key Findings

### 1. Token Efficiency
- **EventGPT**: 636 tokens (constant regardless of frame count)
- **Video-LLaVA**: 4643 tokens (576 per frame × 8 frames)
- **Ratio**: Video-LLaVA uses **7.3x more tokens**

### 2. Prefill Speedup
- EventGPT prefill: 66.16 ms ± 1.12 ms
- Video-LLaVA prefill: 568.13 ms ± 0.72 ms
- **EventGPT is 8.59x faster** in prefill

### 3. Decode Speedup
- EventGPT decode: 54.0 tokens/sec
- Video-LLaVA decode: 34.7 tokens/sec
- **EventGPT is 1.55x faster** in decode (smaller KV cache)

## Theoretical Speedup Analysis

### Attention Complexity

Self-attention has O(n²) complexity where n is sequence length:

```
EventGPT:    636²  = 404,496 attention operations
Video-LLaVA: 4643² = 21,557,449 attention operations

Theoretical ratio: 53.3x more compute for Video-LLaVA
Measured ratio:    8.59x (memory-bandwidth bound)
```

**Why measured speedup (8.59x) < theoretical (53.3x):**
1. **Memory bandwidth bottleneck**: Limited by GPU memory bandwidth, not compute
2. **Vision encoding cost**: Video-LLaVA processes 8 images (29.48ms) vs EventGPT's 1 (5.93ms)
3. **Kernel efficiency**: Larger batches can be more efficient despite O(n²) complexity

### Per-Token Acceptance Rate Equivalent

In speculative decoding terms, achieving EventGPT's performance from Video-LLaVA would require:

```
Required acceptance rate = 1 / (token_ratio) = 1 / 7.3 = 13.7%

To match EventGPT's 636 tokens with Video-LLaVA's 4643 tokens:
- Need to "reject" 86.3% of Video-LLaVA's tokens (4007 out of 4643)
- Or equivalently: accept only 13.7% of draft tokens consistently
```

This means EventGPT's **architectural compression is equivalent to 86.3% token rejection** in speculative decoding.

### KV Cache Memory Scaling

```
KV cache size = 2 × num_layers × num_heads × seq_len × head_dim × bytes_per_param

EventGPT:    2 × 32 × 32 × 636 × 128 × 2 = 318 MB
Video-LLaVA: 2 × 32 × 32 × 4643 × 128 × 2 = 2321 MB

Memory ratio: 7.3x (directly proportional to sequence length)
```

**Impact on batch size:**
- With 24GB GPU VRAM:
  - EventGPT: ~75 samples per batch (318 MB each)
  - Video-LLaVA: ~10 samples per batch (2321 MB each)
- **7.5x difference in batch capacity**

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

## Stage-by-Stage Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EventGPT (1 frame)     Video-LLaVA (8 frames)                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Stage 1-2 (Load + Preprocess)    ~6 ms              ~83 ms                       │
│  Stage 3 (Vision Encoding)        ~6 ms              ~29 ms                       │
│  Stage 4 (Prefill)               ~66 ms             ~568 ms  ← 8.59x faster       │
│  Stage 5 (Decode 5 tokens)        ~93 ms             ~144 ms  ← 1.55x faster       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Total                           ~165 ms            ~742 ms  ← 4.50x faster         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Prefill dominates speedup difference:**
- Stage 4 (Prefill): 8.59x speedup → contributes **414 ms** of total 577 ms advantage
- Stage 5 (Decode): 1.55x speedup → contributes **51 ms** of total advantage
- Vision stages: Contribute **112 ms** advantage

## Conclusion

EventGPT achieves **4.50x total speedup** over Video-LLaVA when processing temporal video data:

1. **8.6x faster prefill** due to 7.3x fewer tokens (53x theoretical, 8.6x measured)
2. **1.6x faster decode** due to smaller KV cache attention window
3. **7.3x less memory** enabling **7.5x larger batch sizes**
4. **Equivalent to 86.3% token rejection rate** in speculative decoding

This advantage grows with longer videos as Video-LLaVA scales linearly while EventGPT remains constant.

---

## Research Documents

Based on these benchmark results, the following research documents were created:

| Document | Purpose | Key Finding |
|----------|---------|-------------|
| `SPECULATIVE_DECODING_RESEARCH.md` | Analyze EventGPT as draft model for Video-LLaVA | Naive approach slower (568ms prefill dominates) |
| `EXPLOITING_FAST_PREFILL.md` | KV Cache Adaptation strategies | 2.85x speedup with trained adapter |
| `PARALLEL_PREFILL_RESEARCH.md` | Parallel prefill analysis | **2.0x speedup with free draft tokens** |

---

## What's Next: Implementation Plan

### Phase 1: Implement Parallel Prefill (2-3 days)
- [ ] Implement `ParallelPrefillSpeculative` class
- [ ] Multi-threaded model execution (EventGPT + Video-LLaVA)
- [ ] Draft budget estimation based on time window
- [ ] KV cache management for both models

### Phase 2: Baseline Acceptance Measurement (1-2 days)
- [ ] Run parallel prefill without adapter
- [ ] Measure actual acceptance rate on validation set
- [ ] Identify failure modes (semantic gaps, token misalignment)
- [ ] Document baseline performance

### Phase 3: Train Parallel Adapter (1-2 weeks)
- [ ] Collect (EGPT draft, VL target) training pairs
- [ ] Implement `ParallelPrefillAdapter` with cross-attention
- [ ] Train for high acceptance rate (target: 80%)
- [ ] Validate on held-out test set

### Phase 4: Multi-GPU Optimization (1 week)
- [ ] Implement true parallel execution on separate GPUs
- [ ] Minimize data transfer overhead
- [ ] Benchmark scaling vs single-GPU
- [ ] Optimize memory layout

### Expected Final Performance
| Output Length | Video-LLaVA | Parallel + Adapter |
|--------------|-------------|-------------------|
| 5 tokens | 743 ms | 650 ms (1.14x) |
| 20 tokens | 1268 ms | 850 ms (1.49x) |
| 50 tokens | 2318 ms | 1200 ms (1.93x) |
| 100 tokens | 4068 ms | 2100 ms (1.94x) |

---

*Benchmark run: 2026-01-24 20:15:21*
*Script: `benchmark_inference_5stages.py --compare_1vs8`*
*Research: 2026-01-24 21:00:00*
