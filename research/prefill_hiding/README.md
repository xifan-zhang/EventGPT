# Prefill Hiding: Concealing Draft Model Prefill Inside Target Model Prefill

**Date:** 2026-01-26
**Status:** Research Proposal
**Focus:** EventGPT (Draft) + Video-LLaVA (Target) Speculative Decoding

---

## Executive Summary

This document explores a novel approach to speculative decoding where the draft model's prefill stage is completely **hidden** inside the target model's longer prefill stage. By exploiting the 8.59x prefill time difference between EventGPT (66ms) and Video-LLaVA (568ms), we can generate **~28 free draft tokens** during what would otherwise be idle waiting time.

**Key Insight:** The draft model's entire prefill + initial token generation can complete before the target model finishes its prefill, effectively making draft speculation "free" from a latency perspective.

---

## 1. The Timing Opportunity

### Benchmark Results (from DSEC 1s dataset)

| Model | Prefill Time | Prefill Tokens | Decode Speed |
|-------|--------------|----------------|--------------|
| **EventGPT** (1 frame) | 66 ms | 636 tokens | 54 tok/s (18.5ms/tok) |
| **Video-LLaVA** (8 frames) | 568 ms | 4,643 tokens | 33 tok/s (30ms/tok) |
| **Ratio** | 8.59x faster | 7.3x fewer | 1.6x faster |

### The "Free Time" Window

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    PREFILL TIMING COMPARISON                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Time (ms):  0    66   100   150   200   250   300   400   500   568      │
│              │    │     │     │     │     │     │     │     │     │       │
│              ▼    ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼       │
│                                                                            │
│  EventGPT:   ████ PREFILL ████|▓▓▓▓▓▓▓▓▓▓▓▓▓▓ FREE DRAFT GENERATION ▓▓▓▓▓│
│              └── 66ms ──┘     └────────── 502ms FREE WINDOW ──────────────┘│
│                                                                            │
│  Video-LLaVA:████████████████████████████████████████████████████ PREFILL █│
│              └───────────────────── 568ms ─────────────────────────────────┘│
│                                                                            │
│  Draft tokens generated in free window: 502ms / 18.5ms ≈ 27 tokens        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**Key Numbers:**
- Free window: 568ms - 66ms = **502ms**
- EventGPT decode speed: 18.5ms/token
- Free draft tokens: 502ms / 18.5ms ≈ **27 tokens**
- Typical response length: 44.5 tokens (from benchmarks)
- **Coverage: 27/44.5 = 61% of response drafted "for free"**

---

## 2. Prefill Hiding Architecture

### 2.1 Parallel Execution Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PREFILL HIDING EXECUTION FLOW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Time ────────────────────────────────────────────────────────────────────► │
│                                                                             │
│  T=0ms: START PARALLEL EXECUTION                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ CUDA Stream 0 (Main)                                                   │ │
│  │ ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │ │             VIDEO-LLAVA PREFILL (568ms)                         │   │ │
│  │ │  • Load 8 RGB frames                                            │   │ │
│  │ │  • CLIP encode: 8 × 576 = 4608 vision tokens                    │   │ │
│  │ │  • LLM prefill: 4643 tokens through 32 layers                   │   │ │
│  │ │  • Output: KV cache ready for decode                            │   │ │
│  │ └─────────────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ CUDA Stream 1 (Draft)                                                  │ │
│  │ ┌──────┐ ┌──────────────────────────────────────────────────────────┐ │ │
│  │ │EGPT  │ │         EVENTGPT DRAFT TOKEN GENERATION                  │ │ │
│  │ │Prefil│ │  • Generate tokens autoregressively                      │ │ │
│  │ │(66ms)│ │  • ~27 tokens in 502ms remaining window                  │ │ │
│  │ │      │ │  • Store draft tokens for later verification             │ │ │
│  │ └──────┘ └──────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  T=568ms: SYNCHRONIZE & VERIFY                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ • Video-LLaVA KV cache ready                                          │ │
│  │ • 27 draft tokens available                                           │ │
│  │ • Begin parallel verification (batched)                               │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Architecture

```python
class PrefillHidingSpeculativeDecoder:
    """
    Hides EventGPT (draft) prefill inside Video-LLaVA (target) prefill.
    Generates ~27 free draft tokens during target's prefill time.
    """

    def __init__(self, eventgpt, videollava):
        self.draft = eventgpt      # Fast: 66ms prefill, 18.5ms/tok decode
        self.target = videollava   # Slow: 568ms prefill, 30ms/tok decode
        self.draft_stream = torch.cuda.Stream()

    def generate(self, event_images, video_frames, query, max_tokens=64):
        """
        Main generation loop with prefill hiding.
        """
        # Phase 1: Parallel Prefill (HIDDEN)
        draft_tokens, target_kv = self._parallel_prefill(
            event_images, video_frames, query
        )

        # Phase 2: Verification + Continued Generation
        output_tokens = self._speculative_decode(
            draft_tokens, target_kv, max_tokens
        )

        return output_tokens

    def _parallel_prefill(self, event_images, video_frames, query):
        """
        Execute prefills in parallel. Draft prefill is HIDDEN in target prefill time.
        """
        draft_tokens = []

        # Start target prefill (slow, 568ms)
        target_future = self._async_target_prefill(video_frames, query)

        # Start draft prefill (fast, 66ms)
        with torch.cuda.stream(self.draft_stream):
            draft_kv = self.draft.prefill(event_images, query)

            # CRITICAL: Use remaining time for draft generation
            # Free window = 568ms - 66ms = 502ms
            free_window_tokens = 27  # 502ms / 18.5ms

            for _ in range(free_window_tokens):
                if target_future.done():
                    break  # Target finished early
                token = self.draft.decode_one(draft_kv)
                draft_tokens.append(token)

        # Wait for target to complete
        target_kv = target_future.result()

        return draft_tokens, target_kv

    def _speculative_decode(self, draft_tokens, target_kv, max_tokens):
        """
        Verify draft tokens and continue generation.
        """
        verified_tokens = []

        # Batch verify all draft tokens at once
        verified, first_mismatch = self.target.batch_verify(
            draft_tokens, target_kv
        )
        verified_tokens.extend(verified)

        # Continue with standard speculative decoding for remaining tokens
        while len(verified_tokens) < max_tokens:
            # Generate more draft tokens
            new_drafts = self.draft.generate_k(k=5)

            # Verify with target
            new_verified, _ = self.target.batch_verify(new_drafts, target_kv)
            verified_tokens.extend(new_verified)

            if self._is_eos(verified_tokens[-1]):
                break

        return verified_tokens
```

---

## 3. Theoretical Speedup Analysis

### 3.1 Baseline: Sequential Video-LLaVA

```
Standard Video-LLaVA (45 tokens):
  Prefill:     568ms
  Decode:      45 × 30ms = 1350ms
  ─────────────────────────────
  Total:       1918ms
```

### 3.2 Prefill Hiding + Speculative Decoding

```
Prefill Hiding with EventGPT (45 tokens, 80% acceptance):

Phase 1 - Parallel Prefill (Hidden):
  Target prefill:  568ms (WALL TIME)
  Draft prefill:    66ms (HIDDEN inside target)
  Draft tokens:     27 tokens (HIDDEN inside target)
  ─────────────────────────────
  Wall time:       568ms  (same as baseline prefill!)

Phase 2 - Verification + Decode:
  Verify 27 drafts: 1 × 30ms = 30ms (batched)
  Accepted tokens:  27 × 0.8 = ~22 tokens

  Remaining tokens: 45 - 22 = 23 tokens
  Spec decode (k=5, 80% accept):
    Iterations: 23 / (5 × 0.8) = 6 iterations
    Time: 6 × (5 × 18.5ms + 30ms) = 735ms
  ─────────────────────────────
  Total:           568 + 30 + 735 = 1333ms

Speedup: 1918ms / 1333ms = 1.44x
```

### 3.3 Speedup vs Acceptance Rate

| Acceptance Rate | Free Tokens Accepted | Remaining | Spec Decode Iters | Total Time | Speedup |
|-----------------|----------------------|-----------|-------------------|------------|---------|
| 100% | 27 | 18 | 4 | 568 + 30 + 530 = **1128ms** | **1.70x** |
| 80% | 22 | 23 | 6 | 568 + 30 + 735 = **1333ms** | **1.44x** |
| 60% | 16 | 29 | 7 | 568 + 30 + 855 = **1453ms** | **1.32x** |
| 40% | 11 | 34 | 9 | 568 + 30 + 1035 = **1633ms** | **1.17x** |
| 20% | 5 | 40 | 10 | 568 + 30 + 1230 = **1828ms** | **1.05x** |

**Key Finding:** Even with modest 60% acceptance rate, prefill hiding provides 1.32x speedup.

---

## 4. Acceptance Rate Considerations

### 4.1 The Cross-Modal Challenge

EventGPT and Video-LLaVA operate on different modalities:
- **EventGPT**: Sparse event data, motion-focused, temporal edge detection
- **Video-LLaVA**: Dense RGB frames, appearance-focused, spatial detail

This leads to naturally different token distributions, potentially lowering acceptance rates.

### 4.2 Strategies to Improve Acceptance

#### Strategy A: Feature-Level Speculation

Instead of token-level matching, match at the embedding level:

```python
class FeatureLevelPrefillHiding:
    """
    Speculate at feature level instead of token level to bypass tokenizer mismatch.
    """

    def __init__(self, eventgpt, videollava, alignment_layer):
        self.draft = eventgpt
        self.target = videollava
        self.align = alignment_layer  # Trained on DSEC dataset

    def _parallel_prefill(self, event_images, video_frames, query):
        # Draft generates hidden states instead of tokens
        draft_hidden_states = []

        target_future = self._async_target_prefill(video_frames, query)

        with torch.cuda.stream(self.draft_stream):
            draft_kv = self.draft.prefill(event_images, query)

            for _ in range(27):
                hidden = self.draft.decode_one_hidden(draft_kv)
                # Align to target feature space
                aligned_hidden = self.align(hidden)
                draft_hidden_states.append(aligned_hidden)

        target_kv = target_future.result()

        return draft_hidden_states, target_kv

    def _speculative_decode(self, draft_hidden, target_kv, max_tokens):
        """
        Verify at feature level using target's LM head.
        """
        verified = []

        for hidden in draft_hidden:
            # Use target's LM head on aligned hidden state
            target_logits = self.target.lm_head(hidden)
            target_token = target_logits.argmax()

            # Verify: is this token close to what target would generate?
            actual_logits = self.target.forward_one(target_kv)
            if self._kl_divergence(target_logits, actual_logits) < threshold:
                verified.append(target_token)
            else:
                break  # First mismatch

        return verified
```

**Expected Improvement:** Feature-level alignment can achieve 60-80% acceptance vs 20-40% token-level.

#### Strategy B: Distilled Draft Head

Add a head to EventGPT that directly predicts Video-LLaVA tokens:

```python
class DistilledDraftHead(nn.Module):
    """
    Trained to predict Video-LLaVA tokens from EventGPT features.
    """
    def __init__(self, eventgpt_dim=4096, videollava_vocab=32000):
        self.proj = nn.Linear(eventgpt_dim, eventgpt_dim * 2)
        self.lm_head = nn.Linear(eventgpt_dim * 2, videollava_vocab)

    def forward(self, eventgpt_hidden):
        x = F.silu(self.proj(eventgpt_hidden))
        return self.lm_head(x)
```

**Training:** Knowledge distillation from Video-LLaVA outputs on aligned DSEC dataset.

#### Strategy C: Semantic Matching

Accept tokens that are semantically equivalent even if not identical:

```python
def semantic_verify(draft_token, target_logits, tokenizer, threshold=0.8):
    """
    Accept if draft token is semantically similar to top-k target predictions.
    """
    draft_text = tokenizer.decode(draft_token)
    top_k = target_logits.topk(10).indices

    for candidate in top_k:
        candidate_text = tokenizer.decode(candidate)
        similarity = compute_semantic_similarity(draft_text, candidate_text)
        if similarity > threshold:
            return True, candidate  # Accept with target's choice

    return False, None
```

---

## 5. Memory Considerations

### 5.1 Dual Model Loading Challenge

Both models need to be in memory simultaneously:
- EventGPT: ~13.5 GB
- Video-LLaVA: ~13.5 GB
- Total: ~27 GB (exceeds 24GB VRAM)

### 5.2 Solutions

#### Option A: Sequential Loading with Feature Caching

```python
class MemoryEfficientPrefillHiding:
    """
    Load one model at a time, cache features for later.
    """

    def generate(self, event_images, video_frames, query):
        # Phase 1: Load EventGPT, generate drafts
        self.draft = load_eventgpt()  # 13.5 GB
        draft_tokens, draft_features = self._generate_drafts(
            event_images, query, n_tokens=40
        )

        # Cache drafts to CPU
        draft_tokens = [t.cpu() for t in draft_tokens]
        draft_features = [f.cpu() for f in draft_features]

        # Unload draft
        del self.draft
        torch.cuda.empty_cache()  # Free 13.5 GB

        # Phase 2: Load Video-LLaVA, verify drafts
        self.target = load_videollava()  # 13.5 GB

        # Move drafts back to GPU for verification
        draft_tokens = [t.cuda() for t in draft_tokens]

        output = self._verify_and_complete(
            video_frames, query, draft_tokens, max_tokens=64
        )

        return output
```

**Trade-off:** Loses parallel execution benefit, but still provides free draft tokens that can be verified in one batch.

#### Option B: Quantized Models

```python
# Load both models in 4-bit quantization
eventgpt_4bit = load_model("eventgpt", quantization="int4")   # ~3.5 GB
videollava_4bit = load_model("videollava", quantization="int4")  # ~3.5 GB
# Total: ~7 GB, fits easily in 24 GB with room for KV cache
```

**Trade-off:** Some quality loss from quantization.

#### Option C: Offloading

```python
class OffloadedPrefillHiding:
    """
    Keep one model on GPU, other on CPU, swap as needed.
    """

    def __init__(self):
        self.draft = load_eventgpt(device="cuda")
        self.target = load_videollava(device="cpu")  # Offloaded

    def generate(self, ...):
        # Draft operations on GPU
        draft_tokens = self._generate_drafts_gpu(...)

        # Swap: draft to CPU, target to GPU
        self.draft.to("cpu")
        self.target.to("cuda")

        # Target operations on GPU
        output = self._verify_and_complete_gpu(...)

        return output
```

**Trade-off:** Swap time (~2-3s) negates some speedup.

---

## 6. Implementation Roadmap

### Phase 1: Proof of Concept (Week 1-2)
- [ ] Implement basic parallel prefill with CUDA streams
- [ ] Measure actual timing overlap
- [ ] Validate draft token generation in free window
- [ ] Test with sequential model loading (memory-safe)

### Phase 2: Feature-Level Alignment (Week 3-4)
- [ ] Train alignment layer on DSEC dataset
- [ ] Implement feature-level verification
- [ ] Measure acceptance rate improvement
- [ ] Compare to token-level baseline

### Phase 3: Optimization (Week 5-6)
- [ ] Implement quantized model loading
- [ ] Optimize batch verification
- [ ] Profile end-to-end performance
- [ ] Document speedup under various conditions

### Phase 4: Production Integration (Week 7-8)
- [ ] Integrate with EventGPT inference pipeline
- [ ] Add fallback for edge cases
- [ ] Benchmark on full DSEC test set
- [ ] Write technical report

---

## 7. Expected Outcomes

### Best Case (80% acceptance, quantized models)
- **Speedup:** 1.44x over baseline Video-LLaVA
- **First token latency:** Same as baseline (568ms)
- **Memory:** ~14 GB (both models in 4-bit)

### Realistic Case (60% acceptance, sequential loading)
- **Speedup:** 1.25-1.32x over baseline
- **First token latency:** Higher due to model swap
- **Memory:** ~13.5 GB (one model at a time)

### Conservative Case (40% acceptance)
- **Speedup:** 1.17x over baseline
- **Still worth it:** Free draft tokens provide some benefit regardless of acceptance

---

## 8. Relation to Existing Work

### Comparison to Standard Speculative Decoding

| Aspect | Standard Spec Decoding | Prefill Hiding |
|--------|----------------------|----------------|
| **Draft prefill cost** | Added to total time | Hidden (free) |
| **First token latency** | Higher | Same as baseline |
| **Free draft tokens** | 0 | ~27 tokens |
| **Complexity** | Medium | Higher |

### Comparison to Parallel Prefill Research

Our approach differs from existing parallel prefill work:
1. **Disaggregated prefill-decode** separates phases for throughput
2. **Prefill hiding** overlaps draft generation with target prefill for latency

### Novel Contribution

**First work to exploit draft model speed for "free" token generation during target prefill.**

---

## 9. References

1. EventGPT + Video-LLaVA Benchmark Results (internal, 2026-01-24)
2. EAGLE-3: Multi-layer feature fusion for speculative decoding (NeurIPS 2025)
3. SpecVLM: EAGLE-2 baseline for VLMs (Sept 2025)
4. Disaggregated Prefill and Decode (Perplexity 2025)

---

## 10. Conclusion

Prefill hiding represents a novel approach to speculative decoding that exploits the fundamental timing asymmetry between fast event-based models and slower video-based models. By hiding the draft model's prefill inside the target's prefill window, we can generate **~27 free draft tokens** that provide **1.3-1.7x speedup** depending on acceptance rates.

The approach is particularly attractive because:
1. Draft tokens are generated at zero additional latency cost
2. Even low acceptance rates (40%) provide meaningful speedup
3. Feature-level alignment can improve acceptance without changing model architectures
4. Memory constraints can be addressed through quantization or sequential loading

**Recommended first step:** Implement proof-of-concept with sequential loading to validate timing assumptions before optimizing for memory.

---

**Document Created:** 2026-01-26
**Author:** Alice Zhang
**Status:** Proposal - Ready for Implementation

