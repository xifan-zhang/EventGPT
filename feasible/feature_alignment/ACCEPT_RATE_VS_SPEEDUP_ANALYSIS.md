# Accept Rate vs Speedup Analysis

> **Updated:** 2026-02-07 — E2E wall-clock benchmark results (L4, 10,970 samples)
> **Previous:** 2026-02-06 — Theoretical projections (L1, cos_sim-based estimates)

---

## UPDATE: Actual E2E Wall-Clock Results (2026-02-07)

### Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | `my_egpt_dsec_seq_1s` (10,970 samples after 3 warmup) |
| Adapter | L4 (100M params, 50 epochs, val_loss=1.2458) |
| Max tokens | 30, gamma=5 |
| Checkpoint | `tasks/L4/L4_20260206_192256/best_model.pt` |

### Results: 1.03x Speedup (Not 5.77x)

| Config | Prefill (ms) | Decode (ms) | Total (ms) | Accept | Speedup |
|--------|-------------|------------|-----------|--------|---------|
| VL baseline | 317 | 419 | 736 | --- | 1.00x |
| L4+VL SD | 317 | 404 | 721 | 21.2% | **1.03x** |

### Why Speedup Is So Low

**Pipeline per sample:**
1. VL prefill: 317 ms (identical to baseline, unchanged by SD)
2. EGPT generates ~22 draft tokens during VL prefill gap (free)
3. VL verify batch: one forward pass over ~21 drafts → **~70 ms**
4. VL autoregressive: remaining tokens after last accepted

**The math:**

| Component | Value |
|-----------|-------|
| VL per-token cost (autoregressive) | 14.0 ms |
| Mean drafted tokens | 20.9 |
| Mean accepted tokens | 4.4 |
| Tokens saved per sample | 5.4 (accepted + 1 bonus) |
| Value of saved tokens | 5.4 × 14.0 = **75 ms** |
| Verify batch cost | **~70 ms** |
| **Net saving** | **~5 ms** per sample |

### Root Causes

**1. Prefill is 43% of wall time**
- Prefill: 317 ms, Decode: 419 ms → SD only helps decode phase
- Even with perfect acceptance, max total speedup = 2.3x

**2. Only 21% acceptance rate**
- Mean 4.4 accepted of 20.9 drafted per iteration
- 32% of samples get 0-1 accepted → **net slowdown** (verify > savings)

**3. Verify batch nearly cancels savings**
- 70 ms verify cost eats almost all of the 75 ms saved
- Net saving only ~5 ms per 736 ms sample

### Acceptance Distribution (10,970 samples)

```
Accepted  Count    %     Speedup
   0       1000   9.1%   0.94x  ← slower
   1       2456  22.4%   0.96x  ← slower
   2        248   2.3%   0.97x  ← slower
   3        899   8.2%   0.99x  ← breakeven
   4        753   6.9%   1.01x
   5       1821  16.6%   1.03x
   6       1135  10.3%   1.06x
   7        950   8.7%   1.08x
   8        595   5.4%   1.10x
   9        309   2.8%   1.13x
  10        189   1.7%   1.15x
  11        118   1.1%   1.22x
  12        486   4.4%   1.25x
```

31.5% of samples are **slower** with SD (0-1 accepted). The average is dragged down by these.

### Projected Speedup at Higher Acceptance Rates

| Accept% | Accepted | AR-remain | Decode (ms) | Total (ms) | Speedup |
|---------|----------|-----------|-------------|-----------|---------|
| 21% | 4.4 | 24.6 | 414 | 730 | **1.01x** ← current |
| 30% | 6.3 | 22.7 | 388 | 704 | 1.04x |
| 40% | 8.4 | 20.6 | 358 | 675 | 1.09x |
| 50% | 10.4 | 18.6 | 329 | 646 | 1.14x |
| 60% | 12.5 | 16.5 | 300 | 617 | 1.19x |
| 70% | 14.6 | 14.4 | 271 | 588 | 1.25x |
| 80% | 16.7 | 12.3 | 242 | 558 | 1.32x |
| 90% | 18.8 | 10.2 | 213 | 529 | 1.39x |
| 100% | 20.9 | 8.1 | 183 | 500 | 1.47x |

### Correction to Previous Estimates

The earlier 5.77x estimate was based on:
- Theoretical consecutive accept counts from cos_sim thresholds
- Assumption that prefill hiding contributes directly to speedup
- Not accounting for verify batch overhead

**Actual finding:** Prefill hiding generates free draft tokens, but VL acceptance rate (21%) means most are rejected. The verify overhead (~70 ms) nearly cancels the savings from accepted tokens. The bottleneck is **acceptance rate**, not token generation speed.

### Next Steps
- L4 retraining with 300 epochs (previous 50 epochs didn't converge, T_max bug fixed)
- B1 VLM-only EAGLE baseline to determine upper bound (cross-modal gap vs adapter capacity)

---

## Original Analysis (2026-02-06) — Theoretical Projections

> **Note:** The projections below were based on cos_sim thresholds before E2E benchmarks existed. Actual E2E results above show significantly lower speedup due to verify overhead and prefill-dominated latency.

### Previous TL;DR (Theoretical)

**20% overall Accept@0.90 sounds low but yields 5.77x speedup** because:
1. Positions 0-2 have ~100% acceptance (consecutive metric >> overall metric)
2. Parallel prefill hides ~500ms of Video-LLaVA latency
3. L1 adapter overhead is < 0.5ms

| What 20% means | Same-model SD | Cross-modal SD (ours) |
|-----------------|---------------|----------------------|
| Classification | Marginal | **Useful** |
| Expected speedup | ~1.5x | **5.77x** |
| Why different | No prefill hiding | Parallel prefill + low overhead |

---

## 1. Prior Works: Acceptance Rates and Speedups

### Same-Model Speculative Decoding

| Method | Venue | Avg Accepted Tokens | Speedup | Accept Rate |
|--------|-------|---------------------|---------|-------------|
| Speculative Sampling | PMLR 2023 | 2.27 | 1.93x | ~50% |
| Medusa | 2024 | 2.59-3.50 | 2.07-2.93x | ~55-65% |
| EAGLE | ICML 2024 | 3.62-4.42 | 2.78-3.76x | 67-85% |
| EAGLE-2 | EMNLP 2024 | 4.83 | 4.26x | ~75% |
| HASS | ICLR 2025 | 5.15-5.21 | 3.24-4.05x | ~70%+ |
| EAGLE-3 | NeurIPS 2025 | **6.65** | **5.58x** | **80%+** |

### VLM-Specific Speculative Decoding (SpecVLM)

| Target Model | Method | Avg Accepted | Speedup |
|--------------|--------|-------------|---------|
| LLaVA-1.5-7B | EagleVLM | 3.59 | 2.01x |
| LLaVA-1.5-7B | SpecVLM | 3.67 | 2.11x |
| LLaVA-1.6-13B | EagleVLM | 4.09 | 2.29x |
| LLaVA-1.6-13B | SpecVLM | **4.15** | **2.39x** |

### Our Cross-Modal SD

| Metric | L1 Pilot (100 samples) | L1 Full (11k val) |
|--------|------------------------|-------------------|
| Cosine Similarity | 0.764 | 0.772 |
| Accept@0.90 (overall) | 19.5% | 20.6% |
| **Consecutive mean @0.90** | **6.35 tokens** | TBD |
| **Estimated speedup** | **5.77x** | TBD |

---

## 2. The Critical Distinction: Overall vs Consecutive

**Speculative decoding STOPS at the first rejection.** The overall Accept@0.90 is misleading.

```
Position:      0    1    2    3    4    5    6    7    8    9
cos_sim:      0.99 0.99 0.99 0.90 0.90 0.89 0.88 0.84 0.83 0.78
Accept@0.90:   Y    Y    Y    Y    Y    N    N    N    N    N

Overall Accept@0.90 = 5/10 = 50%    <-- What we report
Consecutive accepts = 5 tokens       <-- What determines speedup
```

### Our Per-Position Acceptance (Pilot Study, threshold 0.90)

| Position | Mean cos_sim | Accept@0.90 | Note |
|----------|-------------|-------------|------|
| 0 | 0.989 | **100.0%** | Vision-conditioned |
| 1 | 0.985 | **100.0%** | Vision-conditioned |
| 2 | 0.985 | **99.9%** | Vision-conditioned |
| 3 | 0.901 | 69.2% | Transition zone |
| 4 | 0.901 | 65.3% | |
| 5 | 0.887 | 55.6% | |
| 6 | 0.879 | 50.4% | |
| 7 | 0.839 | 42.3% | |
| 8 | 0.825 | 26.4% | Text divergence |
| 9 | 0.782 | 18.2% | |

**Positions 0-2 are nearly perfect** (~100%). This creates long consecutive accept chains from the start, yielding 6.35 mean consecutive tokens despite only 19.5% overall.

### Why Early Positions Align Well

Early positions correspond to vision-conditioned tokens where both EventGPT (events) and Video-LLaVA (RGB frames) encode similar scene information. Later positions (pure text generation) diverge due to the modality gap and different generation trajectories.

---

## 3. How cos_sim Maps to Token-Level Acceptance

### Mapping from Literature (Synthesized from ASD, EAGLE, HASS, SpecVLM)

| cos_sim | Token Match Prob | Avg Chain (gamma=5) | SD Speedup |
|---------|-----------------|---------------------|------------|
| 0.70 | ~15% | ~2 | ~1.5x |
| 0.75 | ~20% | ~4 | ~2x |
| **0.77** | **~22%** | **~5** | **~2.5x** |
| 0.80 | ~28% | ~6 | ~3x |
| 0.85 | ~38% | ~10 | ~4x |
| 0.90 | ~55% | ~15 | ~5x |
| 0.95 | ~75% | ~25 | ~7x |

**Our L1 at cos_sim 0.77 should yield ~2.5x in same-model SD.** But we get 5.77x because of cross-modal parallel prefill.

### EAGLE's Graceful Degradation (from paper, Table 2)

EAGLE shows that feature prediction errors don't cascade badly:

| Draft Position | Accept Rate (temp=0, LLaMA2-70B) |
|----------------|----------------------------------|
| 0-alpha (perfect features) | 75% |
| 1-alpha (1 predicted) | 69% |
| 2-alpha | 65% |
| 3-alpha | 64% |
| 4-alpha | 64% |

Only 2-3% drop per position. This means even imperfect adapters can maintain decent acceptance at early positions.

---

## 4. Why Cross-Modal SD Outperforms at Lower Accept Rates

### The Parallel Prefill Advantage

```
Same-model SD (SpecVLM):
  [Target encode 600ms] → [Draft 30ms] → [Verify 20ms] = 650ms total
  Sequential -- draft DEPENDS on target features

Cross-modal SD (ours):
  [EventGPT  72ms]──┐
                     ├── max(72, 597) = 597ms + [Verify 20ms] = 617ms
  [VideoLLaVA 597ms]─┘
  Parallel -- draft is INDEPENDENT
```

Even though SpecVLM has higher acceptance (3.67-4.15 accepted tokens), it can't hide the target's 600ms prefill. We accept fewer tokens but save ~500ms of prefill latency.

### Speedup Formula Comparison

| Component | Same-model (SpecVLM) | Cross-modal (ours) |
|-----------|---------------------|-------------------|
| Draft overhead | 80ms (encode) + 30ms (draft) | ~0.5ms (adapter only) |
| Prefill saved | 0ms (sequential) | ~525ms (parallel) |
| Tokens accepted | 4.15 | 6.35 |
| **Total speedup** | **2.39x** | **5.77x** |

### The Cross-Modal Equation

From the threshold analysis:
```
Same-model:  cos_sim = 0.92, accept = 70%, but overhead = 0.3  --> Speedup = 3.1x
Cross-modal: cos_sim = 0.77, accept = 20%, but overhead = 0.01 --> Speedup = 5.8x
                                                       ^
                                           EventGPT is MUCH faster
```

---

## 5. Is 20% Accept@0.90 Enough? Verdict by Use Case

| Use Case | Accept@0.90 Needed | Our Status | Viable? |
|----------|-------------------|------------|---------|
| Cross-modal prefill SD | > 15% | 20% | **YES** |
| Same-model decode SD | > 40% | 20% | No |
| Lossless SD (token verify) | > 15% consecutive | 100% at pos 0-2 | **YES** |
| Production (> 3x speedup) | > 30% same-model | 20% + parallel | **YES** (5.77x) |

**For our specific cross-modal pipeline, 20% Accept@0.90 is already operationally useful.** The parallel prefill advantage compensates for lower hidden-state alignment quality.

---

## 6. What Improvement Can We Expect?

### SpecVLM Training Scaling (from paper, Figure 5)

| Epoch | Accepted Tokens | Speedup |
|-------|-----------------|---------|
| 1 | 3.68 | 2.20x |
| 3 | 4.82 | 2.54x |
| **5** | **5.27** | **2.66x** |

More training consistently improves acceptance. Our L1 is still improving at epoch 12.

### Projected Results for Larger Adapters

| Adapter | Params | Expected cos_sim | Expected Accept@0.90 | Expected Consecutive | Expected Speedup |
|---------|--------|-----------------|----------------------|---------------------|------------------|
| **L1** | **2.1M** | **0.77 (actual)** | **20% (actual)** | **~6 tokens** | **~5.8x** |
| L2 | 6.3M | ~0.80 | ~25% | ~10 | ~6-7x |
| L3 | 16.8M | ~0.82 | ~28% | ~12 | ~6-8x |
| L4 | 100M | ~0.85 | ~32% | ~15 | ~7-9x |
| **L5** | **100M** | **~0.88** | **~40%** | **~20** | **~10-15x** |

### HASS Insight (ICLR 2025)

> "Acceptance rate is primarily influenced by alignment on DESIRED tokens, not entire vocabulary."

This suggests focusing adapter training on early/high-impact positions could improve consecutive acceptance without requiring overall Accept@0.90 to increase. A position-weighted loss could help.

---

## 7. Comparison Table: Our L1 vs All Prior Works

| System | Type | Accept Metric | Consec. Tokens | Speedup | cos_sim |
|--------|------|--------------|----------------|---------|---------|
| Spec. Sampling | Same-model token | ~50% token | 2.27 | 1.93x | N/A |
| Medusa | Same-model multi-head | ~60% token | 3.50 | 2.93x | N/A |
| EAGLE | Same-model feature | 67-85% token | 3.94 | 3.07x | ~0.90 (implied) |
| EAGLE-3 | Same-model multi-layer | 80%+ token | 6.65 | 5.58x | ~0.95 (implied) |
| HASS | Same-model aligned | 70%+ token | 5.21 | 4.05x | ~0.92 (implied) |
| EagleVLM | Same-VLM feature | ~65% token | 4.09 | 2.29x | N/A |
| SpecVLM | Same-VLM distilled | ~65% token | 4.15 | 2.39x | N/A |
| **Our L1** | **Cross-modal feature** | **20% @0.90** | **6.35** | **5.77x** | **0.77** |

**Takeaway:** Our low overall Accept@0.90 is deceptive. The combination of front-loaded acceptance + parallel prefill + tiny adapter overhead makes cross-modal SD competitive with EAGLE-3 despite much lower hidden-state alignment quality.

---

## References

1. EAGLE (Li et al., 2024) - ICML 2024
2. EAGLE-3 (Li et al., 2025) - NeurIPS 2025
3. Medusa (Cai et al., 2024)
4. ASD (arXiv 2410.01028) - Adaptive Self-Speculative Decoding
5. HASS (ICLR 2025) - Harmonized Speculative Sampling
6. SpecVLM (2025) - Fast Speculative Decoding for VLMs
7. Leviathan et al. (2023) - Original SD Theory
