# Hidden-State vs Token-Level Metrics for Cross-Modal Speculative Decoding

> **Last Updated:** 2026-02-06

This document explains the two metric levels used in `evaluate_e2e_sd.py` and why both matter.

---

## Why Two Metric Levels?

Hidden-state alignment (cosine similarity) and token-level agreement (argmax match) measure different things:

| Aspect | Hidden-State Level | Token Level |
|--------|-------------------|-------------|
| **What it measures** | Geometric closeness of representation vectors | Whether the same discrete token is selected |
| **Metric** | Cosine similarity > threshold | argmax(LM_head(aligned)) == argmax(LM_head(vl)) |
| **Sensitivity** | Smooth, continuous (0.0 to 1.0) | Binary (match or no match) |
| **Threshold** | Configurable (0.80, 0.85, 0.90, 0.95) | Implicit (argmax is strict) |
| **Failure mode** | Over-optimistic: high cos_sim does not guarantee token match | Over-pessimistic: token match ignores near-misses |
| **SD relevance** | Approximation of acceptance probability | Ground truth for SD acceptance |

### The Gap

Hidden-state metrics consistently show higher acceptance than token-level metrics:
- Hidden-state Accept@0.90 might be ~20%
- Token match rate might be ~2-5%

This is because the LM head amplifies small representation differences. Two vectors with cos_sim=0.92 may still project to different argmax tokens if they're near a decision boundary in the 32k-dimensional token space.

### Why Hidden-State Metrics Still Matter

1. **Training signal:** Adapter training optimizes cosine similarity / MSE loss. Token-level feedback is too sparse for gradient-based optimization.
2. **Top-K match:** Even if argmax differs, the correct token often appears in top-5 (useful for top-k SD variants).
3. **Continuous progress tracking:** cos_sim improves smoothly during training, while token match jumps discretely.
4. **Adapter comparison:** cos_sim reliably ranks adapter quality across L1-L5.

---

## Metric Definitions

### A. Hidden-State Level

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| cos_sim | `(h_aligned . h_vl) / (||h_aligned|| * ||h_vl||)` | 1.0 = identical direction |
| Accept@tau | `mean(cos_sim > tau)` | Fraction of positions passing threshold |
| Consecutive@tau | `cumprod(cos_sim > tau).sum()` | Tokens accepted before first rejection |

### B. Token Level

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Token Match | `argmax(LM_head(aligned)) == argmax(LM_head(vl))` | Exact argmax agreement |
| Top-5 Match | `argmax(LM_head(aligned)) in top5(LM_head(vl))` | Draft token in target's top-5 |
| Consecutive Token Match | `cumprod(token_match).sum()` | Sequential token matches from position 0 |
| Prefill Acceptance | Token match over gamma_prefill positions | Prefill phase effectiveness |
| Decode Acceptance | Token match over gamma_decode positions | Decode phase effectiveness (L5) |

---

## Two-Phase SD Speedup Model

### Phase 1: Prefill Hiding (All Adapters L1-L5)

```
Timeline:
  VL:   |---- VL Prefill (310ms) ----|
  EGPT: |-- EGPT Prefill (130ms) --|- draft tokens --|
                                    ^
                                    free_time = 180ms
                                    gamma_prefill = floor(180 / 25) = 7 tokens
```

- During VL's long prefill (~310ms), EGPT can generate draft tokens
- Accepted tokens = min(consecutive_accepts, gamma_prefill)
- These tokens skip VL's decode phase entirely

### Phase 2: Decode (L5 Only)

| Adapter | Draft Cost | Decode Acceleration |
|---------|-----------|---------------------|
| L1-L4 | EGPT full decode (~25ms/tok) | NONE (same cost as VL) |
| L5 EAGLE | Adapter forward (~3ms/tok) | YES (8x cheaper drafting) |

L5 EAGLE generates draft hidden states autoregressively at ~3ms/token, enabling standard speculative decoding during the decode phase.

### E2E Speedup Formula

```
Baseline = VL_prefill + N * VL_per_token

L1-L4 SD = max(VL_prefill, EGPT_total) + (N - prefill_accepted) * VL_per_token

L5 SD    = max(VL_prefill, EGPT_total) + decode_iterations * (gamma * L5_draft + VL_verify)
         where decode_iterations = (N - prefill_accepted) / (accepted_per_iter + 1)
```

---

## Practical Implications

1. **L1 adapter is sufficient for prefill hiding.** Since L1-L4 only gain speedup from prefill hiding, a cheap L1 adapter (~2M params) captures most of the benefit.

2. **L5 is needed for decode acceleration.** Only L5's EAGLE-style autoregressive prediction enables cheaper drafting than full model inference.

3. **Token-level metrics are the true SD metric.** Always report token match rate alongside hidden-state metrics. The hidden-state metrics are useful for training but misleading for speedup estimation.

4. **Per-position analysis reveals quality degradation.** Both hidden-state and token-level acceptance typically degrade at later positions, affecting consecutive accept counts.
