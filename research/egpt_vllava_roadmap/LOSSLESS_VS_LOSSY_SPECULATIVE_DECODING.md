# Lossless vs Lossy Speculative Decoding

## Table of Contents
1. [Standard Speculative Decoding is Lossless](#standard-speculative-decoding-is-lossless)
2. [The Mathematical Proof](#the-mathematical-proof)
3. [Speculative Sampling Algorithm](#speculative-sampling-algorithm)
4. [Lossy Speculative Decoding](#lossy-speculative-decoding)
5. [Comparison](#comparison)

---

## Standard Speculative Decoding is Lossless

### What "Lossless" Means

**Lossless speculative decoding** guarantees that the output distribution is **identical** to what the target model would produce with standard autoregressive decoding.

```
P_speculative(x1, x2, ..., xn) = P_target(x1, x2, ..., xn)
```

**Key property**: The final output is statistically indistinguishable from running the target model alone. You get the same quality, just faster.

### Why This Matters

- No degradation in output quality
- No need for quality-speed trade-off tuning
- Mathematically guaranteed equivalence
- Safe for production deployment

---

## The Mathematical Proof

### Setup

- **Target model**: p(x) - the large, slow model we want to match
- **Draft model**: q(x) - the small, fast model that proposes tokens
- **Goal**: Sample from p(x) while mostly running q(x)

### The Rejection Sampling Trick

For each drafted token x:

1. **Compute acceptance probability**:
   ```
   α(x) = min(1, p(x) / q(x))
   ```

2. **Accept with probability α(x)**:
   - If accepted: keep the token
   - If rejected: resample from adjusted distribution

3. **Adjusted distribution for rejection**:
   ```
   p_adjusted(x) = max(0, p(x) - q(x)) / Z

   Where Z = Σ_x max(0, p(x) - q(x)) = 1 - Σ_x min(p(x), q(x))
   ```

### Why This Works (Proof Sketch)

The probability of outputting token x is:

```
P(output = x) = P(accept x from draft) + P(reject draft) × P(resample x)
              = q(x) × min(1, p(x)/q(x)) + (1 - Σ_y q(y)×min(1, p(y)/q(y))) × p_adjusted(x)
```

**Case 1**: p(x) ≤ q(x)
```
P(output = x) = q(x) × p(x)/q(x) + ... × 0 = p(x)  ✓
```

**Case 2**: p(x) > q(x)
```
P(output = x) = q(x) × 1 + Z × (p(x) - q(x))/Z = q(x) + p(x) - q(x) = p(x)  ✓
```

**In both cases, P(output = x) = p(x)**. The speculative sampling exactly recovers the target distribution.

---

## Speculative Sampling Algorithm

### Algorithm (Leviathan et al., 2023)

```python
def speculative_sampling(target_model, draft_model, prompt, gamma=4):
    """
    Lossless speculative decoding.

    Args:
        target_model: Large model p(x)
        draft_model: Small model q(x)
        prompt: Input tokens
        gamma: Number of draft tokens per round

    Returns:
        tokens: Generated tokens (same distribution as target alone)
    """
    tokens = prompt

    while not done:
        # Step 1: Draft generates gamma tokens autoregressively
        draft_tokens = []
        draft_probs = []
        for _ in range(gamma):
            q = draft_model.get_probs(tokens + draft_tokens)
            x = sample(q)
            draft_tokens.append(x)
            draft_probs.append(q[x])

        # Step 2: Target evaluates all gamma tokens in ONE forward pass
        target_probs = target_model.get_probs_batch(tokens, draft_tokens)

        # Step 3: Accept/reject each token sequentially
        accepted = []
        for i, x in enumerate(draft_tokens):
            q_x = draft_probs[i]
            p_x = target_probs[i][x]

            # Acceptance probability
            alpha = min(1, p_x / q_x)

            if random() < alpha:
                # ACCEPT: keep draft token
                accepted.append(x)
            else:
                # REJECT: resample from adjusted distribution
                p_adjusted = np.maximum(0, target_probs[i] - draft_model.get_probs(tokens + accepted))
                p_adjusted /= p_adjusted.sum()
                x_new = sample(p_adjusted)
                accepted.append(x_new)
                break  # Stop after first rejection

        tokens = tokens + accepted

    return tokens
```

### Key Insight: Why It's Lossless

The magic is in the **adjusted distribution**:

```
p_adjusted(x) = max(0, p(x) - q(x)) / Z
```

This distribution:
1. Is zero where draft over-estimates (q(x) > p(x))
2. Is proportional to the "excess" where target prefers more (p(x) > q(x))
3. Exactly compensates for the draft's bias

**Result**: Every token comes from p(x), not q(x).

---

## Lossy Speculative Decoding

### What Makes It "Lossy"

**Lossy speculative decoding** trades output quality for speed by relaxing the strict acceptance criterion.

```
P_speculative(x) ≠ P_target(x)   (but hopefully close)
```

### Types of Lossy Methods

#### 1. Fuzzy Speculative Decoding (FSD)

**Idea**: Accept tokens when distributions are "close enough" instead of exact match.

```python
def fuzzy_acceptance(p, q, x, tau):
    """
    Accept if divergence is below threshold tau.
    """
    # Instead of: accept with prob min(1, p[x]/q[x])
    # Use: accept if Div(p, q) <= tau

    divergence = compute_divergence(p, q)  # e.g., KL, JS, or top-k diff
    return divergence <= tau
```

**Trade-off**: Higher τ → more acceptance → faster but lower quality

#### 2. Top-k Relaxation

**Idea**: Accept if draft token is in target's top-k predictions.

```python
def topk_acceptance(p, q, x, k=5):
    """
    Accept if x is in target's top-k tokens.
    """
    top_k_tokens = np.argsort(p)[-k:]
    return x in top_k_tokens
```

**Lossy because**: Accepts tokens that target wouldn't sample with high probability.

#### 3. Temperature Mismatch

**Idea**: Use different temperatures for draft and target.

```python
# Draft with higher temperature (more diverse)
q = softmax(logits_draft / T_draft)  # T_draft = 1.0

# Target with lower temperature (more focused)
p = softmax(logits_target / T_target)  # T_target = 0.7
```

**Lossy because**: Distribution mismatch even with standard acceptance.

#### 4. Assisted Decoding (Greedy)

**Idea**: Accept if draft's argmax matches target's argmax.

```python
def assisted_decoding(p, q, x):
    """
    Simple greedy matching.
    """
    return np.argmax(p) == np.argmax(q) == x
```

**Lossy because**: Ignores probability mass, only checks mode.

### PyramidSD's Lossy Variant (PSD_F)

PyramidSD with fuzzy acceptance at both stages:

```python
# Stage 1: Draft → Qualifier
accept_1 = Div(P_MQ, P_MD) <= tau_Q  # Lossy

# Stage 2: Qualifier → Target
accept_2 = Div(P_MT, P_MQ) <= tau_T  # Lossy

# Compound effect: errors propagate
```

**Why lossy**: Neither stage samples from the exact target distribution.

---

## Comparison

### Lossless vs Lossy Summary

| Aspect | Lossless SD | Lossy SD |
|--------|-------------|----------|
| **Output distribution** | Exact match to target | Approximate |
| **Quality guarantee** | Mathematically proven | Empirical |
| **Speedup** | Limited by acceptance rate | Can be higher |
| **Tuning required** | None (for quality) | τ, k, temperature |
| **Use case** | Production, safety-critical | Research, speed-critical |

### When to Use Each

**Use Lossless When**:
- Output quality is critical
- Deployment in production
- Need reproducibility
- Regulatory/safety requirements

**Use Lossy When**:
- Speed is more important than exact quality
- Acceptable quality degradation
- Exploratory/research settings
- Resource-constrained environments

### Speedup Comparison

| Method | Type | Typical Speedup | Quality |
|--------|------|-----------------|---------|
| Standard SD | Lossless | 2-3× | 100% |
| FSD (τ=0.3) | Lossy | 3-4× | ~98% |
| FSD (τ=0.5) | Lossy | 4-5× | ~95% |
| PyramidSD (PSD_A) | Mostly lossless | 1.44× over SD | ~99% |
| PyramidSD (PSD_F) | Lossy | 1.91× over SD | ~95% |

---

## Mathematical Formulation

### Lossless Acceptance Criterion

```
α(x) = min(1, p(x) / q(x))

Properties:
- α(x) ∈ [0, 1]
- α(x) = 1 when p(x) ≥ q(x) (target prefers at least as much)
- α(x) = p(x)/q(x) when p(x) < q(x) (proportional downweighting)
```

### Lossy Acceptance Criterion (Fuzzy)

```
α(x) = 1  if Div(p, q) ≤ τ
α(x) = 0  otherwise

Where Div can be:
- KL divergence: KL(p || q) = Σ p(x) log(p(x)/q(x))
- Top-k difference: |argmax_k(p) ∩ argmax_k(q)|
- Logit difference: |logit_p(x) - logit_q(x)|
```

### Why Lossy is Faster

**Lossless**: Must compute full distribution, do rejection sampling
```
Cost = γ × T_draft + T_target + T_resample
Effective tokens = E[accepted] + 1 (from resample)
```

**Lossy**: Simple threshold check, no resampling
```
Cost = γ × T_draft + T_target
Effective tokens = E[accepted] (often higher due to relaxed τ)
```

---

## Practical Implications

### For SpecVLM

SpecVLM is **lossless** by default:
- Uses standard speculative sampling
- Output matches target VLM exactly
- Speedup comes from efficient drafting, not quality trade-off

### For PyramidSD

- **PSD_A**: Near-lossless (assisted decoding fallback)
- **PSD_F**: Lossy (fuzzy at both stages, τ controls trade-off)

### For EventGPT Application

**Recommendation**: Start with lossless, move to lossy only if needed.

```python
# Lossless (recommended for quality)
if p(event_token) / q(event_token) >= random():
    accept(event_token)
else:
    resample_from_target()

# Lossy (if speed critical)
if divergence(event_features, video_features) <= tau:
    accept(event_draft)  # May not match target exactly
```

---

## Summary

| Question | Answer |
|----------|--------|
| **Why is standard SD lossless?** | Rejection sampling + adjusted distribution exactly recovers p(x) |
| **What makes lossy SD lossy?** | Relaxed acceptance (threshold, top-k, greedy) doesn't preserve p(x) |
| **Which is faster?** | Lossy (higher acceptance, no resampling overhead) |
| **Which is better quality?** | Lossless (mathematically guaranteed) |
| **What's the trade-off?** | Lossless: quality=100%, speed limited; Lossy: quality<100%, speed higher |

---

## References

- Leviathan et al., 2023 - "Fast Inference from Transformers via Speculative Decoding" (original lossless)
- Chen et al., 2023 - "Accelerating LLM Decoding with Speculative Sampling" (parallel proof)
- Holsman et al., 2025 - "Fuzzy Speculative Decoding" (lossy variant)
- PyramidSD - 3-model with fuzzy acceptance (lossy variant)

---

**Document Created**: January 28, 2026
