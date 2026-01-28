# Cross-Modal Speculative Decoding: Theoretical Foundations

**Research Focus**: EventGPT (sparse events) → VideoLLaVA (dense video)
**Document Created**: January 28, 2026

---

## Table of Contents
1. [Why Speculative Decoding Works](#why-speculative-decoding-works)
2. [Mathematical Foundation of Lossless SD](#mathematical-foundation-of-lossless-sd)
3. [Cross-Modal Extension: Theoretical Basis](#cross-modal-extension-theoretical-basis)
4. [Modality Gap and Distribution Alignment](#modality-gap-and-distribution-alignment)
5. [Entropy Gradient Across Modalities](#entropy-gradient-across-modalities)
6. [Cross-Attention Mechanisms for Modal Bridging](#cross-attention-mechanisms-for-modal-bridging)
7. [Acceptance Rate Bounds for Cross-Modal SD](#acceptance-rate-bounds-for-cross-modal-sd)
8. [Key Papers and Insights](#key-papers-and-insights)
9. [Theoretical Framework for EventGPT → VideoLLaVA](#theoretical-framework-for-eventgpt--videollava)

---

## Why Speculative Decoding Works

### The Core Insight: Parallelization of Verification

**Standard Autoregressive Decoding**:
```
Token 1 → Target forward pass (T_target)
Token 2 → Target forward pass (T_target)
Token 3 → Target forward pass (T_target)
...
Token n → Target forward pass (T_target)

Total time: n × T_target
```

**Speculative Decoding**:
```
Draft model generates γ tokens: γ × T_draft (fast, sequential)
Target model verifies ALL γ tokens: ~T_target (parallel, ONE forward pass!)

If α tokens accepted (α ≤ γ):
    Effective tokens per round: α + 1 (accepted + resampled)
    Time per round: γ × T_draft + T_target
```

### Why Parallel Verification is Possible

The target model's forward pass time scales **sublinearly** with sequence length due to:

1. **GPU Parallelism**: Matrix multiplications parallelize across tokens
2. **Memory Bandwidth**: The bottleneck is loading model weights, not computation
3. **KV Cache Reuse**: Cached key-value pairs from previous tokens are reused

```
T_target(γ tokens) ≈ T_target(1 token) × (1 + small_overhead)

NOT: T_target(γ tokens) = γ × T_target(1 token)
```

### Speedup Formula

```
Speedup = (Tokens generated) / (Time spent)

Standard: n tokens / (n × T_target) = 1/T_target

Speculative: n tokens / (R × (γ × T_draft + T_target))
           where R = n / (α + 1) rounds

Speedup ≈ (α + 1) / (γ × T_draft/T_target + 1)
```

**Key factors**:
- Higher acceptance rate α → more speedup
- Lower draft latency T_draft/T_target → more speedup
- Optimal γ balances acceptance degradation vs. parallelization gain

---

## Mathematical Foundation of Lossless SD

### The Rejection Sampling Guarantee

**Goal**: Sample from target distribution p(x) while mostly running draft model q(x)

**Algorithm**:
1. Draft proposes token x with probability q(x)
2. Accept with probability: `α(x) = min(1, p(x)/q(x))`
3. If rejected, resample from adjusted distribution

**Adjusted Distribution**:
```
p_adjusted(x) = max(0, p(x) - q(x)) / Z

Where Z = Σ_x max(0, p(x) - q(x)) = 1 - Σ_x min(p(x), q(x))
```

### Proof of Losslessness

The probability of outputting token x:

```
P(output = x) = P(accept x) + P(reject) × P(resample x)
              = q(x) × min(1, p(x)/q(x)) + Z × max(0, p(x)-q(x))/Z
```

**Case 1: p(x) ≤ q(x)**
```
P(output = x) = q(x) × p(x)/q(x) + Z × 0 = p(x) ✓
```

**Case 2: p(x) > q(x)**
```
P(output = x) = q(x) × 1 + Z × (p(x)-q(x))/Z = p(x) ✓
```

**In both cases, P(output = x) = p(x)**. The output distribution exactly matches the target.

### Why This Saves Time Despite Target Verification

The key insight that confuses many:

> "If we still need the target model, how do we save time?"

**Answer**: The target model runs **fewer forward passes**.

```
Standard (100 tokens):
    100 forward passes × T_target each

Speculative (100 tokens, α=3, γ=4):
    ~25 rounds × (4 × T_draft + 1 × T_target)
    = 100 × T_draft + 25 × T_target

If T_draft << T_target:
    Savings ≈ 75 × T_target
```

The draft model "front-loads" the sequential work, converting it to parallel verification.

---

## Cross-Modal Extension: Theoretical Basis

### From Text-Only to Multimodal SD

Standard SD assumes:
- Draft and target share the same vocabulary
- Both models produce probability distributions over tokens
- Acceptance is computed at the token level

**Cross-modal challenge**: Different modalities have different representations!

```
Event features ∈ R^{N_e × D_e}  (sparse, high temporal resolution)
Video features ∈ R^{N_v × D_v}  (dense, rich spatial information)
```

### Bridging the Modality Gap

**Key insight from MSD (Multimodal Speculative Decoding)**:

> "Text and visual tokens have fundamentally different characteristics and need to be processed separately during drafting."

**Visual tokens**: Encoded with bidirectional attention → related to ALL tokens
**Text tokens**: Encoded with causal attention → related only to previous tokens

### Cross-Modal Acceptance Criterion

For cross-modal SD, we need to define acceptance in feature space:

```python
# Standard (token-level):
accept = random() < min(1, p_target(x) / p_draft(x))

# Cross-modal (feature-level):
accept = divergence(f_target, f_draft) <= threshold

# Or probabilistic:
accept = random() < alignment_score(f_target, f_draft)
```

**Alignment score options**:
1. Cosine similarity in aligned feature space
2. KL divergence after projecting to probability space
3. Learned acceptance predictor

---

## Modality Gap and Distribution Alignment

### The Modality Gap Problem

Research shows that different modalities occupy distinct regions of the embedding space:

> "Despite the content diversity within images or texts, they show a relatively uniform distribution within each modality and a clear distribution gap between modalities."

**Implications for cross-modal SD**:
- Event features and video features live in different subspaces
- Direct feature comparison yields low acceptance rates
- Alignment is necessary before speculative verification

### Information Imbalance

From [Two Effects, One Trigger](https://arxiv.org/abs/2404.07983):

> "The driving factor behind both the modality gap and the object bias is an information imbalance between images and captions."

**For EventGPT → VideoLLaVA**:
- Events: Sparse, high temporal resolution, edge-focused
- Video: Dense, lower temporal resolution, texture-rich

The information content differs fundamentally, creating a natural entropy gradient.

### Measuring the Gap

**Wasserstein distance** is recommended for measuring modality gap:

```python
def modality_gap(event_features, video_features):
    """
    Measure distributional distance between modalities.
    """
    # Option 1: Wasserstein distance
    gap = wasserstein_distance(event_features, video_features)

    # Option 2: Maximum Mean Discrepancy (MMD)
    gap = mmd(event_features, video_features, kernel='rbf')

    # Option 3: Learned metric
    gap = alignment_network.predict_gap(event_features, video_features)

    return gap
```

### Alignment Strategies

1. **Contrastive Alignment**: Train with InfoNCE loss to align event-video pairs
2. **Projector Networks**: Learn MLP to map event features → video feature space
3. **Cross-Attention Fusion**: Use DREAM-style cross-attention for alignment

---

## Entropy Gradient Across Modalities

### PyramidSD Insight Extended to Modalities

PyramidSD observed that model size correlates with entropy:

| Model Size | Entropy | Confidence |
|------------|---------|------------|
| 1B (draft) | 2.433 | Low |
| 3B (qualifier) | 2.203 | Medium |
| 8B (target) | 1.900 | High |

**Cross-modal analog**:

| Modality | Information Density | Entropy | Role |
|----------|---------------------|---------|------|
| Events (sparse) | Low | High | Draft |
| Frames (medium) | Medium | Medium | Qualifier |
| Video (dense) | High | Low | Target |

### Why Sparse Modalities Make Good Drafters

1. **Fast encoding**: Fewer pixels/events → lower latency
2. **High uncertainty**: Sparse information → broader distribution → more exploration
3. **Complementary**: Captures temporal dynamics that video may miss

### Entropy-Adaptive Selection (DREAM insight)

From [DREAM](https://arxiv.org/abs/2505.19201):

> "Adaptive intermediate feature selection based on attention entropy guides efficient draft model training."

**Application**: Use entropy to decide when event features suffice vs. need video verification:

```python
def adaptive_verification(event_features, threshold=0.5):
    """
    Decide verification level based on draft entropy.
    """
    entropy = compute_entropy(event_features)

    if entropy < threshold:
        # High confidence → likely correct → skip video verification
        return "accept_event_draft"
    else:
        # Low confidence → need video verification
        return "verify_with_video"
```

---

## Cross-Attention Mechanisms for Modal Bridging

### DREAM's Cross-Attention Approach

Standard SD concatenates features. DREAM introduces cross-attention:

```python
class CrossModalAttention(nn.Module):
    """
    DREAM-style cross-attention for multimodal speculative decoding.
    """
    def forward(self, draft_features, target_cached_features):
        # Draft features as queries
        Q = self.q_proj(draft_features)

        # Target cached features as keys and values
        K = self.k_proj(target_cached_features)
        V = self.v_proj(target_cached_features)

        # Cross-attention: draft queries attend to target cache
        attention = softmax(Q @ K.T / sqrt(d_k))
        output = attention @ V

        return output
```

### Application to EventGPT → VideoLLaVA

```python
class EventVideoFusion(nn.Module):
    """
    Fuse event features with cached video features via cross-attention.
    """
    def __init__(self, event_dim, video_dim, hidden_dim):
        self.event_proj = nn.Linear(event_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8)

    def forward(self, event_features, video_kv_cache):
        # Project event features
        event_hidden = self.event_proj(event_features)

        # Cross-attend to video cache
        fused, _ = self.cross_attn(
            query=event_hidden,
            key=video_kv_cache.keys,
            value=video_kv_cache.values
        )

        return fused
```

### Benefits of Cross-Attention for Cross-Modal SD

1. **Selective alignment**: Queries select relevant target information
2. **Efficient caching**: Video features computed once, reused for verification
3. **Gradient flow**: End-to-end trainable alignment

---

## Acceptance Rate Bounds for Cross-Modal SD

### Standard SD Acceptance Rate

```
α = E[min(1, p(x)/q(x))] = Σ_x min(p(x), q(x))

Upper bound: α ≤ 1 (perfect match)
Lower bound: α ≥ 0 (complete mismatch)

Practical: α ∈ [0.6, 0.9] for well-trained drafts
```

### Cross-Modal Acceptance Rate

For cross-modal SD with alignment function A:

```
α_cross = E[accept | f_draft, f_target]
        = P(A(f_draft, f_target) ≤ threshold)
```

**Factors affecting α_cross**:
1. **Alignment quality**: Better A → higher α
2. **Modality similarity**: More similar content → higher α
3. **Threshold tuning**: Looser threshold → higher α but lower quality

### Theoretical Lower Bound

From entropy analysis:

```
α ≥ 2^{-H(p)} where H(p) is the entropy of target distribution
```

**For cross-modal**: The bound depends on mutual information:

```
α_cross ≥ f(I(X_event; X_video))

Higher mutual information → higher acceptance potential
```

### Empirical Expectations for EventGPT → VideoLLaVA

| Configuration | Expected α | Reasoning |
|---------------|-----------|-----------|
| Raw event features | 0.3-0.4 | Large modality gap |
| Aligned event features | 0.5-0.6 | Reduced gap via MLP |
| Cross-attention fusion | 0.6-0.7 | Selective alignment |
| With frame qualifier | 0.7-0.8 | Hierarchical refinement |

---

## Key Papers and Insights

### 1. MSD: Multimodal Speculative Decoding (May 2025)
- **Paper**: [arXiv:2505.14260](https://arxiv.org/abs/2505.14260)
- **Key insight**: Decouple text and visual token processing in drafting
- **Result**: 2.46× speedup on LLaVA-1.5-13B

### 2. DREAM: Cross-Attention for Multimodal SD (NeurIPS 2025)
- **Paper**: [arXiv:2505.19201](https://arxiv.org/abs/2505.19201)
- **Key insight**: Cross-attention + entropy-adaptive feature selection
- **Result**: Up to 3.6× speedup, outperforms EAGLE variants

### 3. ViSpec: Vision-Aware Speculative Decoding (NeurIPS 2025)
- **Paper**: [arXiv:2509.15235](https://arxiv.org/abs/2509.15235)
- **Key insight**: Compress visual tokens + inject global features
- **Result**: 3.22× speedup on VLMs

### 4. MASSV: Self-Distillation for VLM Draft Models (July 2025)
- **Paper**: [OpenReview](https://openreview.net/forum?id=ukDi9JyaL3)
- **Key insight**: Transform language-only models into multimodal drafters
- **Method**: Lightweight projector + self-distilled visual instruction tuning

### 5. Two Effects, One Trigger (ICLR 2024)
- **Paper**: [arXiv:2404.07983](https://arxiv.org/abs/2404.07983)
- **Key insight**: Information imbalance causes modality gap
- **Relevance**: Explains why cross-modal SD needs explicit alignment

### 6. VISTA: Cross-Modal Mutual Information (2025)
- **Paper**: [arXiv:2505.10917](https://arxiv.org/html/2505.10917v1)
- **Key insight**: Cross-entropy alignment degrades with sequence length
- **Solution**: Maximize cross-modal mutual information explicitly

---

## Theoretical Framework for EventGPT → VideoLLaVA

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 CROSS-MODAL SPECULATIVE DECODING                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STAGE 1: Event Drafting (2-5ms)                               │
│  ├── Input: Sparse event stream                                 │
│  ├── Model: EventGPT encoder (frozen)                          │
│  ├── Output: Event features f_e ∈ R^{N_e × D}                  │
│  └── Latency: ~2ms (sparse computation)                        │
│                                                                 │
│  STAGE 2: Feature Alignment                                     │
│  ├── Alignment MLP: f_e → f_aligned                            │
│  ├── Cross-attention with VideoLLaVA cache (if available)      │
│  └── Output: Aligned draft features                             │
│                                                                 │
│  STAGE 3: Acceptance Decision                                   │
│  ├── Compute: α = alignment_score(f_aligned, f_video_expected) │
│  ├── If α > threshold: Accept event draft (fast path)          │
│  └── Else: Fall back to video encoding (slow path)             │
│                                                                 │
│  STAGE 4: Video Verification (only if needed, ~50ms)           │
│  ├── Encode video frames with VideoLLaVA                       │
│  ├── Verify draft against video features                       │
│  └── Resample if rejected                                       │
│                                                                 │
│  STAGE 5: LLM Decoding                                          │
│  ├── Use accepted features for text generation                 │
│  └── Standard speculative decoding for text tokens             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Mathematical Formulation

**Objective**: Minimize expected latency while maintaining output quality

```
min E[Latency] = P(accept) × T_event + P(reject) × (T_event + T_video)
              = T_event + P(reject) × T_video

Subject to: Quality(output) ≥ Quality(VideoLLaVA baseline)
```

**Optimization**:
- Maximize P(accept) via alignment training
- Minimize T_event via efficient event encoding
- Maintain quality via lossless acceptance criterion

### Training Strategy

**Phase 1: Alignment Pretraining**
```python
loss_align = MSE(MLP(event_features), video_features)
```

**Phase 2: Acceptance Distillation**
```python
loss_accept = BCE(predictor(event_features), should_accept_labels)
```

**Phase 3: End-to-End Fine-tuning**
```python
loss_total = loss_LM + λ_align × loss_align + λ_accept × loss_accept
```

### Expected Performance

| Metric | Baseline (VideoLLaVA) | With Event SD |
|--------|----------------------|---------------|
| Vision latency | 50ms | 10ms (avg) |
| Acceptance rate | N/A | 0.65-0.75 |
| End-to-end speedup | 1.0× | 2.5-3.5× |
| Quality degradation | 0% | <1% (lossless) |

---

## Open Research Questions

### Theoretical Questions

1. **Optimal modality hierarchy**: How many intermediate modalities are beneficial?
2. **Acceptance rate ceiling**: What is the theoretical maximum α for event→video?
3. **Information-theoretic bounds**: How does I(event; video) limit speedup?

### Practical Questions

1. **Training data**: How to generate paired event-video data efficiently?
2. **Dynamic threshold**: Should acceptance threshold adapt per-query?
3. **Streaming SD**: How to apply cross-modal SD to real-time event streams?

### Empirical Questions

1. **Task dependence**: Which VQA tasks benefit most from event drafting?
2. **Event quality**: How does event camera noise affect acceptance rate?
3. **Temporal alignment**: How to handle asynchronous event-video timestamps?

---

## Summary

### Key Theoretical Insights

1. **SD saves time via parallelization**, not by avoiding target computation
2. **Cross-modal SD requires explicit alignment** due to modality gap
3. **Entropy gradient across modalities** enables hierarchical drafting
4. **Cross-attention is superior to concatenation** for modal fusion
5. **Acceptance rate bounded by mutual information** between modalities

### Actionable Next Steps

1. Implement alignment MLP for EventGPT → VideoLLaVA features
2. Measure baseline acceptance rate without alignment
3. Apply DREAM-style cross-attention for improved fusion
4. Benchmark against VideoLLaVA baseline on DSEC dataset

---

## References

- [MSD: Speculative Decoding Reimagined for MLLMs](https://arxiv.org/abs/2505.14260)
- [DREAM: Cross-Attention for Multimodal SD](https://arxiv.org/abs/2505.19201)
- [ViSpec: Vision-Aware Speculative Decoding](https://arxiv.org/abs/2509.15235)
- [MASSV: Multimodal Adaptation for VLM SD](https://openreview.net/forum?id=ukDi9JyaL3)
- [Two Effects, One Trigger: Modality Gap Analysis](https://arxiv.org/abs/2404.07983)
- [VISTA: Cross-Modal Mutual Information](https://arxiv.org/html/2505.10917v1)
- [SpecVLM: Fast SD in VLMs](https://arxiv.org/abs/2509.11815)
- [Speculative Decoding Papers Repository](https://github.com/hemingkx/SpeculativeDecodingPapers)

---

**Document Created**: January 28, 2026
**Research Focus**: Theoretical foundations for EventGPT → VideoLLaVA cross-modal speculative decoding
