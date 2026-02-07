# Hybrid Speculative Decoding: Combining Embedding and Token-Level Approaches

## Overview

Hybrid speculative decoding combines **token-level** and **feature/embedding-level** approaches to leverage the strengths of both. These methods often use multiple draft strategies or combine different prediction mechanisms.

---

## Key Methods

### 1. Chimera (2024)
**Paper:** "Chimera: A Lossless Decoding Method for Accelerating Large Language Models Inference by Fusing all Tokens"
- **arXiv:** [2402.15758](https://arxiv.org/abs/2402.15758)
- **Authors:** Chen et al. (South China University of Technology, HKUST)
- **Speedup:** ~2.7x
- **Citations:** 77+

**Architecture Components:**
1. **Trigram Encoder** - Short-range dependency modeling
   - Pre-computed lookup table for trigram hidden states
   - Efficient feature extraction from three consecutive positions

2. **Full Context Encoder** - Long-range dependency capture
   - Single Transformer block
   - Fuses trigram encoder outputs with original LLM hidden states
   - Captures dependencies across the full context

3. **Residual Decoding Heads** - Parallel draft generation
   - Multiple heads predict tokens at different offset positions
   - Similar to Medusa's multi-head approach

**Key Innovation:** Combines efficient n-gram modeling (token-level patterns) with deep feature fusion (embedding-level) for lossless acceleration.

---

### 2. SpecBranch (2025)
**Paper:** "Speculative Decoding via Hybrid Drafting and Rollback-Aware Branch Parallelism"
- **arXiv:** [2506.01979](https://arxiv.org/abs/2506.01979)
- **OpenReview:** [Forum](https://openreview.net/forum?id=BrnlCSqO6n)
- **Authors:** Shen et al.
- **Published:** June 2025

**Core Concepts:**
1. **Hybrid Drafting** - Combines explicit and implicit draft models
   - Improves prediction accuracy
   - Reduces compounding errors from single draft model

2. **Rollback-Aware Branch Parallelism**
   - Inspired by branch prediction in modern CPUs
   - Parallel speculative branches hedge against rejections
   - Addresses static draft length limitations

3. **Adaptive Draft Lengths**
   - Jointly orchestrates draft lengths with hybrid draft models
   - More aware of rollback and rejected tokens

**Why Hybrid Matters:**
Traditional speculative decoding uses static draft lengths that don't account for rollbacks. SpecBranch introduces parallel speculative branches to preemptively handle likely rejections, combining multiple drafting strategies.

---

### 3. Gumiho (2025)
**Paper:** "Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding"
- **arXiv:** [2503.10135](https://arxiv.org/abs/2503.10135)
- **AMD ROCm Blog:** [Post](https://rocm.blogs.amd.com/software-tools-optimizations/gumiho/README.html)

**Key Idea:** Prioritizes early tokens in generation with a hybrid architecture that adapts to the decoding process.

---

### 4. GLIDE with CaPE (2024)
**Paper:** "Glide with a Cape: A Low-Hassle Method to Accelerate Speculative Decoding"
- **arXiv:** [2402.02082](https://arxiv.org/abs/2402.02082)
- **Authors:** Du et al. (Singapore Management University, NUS, HK PolyU, Harbin Institute of Technology, Tencent AI Lab)
- **Speedup:** 2.5-2.6x

**Components:**
1. **GLIDE (Glimpse Draft Model)**
   - Reuses target model's KV cache via cross-attention
   - No need for separate draft model training from scratch

2. **CAPE (Proposal Expansion)**
   - Uses draft model confidence scores
   - Leverages "dark knowledge" from lower-confidence candidates
   - Expands proposals based on confidence

**Hybrid Aspect:** Combines KV cache reuse (feature-level) with proposal expansion (token-level selection) for acceleration.

---

## Comparison Table: Hybrid Approaches

| Method | Year | Hybrid Strategy | Token Component | Feature Component | Speedup |
|--------|------|-----------------|-----------------|-------------------|---------|
| **Chimera** | 2024 | N-gram + Feature Fusion | Trigram Encoder | Full Context Encoder | 2.7x |
| **SpecBranch** | 2025 | Explicit + Implicit Drafting | Multiple Draft Models | Rollback-Aware Branching | TBD |
| **GLIDE+CaPE** | 2024 | KV Cache + Proposal Expansion | CaPE Token Selection | GLIDE Cross-Attention | 2.5-2.6x |
| **Gumiho** | 2025 | Early Token Prioritization | Early Token Focus | Adaptive Architecture | TBD |

---

## Common Patterns in Hybrid Approaches

### 1. Multi-Stage Drafting
```python
# Pseudo-code for hybrid drafting
def hybrid_draft(context):
    # Stage 1: Fast token-level drafting (n-gram, small model)
    token_draft = fast_token_model(context)

    # Stage 2: Feature-level refinement
    feature_draft = feature_model(context, token_draft)

    # Stage 3: Combine or select best candidates
    final_draft = combine_or_select(token_draft, feature_draft)

    return final_draft
```

### 2. Adaptive Branch Selection
- Use simple methods for easy tokens
- Use complex methods for difficult tokens
- Reduce computation while maintaining quality

### 3. Cascaded Verification
- Verify token-level drafts first (fast rejection)
- Verify feature-level drafts for remaining candidates
- Parallel verification where possible

---

## Advantages of Hybrid Approaches

1. **Best of Both Worlds**
   - Token-level: Fast, simple, interpretable
   - Feature-level: Rich semantic information
   - Hybrid: Adaptive selection based on context

2. **Reduced Error Propagation**
   - Multiple drafting strategies reduce single-point failures
   - Branch parallelism provides backup candidates

3. **Better Resource Utilization**
   - Use cheap methods when they work
   - Fall back to expensive methods only when needed
   - Dynamic allocation of computational budget

---

## Challenges

1. **Complexity**
   - More components to design and tune
   - Higher engineering overhead

2. **Coordination Overhead**
   - Synchronizing multiple draft strategies
   - Managing branch prediction and rollback

3. **Memory Usage**
   - Storing multiple draft candidates
   - KV cache management for cross-attention

---

## Cascaded Speculative Decoding

### Concept: Draft A → Draft B → Target Model

A natural extension of hybrid approaches is **cascaded speculative decoding**, where multiple draft models are chained:

```
Input Context
      │
      ▼
┌─────────────┐
│  Model A    │  ← Smallest/Fastest (e.g., 1B)
│ (Draft #1)  │
└──────┬──────┘
       │ Draft tokens/features
       ▼
┌─────────────┐
│  Model B    │  ← Medium Draft Model (e.g., 7B)
│ (Draft #2)  │    verifies Model A's drafts
└──────┬──────┘       + generates its own refined drafts
       │ Refined drafts
       ▼
┌─────────────┐
│ Target      │  ← Largest Model (e.g., 70B)
│  Model      │     verifies Model B's drafts
└─────────────┘
```

### Mathematical Formulation

Given:
- **P_A**: Distribution of Model A (smallest)
- **P_B**: Distribution of Model B (medium)
- **P_T**: Distribution of Target Model (largest)

**Stage 1: Model A → Model B**
```
For k draft tokens from Model A:
Acceptance probability for token t_i:
    α_A(i) = min(1, P_B(t_i) / P_A(t_i))

Expected accepted tokens: k_A = Σ α_A(i)
```

**Stage 2: Model B → Target Model**
```
For m draft tokens from Model B:
Acceptance probability for token t_j:
    α_B(j) = min(1, P_T(t_j) / P_B(t_j))

Expected accepted tokens: k_B = Σ α_B(j)
```

**Total Expected Speedup:**
```
Speedup = (Time_T) / (Time_A + Time_B + Time_verify)

Where:
- Time_A = Cost of generating k drafts with Model A
- Time_B = Cost of generating m refined drafts with Model B
- Time_verify = Cost of parallel verification with Target Model

Optimal when:
Time_A + Time_B << Time_T
And: k_A + k_B >> 1 (high acceptance rate)
```

### Feasibility Analysis

#### Advantages
1. **Higher Parallelization**
   - Model A is very fast → generates many candidate tokens quickly
   - Model B filters and refines → better quality drafts for Target

2. **Better Distribution Matching**
   - Model A → Model B is a smaller gap than Model A → Target
   - Model B → Target also has smaller distribution gap
   - Each step has higher acceptance probability

3. **Flexibility**
   - Can use different model architectures at each stage
   - Model A: Simple n-gram or tiny LLM
   - Model B: Feature-level draft model
   - Target: Full model

#### Challenges
1. **Error Accumulation**
   - Errors from Model A propagate to Model B
   - Model B must be robust to imperfect drafts

2. **Coordination Complexity**
   - Need to manage acceptance/rejection at each stage
   - Rollback becomes more complex with multiple stages

3. **Diminishing Returns**
   - Each additional stage adds overhead
   - Too many stages → net slowdown

#### Conditions for Feasibility

**Cascaded decoding is feasible when:**

1. **Strong Hierarchy in Model Sizes**
   ```
   Size_A << Size_B << Size_Target
   e.g., 1B → 7B → 70B
   ```

2. **High Acceptance at Each Stage**
   ```
   α_A > 0.7 (Model A → Model B)
   α_B > 0.7 (Model B → Target)

   If acceptance drops too low, cascading provides no benefit
   ```

3. **Low Verification Cost**
   ```
   Time_verify(k drafts) ≈ Time_single_forward

   Requires efficient tree attention or parallel verification
   ```

4. **Complementary Strengths**
   ```
   Model A: Fast at simple patterns (n-grams, common phrases)
   Model B: Good at semantic coherence (feature-level prediction)
   Target: Full capability for complex reasoning
   ```

### Related Works

While no published work (as of 2026) implements the exact 3-stage cascade above, related concepts include:

1. **Tree Speculative Decoding** (EAGLE, Medusa)
   - Multiple branches explored in parallel
   - Similar concept: hierarchical draft generation

2. **Recall That SpecTr** (Sun et al., 2023)
   - Uses optimal transport to match draft and target distributions
   - Could be extended to multiple stages

3. **Draft & Verify** (Zhang et al., 2023)
   - Early exit layers act as intermediate draft models
   - Implicitly creates a cascade within the same model

### Practical Recommendations

**Use cascaded decoding when:**
- You have 3+ model sizes available (1B, 7B, 70B)
- Each stage has >70% acceptance rate
- Verification cost is amortized over many draft tokens

**Avoid cascaded decoding when:**
- Models are similar in size (no speed benefit)
- Acceptance rates are low (<50%)
- Tree attention / parallel verification not available

### Example Configuration

For a **VideoLLaVA** setup with EventGPT:

```
Stage 1: EventGPT Vision Encoder (smallest)
    → Extract event features from video
    → Generate initial frame-level tokens

Stage 2: EventGPT Language Model (medium)
    → Refine frame tokens using context
    → Generate draft captions

Stage 3: VideoLLaVA-7B (target)
    → Verify and finalize captions
    → Parallel verification of all draft tokens
```

**Key requirement:** Feature alignment between EventGPT and VideoLLaVA (see Feasibility Analysis section).

---

## Future Directions

1. **Learned Hybrid Policies**
   - Train models to select best drafting strategy per token
   - Reinforcement learning for optimal resource allocation

2. **Hardware-Aware Hybrid Design**
   - Optimize for specific hardware architectures (GPUs, TPUs)
   - Consider memory bandwidth and compute trade-offs

3. **Multimodal Hybrid Decoding**
   - Extend hybrid approaches to vision-language models
   - Combine modality-specific drafting strategies

4. **Cascaded Speculative Decoding**
   - Chain multiple draft models (A → B → Target)
   - Requires careful balancing of stage-specific acceptance rates
   - Active research area (no published implementations yet)

---

## References

### Papers
- [Chimera: A Lossless Decoding Method](https://arxiv.org/abs/2402.15758) - Chen et al., 2024
- [SpecBranch: Speculative Decoding via Hybrid Drafting](https://arxiv.org/abs/2506.01979) - Shen et al., 2025
- [GLIDE with CaPE](https://arxiv.org/abs/2402.02082) - Du et al., 2024
- [Gumiho: A Hybrid Architecture](https://arxiv.org/abs/2503.10135) - AMD, 2025

### Related Works
- [Medusa: Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) - Cai et al., 2024
- [Hydra: Sequentially-Dependent Draft Heads](https://arxiv.org/abs/2402.05109) - Ankner et al., 2024
- [Speculative Decoding Survey (ACL Findings 2024)](https://aclanthology.org/2024.findings-acl.456.pdf) - Xia et al., 198 citations

### Resources
- [HuggingFace Assisted Generation](https://huggingface.co/blog/assisted-generation)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [Speculative Decoding Papers Collection](https://github.com/hemingkx/SpeculativeDecodingPapers)

---

## Summary

Hybrid speculative decoding represents the cutting edge of LLM inference acceleration, combining token-level efficiency with feature-level richness. Methods like Chimera, SpecBranch, and GLIDE+CaPE demonstrate that **adaptive, multi-strategy approaches** can achieve better speedups than single-method approaches.

The key insight: **don't rely on one drafting strategy** — use the right tool for each token, and have backup plans ready when drafts fail.

As we move forward, hybrid approaches will likely incorporate:
1. Learned policies for strategy selection
2. Hardware-aware optimization
3. Multimodal extensions for vision-language models

---

**Last Updated:** January 2026
