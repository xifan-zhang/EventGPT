# Embedding-Level Speculative Decoding Research

## Overview

**Embedding-level speculative decoding** (also called **feature-level speculative decoding**) IS a real and active research area. Instead of predicting tokens directly, these methods predict the next hidden state/feature/embedding and then pass it through the target model's LM head to obtain tokens.

## Key Insight

> "特征层中会比最终结果隐藏大量的 dark knowledge" - Feature layers contain more dark knowledge than final results

The core advantage is that feature/hidden state sequences exhibit more regularity than token sequences because token sampling introduces randomness/uncertainty. Different next tokens (e.g., "am" vs "always") can lead to completely different feature trajectories.

## Major Methods

### 1. EAGLE (ICML 2024)
**Paper:** "Speculative Sampling Requires Rethinking Feature Uncertainty"

- **Key innovation:** Autoregression at the **feature level** instead of token level
- Uses the second-to-last layer features of the target model
- Introduces the previous timestep's token sequence to handle uncertainty
- Draft model is lightweight (~1B parameters for 70B target model)
- Shares embedding layer and LM head with target model (no extra training needed)

**Speedup:** 2.5-4x over vanilla autoregressive decoding

### 2. EAGLE-2
**Paper:** "Faster Inference of Language Models with Dynamic Draft Trees"

- Improvement over EAGLE with **dynamic draft trees**
- Observation: Draft model acceptance rate depends on context, not just position
- Uses draft model confidence scores to dynamically prune/expand the draft tree
- Static trees (EAGLE-1, Medusa) waste computation on low-probability branches

**Speedup:** 20-40% faster than EAGLE-1

### 3. EAGLE-3 (NeurIPS 2025)
**Paper:** "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"
- **arXiv:** [2503.01840](https://arxiv.org/abs/2503.01840)
- **Venue:** NeurIPS 2025 (accepted)
- **GitHub:** [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) - Official implementation
- **Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang (Peking University, Microsoft Research)

**Key Innovations:**
- **Removes feature prediction constraint** - directly predicts tokens
- **Multi-layer feature fusion** instead of just top-layer features
- Uses "training-time test" - simulates multi-step generation during training
- Discovers a new scaling law: more training data → proportional speedup increase
- Inputs can be fused features (g) from target model OR draft model outputs (a)

**Key Architecture Changes:**
1. Concatenates low, middle, and high-level features from target model
2. Passes through FC layer to create fused feature representation
3. Removes feature prediction loss, only uses token prediction loss
4. Achieves up to **6.5x speedup** (1.4x better than EAGLE-2)

**Implementation Details:**
```python
# Training-time test implementation (from GitHub)
# Files: train/train_eagle_ttt.py, train/modules/trainer/trainer_eagle_ttt.py

# 1. Multi-layer feature fusion
low_features = target_model.get_layer_features(layer=8)   # Low-level
mid_features = target_model.get_layer_features(layer=16)  # Mid-level
high_features = target_model.get_layer_features(layer=31) # High-level

# Concatenate and project
fused = torch.cat([low_features, mid_features, high_features], dim=-1)
g = projection_layer(fused)  # (batch, seq_len, hidden_size)

# 2. Training-time test: Feed back predictions
for step in range(max_steps):
    if step == 0:
        # Native: use real target model features
        features = g_target
    else:
        # Simulated: use draft model's own predictions
        features = a_draft  # From previous step

    # Predict next token/future
    output = draft_model(features, text_embedding)
    a_draft = output.features  # Store for next iteration
```

**Community Implementations:**
- [BaldEagle](https://github.com/NickL77/BaldEagle) - 3x faster inference implementation
- [fast-llm-inference](https://github.com/ccs96307/fast-llm-inference) - EAGLE-3 implementation (April 2025)
- Training code in `train/train_eagle_ttt.py` (official repo)

### 4. GLIDE with CaPE
**Paper:** "Glide with a Cape: A Low-Hassle Method to Accelerate Speculative Decoding"

- **GLIDE:** Reuses target model's KV cache in draft model via cross-attention
- **CAPE:** Proposal expansion using draft model confidence scores
- Leverages dark knowledge from lower-confidence candidates

**Speedup:** 2.5-2.6x

### 5. Chimera
**Paper:** "A Lossless Decoding Method for Accelerating Large Language Models Inference by Fusing all Tokens"

- Uses **two draft models**: Trigram Encoder + Full Context Encoder
- Trigram encoder captures short-range dependencies (shallow layers)
- Full context encoder reuses target model's hidden states with distillation
- Multiple residual decoding heads (like Medusa)

**Speedup:** 2.7x

## Why Feature-Level Works Better

1. **Less randomness:** Features are deterministic; tokens involve sampling
2. **More information:** Features contain rich semantic info beyond top-1 token
3. **Better alignment:** Predicting features is more direct than predicting final distribution
4. **Layer utilization:** Different layers capture different abstraction levels

## Training-Time Test (EAGLE-3 Innovation)

The "training-time test" technique simulates the test-time multi-step generation process during training:

1. **Step 1 (native):** Train on real features f1, f2, ..., ft → predict token
2. **Step 2 (simulated):** Feed back model's own predictions as input
3. **Step 3 (simulated):** Continue the autoregressive process

This ensures the draft model learns to handle its own predictions as inputs, reducing distribution shift.

## Acceptance Rate Metrics

Key metrics for evaluating feature-level speculative decoding:

- **τ (tau):** Average acceptance length - tokens accepted per cycle
- **n-α:** Acceptance rate when input contains n estimated features

EAGLE-3 maintains high acceptance rates even with multiple self-predicted values in input, while EAGLE degrades significantly.

## Scaling Law Discovery

EAGLE-3 discovered that with the new architecture:
- More training data → proportional speedup increase
- This scaling behavior was NOT observed in EAGLE-1/2
- Feature prediction constraints in earlier methods limited benefits from data scaling

## Performance Comparison (MT-bench, LLaMA-Instruct 3.1 8B)

| Method | Speedup | Avg Acceptance Length τ |
|--------|---------|------------------------|
| Vanilla | 1.00x | 1.00 |
| Speculative Sampling | 1.93x | 2.27 |
| Medusa | 2.07x | 2.59 |
| EAGLE | 3.07x | 3.98 |
| EAGLE-2 | 4.26x | 4.83 |
| **EAGLE-3** | **5.58x** | **6.65** |

## References

- [EAGLE-3 Paper (arXiv 2025)](https://arxiv.org/html/2503.01840v1)
- [EAGLE GitHub](https://github.com/SafeAILab/EAGLE)
- [Speculative Decoding Survey](https://arxiv.org/abs/2401.07851)
- NVIDIA: [Introduction to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- vLLM: [Speculative Decoding Documentation](https://docs.vllm.ai/en/latest/features/spec_decode/)

## Research Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE-LEVEL SPECULATIVE DECODING                       │
│                          RESEARCH ROADMAP                                   │
└─────────────────────────────────────────────────────────────────────────────┘

   2023                    2024                          2025
    │                        │                             │
    ▼                        ▼                             ▼
┌─────────┐           ┌──────────┐                 ┌─────────────┐
│ Vanilla │           │  EAGLE   │                 │  EAGLE-3    │
│ Spec    │           │  (ICML)  │                 │   (SOTA)    │
└────┬────┘           └─────┬────┘                 └──────┬──────┘
     │                      │                              │
     │  Token-level         │  Feature-level               │  Training-Time Test
     │  prediction          │  autoregression              │  + Multi-layer fusion
     │  Small draft LLM     │  Reuse target features      │  Remove feature constraint
     │                      │  Tree attention             │  New scaling law discovery
     │                      │                              │
     ▼                      ▼                              ▼
   ~2x                    ~3-4x                         ~5-6.5x

┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVOLUTION OF KEY IDEAS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: TOKEN-LEVEL (Pre-2024)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Vanilla Speculative Sampling (Chen et al., Leviathan et al.)      │   │
│  │ • Small separate LLM as draft model                                │   │
│  │ • Limit: Distribution mismatch between draft and target            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Phase 2: FEATURE-LEVEL (2024)                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • EAGLE: Predict next feature, then LM head → token                 │   │
│  │ • Reuse target model's top-layer features                           │   │
│  │ • Key insight: Features have less randomness than tokens            │   │
│  │ • EAGLE-2: Dynamic draft trees based on confidence                 │   │
│  │ • GLIDE: Reuse KV cache via cross-attention                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Phase 3: MULTI-LAYER + TRAINING-TIME TEST (2025)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • EAGLE-3: Fuse low/mid/high features from target model             │   │
│  │ • Remove feature prediction constraint (direct token prediction)    │   │
│  │ • Training-time test: Simulate multi-step during training          │   │
│  │ • Discovery: Scaling law for data → speedup                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Phase 4: FUTURE DIRECTIONS                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Multimodal speculative decoding (vision + language)              │   │
│  │ • Long-context generation optimization                              │   │
│  │ • Reasoning models (o1, DeepSeek-R1) acceleration                   │   │
│  │ • Cross-architecture speculative decoding                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Feasibility Analysis: EventGPT → VideoLLaVA

**Question:** Can EventGPT serve as a feature-level draft model for VideoLLaVA?

### Challenges

1. **Architecture Mismatch**
   - EventGPT: Event-centric vision encoder + language model
   - VideoLLaVA: Different vision encoder (likely CLIP/LLaVA) + language model
   - Feature spaces may not align without significant adaptation

2. **Modality Gap**
   - EventGPT processes events (sparse, temporal)
   - VideoLLaVA processes frames (dense, visual)
   - Different preprocessing → different feature distributions

3. **Sequence Length**
   - VideoLLaVA generates long responses (multi-sentence descriptions)
   - Draft model needs to maintain coherence across many tokens

### Potential Approaches

| Approach | Feasibility | Effort | Expected Speedup |
|----------|-------------|--------|------------------|
| **A. Feature Alignment** | Medium | High | 2-3x |
| Train adapter to map EventGPT features → VideoLLaVA feature space | | | |
| **B. Hybrid Draft** | Low | Medium | 1.5-2x |
| Use EventGPT for video understanding, separate draft for text generation | | | |
| **C. Knowledge Distillation** | Medium | High | 2-4x |
| Distill VideoLLaVA's hidden states into EventGPT-like architecture | | | |
| **D. Vision Cache Reuse** | High | Low-Medium | 2-3x |
| Reuse VideoLLaVA's vision features, draft only language part (EAGLE-style) | | | |

### Recommended Approach: **Option D (Vision Cache Reuse)**

This is most similar to EAGLE/GLIDE and has highest feasibility:

```
Video Input
    │
    ▼
┌─────────────────┐
│ VideoLLaVA      │
│ Vision Encoder  │  ← Keep as-is (expensive but necessary)
└────────┬────────┘
         │ Vision Features (cache these)
         ▼
    ┌─────────┐
    │ Draft   │  ← Lightweight, trained on these features
    │ Model   │
    └────┬────┘
         │ Predicted Features
         ▼
    ┌─────────┐
    │ VideoLLaVA│  ← Verify using LM head only
    │ LM Head  │
    └──────────┘
```

**Why this works:**
1. Vision encoding is done once and cached
2. Draft model operates on cached vision features + text history
3. No architectural mismatch - same model's features
4. Similar to EAGLE's success with LLM features

### Implementation Steps

1. **Extract VideoLLaVA features** from penultimate layer during vision encoding
2. **Train draft model** (small transformer) to:
   - Take: Vision features + text embeddings → predict next feature
   - Use training-time test (EAGLE-3 technique)
3. **Verify** with VideoLLaVA's LM head (parallel token verification)
4. **Use tree attention** for batch verification

### Summary

Yes, it's feasible! The most promising approach is **not using EventGPT directly**, but rather:

**Train a lightweight draft model on VideoLLaVA's own vision+language features**, using EAGLE-3's multi-layer fusion and training-time test techniques.

Using EventGPT as draft would require feature alignment and may not work well due to architectural differences. The EAGLE family's success comes from reusing the **target model's own features**, not cross-model features.

## Latest Research (December 2025 - January 2026)

### Entropy-Aware Speculative Decoding (December 2025)
**Paper:** "Entropy-Aware Speculative Decoding Toward Improved..."
- **arXiv:** [2512.23765](https://arxiv.org/abs/2512.23765)
- **Authors:** T. Su et al.
- **Innovation:** Incorporates entropy awareness to improve draft quality

### SpecPV: Partial Verification for Long Context (December 2025)
**Paper:** "SpecPV: Improving Self-Speculative Decoding for Long-Context Generation via Partial Verification"
- **arXiv:** [2512.02337](https://arxiv.org/abs/2512.02337)
- **Authors:** Z. Tan et al.
- **Innovation:** Partial verification mechanism for long-context tasks

### Accelerate with Sparse Computation (December 2025)
**Paper:** "Accelerate Speculative Decoding with Sparse Computation..."
- **arXiv:** [2512.21911](https://arxiv.org/abs/2512.21911)
- **Authors:** J. Wang et al.
- **Innovation:** Integrates sparse computation with speculative decoding

### Efficient Adaptive Rejection Sampling (December 2025)
**Paper:** "Efficient Adaptive Rejection Sampling for Accelerating..."
- **arXiv:** [2512.13194](https://arxiv.org/abs/2512.13194)
- **Authors:** C. Sun et al.
- **Innovation:** Adaptive rejection sampling for improved efficiency

### Optimal Lower Bounds (December 2025)
**Paper:** "Speculative Decoding Speed-of-Light: Optimal Lower Bounds via Branching Random Walks"
- **arXiv:** [2512.11718](https://arxiv.org/abs/2512.11718)
- **Authors:** S. Pankratov et al.
- **Innovation:** Theoretical analysis establishing optimal lower bounds

### Speculative Sampling with RL (January 2026)
**Paper:** "Speculative Sampling with Reinforcement Learning"
- **arXiv:** [2601.12212](https://arxiv.org/abs/2601.12212)
- **Innovation:** Re-SpS introduces efficient feature reuse with RL

### Revisiting Judge Decoding (January 2026)
**Paper:** "Revisiting Judge Decoding from First Principles..."
- **arXiv:** [2601.04766](https://arxiv.org/abs/2601.04766)
- **Authors:** S. Sun et al.
- **Innovation:** First principles re-examination of Judge Decoding

### Multi-Scale for Image Generation (January 2026)
**Paper:** "Multi-Scale Local Speculative Decoding for Image Generation"
- **arXiv:** [2601.05149](https://arxiv.org/abs/2601.05149)
- **Authors:** E. Peruzzo et al.
- **Innovation:** Applies speculative decoding to image synthesis

### SRT for RL Acceleration (January 2026)
**Paper:** "SRT: Accelerating Reinforcement Learning via Speculative Rollout..."
- **arXiv:** [2601.09083](https://arxiv.org/abs/2601.09083)
- **Authors:** C.C. Chang et al.
- **Innovation:** Tree-structured cache for RL acceleration

### Plan, Verify and Fill (January 2026)
**Paper:** "Plan, Verify and Fill: A Structured Parallel Decoding Approach for Diffusion Language Models"
- **arXiv:** [2601.12247](https://arxiv.org/abs/2601.12247)
- **Innovation:** Structured parallel decoding for diffusion models

### Research Trends

The December 2025 - January 2026 papers show emerging trends:

1. **Long-Context Optimization** - SpecPV addresses long-context generation
2. **Cross-Domain Applications** - Image generation, reinforcement learning
3. **Theoretical Foundations** - Optimal lower bounds, first principles analysis
4. **Hybrid Approaches** - Combining with RL, sparse computation
5. **Verification Improvements** - Adaptive rejection sampling, entropy-aware methods

---

## Summary

Yes, embedding/feature-level speculative decoding is definitely "a thing" and represents one of the most promising directions in LLM inference acceleration. The key idea is:

**Predict hidden states, not tokens.** Hidden states contain more information, exhibit more regularity, and can be predicted more accurately than final token distributions.

The evolution from EAGLE → EAGLE-2 → EAGLE-3 shows rapid improvement, with EAGLE-3 achieving up to 6.5x speedup through multi-layer feature fusion and training-time test techniques.

**Latest developments (Dec 2025 - Jan 2026):** The field continues to evolve with:
- Long-context optimization (SpecPV)
- Cross-domain applications (images, RL)
- Theoretical foundations (optimal lower bounds)
- Hybrid approaches (RL + speculative)
- Improved verification (entropy-aware, adaptive rejection)

---

**Last Updated:** January 2026
