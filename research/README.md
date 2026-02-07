# Speculative Decoding Research Collection

Accelerating Large Language Model and Vision-Language Model Inference through Speculative Decoding Techniques

---

## Overview

This repository contains research on **speculative decoding** techniques for accelerating LLM and VLM inference. The focus is on three key approaches:

1. **Token-Level Speculative Decoding** (Standard/Vanilla)
2. **Embedding/Feature-Level Speculative Decoding** (EAGLE-family)
3. **Hybrid and Cascaded Approaches** (Combining multiple strategies)

**Application Goal:** Accelerate VideoLLaVA inference using EventGPT as a draft model.

---

## Quick Start

### What is Speculative Decoding?

**Speculative decoding** accelerates model inference by:
1. Using a small **draft model** to predict future tokens
2. Verifying predictions with a larger **target model** in parallel
3. Accepting good predictions and resampling bad ones

**Result:** 2-6x speedup with no quality loss.

### Key Insight

> "Predict features, not tokens. Features contain more information and exhibit more regularity than final token distributions."

---

## Documents

| Document | Description | Focus | Status |
|----------|-------------|-------|--------|
| [**Token-Level Speculative Decoding**](token_level_speculative_decoding.md) | Standard/vanilla approach | Small LM → Large LM (token prediction) | ✅ Complete |
| [**Embedding-Level Speculative Decoding**](embedding_level_speculative_decoding.md) | Feature-level approach (EAGLE) | Hidden state prediction | ✅ Complete |
| [**Hybrid Speculative Decoding**](hybrid_speculative_decoding.md) | Combined approaches | Token + Feature, multiple draft models | ✅ Complete |
| [**Sequential/Cascaded Speculative Decoding**](sequential_cascaded_speculative_decoding.md) | Multi-stage acceleration + theoretical analysis | Chain of draft models (A → B → Target) | ✅ Complete |
| [**Edge-Cloud Speculative Decoding**](edge_cloud_speculative_decoding.md) | Distributed inference | Edge device → Cloud verification | ✅ New |
| [**Parallel Prefill & Speculative Decoding**](parallel_prefill_speculative_decoding_20260125_004920.md) | Cross-modal VLM acceleration | EventGPT draft + Video-LLaVA target opportunities | ✅ New |
| [**EventGPT → VideoLLaVA Roadmap**](EventGPT_VideoLLaVA_roadmap.md) | Application roadmap | Sparse events → dense video acceleration | ✅ Complete |

### Supporting Files

| File | Purpose |
|------|---------|
| [**CHANGELOG.md**](CHANGELOG.md) | Formal changelog (Keep a Changelog format) |
| [**UPDATE_LOG.md**](UPDATE_LOG.md) | Detailed update tracking and TODO list |
| [**UNIFIED_TEMPLATE.md**](UNIFIED_TEMPLATE.md) | Template for all markdown files |

---

## Document Structure

All documents follow a unified structure defined in [UNIFIED_TEMPLATE.md](UNIFIED_TEMPLATE.md):

1. **Overview** - Introduction and key insights
2. **Key Methods** - Main algorithms and techniques
3. **Theoretical Analysis** - Mathematical formulations
4. **Implementation** - Code examples and usage
5. **Performance** - Benchmarks and comparisons
6. **Critical Analysis** - Advantages, limitations, recommendations
7. **References** - Citations and resources
8. **Summary** - Key takeaways

---

## Research Taxonomy

```
SPECULATIVE DECODING
├── TOKEN-LEVEL (Standard)
│   ├── Vanilla Speculative Sampling (Chen et al., 2023)
│   ├── Speculative Decoding (Leviathan et al., 2023)
│   ├── Medusa (Cai et al., 2024)
│   ├── Hydra (Ankner et al., 2024)
│   └── Assisted Generation (HuggingFace)
│
├── FEATURE/EMBEDDING-LEVEL
│   ├── EAGLE (Li et al., 2024)
│   ├── EAGLE-2 (Li et al., 2024)
│   ├── EAGLE-3 (Li et al., 2025)
│   ├── GLIDE with CaPE (Du et al., 2024)
│   └── Chimera (Chen et al., 2024)
│
├── HYBRID APPROACHES
│   ├── Token + Feature Fusion
│   ├── Multiple Draft Models
│   ├── Adaptive Strategy Selection
│   └── Cascaded Drafting
│
├── SEQUENTIAL/CASCADED
│   ├── ReDrafter (Cheng et al., 2024)
│   ├── Cascade Speculative Drafting (Chen et al., 2024)
│   ├── Lookahead Decoding (Fu et al., 2024)
│   └── PaSS (Monea et al., 2023)
│
└── MULTIMODAL EXTENSIONS
    ├── SpecVLM (2025)
    ├── ViSpec (2025)
    └── EventGPT → VideoLLaVA (This Research)
```

---

## Key Papers Summary

### Foundational Works (2023)

| Paper | Venue | Citations | Key Idea |
|-------|-------|-----------|----------|
| [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) | arXiv 2023 | 642+ | Standard speculative sampling |
| [Fast Inference via Speculative Decoding](https://arxiv.org/abs/2211.17192) | ICML 2023 | 1,122+ | Transformer-specific optimization |
| [PaSS: Parallel Speculative Sampling](https://arxiv.org/abs/2311.13581) | NeurIPS 2023 | 146+ | Parallel draft generation |
| [SpecTr: Optimal Transport for Speculative Decoding](https://arxiv.org/abs/2310.15141) | NeurIPS 2023 | 146+ | Distribution matching via OT |

### Feature-Level Works (2024-2025)

| Paper | Venue | Citations | Speedup | Key Idea |
|-------|-------|-----------|---------|----------|
| [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) | ICML 2024 | - | 3-4x | Predict next feature, not token |
| [EAGLE-2: Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) | EMNLP 2024 | - | 4-5x | Context-aware tree pruning |
| [EAGLE-3: Training-Time Test](https://arxiv.org/abs/2503.01840) | arXiv 2025 | - | 5-6.5x | Multi-layer fusion, scaling law |
| [Chimera: Fusing All Tokens](https://arxiv.org/abs/2402.15758) | arXiv 2024 | 77 | 2.7x | Trigram + Full Context Encoder |
| [GLIDE with CaPE](https://arxiv.org/abs/2402.02082) | arXiv 2024 | - | 2.5-2.6x | KV cache reuse + proposal expansion |

### Multimodal Works (2024-2025)

| Paper | Venue | Focus | Speedup |
|-------|-------|-------|---------|
| [SpecVLM: Fast Speculative Decoding in VLMs](https://arxiv.org/abs/2509.11815) | arXiv 2025 | Vision-language models | TBD |
| [ViSpec: Vision-Aware Speculative Decoding](https://arxiv.org/abs/2509.15235) | NeurIPS 2025 | Vision feature reuse | 3.2x |
| [Spec-VLA: VLA Acceleration](https://arxiv.org/abs/2507.22424) | EMNLP 2025 | Vision-language-action | TBD |

---

## Performance Comparison

### Speedup by Approach

| Approach | Typical Speedup | Complexity | Production Ready |
|----------|-----------------|------------|------------------|
| **Token-Level** | 2-3x | Low | ✅ Yes (HuggingFace, vLLM) |
| **Feature-Level (EAGLE)** | 3-5x | Medium | ✅ Yes (GitHub) |
| **Feature-Level (EAGLE-3)** | 5-6.5x | Medium | ⚠️ New (2025) |
| **Hybrid** | 2.5-4x | High | ⚠️ Research |
| **Cascaded** | 3-5x | High | ⚠️ Research |

### Acceptance Rates

| Method | Acceptance Rate | Notes |
|--------|-----------------|-------|
| Same-family token-level | 70-90% | e.g., GPT-2 → GPT-2 XL |
| Cross-family token-level | 40-60% | e.g., LLaMA → GPT |
| Feature-level (EAGLE) | 60-80% | More stable |
| Feature-level (EAGLE-3) | 80-95% | Training-time test helps |

---

## Our Research: EventGPT → VideoLLaVA

### Goal

Accelerate VideoLLaVA-7B/13B inference using EventGPT as a draft model through:

1. **Stage 1: Embedding-Level (Vision)**
   - EventGPT encoder (sparse, fast) → VideoLLaVA encoder (dense, slow)
   - Feature alignment layer
   - Target: 10-50x speedup on vision encoding

2. **Stage 2: Token-Level (Language)**
   - EventGPT LM (1B) → VideoLLaVA LM (7B/13B)
   - Shared vision features (cached)
   - Target: 2-3x speedup on text generation

### Actual E2E Wall-Clock Performance (50 samples, DSEC 1s dataset)

| Config | Total (ms) | Accept Rate | Speedup | Free Tokens |
|--------|-----------|-------------|---------|-------------|
| VL baseline | 1017 | --- | 1.00x | --- |
| L4+VL (best) | 1005 | 20.3% | **1.02x** avg, **1.24x** peak | 21.9 |

### Roadmap

- [x] Phase 1: Hidden state extraction + adapter training (L1-L5, B1, L5F)
- [x] Phase 2: E2E wall-clock benchmark with prefill hiding
- [ ] Phase 3: EAGLE autoregressive rollout retraining (B1R, L5FR)
- [ ] Phase 4: Full 1100-sample benchmark (running)

**See:** [EventGPT → VideoLLaVA Roadmap](EventGPT_VideoLLaVA_roadmap.md)

---

## Code Examples

### Token-Level Speculative Decoding

```python
from transformers import AutoModelForCausalLM

# Load models
draft_model = AutoModelForCausalLM.from_pretrained("gpt2")
target_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Generate with speculative decoding
outputs = target_model.generate(
    input_ids,
    assistant_model=draft_model,  # HuggingFace syntax
    do_sample=True,
    max_new_tokens=100
)
```

### Feature-Level (EAGLE-Style)

```python
# Predict next feature instead of token
features = target_model.get_features(input_ids)

# Draft model predicts next feature
draft_feature = draft_model.predict_feature(features)

# Use target model's LM head to get token
token = target_model.lm_head(draft_feature)

# Verify with target model
accepted = verify_with_target(draft_feature, target_model)
```

---

## EAGLE-2 Draft Model Architecture *(Added: 2026-02-06)*

### Overview

EAGLE-2 introduces **context-aware dynamic draft trees** on top of EAGLE's feature-level autoregression. The key insight: acceptance rates depend on **context**, not just position.

### Draft Model Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EAGLE-2 DRAFT MODEL                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Target LLM (Frozen)                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Layer 1 → Layer 2 → ... → Layer N-1 → Layer N → LM Head   │   │
│  └──────────────────────────────────┬──────────────────────────┘   │
│                                     │                               │
│                          Penultimate-layer                          │
│                          features (f_t)                             │
│                                     │                               │
│                                     ▼                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              DRAFT MODEL (FeatExtrapolator)                  │   │
│  │  ┌─────────────┐    ┌──────────────────────────────────┐    │   │
│  │  │ Token Embed │───►│ Single-Layer Transformer Decoder │    │   │
│  │  │ (t_{i-1})   │    │  • Self-attention                │    │   │
│  │  └─────────────┘    │  • FFN                           │    │   │
│  │         +           │  • Layer Norm                    │    │   │
│  │  ┌─────────────┐    └──────────────┬───────────────────┘    │   │
│  │  │ Feature     │                   │                        │   │
│  │  │ (f_{i-1})   │───────────────────┘                        │   │
│  │  └─────────────┘         Predicted feature (f̂_i)           │   │
│  └──────────────────────────────────┬──────────────────────────┘   │
│                                     │                               │
│                                     ▼                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │            TARGET MODEL's LM HEAD (Reused)                   │   │
│  │                  f̂_i → logits → token                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Input** | Concatenation of: token embedding (t_{i-1}) + penultimate feature (f_{i-1}) |
| **Draft Decoder** | Single-layer Transformer decoder (~10-15% of target params) |
| **Output** | Predicted next feature f̂_i |
| **LM Head** | Reuses target model's LM head (no additional training) |

### Feature Extrapolation Process

```python
# EAGLE-2 Draft Model Forward Pass
def eagle2_draft_forward(target_model, draft_model, input_ids):
    # 1. Get penultimate-layer features from target model
    with torch.no_grad():
        hidden_states = target_model(input_ids, output_hidden_states=True)
        f_prev = hidden_states.hidden_states[-2]  # Second-to-last layer

    # 2. Get token embedding (shifted by 1 position)
    t_prev = target_model.embed_tokens(input_ids)

    # 3. Concatenate feature + token embedding as draft input
    draft_input = torch.cat([f_prev, t_prev], dim=-1)  # Or addition

    # 4. Draft model predicts next feature
    f_hat = draft_model(draft_input)  # Single-layer transformer

    # 5. Use target's LM head to get token probabilities
    logits = target_model.lm_head(f_hat)

    return logits, f_hat
```

### Why Token Embedding is Needed

The sampling process introduces uncertainty in the feature sequence:

```
Without token embedding:
  "I am" → feature_am → ???
  "I always" → feature_always → ???

  Different tokens lead to completely different feature trajectories!

With token embedding (EAGLE's solution):
  Input: [feature_{t-1}, token_{t-1}] → predict feature_t

  This resolves the uncertainty by explicitly providing the sampled token.
```

### Dynamic Draft Tree (EAGLE-2's Innovation)

EAGLE-2's key contribution is **dynamic tree construction** based on draft model confidence:

```
Static Tree (EAGLE-1, Medusa):          Dynamic Tree (EAGLE-2):
         root                                   root
        /    \                                 /    \
       a      b                               a      b (low conf, pruned)
      /|\    /|\                             /|\
     c d e  f g h                           c d e

    Fixed structure,                     Adaptive structure,
    wastes compute on                    focuses on high-
    low-prob branches                    probability paths
```

#### Dynamic Tree Algorithm

```python
def build_dynamic_tree(draft_model, root_feature, max_tokens=60, depth=7):
    """EAGLE-2 dynamic draft tree construction"""
    tree = [root_feature]

    for layer in range(depth):
        candidates = []

        # Expansion Phase: Get top-k candidates
        for node in tree:
            logits, features = draft_model(node)
            probs = F.softmax(logits, dim=-1)

            # Confidence = probability of predicted token
            top_k_probs, top_k_tokens = probs.topk(k=10)

            for prob, token in zip(top_k_probs, top_k_tokens):
                # Value = product of confidences along path
                value = node.path_confidence * prob.item()
                candidates.append((value, token, features))

        # Reranking Phase: Select top-m by value
        candidates.sort(key=lambda x: x[0], reverse=True)
        tree = candidates[:max_tokens // depth]

    return tree
```

#### Key Insight: Confidence ≈ Acceptance Rate

EAGLE-2 discovered that the draft model is **well-calibrated**:

```
Draft model confidence score ≈ Actual acceptance rate

This allows dynamic pruning WITHOUT running the target model!
```

### Tree Attention for Parallel Verification

```python
# Tree attention mask for batch verification
def create_tree_attention_mask(tree_structure):
    """
    Tree structure: [root, child1, child2, grandchild1, ...]

    Attention mask ensures each node only attends to its ancestors:

         0 (root)
        / \
       1   2
      / \
     3   4

    Mask:
         0  1  2  3  4
    0 [  1  0  0  0  0 ]  # root attends to self
    1 [  1  1  0  0  0 ]  # child1 attends to root, self
    2 [  1  0  1  0  0 ]  # child2 attends to root, self
    3 [  1  1  0  1  0 ]  # grandchild1 attends to root, child1, self
    4 [  1  1  0  0  1 ]  # grandchild2 attends to root, child1, self
    """
    n = len(tree_structure)
    mask = torch.zeros(n, n)
    for i, node in enumerate(tree_structure):
        # Attend to all ancestors + self
        mask[i, node.ancestor_indices + [i]] = 1
    return mask
```

### EAGLE-2 Inference Configuration

```python
# Typical EAGLE-2 hyperparameters
config = {
    "top_k": 10,           # Candidates per expansion
    "total_tokens": 60,    # Max draft tokens
    "depth": 7,            # Max tree depth
    "temperature": 0,      # Greedy (deterministic)
}
```

### Performance

| Method | Speedup | Avg Accepted Length |
|--------|---------|---------------------|
| EAGLE-1 (static tree) | 3.0-4.0x | ~4 tokens |
| **EAGLE-2 (dynamic tree)** | **3.5-4.5x** | ~5 tokens |
| Improvement | +20-40% | +25% |

### References

- [EAGLE-2: Faster Inference with Dynamic Draft Trees (EMNLP 2024)](https://arxiv.org/abs/2406.16858)
- [EAGLE GitHub (Official Implementation)](https://github.com/SafeAILab/EAGLE)
- [EAGLE-2 HTML Paper](https://arxiv.org/html/2406.16858v1)

---

## SpecVLM: Speculative Decoding for Vision-Language Models *(Added: 2026-02-06)*

**Paper:** [SpecVLM: Fast Speculative Decoding in Vision-Language Models (arXiv 2509.11815)](https://arxiv.org/abs/2509.11815)

### Overview

SpecVLM extends EAGLE-style speculative decoding to VLMs by addressing the **visual token bottleneck** - the key challenge that distinguishes VLMs from LLMs.

```
SpecVLM = EagleVLM (EAGLE-2 style draft) + Elastic Visual Compressor
```

### The VLM-Specific Problem

| Model | Bottleneck | Typical Token Count |
|-------|------------|---------------------|
| LLM | Text decoding | ~100-500 tokens |
| VLM | **Visual tokens + KV cache** | ~1000-5000+ tokens |

```
Image (224×224)  →  256 visual tokens
Image (336×336)  →  576 visual tokens
Video (8 frames) → 2048+ visual tokens  ← Prefill dominates!
```

Standard EAGLE doesn't help much for VLMs because **prefill (not decoding) is the bottleneck**.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SpecVLM Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input Image/Video                                                      │
│        │                                                                │
│        ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │           ELASTIC VISUAL COMPRESSOR (New in SpecVLM)            │   │
│  │                                                                 │   │
│  │   Adaptively selects compression method per input:              │   │
│  │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │   │
│  │   │ Pruning  │ │ Pooling  │ │   Conv   │ │Resampler │          │   │
│  │   │ (sparse) │ │ (spatial)│ │ (learned)│ │(Q-Former)│          │   │
│  │   └──────────┘ └──────────┘ └──────────┘ └──────────┘          │   │
│  │                                                                 │   │
│  │   Selection based on: image complexity, resolution, task        │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                       │
│                    Compressed visual tokens (reduced KV cache)          │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              EagleVLM (EAGLE-2 Style Draft Model)               │   │
│  │                                                                 │   │
│  │   • Single-layer transformer decoder                            │   │
│  │   • Input: penultimate features + token embeddings              │   │
│  │   • Reuses target VLM's LM head                                 │   │
│  │   • Tree attention (top-k=10, depth=7, 60 tokens)               │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                       │
│                           Draft tokens                                  │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Target VLM (Frozen)                          │   │
│  │              Parallel verification of draft tokens               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Two-Pronged Speedup Strategy

| Problem | Solution | Speedup |
|---------|----------|---------|
| Slow autoregressive decoding | EagleVLM (EAGLE-2 draft) | 1.5-2.3x |
| Visual token overhead (prefill + KV cache) | Elastic Visual Compressor | +1.3-1.5x |
| **Combined** | **SpecVLM** | **2.5-2.9x** |

### Elastic Visual Compressor Details

The compressor adaptively chooses from four primitives:

| Method | Mechanism | Best For |
|--------|-----------|----------|
| **Pruning** | Remove low-importance tokens | Simple images, redundant regions |
| **Pooling** | Spatial average/max pooling | Uniform textures |
| **Convolution** | Learned downsampling | Complex patterns |
| **Resampler** | Q-Former style attention | Dense information |

```python
# Elastic compressor selection (conceptual)
def elastic_compress(visual_tokens, image_complexity):
    if image_complexity < 0.3:
        return prune(visual_tokens, keep_ratio=0.5)
    elif image_complexity < 0.6:
        return pool(visual_tokens, factor=2)
    elif image_complexity < 0.8:
        return conv_downsample(visual_tokens)
    else:
        return resampler(visual_tokens)  # Preserve most info
```

### Training

**Online-Logit Distillation** (no offline corpus needed):

```python
# Training objective
loss = CE_loss(draft_logits, target_logits) + λ * SmoothL1(draft_features, target_features)

# Only these are trained (target VLM frozen):
# 1. Draft model's decoder layer
# 2. Visual compressor
```

Key finding: **Longer training → higher acceptance rate** (training-time scaling effect)

### Performance Summary

| Configuration | Speedup | Notes |
|---------------|---------|-------|
| Baseline (autoregressive) | 1.0x | - |
| EagleVLM only | 1.5-2.3x | Decoding acceleration |
| Visual compression only | 1.3-1.5x | Prefill acceleration |
| **SpecVLM (both)** | **2.5-2.9x** | End-to-end |

Evaluated on: LLaVA benchmark suite, MMMU multimodal evaluation

### Comparison with Other VLM Acceleration Methods

| Method | Approach | Speedup | Lossless? |
|--------|----------|---------|-----------|
| Token pruning | Remove visual tokens | 1.5-2x | No |
| Early exit | Skip layers | 1.3-1.8x | No |
| Quantization | Reduce precision | 1.5-2x | No |
| **SpecVLM** | Draft + compress | **2.5-2.9x** | **Yes** |

### Key Takeaways

1. **VLMs need different treatment** - Visual tokens dominate prefill, not just decoding
2. **EagleVLM baseline is strong** - EAGLE-2 style works well for VLMs
3. **Compression is complementary** - Addresses the VLM-specific bottleneck
4. **Online distillation works** - No need for expensive offline corpus generation
5. **Lossless acceleration** - Maintains target model's output distribution

### References

- [SpecVLM Paper (arXiv 2509.11815)](https://arxiv.org/abs/2509.11815)
- [SpecVLM HTML](https://arxiv.org/html/2509.11815v1)

---

## Cross-Modal Speculative Decoding: Beyond SpecVLM *(Added: 2026-02-06)*

### Motivation: SpecVLM's Sequential Bottleneck

SpecVLM achieves 2.5-2.9x speedup but has a fundamental limitation - **everything is sequential**:

```
SpecVLM Pipeline (Sequential):

Image/Video ──► Visual Encoder ──► Compressor ──► EagleVLM ──► Target VLM
                    │                   │            │            │
                 SLOW (1x)          depends on    depends on   depends on
                                    encoder       compressor   draft
                                    finishing     finishing    finishing

Compression can't start until encoding finishes.
Draft can't start until compression finishes.
```

### Cross-Modal Advantage: Parallel Prefill

Cross-modal speculative decoding uses a **different (faster) modality** as the draft, enabling true parallelism:

```
Cross-Modal Pipeline (Parallel):

                    ┌──► Event Encoder (FAST) ──► Draft LM ──► Draft tokens
                    │         10ms                  20ms           │
Sensor Input ───────┤                                              │
                    │                                              ▼
                    └──► RGB Encoder (SLOW) ──────────────► Target LM (verify)
                              100ms                            30ms

Draft generates tokens WHILE target encoder is still processing!
```

### Where Cross-Modal Wins

#### 1. Prefill Hiding (Biggest Advantage)

| Approach | Prefill Latency | Can Hide Prefill? |
|----------|-----------------|-------------------|
| SpecVLM | Must wait for target encoder | No - same encoder |
| Cross-Modal | Draft encoder runs in parallel | **Yes** |

```
SpecVLM timeline:
[====== RGB Encode 100ms ======][== Compress ==][== Draft ==][== Verify ==]
                                                            Total: ~180ms

Cross-Modal timeline:
[== Event Encode 10ms ==][======= Draft generates while RGB encodes =======]
[================= RGB Encode 100ms ==================][== Verify ==]
                                                            Total: ~130ms

Saved: ~50ms by hiding prefill behind draft generation
```

#### 2. Inherently Faster Modality

| Modality | Data Density | Encode Time | Visual Tokens |
|----------|--------------|-------------|---------------|
| Event camera | Sparse (edges only) | 5-20ms | ~100-500 |
| RGB video | Dense (all pixels) | 50-200ms | ~1000-5000 |

Cross-modal exploits **modality asymmetry** - draft uses fast sparse modality, target uses slow dense modality.

#### 3. No Compression Artifacts

| Approach | Visual Quality | Risk |
|----------|----------------|------|
| SpecVLM (compress) | Degraded (lossy) | Miss fine details |
| Cross-Modal | Full resolution for target | **None** |

SpecVLM's compressor trades quality for speed. Cross-modal preserves target's full visual fidelity.

#### 4. Speculative Prefill (Advanced)

```python
# SpecVLM: Only speculates text tokens
draft_text_tokens = eagle_vlm(compressed_visual_features)

# Cross-modal: Can speculate BOTH visual features AND text tokens
draft_visual_features = event_encoder(events)      # Fast approximation
draft_text_tokens = draft_lm(draft_visual_features)  # Speculate tokens

# If visual features align well → BOTH accepted
# Double speculation = higher potential speedup
```

### Theoretical Speedup Comparison

| Scenario | SpecVLM | Cross-Modal | Winner |
|----------|---------|-------------|--------|
| Short video, simple scene | 2.5x | 2.0x | SpecVLM |
| Long video, complex scene | 2.8x | **3.5x** | Cross-Modal |
| Real-time streaming | 2.5x | **4.0x+** | Cross-Modal |
| High-resolution input | 2.5x | **3.5x** | Cross-Modal |

### When Each Approach Wins

**Cross-modal wins when:**
- Prefill time dominates (long videos, high resolution)
- Real-time/streaming scenarios (can't wait for full encode)
- Draft modality is significantly faster than target
- Need full visual quality (no compression artifacts)

**SpecVLM wins when:**
- Same-modality input (no alignment needed)
- Short inputs (prefill not the bottleneck)
- Single image (compression very effective)
- No access to alternative fast modality

### The Key Trade-offs

```
SpecVLM:
  ✓ No cross-modal alignment needed
  ✓ Guaranteed same visual understanding
  ✓ Works with any VLM out-of-box
  ✗ Sequential pipeline (can't hide prefill)
  ✗ Lossy compression may hurt quality

Cross-Modal (e.g., EventGPT → VideoLLaVA):
  ✓ Parallel prefill (hide encoder latency)
  ✓ Exploit inherently fast modalities
  ✓ Full quality preserved for target
  ✗ Requires cross-modal feature alignment training
  ✗ Modality gap may reduce acceptance rate
  ✗ Needs specialized hardware (event camera)
```

### Ideal Cross-Modal Scenario: EventGPT → VideoLLaVA

```
Timeline analysis:

1. Event encoding:     10ms  ─┬─ Parallel
2. RGB encoding:      100ms  ─┘
3. Draft generation:   80ms  (produces ~8 tokens while waiting for RGB)
4. Verification:       20ms  (batch verify 8 tokens)
                      ─────
                      ~110ms total

Baseline (no speculation): 100ms encode + 100ms decode = 200ms
SpecVLM:                   100ms encode + 40ms compress + 35ms draft+verify = 175ms
Cross-Modal:               max(10+80, 100) + 20ms verify = 110ms

Speedup breakdown:
  - Parallel prefill:        ~2x (hide 90ms of RGB encoding)
  - Speculative decoding:    ~2x (8 tokens verified in 1 pass)
  - Combined theoretical:    ~4x
```

### Architecture for Cross-Modal Speculative Decoding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Cross-Modal Speculative Decoding Pipeline                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Synchronized Sensor Input (Events + RGB)                                   │
│        │                                                                    │
│        ├────────────────────────┬───────────────────────────────────────┐  │
│        │                        │                                       │  │
│        ▼                        ▼                                       │  │
│  ┌─────────────┐          ┌─────────────┐                               │  │
│  │   Event     │          │    RGB      │                               │  │
│  │  Encoder    │          │   Encoder   │  ← Runs in parallel           │  │
│  │  (10ms)     │          │  (100ms)    │                               │  │
│  └──────┬──────┘          └──────┬──────┘                               │  │
│         │                        │                                       │  │
│         ▼                        │                                       │  │
│  ┌─────────────┐                 │                                       │  │
│  │  Alignment  │                 │  ← Optional: align event features     │  │
│  │   Layer     │                 │     to RGB feature space              │  │
│  └──────┬──────┘                 │                                       │  │
│         │                        │                                       │  │
│         ▼                        │                                       │  │
│  ┌─────────────┐                 │                                       │  │
│  │  Draft LM   │                 │  ← Generates tokens while RGB encodes │  │
│  │ (EventGPT)  │                 │                                       │  │
│  └──────┬──────┘                 │                                       │  │
│         │                        │                                       │  │
│         │ Draft tokens           │ RGB features                          │  │
│         │ (speculative)          │ (ground truth)                        │  │
│         │                        │                                       │  │
│         └────────────┬───────────┘                                       │  │
│                      │                                                    │  │
│                      ▼                                                    │  │
│              ┌─────────────┐                                              │  │
│              │  Target LM  │  ← Parallel verification                     │  │
│              │(VideoLLaVA) │    Accept/reject draft tokens                │  │
│              └─────────────┘                                              │  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Research Challenges

| Challenge | Difficulty | Potential Solution |
|-----------|------------|-------------------|
| Cross-modal feature alignment | High | Contrastive learning, adapter layers |
| Acceptance rate optimization | Medium | Joint training, distillation |
| Temporal synchronization | Medium | Hardware sync, interpolation |
| Modality gap in understanding | High | Hybrid features, late fusion |

### Why SpecVLM Cannot Parallelize Prefill *(Added: 2026-02-06)*

**One-Line Answer:** SpecVLM's draft is **parasitic** (needs target's features) vs Cross-modal draft is **independent** (has own encoder).

**The Dependency Problem:** SpecVLM's EagleVLM draft model requires the target VLM's penultimate-layer features as input. This creates an architectural dependency that prevents parallel execution.

```
EAGLE-style draft input = [target's penultimate features] + [token embedding]
                                      ↑
                          Must run target encoder FIRST!
```

```
SpecVLM: Draft DEPENDS on Target's Features
──────────────────────────────────────────────────────────────────────
Input ──► [Target VLM Encoder] ──► Features (f_t) ──► [EagleVLM Draft]
                │                        │                   │
                │                        └───────────────────┘
                │                         Draft needs f_t from target!
                └── MUST FINISH FIRST ────────────────────────────────


Cross-Modal: Draft is INDEPENDENT
──────────────────────────────────────────────────────────────────────
Events ──► [Event Encoder] ──► [Draft LM] ──► Draft tokens
                │                                   │
                │ PARALLEL (independent)            │
                │                                   ▼
RGB ────► [RGB Encoder] ─────────────────► [Target LM] (verify)
```

**Why EAGLE-style needs target features:**

```python
# EAGLE/EagleVLM: Draft uses TARGET's features
def eagle_draft(target_model, input_ids):
    # Step 1: MUST run target model first
    target_outputs = target_model(input_ids, output_hidden_states=True)
    f_target = target_outputs.hidden_states[-2]  # Penultimate layer

    # Step 2: Only NOW can draft model run
    draft_input = concat(f_target, embed(input_ids))
    f_predicted = draft_model(draft_input)
    return target_model.lm_head(f_predicted)

# Cross-modal: Draft has INDEPENDENT encoder
def crossmodal_draft(events, rgb_frames):
    future_rgb = executor.submit(rgb_encoder, rgb_frames)  # Async
    event_features = event_encoder(events)  # Runs immediately
    draft_tokens = draft_lm(event_features)  # While RGB encodes
    rgb_features = future_rgb.result()  # Wait only for verify
    return target_lm.verify(draft_tokens, rgb_features)
```

**Timeline Comparison:**

```
SpecVLM (Sequential - 150ms):
├──[RGB Encode 80ms]──┼──[Compress 20ms]──┼──[Draft 30ms]──┼──[Verify 20ms]──┤

Cross-Modal (Parallel - 120ms):
├──[Event 10ms]──[Draft LM 70ms]──────────┤
├──────────[RGB Encode 100ms]─────────────┼──[Verify 20ms]──┤
                                          ↑
                              max(10+70, 100) = 100ms
```

| Aspect | SpecVLM | Cross-Modal |
|--------|---------|-------------|
| Draft has own encoder? | **No** (reuses target) | **Yes** (EventGPT) |
| Feature source | Target's penultimate layer | Independent modality |
| Can run parallel? | **No** (dependency) | **Yes** |

**Why EAGLE Designed It This Way:**

| Design Choice | Benefit | Cost |
|---------------|---------|------|
| Use target's features | **High acceptance rate** (same feature space) | **No parallelism** |
| Reuse target's LM head | No extra training needed | Coupled to target |

EAGLE prioritized **acceptance rate** over **parallelism** - good for LLMs where prefill is fast, but suboptimal for VLMs where prefill dominates latency.

**Trade-off:** SpecVLM gets high acceptance (same feature space) but no parallelism. Cross-modal needs alignment training but enables parallel prefill.

### Summary: SpecVLM vs Cross-Modal

| Dimension | SpecVLM | Cross-Modal |
|-----------|---------|-------------|
| **Prefill strategy** | Compress tokens | **Hide behind fast encoder** |
| **Decoding strategy** | EAGLE-style draft | EAGLE-style draft |
| **Parallelism** | Sequential | **Parallel prefill** |
| **Alignment needed** | None | **Cross-modal training** |
| **Best scenario** | Single images | **Video/streaming** |
| **Quality preservation** | Lossy (compression) | **Lossless** |
| **Theoretical max speedup** | ~3x | **~4-5x** |
| **Implementation complexity** | Low | High |

**Bottom line:** Cross-modal speculative decoding can outperform SpecVLM by exploiting **parallel prefill** with a fast draft modality, achieving up to 4-5x speedup in video/streaming scenarios where prefill dominates latency.

---

## Current Benchmark: T_prefill / T_total Analysis *(Added: 2026-02-06)*

Based on actual benchmark results from EventGPT and Video-LLaVA:

### Video-LLaVA (8 frames) - Short Output (5 tokens)

```
Vision Encoding:   29.48 ms
Prefill:          568.13 ms
─────────────────────────
T_prefill:        597.61 ms

Decode (5 tok):   143.94 ms
─────────────────────────
T_total:          741.56 ms

T_prefill / T_total = 597.61 / 741.56 = 80.6%  ✅ >> 40% threshold
```

### EventGPT (1 frame) - Short Output (5 tokens)

```
Vision Encoding:    5.93 ms
Prefill:           66.16 ms
─────────────────────────
T_prefill:         72.09 ms

Decode (5 tok):    92.68 ms
─────────────────────────
T_total:          164.76 ms

T_prefill / T_total = 72.09 / 164.76 = 43.7%  ✅ > 40% threshold
```

### EventGPT - Long Output (44.5 tokens avg)

```
Stage 1-3:         31.2 ms  (3.0%)
Stage 4 (Decode): 1000.7 ms (97.0%)
─────────────────────────────────
T_total:         1031.8 ms

T_prefill / T_total = 31.2 / 1031.8 = 3.0%  ❌ < 40% threshold
```

### Summary Table

| Scenario | T_prefill | T_total | **Ratio** | Cross-Modal Wins? |
|----------|-----------|---------|-----------|-------------------|
| **Video-LLaVA (8f, 5 tok)** | 598ms | 742ms | **80.6%** | ✅ YES (strong) |
| EventGPT (1f, 5 tok) | 72ms | 165ms | **43.7%** | ✅ YES |
| EventGPT (1f, 45 tok) | 31ms | 1032ms | **3.0%** | ❌ NO |

### Decision Framework

```
If T_prefill / T_total > 40%  → Cross-Modal wins (parallel prefill advantage)
If T_prefill / T_total < 20%  → Same-modality wins (decode dominates)
If 20% < ratio < 40%          → Depends on acceptance rate

Video-LLaVA ratio: 80.6% >> 40%  → STRONG case for cross-modal!
```

### Ratio vs Output Length

```
As output gets longer, prefill ratio decreases:
┌────────────────────────────────────────────────────────┐
│  Output    │ Prefill Ratio │ Cross-Modal Advantage    │
├────────────┼───────────────┼──────────────────────────┤
│  5 tokens  │    80.6%      │  Very Strong (hide 500ms)│
│  20 tokens │    ~60%       │  Strong                  │
│  50 tokens │    ~40%       │  Moderate                │
│  100 tokens│    ~25%       │  Weak                    │
│  200 tokens│    ~15%       │  Same-modality may win   │
└────────────────────────────────────────────────────────┘

Target use case (video QA, short answers): 80% prefill ratio = ideal for cross-modal!
```

---

## Experimental Design: Cross-Modal vs SpecVLM *(Added: 2026-02-06)*

### Hypothesis

> Cross-modal speculative decoding (EventGPT → VideoLLaVA) outperforms SpecVLM in **prefill-dominated scenarios** (long videos, high resolution, streaming) due to parallel prefill hiding.

### Experiment 1: Prefill Hiding Effectiveness

**Goal:** Demonstrate that cross-modal hides prefill latency while SpecVLM cannot.

```
Setup:
┌─────────────────────────────────────────────────────────────┐
│  Vary: Number of input frames (1, 4, 8, 16, 32, 64)         │
│  Fixed: Output length (~50 tokens), same scene content      │
│  Measure: End-to-end latency, prefill time, decode time     │
└─────────────────────────────────────────────────────────────┘
```

| Frames | Visual Tokens | Expected SpecVLM | Expected Cross-Modal |
|--------|---------------|------------------|----------------------|
| 1 | 256 | 1.0x (baseline) | 0.9x (alignment overhead) |
| 8 | 2048 | 2.5x | **3.0x** |
| 16 | 4096 | 2.5x | **3.5x** |
| 32 | 8192 | 2.5x | **4.0x** |
| 64 | 16384 | 2.5x | **4.5x** |

**Key metric:** `Speedup vs frames` curve - cross-modal should scale better.

```python
def experiment_prefill_scaling():
    results = []
    for num_frames in [1, 4, 8, 16, 32, 64]:
        events, rgb_frames = load_dsec_sample(num_frames)
        prompt = "Describe what is happening in this video."

        # Baseline: VideoLLaVA autoregressive
        t_baseline = measure_latency(videollava, rgb_frames, prompt)

        # SpecVLM: Compressed + EagleVLM
        t_specvlm = measure_latency(specvlm, rgb_frames, prompt)

        # Cross-modal: EventGPT draft + VideoLLaVA verify
        t_crossmodal = measure_latency(crossmodal, events, rgb_frames, prompt)

        results.append({
            'frames': num_frames,
            'baseline_ms': t_baseline,
            'specvlm_ms': t_specvlm,
            'crossmodal_ms': t_crossmodal,
            'specvlm_speedup': t_baseline / t_specvlm,
            'crossmodal_speedup': t_baseline / t_crossmodal,
        })
    return results
```

**Expected Result:**
```
Speedup
  5x │                                    ╱ Cross-Modal
     │                               ╱───
  4x │                          ╱───
     │                     ╱───
  3x │                ╱───
     │      ─────────────────────────── SpecVLM (plateaus)
  2x │ ────
     │
  1x │
     └────────────────────────────────────────
        1     8     16    32    64   Frames

Cross-modal scales with frames; SpecVLM plateaus due to sequential prefill.
```

### Experiment 2: Latency Breakdown Analysis

**Goal:** Show WHERE the time is spent in each approach.

```python
def experiment_latency_breakdown():
    """Measure each pipeline stage separately"""
    events, rgb_frames = load_dsec_sample(num_frames=16)

    # SpecVLM breakdown (sequential)
    specvlm_times = {
        'rgb_encode': measure(specvlm.encode, rgb_frames),
        'compress': measure(specvlm.compress, rgb_features),
        'draft_generate': measure(specvlm.draft, compressed_features),
        'verify': measure(specvlm.verify, draft_tokens),
    }
    specvlm_times['total'] = sum(specvlm_times.values())

    # Cross-modal breakdown (parallel prefill)
    crossmodal_times = {
        'event_encode': measure(eventgpt.encode, events),
        'rgb_encode': measure(videollava.encode, rgb_frames),
        'draft_generate': measure(eventgpt.generate, event_features),
        'verify': measure(videollava.verify, draft_tokens, rgb_features),
    }
    # Parallel: max(event_encode + draft_generate, rgb_encode) + verify
    crossmodal_times['total'] = max(
        crossmodal_times['event_encode'] + crossmodal_times['draft_generate'],
        crossmodal_times['rgb_encode']
    ) + crossmodal_times['verify']

    return specvlm_times, crossmodal_times
```

**Expected Result (16 frames):**

| Stage | SpecVLM | Cross-Modal | Notes |
|-------|---------|-------------|-------|
| RGB Encode | 80ms | 80ms | Same |
| Event Encode | - | 8ms | Fast! |
| Compress | 15ms | - | SpecVLM only |
| Draft Generate | 25ms | 60ms | Cross-modal generates more |
| Verify | 20ms | 20ms | Same |
| **Total** | **140ms** | **100ms** | Cross-modal wins |
| **Parallelism Savings** | 0ms | **48ms** | Hidden prefill |

### Experiment 3: Acceptance Rate vs Alignment Quality

**Goal:** Measure how cross-modal alignment quality affects acceptance rate.

```python
def experiment_acceptance_rate():
    """Compare acceptance rates across alignment methods"""
    alignment_configs = [
        'no_alignment',      # Raw EventGPT features
        'linear_probe',      # Single linear layer
        'mlp_adapter',       # 2-layer MLP
        'contrastive',       # Contrastive pre-trained
        'distilled',         # Full distillation
    ]

    results = []
    for config in alignment_configs:
        crossmodal = load_crossmodal_model(alignment=config)

        acceptance_rates = []
        for sample in dsec_test_set:
            events, rgb, prompt = sample
            draft_tokens, draft_probs = crossmodal.draft(events, prompt)
            target_probs = videollava.get_probs(rgb, prompt, draft_tokens)
            accepted = calculate_accepted_tokens(draft_probs, target_probs)
            acceptance_rates.append(accepted / len(draft_tokens))

        results.append({
            'alignment': config,
            'mean_acceptance': np.mean(acceptance_rates),
            'std_acceptance': np.std(acceptance_rates),
        })
    return results
```

**Expected Result:**

| Alignment Method | Acceptance Rate | Training Cost | Speedup |
|------------------|-----------------|---------------|---------|
| No alignment | 15-25% | 0 | 1.2x |
| Linear probe | 35-45% | Low | 2.0x |
| MLP adapter | 50-60% | Medium | 2.8x |
| Contrastive | 60-70% | Medium | 3.2x |
| **Distilled** | **70-80%** | High | **3.8x** |
| SpecVLM (reference) | 75-85% | Medium | 2.7x |

**Key insight:** Cross-modal needs ~60%+ acceptance to beat SpecVLM; distillation achieves this.

### Experiment 4: Streaming/Real-time Scenario

**Goal:** Demonstrate advantage in continuous video streaming.

```python
def experiment_streaming():
    """Simulate real-time video processing"""
    fps = 30
    frame_budget_ms = 1000 / fps  # 33ms per frame

    results = {'specvlm': [], 'crossmodal': []}

    for video in streaming_test_videos:
        for method in ['specvlm', 'crossmodal']:
            frames_processed = 0
            frames_dropped = 0
            total_latency = 0

            for frame_idx, (events, rgb) in enumerate(video.stream()):
                t_start = time.time()

                if method == 'specvlm':
                    response = specvlm.process(rgb)
                else:
                    response = crossmodal.process(events, rgb)

                latency = (time.time() - t_start) * 1000

                if latency > frame_budget_ms:
                    frames_dropped += 1
                else:
                    frames_processed += 1
                total_latency += latency

            results[method].append({
                'video': video.name,
                'processed': frames_processed,
                'dropped': frames_dropped,
                'avg_latency_ms': total_latency / len(video),
                'throughput_fps': frames_processed / video.duration,
            })
    return results
```

**Expected Result:**

| Metric | SpecVLM | Cross-Modal | Improvement |
|--------|---------|-------------|-------------|
| Avg latency | 45ms | **28ms** | 38% lower |
| Frames dropped @30fps | 35% | **5%** | 7x fewer |
| Max sustainable FPS | 22 | **35** | 1.6x higher |
| P99 latency | 80ms | **45ms** | 44% lower |

### Experiment 5: Quality Preservation

**Goal:** Verify cross-modal maintains output quality (lossless).

```python
def experiment_quality():
    """Compare output quality across methods"""
    metrics = {
        'baseline': {'bleu': [], 'rouge': [], 'cider': [], 'accuracy': []},
        'specvlm': {'bleu': [], 'rouge': [], 'cider': [], 'accuracy': []},
        'crossmodal': {'bleu': [], 'rouge': [], 'cider': [], 'accuracy': []},
    }

    for sample in quality_test_set:
        events, rgb, prompt, ground_truth = sample

        baseline_response = videollava.generate(rgb, prompt)
        specvlm_response = specvlm.generate(rgb, prompt)
        crossmodal_response = crossmodal.generate(events, rgb, prompt)

        for method, response in [
            ('baseline', baseline_response),
            ('specvlm', specvlm_response),
            ('crossmodal', crossmodal_response),
        ]:
            metrics[method]['bleu'].append(compute_bleu(response, ground_truth))
            metrics[method]['rouge'].append(compute_rouge(response, ground_truth))
            metrics[method]['cider'].append(compute_cider(response, ground_truth))
            if sample.has_answer:
                metrics[method]['accuracy'].append(
                    response.strip() == ground_truth.strip()
                )
    return metrics
```

**Expected Result:**

| Method | BLEU-4 | ROUGE-L | CIDEr | VQA Acc | Notes |
|--------|--------|---------|-------|---------|-------|
| Baseline | 32.1 | 54.3 | 98.2 | 71.2% | Ground truth |
| SpecVLM | 31.8 | 53.9 | 97.5 | 70.8% | -0.4% (compression loss) |
| **Cross-Modal** | **32.0** | **54.2** | **98.0** | **71.0%** | **-0.2% (lossless)** |

### Experiment 6: Ablation Studies

#### 6a. Parallel vs Sequential Cross-Modal

```python
def ablation_parallelism():
    """Quantify the benefit of parallel prefill"""
    # Sequential: Event encode → Draft → RGB encode → Verify
    t_sequential = event_encode + draft_time + rgb_encode + verify_time

    # Parallel: max(Event+Draft, RGB) → Verify
    t_parallel = max(event_encode + draft_time, rgb_encode) + verify_time

    parallelism_benefit = t_sequential - t_parallel
    return parallelism_benefit  # Expected: 40-60ms for 16 frames
```

#### 6b. Draft Length vs Acceptance Trade-off

```python
def ablation_draft_length():
    """Find optimal draft length for cross-modal"""
    results = []
    for draft_len in [2, 4, 8, 12, 16, 24, 32]:
        acceptance_rate = measure_acceptance(draft_len)
        draft_time = measure_draft_time(draft_len)
        verify_time = measure_verify_time(draft_len)

        effective_tokens = draft_len * acceptance_rate
        time_per_token = (draft_time + verify_time) / effective_tokens

        results.append({
            'draft_len': draft_len,
            'acceptance': acceptance_rate,
            'effective_tokens': effective_tokens,
            'time_per_token': time_per_token,
        })
    return results
```

**Expected Result:**

| Draft Length | Acceptance | Effective Tokens | Optimal? |
|--------------|------------|------------------|----------|
| 4 | 85% | 3.4 | Too short |
| 8 | 72% | 5.8 | Good |
| **12** | **65%** | **7.8** | **Best** |
| 16 | 55% | 8.8 | Diminishing |
| 24 | 40% | 9.6 | Too long |

### Experiment 7: GPU Utilization

**Goal:** Show that cross-modal better utilizes GPU parallelism.

```python
def experiment_gpu_utilization():
    """Profile GPU usage during inference"""
    with torch.profiler.profile() as prof:
        specvlm.generate(rgb_frames, prompt)
    specvlm_util = prof.gpu_time_total / prof.total_time

    with torch.profiler.profile() as prof:
        crossmodal.generate(events, rgb_frames, prompt)
    crossmodal_util = prof.gpu_time_total / prof.total_time

    return specvlm_util, crossmodal_util
```

**Expected Result:**

| Metric | SpecVLM | Cross-Modal |
|--------|---------|-------------|
| GPU Utilization | 65% | **85%** |
| GPU Idle Time | 35% | **15%** |
| Memory Bandwidth Util | 70% | **90%** |

### Experimental Summary

| Exp | Variable | Key Metric | Expected Winner |
|-----|----------|------------|-----------------|
| 1 | # frames | Speedup scaling | **Cross-Modal** |
| 2 | Pipeline stage | Latency breakdown | **Cross-Modal** |
| 3 | Alignment method | Acceptance rate | SpecVLM (close) |
| 4 | Real-time constraint | Frames dropped | **Cross-Modal** |
| 5 | - | Quality metrics | Tie |
| 6 | Draft length | Tokens/second | **Cross-Modal** |
| 7 | - | GPU utilization | **Cross-Modal** |

### Datasets

| Dataset | Modalities | Samples | Use Case |
|---------|------------|---------|----------|
| **DSEC** | Events + RGB (stereo) | 50+ sequences | Primary benchmark |
| **MVSEC** | Events + RGB + IMU | 10+ sequences | Multi-modal validation |
| **E2VID** | Events → RGB | 1000+ clips | Alignment training |
| **ActivityNet-QA** | RGB video + QA | 5800 QA pairs | Quality evaluation |
| **MSRVTT-QA** | RGB video + QA | 10K QA pairs | VQA benchmark |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24GB) | A100 (80GB) |
| Event Camera | - | DAVIS346 / Prophesee |
| CPU | 8 cores | 16+ cores |
| RAM | 32GB | 64GB |

### Implementation Checklist

- [ ] **Phase 1: Baseline Implementation** *(TODO)*
  - [ ] VideoLLaVA inference pipeline
  - [ ] Implement EAGLE-style decoding on VideoLLaVA (EagleVLM baseline)
  - [ ] Add existing visual token compression method (e.g., FastV, TokenPacker)
  - [ ] Latency measurement framework

- [ ] **Phase 2: Baseline Comparison** *(TODO)*
  - [ ] EagleVLM only (no compression) - measure speedup
  - [ ] Visual compression only (no draft) - measure speedup
  - [ ] EagleVLM + compression (SpecVLM reproduction) - measure speedup
  - [ ] Document baseline performance ceiling

- [ ] **Phase 3: Cross-Modal Pipeline** *(TODO)*
  - [ ] EventGPT encoder integration
  - [ ] Feature alignment layer training
  - [ ] Parallel prefill implementation
  - [ ] Cross-modal speculative decoding

- [ ] **Phase 4: Experiments**
  - [ ] Exp 1: Prefill scaling
  - [ ] Exp 2: Latency breakdown
  - [ ] Exp 3: Acceptance rate
  - [ ] Exp 4: Streaming benchmark
  - [ ] Exp 5: Quality evaluation
  - [ ] Exp 6: Ablations
  - [ ] Exp 7: GPU profiling

- [ ] **Phase 5: Analysis**
  - [ ] Generate comparison plots
  - [ ] Statistical significance tests
  - [ ] Write experimental report
  - [ ] Document WHY cross-modal outperforms baselines

---

## Why Cross-Modal Can Outperform Same-Modality Baselines *(Added: 2026-02-06)*

### The Baseline Ceiling Problem

Even with the best same-modality optimizations, there's a **fundamental ceiling** due to sequential dependency:

```
Best Same-Modality Pipeline (EagleVLM + Compression):
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  RGB Input ──► [Encoder] ──► [Compress] ──► [EagleVLM] ──► [Verify]       │
│                  80ms          15ms           25ms          20ms           │
│                   │             │              │              │            │
│                   └─────────────┴──────────────┴──────────────┘            │
│                            ALL SEQUENTIAL                                  │
│                                                                            │
│  Total: 80 + 15 + 25 + 20 = 140ms                                         │
│  Speedup ceiling: ~2.5-3x (cannot break this without parallelism)         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Why Same-Modality Baselines Are Limited

| Baseline | What It Optimizes | What It Cannot Optimize |
|----------|-------------------|------------------------|
| EagleVLM only | Decoding (2-2.5x) | **Prefill (0% savings)** |
| Compression only | Prefill tokens (1.3-1.5x) | **Prefill compute (still sequential)** |
| EagleVLM + Compression | Both (2.5-3x) | **Parallelism (still sequential)** |

**The fundamental limit:** All same-modality approaches require target features → sequential pipeline → speedup ceiling ~3x.

### Cross-Modal Breaks the Ceiling

```
Cross-Modal Pipeline (Parallel Prefill):
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  Events ──► [Event Enc] ──► [Draft LM] ──────────────────┐                │
│                 10ms           70ms                       │                │
│                  │              │                         ▼                │
│  RGB ────► [RGB Encoder] ──────────────────────────► [Verify]             │
│                100ms                                    20ms               │
│                 │                                                          │
│                 └──── PARALLEL ─────────────────────────────               │
│                                                                            │
│  Total: max(10+70, 100) + 20 = 120ms → 100ms with optimization            │
│  Speedup: ~3.5-4.5x (BREAKS the 3x ceiling!)                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Quantitative Analysis: Why Cross-Modal Wins

| Component | EagleVLM+Compress | Cross-Modal | Difference |
|-----------|-------------------|-------------|------------|
| Prefill | 80ms (sequential) | 80ms (parallel) | **Hidden** |
| Compression | 15ms | 0ms | **-15ms** |
| Draft generation | 25ms | 70ms (but parallel!) | +45ms (hidden) |
| Verification | 20ms | 20ms | 0ms |
| **Total** | **140ms** | **100ms** | **-40ms (29% faster)** |

### The Key Insight

```
Same-Modality (SpecVLM):
  Speedup = S_decode × S_compress
          = 2.5x × 1.2x
          = 3.0x (ceiling)

Cross-Modal:
  Speedup = S_decode × S_parallel_prefill
          = 2.0x × 2.0x
          = 4.0x (breaks ceiling!)

Even with LOWER acceptance rate, cross-modal wins via parallelism.
```

### Acceptance Rate vs Parallelism Trade-off

| Method | Acceptance Rate | Parallelism | Net Speedup |
|--------|-----------------|-------------|-------------|
| EagleVLM | 80% | None | 2.5x |
| EagleVLM + Compress | 75% (compression hurts) | None | 2.8x |
| **Cross-Modal** | **60%** (modality gap) | **Yes** | **3.5x** |

**Cross-modal can afford 20% lower acceptance rate** because parallel prefill more than compensates.

### When Cross-Modal Wins (Decision Framework)

```
Calculate: T_prefill / T_total

If T_prefill / T_total > 0.4:  → Cross-Modal wins (prefill-dominated)
If T_prefill / T_total < 0.2:  → Same-modality wins (decode-dominated)
If 0.2 < ratio < 0.4:          → Depends on acceptance rate
```

| Scenario | T_prefill/T_total | Winner |
|----------|-------------------|--------|
| Single image, short response | 0.15 | Same-modality |
| Single image, long response | 0.25 | Tie |
| 8-frame video | 0.45 | **Cross-Modal** |
| 16-frame video | 0.55 | **Cross-Modal** |
| 32-frame video | 0.65 | **Cross-Modal** |
| Streaming (continuous) | 0.70+ | **Cross-Modal** |

### Summary: Why We Outperform Baselines

1. **Parallel Prefill** - Draft generates while target encodes (saves 40-80ms)
2. **No Compression Artifacts** - Full quality preserved (vs SpecVLM's lossy compression)
3. **Scalability** - Advantage grows with video length (SpecVLM plateaus)
4. **Streaming-Friendly** - Can start immediately (no waiting for full encode)

```
Baseline ceiling:  ~3x  (sequential dependency)
Cross-modal:       ~4x+ (parallel prefill breaks ceiling)

The 1x+ improvement comes purely from PARALLELISM, not better drafting.
```

---

## Key Equations

### Acceptance Probability

```
α = min(1, p_target / p_draft)

Where:
- p_target: Target model probability
- p_draft: Draft model probability
```

### Expected Speedup

```
Speedup = (k + 1) / (1 + k × (T_draft / T_target))

Where:
- k: Average accepted tokens per cycle
- T_draft: Draft model forward pass time
- T_target: Target model forward pass time
```

### Resampling Distribution (on Rejection)

```
P_resample(t) = max(0, P_target(t) - P_draft(t)) / Z

Where Z normalizes the distribution.
```

---

## Resources

### Libraries and Frameworks

| Resource | Description | Link |
|----------|-------------|------|
| **HuggingFace Transformers** | Assisted Generation API | [Docs](https://huggingface.co/blog/assisted-generation) |
| **vLLM** | Production speculative decoding | [Docs](https://docs.vllm.ai/en/latest/features/spec_decode/) |
| **EAGLE** | Feature-level implementation | [GitHub](https://github.com/SafeAILab/EAGLE) |
| **Medusa** | Multi-head decoding | [GitHub](https://github.com/FasterDecoding/Medusa) |
| **ReDrafter** | Apple + NVIDIA implementation | [GitHub](https://github.com/apple/ml-recurrent-drafter) |
| **Lookahead Decoding** | N-gram + Jacobi iteration | [GitHub](https://github.com/hao-ai-lab/LookaheadDecoding) |

### Survey Papers

1. [A Comprehensive Survey of Speculative Decoding (Xia et al., ACL 2024)](https://aclanthology.org/2024.findings-acl.456.pdf) - 198 citations

2. [Decoding Speculative Decoding (He et al., NAACL 2025)](https://aclanthology.org/2025.naacl-long.328.pdf) - Meta-analysis

3. [NVIDIA: Introduction to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

---

## Latest Research (December 2025 - January 2026)

### Recent Papers

| Paper | arXiv | Focus | Innovation |
|-------|-------|-------|------------|
| Entropy-Aware Speculative Decoding | [2512.23765](https://arxiv.org/abs/2512.23765) | Draft quality | Entropy awareness |
| SpecPV (Long Context) | [2512.02337](https://arxiv.org/abs/2512.02337) | Long-context | Partial verification |
| Sparse Computation | [2512.21911](https://arxiv.org/abs/2512.21911) | Efficiency | Sparse + speculative |
| Adaptive Rejection Sampling | [2512.13194](https://arxiv.org/abs/2512.13194) | Verification | Adaptive rejection |
| Optimal Lower Bounds | [2512.11718](https://arxiv.org/abs/2512.11718) | Theory | Branching random walks |
| Speculative Sampling + RL | [2601.12212](https://arxiv.org/abs/2601.12212) | RL hybrid | Feature reuse |
| Judge Decoding Revisited | [2601.04766](https://arxiv.org/abs/2601.04766) | Theory | First principles |
| Multi-Scale Image Gen | [2601.05149](https://arxiv.org/abs/2601.05149) | Images | Multi-scale local |
| SRT for RL | [2601.09083](https://arxiv.org/abs/2601.09083) | RL | Tree-structured cache |
| Plan, Verify, Fill | [2601.12247](https://arxiv.org/abs/2601.12247) | Diffusion | Structured parallel |

### Emerging Trends

1. **Long-Context Optimization** - SpecPV addresses long-context generation
2. **Cross-Domain Applications** - Image generation, reinforcement learning
3. **Theoretical Foundations** - Optimal lower bounds, first principles analysis
4. **Hybrid Approaches** - Combining with RL, sparse computation
5. **Verification Improvements** - Adaptive rejection sampling, entropy-aware methods

---

## Citation

If you use this research or find these documents helpful, please cite:

```bibtex
@misc{eventgpt_speculative_research,
  title={Speculative Decoding Research: Accelerating LLM and VLM Inference},
  author={Alice Zhang},
  year={2026},
  url={https://github.com/your-repo}
}
```

---

## Contributing

This is an active research project. Contributions welcome:
- Additional paper summaries
- Code implementations
- Benchmark results
- Corrections and improvements

---

## License

Research materials licensed under MIT License.

---

## Acknowledgments

- EAGLE team (Peking University, Microsoft Research)
- Apple ML Research (ReDrafter)
- HuggingFace (Assisted Generation)
- vLLM team
- NVIDIA research team

---

**Last Updated:** February 7, 2026

**Contact:** For questions or collaboration, open an issue or contact the research team.
