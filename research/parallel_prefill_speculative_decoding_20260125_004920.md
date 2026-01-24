# Parallel Prefilling & Speculative Decoding Research Opportunities

**Date:** 2026-01-25
**Focus:** EventGPT (Draft) + Video-LLaVA (Target) Speculative Decoding

---

## Table of Contents

1. [Overview](#overview)
2. [Parallel Prefilling](#parallel-prefilling)
3. [Cross-Model Speculative Decoding](#cross-model-speculative-decoding)
4. [Research Opportunities for EventGPT + Video-LLaVA](#research-opportunities)
5. [Key Challenges](#key-challenges)
6. [Proposed Approaches](#proposed-approaches)
7. [References](#references)

---

## Overview

This document surveys the latest research (2025-2026) on **parallel prefilling** and **speculative decoding** for Vision-Language Models (VLMs), with a focus on identifying research opportunities for using EventGPT as a draft model to accelerate Video-LLaVA inference.

### Key Insight

> "VLMs differ from text-only LLMs by requiring seamless integration of visual and textual information. The prefill stage is dominated by visual tokens whose count scales with image resolution and video length, inflating both compute and memory."

---

## Parallel Prefilling

### Background

LLM inference consists of two distinct phases:
1. **Prefill Phase**: Process all input tokens in parallel, populate KV caches
2. **Decode Phase**: Generate tokens one-by-one autoregressively

For VLMs, the prefill phase is significantly more expensive due to visual token processing.

### Disaggregated Prefill-Decode Architecture

**Key Papers (2025):**

| Paper | Date | Key Contribution |
|-------|------|------------------|
| Nexus: Proactive Intra-GPU Disaggregation | Aug 2025 | Intra-GPU separation of prefill/decode |
| TPLA: Tensor Parallel Latent Attention | Aug 2025 | Optimized disaggregated inference |
| SPAD: Specialized Prefill and Decode Hardware | Oct 2025 | Hardware-level optimization |

**Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                DISAGGREGATED INFERENCE                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐         ┌─────────────┐               │
│  │  PREFILL    │ ──KV──► │  DECODE     │               │
│  │  (GPU/TPU)  │  Cache  │  (GPU/Edge) │               │
│  └─────────────┘         └─────────────┘               │
│       │                        │                        │
│       ▼                        ▼                        │
│  Heavy visual tokens     Sequential text gen            │
│  High throughput         Low latency                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Benefits:**
- Prefill and decode can be optimized independently
- Prefill optimized for throughput (batch many visual tokens)
- Decode optimized for latency (fast autoregressive)
- Can use different hardware for each phase

### Parallel Prefill Techniques

**1. Visual Token Parallelization:**
```
Input Video (8 frames) → Parallel Vision Encoding → Merged KV Cache
                            ├── Frame 0 → GPU 0
                            ├── Frame 1 → GPU 1
                            ├── ...
                            └── Frame 7 → GPU 7
```

**2. Speculative Prefill (NEW OPPORTUNITY):**
```
EventGPT (sparse, fast) ─────────┐
                                  │ Parallel Prefill
Video-LLaVA (dense, slow) ───────┘
                                  │
                                  ▼
                          Merged/Aligned Features
```

---

## Cross-Model Speculative Decoding

### The Tokenizer Mismatch Problem

EventGPT and Video-LLaVA use different tokenizers, leading to:
- Incompatible vocabulary mappings
- Different BPE segmentation
- Misaligned token boundaries

**Current Results (from your benchmarks):**
- Token-level acceptance rate: ~2-5% (very low due to tokenizer mismatch)
- This makes vanilla speculative decoding impractical

### Solutions from Recent Research

#### 1. OmniDraft Framework (2025)

**Paper:** "OmniDraft: A cross-vocabulary, online adaptive drafter"

**Approach:**
- Cross-vocabulary mappings from draft tokens to target tokens
- N-gram cache integration to alleviate vocabulary mismatches
- Online knowledge distillation during inference

**Relevance to EventGPT:**
```
EventGPT tokens ──► Vocabulary Mapping ──► Video-LLaVA tokens
                         │
                         ▼
                   N-gram Cache
                   (learned mapping)
```

#### 2. CTPD: Cross Tokenizer Preference Distillation (2026)

**Paper:** arXiv 2601.11865

**Key Innovation:** Aligned spans mechanism
- Aggregate signals (log-probs, weights) from teacher tokens within a span
- Project onto corresponding student tokens in same span
- Guarantees sound basis for cross-tokenizer distillation

**Application:**
```
EventGPT span: ["The", "car", "is"]
Video-LLaVA span: ["The", " car", " is"]
                      │
              Span Alignment
                      │
                      ▼
         Feature-level distillation
```

#### 3. GRIFFIN: Token Alignment for Speculative Decoding (Feb 2025)

**Paper:** "GRIFFIN: Effective Token Alignment for Faster Speculative Decoding"

**Approach:** Aligns tokens across different model vocabularies for cross-family speculative decoding.

#### 4. Pyramid Speculative Decoding (PyramidSD)

**Key Insight:** Insert intermediate "qualifier" model between draft and target to bridge distributional gap.

**For EventGPT + Video-LLaVA:**
```
EventGPT (1B) ──► Qualifier (3B) ──► Video-LLaVA (7B)
  (event)          (bridge)           (video)
```

---

## Research Opportunities

### Opportunity 1: Cross-Modal Speculative Prefill

**Novelty:** No existing work on event-to-video speculative prefilling.

**Concept:**
- EventGPT processes sparse event data rapidly
- Generates "draft" visual features
- Video-LLaVA verifies/refines in parallel

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│           CROSS-MODAL SPECULATIVE PREFILL                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Event Images ──► EventGPT Encoder ──► Draft Features        │
│       │                                      │               │
│       │                                      ▼               │
│       │                              Alignment Layer         │
│       │                                      │               │
│       ▼                                      ▼               │
│  Video Frames ──► VideoLLaVA Encoder ──► Target Features     │
│                          │                   │               │
│                          └───────────────────┘               │
│                                  │                           │
│                                  ▼                           │
│                          Verification + Merge                │
│                                  │                           │
│                                  ▼                           │
│                          KV Cache (optimized)                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Expected Speedup:** 2-5x on prefill phase (vision encoding)

**Research Questions:**
1. How to align event features to video features efficiently?
2. What acceptance rate can be achieved at feature level?
3. Can we use sparse event attention to skip redundant video tokens?

### Opportunity 2: Feature-Level Speculative Decoding (EAGLE-style)

**Current State:** EAGLE-3 achieves 5-6.5x speedup on text LLMs.

**Adaptation for EventGPT + Video-LLaVA:**

Instead of predicting tokens, predict features:
```
EventGPT hidden states ──► Draft Head ──► Predicted Video-LLaVA features
                                               │
                                               ▼
                                     Video-LLaVA LM Head
                                               │
                                               ▼
                                          Verification
```

**Key Insight from EAGLE:**
> "Features contain more information and exhibit more regularity than final token distributions."

**Architecture:**
```python
# Feature-level speculative decoding
class EventToVideoEAGLE(nn.Module):
    def __init__(self, eventgpt_dim=4096, videollava_dim=4096):
        self.feature_predictor = nn.Sequential(
            nn.Linear(eventgpt_dim * 2, eventgpt_dim * 4),
            nn.SiLU(),
            nn.Linear(eventgpt_dim * 4, videollava_dim),
        )

    def predict_feature(self, eventgpt_features, prev_feature):
        concat = torch.cat([eventgpt_features, prev_feature], dim=-1)
        return self.feature_predictor(concat)
```

**Training Data:** Use your existing DSEC alignment dataset.

### Opportunity 3: Medusa-style Multi-Head Drafting

**Concept:** Add multiple prediction heads to EventGPT that predict Video-LLaVA token distributions.

**Architecture:**
```
EventGPT backbone
       │
       ├──► Head 0: Predict t+1 (Video-LLaVA vocab)
       ├──► Head 1: Predict t+2 (Video-LLaVA vocab)
       ├──► Head 2: Predict t+3 (Video-LLaVA vocab)
       └──► Head 3: Predict t+4 (Video-LLaVA vocab)
```

**Training:** Knowledge distillation from Video-LLaVA.

### Opportunity 4: ViSpec/SpecVLM Adaptation

**SpecVLM (Sept 2025):** EAGLE-2 style baseline for VLMs achieving 1.5-2.3x speedup.

**ViSpec (Sept 2025):** Vision-aware speculative decoding with:
- Lightweight vision adaptor for token compression
- Integration of compressed tokens in draft model attention

**Adaptation for EventGPT:**
```
Events ──► EventGPT Vision Adaptor ──► Compressed Event Tokens
                                              │
                                              ▼
                                     Integrate into Draft
                                              │
                                              ▼
Video-LLaVA ──────────────────────────► Verification
```

### Opportunity 5: DREAM-style Cross-Attention Fusion

**DREAM (May 2025):** Cross-attention mechanism for multimodal speculative decoding.

**Key Innovations:**
1. Cross-attention to inject target features into draft
2. Entropy-adaptive feature selection
3. Visual token compression

**Application:**
```
EventGPT features ◄───────────────────┐
       │                              │
       ▼                              │
Cross-Attention Layer ◄── Video-LLaVA intermediate features
       │
       ▼
Enhanced Draft Predictions
```

**Expected Speedup:** Up to 3.6x (based on DREAM results)

---

## Key Challenges

### Challenge 1: Different Visual Representations

| Model | Visual Encoder | Token Count | Resolution |
|-------|----------------|-------------|------------|
| EventGPT | CLIP + Event Processing | ~576 tokens | 336×336 |
| Video-LLaVA | LanguageBind | ~2048 tokens | 224×224 per frame |

**Solutions:**
- Learnable alignment layer (your current approach)
- Pooling/projection to match dimensions
- Cross-attention instead of direct replacement

### Challenge 2: Temporal Alignment

Events capture different temporal characteristics than video frames:
- Events: microsecond resolution, sparse, edge-focused
- Video: millisecond resolution, dense, full appearance

**Research Direction:** Learn temporal alignment that maps event temporal resolution to video frame semantics.

### Challenge 3: Different Tokenizers

As shown in your speculative_decoding_S1.py results:
- Very low token-level acceptance (~2-5%)
- Different vocabularies prevent direct token matching

**Solutions:**
1. **Feature-level decoding** (bypass tokenizer)
2. **Shared LM head** (use Video-LLaVA's LM head on aligned features)
3. **Cross-vocabulary mapping** (OmniDraft approach)
4. **Distillation** to shared vocabulary

### Challenge 4: Memory Constraints

24GB VRAM is limited for both models:
- Your current approach: Sequential model loading
- Better: Feature caching + partial model loading

---

## Proposed Approaches

### Approach A: Two-Stage Speculative Decoding (Recommended)

**Stage 1: Vision Speculative Prefill**
```
Events ──► EventGPT Encoder (fast) ──► Draft Vision Features
                                              │
                                    Alignment Layer (trained)
                                              │
Video ──► Video-LLaVA Encoder (slow) ──► Verify/Refine Features
```
- Cache verified features
- Skip redundant video encoding when event prediction is confident

**Stage 2: Text Speculative Decoding**
```
Cached Vision Features + Text ──► EventGPT LM (fast) ──► Draft Tokens
                                                              │
                                     (Using Video-LLaVA's LM head)
                                                              │
                                    Video-LLaVA LM (slow) ──► Verify
```

**Expected Speedup:**
| Stage | Speedup | Notes |
|-------|---------|-------|
| Vision Prefill | 3-5x | Event vs video encoding |
| Text Decode | 1.5-2x | Shared feature, different LM |
| **Total** | **2-3x** | End-to-end |

### Approach B: EAGLE-3 Style Feature Prediction

Train a lightweight feature predictor:
```python
class EventVideoEAGLE3(nn.Module):
    """EAGLE-3 style feature predictor for EventGPT → Video-LLaVA"""

    def __init__(self):
        # Multi-layer feature fusion (EAGLE-3 innovation)
        self.low_proj = nn.Linear(4096, 1024)   # Layer 8
        self.mid_proj = nn.Linear(4096, 1024)   # Layer 16
        self.high_proj = nn.Linear(4096, 1024)  # Layer 24

        self.fusion = nn.Linear(3072, 4096)

        # Feature predictor
        self.predictor = nn.TransformerEncoderLayer(
            d_model=4096, nhead=16, dim_feedforward=16384
        )

    def forward(self, eventgpt_features, layer_features):
        # Fuse multi-layer features
        low = self.low_proj(layer_features['layer_8'])
        mid = self.mid_proj(layer_features['layer_16'])
        high = self.high_proj(layer_features['layer_24'])

        fused = self.fusion(torch.cat([low, mid, high], dim=-1))

        # Predict next Video-LLaVA feature
        return self.predictor(fused)
```

**Training:**
1. Extract features from both models on aligned dataset
2. Train predictor to minimize MSE between predicted and actual Video-LLaVA features
3. Use Video-LLaVA's LM head for token generation

### Approach C: Medusa Heads for Cross-Model

Add Medusa-style heads to EventGPT:
```python
class EventGPTMedusa(nn.Module):
    def __init__(self, eventgpt, videollava_vocab_size=32000):
        self.base = eventgpt
        self.videollava_vocab = videollava_vocab_size

        # Medusa heads predicting Video-LLaVA tokens
        self.medusa_heads = nn.ModuleList([
            nn.Linear(4096, videollava_vocab_size)
            for _ in range(4)  # Predict 4 tokens ahead
        ])

    def forward(self, x):
        hidden = self.base.get_hidden_states(x)

        # Predict multiple future Video-LLaVA tokens
        predictions = [head(hidden) for head in self.medusa_heads]
        return predictions
```

**Training:** Distillation from Video-LLaVA outputs.

---

## Implementation Roadmap

### Phase 1: Feature-Level Alignment (Current)
- [x] Train alignment layer for vision features
- [ ] Measure feature-level acceptance rate
- [ ] Benchmark feature caching approach

### Phase 2: Feature-Level Speculative Decoding
- [ ] Implement EAGLE-style feature predictor
- [ ] Train on DSEC alignment dataset
- [ ] Benchmark end-to-end speedup

### Phase 3: Cross-Vocabulary Mapping
- [ ] Implement OmniDraft-style vocabulary mapping
- [ ] Train span alignment for EventGPT → Video-LLaVA
- [ ] Measure token-level improvement

### Phase 4: Full Integration
- [ ] Combine vision prefill + text decode speculation
- [ ] Optimize for 24GB VRAM constraint
- [ ] Production benchmarks

---

## Expected Contributions

1. **First cross-modal speculative decoding** for event camera + video VLMs
2. **Sparse-to-dense speculative prefilling** technique
3. **Cross-vocabulary mapping** for event-video domain
4. **End-to-end acceleration** with quality preservation

---

## References

### Parallel Prefilling
- [Disaggregated Prefill and Decode](https://www.perplexity.ai/hub/blog/disaggregated-prefill-and-decode) - Perplexity 2025
- [Nexus: Proactive Intra-GPU Disaggregation](https://arxiv.org/) - Aug 2025
- [LLM Inference Optimization Techniques](https://www.clarifai.com/blog/llm-inference-optimization/) - Clarifai 2025

### Speculative Decoding
- [EAGLE-3](https://arxiv.org/abs/2503.01840) - NeurIPS 2025
- [SpecForge Training](https://lmsys.org/blog/2025-07-25-spec-forge/) - LMSYS 2025
- [Decoding Speculative Decoding](https://aclanthology.org/2025.naacl-long.328.pdf) - NAACL 2025

### Multimodal Speculative Decoding
- [SpecVLM](https://arxiv.org/abs/2509.11815) - Sept 2025
- [ViSpec](https://arxiv.org/abs/2509.15235) - NeurIPS 2025
- [DREAM](https://arxiv.org/abs/2505.19201) - May 2025
- [Speculative Decoding Reimagined for MLLMs](https://arxiv.org/abs/2505.14260) - May 2025
- [MASSV](https://arxiv.org/abs/2505.10526) - May 2025

### Cross-Tokenizer Methods
- [OmniDraft](https://openreview.net/pdf?id=RALtozQipi) - 2025
- [CTPD: Cross Tokenizer Preference Distillation](https://arxiv.org/abs/2601.11865) - Jan 2026
- [GRIFFIN](https://arxiv.org/) - Feb 2025
- [Pyramid Speculative Decoding](https://arxiv.org/abs/2510.12966) - 2025

### Event Camera Vision
- [Hybrid Spiking Vision Transformer](https://arxiv.org/abs/2505.07715) - ICML 2025
- [PMRVT: Parallel Attention MLP RVT](https://pmc.ncbi.nlm.nih.gov/articles/PMC12610684/) - 2025
- [Event Cameras in 2025](https://lenzgregor.com/posts/event-cameras-2025-part2/) - Gregor's Blog

### Production Frameworks
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [EAGLE GitHub](https://github.com/SafeAILab/EAGLE)
- [Medusa GitHub](https://github.com/FasterDecoding/Medusa)
- [NVIDIA Speculative Decoding Guide](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

---

## Summary

**Key Findings:**
1. **Parallel prefilling** is well-established for LLMs but unexplored for cross-modal VLM settings
2. **Cross-tokenizer speculative decoding** is an active research area with recent solutions (OmniDraft, CTPD, GRIFFIN)
3. **Feature-level decoding** (EAGLE-style) bypasses tokenizer mismatch and achieves higher acceptance rates
4. **Multimodal speculative decoding** is nascent (SpecVLM, ViSpec, DREAM) with 1.5-3.6x speedups

**Research Opportunity:**
EventGPT + Video-LLaVA represents a **unique cross-modal speculative decoding** scenario:
- Sparse (events) → Dense (video) visual encoding
- Different temporal resolutions
- Different tokenizers

**Recommended Approach:**
1. Start with **feature-level speculative decoding** to bypass tokenizer issues
2. Use your **trained alignment layer** as the bridge
3. Implement **EAGLE-3 style multi-layer feature fusion**
4. Add **cross-attention** (DREAM-style) for better draft quality

**Expected Outcome:** 2-3x end-to-end speedup with quality preservation.

---

**Last Updated:** 2026-01-25 00:49:20

**Status:** Complete

**Related Documents:**
- [EventGPT → VideoLLaVA Roadmap](EventGPT_VideoLLaVA_roadmap.md)
- [Feature-Level Speculative Decoding](embedding_level_speculative_decoding.md)
- [Cross-Modal Speculative Prefill](cross_modal_speculative_prefill.md)
