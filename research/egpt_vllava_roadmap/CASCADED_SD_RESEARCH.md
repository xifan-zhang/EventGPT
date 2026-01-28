# Cascaded Speculative Decoding: Research Opportunities

## Table of Contents
1. [Research Opportunity](#research-opportunity)
2. [Advantages of Cascaded SD](#advantages-of-cascaded-sd)
3. [Light → Middle → Dense Modality Cascade](#light--middle--dense-modality-cascade)
4. [Top 5 Papers](#top-5-papers)
5. [Justification for Paper Selection](#justification-for-paper-selection)
6. [Research Gaps & Opportunities](#research-gaps--opportunities)

---

## Research Opportunity

Cascaded speculative decoding (SD) extends the two-model draft-verify paradigm to **multi-stage hierarchies**, where progressively larger models refine drafts before final verification. This is a rapidly evolving field with significant opportunities, especially for:

### 1. Cross-Modality Cascades (Novel Direction)
**Unexplored territory**: Light modality → Middle modality → Dense modality

```
Event Camera (sparse) → RGB Frame (medium) → Video (dense) → VLM
     ↓                      ↓                    ↓
 EventGPT draft      Frame-aligned draft    VideoLLaVA verify
```

**Why this matters:**
- Different modalities have natural "entropy gradients" - sparse representations are faster but less precise
- Cross-modal cascading can exploit modality-specific speedups (e.g., event cameras are 100x sparser than video)
- No existing work addresses modality cascading for SD

### 2. Hierarchical Model Size Cascades (Established)
```
1B draft → 3B qualifier → 8B target (PyramidSD achieves 1.91x speedup)
```

### 3. Hybrid Vision-Language Cascades
```
Vision Encoder (fast) → Draft LLM → Target VLM
```

---

## Advantages of Cascaded SD

### Theoretical Advantages

| Advantage | Description | Evidence |
|-----------|-------------|----------|
| **Entropy Gradient Exploitation** | Larger models have lower entropy (higher confidence); intermediate models bridge the gap | PyramidSD: 1B→3B→8B achieves 124 tok/s |
| **Incremental Verification** | Each stage filters low-quality drafts before expensive verification | HiSpec: 4x faster verification with early-exit |
| **Distribution Alignment** | Smaller gaps between consecutive models → higher acceptance rates | α_total = Π α_i improves with smaller gaps |
| **Parallel Pipeline Potential** | Multiple models can run asynchronously | PipeSpec: eliminates idle periods |

### Empirical Speedups

| Method | Configuration | Speedup vs Vanilla | Speedup vs 1-Stage SD |
|--------|---------------|--------------------|-----------------------|
| PyramidSD | 1B→3B→8B | 1.91x | +28% over single stage |
| HiSpec | 3B→70B w/ early-exit | 2.5x | +15-20% |
| PipeSpec | k-model pipeline | 2.8x | +25-35% |
| Faster Cascades | Multiple drafters | 2.92x | +17% |

### When Cascading Works Best

1. **Clear model hierarchy** (sizes differ by 3-10x)
2. **High per-stage acceptance** (>75% each stage)
3. **Compute-abundant environments** (multiple GPUs)
4. **Trained alignment layers** (feature matching between stages)

---

## Light → Middle → Dense Modality Cascade

### Concept: Cross-Modal Speculative Decoding

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LIGHT MODALITY → DENSE MODALITY                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stage 1: Light Modality (Event Camera)                            │
│  ├── Input: Sparse event stream (100x fewer pixels)                │
│  ├── Model: EventGPT encoder (~0.5B params)                        │
│  ├── Output: Draft visual tokens                                   │
│  └── Latency: ~2ms                                                 │
│                                                                     │
│  Stage 2: Middle Modality (RGB Frame)                              │
│  ├── Input: Single frame + Stage 1 features                        │
│  ├── Model: ViT encoder (~0.3B params)                             │
│  ├── Output: Refined visual tokens                                 │
│  └── Latency: ~10ms                                                │
│                                                                     │
│  Stage 3: Dense Modality (Video/VLM)                               │
│  ├── Input: Multi-frame + Stage 2 features                         │
│  ├── Model: VideoLLaVA-7B                                          │
│  ├── Output: Verified tokens + text generation                     │
│  └── Latency: ~100ms                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Cascade Makes Sense

1. **Natural Entropy Gradient Across Modalities**
   - Event cameras capture change (sparse, low entropy)
   - RGB frames capture appearance (medium entropy)
   - Video captures temporal context (high entropy, high information)

2. **Compute Scaling Matches Information Density**
   - Sparse modalities → small models → fast drafts
   - Dense modalities → large models → accurate verification

3. **Existing Infrastructure**
   - EventGPT already trained for event→language
   - VideoLLaVA handles video→language
   - Gap: alignment between event features and video features

### Research Questions

| Question | Current Status | Opportunity |
|----------|----------------|-------------|
| Can event features draft for video models? | Unexplored | Train alignment adapter |
| What's the optimal modality cascade depth? | Unknown | Empirical study needed |
| How to handle temporal misalignment? | Partial (SpecVLM) | Cross-modal attention |
| Can lightweight vision models draft for VLMs? | SpecVLM, ViSpec | Extend to event→video |

### Proposed EventGPT → VideoLLaVA Cascade

```python
# Proposed architecture
class CrossModalCascadeSD:
    """
    Stage 1: EventGPT (sparse events → draft features)
    Stage 2: VideoLLaVA (dense frames → verified features)
    Stage 3: LLM decoder (features → text)
    """

    def __init__(self):
        self.event_encoder = EventGPTEncoder()      # ~0.5B
        self.alignment_mlp = AlignmentMLP()          # ~10M (trainable)
        self.video_llava = VideoLLaVA7B()            # ~7B

    def forward(self, events, frames, text_prompt):
        # Stage 1: Fast event drafting
        event_features = self.event_encoder(events)  # 2ms
        draft_features = self.alignment_mlp(event_features)

        # Stage 2: Video verification
        video_features = self.video_llava.encode_video(frames)  # 50ms

        # Stage 3: Speculative verify
        acceptance = self.compute_acceptance(draft_features, video_features)

        if acceptance > threshold:
            # Use draft features (fast path)
            return self.video_llava.decode(draft_features, text_prompt)
        else:
            # Fall back to video features (slow path)
            return self.video_llava.decode(video_features, text_prompt)
```

---

## Top 5 Papers

### Paper 1: Faster Cascades via Speculative Decoding (ICLR 2025)

| Attribute | Value |
|-----------|-------|
| **Authors** | H. Narasimhan et al. |
| **Venue** | ICLR 2025 |
| **arXiv** | [2405.19261](https://arxiv.org/abs/2405.19261) |
| **Local PDF** | `pdf/faster_cascades_via_speculative_decoding_ICLR2025.pdf` |

**Key Contributions:**
- Unifies cascades (deferral-based) with speculative decoding (speculation-based)
- Introduces "lossy speculative sampling" for general target distributions
- Shows 2-3 draft stages provide 17%+ improvement over single stage

**Relevance to EventGPT:**
- Provides theoretical foundation for cascading different models
- Deferral rules can be adapted for modality-based routing (event vs. video)

---

### Paper 2: HiSpec - Hierarchical Speculative Decoding (arXiv 2510.01336)

| Attribute | Value |
|-----------|-------|
| **Authors** | Research team 2025 |
| **arXiv** | [2510.01336](https://arxiv.org/abs/2510.01336) |
| **Local PDF** | `pdf/HiSpec_hierarchical_speculative_decoding.pdf` |

**Key Contributions:**
- Uses early-exit (EE) models for intermediate verification
- Verification is 4x faster than drafting in some configurations
- Addresses verification bottleneck (ignored by most prior work)

**Relevance to EventGPT:**
- Early-exit can be applied to vision encoders
- EventGPT encoder could serve as "early exit" for VideoLLaVA vision

---

### Paper 3: PyramidSD - 3-Model Speculative Decoding (arXiv 2510.12966)

| Attribute | Value |
|-----------|-------|
| **Authors** | S. Byun et al. |
| **Venue** | NeurIPS 2025 Workshop |
| **arXiv** | [2510.12966](https://arxiv.org/abs/2510.12966) |
| **Local PDF** | `pdf/PyramidSD_3model_speculative_decoding.pdf` |

**Key Contributions:**
- 3-model cascade: draft → qualifier → target
- Exploits "entropy gradient" across model sizes
- Achieves 1.91x speedup (124 tok/s on RTX 4090)
- No additional training required

**Relevance to EventGPT:**
- Entropy gradient concept applies to modalities (event = low entropy, video = high entropy)
- Qualifier model concept → EventGPT as qualifier for VideoLLaVA

---

### Paper 4: PipeSpec - Breaking Stage Dependencies (arXiv 2505.01572)

| Attribute | Value |
|-----------|-------|
| **Authors** | Research team 2025 |
| **arXiv** | [2505.01572](https://arxiv.org/abs/2505.01572) |
| **Local PDF** | `pdf/PipeSpec_hierarchical_LLM_decoding.pdf` |

**Key Contributions:**
- k-model asynchronous pipeline architecture
- Producer-consumer relationship between model pairs
- Eliminates idle periods in hierarchical decoding
- Higher-quality drafts from intermediate models

**Relevance to EventGPT:**
- Pipelining applicable to event processing + video encoding
- Asynchronous design suits real-time event camera applications

---

### Paper 5: SpecVLM - Fast Speculative Decoding in VLMs (arXiv 2509.11815)

| Attribute | Value |
|-----------|-------|
| **Authors** | Research team 2025 |
| **arXiv** | [2509.11815](https://arxiv.org/abs/2509.11815) |
| **Local PDF** | `pdf/SpecVLM_vision_language_speculative_decoding.pdf` |

**Key Contributions:**
- First comprehensive SD framework for Vision-Language Models
- Elastic visual compressor (pruning, pooling, convolution, resampler)
- Online-logit distillation (no offline corpus needed)
- 1.5-2.3x end-to-end speedup on VLMs

**Relevance to EventGPT:**
- Directly applicable to EventGPT→VideoLLaVA
- Visual compression techniques can reduce event→video gap
- Online distillation avoids expensive dataset creation

---

## Justification for Paper Selection

### Selection Criteria

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| **Recency** | High | 2024-2025 papers capture latest advances |
| **Cascade Architecture** | High | Multi-stage is core to our research |
| **VLM Applicability** | High | EventGPT→VideoLLaVA is vision-language |
| **Theoretical Foundation** | Medium | Need principled framework |
| **Implementation Feasibility** | Medium | Must be reproducible |

### Why These 5 Papers?

1. **Faster Cascades (ICLR 2025)**: Foundational theory for combining cascades + SD
2. **HiSpec**: Novel verification acceleration (usually overlooked)
3. **PyramidSD**: Practical 3-model cascade with clear entropy gradient insight
4. **PipeSpec**: Asynchronous pipelining for real-time applications
5. **SpecVLM**: Direct VLM application with visual-specific optimizations

### Papers NOT Selected (and why)

| Paper | Reason for Exclusion |
|-------|---------------------|
| EAGLE/EAGLE-2 | Focus on single-model internal drafting, not cascade |
| Medusa | Multi-head drafting, but not multi-model cascade |
| Draft & Verify | Self-speculative (single model), not cascade |
| Lookahead | N-gram cache, not model cascade |

---

## Research Gaps & Opportunities

### Gap 1: Cross-Modal Cascading (Major Opportunity)

**Current state**: No work on light modality → dense modality cascade
**Opportunity**: EventGPT (event) → ViT (frame) → VideoLLaVA (video)

**Research questions:**
- How to align features across modalities?
- What's the acceptance rate for cross-modal verification?
- Can events "draft" for video models effectively?

### Gap 2: Modality-Specific Draft Acceptance

**Current state**: Acceptance based on token-level probability
**Opportunity**: Modality-aware acceptance (e.g., spatial vs. temporal confidence)

### Gap 3: Real-Time Cascading for Event Cameras

**Current state**: All SD work assumes batch processing
**Opportunity**: Streaming SD for event camera applications

### Gap 4: Training-Free Cross-Modal Alignment

**Current state**: SpecVLM requires online distillation
**Opportunity**: Zero-shot alignment using frozen encoders + learned MLP

---

## Recommended Next Steps

1. **Read SpecVLM paper** - Most directly applicable to EventGPT→VideoLLaVA
2. **Implement PyramidSD baseline** - Understand entropy gradient empirically
3. **Design cross-modal acceptance metric** - Event features vs. video features
4. **Train alignment MLP** - EventGPT features → VideoLLaVA compatible features
5. **Benchmark cascaded vs. single-stage** - Measure actual speedup on DSEC

---

## Summary Table: Downloaded Papers

| # | Paper | Year | Venue | Key Concept | PDF Location |
|---|-------|------|-------|-------------|--------------|
| 1 | Faster Cascades via SD | 2025 | ICLR | Lossy speculative sampling | `pdf/faster_cascades_via_speculative_decoding_ICLR2025.pdf` |
| 2 | HiSpec | 2025 | arXiv | Early-exit verification | `pdf/HiSpec_hierarchical_speculative_decoding.pdf` |
| 3 | PyramidSD | 2025 | NeurIPS-W | 3-model entropy gradient | `pdf/PyramidSD_3model_speculative_decoding.pdf` |
| 4 | PipeSpec | 2025 | arXiv | Async k-model pipeline | `pdf/PipeSpec_hierarchical_LLM_decoding.pdf` |
| 5 | SpecVLM | 2025 | arXiv | VLM-specific SD | `pdf/SpecVLM_vision_language_speculative_decoding.pdf` |

---

**Document Created:** January 28, 2026
**Research Focus:** Cascaded Speculative Decoding for EventGPT → VideoLLaVA

## Sources

- [Faster Cascades via Speculative Decoding](https://arxiv.org/abs/2405.19261)
- [HiSpec: Hierarchical Speculative Decoding for LLMs](https://arxiv.org/abs/2510.01336)
- [PyramidSD / 3-Model Speculative Decoding](https://arxiv.org/abs/2510.12966)
- [PipeSpec: Breaking Stage Dependencies](https://arxiv.org/abs/2505.01572)
- [SpecVLM: Fast Speculative Decoding in VLMs](https://arxiv.org/abs/2509.11815)
- [Speculative Cascades - Google Research](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)
- [SpeculativeDecodingPapers Repository](https://github.com/hemingkx/SpeculativeDecodingPapers)
