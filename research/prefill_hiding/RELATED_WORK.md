# Related Work: Parallel and Asynchronous Speculative Decoding

**Date:** 2026-01-26
**Research Survey for Prefill Hiding**

---

## Executive Summary

Our "Prefill Hiding" approach—where the draft model's prefill is hidden inside the target model's longer prefill window—has **strong connections to but remains distinct from** recent parallel/asynchronous speculative decoding research. The key papers surveyed demonstrate growing interest in overlapping draft and target execution, but **none specifically address cross-modal (event→video) speculation with prefill-time exploitation**.

---

## 1. Most Related: Parallel Execution of Draft and Target

### 1.1 DSI: Distributed Speculative Inference (ICLR 2025)

**Paper:** [Distributed Speculative Inference (DSI)](https://arxiv.org/abs/2405.14105)

**Key Innovation:** Overlaps verification with drafting by running them on separate threads/GPUs.

> "Unlike SI, which blocks drafting until verification is complete, DSI overlaps verification with drafting, transforming SI into a non-blocking algorithm and **effectively hiding verification latency**."

**How it works:**
- Uses "Speculation Parallelism (SP)" to orchestrate target and drafter instances
- While target verifies iteration T, drafter generates iteration T+1
- Requires multiple GPUs (≥2)

**Results:** 1.29-1.92x faster than standard speculative inference

**Relation to Our Work:**
| Aspect | DSI | Prefill Hiding (Ours) |
|--------|-----|----------------------|
| What's hidden | Verification latency | Draft prefill + generation |
| When | During decode phase | During target prefill phase |
| Multi-GPU required | Yes | No (can be sequential) |
| Cross-modal | No | Yes (event→video) |

**Gap:** DSI focuses on the decode phase overlap, not prefill phase.

---

### 1.2 PEARL: Parallel Speculative Decoding (ICLR 2025)

**Paper:** [PEARL: Parallel Speculative Decoding with Adaptive Draft Length](https://arxiv.org/abs/2408.11850)

**Key Innovation:** Pre-verify and post-verify strategies to overlap drafting and verification.

> "PEARL proposes **pre-verify** to verify the first draft token in advance during the drafting phase, and **post-verify** to generate more draft tokens during the verification phase."

**How it works:**
- Pre-verify: Verify first draft token while still drafting remaining tokens
- Post-verify: Generate additional drafts while verification is running
- Adaptive draft length based on scenario

**Results:** Up to 4.43x speedup over autoregressive, 1.50x over vanilla speculative decoding

**Relation to Our Work:**
- Similar spirit: overlap draft and target operations
- Different focus: PEARL targets decode phase, we target prefill phase
- **Our contribution:** First to exploit the asymmetric prefill times of different modalities

---

### 1.3 SSD: Speculative Speculative Decoding (ICLR 2026 Submission)

**Paper:** [Speculative Speculative Decoding](https://openreview.net/forum?id=aL1Wnml9Ef)

**Key Innovation:** Predict likely verification outcomes and pre-compute speculations for them.

> "While a verification is ongoing, the draft model **predicts** likely verification outcomes and prepares speculations pre-emptively for them. If the actual verification outcome is then in the predicted set, a speculation can be returned immediately, thereby **eliminating all speculation overhead**."

**How it works:**
- Draft model runs continuously on separate hardware
- Builds "speculation cache" of pre-computed outcomes
- When verification completes, checks if outcome was predicted

**Relation to Our Work:**
- SSD focuses on **eliminating draft overhead** during decode
- Our focus: exploit prefill time asymmetry for **free draft generation**
- Both hide draft computation, but in different phases

---

### 1.4 AMUSD: Asynchronous Multi-Device Speculative Decoding

**Paper:** [AMUSD](https://arxiv.org/abs/2410.17375)

**Key Innovation:** Continuous parallel execution of draft and verify on separate devices.

> "AMUSD leverages multiple devices (e.g., GPUs) to enable simultaneous and independent predictions from both the draft and verify models."

**How it works:**
- Draft model produces tokens continuously
- Verify model validates asynchronously
- Rollback mechanism handles conflicts

**Results:** Up to 1.96x speedup

**Relation to Our Work:**
- AMUSD requires multi-GPU setup
- Our approach can work with single GPU + sequential loading
- Both exploit parallelism, different hardware assumptions

---

### 1.5 SwiftSpec: Ultra-Low Latency Speculative Decoding (June 2025)

**Paper:** [SwiftSpec](https://arxiv.org/abs/2506.11309)

**Key Innovation:** Disaggregated, asynchronous pipeline that removes draft from critical path.

> "SwiftSpec redesigns the speculative decoding pipeline in an asynchronous and disaggregated manner, so that each component can be scaled flexibly and **remove draft overhead from the critical path**."

**Key techniques:**
- Parallel tree generation across GPU groups
- Asymmetric scaling (different TP for draft vs target)
- KV-cache consistency management

**Results:** 1.75x speedup over SOTA, Llama3-70B at 348 tokens/s on 8 GPUs

**Relation to Our Work:**
- SwiftSpec focuses on system-level disaggregation
- Our focus: exploit inherent timing asymmetry between modalities
- SwiftSpec requires multiple GPUs; our approach can work single-GPU

---

## 2. Vision-Language Model Speculative Decoding

### 2.1 HiViS: Hiding Visual Tokens from Drafter (September 2025)

**Paper:** [HiViS](https://arxiv.org/abs/2509.23928)

**Key Innovation:** Remove visual tokens from drafter entirely, use target's hidden states.

> "All visual tokens are removed from the drafter's input, retaining only textual tokens as explicit inputs, while directly reusing the target VLM's corresponding last-layer hidden states."

**Problem solved:**
1. Visual tokens from target misaligned with drafter's encoding
2. Large visual token count slows drafter's self-attention

**Results:**
- Compresses drafter prefill to **0.7%-1.3%** of target's input length
- Up to **2.65x speedup**

**Relation to Our Work:**
| Aspect | HiViS | Prefill Hiding (Ours) |
|--------|-------|----------------------|
| Approach | Hide visual tokens from drafter | Hide draft prefill inside target prefill |
| Modality | Same (RGB→RGB) | Cross-modal (Event→RGB) |
| Draft input | Text only + target hidden states | Full event input |
| Novelty | Token hiding | Temporal exploitation |

**Gap filled by us:** HiViS hides tokens, we hide the entire prefill phase.

---

### 2.2 SpecVLM: Fast Speculative Decoding for VLMs (September 2025)

**Paper:** [SpecVLM](https://arxiv.org/abs/2509.11815)

**Key Innovation:** EAGLE-2 baseline for VLMs with elastic visual compression.

> "SpecVLM introduces a practical system that establishes a strong EAGLE-2-style baseline (EagleVLM), delivering 1.5–2.3x end-to-end speedups."

**Key techniques:**
- Visual token compression (pruning, pooling, convolution, resampler)
- Question-adaptive compression selection
- Lossless decoding guarantee

**Results:** 2.5-2.9x speedup within 5 epochs

**Relation to Our Work:**
- SpecVLM focuses on compressing visual tokens
- Our approach uses fundamentally different modality (events)
- Complementary: could combine event camera + visual compression

---

### 2.3 ViSpec: Vision-Aware Speculative Decoding (September 2025)

**Paper:** [ViSpec](https://arxiv.org/abs/2509.15235)

**Key Innovation:** Lightweight vision adaptor for token compression in speculative setting.

> "ViSpec proposes Vision-Aware Speculative Decoding, incorporating a lightweight vision adaptor module to compress numerous image tokens into a compact, informative representation."

**Relation to Our Work:**
- ViSpec compresses within same modality
- Our approach: fundamentally different modality with inherent compression

---

## 3. Key Gap in Literature

### No Existing Work On:

1. **Cross-modal speculative decoding** where draft and target use different sensor modalities
2. **Exploiting prefill time asymmetry** between models (not just token count asymmetry)
3. **Event camera → Video VLM** speculation pipeline
4. **"Free" draft generation** during target's prefill window

### Our Novel Contributions:

| Contribution | Novelty |
|--------------|---------|
| **Cross-modal speculation** | First to use event camera as draft for video VLM target |
| **Prefill time exploitation** | First to systematically hide draft prefill in target prefill |
| **Free draft tokens** | ~27 tokens generated at zero additional latency cost |
| **Sparse→Dense speculation** | New paradigm: sparse temporal data drafts for dense RGB |

---

## 4. Positioning Our Work

### Research Landscape

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPECULATIVE DECODING RESEARCH LANDSCAPE                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DECODE PHASE PARALLELISM                    PREFILL OPTIMIZATION            │
│  ┌─────────────────────────┐                 ┌─────────────────────────┐   │
│  │ DSI (ICLR 2025)         │                 │ HiViS (Sept 2025)       │   │
│  │ PEARL (ICLR 2025)       │                 │ SpecVLM (Sept 2025)     │   │
│  │ SSD (ICLR 2026)         │                 │ ViSpec (Sept 2025)      │   │
│  │ AMUSD (Oct 2024)        │                 │                         │   │
│  │ SwiftSpec (June 2025)   │                 │                         │   │
│  └─────────────────────────┘                 └─────────────────────────┘   │
│           │                                            │                     │
│           │                                            │                     │
│           └────────────────┬───────────────────────────┘                     │
│                            │                                                  │
│                            ▼                                                  │
│           ┌─────────────────────────────────────────────┐                   │
│           │         PREFILL HIDING (OURS)               │                   │
│           │  ─────────────────────────────────────────  │                   │
│           │  • Cross-modal: Event → Video               │                   │
│           │  • Exploits prefill time asymmetry          │                   │
│           │  • Free draft tokens during target prefill  │                   │
│           │  • Single-GPU feasible                      │                   │
│           └─────────────────────────────────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Paper Positioning Statement

> "While recent work has explored parallel execution of draft and target models during the decode phase (DSI, PEARL, SSD) and visual token optimization during prefill (HiViS, SpecVLM), no prior work has exploited the **fundamental timing asymmetry between different visual modalities**. We introduce Prefill Hiding, which leverages the 8.59x prefill speed advantage of event-based vision (EventGPT) over frame-based vision (Video-LLaVA) to generate ~27 draft tokens entirely within the target's prefill window, achieving effective 'free' speculation."

---

## 5. References

### Parallel/Asynchronous Speculative Decoding
- [DSI: Distributed Speculative Inference](https://arxiv.org/abs/2405.14105) - ICLR 2025
- [PEARL: Parallel Speculative Decoding](https://arxiv.org/abs/2408.11850) - ICLR 2025
- [SSD: Speculative Speculative Decoding](https://openreview.net/forum?id=aL1Wnml9Ef) - ICLR 2026 Submission
- [AMUSD: Asynchronous Multi-Device Speculative Decoding](https://arxiv.org/abs/2410.17375) - IEEE 2025
- [SwiftSpec: Ultra-Low Latency Speculative Decoding](https://arxiv.org/abs/2506.11309) - June 2025

### Vision-Language Model Speculative Decoding
- [HiViS: Hiding Visual Tokens](https://arxiv.org/abs/2509.23928) - September 2025
- [SpecVLM: Fast Speculative Decoding in VLMs](https://arxiv.org/abs/2509.11815) - September 2025
- [ViSpec: Vision-Aware Speculative Decoding](https://arxiv.org/abs/2509.15235) - September 2025
- [MASSV: Multimodal Adaptation for VLM Speculative Decoding](https://arxiv.org/abs/2505.10526) - May 2025

### Speculative Decoding Foundations
- [Decoding Speculative Decoding](https://aclanthology.org/2025.naacl-long.328.pdf) - NAACL 2025
- [EAGLE-3](https://arxiv.org/abs/2503.01840) - NeurIPS 2025
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [NVIDIA Speculative Decoding Guide](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

### Event Camera Research
- [Low-latency automotive vision with event cameras](https://www.nature.com/articles/s41586-024-07409-w) - Nature 2024
- [Event cameras in 2025](https://lenzgregor.com/posts/event-cameras-2025-part2/) - Gregor's Blog
- [High-efficiency sparse convolution for event cameras](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1537673/full) - Frontiers 2025

---

**Document Created:** 2026-01-26
**Status:** Complete
