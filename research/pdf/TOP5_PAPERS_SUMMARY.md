# Top 5 Papers for Prefill Hiding Research

**Date:** 2026-01-26
**Selection Criteria:** Most relevant to cross-modal speculative decoding and prefill hiding

---

## Reading Order & Priority

| # | Paper | Venue | Date | Why Read First |
|---|-------|-------|------|----------------|
| 1 | **DSI** | ICLR 2025 | May 2024 | Foundational: hides verification latency via parallelism |
| 2 | **PEARL** | ICLR 2025 | Aug 2024 | Key technique: pre-verify/post-verify overlap |
| 3 | **HiViS** | arXiv | Sept 2025 | VLM-specific: hides visual tokens from drafter |
| 4 | **SpecVLM** | arXiv | Sept 2025 | VLM baseline: EAGLE-2 for vision-language models |
| 5 | **SSD** | ICLR 2026 | Oct 2025 | Advanced: eliminates all speculation overhead |

---

## 1. DSI: Distributed Speculative Inference (ICLR 2025)

**File:** `1_DSI_Distributed_Speculative_Inference_ICLR2025.pdf`
**arXiv:** [2405.14105](https://arxiv.org/abs/2405.14105)
**Date:** May 2024

### Summary
DSI introduces "Speculation Parallelism" (SP) to overlap verification with drafting, transforming speculative inference from a blocking to non-blocking algorithm.

### Key Contributions
1. **Speculation Parallelism:** New type of task parallelism that orchestrates target and drafter to overlap in time
2. **Provably faster:** DSI is proven faster than both SI and non-SI for any drafters
3. **Hardware flexibility:** Works with arbitrary number of GPUs (≥2)

### Core Insight
> "Unlike SI, which blocks drafting until verification is complete, DSI overlaps verification with drafting, effectively **hiding verification latency**."

### Results
- 1.29-1.92x faster than standard speculative inference
- Never slower than either SI or non-SI for any configuration

### Relevance to Our Work
- **Similar goal:** Hide latency through parallel execution
- **Different phase:** DSI hides verification; we hide prefill
- **Different scope:** DSI needs multi-GPU; we can work single-GPU

---

## 2. PEARL: Parallel Speculative Decoding (ICLR 2025)

**File:** `2_PEARL_Parallel_Speculative_Decoding_ICLR2025.pdf`
**arXiv:** [2408.11850](https://arxiv.org/abs/2408.11850)
**Date:** August 2024

### Summary
PEARL addresses the "mutual waiting problem" in speculative decoding through two strategies: pre-verify and post-verify.

### Key Contributions
1. **Pre-verify:** Verify first draft token while drafting remaining tokens
2. **Post-verify:** Generate more drafts while verification is running
3. **Adaptive draft length:** Dynamically adjust based on scenario

### Core Insight
> "The target model gets stuck when the draft model is guessing tokens, and vice versa. This mutual waiting problem is exacerbated due to fixed draft length."

### Results
- Up to **4.43x** speedup over autoregressive decoding
- **1.50x** speedup over vanilla speculative decoding

### Relevance to Our Work
- **Similar spirit:** Overlap draft and target operations
- **Applicable:** PEARL can be combined with our prefill hiding
- **Key technique:** Post-verify is analogous to our hidden drafting

---

## 3. HiViS: Hiding Visual Tokens from Drafter (Sept 2025)

**File:** `3_HiViS_Hiding_Visual_Tokens_2025.pdf`
**arXiv:** [2509.23928](https://arxiv.org/abs/2509.23928)
**Date:** September 2025

### Summary
HiViS removes all visual tokens from the drafter's input, using only text tokens while reusing target VLM's hidden states.

### Key Contributions
1. **Explicit-implicit decomposition:** Text as explicit input, visual via hidden states
2. **Massive compression:** Drafter prefill reduced to 0.7%-1.3% of target input
3. **Multi-step self-feedback training:** Simulate reasoning during training

### Core Insight
> "Visual tokens in large VLMs are highly redundant, and most can be removed without compromising generation quality."

### Results
- Up to **2.65x** speedup
- Maintains lossless generation quality

### Relevance to Our Work
- **Related concept:** "Hiding" tokens/compute from drafter
- **Different approach:** HiViS hides tokens; we hide prefill time
- **Complementary:** Could combine HiViS-style token hiding with our prefill hiding

---

## 4. SpecVLM: Fast Speculative Decoding in VLMs (Sept 2025)

**File:** `4_SpecVLM_Fast_Speculative_Decoding_VLM_2025.pdf`
**arXiv:** [2509.11815](https://arxiv.org/abs/2509.11815)
**Date:** September 2025

### Summary
SpecVLM establishes an EAGLE-2 style baseline for VLMs with elastic visual token compression.

### Key Contributions
1. **EagleVLM:** EAGLE-2 baseline for vision-language models
2. **Elastic compressor:** Adaptive selection among pruning, pooling, convolution, resampler
3. **Question-adaptive:** Compression adapts to query type

### Core Insight
> "The draft model's latency is increasingly dominated by KV cache size rather than parameter count alone, especially for video/long-context inputs."

### Results
- 1.5-2.3x end-to-end speedups (EagleVLM baseline)
- 2.5-2.9x speedups with elastic compression

### Relevance to Our Work
- **Baseline comparison:** SpecVLM is current SOTA for VLM speculation
- **Different approach:** SpecVLM compresses same modality; we use different modality
- **Potential combination:** Could use SpecVLM compression on Video-LLaVA

---

## 5. SSD: Speculative Speculative Decoding (ICLR 2026 Submission)

**File:** `5_SSD_Speculative_Speculative_Decoding_ICLR2026.pdf`
**OpenReview:** [aL1Wnml9Ef](https://openreview.net/forum?id=aL1Wnml9Ef)
**Date:** October 2025

### Summary
SSD predicts likely verification outcomes and pre-computes speculations, eliminating all speculation overhead.

### Key Contributions
1. **Speculation cache:** Pre-compute speculations for predicted outcomes
2. **Outcome prediction:** Draft model predicts verification results
3. **Zero overhead:** If outcome in cache, return immediately

### Core Insight
> "While verification is ongoing, the draft model predicts likely verification outcomes and prepares speculations pre-emptively. If actual outcome is in the predicted set, speculation can be returned immediately."

### Results
- Eliminates speculation overhead when predictions are correct
- Higher throughput by only increasing draft compute (cheap)

### Relevance to Our Work
- **Advanced technique:** Could extend our approach with speculation caching
- **Different focus:** SSD caches outcomes; we exploit timing asymmetry
- **Complementary:** SSD decode-phase + our prefill-phase = combined speedup

---

## Comparison Matrix

| Paper | What's Hidden | Phase | Multi-GPU | VLM-Specific | Speedup |
|-------|--------------|-------|-----------|--------------|---------|
| **DSI** | Verification latency | Decode | Yes | No | 1.3-1.9x |
| **PEARL** | Mutual waiting | Decode | No | No | 1.5x over SD |
| **HiViS** | Visual tokens | Prefill+Decode | No | Yes | 2.65x |
| **SpecVLM** | Visual tokens (compressed) | Prefill+Decode | No | Yes | 2.5-2.9x |
| **SSD** | Speculation overhead | Decode | Yes | No | Variable |
| **Ours** | Draft prefill + generation | Prefill | No | Yes | 1.3-1.7x |

---

## Reading Strategy

### Day 1: Foundations
- Read **DSI** (Paper 1) - Understand parallel speculation concept
- Read **PEARL** (Paper 2) - Understand pre-verify/post-verify

### Day 2: VLM-Specific
- Read **SpecVLM** (Paper 4) - Understand VLM speculation baseline
- Read **HiViS** (Paper 3) - Understand visual token hiding

### Day 3: Advanced
- Read **SSD** (Paper 5) - Understand speculation caching
- Re-read relevant sections from all papers

### Key Sections to Focus On
| Paper | Key Sections |
|-------|--------------|
| DSI | Section 3 (Method), Section 4.1 (Proofs) |
| PEARL | Section 3.2 (Pre-verify), Section 3.3 (Post-verify) |
| HiViS | Section 3 (Method), Section 4.2 (Training) |
| SpecVLM | Section 3 (EagleVLM), Section 4 (Compressor) |
| SSD | Section 3 (Framework), Section 4 (Caching) |

---

## How These Inform Our Work

### From DSI:
- Formalization of "hiding latency" as a parallelization problem
- Theoretical framework for proving speedup guarantees

### From PEARL:
- Pre-verify/post-verify can be combined with our prefill hiding
- Adaptive draft length could help optimize our ~27 token window

### From HiViS:
- Concept of "hiding" visual information from drafter
- Training strategies for cross-modal alignment

### From SpecVLM:
- Baseline comparison for VLM speculative decoding
- Visual compression techniques as complement to our approach

### From SSD:
- Speculation caching could extend our method post-prefill
- Outcome prediction for better acceptance rates

---

**Document Created:** 2026-01-26
**PDFs Downloaded:** 5/5 ✅
