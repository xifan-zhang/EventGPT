# EventGPT → VideoLLaVA Acceleration Roadmap

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EVENTGPT → VIDEOLLaVA ACCELERATION ROADMAP                  │
│              Sparse Events → Dense Video Embedding + Token Decoding            │
└─────────────────────────────────────────────────────────────────────────────────┘

VIDEO INPUT
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: VISION ENCODING                            │
│                     (EMBEDDING-LEVEL SPECULATIVE)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Event Camera ────┐                                                        │
│    (Sparse frames)  │                                                        │
│                      ▼                                                        │
│              ┌──────────────┐                                               │
│              │  EventGPT    │  ← DRAFT MODEL (Fast, Sparse)                │
│              │   Encoder    │     - Event-based feature extraction         │
│              │              │     - Processes sparse frames                │
│              └──────┬───────┘                                               │
│                     │ Draft Features (g_event)                              │
│                     │ ~100x less computation                                │
│                     ▼                                                        │
│              ┌──────────────┐                                               │
│              │  Alignment   │     • Feature mapping layer                  │
│              │     Layer    │     • Projects EventGPT → VideoLLaVA space   │
│              │  (Trainable) │     • Single MLP or adapter                  │
│              └──────┬───────┘                                               │
│                     │ Aligned Features (g_aligned)                         │
│                     ▼                                                        │
│              ┌──────────────┐                                               │
│              │  VideoLLaVA  │  ← TARGET MODEL (Verify)                      │
│              │   Encoder    │     - Dense frame processing                 │
│              │              │     - Verifies features                       │
│              └──────┬───────┘                                               │
│                     │                                                        │
│              Accepted Features (if high similarity)                          │
│              Rejected → Fallback to full VideoLLaVA encoding                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     │ Vision Features (Cached for all subsequent tokens)
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 2: LANGUAGE DECODING                          │
│                      (TOKEN-LEVEL SPECULATIVE)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Text Prompt + Vision Features                                            │
│                     │                                                        │
│                     ▼                                                        │
│              ┌──────────────┐                                               │
│              │  EventGPT    │  ← DRAFT MODEL (Lightweight)                  │
│              │  Language    │     - ~1B parameters                         │
│              │    Model     │     - Trained on vision+text features         │
│              └──────┬───────┘                                               │
│                     │ Draft Tokens (k tokens)                               │
│                     │ Sequential generation                                 │
│                     ▼                                                        │
│              ┌──────────────┐                                               │
│              │  VideoLLaVA  │  ← TARGET MODEL (Verify)                      │
│              │     LM       │     - 7B or 13B parameters                    │
│              │              │     - Parallel verification of k tokens       │
│              └──────┬───────┘                                               │
│                     │                                                        │
│              Accept/Reject each token                                        │
│              Accepted → append to output                                     │
│              Rejected → resample from target distribution                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
GENERATED TEXT (Video Description)
```

---

## Detailed Specifications

### Stage 1: Embedding-Level Speculative Decoding (Vision)

#### Draft Model: EventGPT Encoder
```
Input: Sparse event frames (1-5 fps vs 30 fps dense)
Output: Feature embeddings g_event ∈ R^{d_model}

Characteristics:
- Processes only key frames (sparse sampling)
- Event-based attention reduces computation
- 50-100x faster than dense video encoder
- Captures temporal dynamics efficiently
```

#### Alignment Layer
```
g_aligned = W_align @ g_event + b_align

Where:
- W_align ∈ R^{d_target × d_event}
- Training objective: ||g_aligned - g_videollava||²
- Can be:
  * Simple linear projection
  * MLP (2-3 layers)
  * Adapter (Houlsby et al., 2019)
  * LoRA (Hu et al., 2021)
```

#### Verification Strategy
```
Similarity Score:
    S = cosine_similarity(g_aligned, g_videollava)

Accept if:
    S > threshold AND S > S_baseline

Where:
- threshold: empirically set (~0.8-0.9)
- S_baseline: similarity of random projection
```

#### Fallback Strategy
```
If rejected:
    1. Compute full VideoLLaVA encoding (dense)
    2. Use dense features for this frame
    3. Update alignment layer (online learning)

If accepted:
    1. Use aligned features directly
    2. Save ~100x computation
```

---

### Stage 2: Token-Level Speculative Decoding (Language)

#### Draft Model: EventGPT Language Model
```
Input: Vision features + Text prompt
Output: k draft tokens

Characteristics:
- Trained on VideoLLaVA's output distribution
- Uses cached vision features (from Stage 1)
- Standard autoregressive generation
- Target: VideoLLaVA-7B or VideoLLaVA-13B
```

#### Training Objective
```
L_draft = Σ_t KL(P_EventGPT(·|context, vision) || P_VideoLLaVA(·|context, vision))

Where:
- KL divergence aligns draft distribution with target
- Vision features are the same (from Stage 1)
- Context includes text history
```

#### Acceptance Algorithm
```
For each draft token t_i:
    q_i = P_EventGPT(t_i | context, vision)
    p_i = P_VideoLLaVA(t_i | context, vision)

    α_i = min(1, p_i / q_i)

    if random() < α_i:
        accept(t_i)
    else:
        resample from p_adjusted
        break  # Stop after rejection

Expected Acceptance Rate: 60-80% (with good training)
```

---

## Implementation Plan

### Phase 1: Alignment Layer Training (Weeks 1-4)

**Goal:** Train EventGPT → VideoLLaVA feature alignment

**Steps:**
1. Extract features from both models
   ```python
   # EventGPT features
   g_event = eventgpt_encoder.extract(video_sparse)

   # VideoLLaVA features (ground truth)
   g_target = videollava_encoder.extract(video_dense)

   # Training data: 10K video clips
   ```

2. Train alignment layer
   ```python
   g_aligned = alignment_layer(g_event)
   loss = MSE(g_aligned, g_target)

   # Train until cosine similarity > 0.85
   ```

3. Verify on holdout set
   ```python
   # Target: >80% acceptance rate
   # Speedup: >50x on vision encoding
   ```

### Phase 2: Language Model Training (Weeks 5-8)

**Goal:** Train EventGPT LM as draft model for VideoLLaVA

**Steps:**
1. Collect training data
   ```python
   # VideoLLaVA generations (target)
   # Vision features (from Phase 1)
   # Text prompts
   ```

2. Train EventGPT LM with KL divergence
   ```python
   loss = KL(
       P_eventgpt(token | vision, text),
       P_videollava(token | vision, text)
   )

   # Train for ~100K examples
   # Target: perplexity close to VideoLLaVA
   ```

3. Evaluate acceptance rate
   ```python
   # Target: >60% token acceptance
   # Speedup: 2-3x on language generation
   ```

### Phase 3: End-to-End Integration (Weeks 9-12)

**Goal:** Combine Stage 1 + Stage 2

**Steps:**
1. Implement speculative decoding pipeline
2. Optimize batch processing
3. Add fallback mechanisms
4. Benchmark on video QA tasks

---

## Expected Performance

### Speedup Breakdown

| Stage | Baseline | With Speculative | Speedup |
|-------|----------|------------------|---------|
| Vision Encoding (Dense) | 100 ms | 2 ms (EventGPT) | 50x |
| Vision Encoding (Sparse acceptance) | - | 10 ms (w/ verification) | 10x |
| Language Generation | 500 ms | 200 ms (draft + verify) | 2.5x |
| **Total** | **600 ms** | **210 ms** | **~2.9x** |

### Accuracy

| Metric | VideoLLaVA (Baseline) | EventGPT Draft | End-to-End |
|--------|----------------------|----------------|------------|
| Video QA Accuracy | 75% | - | 74-75% |
| Caption Quality (CIDEr) | 85 | - | 84-85 |
| Feature Similarity | - | 0.85 | - |

---

## Challenges and Mitigations

### Challenge 1: Feature Space Mismatch
**Problem:** EventGPT and VideoLLaVA use different architectures

**Solution:**
- Train alignment layer on large dataset (10K+ videos)
- Use adversarial training for better alignment
- Consider LoRA adapters for fine-tuning

### Challenge 2: Temporal Granularity
**Problem:** Events are sparse, VideoLLaVA expects dense frames

**Solution:**
- EventGPT processes sparse keyframes
- Interpolation for missing frames
- Use event representations (temporal windows)

### Challenge 3: Error Propagation
**Problem:** Bad vision features → bad language generation

**Solution:**
- Strict acceptance threshold (>0.85 similarity)
- Fallback to dense encoding when uncertain
- Online learning to adapt alignment

### Challenge 4: Training Data
**Problem:** Need paired EventGPT-VideoLLaVA training data

**Solution:**
- Use VideoLLaVA's training data (videos + captions)
- Extract events from videos using event simulator
- Or use sparse sampling from dense videos

---

## Research Questions

1. **What's the optimal sparsity for EventGPT draft?**
   - Test: 1 fps, 2 fps, 5 fps, 10 fps
   - Trade-off: Speed vs. feature quality

2. **How does temporal resolution affect acceptance?**
   - Events capture motion differently
   - May need temporal interpolation

3. **Can we train EventGPT end-to-end for this task?**
   - Joint training of encoder + alignment
   - Potentially better than post-hoc alignment

4. **What's the best architecture for alignment?**
   - Linear layer vs MLP vs Adapter vs LoRA
   - Trade-off: Expressiveness vs overfitting

5. **How does this generalize to other video tasks?**
   - Video QA (primary focus)
   - Video captioning
   - Action recognition

---

## Milestones

| Milestone | Target | Deadline | Status |
|-----------|--------|----------|--------|
| Feature extraction setup | Both models working | Week 2 | ⬜ |
| Alignment layer trained | Cosine sim > 0.8 | Week 4 | ⬜ |
| Language model trained | KL loss minimized | Week 8 | ⬜ |
| End-to-end pipeline | Working integration | Week 10 | ⬜ |
| Benchmarking | VideoQA eval | Week 12 | ⬜ |

---

## References

### Feature Alignment
- [Adapter Layers (Houlsby et al., 2019)](https://arxiv.org/abs/1902.00751)
- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)

### Speculative Decoding
- [EAGLE: Feature-Level Speculative Decoding](https://arxiv.org/abs/2401.15077)
- [Token-Level Speculative Sampling (Chen et al., 2023)](https://arxiv.org/abs/2302.01318)

### Multimodal Acceleration
- [SpecVLM: Fast Speculative Decoding in VLMs](https://arxiv.org/abs/2509.11815)
- [ViSpec: Vision-Aware Speculative Decoding](https://arxiv.org/abs/2509.15235)

---

## Summary

**Key Insight:** Use EventGPT's sparse efficiency at both levels:
- **Vision (Embedding):** Sparse event features → dense video features
- **Language (Token):** Small LM → large LM with shared vision context

**Expected Outcome:** 2-3x overall speedup with minimal quality loss

**Novelty:** First work combining:
1. Cross-architecture speculative decoding (EventGPT → VideoLLaVA)
2. Two-level acceleration (vision + language)
3. Sparse → dense feature alignment

---

## LLaVA Video Model Family Analysis

### Available Video-LLM Families (January 2026)

Three main families support video understanding with multiple model sizes:

#### 1. LLaVA-NeXT-Video (May 2024)
| Model | Parameters | Base LLM | HuggingFace ID | Video Support |
|-------|------------|----------|----------------|---------------|
| **LLaVA-NeXT-Video-7B** | 7B | Vicuna-7B | `llava-hf/LLaVA-NeXT-Video-7B-hf` | ✅ 32 frames |
| **LLaVA-NeXT-Video-32B** | 32B | Yi-34B | `lmms-lab/LLaVA-NeXT-Video-32B` | ✅ 32 frames |
| **LLaVA-NeXT-Video-34B-DPO** | 34B | Yi-34B | `llava-hf/LLaVA-NeXT-Video-34B-DPO-hf` | ✅ 32 frames |

#### 2. LLaVA-OneVision (August 2024) - Recommended
| Model | Parameters | Base LLM | HuggingFace ID | Video Support |
|-------|------------|----------|----------------|---------------|
| **LLaVA-OneVision-0.5B** | 0.5B | Qwen2-0.5B | `lmms-lab/llava-onevision-qwen2-0.5b-ov` | ✅ |
| **LLaVA-OneVision-7B** | 7B | Qwen2-7B | `lmms-lab/llava-onevision-qwen2-7b-ov` | ✅ |
| **LLaVA-OneVision-72B** | 72B | Qwen2-72B | `lmms-lab/llava-onevision-qwen2-72b-ov-chat` | ✅ |

#### 3. LLaVA-Video (November 2024) - Latest
| Model | Parameters | Base LLM | HuggingFace ID | Video Support |
|-------|------------|----------|----------------|---------------|
| **LLaVA-Video-7B** | 7B | Qwen2-7B | `lmms-lab/LLaVA-Video-7B-Qwen2` | ✅ 64 frames |
| **LLaVA-Video-72B** | 72B | Qwen2-72B | `lmms-lab/LLaVA-Video-72B-Qwen2` | ✅ 64 frames |

### Optimal Light/Heavy Pairs for Cascaded Speculative Decoding

#### Option A: LLaVA-OneVision Family (Best for Cascade)
```
Light Draft:   LLaVA-OneVision-0.5B  (~1 GB)   ← Ultra-fast drafting
     ↓
Mid Draft:     LLaVA-OneVision-7B    (~14 GB)  ← Good quality/speed balance
     ↓
Heavy Target:  LLaVA-OneVision-72B   (~144 GB) ← Highest quality
```

**Why Best for Cascade:**
- Same Qwen2 tokenizer across all sizes → high token acceptance
- Same architecture → minimal distribution gap
- 0.5B option enables 3-stage cascade (0.5B → 7B → 72B)

#### Option B: LLaVA-NeXT-Video Family
```
Light Draft:   LLaVA-NeXT-Video-7B   (~14 GB)
     ↓
Heavy Target:  LLaVA-NeXT-Video-34B  (~68 GB)
```

**Pros:** Well-tested, DPO-optimized 34B variant
**Cons:** No ultra-light (<7B) option, different base LLMs (Vicuna vs Yi)

#### Option C: LLaVA-Video Family (Latest)
```
Light Draft:   LLaVA-Video-7B   (~14 GB)
     ↓
Heavy Target:  LLaVA-Video-72B  (~144 GB)
```

**Pros:** Latest models, 64-frame support, best video understanding
**Cons:** Large gap (7B → 72B), no intermediate sizes

### Expected Token Acceptance Rates

| Cascade Configuration | Expected α | Rationale |
|----------------------|------------|-----------|
| **OneVision 0.5B → 7B** | 60-70% | Same family, small gap |
| **OneVision 7B → 72B** | 55-65% | Same family, large gap |
| **OneVision 0.5B → 7B → 72B** | 45-55% | Compound (0.65 × 0.60) |
| **NeXT-Video 7B → 34B** | 50-60% | Different base LLMs |
| **LLaVA-Video 7B → 72B** | 55-65% | Same Qwen2 base |

### Memory Requirements by Configuration

| Configuration | Draft VRAM | Target VRAM | Total | Hardware |
|--------------|------------|-------------|-------|----------|
| OneVision 0.5B → 7B | 1 GB | 14 GB | **15 GB** | RTX 4090 ✅ |
| OneVision 7B → 72B | 14 GB | 144 GB | **158 GB** | 2× H100 |
| OneVision 0.5B → 7B → 72B | 1+14 GB | 144 GB | **159 GB** | 2× H100 |
| NeXT-Video 7B → 34B | 14 GB | 68 GB | **82 GB** | A100-80GB |
| LLaVA-Video 7B → 72B | 14 GB | 144 GB | **158 GB** | 2× H100 |

### Recommendation: LLaVA-OneVision for Cascaded Speculative

**Primary Choice:** `LLaVA-OneVision 0.5B → 7B → 72B`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          RECOMMENDED 3-STAGE CASCADE WITH LLAVA-ONEVISION                   │
└─────────────────────────────────────────────────────────────────────────────┘

EventGPT (sparse encoder)
    │
    │ Embedding alignment (86.8% acceptance)
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Ultra-Light Draft                                                 │
│  LLaVA-OneVision-0.5B (~1 GB, ~5ms/token)                                   │
│  → Generate 8-10 draft tokens                                               │
│  → Expected acceptance to 7B: 60-70%                                        │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Mid-Weight Verifier + Draft                                       │
│  LLaVA-OneVision-7B (~14 GB, ~25ms/token)                                   │
│  → Verify 0.5B tokens, accept ~65%                                          │
│  → Generate refined drafts for 72B                                          │
│  → Expected acceptance to 72B: 55-65%                                       │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Heavy Target                                                      │
│  LLaVA-OneVision-72B (~144 GB, ~100ms/token)                                │
│  → Final verification                                                       │
│  → Highest quality output                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
OUTPUT (72B quality at ~2.5-3x speed)
```

**Why This Configuration:**
1. **0.5B is extremely fast** (~20x faster than 7B) → aggressive drafting
2. **Same Qwen2 tokenizer** across all three → high acceptance
3. **7B serves dual role:** verifies 0.5B drafts AND drafts for 72B
4. **Proven architecture** from LLaVA team with video support

**Alternative (Lower Memory):** `LLaVA-OneVision 7B → 72B` (skip 0.5B stage)

---

## Cascaded Speculative Decoding: EventGPT → Light → Heavy

### Architecture Overview

A two-stage (or three-stage) cascade where EventGPT accelerates a lightweight decoder, which in turn accelerates a heavy decoder.

**Recommended Configuration (using LLaVA-OneVision):**
- **Light:** LLaVA-OneVision-0.5B or 7B
- **Heavy:** LLaVA-OneVision-72B (or LLaVA-NeXT-Video-34B for lower memory)

See [LLaVA Video Model Family Analysis](#llava-video-model-family-analysis) above for all options.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            CASCADED SPECULATIVE DECODING ARCHITECTURE                       │
│          EventGPT → VideoLLaVA-7B (draft) → Heavy-13B (target)             │
└─────────────────────────────────────────────────────────────────────────────┘

VIDEO INPUT (Event Camera)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: EMBEDDING-LEVEL (EventGPT → VideoLLaVA-7B)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EventGPT Encoder (sparse, ~2ms)                                            │
│       │                                                                     │
│       ▼                                                                     │
│  Lightweight Adapter (~1ms)        ← Trained, 86.8% acceptance              │
│       │                                                                     │
│       ▼                                                                     │
│  VideoLLaVA-7B Encoder (verify)                                             │
│       │                                                                     │
│  ├─ Accept (cos_sim > 0.8): Use aligned features, skip dense encoding       │
│  └─ Reject (cos_sim < 0.5): Fallback to full VideoLLaVA-7B encoding         │
│                                                                             │
│  [SPEEDUP: 5-8x on vision encoding]                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    │ Vision Features (cached)
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: TOKEN-LEVEL (VideoLLaVA-7B → Heavy-13B)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VideoLLaVA-7B LM (draft model)                                             │
│       │                                                                     │
│       │ Generate k draft tokens                                             │
│       ▼                                                                     │
│  Heavy-13B LM (target model)      ← Parallel verification                   │
│       │                                                                     │
│  ├─ Accept matching tokens (expected α: 40-75%)                             │
│  └─ Reject → resample from 13B distribution                                 │
│                                                                             │
│  [SPEEDUP: 1.5-2.5x on language generation]                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
OUTPUT TEXT (High-quality 13B generation)
```

### Critical Finding: Video-LLaVA 13B Does NOT Exist

**Available 13B Models (as of January 2026):**

| Model | Parameters | Type | Availability |
|-------|------------|------|--------------|
| **LLaVA-1.5-13B** | 13B | Image-only | ✅ Production |
| **LLaVA-NeXT-13B** | 13B | Image-only | ✅ Production |
| **Video-LLaVA-13B** | 13B | Video | ❌ Does NOT exist |
| **Qwen2-VL-72B** | 72B | Video | ✅ (but very heavy) |

**Implications:**
- Stage 2 target must be LLaVA-13B (image) or custom Video-LLaVA-13B
- Building Video-LLaVA-13B would require training from LLaMA-13B base
- Alternative: Use Qwen2-VL or other video-capable 13B+ models

### Token Acceptance Rates: 7B → 13B

| Configuration | Acceptance (α) | Source | Notes |
|--------------|----------------|--------|-------|
| **Same-family 7B → 13B** (LLaMA) | 70-75% | ICLR 2025 | Same tokenizer, similar training |
| **Cross-model 7B → 13B** (different arch) | 40-50% | SpecVLM 2025 | Distribution mismatch |
| **LLaVA-7B → LLaVA-13B** | 65-70% | Spec-LLaVA | Same family, vision component |
| **VideoLLaVA-7B → LLaVA-13B** | ~50-60% | Estimated | Cross-modal gap |

### Expected Speedup Analysis

**Theoretical Formula:**
```
Speedup = E[accepted_tokens] / (c × γ + 1)

Where:
- E[accepted] = (1 - α^(γ+1)) / (1 - α)
- α = token acceptance rate
- γ = draft tokens per cycle (typically 5)
- c = cost ratio (draft_time / verify_time ≈ 0.3 for 7B/13B)
```

**Projected Performance:**

| Scenario | Stage 1 (Vision) | Stage 2 (Language) | Overall |
|----------|------------------|-------------------|---------|
| Conservative (α=50%) | 5x | 1.5x | **1.8-2.0x** |
| Moderate (α=65%) | 6x | 2.0x | **2.2-2.5x** |
| Optimistic (α=75%) | 8x | 2.5x | **2.8-3.0x** |

**Comparison to Literature:**

| Method | Speedup | Notes |
|--------|---------|-------|
| EAGLE-3 (SOTA, 2025) | 5.58x | Single model, feature-level |
| 3-Model Cascade | 3.3x | 0.5B → 3B → 7B → 13B |
| **Proposed 7B → 13B** | 2.0-2.5x | With alignment |
| Single Draft 7B → 13B | 2.39x | No cascade overhead |

### Memory Requirements

**Simultaneous Loading (Required for Speculative):**
```
EventGPT-7B:      ~14 GB (FP16)
VideoLLaVA-7B:    ~14 GB (FP16)
Heavy-13B:        ~26 GB (FP16)
KV Cache (8K):    ~12 GB combined
─────────────────────────────────
Total Peak:       ~50-60 GB
```

**Hardware Options:**

| Setup | VRAM | Feasibility | Latency Overhead |
|-------|------|-------------|------------------|
| A100-80GB / H100 | 80GB | ✅ Ideal | 0% |
| 2× RTX 4090 (24GB each) | 48GB | ⚠️ Tight | +5-10% |
| Sequential loading | 26GB | ✅ Works | +10-15% |
| INT8 quantization | ~30GB | ✅ Works | Minimal quality loss |

### Benefits of Two-Stage Cascade

**Compounding Effect:**
```
Direct EventGPT → 13B:
- Very low acceptance (~5%) due to large distribution gap
- NO speedup benefit

With 7B intermediate:
- Stage 1: EventGPT → 7B (86.8% embedding acceptance)
- Stage 2: 7B → 13B (50-75% token acceptance)
- Cascade bridges the distribution gap
```

**Why This Works:**
1. EventGPT's sparse features align well with 7B (trained adapter)
2. 7B and 13B share similar LLaMA architecture → high token acceptance
3. 7B acts as "distribution translator" between sparse events and dense 13B

### Cascade Overhead Analysis

| Source | Overhead |
|--------|----------|
| Coordination between stages | 5-10% |
| Memory bandwidth (multi-model) | 5-10% |
| Fallback to full computation | ~13% of samples |
| **Total Overhead** | **15-25%** |

**Net Benefit:**
```
Gross speedup (ideal):     3.0x
Overhead penalty:          -0.5x
Net speedup (realistic):   2.0-2.5x
```

### Implementation Roadmap

**Phase 1: Embedding-Level Integration (Weeks 1-2)**
- ✅ Alignment adapter trained (86.8% acceptance)
- ⬜ Integrate into VideoLLaVA-7B inference pipeline
- ⬜ Implement acceptance threshold logic
- ⬜ Add fallback mechanism

**Phase 2: 7B → 13B Token Alignment (Weeks 3-4)**
- ⬜ Set up LLaVA-13B as target model
- ⬜ Measure baseline token acceptance (VideoLLaVA-7B → LLaVA-13B)
- ⬜ Train token alignment if acceptance < 50%
- ⬜ Implement draft-verify loop

**Phase 3: Cascade Integration (Weeks 5-6)**
- ⬜ Connect Stage 1 output to Stage 2 input
- ⬜ Optimize memory management (multi-GPU or sequential)
- ⬜ Profile end-to-end latency
- ⬜ Benchmark on video QA tasks

**Phase 4: Optimization (Weeks 7-8)**
- ⬜ Tree attention for multi-token verification
- ⬜ Dynamic draft length based on acceptance history
- ⬜ Quantization for memory reduction
- ⬜ Production hardening

### Feasibility Verdict

| Criterion | Assessment |
|-----------|------------|
| **Technical Feasibility** | ✅ HIGH - Architecture is sound |
| **13B Model Availability** | ⚠️ MEDIUM - Video-LLaVA-13B doesn't exist, use LLaVA-13B |
| **Expected Speedup** | ✅ 2.0-2.5x realistic |
| **Memory Requirements** | ⚠️ HIGH - Need 50GB+ or multi-GPU |
| **Implementation Effort** | ⚠️ MEDIUM - 6-8 weeks |
| **Cascade Benefit vs Single-Stage** | ⚠️ MARGINAL - +10-20% over direct 7B→13B |

### Recommendation

**PROCEED with caveats:**
1. Use LLaVA-13B as target (Video-LLaVA-13B unavailable)
2. Start with embedding-level only (proven 86.8% acceptance)
3. Measure 7B → 13B token acceptance before full integration
4. Consider EAGLE-3 as alternative (5.58x speedup, simpler)

---

## Research Analysis: Challenges, Novelties, and Solutions

### Key Challenges

#### Challenge 1: Cross-Modal Alignment (Event → Video Features)

| Issue | Description | Severity |
|-------|-------------|----------|
| **Architectural Incompatibility** | Event cameras produce asynchronous sparse streams vs synchronous dense RGB frames | HIGH |
| **Distribution Mismatch** | Event features through CLIP (designed for RGB) don't align with video features | HIGH |
| **Information Loss** | Sparse-to-dense conversion via image rendering loses temporal information | MEDIUM |
| **Temporal Mismatch** | Events have sub-ms latency; RGB frames have 30-60ms latency | MEDIUM |

#### Challenge 2: Token Distribution Mismatch Across Stages

**Cascade Degradation Effect:**
```
Stage 1 (Event → Light VLM):  α₁ ≈ 0.65
Stage 2 (Light → Heavy VLM):  α₂ ≈ 0.60
Combined acceptance:          α_total = 0.65 × 0.60 = 0.39 (multiplicative!)
```

**Impact on Speedup:**
- For γ=4 draft tokens: expected_accepted = (1-α^(γ+1))/(1-α)
- If α drops 0.8 → 0.5 → 0.3: speedup drops from ~4x → ~2x

#### Challenge 3: Error Propagation in Autoregressive Generation

```
Draft token sequence: [t₁, t₂, t₃, t₄]
                        ↓
If t₁ error from misaligned features → compounds to t₂, t₃, t₄
                        ↓
Position-wise acceptance degrades: 0.81 → 0.74 → 0.68 → 0.62
```

**Root Cause:** Feature errors are continuous (not discrete like text) and compound through transformer layers.

#### Challenge 4: Memory Management for Multi-Model Inference

| Model Configuration | Peak VRAM | Challenge |
|--------------------|-----------|-----------|
| EventGPT-7B + Light-7B + Heavy-72B | ~160 GB | Exceeds single GPU |
| Sequential loading | ~72 GB | GPU idle during swap |
| KV cache for 8 video frames | +12-16 GB | Visual tokens dominate |

#### Challenge 5: Training Data Scarcity

| Data Type | Required | Available | Gap |
|-----------|----------|-----------|-----|
| Paired (event, video, text) triplets | 50K+ | ~15K (DSEC) | 35K |
| Target model acceptance pairs | 10K+ | ~200 | 9.8K |
| Cross-domain scenarios | Multiple | Driving only | HIGH |

---

### Potential Novelties for Publication (Updated with 2025 Literature)

#### State-of-the-Art Landscape (2025)

| Method | Venue | Speedup | Key Innovation |
|--------|-------|---------|----------------|
| **EAGLE-3** | NeurIPS 2025 | 5.58x | Multi-layer feature fusion + training-time test |
| **Spec-LLaVA** | 2025 | 3.28x | Tree-based draft with shared vision encoder |
| **SpecVLM** | 2025 | 2.9x | Elastic visual compressor + online distillation |
| **MASSV** | 2025 | 2.5x | Self-data distillation for multimodal drafters |
| **AdaSPEC** | NeurIPS 2025 | +15% α | Selective token filtering in distillation |
| **FastVLM** | CVPR 2025 | 3.2x TTFT | Hybrid vision encoder for high-res |
| **MSD** | 2025 | +0.65 τ | Visual-textual token decoupling |

#### What Makes This Approach Novel

| Aspect | Prior Work | Our Approach |
|--------|------------|--------------|
| **Modality** | Frame-based VLMs (MASSV, SpecVLM) | Event cameras (sparse, async) |
| **Cascade** | Same architecture (EAGLE) | Cross-modal + cross-scale |
| **Sparsity** | Not leveraged | Event sparsity → fewer visual tokens |
| **Temporal** | Fixed frame rate | Sub-millisecond event timing |

#### Unique Technical Contributions

**1. Event-Aware Token Compression**
- Events are sparse → visual tokens clustered in high-activity regions
- Compress using event density-based attention masks
- **Expected:** 20-30% KV cache reduction vs frame-based VLMs

**2. Asynchronous Feature Alignment**
- Align at multiple temporal resolutions (events @ 50Hz, frames @ 30Hz)
- Temporal interpolation loss for consistency
- **Expected:** Better alignment by respecting event temporal structure

**3. Cascade-Aware Distillation**
- Per-stage acceptance rate weighting in loss function
- `Loss = CE + λ(stage) × distillation_loss` where λ ∝ (1 - acceptance_rate)
- **Expected:** Reduce cascade error from 0.39 → 0.50+ combined acceptance

**4. Adaptive Draft Length Based on Event Sparsity**
- Sparse regions → shorter drafts (γ=2)
- Dense regions → longer drafts (γ=6)
- Learn optimal γ from hidden states
- **Expected:** Higher speedup in sparse scenes

#### Additional Novelties from 2025 Literature

**5. Visual-Textual Token Decoupling (from MSD)**
- Separate handling of visual and textual tokens during speculative decoding
- Visual tokens have different generation characteristics than text
- Decoupling improves average acceptance length by +0.37
- **Adaptation for Events:** Decouple event tokens (sparse, temporal) from language tokens

**6. Selective Token Filtering in Distillation (from AdaSPEC)**
- Standard KL divergence distillation is misaligned with acceptance rate objective
- Filter out "difficult-to-fit" tokens during distillation
- Focus training on tokens the draft model can realistically match
- **Expected:** +15% acceptance rate improvement over DistillSpec
- **Adaptation:** Filter tokens where event→video alignment is weakest

**7. Shared Vision Encoder Strategy (from Spec-LLaVA)**
- Draft and target models share the same CLIP vision encoder
- Eliminates redundant image encoding computations
- Training draft with identical data/methodology minimizes KL divergence
- **Adaptation:** Share EventGPT's event encoder across cascade stages

**8. Tree-Based Draft with Traversal Verification (2025)**
- Construct token tree covering multiple continuation paths
- Traversal Verification: leaf-to-root traversal preserves valid subsequences
- Tree attention enables single-pass verification of entire draft tree
- **Adaptation:** Build event-aware draft trees based on scene activity

**9. Elastic Visual Compressor (from SpecVLM)**
- Adaptively select compression primitives: pruning, pooling, convolution, resampler
- Online logit distillation without offline corpus
- Training-time scaling: longer training → higher acceptance length
- **Adaptation:** Select compression based on event density per region

**10. MASSV-Style Drafter Construction**
- Graft target VLM's vision encoder onto smaller LM via trainable projector
- Self-distilled visual instruction tuning using target's own responses
- Connect target's multimodal projector to draft model
- **Adaptation:** Graft Video-LLaVA encoder onto EventGPT-Light

**11. CoreMatching Sparse Inference (ICML 2025)**
- Identify "Core Neurons" (most frequently activated in FFN)
- Select "Core Tokens" with largest intersection with core neurons
- Only Core Tokens pass to subsequent layers
- **Adaptation:** Event sparsity naturally identifies core active regions

**12. Hardware-Aware Heterogeneous Decoding (DuoDecoding)**
- Deploy draft model on CPU, target on GPU for concurrent execution
- Dynamic multi-sequence drafting based on output uncertainty
- Hardware-aware optimal draft budget to minimize idle time
- **Adaptation:** EventGPT on edge device, heavy VLM on cloud

**13. Hierarchical Speculative Decoding (HSD)**
- PyramidSD: 1B → 3B → 8B cascade achieves 1.91× speedup
- Partition tasks via clustering, use heterogeneous drafts per cluster
- Data-driven task allocation
- **Adaptation:** 0.5B → 7B → 72B cascade with event-aware routing

**14. FastVLM Hybrid Vision Encoder (CVPR 2025)**
- 3.2× improvement in time-to-first-token (TTFT)
- 85× faster TTFT than LLaVA-OneVision at high resolution
- Vision encoder 3.4× smaller
- **Adaptation:** Design event-specific lightweight encoder

#### Comparison to State-of-the-Art

| Method | Speedup | Applicability to Events |
|--------|---------|------------------------|
| EAGLE-3 | 5.58x | ✓ Apply to LLM stage only |
| MASSV | 2.5x | ⚠️ Frame-based, needs adaptation |
| SpecVLM | 2.8x | ✓ Visual compression applicable |
| DREAM | 2.3x | ✓ Cross-attention for event-video |
| **Proposed** | 2.5-3.5x | Native event support + cascade |

---

### Proposed Solutions

#### Solution 1: Reconstruction-Aware Event Alignment

```python
# Pretrain event encoder on frame reconstruction
event_tensor → Event_Encoder → reconstructed_frame
loss = L2(reconstructed_frame, original_frame)

# Then freeze and use for alignment
pretrained_event_features → Alignment_Adapter → video_space
```

**Benefit:** Event features learn RGB-like semantics before alignment
**Expected:** Cosine similarity 0.65 → 0.80+

#### Solution 2: Layer-Wise Supervision (FastEagle approach)

```python
for i, (light_layer, heavy_layer) in enumerate(zip(light.layers, heavy.layers)):
    light_hidden = light_layer(x)
    heavy_hidden = heavy_layer(x)  # frozen

    # Earlier layers weighted higher
    layer_loss = MSE(light_hidden, heavy_hidden) * (1.0 - i/num_layers)
    total_loss += layer_loss
```

**Benefit:** Each layer corrects errors before passing to next
**Expected:** Position-wise degradation 0.81→0.74 improved to 0.81→0.79

#### Solution 3: Visual Token Pruning for Draft Stage

```python
# Prune low-importance visual tokens based on event density
event_density = compute_event_density(event_stream)
importance_scores = attention_scores * event_density
top_k_indices = torch.topk(importance_scores, k=128)  # Keep 128 of 257

pruned_visual_tokens = visual_tokens[top_k_indices]
```

**Benefit:** 50% reduction in draft stage KV cache
**Expected:** Memory 160GB → 120GB for full cascade

#### Solution 4: Synthetic Data Distillation

```python
# Generate training pairs using target model itself
for event_sample in unlabeled_events:
    # Heavy model generates reference
    with torch.no_grad():
        heavy_output = heavy_vlm(event_to_video(event_sample))

    # Train light model to match
    light_output = light_vlm(event_sample)
    loss = KL_div(light_output, heavy_output)
```

**Benefit:** 80-93% of real paired data performance
**Expected:** Reduce data requirement from 50K → 15K pairs

#### Solution 5: Adaptive Batch Pipelining

```
Time →
GPU:  [EventGPT B0] [EventGPT B1] [EventGPT B2] ...
           ↓
GPU:       [Light B0] [Light B1] [Light B2] ...
                ↓
GPU:            [Heavy B0] [Heavy B1] [Heavy B2] ...
```

**Benefit:** Hide ~70% of stage latency through overlap
**Expected:** Throughput improvement 1.5-2x

---

---

## Experimental Baselines and Evaluation Settings

### Baseline Categories

We organize baselines into four categories to comprehensively evaluate the proposed cascaded speculative decoding:

#### Category 1: Autoregressive Baselines (No Acceleration)

| Baseline | Description | Why Include | Settings |
|----------|-------------|-------------|----------|
| **EventGPT-7B (AR)** | Standard autoregressive EventGPT | Lower bound for event-based speed | `max_new_tokens=256`, greedy |
| **Video-LLaVA-7B (AR)** | Standard autoregressive Video-LLaVA | Frame-based VLM baseline | `max_new_tokens=256`, greedy |
| **LLaVA-OneVision-7B (AR)** | Latest LLaVA family | Strong frame-based baseline | `max_new_tokens=256`, greedy |
| **LLaVA-OneVision-72B (AR)** | Heavy target model | Quality upper bound | `max_new_tokens=256`, greedy |

**Rationale:** Establishes baseline latency and quality without any acceleration. EventGPT-7B shows natural event-based speedup; 72B shows quality ceiling.

#### Category 2: Speculative Decoding Baselines (Single-Stage)

| Baseline | Description | Why Include | Settings |
|----------|-------------|-------------|----------|
| **EAGLE-2** | Feature-level draft head on LLaVA | SOTA single-model speculative | Draft head 2 layers, γ=6 |
| **EAGLE-3** | Multi-layer fusion + training-time test | Latest SOTA (5.58x on LLMs) | Multi-layer fusion, γ=6 |
| **Medusa** | Parallel token prediction heads | Popular alternative to EAGLE | 4 Medusa heads |
| **Spec-LLaVA** | Tree-based VLM speculative | VLM-specific SOTA (3.28x) | 160M draft, tree depth=6 |
| **SpecVLM** | Elastic visual compressor | Visual token optimization | Online distillation |

**Rationale:** Shows how proposed method compares to existing speculative decoding techniques. EAGLE-3 and Spec-LLaVA are primary competitors.

**Settings for Fair Comparison:**
```python
# Common settings across all speculative baselines
max_new_tokens = 256
gamma = 6  # draft tokens per step (except tree-based)
temperature = 0.0  # greedy for reproducibility
batch_size = 1  # single sample for latency measurement
num_samples = 500  # statistical significance
```

#### Category 3: Visual Token Efficiency Baselines

| Baseline | Description | Why Include | Settings |
|----------|-------------|-------------|----------|
| **SparseVLM** | Text-guided token pruning | 54% FLOPs reduction, 97% acc | Adaptive ratio per layer |
| **FastVLM** | Hybrid vision encoder | 3.2x TTFT improvement | FastViTEncoder |
| **LLaVA-PruMerge** | Adaptive token reduction | Popular pruning method | Top-k=128 tokens |
| **VisPruner** | 94.4% visual token pruning | Extreme pruning capability | Threshold=0.1 |

**Rationale:** Our event-aware token compression should outperform frame-based pruning by leveraging event sparsity. These baselines test visual efficiency claims.

#### Category 4: Cascaded/Multi-Stage Baselines

| Baseline | Description | Why Include | Settings |
|----------|-------------|-------------|----------|
| **Direct 7B→72B** | Single-stage speculative (no cascade) | Ablation: is cascade beneficial? | γ=6 |
| **PyramidSD (1B→3B→8B)** | HSD cascade from literature | Existing cascade approach | 3-stage |
| **SLED** | Edge-cloud speculative | Distributed baseline | Edge draft, cloud verify |
| **Sequential Loading** | Load 7B, unload, load 72B | Memory-constrained baseline | No parallel loading |

**Rationale:** Validates the cascaded architecture provides benefit over single-stage approaches.

---

### Evaluation Metrics

#### Efficiency Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Tokens/Second (TPS)** | `generated_tokens / total_time` | Primary throughput metric |
| **Speedup** | `TPS_method / TPS_autoregressive` | Relative improvement |
| **Time-to-First-Token (TTFT)** | `prefill_time` | Critical for interactive applications |
| **Token Acceptance Rate (α)** | `accepted_tokens / draft_tokens` | Speculative decoding efficiency |
| **Average Acceptance Length (τ)** | `E[consecutive_accepted]` | Expected tokens accepted per step |
| **Memory Peak (GB)** | `max(GPU_memory)` | Hardware feasibility |

#### Quality Metrics (Accuracy Preservation)

| Benchmark | Type | Metric | Target Models |
|-----------|------|--------|---------------|
| **DSEC-QA** | Video QA (Driving) | Accuracy | Primary eval (in-domain) |
| **VideoMME** | Video QA (General) | Accuracy | Zero-shot generalization |
| **MVBench** | Video Understanding | Accuracy | Temporal reasoning |
| **MSVD-QA** | Video Captioning | CIDEr, BLEU-4 | Generation quality |
| **ActivityNet-QA** | Long Video QA | Accuracy | Long-form video |

**Quality Preservation Target:** <1% accuracy drop vs autoregressive baseline (lossless decoding guarantee).

---

### Experimental Settings

#### Hardware Configuration

| Setup | GPU | VRAM | Use Case |
|-------|-----|------|----------|
| **Single GPU** | RTX 4090 | 24GB | 7B models, sequential loading |
| **Dual GPU** | 2× RTX 4090 | 48GB | 7B + 34B parallel |
| **High-Memory** | A100-80GB | 80GB | Full cascade parallel |
| **Multi-GPU** | 2× H100 | 160GB | 72B target model |

#### Model Configurations

```python
# EventGPT Configuration
eventgpt_config = {
    "model": "EventGPT-7B",
    "vision_encoder": "CLIP-ViT-L/14",
    "event_representation": "event_image",  # or "voxel_grid"
    "fps": 1,  # effective frame rate from events
}

# Light VLM (Draft) Configuration
light_vlm_config = {
    "model": "LLaVA-OneVision-0.5B",  # or 7B
    "vision_encoder": "SigLIP",
    "shared_with_target": True,  # share vision encoder
}

# Heavy VLM (Target) Configuration
heavy_vlm_config = {
    "model": "LLaVA-OneVision-72B",  # or 34B
    "precision": "bfloat16",
    "context_length": 32768,
}

# Alignment Adapter Configuration
adapter_config = {
    "type": "lightweight",  # or "contrastive"
    "hidden_dim": 512,
    "num_layers": 3,
    "acceptance_threshold": 0.8,
}
```

#### Speculative Decoding Settings

```python
speculative_config = {
    "gamma": 6,              # draft tokens per step
    "tree_depth": 4,         # for tree-based methods
    "tree_width": 3,         # branches per node
    "acceptance_threshold": 0.8,  # for visual features
    "max_new_tokens": 256,
    "temperature": 0.0,      # greedy (deterministic)
}
```

---

### Dataset Configuration

#### Primary Dataset: DSEC (Driving Scenarios)

| Split | Samples | Duration | Use |
|-------|---------|----------|-----|
| **Train** | 5,208 | 1s clips | Alignment training |
| **Val** | 1,302 | 1s clips | Hyperparameter tuning |
| **Test** | 1,000 | 1s clips | Final evaluation |
| **Long-form** | 596 | 8s clips | Temporal generalization |

#### Secondary Datasets (Zero-Shot)

| Dataset | Samples | Domain | Purpose |
|---------|---------|--------|---------|
| **N-Caltech101** | 8,246 | Object recognition | Event camera generalization |
| **N-ImageNet** | ~1.7M | General objects | Large-scale event eval |
| **MVSEC** | 4 sequences | Indoor/outdoor | Cross-domain events |

---

### Ablation Studies

| Ablation | Question | Comparison |
|----------|----------|------------|
| **Cascade Stages** | Is 3-stage better than 2-stage? | 0.5B→72B vs 0.5B→7B→72B |
| **Alignment Method** | Which adapter works best? | Lightweight vs Contrastive vs MASSV |
| **Draft Tokens (γ)** | Optimal draft length? | γ ∈ {2, 4, 6, 8} |
| **Acceptance Threshold** | Optimal cosine similarity cutoff? | θ ∈ {0.5, 0.7, 0.8, 0.9} |
| **Visual Token Pruning** | How much pruning before quality drops? | k ∈ {64, 128, 192, 256} |
| **Distillation Method** | Which improves acceptance most? | DistillSpec vs AdaSPEC vs SpecKD |

---

### Expected Results Summary

| Method | Speedup | α (acceptance) | Memory | Quality |
|--------|---------|----------------|--------|---------|
| EventGPT-7B (AR) | 1.0x | - | 14GB | Baseline |
| LLaVA-72B (AR) | 0.3x | - | 144GB | Upper bound |
| EAGLE-3 (7B) | 3.5x | 0.72 | 16GB | Lossless |
| Spec-LLaVA (7B) | 3.28x | 0.68 | 15GB | Lossless |
| Direct 7B→72B | 2.0x | 0.55 | 160GB | Lossless |
| **Ours (0.5B→7B→72B)** | **2.5-3.5x** | **0.45-0.55** | **160GB** | **Lossless** |
| **Ours (7B→34B)** | **2.0-2.5x** | **0.50-0.60** | **82GB** | **Lossless** |

**Key Comparisons:**
1. vs EAGLE-3: We target 72B quality (higher) with comparable speedup
2. vs Spec-LLaVA: We handle events (novel modality) with cascade
3. vs Direct 7B→72B: Cascade improves acceptance via intermediate stage

---

### Paper Structure Recommendation

**Title:** "EventSpec: Cascaded Speculative Decoding for Event-Camera Vision-Language Models"

**Key Claims:**
1. First application of speculative decoding to event cameras
2. Novel three-stage cascade (sparse → dense → language)
3. Event sparsity enables 20-30% KV cache reduction
4. Achieve 2.5-3.5x speedup with <5% accuracy loss

**Sections:**
1. Introduction: Event cameras + speculative decoding gap
2. Related Work: Speculative decoding, VLMs, event vision
3. Method:
   - 3.1: Event-to-Video Alignment
   - 3.2: Cross-Scale Token Distribution Alignment
   - 3.3: Cascade Error Mitigation
   - 3.4: System Optimizations
4. Experiments:
   - Alignment quality (cosine sim, acceptance rate)
   - End-to-end speedup vs baselines
   - Ablation studies
5. Analysis & Conclusion

---

### References

#### Speculative Decoding - Core Methods
- [EAGLE-3: Scaling up Inference Acceleration (NeurIPS 2025)](https://arxiv.org/abs/2503.01840) - Multi-layer feature fusion + training-time test
- [Cascade Speculative Drafting (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/9cb5b083ba4f5ca6bd05dd307a2fb354-Paper-Conference.pdf)
- [Faster Cascades via Speculative Decoding (ICLR 2025)](https://openreview.net/forum?id=vo9t20wsmd)
- [FastEagle: Cascaded Drafting (2025)](https://arxiv.org/html/2509.20416)

#### Speculative Decoding - VLM-Specific
- [Spec-LLaVA: Tree-Based Speculative Decoding (2025)](https://arxiv.org/abs/2509.11961) - Shared vision encoder, 3.28x speedup
- [SpecVLM: Fast Speculative Decoding in VLMs (2025)](https://arxiv.org/abs/2509.11815) - Elastic visual compressor
- [MASSV: Multimodal Adaptation and Self-Data Distillation (2025)](https://arxiv.org/html/2505.10526v1) - Drafter construction via projector
- [MSD: Multimodal Speculative Decoding (2025)](https://arxiv.org/html/2505.14260v1) - Visual-textual token decoupling
- [ViSpec: Vision-Aware Speculative Decoding (2025)](https://arxiv.org/html/2509.15235v5) - Dual integration mechanisms

#### Distillation & Acceptance Rate
- [AdaSPEC: Selective Knowledge Distillation (NeurIPS 2025)](https://arxiv.org/abs/2510.19779) - +15% acceptance via selective filtering
- [DistillSpec: Knowledge Distillation for SD (2023)](https://arxiv.org/abs/2310.08461) - On-policy distillation baseline
- [SpecKD: Speculative Knowledge Distillation (2025)](https://arxiv.org/html/2510.24021v1) - Token acceptance rate guarantee
- [Training Domain Draft Models (2025)](https://arxiv.org/abs/2503.07807) - Domain-specific draft training

#### Tree-Based & Parallel Verification
- [Traversal Verification for Speculative Tree Decoding (2025)](https://arxiv.org/abs/2505.12398) - Leaf-to-root verification
- [OPT-Tree: Adaptive Draft Tree Structure (TACL 2025)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00735/128189/) - Dynamic tree construction
- [Talon: Confidence-Aware Adaptive Token Trees (2025)](https://arxiv.org/html/2601.07353) - Budget-driven framework

#### Heterogeneous & Edge-Cloud
- [DuoDecoding: Hardware-aware Heterogeneous SD (2025)](https://arxiv.org/html/2503.00784) - CPU draft + GPU target
- [SpecRouter: Multi-Level Speculative Routing (2025)](https://arxiv.org/pdf/2505.07680) - Adaptive model selection
- [SLED: Speculative Decoding for Edge (2025)](https://arxiv.org/html/2506.09397v5) - Edge-server collaboration
- [LiteVLM: Low-Latency VLM Pipeline (2025)](https://arxiv.org/html/2506.07416v1) - Patch selection + token compression
- [Distributed VLMs: Cloud-Edge Collaboration (2025)](https://wimnet.ee.columbia.edu/wp-content/uploads/2025/04/DistributedVLMs_Efficient_Vision-Language_Processing_through_Cloud-Edge_Collaboration.pdf)

#### Visual Token Efficiency
- [SparseVLM: Visual Token Sparsification (ICML 2025)](https://arxiv.org/abs/2410.04417) - 54% FLOPs reduction, 97% accuracy
- [TopV: Compatible Token Pruning (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_TopV_Compatible_Token_Pruning_with_Inference_Time_Optimization_for_Fast_CVPR_2025_paper.pdf)
- [VisPruner: 94.4% Visual Token Pruning (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.pdf)
- [PruneVid: Video Token Pruning (ACL 2025)](https://aclanthology.org/2025.findings-acl.1024.pdf)
- [FastVLM: Efficient Vision Encoding (CVPR 2025)](https://arxiv.org/abs/2412.13303) - 3.2x TTFT, 85x faster than LLaVA-OneVision
- [CoreMatching: Co-adaptive Sparse Inference (ICML 2025)](https://github.com/wangqinsi1/2025-ICML-CoreMatching)

#### Event Camera & Neuromorphic
- [EventGPT: Event Stream Understanding (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_EventGPT_Event_Stream_Understanding_with_Multimodal_Large_Language_Models_CVPR_2025_paper.pdf)
- [High-efficiency Sparse Convolution for Event Cameras (2025)](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1537673/full)
- [Event2Vec: Processing Events in Vector Space (2025)](https://arxiv.org/abs/2504.15371)
- [SEED: Sparse Event-based Efficient Detector (2025)](https://arxiv.org/html/2506.13440) - 92% activation sparsity

#### Other Relevant Works
- [OmniDraft: Cross-vocabulary Speculative Decoding (2025)](https://arxiv.org/html/2507.02659)
- [DREAM: Drafting with Refined Target Features (2025)](https://arxiv.org/html/2505.19201v1)
- [Cross-Attention Speculative Decoding (Beagle, 2025)](https://www.arxiv.org/pdf/2505.24544)

---

**Last Updated:** January 2026
