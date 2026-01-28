# SpecVLM: Fast Speculative Decoding in Vision-Language Models

**Paper**: arXiv:2509.11815v2
**Authors**: Haiduo Huang et al. (AMD + Xi'an Jiaotong University)
**PDF**: `pdf/SpecVLM_vision_language_speculative_decoding.pdf`

---

## Core Problem

VLM speculative decoding faces unique challenges vs. text-only LLMs:

| Challenge | Description |
|-----------|-------------|
| **Prefill bottleneck** | Visual tokens scale with image resolution/video length |
| **KV cache inflation** | More visual tokens → larger memory footprint |
| **Dual bottleneck** | LLM prefill + autoregressive decoding both slow |

### Latency Breakdown (LLaVA-1.6-7B)
- Vision Encoder + Projector: ~45ms
- **LLM Prefill: ~261ms** (dominant bottleneck)
- LLM Decode: remaining

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TARGET VLM (frozen)                         │
├─────────────────────────────────────────────────────────────────┤
│  Input Image → Vision Encoder → Vision Projector                │
│                                      ↓                          │
│  Input Text  → Text Tokenizer  → Text Embedding                 │
│                                      ↓                          │
│              [V1, V2, ..., Vm, T1, T2, ..., Tn]                 │
│                                      ↓                          │
│                    Transformer Layers × N                       │
│                                      ↓                          │
│                         Head Layer → Logits                     │
│                                      ↓                          │
│                      Autoregressive Sampling                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     DRAFT MODEL (trained)                       │
├─────────────────────────────────────────────────────────────────┤
│  Shared: Vision Encoder, Vision Projector, Text Embedding       │
│                                      ↓                          │
│                    Vision Compressor (elastic)                  │
│                                      ↓                          │
│                    Dimension Reduction FC                       │
│                                      ↓                          │
│               Transformer Layer × 1 (single layer!)             │
│                                      ↓                          │
│              Head Layer (shared with target)                    │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: Only a single transformer decoder layer needed for draft model.

---

## Two Key Innovations

### 1. Elastic Visual Compressor (Section 3.4)

Four compression primitives with adaptive selection:

| Primitive | Parameters | Compression Ratio | Best Use Case |
|-----------|------------|-------------------|---------------|
| **Pruning** | None (parameter-free) | Up to 20-30× | Simple/text-based tasks |
| **Pooling** | None (parameter-free) | Up to 20-30× | Simple tasks, global features |
| **Convolution** | Few parameters | 3-5× max | Visual detail tasks |
| **Resampler** | Q-Former style | 2 queries optimal | Complex reasoning |

#### Three Selection Modes

**(a) Weighted Combination of Experts**
- MoE-inspired question-conditioned gate
- Fuses operators with same compression ratio but different abstraction
- Cap ratio at ≤5×, use 3× by default

**(b) Multi-Granularity Feature Concatenation**
- Combine heterogeneous compression scales
- Config: Pruning/Pooling at 20×, Convolution at 3×, Resampler with 2 queries
- Concatenate all outputs

**(c) Dynamic Selection of Compression Experts (DEFAULT)**
- Question-aware selector chooses ONE operator per input
- Based on gate's top-1 logit
- Includes **Text-only branch** (drops all vision tokens) for LLM zero-shot
- Uses Gumbel-Softmax during training, argmax at inference

#### Compression Ratio Analysis (Table 4)

| Branch | Ratio | Speedup (τ) | Accepted Length (σ) |
|--------|-------|-------------|---------------------|
| Pruning | 20× | **2.03** | **3.40** |
| Pooling | 20× | **2.03** | **3.40** |
| Convolution | 3× | **2.03** | **3.40** |
| Convolution | 10× | 1.91 | 3.29 (degraded) |
| Convolution | 30× | 1.67 | 2.73 (broken) |
| Resampler | 2 queries | **2.03** | **3.40** |

**Takeaway**:
- Pruning/pooling robust up to 10-20×
- Convolution degrades beyond 5×
- Resampler sweet spot is 2 queries

---

### 2. Online-Logit Distillation (Section 3.5)

**Problem with offline distillation**:
- Precomputes and stores large volumes of teacher outputs
- Storage-intensive and computationally expensive for VLMs

**Solution**: Distill on-the-fly during training

#### Loss Function
```
L_online = λ_logit × L_CE(z_q, z_p) + λ_feat × L_SmoothL1(f_q, f_p)

Where:
- z_q, z_p = draft and target logits
- f_q, f_p = draft and target penultimate-layer features
- λ_logit = 0.1 (default)
- λ_feat = 1.0 (default)
```

#### Training-Time Scaling Effect

| Epochs | LLaVA-1.5-7B τ | LLaVA-1.5-7B σ | LLaVA-1.6-13B τ | LLaVA-1.6-13B σ |
|--------|----------------|----------------|-----------------|-----------------|
| 1 | 2.20 | 3.68 | 2.38 | 3.87 |
| 3 | 2.47 | 4.84 | 2.55 | 4.97 |
| 5 | **2.66** | **5.27** | **2.82** | **5.34** |

**Key finding**: Longer online training monotonically increases acceptance rate.

---

## Speedup Analysis

### Theoretical Framework (Equation 2)

```
Speedup = T_AR / T_SD = (S/R) × 1/(γ × Tq(1)/Tp(1) + Tp(γ)/Tp(1) + Tsample/Tp(1))

Where:
- S = total tokens generated
- R = number of SD rounds
- σ = S/R = average accepted tokens per round
- γ = draft tokens per round
- Tq(1) = draft model single-token latency
- Tp(1) = target model single-token latency
- Tp(γ) = target model γ-token verification latency
```

**Key factors for speedup**:
1. **Maximize σ** (average accepted length) → better draft-target alignment
2. **Minimize Tq(1)/Tp(1)** (draft latency ratio) → smaller/faster draft
3. **Reduce KV cache** via visual compression → lower memory bandwidth

---

## Experimental Results

### Main Results (Table 1 - Temperature=0, Epoch=1)

| Model | Method | LLaVA-Wild | MMBench | Avg τ | Avg σ |
|-------|--------|------------|---------|-------|-------|
| LLaVA-1.5-7B | EagleVLM | 2.09× | 1.83× | 2.01× | 3.59 |
| LLaVA-1.5-7B | **SpecVLM** | **2.20×** | **1.93×** | **2.11×** | **3.67** |
| LLaVA-1.6-7B | EagleVLM | 1.91× | 2.04× | 1.88× | 3.53 |
| LLaVA-1.6-7B | **SpecVLM** | **2.03×** | **2.17×** | **2.03×** | **3.69** |
| LLaVA-1.5-13B | EagleVLM | 2.31× | 2.09× | 2.15× | 3.64 |
| LLaVA-1.5-13B | **SpecVLM** | **2.41×** | **2.29×** | **2.27×** | **3.71** |
| LLaVA-1.6-13B | EagleVLM | 2.29× | 2.54× | 2.26× | 4.09 |
| LLaVA-1.6-13B | **SpecVLM** | **2.38×** | **2.70×** | **2.39×** | **4.15** |

### After 5 Epochs Training
- **2.5-2.9× end-to-end speedup**
- σ increases from ~3.5 to ~5.3 tokens per round

### MMMU Benchmark (Table 2)
- Consistent gains across Art, Biology, Chemistry, Economics, Math, Physics
- Best on Math: up to 2.63× speedup

---

## Implementation Details

### What to Train (Figure 3b)
- **Frozen**: Target VLM (all components)
- **Trainable**:
  - Draft model's single decoder layer
  - Vision compressor (gate + optional resampler)
  - Dimension reduction FC

### Training Setup
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Batch size | 128 |
| Learning rate (7B) | 8e-5 |
| Learning rate (13B) | 5e-5 |
| Weight decay | 0 |
| Warmup | None |
| Training time (7B) | 15 hours on 8× MI250 |
| Training time (13B) | 20 hours on 8× MI250 |

### Inference Setup
| Parameter | Value |
|-----------|-------|
| Tree attention Top-K | 10 |
| Total draft tokens | 60 |
| Tree depth | 7 |
| Batch size | 1 (latency-focused) |

### Key Implementation Note
> "Compared with the original EAGLE draft model, **the only effective modification is the introduction of an input layer normalization**, which stabilizes training and prevents numerical overflow."

---

## Ablation Studies (Table 3)

| Configuration | Avg τ | Avg σ |
|---------------|-------|-------|
| EagleVLM (baseline) | 1.88× | 3.53 |
| + Weighted combination | 1.99× | 3.71 |
| + Multi-granularity concat | 1.96× | 3.67 |
| + Dynamic selection (SpecVLM) | **2.03×** | **3.69** |

**Finding**: Dynamic selection yields best overall performance.

---

## Application to EventGPT → VideoLLaVA

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              TARGET: VideoLLaVA-7B (frozen)                     │
├─────────────────────────────────────────────────────────────────┤
│  Video Frames → Video Encoder (dense) → Projector               │
│                                              ↓                  │
│  Text Query   → Text Tokenizer → Text Embedding                 │
│                                              ↓                  │
│                      LLM Layers × N                             │
│                                              ↓                  │
│                    Verification + Sampling                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              DRAFT: EventGPT-based (trainable)                  │
├─────────────────────────────────────────────────────────────────┤
│  Event Stream → Event Encoder (sparse) ← INHERENTLY FAST        │
│                           ↓                                     │
│            Elastic Compressor (10-20× pruning/pooling)          │
│                           ↓                                     │
│            Alignment MLP (event → video feature space)          │
│                           ↓                                     │
│               Single Transformer Layer                          │
│                           ↓                                     │
│            Head Layer (shared with VideoLLaVA)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Should Work

1. **Event features are naturally "compressed"**
   - Sparse temporal representation (only changes)
   - 10-100× fewer "pixels" than dense video frames

2. **Elastic compressor directly applicable**
   - Use 10-20× pooling/pruning on event tokens
   - Question-aware routing: text-only when events sufficient

3. **Lightweight training**
   - Single decoder layer + online distillation
   - Freeze VideoLLaVA, train only alignment + draft layer

4. **Modality-specific speedup**
   - Event encoder: ~2ms (sparse)
   - Video encoder: ~50ms (dense)
   - Additional 25× speedup on vision side

### Training Recipe (from paper)

```python
# Freeze target VLM
for param in videollava.parameters():
    param.requires_grad = False

# Trainable components
draft_decoder = nn.TransformerDecoderLayer(d_model=4096, nhead=32)
alignment_mlp = nn.Sequential(
    nn.Linear(event_dim, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096)
)
vision_compressor = ElasticCompressor(modes=['prune', 'pool', 'conv', 'resample'])

# Loss
loss = 0.1 * CE(draft_logits, target_logits) + 1.0 * SmoothL1(draft_features, target_features)

# Training: 1-5 epochs, ~15-35 hours on 8× GPUs
```

### Expected Gains

| Component | EagleVLM-style | SpecVLM-style | EventGPT Draft |
|-----------|----------------|---------------|----------------|
| Vision latency | 50ms | 30ms (compressed) | **5ms** (sparse) |
| Draft latency | 10ms | 8ms | **3ms** |
| Overall speedup | 2.0× | 2.5× | **3.0-3.5×** (estimated) |

---

## Key Takeaways

1. **Single decoder layer is sufficient** for draft model
2. **Visual compression is critical** - pruning/pooling at 10-20× for simple tasks
3. **Question-aware routing** enables task-adaptive compression
4. **Online distillation** eliminates need for offline corpus
5. **Training-time scaling** - more epochs = better acceptance rate
6. **Layer normalization** is the only essential modification to EAGLE-2

---

## Limitations (from paper)

1. Compression ratios still manually configured
2. No dynamic KV cache compression at inference
3. Hyperparameters not exhaustively tuned
4. Focus on latency, not throughput optimization

---

## References

- EAGLE-2: Li et al., 2024d - Base architecture
- BLIP-2: Li et al., 2023 - Q-Former resampler
- LLaVA: Liu et al., 2023, 2024 - Target VLM
- Gumbel-Softmax: Jang et al., 2016 - Differentiable selection

---

**Document Created**: January 28, 2026
**Source**: arXiv:2509.11815v2
