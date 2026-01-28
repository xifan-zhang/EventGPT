# HiSpec: Hierarchical Speculative Decoding for LLMs

**Paper**: arXiv:2510.01336v1
**Authors**: Avinash Kumar, Sujay Sanghavi, Poulami Das (UT Austin)
**PDF**: `pdf/HiSpec_hierarchical_speculative_decoding.pdf`

---

## Core Problem: The Verification Wall

**Key insight**: Verification is the bottleneck, not drafting.

```
Verification latency vs. Draft generation latency:

Target: OPT-66B      → Verification is 2× slower
Target: Llama-70B    → Verification is 4-6× slower
Target: Llama-405B   → Verification is 10.3× slower!

Verification accounts for 60-90% of total response time.
```

**Most prior works focus on accelerating drafting** (EAGLE, Medusa, Lookahead), but this yields limited returns because verification dominates runtime.

---

## HiSpec Solution: Intermediate Verification with Early-Exit Models

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SINGLE EARLY-EXIT MODEL                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1-4:    Draft Layer (L_d)         → Fast draft tokens   │
│                     ↓                                           │
│  Layer 5-8:    Intermediate Verifier (L_i) → Early rejection   │
│                     ↓                                           │
│  Layer 9-32:   Target/Full Model (L_f)   → Final verification  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Innovation

**Use early-exit (EE) models** for both drafting AND intermediate verification:
- EE models are trained so hidden states at selected layers can be interpreted
- No need for separate auxiliary models
- No training overhead for intermediate verifier
- KV caches and hidden states can be reused across layers

---

## How HiSpec Works (Figure 3)

### Standard Speculative Decoding
```
Draft: [A B C D] → Wait for target verification → [A B ✓ C ✗] → 2/4 accepted
                   (long wait)
Draft: [D E F G] → Wait for target verification → [D E F ✓ G ✗] → 3/4 accepted
```

### HiSpec with Intermediate Verification
```
Draft: [A B C D] → Intermediate verify → [A B ✓ C ✗] → 2/4 tentative
                   (fast, early rejection of C)
Draft: [D E F G] → Intermediate verify → [D E F G ✓] → 4/4 tentative
                   (faster draft starts sooner)
Target verify: [A B D E F G] → All 7 accepted (verified in batch)
```

**Benefits**:
1. Early rejection of bad tokens
2. Faster subsequent draft generation
3. KV cache reuse across all stages

---

## Key Design Decisions

### 1. Layer Positioning (Section 3.2)

**Critical finding**: Early layers (1/4 of model depth) generate up to 69% of tokens correctly.

| Layer Depth | Role | Accuracy |
|-------------|------|----------|
| 1/8 of model | Draft (L_d) | ~40-50% |
| 1/4 of model | Intermediate Verifier (L_i) | ~69% |
| Full model | Target (L_f) | 100% |

**Default HiSpec configuration**:
- Draft layer: ~1/8 of total layers
- Intermediate verifier: ~1/4 of total layers
- Target: full model

### 2. Dynamic KV Cache Management (Section 3.1)

```python
# Problem: Misaligned KV caches corrupt output

# Solution: Two-step approach
1. Buffer KV caches and hidden states during draft generation
2. Prune entries for rejected tokens after intermediate verification

# This differs from single-layer speculation where KV is not managed
# across arbitrary intermediate layers
```

### 3. Balancing Intermediate and Target Verification (Section 3.3)

**Trade-off**:
- Frequent target verification → compute-intensive, slow
- Infrequent target verification → risk of flushing long token sequences

**Solution**: Dynamic policy based on tentatively accepted token count

```
Default: Wait for N_i = 4 tokens tentatively accepted before target verification
```

**Improvement**: HiSpec increases acceptance rate from 39.7% to 58.1% (46% improvement) by using intermediate verifier.

---

## Algorithm

```python
def hispec_decode(L_d, L_i, L_f, prompt, N_d=2, N_i=4):
    """
    L_d: Draft layer
    L_i: Intermediate verifier layer
    L_f: Full/target layer
    N_d: Tokens proposed by draft per step
    N_i: Tokens verified by L_i before L_f verification
    """
    context = prompt
    V = []  # Buffer for tokens accepted by L_i but not committed

    while not end_of_sequence:
        # Intermediate verification loop
        while len(V) < N_i and not end_of_sequence:
            # Generate draft tokens
            S = L_d.generate(context + V, N_d)

            # Intermediate verification
            V_accepted = leading_substring_verify(S, L_i, context + V)
            V.extend(V_accepted)

        # Full-model verification (periodic)
        F = leading_substring_verify(V, L_f, context)
        context.extend(F)
        V = []  # Clear buffer

    return context


def leading_substring_verify(draft_tokens, verifier_layer, context):
    """Accept longest prefix where each token is in top predictions"""
    accepted = []
    for token in draft_tokens:
        if token in verifier_layer.top_predictions(context + accepted):
            accepted.append(token)
        else:
            # Emit one token from verifier and stop
            accepted.append(verifier_layer.generate_one(context + accepted))
            break
    return accepted
```

---

## Experimental Results

### Setup
- **Hardware**: 4× NVIDIA H100 GPUs (94GB HBM3)
- **Models**: Llama2-7B/13B/70B, Llama3-8B, CodeLlama-7B/34B
- **Benchmarks**: ShareGPT, CNN/DM, XSum, HumanEval, GSM8K

### Main Results (Table 1)

| Model | Method | ShareGPT | CNN/DM | HumanEval | GSM8K | XSum |
|-------|--------|----------|--------|-----------|-------|------|
| **Llama3-8B** | AdaDecode | 1.00× | 1.12× | 1.36× | 1.39× | 1.06× |
| | LayerSkip | 1.14× | 1.70× | 1.54× | 1.77× | 1.31× |
| | LookAhead | 1.20× | 1.44× | 1.19× | 1.20× | 1.37× |
| | **HiSpec** | **2.01×** | **2.08×** | **1.91×** | **1.93×** | **1.84×** |
| **Llama2-70B** | LayerSkip | 1.30× | 1.47× | 1.32× | 1.48× | 1.29× |
| | SWIFT | 1.10× | 1.17× | 1.12× | 1.09× | 1.10× |
| | **HiSpec** | **1.64×** | **1.93×** | **1.69×** | **1.72×** | **1.63×** |

**Key result**: HiSpec improves throughput by **1.28× on average** and up to **2.01×** compared to baseline single-layer speculation.

### Throughput vs Acceptance Rate (Figure 6)

For CodeLlama-34B (48 layers):
- **Baseline (LayerSkip)**: Optimal at Layer-7 draft
- **HiSpec**: Draft Layer-7, Intermediate Layer-12
  - Throughput: **1.25× improvement**
  - Acceptance rate: **1.15× improvement**

---

## Ablation Studies (Section 6)

### Impact of Draft Tokens Per Step (N_d)

| N_d | Throughput |
|-----|------------|
| 2 | **~60** (optimal) |
| 4 | ~52 |
| 6 | ~48 |
| 8 | ~45 |

**Finding**: Lower N_d yields higher throughput (limits unverified token chains).

### Impact of Tentative Acceptance Window (N_i)

| N_i | Throughput |
|-----|------------|
| 4 | **~60** (optimal) |
| 8 | ~55 |
| 12 | ~50 |
| 16 | ~48 |

**Finding**: Lower N_i yields higher throughput (reduces flush penalty).

### Default HiSpec Parameters
- N_d = 2 (draft tokens per step)
- N_i = 4 (tentative acceptance window)
- L_d = 1/8 of model depth
- L_i = 1/4 of model depth

---

## Comparison with Prior Works

| Method | Focus | Training | Memory | Accuracy |
|--------|-------|----------|--------|----------|
| EAGLE | Draft quality | Required | +Draft model | Same |
| Medusa | Multiple heads | Required | +Heads | Same |
| LayerSkip | Early-exit draft | None | Same | Same |
| SPRINTER | Intermediate verify | Required | +Verifier model | Degraded |
| **HiSpec** | Intermediate verify | **None** | **Same** | **Same** |

**HiSpec advantages**:
1. No training required
2. No additional memory for auxiliary models
3. Maintains target model accuracy
4. Works with any early-exit model (≥2 exit layers)

---

## Application to EventGPT → VideoLLaVA

### Analogy

```
HiSpec (within single model):
    Layer 1-8 (draft) → Layer 9-16 (intermediate) → Layer 17-32 (target)

Cross-Modal HiSpec:
    Event Encoder (draft) → Frame Encoder (intermediate) → VideoLLaVA (target)
```

### Key Insights for Cross-Modal Application

1. **Verification is the bottleneck**
   - Video encoding (dense) is slow → analogous to target verification
   - Event encoding (sparse) is fast → analogous to draft generation

2. **Early rejection saves compute**
   - Intermediate verification rejects bad drafts before expensive video encoding
   - Frame-based verification can filter event drafts cheaply

3. **KV cache reuse across modalities**
   - Event features can be extended to frame features
   - Frame features can be extended to video features
   - Avoid redundant encoding

### Proposed Cross-Modal HiSpec

```python
class CrossModalHiSpec:
    """
    L_d: Event encoder (1/8 compute)
    L_i: Frame encoder (1/4 compute)
    L_f: Video encoder (full compute)
    """

    def __init__(self):
        self.event_encoder = EventGPT()      # Draft
        self.frame_encoder = FrameEncoder()   # Intermediate verifier
        self.video_llava = VideoLLaVA()       # Target

    def forward(self, events, frames, video):
        # Stage 1: Event drafting (fast)
        event_features = self.event_encoder(events)
        draft_tokens = self.event_encoder.predict_tokens(event_features)

        # Stage 2: Frame intermediate verification
        frame_features = self.frame_encoder(frames)
        tentative_tokens = []
        for token in draft_tokens:
            if token in self.frame_encoder.top_predictions(frame_features):
                tentative_tokens.append(token)
            else:
                break  # Early rejection

        # Stage 3: Video target verification (expensive, but batched)
        if len(tentative_tokens) >= N_i:
            video_features = self.video_llava.encode(video)
            accepted = self.verify_against_target(tentative_tokens, video_features)
            return accepted

        # Fallback to video encoding
        return self.video_llava.full_inference(video)
```

### Expected Benefits

| Stage | Latency | Tokens Processed |
|-------|---------|------------------|
| Event draft | 2ms | Generate 4 tokens |
| Frame intermediate | 10ms | Verify 4, accept 3 |
| Video target | 50ms | Verify 3, accept 3 |
| **Total** | **62ms** | 3 tokens (vs 150ms baseline) |

---

## Key Takeaways

### For Research

1. **Verification is 2-10× slower than drafting** - focus on verification acceleration
2. **Early-exit models enable hierarchical verification** without training overhead
3. **1/4 of model depth generates ~69% correct tokens** - position intermediate verifier there
4. **KV cache reuse is critical** for efficiency across draft/intermediate/target
5. **Periodic target verification maintains accuracy** while improving throughput

### For EventGPT Application

1. **Event features can draft** for video features with intermediate frame verification
2. **Frame encoder as intermediate verifier** - cheaper than full video encoding
3. **Early rejection** of bad event drafts saves expensive video computation
4. **Dynamic verification policy** - verify against video after N tentative acceptances

---

## Limitations

1. Requires early-exit models (with interpretable intermediate hidden states)
2. Default layer positioning (1/8, 1/4) may not be optimal for all tasks
3. Hyperparameter tuning (N_d, N_i) needed for optimal performance
4. Focus on single-model hierarchy (not cross-model like PyramidSD)

---

## References

- LayerSkip: Elhoushi et al., 2024 - Early-exit for draft generation
- AdaDecode: Wei et al., 2025 - Dynamic draft layer selection
- SPRINTER: Zhong et al., 2025 - Intermediate verification (with training)
- Lookahead: Fu et al., 2024 - Parallel draft generation

---

**Document Created**: January 28, 2026
**Source**: arXiv:2510.01336v1
