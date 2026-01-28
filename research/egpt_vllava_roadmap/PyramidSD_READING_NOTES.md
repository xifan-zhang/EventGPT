# PyramidSD: 3-Model Speculative Decoding

**Paper**: arXiv:2510.12966v1
**Authors**: Sanghyun Byun et al. (LG Electronics USA)
**PDF**: `pdf/PyramidSD_3model_speculative_decoding.pdf`

---

## Core Idea

PyramidSD introduces a **3-model hierarchy** for speculative decoding:

```
Draft (1B) → Qualifier (3B) → Target (8B)
   M_D           M_Q              M_T
```

**Key insight**: Larger models produce lower-entropy, higher-confidence distributions. The qualifier model naturally sits between draft and target, bridging the distributional gap.

---

## Problem with Standard 2-Model SD

### Acceptance Rate Limitation

```
Acceptance rate: β = P(Div(P_MT(xt), P_MD(xt)) ≤ τ)

As size gap increases → distributions diverge → lower acceptance → reduced speedup
```

### Why Adding a Third Model Naively Fails

Standard SD with strict equality matching makes the qualifier redundant:

```
Token accepted only if: P_MT(xt) ≥ P_MQ(xt) ≥ P_MD(xt)
```

Since the draft must already match the target, the qualifier provides no benefit.

---

## PyramidSD Solution: Fuzzy Two-Stage Acceptance

### Two-Stage Acceptance Criterion

```
Stage 1 (Draft → Qualifier):
    Div(P_MQ(xt), P_MD(xt)) ≤ τ_Q

Stage 2 (Qualifier → Target):
    Div(P_MT(xt), P_MQ(xt)) ≤ τ_T
```

**Key**: Relaxed divergence thresholds at each stage enable higher acceptance rates.

### How It Works

1. Draft model M_D generates ℓ_D tokens
2. Qualifier M_Q verifies with threshold τ_Q until ℓ_Q tokens generated
3. Target M_T performs final verification with threshold τ_T
4. P_MD is replaced with P_MQ in target comparison (closer distribution)

---

## Throughput Analysis

### Standard SD Throughput

```
V_SD = β_T,D × (ℓ_D + 1) / (ℓ_D/V_D + 1/V_T)

Where:
- β_T,D = acceptance rate (target vs draft)
- ℓ_D = speculative length
- V_D, V_T = model speeds (tokens/sec)
```

### PyramidSD Throughput (Nested Speculation)

**Step 1**: Compute effective speed of draft → qualifier speculation:
```
V_FSD_Q,D = β_Q,D × (ℓ_D + 1) / (ℓ_D/V_D + 1/V_Q)
```

**Step 2**: Compute overall PyramidSD throughput:
```
V_PSD = β_T,Q × (ℓ_Q + 1) / (ℓ_Q/V_FSD_Q,D + 1/V_T)
```

### Design Levers

| Lever | Description | Trade-off |
|-------|-------------|-----------|
| ℓ_D | Draft speculation length | More parallelism vs. error compounding |
| ℓ_Q | Qualifier speculation length | Longer windows vs. acceptance drop |
| τ_Q | Draft-qualifier threshold | Throughput vs. quality filtering |
| τ_T | Qualifier-target threshold | Speed vs. final correctness |

---

## Two Variants

### PSD_A (Assisted Decoding Variant)

- Uses **assisted decoding** instead of τ_Q
- When draft tokens rejected by M_Q, samples directly from qualifier's distribution
- Provides guaranteed quality floor matching M_Q
- **Trade-off**: More stable, but reduced acceleration

### PSD_F (Fuzzy Variant)

- Applies **fuzzy relaxation at both stages**
- Higher average throughput
- **Trade-off**: Larger variance due to compounding fuzzy acceptance

| Variant | Speedup vs SD | Stability | Best For |
|---------|---------------|-----------|----------|
| **PSD_A** | 1.44× | High (low variance) | Production |
| **PSD_F** | 1.91× | Lower (high variance) | Peak performance |

---

## Entropy Gradient Analysis (Figure 3)

### Key Observation

| Model | Size | Mean Entropy | Confidence | Behavior |
|-------|------|--------------|------------|----------|
| LLaMA-1B | Draft | 2.433 (high) | Low | Uniform distributions, widespread uncertainty |
| LLaMA-3B | Qualifier | 2.203 (medium) | Medium | Sharper, better aligned with target |
| LLaMA-8B | Target | 1.900 (low) | High | Right-skewed, selective predictions |

### Why This Matters

```
Small model (high entropy) → Medium model (medium entropy) → Large model (low entropy)
        Fast, uncertain              Filters bad drafts           Slow, confident
```

The qualifier naturally operates in an intermediate regime:
- Validates high-quality draft tokens
- Rejects low-confidence outputs before they reach expensive target
- Enables higher effective acceptance rates

---

## Experimental Results

### Setup

- **Draft**: LLaMA-3.2-1B (instruction-tuned)
- **Qualifier**: LLaMA-3.2-3B (instruction-tuned)
- **Target**: LLaMA-3.1-8B (instruction-tuned)
- **Hardware**: RTX 4090 (24GB)
- **Benchmark**: CommonsenseQA (CSQA)

### Main Results

| Method | Speedup vs SD | Peak tok/s | CSQA Score |
|--------|---------------|------------|------------|
| SD (baseline) | 1.00× | 65 | 69.58±2.20 |
| FSD (fuzzy) | ~1.2× | 83 | ~70.0 |
| **PSD_A** | **1.44×** | 93 | 70.73±2.57 |
| **PSD_F** | **1.91×** | **124** | 72.63±10.06 |

### Best Configuration

```
τ_T = 0.5
τ_Q = 0.4
ℓ_Q = 25
ℓ_D = 15

Result: 124 tokens/second on RTX 4090
```

### Speed vs Threshold (Figure 2)

| τ_T | FSD tok/s | PSD_A tok/s | PSD_F tok/s |
|-----|-----------|-------------|-------------|
| 0.2 | ~70 | ~75 | ~80 |
| 0.3 | ~75 | ~85 | ~95 |
| 0.4 | ~80 | ~90 | ~110 |
| 0.5 | ~82 | ~93 | ~120 |

---

## Key Findings

### 1. Threshold Ordering: τ_Q ≤ τ_T

**Best practice**: Qualifier filters strict, target verifies relaxed

```
τ_Q = 0.3, τ_T = 0.5  ✓ Good
τ_Q = 0.5, τ_T = 0.3  ✗ Bad (forwards low-quality to target)
```

### 2. Optimal Speculation Length Ratio

**ℓ_D : ℓ_Q ratio of 1:2 to 1:3 works best**

| ℓ_D | ℓ_Q | Ratio | Performance |
|-----|-----|-------|-------------|
| 2 | 4 | 1:2 | Good |
| 4 | 10 | 1:2.5 | Better |
| 5 | 15 | 1:3 | Best |
| 10 | 15 | 1:1.5 | Worse (ℓ_D too long) |

**Why**:
- Too long ℓ_D → acceptance drops due to error compounding
- Too short ℓ_D → insufficient parallelism benefit

### 3. Non-Monotonic Performance

More aggressive speculation doesn't always help:
- Medium thresholds often outperform extreme settings
- Aggressive speculation increases rejection cascades
- **Sweet spot**: τ_T ≈ 0.4-0.5, τ_Q ≈ 0.3-0.4

### 4. No Training Required

Works with off-the-shelf model families that share:
- Tokenizers
- Vocabularies
- Architectural compatibility

---

## Ablation Studies

### Threshold Sensitivity (Table 1)

| Config | τ_Q | τ_T | CSQA Score |
|--------|-----|-----|------------|
| SD | - | - | 69.58±2.20 |
| FSD | - | 0.4 | 71.20±2.55 |
| PSD_A | - | 0.4 | 70.73±2.57 |
| PSD_F | 0.4 | 0.4 | 72.63±10.06 |
| PSD_F | 0.2 | 0.2 | 72.31±8.85 |
| PSD_F | 0.5 | 0.5 | 70.00±10.11 |

### Speculation Length Sensitivity (Tables 2-3)

**Best configurations for PSD_F**:

| ℓ_Q | ℓ_D | τ_Q | τ_T | tok/s |
|-----|-----|-----|-----|-------|
| 25 | 15 | 0.4 | 0.5 | **124.13** |
| 25 | 25 | 0.5 | 0.5 | 122.89 |
| 20 | 20 | 0.4 | 0.4 | 115.86 |
| 15 | 15 | 0.5 | 0.5 | 115.62 |

---

## Application to EventGPT → VideoLLaVA

### Direct Analogy

```
PyramidSD:
    LLaMA-1B (draft) → LLaMA-3B (qualifier) → LLaMA-8B (target)
    High entropy         Medium entropy         Low entropy

Cross-Modal Cascade:
    EventGPT (sparse) → Frame Encoder (medium) → VideoLLaVA (dense)
    High entropy          Medium entropy           Low entropy
    (sparse events)       (single frames)          (video context)
```

### Why This Maps Well

| PyramidSD Concept | EventGPT Application |
|-------------------|---------------------|
| Entropy gradient across model sizes | Entropy gradient across modalities |
| 1B draft → 3B qualifier → 8B target | Event encoder → Frame encoder → Video VLM |
| Fuzzy acceptance thresholds | Modality-aware acceptance criteria |
| No training required | Feature alignment MLP only |
| Shared tokenizer requirement | Shared feature space via alignment |

### Proposed Cross-Modal Cascade

```python
class CrossModalPyramidSD:
    """
    Stage 1: Event → Frame (fuzzy acceptance)
    Stage 2: Frame → Video (fuzzy acceptance)
    """

    def __init__(self, event_encoder, frame_encoder, video_llava,
                 tau_Q=0.3, tau_T=0.5):
        self.event_encoder = event_encoder    # Draft (fast, sparse)
        self.frame_encoder = frame_encoder    # Qualifier (medium)
        self.video_llava = video_llava        # Target (slow, dense)
        self.tau_Q = tau_Q
        self.tau_T = tau_T

    def forward(self, events, frames, video):
        # Stage 1: Event drafting
        event_features = self.event_encoder(events)  # Fast, high entropy

        # Stage 1 verification: Event vs Frame
        frame_features = self.frame_encoder(frames)
        div_Q = self.compute_divergence(frame_features, event_features)

        if div_Q <= self.tau_Q:
            # Stage 2: Frame → Video verification
            video_features = self.video_llava.encode(video)
            div_T = self.compute_divergence(video_features, frame_features)

            if div_T <= self.tau_T:
                # Accept event-based draft (fast path)
                return self.video_llava.decode(event_features)

        # Fallback to full video encoding (slow path)
        return self.video_llava.decode(video_features)
```

### Expected Benefits

| Metric | 2-Model SD | PyramidSD (3-Model) |
|--------|------------|---------------------|
| Vision latency | 50ms | 20ms (event draft accepted) |
| Acceptance rate | 70% | 85% (better alignment) |
| Overall speedup | 2.0× | 2.5-3.0× |

### Modality-Specific Thresholds

| Stage | Modality Pair | Recommended τ |
|-------|---------------|---------------|
| 1 | Event → Frame | τ_Q = 0.2-0.3 (strict, filter noise) |
| 2 | Frame → Video | τ_T = 0.4-0.5 (relaxed, temporal tolerance) |

---

## Limitations

1. **Complex hyperparameter space**
   - (τ_Q, τ_T, ℓ_D, ℓ_Q) interactions vary by task/dataset
   - Requires grid search for optimal configuration

2. **Variance in PSD_F**
   - Multiplicative acceptance rates amplify fluctuations
   - Errors at draft stage propagate downstream

3. **Requires compatible model families**
   - Shared tokenizers and vocabularies
   - May limit adoption for heterogeneous architectures

4. **Non-plug-and-play**
   - Optimal config varies across tasks
   - Diminishes ease-of-use compared to standard SD

---

## Key Takeaways

### For Research

1. **Entropy gradient is exploitable** - smaller models are fast but uncertain; use them as filters
2. **Hierarchical filtering** - qualifier catches obvious mismatches cheaply
3. **Fuzzy acceptance enables cascade** - strict matching makes intermediate models useless
4. **1.91× speedup** achievable with 3 models vs 2-model SD
5. **No training needed** - just need models from same family

### For EventGPT Application

1. **Modality = model size** in terms of entropy gradient
2. **Event features can draft** for video features with fuzzy acceptance
3. **Frame encoder as qualifier** bridges event-video distributional gap
4. **Tune τ_Q strict, τ_T relaxed** for stability with speedup

---

## Comparison with Other Papers

| Paper | Models | Training | Speedup | Key Idea |
|-------|--------|----------|---------|----------|
| **PyramidSD** | 3 (off-shelf) | None | 1.91× | Entropy gradient + fuzzy |
| SpecVLM | 2 (draft trained) | Online distill | 2.5× | Visual compression |
| HiSpec | 2 + early-exit | Partial | 2.5× | Verification acceleration |
| PipeSpec | k models | None | 2.8× | Async pipeline |

**PyramidSD advantage**: No training, simple integration, exploits existing model families.

---

## References

- Fuzzy Speculative Decoding [6]: Holsman et al., 2025
- EAGLE [8]: Li et al., 2024
- Cascade Speculative Drafting [3]: Chen et al., 2024
- Staged Speculative Decoding [10]: Spector & Re, 2023

---

**Document Created**: January 28, 2026
**Source**: arXiv:2510.12966v1
