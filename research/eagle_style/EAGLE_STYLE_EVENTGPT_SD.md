# EAGLE-Style Speculative Decoding: EventGPT → Video VLMs

## Executive Summary

This document explores using EventGPT as a **feature-level draft model** for Video VLMs (e.g., VideoLLaVA) following the EAGLE paradigm. Instead of predicting tokens, we predict hidden states, which sidesteps the cross-modal tokenizer mismatch problem and achieves higher acceptance rates.

**Key Insight:** EAGLE-style feature prediction is fundamentally easier than token prediction because hidden states are continuous vectors with lower entropy than discrete tokens, and semantic information is preserved without surface-level variance.

---

## 1. Background: The EAGLE Family

### 1.1 EAGLE-1 (ICML 2024)
- **Core Innovation:** Predict next hidden state instead of next token
- Uses second-to-last layer features with a small draft head
- Shares embedding layer and LM head with target model
- Achieves **2.5-3.5x speedup**

### 1.2 EAGLE-2 (EMNLP 2024)
- Introduces **dynamic draft trees** with confidence-based expansion/pruning
- Adapts to context difficulty dynamically
- **+20-27% improvement** over EAGLE-1

### 1.3 EAGLE-3 (NeurIPS 2025)
- Fuses multi-layer features (low, middle, high)
- Removes feature prediction constraint, directly predicts tokens
- Achieves up to **6.5x speedup**
- New scaling law: more training data → proportional speedup increase

---

## 2. Why EAGLE-Style for Cross-Modal SD?

### 2.1 The Token Mismatch Problem

Traditional speculative decoding requires token-level alignment:

```
Draft Model:  "The" → "car" → "is" → "red"
Target Model: "The" → "car" → "is" → "red"  ✓ Same tokenizer
```

Cross-modal challenge:
```
EventGPT:     [event_tok_1] → [event_tok_2] → ...
VideoLLaVA:   [video_tok_1] → [video_tok_2] → ...
              ↑ Different tokenizers, incompatible!
```

### 2.2 Feature-Level Solution

EAGLE-style operates at the hidden state level, where semantic alignment is achievable:

```
EventGPT features:    h_e ∈ R^4096  ──┐
                                      ├──→ Adapter → h'_v ≈ h_v
VideoLLaVA features:  h_v ∈ R^4096  ──┘
```

**Key Advantage:** Features encode semantics, not surface tokens. Same scene → similar semantics → alignable features.

---

## 3. Proposed Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                 EAGLE-Style EventGPT → VideoLLaVA                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────────────┐   │
│  │   EventGPT   │    │  Feature    │    │     VideoLLaVA       │   │
│  │   Encoder    │───→│  Adapter    │───→│   Verification       │   │
│  │  (66ms)      │    │  (2-10M)    │    │   (568ms prefill)    │   │
│  └──────────────┘    └─────────────┘    └──────────────────────┘   │
│         │                   │                      │                │
│         ▼                   ▼                      ▼                │
│    Event frames      h'_{t+1...t+k}          Accept/Reject         │
│    → h_e^{L-1}       (draft features)        → Final tokens        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Details

#### A. EventGPT Feature Extractor
- Extract features from **penultimate layer** (layer L-1)
- These contain rich semantic information before final projection
- Dimension: 4096 (Vicuna backbone)

#### B. Feature Adapter (Draft Head)
- **Architecture:** Bottleneck MLP
  ```
  Linear(4096 → 256) → ReLU → Linear(256 → 4096)
  ```
- **Parameters:** ~2-10M (lightweight)
- **Purpose:** Align EventGPT feature space → VideoLLaVA feature space

#### C. Draft Feature Generation
- Given EventGPT features h_e^t, predict VideoLLaVA features h'_v^{t+1}
- Auto-regressive: h'_v^{t+k} = Adapter(h_e^t, h'_v^{t+1}, ..., h'_v^{t+k-1})

#### D. VideoLLaVA Verification
- Convert draft features to logits via shared LM head
- Standard speculative verification with acceptance sampling

---

## 4. Advantages Over Existing Approaches

### 4.1 Comparison Table

| Aspect | Token-Level SD | Parallel Prefill | **EAGLE-Style** |
|--------|---------------|------------------|-----------------|
| Acceptance Rate | 40-70% | N/A (drafts only) | **80-95%** |
| Speedup | 1.5-2x | 1.3-1.6x | **3-5x** |
| Cross-Modal Gap | Tokenizer block | Avoided | **Bridged** |
| Training Req. | Token pairs | Minimal | Feature pairs |
| Complexity | Low | Low | Medium |

### 4.2 Why EventGPT is Ideal

1. **4.5x Faster Prefill**
   - EventGPT: 66ms
   - VideoLLaVA: 568ms
   - Draft generation cost is negligible

2. **7.3x Fewer Visual Tokens**
   - Events are sparse → fewer tokens to process
   - Lighter draft head computation

3. **Semantic Overlap**
   - Same scene captured by both modalities
   - Events and frames contain correlated information
   - Feature alignment is semantically grounded

4. **Parallel Prefill Bonus**
   - Run both prefills simultaneously
   - EventGPT generates **~28 free draft tokens** during VideoLLaVA's prefill
   - Combined with EAGLE-style: **potential 4-6x total speedup**

---

## 5. Implementation Strategy

### 5.1 Phase 1: Feature Extraction & Analysis

```python
# Extract penultimate layer features from both models
def extract_features(model, inputs):
    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        # L-1 layer features (second to last)
        features = outputs.hidden_states[-2]
    return features

# Collect aligned feature pairs
eventgpt_features = extract_features(eventgpt, event_inputs)
videollava_features = extract_features(videollava, video_inputs)
```

### 5.2 Phase 2: Adapter Training

```python
class EAGLEStyleAdapter(nn.Module):
    def __init__(self, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim)
        )
        # Optional: feature fusion for auto-regressive drafting
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, eventgpt_feat, prev_draft_feat=None):
        adapted = self.adapter(eventgpt_feat)
        if prev_draft_feat is not None:
            # Fuse with previous draft for auto-regressive generation
            fused = torch.cat([adapted, prev_draft_feat], dim=-1)
            adapted = self.fusion(fused)
        return adapted

# Training objective: minimize feature distance
def train_adapter(adapter, eg_feats, vl_feats):
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)

    for epoch in range(epochs):
        pred_feats = adapter(eg_feats)
        # L2 loss in feature space
        loss = F.mse_loss(pred_feats, vl_feats)
        # Optional: cosine similarity loss
        loss += (1 - F.cosine_similarity(pred_feats, vl_feats, dim=-1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 5.3 Phase 3: EAGLE-Style Draft Generation

```python
def eagle_style_draft(eventgpt, adapter, event_input, num_drafts=5):
    """Generate draft features using EAGLE-style prediction."""
    # Get EventGPT features
    eg_features = extract_features(eventgpt, event_input)

    draft_features = []
    prev_feat = None

    for i in range(num_drafts):
        # Predict next VideoLLaVA feature
        draft_feat = adapter(eg_features[:, -1], prev_feat)
        draft_features.append(draft_feat)
        prev_feat = draft_feat

    return torch.stack(draft_features, dim=1)  # [B, num_drafts, hidden_dim]
```

### 5.4 Phase 4: Verification with VideoLLaVA

```python
def verify_draft_features(videollava, draft_features, lm_head):
    """Verify draft features using VideoLLaVA's LM head."""
    # Convert features to logits
    draft_logits = lm_head(draft_features)  # [B, num_drafts, vocab_size]
    draft_tokens = draft_logits.argmax(dim=-1)

    # Run VideoLLaVA verification pass
    with torch.no_grad():
        target_outputs = videollava(input_ids=draft_tokens, output_hidden_states=True)
        target_features = target_outputs.hidden_states[-2]
        target_logits = lm_head(target_features)

    # Acceptance sampling
    accepted = speculative_accept(draft_logits, target_logits)

    return accepted, draft_tokens[:, :accepted]
```

---

## 6. Expected Performance

### 6.1 Theoretical Analysis

**Acceptance Rate Improvement:**
- Token-level cross-modal: ~19.5% (baseline, no training)
- Token-level with adapter: ~60-70%
- **EAGLE-style feature-level: ~80-95%**

**Speedup Calculation:**
```
Traditional SD:  Speedup = (k+1) / (1 + k * α * (1-p))
EAGLE-style:     Speedup = (k+1) / (1 + β)

Where:
- k = draft length (5-10)
- α = draft/target time ratio
- p = acceptance rate
- β = adapter overhead (negligible ~0.01)

With p=0.85, k=8:
Traditional: (8+1) / (1 + 8*0.1*(1-0.85)) = 9/1.12 = 8.0x (theoretical max)
Practical: ~3-5x (with overhead and tree verification)
```

### 6.2 Projected Results

| Configuration | Acceptance Rate | Speedup | Notes |
|--------------|-----------------|---------|-------|
| Baseline (no adapter) | 19.5% | 1.1x | Cross-modal gap |
| Token-level + adapter | 60-70% | 1.8-2.2x | Current approach |
| **EAGLE-style** | 80-90% | **3-4x** | Feature alignment |
| **EAGLE + Parallel Prefill** | 80-90% | **4-5x** | Combined benefits |

---

## 7. Challenges and Mitigations

### 7.1 Feature Space Mismatch

**Challenge:** EventGPT and VideoLLaVA may have different internal representations.

**Mitigation:**
- Use cosine similarity loss in addition to L2
- Multi-layer feature fusion (like EAGLE-3)
- Contrastive learning on aligned pairs

### 7.2 Auto-regressive Drift

**Challenge:** Draft features may drift from target distribution over multiple steps.

**Mitigation:**
- Limit draft length (k=5-8)
- Use dynamic tree expansion (EAGLE-2 style)
- Confidence-based early termination

### 7.3 Training Data Requirements

**Challenge:** Need aligned event-video pairs with ground truth.

**Mitigation:**
- Use existing DSEC dataset (aligned events + frames)
- Generate synthetic pairs from same scenes
- Online distillation during inference (like SpecVLM)

---

## 8. Comparison with SpecVLM

| Aspect | SpecVLM | EAGLE-Style EventGPT |
|--------|---------|---------------------|
| Draft Source | Same-family small VLM | Cross-modal EventGPT |
| Visual Handling | Elastic compressor | Separate modality |
| Speedup | 1.5-2.3x | **3-5x (projected)** |
| Training | Online logit distillation | Offline feature alignment |
| Novelty | Within-family SD | **Cross-modal feature SD** |

**Key Differentiator:** SpecVLM uses same-family models; EAGLE-style EventGPT enables **cross-modal** speculative decoding by operating at the feature level.

---

## 9. Research Roadmap

### Phase 1: Validation (2 weeks)
- [ ] Extract and analyze feature distributions from both models
- [ ] Measure baseline feature similarity (cosine, L2)
- [ ] Train simple adapter and measure alignment improvement

### Phase 2: EAGLE-Style Implementation (3 weeks)
- [ ] Implement auto-regressive draft feature generation
- [ ] Integrate with VideoLLaVA verification
- [ ] Benchmark acceptance rates

### Phase 3: Optimization (2 weeks)
- [ ] Dynamic draft tree (EAGLE-2 style)
- [ ] Multi-layer feature fusion (EAGLE-3 style)
- [ ] Parallel prefill integration

### Phase 4: Evaluation (1 week)
- [ ] End-to-end latency benchmarks
- [ ] Quality comparison (BLEU, ROUGE, accuracy)
- [ ] Ablation studies

---

## 10. Conclusion

EAGLE-style speculative decoding with EventGPT offers a compelling approach for accelerating Video VLMs:

1. **Sidesteps tokenizer mismatch** by operating at feature level
2. **Higher acceptance rates** (80-95% vs 60-70% token-level)
3. **Leverages EventGPT's speed advantage** (4.5x faster prefill)
4. **Combinable with parallel prefill** for 4-5x total speedup

This represents a novel contribution: **cross-modal feature-level speculative decoding**, extending EAGLE's principles beyond same-family models to heterogeneous multimodal systems.

---

## References

1. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (ICML 2024)
2. EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (EMNLP 2024)
3. EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test Optimization (NeurIPS 2025)
4. SpecVLM: A Simple Baseline for Speculative Decoding in VLMs (2024)
5. Faster Cascades via Speculative Decoding (ICLR 2025)

---

*Document created: 2026-02-06*
*Status: Research Proposal*
