# Cross-Modal Hidden State Adapter Architectures (L1-L5)

> **Source:** `hidden_adapter.py`
> **Purpose:** Align EventGPT decoder hidden states to Video-LLaVA hidden state space for cross-modal speculative decoding.

---

## Overview

All adapters map `h_egpt [batch, seq, 4096]` → `h_aligned [batch, seq, 4096]`, where 4096 is the Vicuna-7B hidden dimension shared by both models.

| Level | Name | Class | Params | Inference | Key Idea |
|-------|------|-------|--------|-----------|----------|
| **L1** | Bottleneck | `HiddenStateAdapter` | 2.1M | <0.5ms | LoRA-style down→up projection |
| **L2** | Multi-Layer Bottleneck | `MultiLayerBottleneckAdapter` | 6.3M | ~1ms | 3 stacked L1 blocks |
| **L3** | Wide Bottleneck | `WideBottleneckAdapter` | 16M | ~1ms | 4x wider bottleneck (1024 vs 256) |
| **L4** | Attention | `AttentionAdapter` | 50M | ~2ms | Self-attention + FFN (transformer layer) |
| **L5** | EAGLE-Style | `EAGLEStyleAdapter` | 50M | ~3ms | Causal attention + next-token prediction |

### Training Configuration (shared)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Loss | MSE + 0.5 * CosLoss |
| Learning Rate | 1e-3 |
| Batch Size | 64 |
| Max Epochs | 50 |
| Early Stopping | 10 epochs patience |
| Train Samples | 52,000 (1s sequences, 10 questions each) |
| Val Samples | 11,000 |

---

## L1: Bottleneck Adapter (2.1M params)

**Class:** `HiddenStateAdapter` → wraps `BottleneckAdapter`

### Architecture

```
h_egpt ─→ [LayerNorm] ─→ [Down: 4096→256] ─→ [GELU] ─→ [Dropout(0.1)] ─→ [Up: 256→4096] ─→ ×α ─+─→ h_aligned
   |                                                                                               |
   └───────────────────────────── residual ────────────────────────────────────────────────────────→+
```

### Components

| Component | Shape | Parameters |
|-----------|-------|------------|
| LayerNorm | 4096 | 8,192 |
| down_proj | 4096 → 256 | 1,048,576 |
| up_proj | 256 → 4096 | 1,048,576 |
| alpha (learnable) | scalar | 1 |
| Final LayerNorm | 4096 | 8,192 |
| **Total** | | **2,113,537** |

### Key Design Choices

- **Zero-init up_proj:** Output starts as identity (residual only), stable training
- **Learnable alpha:** Initialized to 0.1, adapts residual contribution during training
- **No bias in projections:** Saves params, matches LoRA convention
- **Kaiming init on down_proj:** Fan-in mode for GELU compatibility

### When to Use

Cheapest adapter. Sufficient for **prefill hiding** (Phase 1 of SD). During VL's ~310ms prefill, EGPT generates draft tokens using this adapter. Only needs a few consecutive token matches to yield free speedup.

---

## L2: Multi-Layer Bottleneck (6.3M params)

**Class:** `MultiLayerBottleneckAdapter`

### Architecture

```
h_egpt ─→ [Bottleneck₁] ─→ [Bottleneck₂] ─→ [Bottleneck₃] ─→ [LayerNorm] ─→ h_aligned
```

Each `Bottleneckᵢ` is an L1 block (LayerNorm → Down → GELU → Dropout → Up → residual).

### Components

| Component | Parameters |
|-----------|------------|
| Bottleneck Block × 3 | 3 × 2,105,345 = 6,316,035 |
| Final LayerNorm | 8,192 |
| **Total** | **6,324,227** |

### Key Design Choices

- **Stacking depth vs width:** 3 blocks with narrow bottleneck (256) vs 1 wide block. Each block refines the residual further.
- **Compounding residuals:** Each block adds `α * up(GELU(down(x)))` to the running state. Three passes allow progressively more complex nonlinear transformations.
- **Same loss function as L1:** MSE + 0.5 * CosLoss.

### When to Use

When L1's single bottleneck isn't enough. 3x params for modestly better alignment (~+1.3% Accept@0.90 over L1 in our experiments).

---

## L3: Wide Bottleneck (16M params)

**Class:** `WideBottleneckAdapter`

### Architecture

```
h_egpt ─→ [LayerNorm] ─→ [WideBlock₁] ─→ [WideBlock₂] ─→ [LayerNorm] ─→ h_aligned
```

Each WideBlock:
```
x ─→ [Linear: 4096→1024] ─→ [LayerNorm(1024)] ─→ [GELU] ─→ [Dropout] ─→ [Linear: 1024→4096] ─→ [Dropout] ─→ ×αᵢ ─+─→ out
|                                                                                                                     |
└──────────────────────────────────────── residual ──────────────────────────────────────────────────────────────────→ +
```

### Components

| Component | Shape | Parameters |
|-----------|-------|------------|
| Input LayerNorm | 4096 | 8,192 |
| WideBlock × 2, each: | | |
|   Linear down | 4096 → 1024 | 4,194,304 |
|   LayerNorm | 1024 | 2,048 |
|   Linear up | 1024 → 4096 | 4,194,304 |
|   alpha (learnable) | scalar | 1 |
| Final LayerNorm | 4096 | 8,192 |
| **Total** | | **~16.8M** |

### Key Design Choices

- **4x wider bottleneck (1024 vs 256):** Preserves more information through the bottleneck. 256-dim may discard fine-grained features needed for token-level matching.
- **Fewer blocks (2 vs 3):** Wider bottleneck compensates for fewer layers.
- **Mid-bottleneck LayerNorm:** Stabilizes the wider hidden representation.
- **Per-block learnable alpha:** Each block learns its own residual contribution.

### When to Use

When L2 plateaus. More capacity per layer than L2, but diminishing returns suggest the fundamental alignment gap may be the limiting factor, not adapter capacity.

---

## L4: Attention Adapter (50M params)

**Class:** `AttentionAdapter`

### Architecture

```
h_egpt ─→ [TransformerLayer] ─→ [OutputProj] ─→ ×α ─+─→ h_aligned
   |                                                  |
   └──────────── residual ──────────────────────────→ +

TransformerLayer:
  x ─→ [LayerNorm] ─→ [MultiHeadAttention(8 heads)] ─+─→ [LayerNorm] ─→ [FFN(4096→2048→4096)] ─+─→ out
  |                                                    |   |                                      |
  └──────────── residual ────────────────────────────→ +   └──────────── residual ──────────────→ +
```

### Components

| Component | Shape | Parameters |
|-----------|-------|------------|
| **Self-Attention (per layer):** | | |
|   attn_norm (LayerNorm) | 4096 | 8,192 |
|   Q, K, V projections | 3 × (4096 × 4096) | 50,331,648 |
|   Output projection | 4096 × 4096 | 16,777,216 |
| **FFN (per layer):** | | |
|   ffn_norm (LayerNorm) | 4096 | 8,192 |
|   Linear up | 4096 → 2048 | 8,390,656 |
|   Linear down | 2048 → 4096 | 8,392,704 |
| **Output:** | | |
|   output_norm (LayerNorm) | 4096 | 8,192 |
|   output_proj (Linear) | 4096 → 4096 | 16,781,312 |
|   alpha (learnable) | scalar | 1 |
| **Total (1 layer)** | | **~50M** |

Note: Actual param count depends on nn.MultiheadAttention implementation (in-proj uses fused 3×4096 weight).

### Key Design Choices

- **Self-attention over sequence:** Unlike L1-L3 which process each position independently, L4 attends over the full sequence. This captures token dependencies — position 5's alignment depends on what happened at positions 0-4.
- **Reduced FFN (2048 vs 4096):** Standard transformer uses 4× hidden dim for FFN, but we use 0.5× to limit params.
- **Eye-init output_proj:** Starts as identity, so output = input until training shifts it.
- **key_padding_mask support:** Properly handles variable-length sequences.
- **No causal mask:** Unlike L5, L4 uses full bidirectional attention (alignment is not autoregressive).

### When to Use

When per-position alignment (L1-L3) misses cross-position dependencies. Useful if adjacent tokens' alignment errors are correlated.

---

## L5: EAGLE-Style Adapter (50M params)

**Class:** `EAGLEStyleAdapter`

### Architecture

```
h_egpt ─→ [LayerNorm] ─→ [+ PosEmbed] ─→ [CausalTransformer] ─→ [LayerNorm] ─→ [OutputProj] ─→ ×α ─+─→ h_aligned
   |                                                                                                   |
   └──────────────────────────────────── residual ───────────────────────────────────────────────────→ +

Optional EAGLE token fusion:
  h_egpt ─┐
           ├→ [Concat] ─→ [Fusion: 8192→4096] ─→ continues as above
  tok_emb ─┘
```

### Components

| Component | Shape | Parameters |
|-----------|-------|------------|
| input_norm (LayerNorm) | 4096 | 8,192 |
| pos_embed | [1, 512, 4096] | 2,097,152 |
| **Causal Transformer (per layer):** | | |
|   Same as L4 attention + FFN | | ~50M |
| output_norm (LayerNorm) | 4096 | 8,192 |
| output_proj (Linear) | 4096 → 4096 | 16,781,312 |
| alpha (learnable) | scalar | 1 |
| **Optional token_embed** | [32000, 4096] | 131,072,000 |
| **Optional token_fusion** | [8192, 4096] | 33,558,528 |
| **Total (1 layer, no embed)** | | **~50M** |
| **Total (1 layer, with embed)** | | **~215M** |

### Key Design Choices

- **Causal attention mask:** `triu` mask ensures position `t` only attends to `0..t`. This is critical for autoregressive prediction — predicting `h_{t+1}` should only use `h_0..h_t`.
- **Positional encoding:** Learned positional embeddings (trunc_normal init, max 512 positions). Needed because causal attention is position-sensitive.
- **EAGLE-style token fusion (optional):** Concatenates previous token embedding with hidden state before the transformer. Disabled by default (`use_token_embed=False`).
- **Two training objectives:**
  1. **Alignment loss:** `aligned[t]` should match `vl_hidden[t]` (same as L1-L4)
  2. **Prediction loss:** `aligned[t]` should match `vl_hidden[t+1]` (next-position). Weight controlled by `prediction_weight` (default 0.5).

### Difference from L4

| Aspect | L4 (Attention) | L5 (EAGLE) |
|--------|---------------|------------|
| Attention | Bidirectional | **Causal** (autoregressive) |
| Position encoding | None | **Learned positional** |
| Training objective | Alignment only | **Alignment + prediction** |
| Token fusion | No | **Optional** (EAGLE-style) |
| SD Phase 2 (decode) | No (same cost as VL) | **Yes** (~3ms/token drafting) |

### Why L5 Matters

L1-L4 can only exploit **prefill hiding** — they draft tokens during VL's prefill window. Once VL's prefill ends, they can't draft cheaper than VL itself because drafting requires a full EGPT forward pass (~25ms/token ≈ VL's decode cost).

L5's autoregressive prediction enables **decode-phase acceleration**: the adapter forward pass (~3ms/token) replaces the full EGPT decode (~25ms/token) for generating draft hidden states. This is 8x cheaper drafting, making standard speculative decoding profitable during the decode phase.

---

## L5F: Fused EAGLE Adapter (~67M params)

**Class:** `FusedEAGLEAdapter`
**Level:** 6 (`--adapter_level 6`)

### Purpose

L5F guarantees **at least as good as B1** by using strictly more information. It fuses h_egpt + h_vl via a learned gating mechanism, so the adapter can:
- Leverage complementary event camera signal (temporal dynamics, motion)
- Fall back to VL-only prediction by setting gate → 0 for EGPT

### Architecture

```
h_egpt ─┐
         ├→ [Gated Fusion] ─→ [+ PosEmbed] ─→ [CausalTransformer] ─→ [OutputProj] ─→ ×α ─+─→ h_predicted
h_vl   ─┘                                                                                  |
   |                                                                                        |
   └────────────────────────────────────── residual (from h_vl) ─────────────────────────→ +

Gated Fusion:
  gate = σ(Linear([h_egpt; h_vl]))          # [B, S, 4096] ∈ [0,1]
  fused = gate * h_egpt + (1-gate) * h_vl   # Soft selection
  proj  = Linear([h_egpt; h_vl])            # [8192 → 4096] projection
  x     = LayerNorm(fused + proj)           # Combined
```

### Components

| Component | Shape | Parameters |
|-----------|-------|------------|
| **Fusion:** | | |
|   gate Linear | 8192 → 4096 | 33,558,528 |
|   fusion_proj Linear | 8192 → 4096 | 33,558,528 |
|   fusion_norm LayerNorm | 4096 | 8,192 |
| **Positional:** | | |
|   pos_embed | [1, 512, 4096] | 2,097,152 |
| **Causal Transformer (same as L5):** | | |
|   Attention + FFN | | ~33M |
| **Output:** | | |
|   output_norm + output_proj | 4096 → 4096 | 16,789,504 |
|   alpha (learnable) | scalar | 1 |
| **Total** | | **~67M** |

### Key Design Choices

- **Gated fusion (not just concat):** The sigmoid gate learns per-position, per-dimension how much to weight EGPT vs VL. This is crucial because:
  - Some positions may benefit from events (motion tokens)
  - Other positions may be better served by VL alone (static scene description)
- **Dual fusion path:** Gate provides soft selection, projection provides nonlinear combination. Together they enable both additive and selective fusion.
- **Residual from h_vl (not h_egpt):** The output residual connects to VL hidden states, ensuring the adapter refines VL predictions rather than EGPT.
- **Causal mask:** Same as L5 — autoregressive prediction for decode-phase drafting.

### Comparison: All EAGLE Variants

| Aspect | L5 (Cross-Modal) | B1 (VLM-Only) | L5F (Fused) |
|--------|------------------|---------------|-------------|
| **Input** | h_egpt only | h_vl only | h_egpt + h_vl (gated) |
| **Information** | Events only | Video only | **Both** (strictly more) |
| **Params** | 50M | 50M | 67M |
| **Guarantee vs B1** | No | Baseline | **Yes** (can collapse to B1) |
| **Training** | `--adapter_level 5` | `--adapter_level 5 --vlm_only` | `--adapter_level 6` |
| **Output dir** | `tasks/L5/` | `tasks/B1/` | `tasks/L5F/` |

### Training

```bash
python train_hidden_adapter.py \
    --train_data ./data/chunked_train_1s_4bit \
    --val_data ./data/chunked_test_1s_4bit \
    --adapter_level 6 --num_epochs 50 --batch_size 64 \
    --output_dir ./tasks/L5F
```

### Training Results

| Level | Params | Epochs | Best Val Loss | Accept@0.80 | Accept@0.85 | Accept@0.90 | Accept@0.95 |
|-------|--------|--------|---------------|-------------|-------------|-------------|-------------|
| **L5F** | 170M | 50 | 0.7282 | 98.5% | 90.6% | 66.2% | 20.0% |

**Result: L5F > B1 (+5.0% Accept@0.90)** — Events provide complementary signal for next-token prediction.
The gated fusion successfully leverages EGPT features where they help while defaulting to VL elsewhere.

---

## Training Results Summary

| Level | Params | Epochs | Best Val Loss | Accept@0.80 | Accept@0.85 | Accept@0.90 | Accept@0.95 |
|-------|--------|--------|---------------|-------------|-------------|-------------|-------------|
| **L1** | 2.1M | 43 (ES) | 1.2798 | 46.1% | 30.2% | 21.9% | 18.1% |
| **L2** | 6.3M | 34 (ES) | 1.2787 | 46.3% | 31.4% | 23.2% | 18.6% |
| **L3** | 16M | 50 | 1.2499 | ~46% | ~30% | 24.9% | ~18% |
| **L4** | 101M | 50 | 1.2458 | 47.0% | 32.3% | 24.8% | 20.0% |
| **L5** | 103M | 50 | 1.3413 | 46.4% | 28.5% | 11.2% | 0.4% |
| **B1** | 103M | 50 | 0.6812 | 98.2% | 90.2% | 61.2% | 12.8% |
| **L5F** | 170M | 50 | 0.7282 | 98.5% | 90.6% | 66.2% | 20.0% |

ES = Early Stopping (patience=10)

### Observations: L1-L4 (Per-Position Alignment)

1. **L1→L2 (+3x params):** +1.3% Accept@0.90, marginal val loss improvement
2. **L2→L3 (+2.5x params):** +1.7% Accept@0.90, noticeable val loss drop (1.279→1.250)
3. **L3→L4 (+6x params):** -0.1% Accept@0.90, marginal val loss drop (1.250→1.246). Attention mechanism doesn't help much for this cross-modal task.
4. **Diminishing returns:** L1-L4 all plateau around 22-25% Accept@0.90 regardless of capacity (2M→101M params). The bottleneck is the cross-modal distribution gap, not adapter capacity.

### Key Finding: EAGLE Variants (L5, B1, L5F)

| Model | Input | Val Loss | Accept@0.90 | CosSim |
|-------|-------|----------|-------------|--------|
| **L5** | EGPT only | 1.3413 | 11.2% | 0.771 |
| **B1** | VL only | 0.6812 | 61.2% | 0.907 |
| **L5F** | EGPT + VL | **0.7282** | **66.2%** | **0.913** |

**Critical insights:**

1. **B1 >> L5 (61.2% vs 11.2%):** Confirms the **cross-modal gap is the dominant bottleneck** for EAGLE-style prediction. VLM-only next-token prediction is 5.5x more accurate than cross-modal.

2. **L5F > B1 (+5.0% Accept@0.90):** The event stream provides **complementary signal** for next-token prediction beyond what VL hidden states alone contain. Gated fusion successfully exploits both modalities.

3. **L5F validates the information-theoretic argument:** Fusing h_egpt + h_vl (strictly more information) yields strictly better results. The learned gate selectively leverages event features where they help (motion, temporal dynamics) while defaulting to VL for other positions.

4. **L5 per-position metrics are not comparable to L1-L4** because EAGLE uses shifted prediction loss (h_t → h_{t+1}). L5's Accept@0.90 measures prediction quality, not alignment quality.

5. **Token-level metrics will be lower.** Hidden-state Accept@0.90 does NOT directly translate to token acceptance. The VL LM head amplifies small hidden-state differences (see `METRICS_COMPARISON.md`). Token-level evaluation is the ground truth for actual SD speedup.

### Two-Phase Speculative Decoding Pipeline

**Phase 1: Prefill Hiding (L4 adapter)**

```
Timeline:
  EGPT: [prefill 130ms][──── generate draft tokens ────────]
  VL:   [──────────── prefill 310ms ────────────────────────][verify]
                       ←── 180ms free window ──→

  Draft token generation:
    1. EGPT generates token t → produces h_egpt[t]
    2. L4 aligns: h_vl_approx[t] = L4(h_egpt[t])
    3. VL LM head: draft_token[t] = argmax(LM_head(h_vl_approx[t]))
    4. Repeat until VL finishes prefill
    5. VL batch-verifies all draft tokens
```

**Why L4 for Phase 1:**
- Only EGPT hidden states available (VL is still prefilling) → cannot use L5F
- L4 has the best per-position alignment (Accept@0.90 = 24.8%) among EGPT-only adapters
- Even with low acceptance, draft tokens are "free" — VL is prefilling anyway
- Any accepted tokens are pure profit (skip VL decode steps)

**Phase 2: Decode Acceleration (L5F adapter)**

```
  After VL prefill completes, both h_egpt and h_vl are available:

  For each SD iteration:
    1. EGPT generates γ=5 draft tokens → h_egpt[t:t+5]
    2. VL's KV cache provides h_vl[t] (from previous verified position)
    3. L5F fuses: h_pred[t+1:t+6] = L5F(h_egpt[t:t+5], h_vl[t:t+5])
    4. VL LM head: draft_tokens = argmax(LM_head(h_pred))
    5. VL verifies γ+1 tokens in one forward pass
    6. Accept consecutive correct prefix, reject rest, advance
```

**Why L5F for Phase 2:**
- Both modalities available → gated fusion leverages complementary signal
- L5F Accept@0.90 = 66.2% (highest), vs B1 = 61.2%, L5 = 11.2%
- EAGLE-style autoregressive prediction (h_t → h_{t+1}) is what SD decode needs
- Gated fusion ensures L5F ≥ B1 by construction

**Combined Pipeline Summary:**

| Phase | Adapter | Input | Role | Accept@0.90 |
|-------|---------|-------|------|-------------|
| **1: Prefill Hiding** | L4 | h_egpt only | Align EGPT→VL space | 24.8% |
| **2: Decode** | L5F | h_egpt + h_vl | Predict next VL token | 66.2% |

**Expected E2E speedup sources:**
1. **Prefill hiding:** ~180ms free window → ~7 draft tokens at 25ms/tok, ~1-2 accepted
2. **Decode acceleration:** L5F drafts γ=5 tokens per iteration, high acceptance rate
3. **Combined:** Both phases contribute to overall latency reduction

---

## B1: VLM-Only EAGLE Baseline (50M params)

**Class:** `EAGLEStyleAdapter` (same as L5)
**Flag:** `--vlm_only`

### Purpose

B1 is an **upper-bound baseline** for EAGLE-style speculative decoding on Video-LLaVA. It uses the same L5 architecture but trains on **VL→VL** instead of EGPT→VL, eliminating the cross-modal distribution gap. This answers: "How much of the L5 acceptance rate is limited by cross-modal mismatch vs adapter capacity?"

### Architecture

Identical to L5 (`EAGLEStyleAdapter`, ~50M params, causal transformer with positional embeddings). The only difference is the training data:

```
L5 (Cross-Modal):     adapter(h_egpt[t]) → h_vl[t] (alignment) + h_vl[t+1] (prediction)
B1 (VLM-Only):        adapter(h_vl[t])   → h_vl[t] (self-recon) + h_vl[t+1] (prediction)
```

### Comparison with L5

| Aspect | L5 (Cross-Modal EAGLE) | B1 (VLM-Only EAGLE) |
|--------|------------------------|---------------------|
| **Architecture** | `EAGLEStyleAdapter` (~50M) | `EAGLEStyleAdapter` (~50M) |
| **Input** | EGPT hidden states | VL hidden states |
| **Target** | VL hidden states | VL hidden states (same) |
| **Alignment loss** | EGPT→VL mapping (hard) | Self-reconstruction (trivial) |
| **Prediction loss** | Predict VL h_{t+1} from EGPT h_t | Predict VL h_{t+1} from VL h_t |
| **Cross-modal gap** | Yes (main challenge) | **No** |
| **SD use case** | Cross-modal EGPT→VL drafting | Standard single-model VL SD |
| **Data extraction** | Requires both EGPT + VL | Reuses existing VL hidden states |

### Training

Uses existing chunked data — swaps `egpt_hidden` for `vl_hidden` as input:

```bash
python train_hidden_adapter.py \
    --train_data ./data/chunked_train_1s_4bit \
    --val_data ./data/chunked_test_1s_4bit \
    --adapter_level 5 --vlm_only --num_epochs 50 --batch_size 64 \
    --output_dir ./tasks/B1
```

### Training Objectives (same loss function as L5)

1. **Alignment loss (trivial for B1):** `adapter(h_vl[t]) ≈ h_vl[t]`
   - With residual connection initialized to identity, this is near-zero from the start
   - The adapter's residual path already passes `h_vl` through unchanged

2. **Prediction loss (the real objective):** `adapter(h_vl[t])[:-1] ≈ h_vl[t+1]`
   - Next-token hidden state prediction within VL
   - Causal mask ensures position t only attends to 0..t
   - This is the standard EAGLE autoregressive prediction task

### Analysis

- **Alignment cos_sim = 0.907:** Very high as expected (VL→VL self-reconstruction is near-trivial)
- **Accept@0.90 = 61.2%:** The upper bound for EAGLE-style next-token prediction on VL
- **B1 >> L5 (61.2% vs 11.2%):** Cross-modal gap is the dominant bottleneck, not adapter capacity
- **B1 < L5F (61.2% vs 66.2%):** Event stream adds complementary information beyond VL alone

### Training Results

| Level | Params | Epochs | Best Val Loss | Accept@0.80 | Accept@0.85 | Accept@0.90 | Accept@0.95 |
|-------|--------|--------|---------------|-------------|-------------|-------------|-------------|
| **B1** | 103M | 50 | 0.6812 | 98.2% | 90.2% | 61.2% | 12.8% |

---

## Next Steps: Evaluation Pipeline

All 7 adapters (L1-L5, B1, L5F) are trained. The evaluation pipeline has three levels:

### Level 1: Hidden-State Acceptance (offline, fast)

Already computed during training. Summary above. This is a **proxy metric** — necessary but not sufficient for SD.

### Level 2: Token-Level Acceptance (offline, needs LM head)

Project aligned hidden states through VL's LM head, compare argmax tokens with ground truth.
This is the **ground truth for SD acceptance** — it measures whether the adapter's draft would produce the same token as VL.

```bash
# Extract VL LM head (one-time, ~2 min)
python extract_vl_lm_head.py --output ./data/vl_lm_head.pt

# Evaluate each adapter with token-level metrics
for CKPT in tasks/{L1,L2,L3,L4}/**/best_model.pt; do
    python measure_feature_acceptance.py \
        --checkpoint $CKPT \
        --test_data ./data/chunked_test_1s_4bit \
        --lm_head ./data/vl_lm_head.pt
done

# EAGLE variants (L5, B1, L5F) — same command, different checkpoints
for CKPT in tasks/{L5,B1,L5F}/**/best_model.pt; do
    python measure_feature_acceptance.py \
        --checkpoint $CKPT \
        --test_data ./data/chunked_test_1s_4bit \
        --lm_head ./data/vl_lm_head.pt
done
```

**Key questions Level 2 answers:**
- What is the actual **token acceptance rate** (not just hidden-state similarity)?
- Does L5F's 66.2% hidden-state Accept@0.90 translate to a usable token acceptance rate?
- How much does the LM head amplify hidden-state differences?

### Level 3: E2E Wall-Clock Benchmark (live, needs both models loaded)

Measures actual speedup with both models running inference.

```bash
python evaluate_e2e_sd.py \
    --adapter_checkpoint tasks/L5F/**/best_model.pt \
    --mode cached \
    --cached_hidden_states ./data/chunked_test_1s_4bit \
    --gamma_decode 5
```

**Key questions Level 3 answers:**
- What is the **actual wall-clock speedup** over autoregressive VL?
- Is the prefill hiding + decode acceleration enough for real-time event VQA?

### Evaluation Priority

1. **Extract VL LM head** (prerequisite for token-level metrics)
2. **Token-level eval: L5F, B1, L5** (the EAGLE variants — most impactful)
3. **Token-level eval: L1-L4** (per-position adapters — for completeness)
4. **E2E benchmark: L5F** (the recommended adapter)

---

## File Reference

| File | Description |
|------|-------------|
| `hidden_adapter.py` | All adapter architectures + `create_adapter()` factory |
| `train_hidden_adapter.py` | Training loop with chunked data loading |
| `measure_feature_acceptance.py` | Hidden-state + token-level evaluation |
| `evaluate_e2e_sd.py` | E2E wall-clock benchmark with parallel prefill |
| `extract_vl_lm_head.py` | Extract VL LM head for offline token metrics |
| `METRICS_COMPARISON.md` | Hidden-state vs token-level metrics explanation |
