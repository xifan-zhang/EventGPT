# EAGLE Family: Embedding-Level Speculative Decoding

Analysis of the EAGLE paper series and implications for cross-modal speculative decoding.

---

## Overview

| Paper | Venue | Date | Key Contribution |
|-------|-------|------|------------------|
| [EAGLE](https://arxiv.org/abs/2401.15077) | ICML 2024 | Jan 2024 | Feature-level speculation framework |
| [EAGLE-2](https://arxiv.org/abs/2406.16858) | EMNLP 2024 | Jun 2024 | Context-aware dynamic draft trees |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) | NeurIPS 2025 | Mar 2025 | Training-time test optimization |

**Local PDFs:**
- `EAGLE_2401.15077.pdf` (1.8 MB)
- `EAGLE2_2406.16858.pdf` (1.7 MB)
- `EAGLE3_2503.01840.pdf` (924 KB)

---

## EAGLE-1: Core Framework

### Key Insight

> "Autoregression at the **feature (second-to-top-layer) level** is more straightforward than at the token level."

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  EAGLE-1 ARCHITECTURE                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Target LLM (frozen):                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Embedding → Decoder Layers → [h_{n-1}] → LM Head → p(x) │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                    ↓                            │
│                              Second-to-top                      │
│                              layer features                     │
│                                    ↓                            │
│  Draft Head (trainable):                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [h_t, embed(x_t)] → Single Decoder Layer → h_{t+1}     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                    ↓                            │
│                              Predicted                          │
│                              next feature                       │
│                                    ↓                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  h_{t+1} → LM Head (shared) → draft token               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why Feature-Level Works

1. **Continuity**: Hidden states are continuous vectors, allowing smooth prediction
2. **Semantic compression**: Features encode meaning, not just surface tokens
3. **Lower entropy**: Feature space has lower uncertainty than token space

### Training

```python
# EAGLE training objective
# Given: h_t (current hidden), x_t (current token)
# Predict: h_{t+1} (next hidden state)

loss = MSE(draft_head(h_t, embed(x_t)), h_{t+1})
```

### Results

| Model | Speedup | Throughput |
|-------|---------|------------|
| Vicuna 7B | 2.5-3.0x | 2x |
| Vicuna 13B | 2.6-3.2x | 2x |
| LLaMA2-Chat 70B | 2.7-3.5x | 2x |

---

## EAGLE-2: Dynamic Draft Trees

### Key Innovation

Context-aware dynamic draft tree structure that adapts to input difficulty.

```
┌─────────────────────────────────────────────────────────────────┐
│  EAGLE-2: DYNAMIC DRAFT TREES                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EAGLE-1 (fixed tree):        EAGLE-2 (dynamic tree):          │
│                                                                  │
│       [draft]                      [draft]                      │
│       /    \                       /  |  \                      │
│     [d1]  [d2]                  [d1][d2][d3]                    │
│     / \    |                    / \   |                         │
│   [.][.]  [.]                 [.][.] [.]                        │
│                                                                  │
│  Fixed depth & width      Adapts based on confidence           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Confidence-Based Expansion

```python
# EAGLE-2 expansion rule
if confidence(h_t) > threshold_high:
    expand_more_branches()  # Easy tokens, explore more
elif confidence(h_t) < threshold_low:
    prune_branches()        # Hard tokens, be conservative
```

### Results

| Model | EAGLE-1 | EAGLE-2 | Improvement |
|-------|---------|---------|-------------|
| Vicuna 7B | 2.5x | 3.0x | +20% |
| LLaMA2-Chat 70B | 3.0x | 3.8x | +27% |

---

## EAGLE-3: Training-Time Optimization

### Key Innovation

Optimize draft model during target model's training phase.

```
┌─────────────────────────────────────────────────────────────────┐
│  EAGLE-3: JOINT TRAINING                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Traditional (EAGLE-1/2):                                       │
│    1. Train target LLM (expensive)                              │
│    2. Freeze target, train draft head (cheap)                   │
│                                                                  │
│  EAGLE-3:                                                       │
│    1. Train target LLM + draft head jointly                     │
│    2. Draft head learns better representations                  │
│    3. Even faster inference                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Results

| Method | Speedup | Training Cost |
|--------|---------|---------------|
| EAGLE-1 | 2.5-3.0x | Low (draft only) |
| EAGLE-2 | 3.0-3.8x | Low (draft only) |
| EAGLE-3 | 3.5-4.5x | Medium (joint) |

---

## Relevance to Cross-Modal Speculative Decoding

### Why EAGLE Matters for Our Work

```
┌─────────────────────────────────────────────────────────────────┐
│  EAGLE (Same Model)        vs    Our Approach (Cross-Modal)    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input:  Same text prompt        Different visual modalities    │
│  Draft:  Same LLM                EventGPT (event camera)        │
│  Target: Same LLM                Video-LLaVA (RGB video)        │
│                                                                  │
│  Task:   Predict h_{t+1}         Align h_egpt → h_vl            │
│                                                                  │
│  Why feature-level:              Why feature-level:             │
│  - Lower uncertainty             - Token matching impossible    │
│  - Continuous space              - Semantic alignment possible  │
│  - Small draft head works        - Small adapter works          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Theoretical Foundations from EAGLE

| EAGLE Finding | Our Application |
|---------------|-----------------|
| Feature space has lower entropy than token space | Embedding alignment easier than token matching |
| Second-to-top layer captures semantics | Hidden states capture meaning across modalities |
| Small draft head (~1B for 70B) sufficient | Small adapter (~2M) sufficient for alignment |
| Uncertainty can be resolved with context | Threshold-based acceptance handles variance |

### Differences from EAGLE

| Aspect | EAGLE | Our Approach |
|--------|-------|--------------|
| **Task** | Predict next hidden state | Align cross-modal hidden states |
| **Models** | Same model (draft = target) | Different models (EGPT ≠ VL) |
| **Visual input** | None (text only) | Different (event vs RGB) |
| **Challenge** | Uncertainty in prediction | Modality gap |
| **Solution** | Context-aware prediction | Bottleneck adapter alignment |

---

## Architecture Comparison: EAGLE vs Our Adapter

### EAGLE Draft Head

```
┌─────────────────────────────────────────────────────────────────┐
│  EAGLE DRAFT HEAD (for 7B model)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: [h_t, embed(x_t)]                                       │
│         h_t: [batch, seq, 4096]  (hidden state)                 │
│         embed(x_t): [batch, seq, 4096]  (token embedding)       │
│                                                                  │
│         ↓ Concatenate                                           │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Linear Projection                                       │   │
│  │  [batch, seq, 8192] → [batch, seq, 4096]                │   │
│  │  Params: 8192 × 4096 = 33.5M                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Single Transformer Decoder Layer                        │   │
│  │  - Self-Attention: Q, K, V projections                  │   │
│  │  - FFN: 4096 → 11008 → 4096                             │   │
│  │  Params: ~200M (for 7B-style layer)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  Output: h_{t+1} [batch, seq, 4096]                            │
│                                                                  │
│  TOTAL: ~230-250M parameters (for 7B target)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Our Hidden State Adapter

```
┌─────────────────────────────────────────────────────────────────┐
│  OUR BOTTLENECK ADAPTER                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: h_egpt [batch, seq, 4096]  (EGPT hidden state)         │
│                                                                  │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LayerNorm                                               │   │
│  │  Params: 2 × 4096 = 8K                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Down Projection (Bottleneck)                            │   │
│  │  Linear(4096 → 256, bias=False)                         │   │
│  │  Params: 4096 × 256 = 1.05M                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GELU + Dropout(0.1)                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Up Projection                                           │   │
│  │  Linear(256 → 4096, bias=False)                         │   │
│  │  Params: 256 × 4096 = 1.05M                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Residual + Learnable α                                  │   │
│  │  output = input + α × projected                         │   │
│  │  Params: 1                                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  Output: h_aligned [batch, seq, 4096]                          │
│                                                                  │
│  TOTAL: ~2.1M parameters                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Side-by-Side Comparison

| Component | EAGLE Draft Head | Our Adapter |
|-----------|------------------|-------------|
| **Input** | h_t + embed(x_t) | h_egpt only |
| **Input dim** | 8192 (concat) | 4096 |
| **Output** | h_{t+1} (predicted) | h_aligned (transformed) |
| **Core structure** | Transformer decoder layer | Bottleneck MLP |
| **Attention** | Yes (self-attention) | No |
| **FFN** | 4096→11008→4096 | 4096→256→4096 |
| **Residual** | Yes (standard) | Yes (with learnable α) |
| **LayerNorm** | Multiple | Single |

### Parameter Comparison

| Model Size | EAGLE Draft Head | Our Adapter | Ratio |
|------------|------------------|-------------|-------|
| 7B target | ~230M | **2.1M** | **110x smaller** |
| 13B target | ~400M | **2.1M** | **190x smaller** |
| 70B target | ~1B | **2.1M** | **475x smaller** |

### Why Such Different Sizes?

```
┌─────────────────────────────────────────────────────────────────┐
│  WHY EAGLE NEEDS MORE PARAMETERS                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EAGLE's task: PREDICT next hidden state                        │
│  - Must model complex temporal dynamics                         │
│  - Needs attention to capture dependencies                      │
│  - Requires large FFN for expressive power                      │
│  - Input includes both hidden state AND token embedding         │
│                                                                  │
│  Complexity: h_{t+1} = f(h_t, x_t, context)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  WHY OUR ADAPTER CAN BE SMALLER                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Our task: ALIGN hidden states (not predict)                    │
│  - Linear transformation sufficient for alignment               │
│  - No temporal prediction needed                                │
│  - Bottleneck captures essential mapping                        │
│  - Residual connection preserves input information              │
│                                                                  │
│  Complexity: h_aligned = h_egpt + α × MLP(h_egpt)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Overhead

| Metric | EAGLE (7B) | Our Adapter |
|--------|------------|-------------|
| Parameters | 230M | 2.1M |
| FLOPs per token | ~500M | ~4M |
| Latency | ~2-3ms | **<0.5ms** |
| Memory | ~500MB | ~8MB |

### Design Trade-offs

| Aspect | EAGLE | Our Adapter |
|--------|-------|-------------|
| **Expressiveness** | High (attention + large FFN) | Medium (bottleneck MLP) |
| **Speed** | Moderate | Very fast |
| **Memory** | Moderate | Very low |
| **Task complexity** | Prediction (harder) | Alignment (easier) |
| **Training data** | 70k dialogues | 1k-50k pairs |
| **Training time** | 1-2 days (4×A100) | Minutes-hours (1×4090) |

### Key Insight

```
┌─────────────────────────────────────────────────────────────────┐
│  PREDICTION vs ALIGNMENT                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EAGLE: "What will the next hidden state be?"                   │
│         → Complex function, needs large network                 │
│                                                                  │
│  Ours:  "How to transform EGPT's hidden to match VL's?"        │
│         → Simpler function, small adapter sufficient            │
│                                                                  │
│  Both leverage the KEY INSIGHT:                                 │
│  Embedding space is the right level for speculation!            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Adapter Complexity Analysis: Can We Do Better?

### The Key Argument

```
┌─────────────────────────────────────────────────────────────────────┐
│  IF PREDICTION WORKS, ALIGNMENT SHOULD BE EVEN EASIER!              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  EAGLE (Prediction - HARDER task):                                  │
│  ─────────────────────────────────                                  │
│  Input:  h_t (current hidden state)                                 │
│  Output: h_{t+1} (PREDICT future state)                             │
│  Challenge: Model temporal dynamics, uncertainty                    │
│  Network: ~230M params (Transformer layer)                          │
│  Result: Works! 2.5-3.5x speedup                                    │
│                                                                      │
│  Ours (Alignment - EASIER task):                                    │
│  ────────────────────────────────                                   │
│  Input:  h_egpt (EGPT hidden state)                                 │
│  Output: h_aligned (TRANSFORM to VL space)                          │
│  Challenge: Learn cross-modal mapping                               │
│  Network: ~2M params (Simple bottleneck)                            │
│  Result: Works! 5.77x speedup (train set)                           │
│                                                                      │
│  INSIGHT:                                                           │
│  ─────────────────────────────────────────────────────────────      │
│  If 230M params can solve PREDICTION (harder),                      │
│  then 2M params might be UNDERFITTING for ALIGNMENT (easier)!       │
│                                                                      │
│  → More complex adapter could achieve BETTER alignment              │
│  → Higher acceptance rates → More speedup                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Adapter Design Space

```
┌─────────────────────────────────────────────────────────────────────┐
│  ADAPTER COMPLEXITY SPECTRUM                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Simplest ◄──────────────────────────────────────────► Most Complex │
│                                                                      │
│  Level 1        Level 2        Level 3        Level 4        Level 5│
│  ───────        ───────        ───────        ───────        ───────│
│  Bottleneck    Multi-layer     Wide          Attention      EAGLE   │
│  MLP           Bottleneck      Bottleneck    + MLP          Style   │
│                                                                      │
│  2M params     8M params       16M params    50M params     200M    │
│  <0.5ms        ~1ms            ~1ms          ~2ms           ~3ms    │
│                                                                      │
│  Current ●                                                          │
│  (maybe underfitting?)                                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Proposed Adapter Architectures

#### Level 1: Current (2M) - Simple Bottleneck
```python
# Current implementation
class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim=4096, bottleneck_dim=256):
        self.down = nn.Linear(4096, 256)    # 1M
        self.up = nn.Linear(256, 4096)      # 1M
        self.act = nn.GELU()

    def forward(self, h):
        return h + self.alpha * self.up(self.act(self.down(h)))

# Total: ~2M params
```

#### Level 2: Multi-Layer (8M) - Stacked Bottlenecks
```python
class MultiLayerAdapter(nn.Module):
    def __init__(self, hidden_dim=4096, bottleneck_dim=256, num_layers=3):
        self.layers = nn.ModuleList([
            BottleneckAdapter(hidden_dim, bottleneck_dim)
            for _ in range(num_layers)
        ])

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h

# Total: ~6-8M params
```

#### Level 3: Wide Bottleneck (16M)
```python
class WideAdapter(nn.Module):
    def __init__(self, hidden_dim=4096, bottleneck_dim=1024):
        self.down = nn.Linear(4096, 1024)   # 4M
        self.mid = nn.Linear(1024, 1024)    # 1M
        self.up = nn.Linear(1024, 4096)     # 4M

# Total: ~10-16M params
```

#### Level 4: Attention-Based (50M)
```python
class AttentionAdapter(nn.Module):
    def __init__(self, hidden_dim=4096, num_heads=8):
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        h = h + self.self_attn(h, h, h)[0]
        h = self.norm1(h)
        h = h + self.ffn(h)
        h = self.norm2(h)
        return h

# Total: ~50M params
```

#### Level 5: EAGLE-Style (200M)
```python
class EAGLEStyleAdapter(nn.Module):
    def __init__(self, hidden_dim=4096):
        # Full transformer decoder layer
        self.self_attn = nn.MultiheadAttention(4096, 32)
        self.ffn = nn.Sequential(
            nn.Linear(4096, 11008),  # Vicuna FFN expansion
            nn.SiLU(),
            nn.Linear(11008, 4096),
        )

# Total: ~200M params (like one Vicuna layer)
```

#### Level 6: Cross-Modal Attention (100M) - Novel Design
```python
class CrossModalAdapter(nn.Module):
    """
    Use VL hidden states as guidance during training.
    At inference, use learned cross-modal patterns.
    """
    def __init__(self, hidden_dim=4096, num_heads=8):
        # Cross-attention: EGPT queries, VL provides context
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096),
        )

    def forward(self, h_egpt, h_vl=None):
        # During training: use VL as key/value
        # During inference: use learned patterns or cached VL
        if h_vl is not None:
            h = h_egpt + self.cross_attn(h_egpt, h_vl, h_vl)[0]
        else:
            h = h_egpt + self.self_attn(h_egpt, h_egpt, h_egpt)[0]
        return h + self.ffn(h)

# Total: ~100M params
```

### Expected Performance vs Complexity

| Adapter | Params | Overhead | Expected Accept@0.90 | Speedup |
|---------|--------|----------|---------------------|---------|
| Current (L1) | 2M | <0.5ms | 19.5% (train) | 5.77x |
| Multi-layer (L2) | 8M | ~1ms | ~25%? | ~6x? |
| Wide (L3) | 16M | ~1ms | ~30%? | ~6.5x? |
| Attention (L4) | 50M | ~2ms | ~40%? | ~5x? |
| EAGLE-style (L5) | 200M | ~3ms | ~50%? | ~4x? |

**Note:** More params ≠ always better speedup (overhead increases!)

### Optimal Point Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│  SPEEDUP vs ADAPTER COMPLEXITY                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Speedup                                                            │
│    ▲                                                                │
│    │                    ╭──────╮                                    │
│  6 │              ╭────╯      ╰────╮                                │
│    │         ╭───╯                  ╰───╮                           │
│  5 │    ╭───╯                            ╰──╮                       │
│    │   ╱                                     ╲                      │
│  4 │  ╱                                       ╲                     │
│    │ ╱                                         ╲                    │
│  3 │╱                                           ╲                   │
│    │                                                                │
│  2 │                                                                │
│    │                                                                │
│  1 ├────┬─────┬─────┬─────┬─────┬─────┬─────┬─────►                │
│    │   2M    8M   16M   50M  100M  200M  500M    Params            │
│    │                                                                │
│    │   ◄─── Sweet spot: 8-50M params ───►                          │
│    │                                                                │
│    │  Too simple:        │        Too complex:                      │
│    │  Underfitting       │        Overhead dominates                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Recommendations

1. **Immediate:** Try Level 2 (8M multi-layer) - low risk, potential +20% improvement
2. **If needed:** Try Level 3 (16M wide) - still fast, more capacity
3. **Research:** Try Level 4 (attention) - might capture position dependencies
4. **Avoid:** Level 5 (200M) unless acceptance is critically important - overhead too high

### Experiment Plan

```bash
# Experiment 1: Multi-layer adapter
python train_hidden_adapter.py \
    --bottleneck_dim 256 \
    --num_layers 3 \
    --train_data hidden_states_train_10q.pt

# Experiment 2: Wide bottleneck
python train_hidden_adapter.py \
    --bottleneck_dim 1024 \
    --num_layers 1 \
    --train_data hidden_states_train_10q.pt

# Experiment 3: Attention-based (requires code changes)
python train_attention_adapter.py \
    --num_heads 8 \
    --train_data hidden_states_train_10q.pt
```

### What We Can Learn from EAGLE

1. **Architecture**: Use lightweight adapter (like EAGLE's draft head)
2. **Training**: MSE + context information improves alignment
3. **Inference**: Tree-based verification can improve acceptance
4. **Optimization**: Joint training could improve cross-modal alignment

---

## Potential Extensions

### EAGLE-style Improvements for Cross-Modal SD

```
┌─────────────────────────────────────────────────────────────────┐
│  FUTURE WORK: Combining EAGLE techniques with Cross-Modal SD    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Dynamic threshold (EAGLE-2 style):                          │
│     - High confidence: lower threshold, accept more             │
│     - Low confidence: higher threshold, be conservative         │
│                                                                  │
│  2. Tree-based drafting:                                        │
│     - Generate multiple draft branches from EGPT                │
│     - Pick best match with VL hidden states                     │
│                                                                  │
│  3. Joint training (EAGLE-3 style):                             │
│     - Fine-tune EGPT to produce more VL-aligned hidden states   │
│     - Train adapter during EGPT fine-tuning                     │
│                                                                  │
│  4. Multi-layer features:                                       │
│     - Use features from multiple decoder layers                 │
│     - Better capture of semantic information                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

### EAGLE's Core Contributions

1. **Proves** feature-level speculation is more tractable than token-level
2. **Shows** small draft networks are sufficient for hidden state prediction
3. **Achieves** 2.5-4.5x speedup across model sizes
4. **Provides** theoretical framework for why embedding-level works

### Implications for Cross-Modal SD

1. **Validates** our embedding-level alignment approach
2. **Explains** why token-level failed (high entropy, discrete space)
3. **Suggests** improvements (dynamic thresholds, tree drafting)
4. **Confirms** small adapters can achieve significant speedups

### Key Takeaway

> EAGLE proves that the embedding space is the right abstraction level for speculative decoding. This is the **only** viable approach for cross-modal SD where token matching is fundamentally impossible.

---

## References

- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) (ICML 2024)
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) (EMNLP 2024)
- [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840) (NeurIPS 2025)
- [GitHub: SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
