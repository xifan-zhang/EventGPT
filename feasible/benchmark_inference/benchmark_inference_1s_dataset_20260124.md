# Research Opportunities: EventGPT + Video-LLaVA Benchmark Analysis

## Benchmark Summary (1100 samples, 1s dataset)

| Metric | EventGPT | Video-LLaVA | Insight |
|--------|----------|-------------|---------|
| **Total Time** | 1.958s | 1.979s | Nearly identical |
| **Stage 4 (LLM)** | 0.947s (48.4%) | 1.957s (98.9%) | Video-LLaVA dominated by LLM |
| **Stage 3 (Vision)** | 0.006s (0.3%) | 0.006s (0.3%) | Both very fast |
| **Avg Tokens** | 41.3 | 101.4 | Video-LLaVA 2.5x more verbose |
| **Token Acceptance** | - | - | **3.11%** (very low) |
| **Speedup** | 1.01x | - | Essentially same speed |

---

## Key Finding: Why Speculative Decoding Fails

The **3.11% token acceptance rate** indicates speculative decoding is not viable in the current setup.

### Root Causes:

1. **Different Tokenizers**
   - EventGPT: LLaMA tokenizer (32,000 vocab)
   - Video-LLaVA: Vicuna tokenizer (32,000 vocab, but different token mappings)

2. **Different Output Styles**
   - EventGPT: Concise (~41 tokens): "The scene depicts a road with a car driving ahead..."
   - Video-LLaVA: Verbose (~101 tokens): "The image captures a blurry view of a street from the perspective..."

3. **Different Vision Encoders**
   - EventGPT: Custom event-based CLIP encoder
   - Video-LLaVA: Standard CLIP vision tower

4. **Semantic Gap**
   - Same image → Different feature representations → Different token sequences

---

## Research Opportunity #1: Cross-Tokenizer Speculative Decoding

### Problem
Different tokenizers make direct token matching impossible.

### Proposed Solution
Train a **Token Alignment Adapter** that maps EventGPT token logits to Video-LLaVA token space.

```
EventGPT logits → Linear(32k, 32k) → Video-LLaVA token probabilities
```

### Expected Impact
- Acceptance rate: 3% → potentially 30-50%
- Theoretical speedup: 0.21x → 1.5-2x

### Research Questions
1. Can a linear mapping preserve semantic meaning across tokenizers?
2. What is the minimum training data needed for alignment?
3. Does vocabulary overlap (common words) provide a good starting point?

---

## Research Opportunity #2: Feature Alignment Adapter

### Problem
Vision features from EventGPT and Video-LLaVA are in different representation spaces.

### Proposed Solution
Train a **Feature Alignment Module**:

```
EventGPT vision features → Adapter → Video-LLaVA-compatible features
```

### Architecture Options
1. **Linear Projection**: Simple, fast, but may lose information
2. **MLP Adapter**: 2-layer MLP with residual connection
3. **Cross-Attention**: Attend to original + projected features
4. **Contrastive Learning**: Align features using paired data

### Expected Impact
- Enable feature caching for Video-LLaVA using EventGPT encoder
- Reduce Video-LLaVA Stage 3+4 overhead

### Research Questions
1. How similar are CLIP event features vs CLIP RGB features?
2. Can we achieve alignment without fine-tuning the main models?
3. What's the quality degradation from feature projection?

---

## Research Opportunity #3: Shared Decoder Head

### Problem
Both models use separate LLM backbones (LLaMA-7B vs Vicuna-7B).

### Proposed Solution
Train a **Shared Decoder Adapter** that enables one LLM to generate for both:

```
Vision Features → [Shared Adapter] → Single LLM → Output
```

### Implementation
1. Freeze both vision encoders
2. Train adapter to merge feature spaces
3. Use single LLM (either LLaMA or Vicuna) for generation

### Expected Impact
- 50% memory reduction (one LLM instead of two)
- Potentially higher throughput via batching

### Research Questions
1. Which LLM backbone is better: LLaMA or Vicuna?
2. How much quality is lost with a shared decoder?
3. Can we maintain task-specific output styles?

---

## Research Opportunity #4: Knowledge Distillation

### Problem
Video-LLaVA is verbose (~101 tokens) but EventGPT is concise (~41 tokens).

### Proposed Solution
Distill Video-LLaVA's knowledge into EventGPT:

```
EventGPT (student) ← Learn from → Video-LLaVA (teacher)
```

### Training Objective
1. **Output KL-Divergence**: Match token probability distributions
2. **Feature MSE**: Align intermediate representations
3. **Contrastive**: Same image → similar outputs

### Expected Impact
- EventGPT with Video-LLaVA quality
- Maintain EventGPT speed advantage
- Single model deployment

### Research Questions
1. Does distillation improve EventGPT's descriptiveness?
2. How much data is needed for effective distillation?
3. Can we distill only specific capabilities (e.g., scene understanding)?

---

## Research Opportunity #5: Hybrid Speculative Decoding

### Problem
Current speculative decoding uses token matching which fails at 3.11%.

### Proposed Solution
**Semantic Speculative Decoding**:
1. EventGPT generates draft tokens
2. Convert draft to semantic representation
3. Video-LLaVA verifies semantic meaning (not exact tokens)
4. Accept if semantically equivalent

### Implementation
```python
# Instead of: if draft_token == target_token
# Use: if semantic_similarity(draft_phrase, target_phrase) > threshold
```

### Expected Impact
- Higher acceptance rate based on meaning
- Novel contribution to speculative decoding literature

### Research Questions
1. How to efficiently compute semantic similarity during generation?
2. What similarity threshold balances speed and quality?
3. Can we use embedding distance as a proxy?

---

## Research Opportunity #6: Stage 1 Optimization (EventGPT-Specific)

### Observation
EventGPT spends 50.2% of time in Stage 1 (Loading) vs Video-LLaVA's 0.6%.

### Root Cause
EventGPT has heavy per-sample preprocessing:
- Loading event images from disk
- Event tensor conversion
- Multiple images per sample (5 images, but only 1 used)

### Proposed Solutions
1. **Prefetching**: Load next batch while processing current
2. **Memory Mapping**: Use mmap for large event files
3. **Caching**: Store preprocessed tensors in shared memory
4. **Async Loading**: DataLoader with multiple workers

### Expected Impact
- EventGPT Stage 1: 0.98s → potentially 0.1s
- Overall speedup: 1.01x → 2.0x+

---

## Research Opportunity #7: Quantization Study

### Observation
Both models use 16-bit precision (bfloat16/float16).

### Proposed Study
Benchmark impact of quantization on:
- INT8 (8-bit)
- INT4 (4-bit)
- NF4 (4-bit normalized float)

### Metrics to Measure
1. Inference speed improvement
2. Output quality (BLEU, semantic similarity)
3. Memory reduction
4. Token accuracy preservation

### Expected Impact
- 2-4x speedup with INT4
- 50-75% memory reduction
- Enable larger batch sizes

---

## Priority Ranking

| Priority | Opportunity | Effort | Impact | Novelty |
|----------|------------|--------|--------|---------|
| 1 | Stage 1 Optimization | Low | High | Low |
| 2 | Quantization Study | Low | Medium | Low |
| 3 | Feature Alignment Adapter | Medium | High | Medium |
| 4 | Knowledge Distillation | Medium | High | Low |
| 5 | Cross-Tokenizer Spec. Decoding | High | High | High |
| 6 | Semantic Speculative Decoding | High | High | Very High |
| 7 | Shared Decoder Head | High | Medium | Medium |

---

## Recommended Next Steps

### Immediate (This Week)
1. Implement async data loading for EventGPT
2. Run quantization benchmark (INT8, INT4)

### Short-term (1-2 Weeks)
3. Train a simple linear feature alignment adapter
4. Measure aligned feature quality

### Medium-term (1 Month)
5. Implement cross-tokenizer speculative decoding
6. Knowledge distillation training

### Research Paper Potential
- **Novel Contribution**: Semantic Speculative Decoding
- **Conference Target**: NeurIPS, ICLR (efficiency track)
- **Title Ideas**:
  - "Semantic Speculative Decoding for Cross-Tokenizer Acceleration"
  - "Bridging Vision-Language Models: Feature Alignment for Speculative Inference"

---

## Conclusion

The low token acceptance rate (3.11%) presents both a challenge and a research opportunity. The fundamental mismatch between EventGPT and Video-LLaVA tokenizers/outputs means direct speculative decoding won't work, but opens doors for:

1. **Novel decoding strategies** that work across different tokenizers
2. **Feature/representation alignment** methods for vision-language models
3. **Engineering optimizations** that can provide immediate speedups

The Stage 1 loading bottleneck in EventGPT is the lowest-hanging fruit for immediate performance gains.
