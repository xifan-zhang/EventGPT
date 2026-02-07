# Feature Alignment Tasks for Speculative Decoding

## Problem Statement

Current feature adapter is **too heavy** (154M params vs 45M token adapter).
Need lightweight adapter that aligns hidden states, not raw images.

## Task List

### Phase 1: Lightweight Hidden State Adapter

- [ ] **Task 1.1**: Extract hidden states from EventGPT and Video-LLaVA
  - Input: Same samples used for token alignment
  - Output: `egpt_hidden_states.pt` [N, seq, 4096], `vl_hidden_states.pt` [N, seq, 4096]
  - Script: `extract_hidden_states.py`

- [ ] **Task 1.2**: Design lightweight bottleneck adapter
  - Architecture: Linear(4096→256) + GELU + Linear(256→4096)
  - Parameters: ~2M (vs 154M current, vs 45M token adapter)
  - Add residual connection: `output = input + adapter(input)`

- [ ] **Task 1.3**: Train hidden state adapter
  - Loss: MSE + Cosine similarity
  - Target: cosine similarity > 0.9 on validation set
  - Script: `train_hidden_adapter.py`

### Phase 2: Feature-Level SD Acceptance Measurement

- [ ] **Task 2.1**: Create acceptance measurement script
  - Script: `measure_feature_acceptance.py`
  - Metrics:
    - Per-position cosine similarity
    - Acceptance rate at different thresholds (0.8, 0.85, 0.9, 0.95)
    - Consecutive acceptance rate (for comparison with token-level)
    - Non-consecutive acceptance rate

- [ ] **Task 2.2**: Compare with token-level baseline
  - Token-level consecutive: 0% (from benchmark)
  - Feature-level consecutive: ?
  - Feature-level relaxed: ?

### Phase 3: End-to-End SD Benchmark

- [ ] **Task 3.1**: Implement feature-level speculative decoding
  - Modify SD verification to use hidden state comparison
  - Add threshold parameter for acceptance
  - Script: `feature_sd_inference.py`

- [ ] **Task 3.2**: Measure actual speedup
  - Baseline: VL-only inference time
  - With feature SD: EGPT draft + adapter + VL verify
  - Account for adapter overhead

- [ ] **Task 3.3**: Compare speedup formulas
  - Token-level: `E[accepted] = (1 - α^(γ+1)) / (1 - α)` (consecutive only)
  - Feature-level consecutive: same formula
  - Feature-level relaxed: `E[accepted] = γ × acceptance_rate`

## Target Metrics

| Metric | Token-Level (Current) | Feature-Level (Target) |
|--------|----------------------|------------------------|
| Adapter Params | 45M | **< 5M** |
| Adapter Overhead | ~1ms | **< 0.5ms** |
| Cosine Similarity | N/A | **> 0.9** |
| Consecutive Accept | 0% | **> 30%** |
| Relaxed Accept | N/A | **> 50%** |
| Speedup | 1.0x | **> 1.3x** |

## Speedup Formula for Feature-Level SD

### Consecutive Mode (like token-level)
```
E[tokens] = (1 - α^(γ+1)) / (1 - α)
Speedup = E[tokens] / (1 + overhead)

Where:
- α = consecutive acceptance rate
- γ = draft length
- overhead = (egpt_time + adapter_time) / vl_step_time
```

### Relaxed Mode (feature-level specific)
```
E[tokens] = γ × acceptance_rate
Speedup = E[tokens] / (1 + overhead + rejection_cost)

Where:
- acceptance_rate = fraction of positions with sim > threshold
- rejection_cost = time to recompute rejected positions
```

### Break-even Analysis
```
For speedup > 1.0:
  acceptance_rate > (1 + overhead) / γ

Example with γ=5, overhead=0.2:
  acceptance_rate > 1.2 / 5 = 24%

For speedup > 1.5:
  acceptance_rate > 1.5 × (1 + overhead) / γ
  acceptance_rate > 1.5 × 1.2 / 5 = 36%
```

## Files to Create

```
feature_alignment/
├── TASKS.md                      # This file
├── extract_hidden_states.py      # Task 1.1
├── hidden_adapter.py             # Task 1.2
├── train_hidden_adapter.py       # Task 1.3
├── measure_feature_acceptance.py # Task 2.1
├── feature_sd_inference.py       # Task 3.1
└── benchmark_feature_sd.py       # Task 3.2, 3.3
```

## Priority

1. **High**: Task 1.1, 1.2, 2.1 (understand if feature-level is viable)
2. **Medium**: Task 1.3, 2.2 (train and compare)
3. **Low**: Task 3.x (end-to-end only if Phase 1-2 show promise)

## Key Insight

Feature-level SD may work where token-level failed because:

| Scenario | Token-Level | Feature-Level |
|----------|-------------|---------------|
| "street" vs "road" | REJECT (different tokens) | ACCEPT (similar embeddings) |
| "car driving" vs "vehicle moving" | REJECT | Likely ACCEPT |
| "parked" vs "stationary" | REJECT | Likely ACCEPT |

The semantic similarity captured in hidden states may bridge the gap that token-level alignment cannot.
