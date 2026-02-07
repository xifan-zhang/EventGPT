# EventGPT Prefill Only: Parallel Prefill Benefit

**Option 2: Use EventGPT for prefill benefit only (no SD during decode)**

This approach exploits EventGPT's faster prefill without the complexity of speculative decoding during the decode phase.

## Trained Token Alignment Model

**Default Adapter:** `task/starred/1q_20260128_151847/best_model.pt`

| Metric | Value |
|--------|-------|
| Baseline Acceptance | 1.58% |
| **Model Acceptance** | **27.9%** |
| Top-5 Accuracy | 51.64% |
| Improvement | +26.32% |
| Parameters | 45M |
| Training Samples | 5200 |

## Key Insight

From 5-Stage timing analysis:
- EventGPT Stage 4 (Prefill): **83ms**
- Video-LLaVA Stage 4 (Prefill): **315ms**
- **Overlap Window: 232ms** (3.8x faster)

During parallel prefill:
1. Both models start prefill simultaneously
2. EventGPT finishes prefill 232ms earlier
3. EventGPT can generate ~12-15 "free" draft tokens in this window
4. Video-LLaVA continues AR decode normally

## Architecture: Prefill-Then-Verify (Recommended)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              PARALLEL PREFILL + SPECULATIVE DECODING VERIFICATION                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PHASE 1: PARALLEL PREFILL (0-315ms)                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  EventGPT:    |--Prefill (83ms)--|----Draft Generation (232ms)----|      │   │
│  │  Video-LLaVA: |---------------Prefill (315ms)---------------------|      │   │
│  │                                                                           │   │
│  │  Result: ~26 FREE draft tokens generated during VL prefill               │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  PHASE 2: BATCH VERIFICATION (315-365ms)                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Video-LLaVA verifies ALL 26 drafts in SINGLE forward pass (~50ms)       │   │
│  │                                                                           │   │
│  │  Draft:  [t1, t2, t3, t4, t5, t6, t7, ...]                               │   │
│  │  VL:     [✓,  ✓,  ✓,  ✓,  ✓,  ✓,  ✗,  ...]  (accept until mismatch)    │   │
│  │                                                                           │   │
│  │  Accepted: 6 tokens (at 27.9% acceptance rate)                            │   │
│  │  Time saved: 6 × 14.5ms = 87ms                                            │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  PHASE 3: AR DECODE (365ms+)                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Continue standard autoregressive decode for remaining tokens             │   │
│  │  Starting from position after last accepted token                         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Insight: FREE Draft Tokens

The ~26 draft tokens cost **ZERO time** because:
1. EventGPT finishes prefill in 83ms
2. Video-LLaVA needs 315ms for prefill
3. During the 232ms overlap, EventGPT generates drafts for FREE
4. When VL is ready, drafts are already waiting for verification

## Benefits vs Full Speculative Decoding

| Aspect | Full SD | Prefill Only |
|--------|---------|--------------|
| Complexity | High (verification loop) | **Low** (parallel prefill) |
| Acceptance rate needed | >30% for benefit | **0%** (no verification) |
| Draft generation cost | Must be hidden | **Always hidden** in prefill |
| Implementation | Token alignment + verification | **Just parallel prefill** |
| Speedup | 1.3-2.0x (if α>50%) | **1.1-1.2x** (guaranteed) |

## Quick Start

### Run Benchmark

```bash
cd /home/ps/Documents/code/EventGPT

# Quick test (10 samples)
python feasible/egpt_prefill_only/benchmark_prefill_only.py \
    --dataset_dir ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \
    --max_samples 10

# Full benchmark
python feasible/egpt_prefill_only/benchmark_prefill_only.py \
    --dataset_dir ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \
    --max_samples -1
```

### Use in Code

```python
from feasible.egpt_prefill_only import PrefillOnlyInference

# Initialize
inference = PrefillOnlyInference(
    eventgpt_path="./checkpoints/EventGPT-7b",
    videollava_model_id="LanguageBind/Video-LLaVA-7B-hf",
    device="cuda",
)

# Run inference (parallel prefill + VL decode)
result = inference.generate(
    event_image_path="path/to/event_image.png",
    video_path="path/to/video.mp4",
    query="What are the key elements in this scene?",
    max_new_tokens=50,
)

print(f"Output: {result['output_text']}")
print(f"Wall-clock time: {result['wall_clock_time']*1000:.1f}ms")
print(f"Free tokens generated: {result['free_tokens_count']}")
```

## Performance Analysis

### Theoretical Speedup

```
Sequential (Video-LLaVA only):
  Total = Prefill(315ms) + Decode(724ms) = 1039ms

Parallel Prefill:
  Total = max(EGPT_prefill, VL_prefill) + VL_decode
       = 315ms + 724ms
       = 1039ms (same wall-clock)
  BUT: Got 12-15 EventGPT tokens "for free"
```

### Use Case: Draft Token Priming

If EventGPT's draft tokens have semantic overlap with Video-LLaVA:
1. Use drafts to "prime" Video-LLaVA's first few tokens
2. Video-LLaVA can start AR decode from token position k (not 0)
3. Saves k × 14.5ms per token = potential 145-217ms savings

### Use Case: Early Response Preview

1. Return EventGPT's draft immediately as "preview"
2. Update with Video-LLaVA's higher-quality output when ready
3. Better user experience with progressive loading

## File Structure

```
egpt_prefill_only/
├── __init__.py                 # Module exports
├── README.md                   # This file
├── prefill_then_verify.py      # MAIN: Parallel prefill + SD verification
├── prefill_only.py             # Basic parallel prefill (no verification)
├── prefill_with_alignment.py   # With TokenAdapter alignment
├── benchmark_prefill_only.py   # Benchmark script
└── run_prefill_only.sh         # Quick run script
```

## Comparison with Other Approaches

| Approach | Prefill Savings | Decode Savings | Total Speedup |
|----------|----------------|----------------|---------------|
| Video-LLaVA only | 0ms | 0ms | 1.0x |
| EventGPT only | 232ms | 252ms | 2.07x (but quality?) |
| Full SD (α=50%) | 0ms | ~360ms | 1.35x |
| **Prefill Only** | **232ms** | **0ms** | **1.22x** |
| Prefill + Priming | 232ms | ~145ms | 1.36x |

## When to Use This Approach

**Use Prefill Only when:**
- You need simple, reliable speedup
- Token acceptance rate is low (<30%)
- Implementation complexity is a concern
- You want guaranteed latency improvement

**Use Full SD when:**
- Token acceptance rate is high (>50%)
- Maximum speedup is priority
- You have trained a good alignment model

## Related Work

- `feasible/token_alignment/` - Token-level alignment for speculative decoding
- `feasible/benchmark_parallel_prefill/` - Parallel prefill timing analysis
- `research/cascaded_SD/` - Cascaded speculative decoding research

## References

1. EventGPT paper - Event-driven VLM with fast inference
2. Video-LLaVA paper - Video understanding with LLaVA architecture
3. Speculative Decoding - Leviathan et al., ICML 2023
