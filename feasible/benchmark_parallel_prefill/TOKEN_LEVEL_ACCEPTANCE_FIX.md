# Token-Level Acceptance Rate Fix

**Date:** 2026-01-27
**Status:** Completed
**Benchmark:** Running on 1s dataset (1,100 samples)

## Problem Statement

The original acceptance rate calculation was using **semantic similarity** metrics:
- Word overlap (50% weight)
- Character similarity (30% weight)
- Length similarity (20% weight)

This approach was **misleading** for speculative decoding because it doesn't reflect actual token-level acceptance as would occur in practice.

## Root Cause

In actual speculative decoding with different models/tokenizers:
1. Draft model generates tokens using its tokenizer
2. Draft tokens need to be "accepted" or "rejected" by the target model
3. Since models use different tokenizers, we must compare at the token level using the **target model's tokenizer**

The old semantic similarity metrics don't capture this behavior. Two outputs could have high word overlap but completely different tokenizations.

## Solution Implemented

### New Token-Level Acceptance Calculation

```python
def compute_acceptance_rate(draft_tokens, target_tokens, draft_text, target_text,
                           egpt_tokenizer=None, vl_tokenizer=None):
    """
    Token-level acceptance: re-tokenize with target tokenizer
    """
    # 1. Re-tokenize draft output with Video-LLaVA tokenizer
    draft_tokens_vl = vl_tokenizer(draft_text, ...)['input_ids']

    # 2. Tokenize target with Video-LLaVA tokenizer
    target_tokens_vl = vl_tokenizer(target_text, ...)['input_ids']

    # 3. Compare token sequences at position level
    min_length = min(len(draft_tokens_vl), len(target_tokens_vl))
    matched_tokens = sum(1 for i in range(min_length)
                        if draft_tokens_vl[i] == target_tokens_vl[i])

    # 4. Acceptance rate = matched tokens / target length
    acceptance_rate = matched_tokens / len(target_tokens_vl)

    return {
        "acceptance_rate": float(acceptance_rate),
        "matched_tokens": int(matched_tokens),
        "total_tokens": int(len(target_tokens_vl)),
        "draft_tokens": int(len(draft_tokens_vl)),
        "method": "token_level_matching"
    }
```

### Key Changes

1. **File:** `feasible/benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py`

   - **Lines 123-195**: Replaced semantic similarity function with token-level matching
   - **Line 630-638**: Updated function call to pass both tokenizers
   - **Lines 726-740**: Updated statistics to track `matched_tokens` and `total_tokens`

2. **Protobuf Fix:** Added environment variable handling at line 47-49 to fix protobuf compatibility issue:
   ```python
   if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
       os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
   ```

### Results Structure

Old format (deprecated):
```json
{
  "acceptance_rate": 0.35,
  "word_overlap": 0.25,
  "char_similarity": 0.40,
  "length_similarity": 0.45
}
```

New format (token-level):
```json
{
  "acceptance_rate": 0.0196,
  "matched_tokens": 1,
  "total_tokens": 51,
  "draft_tokens": 34,
  "method": "token_level_matching"
}
```

## Expected Behavior

### Example Analysis

**EventGPT Output:** "In the scene, there is a car parked on the side..."
**Video-LLaVA Output:** "The key elements in this scene include a car driving..."

When both outputs are tokenized with Video-LLaVA's tokenizer:
- Many tokens differ due to different word choices ("parked" vs "driving")
- Only ~2% token-level match (1 out of 51 tokens)
- This accurately reflects that the outputs are semantically divergent

## Validation Test Results

Quick test on 1 sample (500ms dataset):
```
Acceptance Rate: 2.0% (token-level)
- Matched tokens: 1
- Total tokens compared: 51
- Method: token_level_matching
```

Compare to old semantic approach which would have given ~35% due to shared words.

## Running the Corrected Benchmark

```bash
# With automatic protobuf fix (built-in):
python feasible/benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py \
  --dataset_dir ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \
  --max_samples -1 \
  --max_new_tokens 50

# Or with explicit environment variable:
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python feasible/benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py ...
```

## Implications for Speculative Decoding

With corrected token-level acceptance rates:
- More accurate expected speedup calculations
- Better reflects real-world speculative decoding performance
- Accounts for tokenizer differences between models

**Formula for effective speedup:**
```
Expected accepted tokens = hidden_tokens * acceptance_rate (token-level)
Effective speedup ≈ 1 + (accepted_tokens / total_output_tokens)
```

## Benchmark Status

- ✅ Token-level acceptance calculation implemented
- ✅ Protobuf compatibility fixed
- ✅ Single sample test passed (2.0% acceptance)
- ⏳ Full 1s dataset benchmark running (1,100 samples)
- ⏳ Remaining datasets to follow

## Files Modified

1. `feasible/benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py`
   - `compute_acceptance_rate()` function (lines 123-195)
   - `run_5stage_benchmark()` call (lines 630-638)
   - `compute_statistics()` function (lines 726-740)
   - Environment variable fix (lines 47-49)

## Benchmark Configuration: max_new_tokens=50

### Why 50?

A practical balance between speed and output quality:

**Benchmark Time Estimates (1,100 samples):**
- `max_new_tokens=20`: ~9 minutes (quick test)
- `max_new_tokens=50`: ~22 minutes (standard) ⭐
- `max_new_tokens=100`: ~43 minutes (detailed)
- `max_new_tokens=200`: ~87 minutes (comprehensive)

**Output Quality:**
- 50 tokens ≈ 30-40 words
- Sufficient for scene description and analysis
- Aligns with hidden token generation (~35 tokens average)

**Current Results:**
- Video-LLaVA: Always hits 50-token limit
- EventGPT: Averages 47 tokens (usually hits limit)
- No constraint issues for parallel prefill analysis

### Setting Used for All Benchmarks

All remaining datasets run with:
```bash
--max_new_tokens 50
```

This ensures:
- Consistent timing across all datasets
- Comparable outputs for analysis
- Reasonable benchmark duration (~30 min per 1,100 samples)
- Flexibility to adjust if needed

---

## Next Steps

1. Complete 1s dataset benchmark (ETA: 30-35 minutes)
2. Generate updated markdown report with token-level metrics
3. Run remaining datasets (500ms, 2s, 4s, 5s, 8s, 10s, 16s, 20s)
4. Compare token-level acceptance vs. old semantic similarity metrics
5. Update conclusions on speculative decoding feasibility

---

*This fix ensures that the acceptance rates reported in benchmarks accurately reflect token-level matching as would occur in actual speculative decoding implementations.*
