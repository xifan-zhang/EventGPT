# EventGPT Feasible Folder - Changelog

## [2026-01-23] Benchmark Infrastructure Updates

### Changes Made

#### 1. Modified `benchmark_inference/benchmark_inference.py`

**Datetime & Sample Tracking:**
- Added `from datetime import datetime` import
- Output JSON now includes `"benchmark_datetime"` field (ISO format)
- Output filename now includes sample count: `*_results_n{num_samples}_{datetime}.json`
  - Example: `EventGPT_Instruction_Subset_results_n600_20260123_223000.json`

**Model Naming Convention:**
- Changed video model key from `"llava-1.5-7b-hf"` to `"llava"` or `"videollava"`
- Automatically detects which model was used:
  - `--use_video_llava` (default) → all keys use `"videollava"`
  - `--use_llava15` → all keys use `"llava"`
- Distinguishes in JSON output:
  - `videollava`, `videollava_time`, `videollava_stage1_time`, etc.
  - `llava`, `llava_time`, `llava_stage1_time`, etc.

**Default Configuration:**
- `--use_video_llava` now defaults to `True` (was `False`)
- `--use_llava15` now defaults to `False` (was `True`)
- `--use_event_image` now defaults to `True` (unchanged)
- New default behavior: Video-LLaVA + preprocessed event images + MP4

#### 2. Created `benchmark_inference/egpt_faster_analysis.md`

**Content:**
- Comprehensive analysis of why EventGPT is 5.1x faster than LLaVA
- Full 3-stage timing breakdown with detailed explanations
- Token generation analysis (EventGPT: 45.5 avg, LLaVA: 102.4 avg)
- Proposed Stage 3→3+4 split architecture
- Implementation locations for vision encoding vs LLM decoding
- Recommendations for optimization

---

## Benchmark Results (1s Test Set, 200 Samples)

### Performance Metrics

**Overall Speedup:**
- EventGPT: **1.851s** total time
- LLaVA: **9.464s** total time
- **Speedup: 5.1x**

**Stage 1 (Data Loading):**
- EventGPT (event images): 0.851s
- LLaVA (MP4 video): 6.858s
- Speedup: **8.1x** (73% of overall speedup)

**Stage 3 (Generation):**
- EventGPT: 0.977s (45.5 tokens)
- LLaVA: 2.533s (102.4 tokens)
- Speedup: **2.6x** (27% of overall speedup)
- Reason: EventGPT generates 56% fewer tokens

---

## Test Runs Performed

### Run 1: 200 samples (1s duration)
- Status: ✅ Complete
- Date: 2026-01-23 00:45 UTC
- Models: EventGPT + LLaVA 1.5
- Results: `/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_test/my_egpt_dsec_seq_1s/benchmark_results.json`

### Run 2: 600 samples (1s duration) - Video-LLaVA
- Status: ❌ Killed (slow progress: ~80s/sample)
- Date: 2026-01-23 22:00 UTC
- Bottleneck: MP4 video codec operations in Stage 1
- Estimated completion: 12+ hours

---

## Known Issues & TODOs

### Blocking Issues
- [ ] Stage 3 split not implemented
  - Need to extract vision encoding from `model.generate()`
  - Requires custom decoding loop implementation
  - Affects: EventGPT, Video-LLaVA, LLaVA 1.5

### Performance Issues
- [ ] Video-LLaVA Stage 1 is 80s/sample due to MP4 decoding
  - Consider preprocessing videos to frame sequences
  - Or cache extracted frames for repeated use

### Future Enhancements
- [ ] Implement Stage 3→3+4 split for all models
- [ ] Add acceptance rate calculation to output JSON
- [ ] Add memory profiling per stage
- [ ] Optimize MP4 handling (frame caching, parallel decoding)

---

## Files Modified

```
feasible/
├── benchmark_inference/
│   ├── benchmark_inference.py (modified)
│   └── egpt_faster_analysis.md (new)
└── CHANGELOG.md (this file, new)
```

---

## Running Benchmarks

### Full dataset (all samples)
```bash
python benchmark_inference.py \
    --dataset_dir /path/to/1s_test \
    --use_video_llava \
    --use_event_image \
    --warmup_steps 2 \
    --device cuda
```

### Limited samples (recommended for testing)
```bash
python benchmark_inference.py \
    --dataset_dir /path/to/1s_test \
    --use_video_llava \
    --use_event_image \
    --max_samples 100 \
    --warmup_steps 2 \
    --device cuda
```

### With LLaVA 1.5 (instead of Video-LLaVA)
```bash
python benchmark_inference.py \
    --dataset_dir /path/to/1s_test \
    --use_llava15 \
    --use_event_image \
    --max_samples 100 \
    --warmup_steps 2 \
    --device cuda
```

---

## Output Format Example

```json
{
  "id": "sample_001",
  "split": "test",
  "query": "What are the key elements in this scene?",
  "benchmark_datetime": "2026-01-23T22:30:15.123456",
  "event_data": "interlaken_00_a/000000.npy",
  "egpt": "In the scene, there is...",
  "egpt_time": 1.851,
  "egpt_stage1_time": 0.851,
  "egpt_stage2_time": 0.023,
  "egpt_stage3_time": 0.977,
  "egpt_token_ids": [1, 512, 2048, ...],
  "video_data": "interlaken_00_a",
  "videollava": "The scene captures...",
  "videollava_time": 9.464,
  "videollava_stage1_time": 6.858,
  "videollava_stage2_time": 0.072,
  "videollava_stage3_time": 2.533,
  "videollava_token_ids": [1, 256, 4096, ...]
}
```

---

## Next Steps

1. **Performance optimization**
   - Implement Stage 3→3+4 split
   - Profile vision encoding vs LLM decoding separately

2. **Data handling**
   - Consider MP4 caching/preprocessing for Video-LLaVA
   - Test on larger sample sizes (600, 1100)

3. **Analysis**
   - Compare EventGPT vs Video-LLaVA (not LLaVA 1.5)
   - Measure acceptance rates for speculative decoding
   - Profile memory usage per stage

---

**Last Updated:** 2026-01-23 22:30 UTC
