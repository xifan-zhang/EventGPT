# EventGPT Feasible Folder - Changelog

---

## [2026-01-25] Parallel Prefill & Speculative Decoding Research

### Added

**New Scripts:**
- `benchmark_inference_5stages.py` - Extended 5-stage benchmark (adds Stage 5: speculative verification)
- `parallel_prefill_speculative.py` - Parallel prefill implementation for EventGPT + Video-LLaVA
- `benchmark_parallel_prefill.py` - Benchmark for parallel prefilling approach

**New Documentation:**
- `PARALLEL_PREFILL_RESEARCH.md` - Survey of parallel prefilling techniques
- `EXPLOITING_FAST_PREFILL.md` - Strategy for leveraging EventGPT's fast vision encoding
- `SPECULATIVE_DECODING_RESEARCH.md` - Cross-modal speculative decoding opportunities

**Research Findings:**
- Token-level acceptance rate ~2-5% (too low due to tokenizer mismatch)
- Feature-level speculation recommended to bypass tokenizer issues
- Cross-modal speculative prefill is unexplored research opportunity
- Expected 2-3x end-to-end speedup with feature-level approach

### Changed
- Updated `benchmark_inference_properly_decoupled.py` with improved timing

---

## [2026-01-24] 4-Stage Benchmark & Decoupling Analysis

### Added

**New Scripts:**
- `benchmark_inference_4stages.py` - Proper Stage 3+4 decoupling (vision vs LLM)
- `analyze_stage4.py` - Stage 4 bottleneck analysis
- `analyze_stage4_5_shapes.py` - Shape analysis for cached features
- `analyze_1frame_vs_8frames.py` - EventGPT (1 frame) vs Video-LLaVA (8 frames) comparison
- `benchmark_alpha.py` - Acceptance rate (alpha) calculation

**New Documentation:**
- `FINAL_BENCHMARK_REPORT.md` - Comprehensive 200-sample benchmark results
- `STAGE4_ANALYSIS.md` - LLM decoding bottleneck analysis
- `STAGE4_5_SHAPE_ANALYSIS.md` - Feature tensor shape documentation
- `COMPLETE_DECOUPLING_STRATEGY.md` - Full decoupling implementation guide
- `DECOUPLING_SOLUTION_SUMMARY.md` - Summary of decoupling approach
- `VIDEOLLAVA_DECOUPLING_VERIFICATION.md` - Video-LLaVA specific analysis
- `4STAGE_DATA_FLOW_ANALYSIS.md` - Data flow through 4 stages
- `EVENTGPT_1FRAME_VS_VIDEOLLAVA_8FRAMES.md` - Frame count comparison
- `BENCHMARK_1FRAME_VS_8FRAMES_20260124_*.md` - Timestamped benchmark reports
- `benchmark_inference_1s_dataset_20260124.md` - Dataset-specific benchmarks

### Key Results

**Stage Breakdown (200-sample average):**
```
Stage 1 (Load):        0.0079s  (0.8%)   ✅ Fast
Stage 2 (Preprocess):  0.0167s  (1.6%)   ✅ Fast
Stage 3 (Vision):      0.0066s  (0.6%)   ✅ Optimized
Stage 4 (LLM):         1.0007s  (97.0%)  🔴 BOTTLENECK
─────────────────────────────────────────────────────
TOTAL PER SAMPLE:      1.0318s  (100%)
```

**Primary Finding:** LLM decoding dominates at **97.0%** of inference time

### Changed
- Modified `EventChatModel.py` to support cached event features in `generate()`
- Updated `benchmark_inference_4stages.py` with Video-LLaVA as default

---

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
│   ├── benchmark_inference_3stages.py (new)
│   ├── benchmark_inference_4stages.py (new)
│   ├── benchmark_inference_5stages.py (new)
│   ├── benchmark_inference_properly_decoupled.py (new)
│   ├── parallel_prefill_speculative.py (new)
│   ├── benchmark_parallel_prefill.py (new)
│   ├── analyze_stage4.py (new)
│   ├── analyze_stage4_5_shapes.py (new)
│   ├── analyze_1frame_vs_8frames.py (new)
│   ├── benchmark_alpha.py (new)
│   ├── speculative_decoding_S1.py (new)
│   ├── shared_decoder_speculative_S1.py (new)
│   ├── verify_videollava_decoupling.py (new)
│   ├── egpt_faster_analysis.md (new)
│   ├── FINAL_BENCHMARK_REPORT.md (new)
│   ├── COMPLETE_DECOUPLING_STRATEGY.md (new)
│   ├── PARALLEL_PREFILL_RESEARCH.md (new)
│   ├── SPECULATIVE_DECODING_RESEARCH.md (new)
│   └── ... (20+ additional .md files)
└── CHANGELOG.md (this file)

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

**Last Updated:** 2026-01-25 00:55 UTC
