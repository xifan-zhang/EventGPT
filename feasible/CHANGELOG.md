# EventGPT Feasible Folder - Changelog

---

## [2026-02-06] Feature Alignment: L5 EAGLE Adapter + Memory-Efficient Training

### Added

**L5 EAGLE-Style Adapter (`hidden_adapter.py`):**
- `EAGLEStyleAdapter` class with dual training objectives
  - Alignment loss: MSE + cosine (same position, EventGPT → Video-LLaVA)
  - Prediction loss: MSE(predicted[t], target[t+1]) (next-position prediction)
  - Causal self-attention + FFN + positional encoding
  - `speculative_decode()` method for autoregressive draft generation
- ~100M params, configurable num_layers/num_heads/ffn_dim
- Updated `create_adapter(level=5)` and `load_any_adapter()` for L5

**Memory-Efficient Training (`train_hidden_adapter.py`):**
- `ChunkedTrainLoader`: streams 1 chunk (~1.6GB) at a time instead of loading all 80GB
  - Shuffles chunk order + shuffles within each chunk
  - Frees memory after each chunk processed
- `ChunkedValLoader`: streams all val chunks sequentially for evaluation
  - Evaluates on **full 11,000 test samples** (was limited to 2,000)
  - ~1.6GB RAM per chunk instead of 17GB for all chunks
- Acceptance rate tracking: val_accept@0.80/0.85/0.90/0.95 per epoch
- 3-panel training plots: loss, cosine similarity, acceptance rates
- Comprehensive config.json saved with each run (adapter, data, training params)

**Auto-Training Pipeline (`auto_train_pipeline.sh`):**
- Trains L1→L5 sequentially with evaluation after each adapter
- Waits for extraction completion before starting
- Saves to `feasible/feature_alignment/tasks/L{N}/`
- Logs to `feasible/feature_alignment/logs/`

**Analysis Documents:**
- `EMBEDDING_SD_THRESHOLD_ANALYSIS.md` - Literature survey of cosine similarity thresholds
  - Synthesized from EAGLE, EAGLE-3, Medusa, ASD, HASS, SpecVLM papers
  - Key finding: cos_sim > 0.85 is "practical barrier" for useful SD speedup
- `FEATURE_SD_REPORT.md` - Full technical report on feature-level SD approach

### Fixed
- **OOM crash**: 80GB train data no longer loaded into 62GB RAM (ChunkedTrainLoader)
- **OOM crash**: 17GB val data no longer loaded all at once (ChunkedValLoader)
- **Val subset bias**: Now evaluates on all 11,000 test samples, not just 2,000

### Current Training Status
- L1 training in progress (epoch 1/50, auto-pipeline running)
- Train: 52,000 samples (1s train split, EventGPT + Video-LLaVA hidden states)
- Val: 11,000 samples (1s test split, full streaming evaluation)

---

## [2026-02-06] Cross-Modal vs SpecVLM Research Analysis

### Added

**Research Documentation (`research/` folder):**
- `CROSSMODAL_VS_SPECVLM.md` - Comprehensive cross-modal vs SpecVLM analysis
- EAGLE-2 draft model architecture documentation
- SpecVLM (EagleVLM + compression) overview
- Why cross-modal outperforms same-modality baselines
- T_prefill / T_total benchmark analysis
- 7 experimental designs with code
- Implementation TODO checklist

### Key Findings

**Benchmark T_prefill / T_total Ratios:**
| Scenario | T_prefill | T_total | Ratio |
|----------|-----------|---------|-------|
| Video-LLaVA (8f, 5 tok) | 598ms | 742ms | **80.6%** |
| EventGPT (1f, 5 tok) | 72ms | 165ms | 43.7% |
| EventGPT (1f, 45 tok) | 31ms | 1032ms | 3.0% |

**Decision Framework:**
- T_prefill / T_total > 40% → Cross-modal wins
- Video-LLaVA: 80.6% >> 40% → **IDEAL for cross-modal!**

**SpecVLM Limitation:** Draft is "parasitic" (needs target features) → cannot parallelize

**Cross-Modal Advantage:** Draft is "independent" (own encoder) → parallel prefill → 4-5x speedup

### Implementation TODO

- [ ] Implement EAGLE-style decoding on VideoLLaVA (EagleVLM baseline)
- [ ] Add visual token compression (FastV, TokenPacker)
- [ ] Compare baselines: EagleVLM only, compression only, combined
- [ ] Implement cross-modal parallel prefill
- [ ] Run 7 designed experiments

---

## [2026-01-31] Feature Alignment: Chunked Incremental Saving

### Added

**Chunked Incremental Saving (Memory Efficient):**
- `ChunkedHiddenStateWriter` class for incremental saving
- Saves every N samples to separate chunk files (default: 1000)
- Each sample moved to CPU immediately after extraction
- Auto-resume via `index.json` tracking
- `load_chunked_hidden_states()` function to load and concatenate all chunks

**New Arguments:**
- `--chunked` - Enable chunked incremental saving mode
- `--chunk_size N` - Samples per chunk (default: 1000)
- `--quant TAG` - Quantization tag for filename (default: `4bit`)
- `--save_interval N` - Checkpoint interval for batch mode (default: 1000)

**Chunked Output Structure:**
```
hidden_states/chunked_train_1s_4bit/
├── chunks/
│   ├── chunk_000000.pt  # samples 0-999
│   ├── chunk_001000.pt  # samples 1000-1999
│   └── ...
└── index.json           # metadata and chunk tracking
```

**Extraction Robustness (Batch Mode):**
- Signal handlers for emergency save on SIGTERM/SIGHUP/SIGINT
- CPU tensor creation to avoid GPU OOM during save
- Try/except around saves to prevent crashes

### Changed

**`feature_alignment/extract_hidden_states.py`:**
- Added `--chunked` and `--chunk_size` arguments
- Dual mode support: chunked (incremental) or batch (accumulate in memory)
- Tensors moved to CPU before stacking: `egpt_h.cpu().float()`
- Save interval increased to 1000 (was 100)
- Added `ChunkedHiddenStateWriter` class (~100 lines)
- Added `load_chunked_hidden_states()` loader function

### Fixed
- **OOM on save**: Tensors now created on CPU, not GPU
- **Lost data on crash**: Chunked mode saves every 1000 samples automatically
- **Memory pressure**: Each sample moved to CPU immediately in chunked mode

### Current Status
- Train extraction: 🔄 Running in chunked mode
- Output: `hidden_states/chunked_train_1s_4bit/`
- Both models 4-bit quantized (~8GB VRAM)
- Chunk size: 1000 samples

---

## [2026-01-30] Feature Alignment: L1-L4 Adapter Architectures

### Added

**Multi-Level Adapter Implementations (`hidden_adapter.py`):**
- **L1: HiddenStateAdapter** - Simple bottleneck (2.1M params, ~1.5ms)
  - LayerNorm → Down(256) → GELU → Up(4096) + Residual
- **L2: MultiLayerBottleneckAdapter** - 3× stacked bottlenecks (6.3M params, ~4ms)
- **L3: WideBottleneckAdapter** - Wide bottleneck 1024 (16.8M params, ~10ms)
- **L4: AttentionAdapter** - Self-attention + FFN (100M params, ~56ms)

**Factory Functions:**
- `create_adapter(level=1-4, ...)` - Unified adapter creation
- `load_any_adapter(path)` - Auto-detect and load any checkpoint type

**Training Support:**
- `--adapter_level {1,2,3,4}` argument in `train_hidden_adapter.py`
- Level-specific arguments: `--num_blocks`, `--num_heads`, `--num_layers`, `--ffn_dim`
- Output directory format: `L{level}_{timestamp}/`

**Documentation:**
- Complete ASCII architecture diagrams for L1-L4
- Academic citations (LoRA, Adapter, Adapter-v2, EAGLE)
- Comprehensive bash command reference

### Changed
- README.md completely rewritten with full architecture documentation
- Added per-level training commands and usage examples

---

## [2026-01-29] Token Alignment: Benchmark Integration + 10-Question Workflow

### Added

**Benchmark Integration - TokenAdapter Support:**
- `benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py`:
  - Added `--use_token_adapter` flag for aligned evaluation
  - Added `--token_adapter_path` option (auto-detects latest checkpoint)
  - New `compute_aligned_acceptance_rate()` function
  - Reports both baseline and aligned acceptance metrics
  - Updated statistics computation for aligned metrics

- `benchmark_inference/benchmark_inference_5stages.py`:
  - Added same `--use_token_adapter` and `--token_adapter_path` flags
  - Added `compute_aligned_acceptance_rate()` function
  - Integrated aligned evaluation into benchmark pipeline

**New Workflow Script:**
- `token_alignment/run_10q_workflow.sh` - 10-question workflow
  - Balanced option: more data than 1q, faster than 50q
  - Trains for full 50 epochs (no early stopping with `--early_stopping 999`)
  - Auto-waits for extraction if already running
  - Extracts both train and test tokens
  - Estimated ~20 hours total

### Changed

**Benchmark Output Format:**
- Results JSON now includes `use_token_adapter` and `token_adapter_path` in config
- Statistics include aligned metrics when TokenAdapter is used:
  - `aligned_acceptance_rate_avg`
  - `aligned_acceptance_rate_std`
  - `aligned_top5_rate_avg`
  - `aligned_improvement`

**Console Output:**
- Prints both baseline and aligned acceptance rates when TokenAdapter enabled
- Shows improvement percentage

### Usage

```bash
# Run benchmark with TokenAdapter (auto-detects latest checkpoint)
python feasible/benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py \
  --use_token_adapter \
  --max_samples 100

# Run 10-question workflow
nohup ./feasible/token_alignment/run_10q_workflow.sh > /tmp/workflow_10q.log 2>&1 &
```

### Current Status

- 10q train extraction: ✅ Complete (52,000 pairs)
- 10q test extraction: 🔄 Running
- 10q training: ⏳ Pending (50 epochs, no early stopping)

---

## [2026-01-28] Token Alignment: Single-Question Success + Parallel Extraction

### Results

**Single Question (1q) Training - SUCCESS:**
| Dataset | Samples | Baseline | TokenAdapter | Top-5 | Improvement |
|---------|---------|----------|--------------|-------|-------------|
| Train | 5,200 | 1.77% | 27.21% | 51.40% | +25.45% |
| Test | 1,100 | 1.58% | 27.90% | 51.64% | +26.32% |

- **Theoretical speedup:** 1.39x with α=27.9%, γ=5 draft tokens
- **Starred result:** `task/starred/1q_20260128_151847/`

### Added

**New Scripts:**
- `token_alignment/extract_tokens_parallel.py` - **NEW** Parallel extraction (both models loaded)
  - ~8GB VRAM (both models 4-bit quantized)
  - ~1.5-2x faster than sequential (avoids model load/unload)
  - Currently running 10-question extraction (52,080 pairs)

- `token_alignment/extract_tokens_train.py` - Sequential extraction (one model at a time)
  - Uses event_image (PNG) for EventGPT input
  - Uses mp4 (8 frames) for Video-LLaVA input
  - ~4GB VRAM, works on smaller GPUs

- `token_alignment/run_1q_workflow.sh` - Single question automated workflow
- `token_alignment/run_full_workflow.sh` - 50 questions workflow

- `token_alignment/top50_questions.json` - Top 50 DSEC questions for training

**New Features:**
- **Task Folder System**: Auto-creates timestamped folders for each training run
  - Format: `task/{task_name}_{YYYYMMDD}_{HHMMSS}/`
  - Contains: checkpoints, results, training curves
  - Starred folder for important results: `task/starred/`

- **Training Curves Visualization**: Saves plots after training
  - `training_curves.png` - Combined loss and accuracy
  - `loss_curve.png` - Training loss over epochs
  - `accuracy_curve.png` - Acceptance rates with baselines

- **Multi-Question Support**: Extract with top N questions
  - `--single_question` for 1 question (fast testing)
  - `--max_questions N` for top N questions

**New Documentation:**
- `token_alignment/README.md` - Comprehensive usage documentation
  - Quick start (1q ~3hrs, 50q ~72hrs)
  - Sequential vs parallel extraction options
  - TokenAdapter architecture diagram
  - Troubleshooting guide
  - Further directions for research

### Changed

**`token_alignment/train_and_evaluate.py`:**
- Fixed test data format auto-detection (extraction vs benchmark format)
- Fixed model saving: `best_val_acc = -1` ensures first epoch saves
- Added `--task_name` argument for custom task naming
- Auto-detects dataset duration from path (1s, 500ms, etc.)
- Added matplotlib training curves generation
- Added `from_extraction_json()` loader for new token format

**`token_alignment/extract_tokens_train.py`:**
- Fixed Video-LLaVA to save only generated tokens (was saving full sequence with prompt)
- Added multi-question support with `--single_question` flag
- Added `--max_questions` to limit number of questions

### Fixed
- **Val Acc=0% despite Train Acc=100%**: Data format mismatch between train/test
- **FileNotFoundError: best_model.pt**: Model not saving when val_acc started at 0
- **Video-LLaVA tokens 2126 instead of 50**: Was including prompt tokens in output

### Directory Structure

```
token_alignment/
├── task/                          # NEW: Task folders
│   ├── 1s_20260128_HHMMSS/
│   │   ├── best_model.pt
│   │   ├── results.json
│   │   ├── RESULTS.md
│   │   ├── training_curves.png
│   │   ├── loss_curve.png
│   │   └── accuracy_curve.png
│   └── 500ms_20260128_HHMMSS/
├── train_tokens_full.json         # 1s training tokens
├── train_tokens_500ms.json        # 500ms training tokens
└── test_tokens_500ms.json         # 500ms test tokens
```

### Running Full Workflow

```bash
# Run both 1s and 500ms datasets
nohup ./feasible/token_alignment/run_full_workflow.sh > /tmp/full_workflow.log 2>&1 &
tail -f /tmp/full_workflow.log

# Or run single dataset
python3 feasible/token_alignment/train_and_evaluate.py \
    --train_benchmark ./feasible/token_alignment/train_tokens_full.json \
    --test_benchmark ./feasible/benchmark_parallel_prefill/results/parallel_prefill_5stages_20260127_160820.json \
    --task_name 1s
```

---

## [2026-01-27] Token-Level Acceptance Rate Fix

### Fixed

**Acceptance Rate Calculation:**
- Changed from semantic similarity (word overlap, character similarity, length similarity) to proper token-level matching
- Now correctly compares draft tokens against target model's tokenizer output
- More accurate for speculative decoding feasibility analysis

**Protobuf Compatibility:**
- Added environment variable handling to fix `sentencepiece` protobuf compatibility issue
- Automatically sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` if not already set

### Changed

**`benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py`:**
- Lines 47-49: Added protobuf fix
- Lines 123-195: Reimplemented `compute_acceptance_rate()` for token-level matching
- Lines 630-638: Updated call to pass both tokenizers
- Lines 726-740: Updated statistics computation

### Benchmark Results

**Before (Semantic Similarity):**
- Acceptance rate: ~35% (misleading)
- Uses word overlap, character similarity, length similarity

**After (Token-Level Matching):**
- Acceptance rate: 2-5% for divergent outputs (accurate)
- Counts exact token matches using target model's tokenizer
- Better reflects real speculative decoding behavior

### Running Updated Benchmarks

```bash
# With automatic protobuf fix (built-in):
python feasible/benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py \
  --dataset_dir ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \
  --max_samples -1
```

---

## [2026-01-26] Full Dataset Benchmarks & Output Organization

### Added

**New Scripts:**
- `benchmark_parallel_prefill/run_full_1s_benchmark.sh` - Run benchmark on full 1s dataset (1100 samples)
- `benchmark_parallel_prefill/run_full_500ms_benchmark.sh` - Run benchmark on full 500ms dataset (2220 samples)

**New Directory:**
- `benchmark_parallel_prefill/results/` - Centralized location for all benchmark results

### Changed

**`benchmark_parallel_prefill/benchmark_parallel_prefill_4bit.py`:**
- Added `dataset_name` field to output JSON (e.g., "my_egpt_dsec_seq_1s")
- Updated JSON filename format: `{dataset_name}_parallel_prefill_{quantization}_{timestamp}.json`
- Changed default output directory to `./results/`
- Changed default `max_samples` from 20 to -1 (all samples)
- Updated usage documentation in header

**Author Attribution:**
- All files in `feasible/` folder now show "AUTHOR: Alice Zhang"

### Running Benchmarks

**1s Dataset (1100 samples, ETA: 2-3 hours):**
```bash
./feasible/benchmark_parallel_prefill/run_full_1s_benchmark.sh
```

**500ms Dataset (2220 samples, ETA: 4-5 hours):**
```bash
./feasible/benchmark_parallel_prefill/run_full_500ms_benchmark.sh
```

---

## [2026-01-26] Speculative Decoding Benchmark with Metrics

### Added

**New Scripts:**
- `benchmark_parallel_prefill/benchmark_parallel_with_metrics.py` - Parallel prefill with acceptance rate and speedup
- `benchmark_parallel_prefill/benchmark_speculative.py` - Token-by-token speculative decoding (work in progress due to vocab mismatch)
- `run_speculative_benchmark.sh` - Shell script to run full benchmark

### Fixed

**Import Path Issues:**
- Fixed `import common.common` causing module not found errors
- Changed to `import common` with proper sys.path insertion

### Benchmark Results (10 samples, gamma=5)

| Metric | Value |
|--------|-------|
| EventGPT complete (V+P) | 112.5 ms |
| Video-LLaVA prefill | 272.5 ms |
| Video-LLaVA generation | 1089.2 ms |
| Overlap window | 161.1 ms |
| Theoretical draft tokens | 17.4 |
| Tokens to verify (gamma=5) | 14.0 |
| Accepted tokens (40% est.) | 5.6 |
| **Actual speedup** | **9.34x** |
| Ideal speedup | 4.98x |

---

## [2026-01-26] Video-LLaVA 8-Frames & 4-Bit Quantization

### Added

**New Scripts:**
- `benchmark_parallel_prefill/test_videollava_8frames.py` - Video-LLaVA 8-frame debugging script
- `benchmark_parallel_prefill/test_eventgpt_4bit_memory.py` - EventGPT 4-bit memory profiling

### Fixed

**Video-LLaVA 8-Frames Integration:**
- Fixed garbage output issue ("ms ms ms...") by using correct model class and parameters
- Changed from `LlavaForConditionalGeneration` to `AutoModelForVision2Seq`
- Use `pixel_values_images` instead of `pixel_values` as input key
- Use 8 `<image>` tokens in prompt for 8 video frames

**4-Bit Quantization for Both Models:**
- EventGPT now supports 4-bit quantization via `BitsAndBytesConfig`
- Fixed dtype mismatch (bfloat16 vs float16) between vision tower and quantized model
- Cast event tensor and vision features to float16 for compatibility

### Benchmark Results

**Memory Usage (Both 4-bit):**
| Config | EventGPT | Video-LLaVA | Total |
|--------|----------|-------------|-------|
| Before (EGPT BF16 + VL 4-bit) | 13.19 GB | ~4.3 GB | 17.5 GB |
| **After (Both 4-bit)** | **4.11 GB** | **~4.3 GB** | **8.37 GB** |

**Timing (10 samples, 50 tokens):**
| Metric | EventGPT (4-bit) | Video-LLaVA (4-bit, 8 frames) |
|--------|-----------------|------------------------------|
| Vision + Prefill | 129.3 ms | - |
| Prefill | 103.4 ms | 306.4 ms |
| Generation | 1343.9 ms | 1214.4 ms |
| Overlap | 180.7 ms | - |
| Free draft tokens | 5.7 | - |

**Output Quality:** Both models generate meaningful, coherent text.

### Changed
- `benchmark_parallel_prefill/benchmark_parallel_quantized.py` - Updated for Video-LLaVA 8 frames + 4-bit EventGPT

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

**Last Updated:** 2026-02-06 UTC
