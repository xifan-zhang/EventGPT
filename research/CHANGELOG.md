# Changelog

All notable changes to the EventGPT Speculative Decoding Research are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Planned
- Feature-level speculative decoding integration into benchmark
- Cross-vocabulary mapping experiments
- EAGLE-3 style multi-layer feature fusion training
- Prefill hiding proof-of-concept implementation

---

## [2026-02-06] - Comprehensive Metrics & Cross-Modal Analysis

### Added

**Feature Alignment: All SD Metrics Implemented**

Parallel computation of all speculative decoding metrics in `measure_feature_acceptance.py`:

| Metric | Function | Lines |
|--------|----------|-------|
| Cosine similarity | `compute_all_metrics_parallel()` | 85-112 |
| Accept rates @τ | `compute_all_metrics_parallel()` | 114-120 |
| **Consecutive accepts** | `compute_all_metrics_parallel()` | 122-158 |
| Num hidden tokens | `compute_all_metrics_parallel()` | 160-170 |
| SD accept rates | `compute_all_metrics_parallel()` | 172-198 |
| Speedup estimations | `compute_all_metrics_parallel()` | 200-270 |
| Per-position stats | `compute_per_position_stats()` | 294-321 |
| **Stage timeline viz** | `plot_stage_timeline()` | 397-550 |

**Key Implementation: Parallel Consecutive Accepts**
```python
# No loops - O(batch × seq) parallel
cumprod = (cos_sim > thresh).int().cumprod(dim=1)
consecutive = cumprod.sum(dim=1)  # count before first rejection
```

**New Visualization: `stage_timeline.png`**
- Baseline vs SD horizontal timeline
- Parallel prefill visualization
- Time ratio pie charts

---

**New Document: `CROSSMODAL_VS_SPECVLM.md`**
- Comprehensive analysis of why cross-modal speculative decoding outperforms SpecVLM
- EAGLE-2 draft model architecture documentation
- SpecVLM (EagleVLM + Elastic Visual Compressor) overview
- Why SpecVLM cannot parallelize prefill (parasitic dependency on target features)
- Cross-modal parallel prefill advantage analysis
- T_prefill / T_total benchmark analysis from actual results
- Experimental design (7 experiments)
- Implementation TODO checklist

**README.md Major Updates:**
- Added EAGLE-2 Draft Model Architecture section with detailed diagrams
- Added SpecVLM section (EagleVLM + Elastic Visual Compressor)
- Added Cross-Modal Speculative Decoding: Beyond SpecVLM section
- Added "Why SpecVLM Cannot Parallelize Prefill" explanation
- Added "Why Cross-Modal Can Outperform Same-Modality Baselines" analysis
- Added "Current Benchmark: T_prefill / T_total Analysis" with actual data
- Added Experimental Design section (7 experiments with code)
- Added Implementation Checklist with Phase 1-5 TODO items

### Key Research Findings

**T_prefill / T_total Ratios (from benchmarks):**
| Scenario | Ratio | Cross-Modal Wins? |
|----------|-------|-------------------|
| Video-LLaVA (8f, 5 tok) | **80.6%** | ✅ YES (strong) |
| EventGPT (1f, 5 tok) | 43.7% | ✅ YES |
| EventGPT (1f, 45 tok) | 3.0% | ❌ NO |

**Decision Framework:**
- If T_prefill / T_total > 40% → Cross-Modal wins
- Video-LLaVA ratio: 80.6% >> 40% → IDEAL for cross-modal

**Why Cross-Modal Outperforms:**
1. Parallel prefill hides 500ms+ encoding latency
2. No compression artifacts (lossless)
3. Scalability grows with video length
4. Can afford 20% lower acceptance rate due to parallelism benefit

**SpecVLM Limitation:**
- Draft model is "parasitic" - needs target's penultimate features
- Creates sequential dependency → cannot parallelize
- Speedup ceiling: ~3x

**Cross-Modal Advantage:**
- Draft model is "independent" - has own encoder
- Enables parallel prefill → breaks 3x ceiling
- Theoretical speedup: ~4-5x

### Files Added
- `CROSSMODAL_VS_SPECVLM.md` - Dedicated cross-modal vs SpecVLM analysis

### Files Modified
- `README.md` - Major additions (~500 lines)

---

## [2026-01-31] - Feature Alignment: Chunked Hidden State Extraction

### Added

**Chunked Incremental Saving:**
- `ChunkedHiddenStateWriter` class for memory-efficient extraction
- Saves every 1000 samples to separate .pt chunks (~1.6GB each)
- Auto-resume via `index.json` tracking
- `load_chunked_hidden_states()` to concatenate all chunks

**New Arguments:**
- `--chunked` - Enable incremental saving mode
- `--chunk_size N` - Samples per chunk (default: 1000)
- `--quant TAG` - Quantization tag for filename

### Current Status

| Task | Status | Progress |
|------|--------|----------|
| Train hidden state extraction | 🔄 Running | ~20,000/52,080 (38%) |
| Test hidden state extraction | ⏳ Pending | 0/11,000 |
| Hidden adapter training | ⏳ Blocked | - |
| Feature-level SD evaluation | ⏳ Blocked | - |

**ETA:** ~14 hours for train extraction completion

---

## [2026-01-30] - Feature Alignment: L1-L4 Adapter Architectures

### Added

**Multi-Level Adapter Implementations:**
| Level | Architecture | Params | Description |
|-------|--------------|--------|-------------|
| L1 | HiddenStateAdapter | 2.1M | Simple bottleneck (4096→256→4096) |
| L2 | MultiLayerBottleneckAdapter | 6.3M | 3× stacked bottlenecks |
| L3 | WideBottleneckAdapter | 16.8M | Wide bottleneck (4096→1024→4096) |
| L4 | AttentionAdapter | 100M | Self-attention + FFN (EAGLE-style) |

**Factory Functions:**
- `create_adapter(level=1-4)` - Unified creation
- `load_any_adapter(path)` - Auto-detect checkpoint type

**Pilot Results (L1, 100 samples × 10 questions):**
- Cosine similarity: 0.764
- Consecutive accepts @0.90: **6.35 tokens avg**
- Estimated speedup: **5.77x** (vs 1.0x token-level)

### Files Added
- `feasible/feature_alignment/hidden_adapter.py`
- `feasible/feature_alignment/train_hidden_adapter.py`
- `feasible/feature_alignment/extract_hidden_states.py`
- `feasible/feature_alignment/README.md`

---

## [2026-01-29] - Token Alignment: Benchmark Integration + Hidden State Plan

### Added

**Benchmark Integration:**
- TokenAdapter integrated into `benchmark_parallel_prefill_5stages.py`
- TokenAdapter integrated into `benchmark_inference_5stages.py`
- New flags: `--use_token_adapter`, `--token_adapter_path`
- Reports both baseline and aligned acceptance rates
- Auto-detects latest trained checkpoint

**10-Question Workflow:**
- `run_10q_workflow.sh` - Balanced training workflow (50 epochs, no early stopping)
- Currently running: 10q extraction complete (52,000 pairs), test extraction in progress

### Planned

**Hidden State Extraction for Feature Alignment:**
- Modify `extract_tokens_parallel.py` to optionally extract hidden states
- Save last-layer hidden states for both EventGPT and Video-LLaVA
- Enable feature-level alignment training (EAGLE-style) alongside token-level
- Expected improvement: 40-60% acceptance rate (vs 27% token-level)

### Current Status

| Task | Status |
|------|--------|
| 10q train extraction | ✅ Complete (52,000 pairs) |
| 10q test extraction | 🔄 Running |
| 10q training (50 epochs) | ⏳ Pending |
| Benchmark integration | ✅ Complete |

---

## [2026-01-28] - Stream-to-Cloud Speculative Decoding

### Added

**New Directory: `edge_cloud_SD/`**
Consolidated edge-cloud speculative decoding research with new paradigm proposal.

**Documents:**
1. **`VIDEO_STREAMING_COMPUTE_OFFLOAD.md`** - NEW paradigm proposal
   - Core idea: Stream video to cloud, draft tokens on edge
   - Insight: Video streaming is cheap (5G, $0.01/GB); vision encoding is expensive
   - EventGPT advantage: Event streams are 50-500x smaller than video
   - Research support from: SLED, DistServe, NVIDIA Dynamo, 5G streaming

2. **`edge_cloud_speculative_decoding.md`** - Moved from research/ root
   - Original comprehensive survey of edge-cloud SD architectures
   - Challenges: network latency, draft quality, privacy, sync
   - Solutions: adaptive strategy, pipeline parallelism, hierarchical drafts

### Key Research Findings

| Factor | Video Streaming | Vision Encoding |
|--------|----------------|-----------------|
| Cost | $0.01-0.05/GB | GPU compute ($0.50+/hr) |
| Latency (5G) | <1ms network | 50-500ms compute |
| Infrastructure | Mature CDNs | Limited GPUs |

**Stream-to-Cloud Benefits:**
- 3-5x edge power savings
- Same inference quality
- Works best with 5G (<10ms RTT)

---

## [2026-01-28] - Cascaded Speculative Decoding Research

### Added

**New Directory: `egpt_vllava_roadmap/`**
Consolidated research on cascaded/hierarchical speculative decoding for EventGPT → VideoLLaVA.

**Research Documents Created:**
1. **`CASCADED_SD_RESEARCH.md`** - Main research overview
   - Top 5 papers on cascaded SD with justifications
   - Light → Middle → Dense modality cascade concept
   - Research gaps and opportunities identified

2. **`SpecVLM_READING_NOTES.md`** - SpecVLM paper analysis (arXiv:2509.11815)
   - Elastic visual compressor (pruning, pooling, conv, resampler)
   - Online-logit distillation technique
   - 2.5-2.9× speedup on VLMs

3. **`PyramidSD_READING_NOTES.md`** - PyramidSD paper analysis (arXiv:2510.12966)
   - 3-model cascade: 1B → 3B → 8B
   - Entropy gradient exploitation
   - PSD_A (assisted) vs PSD_F (fuzzy) variants
   - 1.91× speedup (124 tok/s on RTX 4090)

4. **`HiSpec_READING_NOTES.md`** - HiSpec paper analysis (arXiv:2510.01336)
   - Verification bottleneck insight (2-10× slower than drafting)
   - Early-exit models for intermediate verification
   - Layer positioning: 1/8 draft, 1/4 intermediate, full target
   - Up to 2.01× speedup

5. **`PipeSpec_READING_NOTES.md`** - PipeSpec paper analysis (arXiv:2505.01572)
   - Asynchronous k-model pipeline architecture
   - Breaks stage dependencies for continuous execution
   - Producer-consumer relationships with rollback
   - 2.54× speedup on LLaMA3.1-70B

6. **`LOSSLESS_VS_LOSSY_SPECULATIVE_DECODING.md`** - Theoretical foundation
   - Mathematical proof of lossless SD (rejection sampling)
   - Lossy variants: fuzzy, top-k, temperature mismatch
   - Comparison table with speedup vs quality trade-offs

7. **`CROSS_MODAL_SD_THEORETICAL_FOUNDATIONS.md`** - Cross-modal theory
   - Why SD saves time (parallel verification)
   - Modality gap and distribution alignment
   - Entropy gradient across modalities
   - Cross-attention mechanisms (DREAM, ViSpec)
   - Acceptance rate bounds for cross-modal SD
   - Proposed EventGPT → VideoLLaVA architecture

**PDFs Downloaded to `cascaded_SD/pdf/`:**
- `faster_cascades_via_speculative_decoding_ICLR2025.pdf`
- `HiSpec_hierarchical_speculative_decoding.pdf`
- `PyramidSD_3model_speculative_decoding.pdf`
- `PipeSpec_hierarchical_LLM_decoding.pdf`
- `SpecVLM_vision_language_speculative_decoding.pdf`

### Key Research Findings

| Paper | Key Insight | Speedup |
|-------|-------------|---------|
| SpecVLM | Elastic visual compression + online distillation | 2.5-2.9× |
| PyramidSD | Entropy gradient across model sizes | 1.91× |
| HiSpec | Verification is the bottleneck, not drafting | 2.01× |
| PipeSpec | Asynchronous execution eliminates idle periods | 2.54× |

**Cross-Modal SD Theory:**
- Modality gap requires explicit alignment (MLP, cross-attention)
- Information imbalance drives gap (sparse events vs dense video)
- Expected acceptance rate: 0.5-0.7 with proper alignment
- Async pipelining suits real-time event camera streams

### Files Moved
- All cascaded SD markdown files from `cascaded_SD/` → `egpt_vllava_roadmap/`

---

## [2026-01-28] - Token Alignment: First Success + Multi-Question Scaling

### Results - Single Question Training

**First successful end-to-end token alignment training:**

| Dataset | Samples | Baseline | TokenAdapter | Top-5 | Improvement |
|---------|---------|----------|--------------|-------|-------------|
| Train | 5,200 | 1.77% | 27.21% | 51.40% | +25.45% |
| Test | 1,100 | 1.58% | **27.90%** | 51.64% | +26.32% |

- **17.6x improvement** over baseline (1.58% → 27.90%)
- **Theoretical speedup:** 1.39x with α=27.9%, γ=5 draft tokens
- **Model:** 4-layer transformer, ~45M parameters

### Added

**Parallel Extraction Script:**
- `feasible/token_alignment/extract_tokens_parallel.py` - Both models loaded simultaneously
- ~8GB VRAM (both models 4-bit), ~1.5s/pair
- Currently extracting 10-question dataset (52,080 pairs, ~24hr ETA)

**Multi-Question Training:**
- `top50_questions.json` - Top 50 DSEC questions for diverse training
- Scaling from 1 question (5,200 pairs) → 10 questions (52,080 pairs)
- Expected improvement: 30-40% acceptance with question diversity

### Analysis

**Token-Level Alignment Validation:**
The 27.9% acceptance rate validates that token-level alignment is feasible:
- Far exceeds the 2-5% baseline (direct token comparison)
- Approaches theoretical ceiling for token-level mapping (~50%)
- Enables practical speculative decoding speedup (1.3-1.5x)

**Why Token Ceiling ~50%:**
1. Models use different vocabulary for same concepts
2. Sentence structure fundamentally differs
3. EventGPT (1 frame) vs Video-LLaVA (8 frames) see different information

**Path to Higher Acceptance:**
- Feature-level alignment (bypass tokenizer mismatch)
- Fine-tune EventGPT LM head on Video-LLaVA outputs
- Medusa-style multi-head speculation

### Documentation
- Updated `feasible/token_alignment/README.md` with comprehensive usage guide
- Added sequential vs parallel extraction options
- Added troubleshooting and further directions

---

## [2026-01-27] - Token-Level Acceptance Rate Validation

### Fixed

**Acceptance Rate Calculation:**
- Corrected acceptance rate metric from semantic similarity to proper token-level matching
- Now accurately reflects real speculative decoding acceptance behavior
- Accounts for tokenizer differences between EventGPT and Video-LLaVA

### Analysis

The token-level acceptance metric shows:
- **Semantic similarity approach:** 30-40% (misleading, based on word overlap)
- **Token-level approach:** 2-5% for divergent outputs (accurate, based on token matching)

This validates that:
1. EventGPT and Video-LLaVA have fundamentally different output distributions
2. Simple token copying won't work effectively (very low acceptance)
3. Feature-level alignment or semantic token mapping needed for good speculative decoding
4. Prefill hiding remains viable strategy despite low token acceptance

### Impact on Feasibility

- **Parallel prefill:** Still valuable (hidden token generation independent of acceptance)
- **Token-level speculative decoding:** Low effectiveness (2-5% acceptance)
- **Feature-level speculative decoding:** Becomes more important for actual speedup
- **Research direction:** Focus on semantic-aware token mapping or draft model adaptation

---

## [2026-01-26] - Prefill Hiding Research

### Added
- **New Directory:** `prefill_hiding/` - Research on hiding draft model prefill inside target model prefill
- **New Document:** `prefill_hiding/README.md` - Comprehensive exploration of prefill hiding technique
  - Exploits 8.59x prefill time difference (EventGPT 66ms vs Video-LLaVA 568ms)
  - Generates ~27 free draft tokens during target prefill window
  - Expected speedup: 1.3-1.7x depending on acceptance rate
  - Memory-efficient implementation strategies
  - Feature-level alignment to improve acceptance rates

- **New Document:** `prefill_hiding/RELATED_WORK.md` - Comprehensive survey of parallel/async speculative decoding
  - DSI (ICLR 2025): Hides verification latency via speculation parallelism
  - PEARL (ICLR 2025): Pre-verify/post-verify strategies, 4.43x speedup
  - SSD (ICLR 2026 submission): Speculative speculative decoding
  - AMUSD: Asynchronous multi-device execution, 1.96x speedup
  - SwiftSpec (June 2025): Disaggregated pipeline, removes draft from critical path
  - HiViS (Sept 2025): Hides visual tokens from drafter, 2.65x speedup
  - SpecVLM, ViSpec: VLM-specific speculative decoding

- **New Document:** `prefill_hiding/NOVELTY_ANALYSIS.md` - Analysis of event camera as draft model novelty
  - First cross-modal speculative decoding (event camera → video VLM)
  - Unique advantages: physics-based speed, temporal understanding, constant token scaling
  - Paper positioning and title suggestions

- **New Document:** `prefill_hiding/ABSTRACT.md` - Paper abstract drafts (4 options)
  - Technical focus, concise, novelty-emphasis, and application-oriented versions
  - Key claims and suggested paper structure

- **New Directory:** `pdf/` - Downloaded top 5 related papers
  - `1_DSI_Distributed_Speculative_Inference_ICLR2025.pdf` - Hides verification latency
  - `2_PEARL_Parallel_Speculative_Decoding_ICLR2025.pdf` - Pre-verify/post-verify
  - `3_HiViS_Hiding_Visual_Tokens_2025.pdf` - Hides visual tokens from drafter
  - `4_SpecVLM_Fast_Speculative_Decoding_VLM_2025.pdf` - VLM speculation baseline
  - `5_SSD_Speculative_Speculative_Decoding_ICLR2026.pdf` - Eliminates speculation overhead

- **New Document:** `pdf/TOP5_PAPERS_SUMMARY.md` - Reading guide and summaries

### Moved
- `cross_modal_speculative_prefill.md` → `prefill_hiding/`
- `parallel_prefill_speculative_decoding_20260125_004920.md` → `prefill_hiding/`

### Key Research Findings
- **Free time window:** 502ms (568ms - 66ms) available for draft generation
- **Free draft tokens:** ~27 tokens at EventGPT's 18.5ms/token decode speed
- **Coverage:** 61% of typical response (27/44.5 tokens) drafted "for free"
- Even 40% acceptance rate provides 1.17x speedup
- **Novel contribution:** First cross-modal speculative decoding using different sensor modalities

---

## [2026-01-26] - 4-Bit Quantization Achieved

### Added
- **4-bit EventGPT:** EventGPT now supports BitsAndBytes 4-bit quantization
- **Memory Optimization:** Both models (EventGPT + Video-LLaVA) run at 4-bit, total memory reduced to **8.37 GB**

### Benchmark Results

**Memory Comparison:**
| Config | EventGPT | Video-LLaVA | Total |
|--------|----------|-------------|-------|
| Before (EGPT BF16 + VL 4-bit) | 13.19 GB | ~4.3 GB | 17.5 GB |
| **After (Both 4-bit)** | **4.11 GB** | **~4.3 GB** | **8.37 GB** |

**Parallel Prefill (10 samples, 50 tokens):**
| Metric | EventGPT (4-bit) | Video-LLaVA (4-bit, 8 frames) |
|--------|-----------------|------------------------------|
| Vision + Prefill | 129.3 ms | - |
| Prefill | 103.4 ms | 306.4 ms |
| Generation | 1343.9 ms | 1214.4 ms |
| Overlap | 180.7 ms | - |
| Free draft tokens | 5.7 | - |

### Fixed
- Video-LLaVA 8-frame integration (fixed garbage output, correct model class)
- EventGPT dtype mismatch (bfloat16 vision tower vs float16 quantized model)

---

## [2026-01-25] - Parallel Prefilling Research

### Added
- **New Document:** `parallel_prefill_speculative_decoding_20260125_004920.md`
  - Comprehensive survey of parallel prefilling techniques (2025)
  - Cross-model speculative decoding with tokenizer mismatch solutions
  - Five research opportunities for EventGPT + Video-LLaVA
  - Implementation roadmap with expected speedups (2-3x)

- **New Papers Surveyed:**
  - OmniDraft: Cross-vocabulary online adaptive drafter
  - CTPD: Cross Tokenizer Preference Distillation (arXiv:2601.11865)
  - GRIFFIN: Token Alignment for Speculative Decoding (Feb 2025)
  - Pyramid Speculative Decoding (arXiv:2510.12966)
  - SpecVLM: Fast Speculative Decoding in VLMs (arXiv:2509.11815)
  - ViSpec: Vision-Aware Speculative Decoding (arXiv:2509.15235)
  - DREAM: Cross-Attention for Multimodal Speculative Decoding (arXiv:2505.19201)
  - MASSV: Multimodal Adaptation for VLM Speculative Decoding (arXiv:2505.10526)
  - Disaggregated Prefill-Decode papers (Nexus, TPLA, SPAD)
  - EAGLE-3 and SpecForge training framework

### Changed
- Updated `README.md` with new document link
- Updated `UPDATE_LOG.md` with session details

### Research Findings
- Token-level acceptance rate (~2-5%) too low due to tokenizer mismatch
- Feature-level speculation recommended to bypass tokenizer issues
- Cross-modal speculative prefill is unexplored research opportunity

---

## [2026-01-23] - Latest Papers Update

### Added
- **New Papers (December 2025 - January 2026):**
  - Entropy-Aware Speculative Decoding (arXiv:2512.23765)
  - SpecPV: Partial Verification for Long Context (arXiv:2512.02337)
  - Accelerate with Sparse Computation (arXiv:2512.21911)
  - Efficient Adaptive Rejection Sampling (arXiv:2512.13194)
  - Optimal Lower Bounds (arXiv:2512.11718)
  - Speculative Sampling with RL (arXiv:2601.12212)
  - Revisiting Judge Decoding (arXiv:2601.04766)
  - Multi-Scale for Image Generation (arXiv:2601.05149)
  - SRT for RL Acceleration (arXiv:2601.09083)
  - Plan, Verify and Fill (arXiv:2601.12247)

- **Implementation Details:**
  - EAGLE-3 training-time test code examples
  - Multi-layer feature fusion implementation
  - GitHub repository links (SafeAILab/EAGLE, BaldEagle, fast-llm-inference)

- **New Document:** `edge_cloud_speculative_decoding.md`
  - Edge device drafting with cloud verification
  - Distributed inference patterns

---

## [2026-01-22] - Initial Research Collection

### Added
- **Core Documents:**
  - `README.md` - Main index and overview
  - `token_level_speculative_decoding.md` - Standard/vanilla approach
  - `embedding_level_speculative_decoding.md` - EAGLE-family methods
  - `hybrid_speculative_decoding.md` - Combined approaches
  - `sequential_cascaded_speculative_decoding.md` - Multi-stage chains
  - `EventGPT_VideoLLaVA_roadmap.md` - Application roadmap

- **Supporting Files:**
  - `UNIFIED_TEMPLATE.md` - Document structure template
  - `UPDATE_LOG.md` - Detailed update tracking
  - `structure.md` - Repository organization

- **ArXiv Paper Summaries:**
  - `arxiv/2024/2404.08856_multimodal_drafting.md`
  - `arxiv/2025/2509.11815_specvlm.md`
  - `arxiv/2025/2509.15235_vispec.md`
  - `arxiv/2025/2505.14260_multimodal_reimagined.md`
  - `arxiv/edge_cloud/spec_vla.md`

### Established
- Research taxonomy covering token-level, feature-level, hybrid, and cascaded approaches
- Performance comparison tables
- Code examples for each major method
- Citation format and quality guidelines

---

## Document Index

| Document | Added | Description |
|----------|-------|-------------|
| `CROSSMODAL_VS_SPECVLM.md` | 2026-02-06 | **NEW** Cross-modal vs SpecVLM analysis |
| `egpt_vllava_roadmap/CASCADED_SD_RESEARCH.md` | 2026-01-28 | Top 5 cascaded SD papers |
| `egpt_vllava_roadmap/SpecVLM_READING_NOTES.md` | 2026-01-28 | **NEW** SpecVLM paper analysis |
| `egpt_vllava_roadmap/PyramidSD_READING_NOTES.md` | 2026-01-28 | **NEW** PyramidSD 3-model cascade |
| `egpt_vllava_roadmap/HiSpec_READING_NOTES.md` | 2026-01-28 | **NEW** HiSpec verification acceleration |
| `egpt_vllava_roadmap/PipeSpec_READING_NOTES.md` | 2026-01-28 | **NEW** PipeSpec async pipeline |
| `egpt_vllava_roadmap/LOSSLESS_VS_LOSSY_SPECULATIVE_DECODING.md` | 2026-01-28 | **NEW** SD theory foundation |
| `egpt_vllava_roadmap/CROSS_MODAL_SD_THEORETICAL_FOUNDATIONS.md` | 2026-01-28 | **NEW** Cross-modal SD theory |
| `prefill_hiding/ABSTRACT.md` | 2026-01-26 | Paper abstract drafts (4 versions) |
| `prefill_hiding/NOVELTY_ANALYSIS.md` | 2026-01-26 | **NEW** Event camera draft novelty analysis |
| `prefill_hiding/RELATED_WORK.md` | 2026-01-26 | **NEW** Parallel/async spec decoding survey |
| `prefill_hiding/README.md` | 2026-01-26 | Prefill hiding technique research |
| `prefill_hiding/parallel_prefill_speculative_decoding_20260125_004920.md` | 2026-01-25 | Cross-modal VLM acceleration research |
| `prefill_hiding/cross_modal_speculative_prefill.md` | 2026-01-22 | Cross-modal prefilling |
| `edge_cloud_speculative_decoding.md` | 2026-01-23 | Distributed inference |
| `EventGPT_VideoLLaVA_roadmap.md` | 2026-01-22 | Application roadmap |
| `sequential_cascaded_speculative_decoding.md` | 2026-01-22 | Multi-stage chains |
| `hybrid_speculative_decoding.md` | 2026-01-22 | Combined approaches |
| `embedding_level_speculative_decoding.md` | 2026-01-22 | EAGLE-family |
| `token_level_speculative_decoding.md` | 2026-01-22 | Standard approach |

---

## Statistics

| Metric | Count |
|--------|-------|
| Total Documents | 20 |
| Papers Surveyed | 68+ |
| Code Examples | 30+ |
| Research Opportunities Identified | 9 |
| Abstract Drafts | 4 |
| Paper Reading Notes | 6 |
| Experiments Designed | 7 |

---

## How to Update This Changelog

When making changes to the research collection:

1. Add entry under `[Unreleased]` or create new dated section
2. Use categories: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`
3. Include file names, paper arXiv IDs, and brief descriptions
4. Update Document Index table if adding new files
5. Update Statistics section

---

**Maintainer:** Alice Zhang
**Last Updated:** 2026-02-06
