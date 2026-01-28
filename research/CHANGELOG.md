# Changelog

All notable changes to the EventGPT Speculative Decoding Research are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Planned
- Feature-level speculative decoding implementation for EventGPT + Video-LLaVA
- Cross-vocabulary mapping experiments
- EAGLE-3 style multi-layer feature fusion training
- Prefill hiding proof-of-concept implementation

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

## [2026-01-28] - Token Alignment Training on Full Dataset

### Added

**Token Alignment Pipeline:**
- Full-scale token extraction from EventGPT and Video-LLaVA
- Uses event_image (PNG) for EventGPT, mp4 (8 frames) for Video-LLaVA
- Task folder system with timestamped outputs
- Training curves visualization (loss, accuracy, top-5)

**Datasets Processed:**
| Dataset | Train Samples | Test Samples |
|---------|---------------|--------------|
| 1s | 5,208 | 1,100 |
| 500ms | 10,475 | 2,220 |

### Analysis

**TokenAdapter Approach:**
- 4-layer transformer (~50M params)
- Learns EventGPT → Video-LLaVA token mapping
- Previous result (200 samples): 26.66% acceptance
- Expected with full dataset: 30-40% acceptance

**Key Insight:**
Token-level alignment has inherent limitations due to semantic gap between models. Even with perfect training, acceptance rate unlikely to exceed ~50% because:
1. Models describe scenes with different vocabulary
2. Sentence structure differs fundamentally
3. Some concepts have no direct mapping

**Next Steps:**
- Analyze training results for both 1s and 500ms datasets
- Compare performance across different event durations
- Consider feature-level fusion for higher acceptance rates

### Documentation
- Updated `feasible/token_alignment/WORKFLOW.md` with complete pipeline documentation

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
| `egpt_vllava_roadmap/CASCADED_SD_RESEARCH.md` | 2026-01-28 | **NEW** Top 5 cascaded SD papers |
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
| Total Documents | 19 |
| Papers Surveyed | 65+ |
| Code Examples | 25+ |
| Research Opportunities Identified | 8 |
| Abstract Drafts | 4 |
| Paper Reading Notes | 5 |

---

## How to Update This Changelog

When making changes to the research collection:

1. Add entry under `[Unreleased]` or create new dated section
2. Use categories: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`
3. Include file names, paper arXiv IDs, and brief descriptions
4. Update Document Index table if adding new files
5. Update Statistics section

---

**Maintainer:** Research Team
**Last Updated:** 2026-01-28
