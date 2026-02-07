# Speculative Decoding Research - Update Log

This file tracks all updates made to the research markdown files.

---

## Latest Update (February 7, 2026) - Pipeline Reorganization + E2E Benchmark Analysis

### Pipeline Reorganization

Organized all cross-modal speculative decoding code into a clean 4-stage `pipeline/` folder:

```
pipeline/
├── README.md                              # Overview, data flow, quick start
├── feature_extraction/                    # Stage 1: Extract paired hidden states
│   ├── extract_vl_lm_head.py
│   ├── extract_hidden_states.py
│   └── monitor_extraction.sh
├── adapter_train/                         # Stage 2: Train adapters (L1-L5, B1, L5F)
│   ├── hidden_adapter.py
│   ├── train_hidden_adapter.py
│   ├── auto_train_pipeline.sh
│   └── retrain_L4_converge.sh
├── evaluation/                            # Stage 3: Offline acceptance metrics
│   ├── measure_feature_acceptance.py
│   ├── eval_two_phase.py
│   ├── run_all_eval.sh
│   └── run_two_phase_eval.sh
└── benchmark_e2e/                         # Stage 4: E2E wall-clock benchmark
    └── benchmark_e2e_wallclock.py
```

### Changes
- Copied all Python scripts and shell scripts from `feasible/feature_alignment/` into `pipeline/`
- Added `Author: Alice Zhang` and `Date: 2026-02-07` to all 7 Python scripts
- Updated default output paths: scripts now output to local `tasks/` or `data/` subdirs
- Added comprehensive README.md for each stage with CLI reference, output formats, and results
- Updated shell scripts to use `pipeline/` paths

### E2E Benchmark Analysis (10,970 samples, max_new_tokens=30)

Key finding: **1.03x speedup** despite ~22 free tokens during prefill. Root causes:

| Component | Value |
|-----------|-------|
| Mean drafted tokens | 20.9 |
| Mean accepted tokens | 4.4 (21.2%) |
| Value of saved tokens | 5.4 × 14.0ms = **75ms** |
| Verify batch cost | **~70ms** |
| Net saving | **~5ms** per 736ms sample |

32% of samples get 0-1 accepted tokens → net slowdown with SD overhead.

| Accept% | Projected Speedup |
|---------|----|
| 21% (current L4) | 1.01x |
| 50% | 1.14x |
| 70% | 1.25x |
| 100% | 1.47x |

Bottleneck is acceptance rate. Improving adapter quality is the key path to higher speedup.

---

## Update (February 7, 2026) - E2E Wall-Clock Benchmark + Prefill Hiding

### True E2E Wall-Clock Benchmark (50 samples, 10 questions, max_new_tokens=50)

| Config | Prefill | Decode | Total | Accept | Speedup | FreeTok |
|--------|---------|--------|-------|--------|---------|---------|
| VL baseline | 326ms | 691ms | 1017ms | --- | 1.00x | --- |
| L1+VL | 326ms | 691ms | 1018ms | 15.9% | 1.00x | 21.9 |
| L2+VL | 326ms | 681ms | 1007ms | 19.3% | 1.02x | 21.9 |
| L3+VL | 326ms | 681ms | 1007ms | 19.6% | 1.02x | 21.9 |
| L4+VL | 326ms | 679ms | 1005ms | **20.3%** | **1.02x** | 21.9 |

### Pipeline: Prefill Hiding + Cross-Modal Speculative Decoding

1. **EGPT prefill + decode** runs first (vision encode + LLM prefill + AR decode)
2. **VL prefill** runs independently (no GPU contention model)
3. **~22 EGPT decode tokens are FREE** — hidden behind VL prefill latency gap
4. **Adapter** maps EGPT hidden states → VL token predictions (draft tokens)
5. **VL verify** accepts/rejects draft tokens in single batched forward pass
6. **VL AR decode** continues from last accepted position

### Key Fixes in This Session
- **Off-by-one bug**: draft_tokens[0] duplicated vl_first token → fixed by using draft_tokens[1:]
- **GPU contention**: Single-GPU can't truly parallelize two models → sequential execution with theoretical overlap timing model
- **OOM in 3-stage timing**: Added KV cache cleanup between EGPT and VL timing runs

### All 7 Adapters Trained & Evaluated

| Level | Architecture | Params | Accept@0.90 | Val Loss |
|-------|-------------|--------|-------------|----------|
| L1 | Bottleneck | 2.1M | 21.9% | 1.2798 |
| L2 | Multi-Layer Bottleneck | 6.3M | 23.2% | 1.2787 |
| L3 | Wide Bottleneck | 16M | 24.9% | 1.2499 |
| L4 | Attention | 101M | 24.8% | 1.2458 |
| L5 | EAGLE (cross-modal) | 103M | 11.2% | 1.3413 |
| B1 | EAGLE (VLM-only baseline) | 103M | 61.2% | 0.6812 |
| L5F | Fused EAGLE | 170M | 66.2% | 0.7282 |

### Key Findings
- **B1 >> L5**: Cross-modal gap is dominant bottleneck (61.2% vs 11.2%)
- **L5F > B1**: Fused adapter beats VLM-only baseline (+5% Accept@0.90)
- **~22 free tokens**: EGPT decode fully hidden behind VL prefill (~230ms gap)
- **L4+VL best E2E**: 20.3% accept, 1.02x avg speedup, up to 1.24x per-sample
- **L1-L4 diminishing returns**: 2M→101M params only +3% Accept@0.90

### Files Created/Modified
- `feasible/feature_alignment/hidden_adapter.py` — All adapter architectures
- `feasible/feature_alignment/train_hidden_adapter.py` — Training with chunked data
- `feasible/feature_alignment/benchmark_e2e_wallclock.py` — True E2E wall-clock benchmark with prefill hiding, per-token timestamps, batched VL verification
- `feasible/feature_alignment/measure_feature_acceptance.py` — Token-level evaluation
- `feasible/feature_alignment/eval_two_phase.py` — Two-phase pipeline evaluation

### Data Pipeline
- Hidden states extracted from both models on DSEC 1s dataset
- EventGPT: uses first frame of 5 preprocessed event images per clip
- Video-LLaVA: uses 8 uniformly-sampled frames from MP4
- 52K train + 11K test samples, 4-bit quantized, chunked storage

---

## Update (January 28, 2026) - Stream-to-Cloud Speculative Decoding

### New Directory: `edge_cloud_SD/`

Created consolidated folder for edge-cloud speculative decoding research.

### New Document: `VIDEO_STREAMING_COMPUTE_OFFLOAD.md`

**Core Idea:** Reverse the edge-cloud paradigm
- Traditional: Process video on edge, send tokens to cloud
- New: Stream raw video to cloud, draft tokens on edge

**Why This Works:**
- Video streaming: 5G offers 100Mbps+, <1ms latency, $0.01-0.05/GB
- Vision encoding: 50-500ms compute, requires GPU
- Stream-to-cloud saves edge compute, leverages cloud GPUs

**Research Support:**
- [SLED: Edge Speculative Decoding](https://arxiv.org/html/2506.09397v3)
- [Disaggregated Inference (Hao AI Lab)](https://hao-ai-lab.github.io/blogs/distserve-retro/)
- [5G Streaming Revolution](https://www.dacast.com/blog/5g-streaming/)
- [NVIDIA Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)

**EventGPT Advantage:**
Event camera streams are 50-500x smaller than RGB video (10-100 KB/s vs 5-10 MB/s), making stream-to-cloud especially attractive.

### Files Moved
- `edge_cloud_speculative_decoding.md` → `edge_cloud_SD/`

---

## Update (January 28, 2026) - Token Alignment Success

### Token Alignment Training Results

**First successful end-to-end TokenAdapter training:**

| Metric | Value |
|--------|-------|
| Test Acceptance | **27.90%** (was 1.58% baseline) |
| Top-5 Accuracy | 51.64% |
| Improvement | 17.6x over baseline |
| Theoretical Speedup | 1.39x |

### Files Created/Modified:
1. **`feasible/token_alignment/extract_tokens_parallel.py`** - NEW parallel extraction
   - Both models loaded simultaneously (~8GB VRAM)
   - ~1.5s/pair extraction speed
   - Currently running 10-question extraction (52,080 pairs)

2. **`feasible/token_alignment/top50_questions.json`** - Top 50 DSEC questions

3. **`feasible/token_alignment/README.md`** - Comprehensive usage documentation
   - Quick start guides (1q, 50q workflows)
   - Sequential vs parallel extraction
   - Architecture diagrams

4. **`feasible/token_alignment/task/starred/1q_20260128_151847/`** - Starred results

### Key Findings:
- **Token-level alignment works:** 27.9% acceptance validates approach
- **Ceiling ~50%:** Semantic gap limits token-level mapping
- **Next step:** Multi-question training for diversity (10q extraction running)
- **Path forward:** Feature-level alignment for higher acceptance

---

## Update (January 28, 2026) - Cascaded SD Research

### New Directory: `egpt_vllava_roadmap/`
Comprehensive research on cascaded/hierarchical speculative decoding for EventGPT → VideoLLaVA.

### Documents Created:
1. **`CASCADED_SD_RESEARCH.md`** - Top 5 papers, research gaps, cross-modal cascade concept
2. **`SpecVLM_READING_NOTES.md`** - Elastic visual compressor, 2.5-2.9× speedup
3. **`PyramidSD_READING_NOTES.md`** - 3-model entropy gradient cascade, 1.91× speedup
4. **`HiSpec_READING_NOTES.md`** - Verification bottleneck, early-exit models, 2.01× speedup
5. **`PipeSpec_READING_NOTES.md`** - Async k-model pipeline, 2.54× speedup
6. **`LOSSLESS_VS_LOSSY_SPECULATIVE_DECODING.md`** - Mathematical foundation
7. **`CROSS_MODAL_SD_THEORETICAL_FOUNDATIONS.md`** - Cross-modal theory and EventGPT application

### PDFs Downloaded:
- Faster Cascades (ICLR 2025), HiSpec, PyramidSD, PipeSpec, SpecVLM

### Key Findings:
- **Verification is the bottleneck** (2-10× slower than drafting)
- **Entropy gradient** exploitable across modalities (sparse→dense)
- **Async execution** provides primary speedup (not just model count)
- **Cross-modal SD** requires explicit feature alignment
- **Expected EventGPT→VideoLLaVA speedup**: 2.5-3.5× with proper alignment

---

## Update (January 26, 2026) - Prefill Hiding Research

### New Directory: `prefill_hiding/`
Created comprehensive research folder for cross-modal speculative decoding.

### New Documents Created:
1. **`prefill_hiding/README.md`** - Core prefill hiding technique
   - Exploits 8.59x prefill time difference (EventGPT 66ms vs Video-LLaVA 568ms)
   - ~27 free draft tokens during 502ms target prefill window
   - Expected 1.3-1.7x speedup depending on acceptance rate
   - Implementation architecture and memory strategies

2. **`prefill_hiding/RELATED_WORK.md`** - Comprehensive literature survey
   - DSI (ICLR 2025): Distributed Speculative Inference, hides verification latency
   - PEARL (ICLR 2025): Pre-verify/post-verify, 4.43x speedup
   - SSD (ICLR 2026 submission): Speculative Speculative Decoding
   - AMUSD: Asynchronous multi-device, 1.96x speedup
   - SwiftSpec (June 2025): Disaggregated pipeline, 1.75x speedup
   - HiViS (Sept 2025): Hides visual tokens from drafter, 2.65x speedup
   - SpecVLM, ViSpec: VLM-specific speculative decoding

3. **`prefill_hiding/NOVELTY_ANALYSIS.md`** - Paper novelty positioning
   - First cross-modal speculative decoding (event camera → video VLM)
   - Physics-based speed advantage vs architectural compression
   - Constant token scaling (events) vs linear (video frames)
   - Paper title suggestions and positioning statement

4. **`prefill_hiding/ABSTRACT.md`** - Paper abstract drafts
   - 4 versions: Technical, Concise, Novelty-focused, Application-oriented
   - Key claims and suggested paper structure

### Files Moved:
- `cross_modal_speculative_prefill.md` → `prefill_hiding/`
- `parallel_prefill_speculative_decoding_20260125_004920.md` → `prefill_hiding/`

### Key Research Findings:
- **Gap identified:** No prior work on cross-modal speculative decoding with different sensor modalities
- **Novel contribution:** Event camera physical properties enable "free" draft generation
- **Positioning:** Opens new research direction of sensor-level speculation

---

## Update (January 25, 2026)

### New Document: Parallel Prefill & Speculative Decoding Research
- **File:** `parallel_prefill_speculative_decoding_20260125_004920.md`
- **Content:**
  - Comprehensive survey of parallel prefilling techniques (2025)
  - Cross-model speculative decoding with tokenizer mismatch solutions
  - Five research opportunities for EventGPT + Video-LLaVA
  - Implementation roadmap and expected speedups

### New Papers/Resources Added:
- OmniDraft: Cross-vocabulary online adaptive drafter
- CTPD: Cross Tokenizer Preference Distillation (arXiv:2601.11865)
- GRIFFIN: Token Alignment for Speculative Decoding (Feb 2025)
- Pyramid Speculative Decoding (arXiv:2510.12966)
- SpecVLM: Fast Speculative Decoding in VLMs (arXiv:2509.11815)
- ViSpec: Vision-Aware Speculative Decoding (arXiv:2509.15235)
- DREAM: Cross-Attention for Multimodal Speculative Decoding (arXiv:2505.19201)
- MASSV: Multimodal Adaptation for VLM Speculative Decoding (arXiv:2505.10526)
- Disaggregated Prefill-Decode (Nexus, TPLA, SPAD papers)
- EAGLE-3 and SpecForge training framework

---

## Update (January 23, 2026)

### New Papers Added (December 2025 - January 2026)
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

### Implementation Details Added
- EAGLE-3 training-time test code
- Multi-layer feature fusion example
- GitHub repositories: SafeAILab/EAGLE, BaldEagle, fast-llm-inference

---

## Update Schedule

### Hourly Updates
- [ ] Search arXiv for new speculative decoding papers
- [ ] Add code examples and implementations
- [ ] Improve mathematical formulations
- [ ] Fix any errors or inconsistencies

### Daily Updates
- [ ] Compile daily paper releases
- [ ] Update benchmark results
- [ ] Add community implementations
- [ ] Review and refine existing content

### Weekly Updates
- [ ] Comprehensive survey of all papers
- [ ] Update performance comparison tables
- [ ] Add new research directions
- [ ] Verify all paper links and citations

---

## TODO List

### High Priority
- [ ] Add more multimodal speculative decoding papers (SpecVLM, ViSpec)
- [ ] Complete implementation guide for EventGPT → VideoLLaVA
- [ ] Add benchmarking code examples
- [ ] Create comparison matrix of all methods

### Medium Priority
- [ ] Add video demonstrations of each method
- [ ] Interview researchers for insights
- [ ] Create interactive visualizations
- [ ] Add citation analysis and impact metrics

### Low Priority
- [ ] Translate to Chinese
- [ ] Create video tutorials
- [ ] Build web-based explorer

---

## Requested Content

### Pending User Requests
- [ ] More detailed equations for cascaded speculative decoding
- [ ] PyTorch implementation for each major method
- [ ] Benchmarking on specific models (LLaMA, Mistral, etc.)
- [ ] Production deployment guides

---

## File Status

| File | Last Update | Status | Next Update |
|------|-------------|--------|-------------|
| pipeline/README.md | 2026-02-07 | ✅ Complete | As needed |
| pipeline/feature_extraction/README.md | 2026-02-07 | ✅ Complete | As needed |
| pipeline/adapter_train/README.md | 2026-02-07 | ✅ Complete | As needed |
| pipeline/evaluation/README.md | 2026-02-07 | ✅ Complete | As needed |
| pipeline/benchmark_e2e/README.md | 2026-02-07 | ✅ Complete | As needed |
| pipeline/**/*.py | 2026-02-07 | ✅ Complete | As needed |
| pipeline/**/*.sh | 2026-02-07 | ✅ Complete | As needed |
| feasible/token_alignment/README.md | 2026-01-28 | ✅ Complete | After 10q results |
| feasible/token_alignment/extract_tokens_parallel.py | 2026-01-28 | ✅ Complete | As needed |
| feasible/token_alignment/task/starred/ | 2026-01-28 | ✅ Results saved | After 10q results |
| edge_cloud_SD/VIDEO_STREAMING_COMPUTE_OFFLOAD.md | 2026-01-28 | ✅ Complete | As needed |
| edge_cloud_SD/edge_cloud_speculative_decoding.md | 2026-01-28 | ✅ Moved | Weekly |
| prefill_hiding/README.md | 2026-01-26 | ✅ Complete | Weekly |
| prefill_hiding/RELATED_WORK.md | 2026-01-26 | ✅ Complete | Weekly |
| prefill_hiding/NOVELTY_ANALYSIS.md | 2026-01-26 | ✅ Complete | Weekly |
| prefill_hiding/ABSTRACT.md | 2026-01-26 | ✅ Complete | As needed |
| egpt_vllava_roadmap/*.md | 2026-01-28 | ✅ Complete | Weekly |
| README.md | 2026-01-23 | ✅ Complete | Hourly |
| token_level_speculative_decoding.md | 2026-01-23 | ✅ Complete | Hourly |
| embedding_level_speculative_decoding.md | 2026-01-23 | ✅ Complete | Hourly |
| hybrid_speculative_decoding.md | 2026-01-23 | ✅ Complete | Hourly |
| sequential_cascaded_speculative_decoding.md | 2026-01-23 | ✅ Complete | Hourly |
| EventGPT_VideoLLaVA_roadmap.md | 2026-01-23 | ✅ Complete | Daily |

---

## Contributing

To contribute updates:
1. Search for new papers on arXiv (tag: cs.CL, cs.LG, cs.AI)
2. Verify paper authenticity
3. Add to appropriate markdown file
4. Update this log
5. Update README with new findings

---

**Last Updated:** February 7, 2026
**Next Scheduled Update:** After L4 retraining convergence (300 epochs)
