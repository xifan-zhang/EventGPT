# Speculative Decoding Research - Update Log

This file tracks all updates made to the research markdown files.

---

## Latest Update (January 28, 2026) - Cascaded SD Research

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
| prefill_hiding/README.md | 2026-01-26 | ✅ Complete | Weekly |
| prefill_hiding/RELATED_WORK.md | 2026-01-26 | ✅ Complete | Weekly |
| prefill_hiding/NOVELTY_ANALYSIS.md | 2026-01-26 | ✅ Complete | Weekly |
| prefill_hiding/ABSTRACT.md | 2026-01-26 | ✅ Complete | As needed |
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

**Last Updated:** January 26, 2026
**Next Scheduled Update:** Hourly
