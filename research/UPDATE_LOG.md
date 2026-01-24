# Speculative Decoding Research - Update Log

This file tracks all updates made to the research markdown files.

---

## Latest Update (January 25, 2026)

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
| README.md | 2026-01-23 | ✅ Complete | Hourly |
| token_level_speculative_decoding.md | 2026-01-23 | ✅ Complete | Hourly |
| embedding_level_speculative_decoding.md | 2026-01-23 | ✅ Complete | Hourly |
| hybrid_speculative_decoding.md | 2026-01-23 | ✅ Complete | Hourly |
| sequential_cascaded_speculative_decoding.md | 2026-01-23 | ✅ Complete | Hourly |
| EventGPT_VideoLLaVA_roadmap.md | 2026-01-23 | ✅ Complete | Daily |
| parallel_prefill_speculative_decoding_20260125.md | 2026-01-25 | ✅ Complete | Weekly |

---

## Contributing

To contribute updates:
1. Search for new papers on arXiv (tag: cs.CL, cs.LG, cs.AI)
2. Verify paper authenticity
3. Add to appropriate markdown file
4. Update this log
5. Update README with new findings

---

**Last Updated:** January 25, 2026
**Next Scheduled Update:** Hourly
