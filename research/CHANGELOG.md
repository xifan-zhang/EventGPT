# Changelog

All notable changes to the EventGPT Speculative Decoding Research are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Planned
- Feature-level speculative decoding implementation for EventGPT + Video-LLaVA
- Cross-vocabulary mapping experiments
- EAGLE-3 style multi-layer feature fusion training

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
| `parallel_prefill_speculative_decoding_20260125_004920.md` | 2026-01-25 | Cross-modal VLM acceleration research |
| `edge_cloud_speculative_decoding.md` | 2026-01-23 | Distributed inference |
| `EventGPT_VideoLLaVA_roadmap.md` | 2026-01-22 | Application roadmap |
| `sequential_cascaded_speculative_decoding.md` | 2026-01-22 | Multi-stage chains |
| `hybrid_speculative_decoding.md` | 2026-01-22 | Combined approaches |
| `embedding_level_speculative_decoding.md` | 2026-01-22 | EAGLE-family |
| `token_level_speculative_decoding.md` | 2026-01-22 | Standard approach |
| `cross_modal_speculative_prefill.md` | 2026-01-22 | Cross-modal prefilling |

---

## Statistics

| Metric | Count |
|--------|-------|
| Total Documents | 8 |
| Papers Surveyed | 50+ |
| Code Examples | 15+ |
| Research Opportunities Identified | 5 |

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
**Last Updated:** 2026-01-25
