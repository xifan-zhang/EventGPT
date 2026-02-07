# arXiv Papers Collection

This folder contains summaries and analysis of multi-modal speculative decoding papers collected from arXiv and other sources.

**Last Updated:** January 23, 2026
**Update Frequency:** Daily

---

## Directory Structure

```
arxiv/
├── README.md                           # This file - index of all papers
├── 2024/
│   ├── 2404.08856_multimodal_drafting.md   # "On Speculative Decoding for Multimodal Large Language Models"
│   └── ...
├── 2025/
│   ├── 2509.11815_specvlm.md             # "SpecVLM: Fast Speculative Decoding in Vision-Language Models"
│   ├── 2509.15235_vispec.md              # "ViSpec: Vision-Aware Speculative Decoding"
│   ├── 2505.14260_multimodal_reimagined.md # "Speculative Decoding Reimagined for Multimodal LLMs"
│   └── ...
└── edge_cloud/                          # Edge-cloud specific approaches
    └── ...
```

---

## Papers by Category

### Vision-Language Model Acceleration

| Paper | arXiv | Year | Venue | Speedup | Focus |
|-------|-------|------|-------|--------|-------|
| [SpecVLM](2025/2509.11815_specvlm.md) | 2509.11815 | 2025 | - | 2.5-2.9x | EAGLE-2 style for VLMs |
| [ViSpec](2025/2509.15235_vispec.md) | 2509.15235 | 2025 | NeurIPS 2025 | 3.22x | Vision-aware adapter |
| [On Multimodal Drafting](2024/2404.08856_multimodal_drafting.md) | 2404.08856 | 2024 | - | - | First multimodal application |
| [Multimodal Reimagined](2025/2505.14260_multimodal_reimagined.md) | 2505.14260 | 2025 | - | MSD framework |

### Vision-Language-Action Models

| Paper | arXiv | Year | Venue | Focus |
|-------|-------|------|-------|-------|
| [Spec-VLA](edge_cloud/spec_vla.md) | 2507.22424 | 2025 | EMNLP 2025 | Vision-Language-Action |

### Edge-Cloud Approaches

| Paper | Year | Focus | Status |
|-------|------|-------|--------|
| Edge-Cloud Speculative Decoding | 2025 | Distributed inference | 📝 In progress |

---

## Paper Summaries

Each paper summary includes:
- **Bibliographic information**
- **Key contributions**
- **Technical approach**
- **Results and benchmarks**
- **Code availability**
-**Relevance to EventGPT → VideoLLaVA**

---

## Collection Methodology

### Sources
- arXiv daily releases (cs.CL, cs.CV, cs.LG)
- NeurIPS, ICML, ICLR, EMNLP proceedings
- Major conference workshops
- Preprint servers

### Selection Criteria
Papers are included if they:
1. Apply speculative decoding to multi-modal models
2. Propose novel acceleration techniques for VLMs
3. Achieve significant speedup (>2x) with quality preservation
4. Include implementation details or code

### Daily Update Process

1. **Morning (6 AM UTC):** Search arXiv for new papers
2. **Afternoon (12 PM UTC):** Review and summarize key papers
3. **Evening (6 PM UTC):** Update indices and cross-references

---

## Contribution Guidelines

To add a paper to this collection:

1. Download the paper (PDF/HTML)
2. Create summary markdown file: `YYYY/arxiv_id_title.md`
3. Include all required sections (see template)
4. Add to this index
5. Commit with descriptive message

### Template

See `PAPER_TEMPLATE.md` for the standard paper summary format.

---

## Statistics

**Total Papers:** 7 (completed summaries)

**By Year:**
- 2024: 1 paper
- 2025: 6 papers

**By Venue:**
- NeurIPS: 1
- EMNLP: 2
- arXiv preprints: 7

**Research Areas:**
- Vision-language model acceleration: 5
- Vision-language-action: 1
- Edge-cloud distributed inference: 1

---

## Completed Summaries

### 2024
- ✅ [2404.08856 - On Speculative Decoding for Multimodal LLMs](2024/2404.08856_multimodal_drafting.md)

### 2025
- ✅ [2509.11815 - SpecVLM](2025/2509.11815_specvlm.md)
- ✅ [2509.15235 - ViSpec](2025/2509.15235_vispec.md)
- ✅ [2505.14260 - Multimodal Reimagined (MSD)](2025/2505.14260_multimodal_reimagined.md)

### Edge-Cloud / VLA
- ✅ [Spec-VLA](edge_cloud/spec_vla.md)

---

## Pending Summaries (from speculative-decoding.github.io)

**Status:** 📋 Track and complete when search quota resets (Feb 1, 2026)

Known tutorial categories:
1. Foundational papers (token-level)
2. Feature-level methods (EAGLE family)
3. Multimodal extensions
4. Edge and distributed approaches
5. Latest advances (2025-2026)

**Action:** Visit https://speculative-decoding.github.io/ after quota reset to extract full paper list

---

## Related Collections

- `../edge_cloud_speculative_decoding.md` - Edge-cloud approaches
- `../EventGPT_VideoLLaVA_roadmap.md` - Our application roadmap
- `../embedding_level_speculative_decoding.md` - Feature-level methods
- `../token_level_speculative_decoding.md` - Token-level methods

---

## Acknowledgments

Papers are collected from:
- arXiv.org
- OpenReview.net
- ACL Anthology
- Direct author submissions

All papers are freely available as of January 2026.

---

**Contact:** For suggestions or additions, open an issue or PR.

**Last Updated:** January 23, 2026
**Next Update:** Daily (quota permitting)
