# Speculative Decoding Research Collection

Accelerating Large Language Model and Vision-Language Model Inference through Speculative Decoding Techniques

---

## Overview

This repository contains research on **speculative decoding** techniques for accelerating LLM and VLM inference. The focus is on three key approaches:

1. **Token-Level Speculative Decoding** (Standard/Vanilla)
2. **Embedding/Feature-Level Speculative Decoding** (EAGLE-family)
3. **Hybrid and Cascaded Approaches** (Combining multiple strategies)

**Application Goal:** Accelerate VideoLLaVA inference using EventGPT as a draft model.

---

## Quick Start

### What is Speculative Decoding?

**Speculative decoding** accelerates model inference by:
1. Using a small **draft model** to predict future tokens
2. Verifying predictions with a larger **target model** in parallel
3. Accepting good predictions and resampling bad ones

**Result:** 2-6x speedup with no quality loss.

### Key Insight

> "Predict features, not tokens. Features contain more information and exhibit more regularity than final token distributions."

---

## Documents

| Document | Description | Focus | Status |
|----------|-------------|-------|--------|
| [**Token-Level Speculative Decoding**](token_level_speculative_decoding.md) | Standard/vanilla approach | Small LM → Large LM (token prediction) | ✅ Complete |
| [**Embedding-Level Speculative Decoding**](embedding_level_speculative_decoding.md) | Feature-level approach (EAGLE) | Hidden state prediction | ✅ Complete |
| [**Hybrid Speculative Decoding**](hybrid_speculative_decoding.md) | Combined approaches | Token + Feature, multiple draft models | ✅ Complete |
| [**Sequential/Cascaded Speculative Decoding**](sequential_cascaded_speculative_decoding.md) | Multi-stage acceleration + theoretical analysis | Chain of draft models (A → B → Target) | ✅ Complete |
| [**Edge-Cloud Speculative Decoding**](edge_cloud_speculative_decoding.md) | Distributed inference | Edge device → Cloud verification | ✅ New |
| [**Parallel Prefill & Speculative Decoding**](parallel_prefill_speculative_decoding_20260125_004920.md) | Cross-modal VLM acceleration | EventGPT draft + Video-LLaVA target opportunities | ✅ New |
| [**EventGPT → VideoLLaVA Roadmap**](EventGPT_VideoLLaVA_roadmap.md) | Application roadmap | Sparse events → dense video acceleration | ✅ Complete |

### Supporting Files

| File | Purpose |
|------|---------|
| [**CHANGELOG.md**](CHANGELOG.md) | Formal changelog (Keep a Changelog format) |
| [**UPDATE_LOG.md**](UPDATE_LOG.md) | Detailed update tracking and TODO list |
| [**UNIFIED_TEMPLATE.md**](UNIFIED_TEMPLATE.md) | Template for all markdown files |

---

## Document Structure

All documents follow a unified structure defined in [UNIFIED_TEMPLATE.md](UNIFIED_TEMPLATE.md):

1. **Overview** - Introduction and key insights
2. **Key Methods** - Main algorithms and techniques
3. **Theoretical Analysis** - Mathematical formulations
4. **Implementation** - Code examples and usage
5. **Performance** - Benchmarks and comparisons
6. **Critical Analysis** - Advantages, limitations, recommendations
7. **References** - Citations and resources
8. **Summary** - Key takeaways

---

## Research Taxonomy

```
SPECULATIVE DECODING
├── TOKEN-LEVEL (Standard)
│   ├── Vanilla Speculative Sampling (Chen et al., 2023)
│   ├── Speculative Decoding (Leviathan et al., 2023)
│   ├── Medusa (Cai et al., 2024)
│   ├── Hydra (Ankner et al., 2024)
│   └── Assisted Generation (HuggingFace)
│
├── FEATURE/EMBEDDING-LEVEL
│   ├── EAGLE (Li et al., 2024)
│   ├── EAGLE-2 (Li et al., 2024)
│   ├── EAGLE-3 (Li et al., 2025)
│   ├── GLIDE with CaPE (Du et al., 2024)
│   └── Chimera (Chen et al., 2024)
│
├── HYBRID APPROACHES
│   ├── Token + Feature Fusion
│   ├── Multiple Draft Models
│   ├── Adaptive Strategy Selection
│   └── Cascaded Drafting
│
├── SEQUENTIAL/CASCADED
│   ├── ReDrafter (Cheng et al., 2024)
│   ├── Cascade Speculative Drafting (Chen et al., 2024)
│   ├── Lookahead Decoding (Fu et al., 2024)
│   └── PaSS (Monea et al., 2023)
│
└── MULTIMODAL EXTENSIONS
    ├── SpecVLM (2025)
    ├── ViSpec (2025)
    └── EventGPT → VideoLLaVA (This Research)
```

---

## Key Papers Summary

### Foundational Works (2023)

| Paper | Venue | Citations | Key Idea |
|-------|-------|-----------|----------|
| [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) | arXiv 2023 | 642+ | Standard speculative sampling |
| [Fast Inference via Speculative Decoding](https://arxiv.org/abs/2211.17192) | ICML 2023 | 1,122+ | Transformer-specific optimization |
| [PaSS: Parallel Speculative Sampling](https://arxiv.org/abs/2311.13581) | NeurIPS 2023 | 146+ | Parallel draft generation |
| [SpecTr: Optimal Transport for Speculative Decoding](https://arxiv.org/abs/2310.15141) | NeurIPS 2023 | 146+ | Distribution matching via OT |

### Feature-Level Works (2024-2025)

| Paper | Venue | Citations | Speedup | Key Idea |
|-------|-------|-----------|---------|----------|
| [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) | ICML 2024 | - | 3-4x | Predict next feature, not token |
| [EAGLE-2: Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) | EMNLP 2024 | - | 4-5x | Context-aware tree pruning |
| [EAGLE-3: Training-Time Test](https://arxiv.org/abs/2503.01840) | arXiv 2025 | - | 5-6.5x | Multi-layer fusion, scaling law |
| [Chimera: Fusing All Tokens](https://arxiv.org/abs/2402.15758) | arXiv 2024 | 77 | 2.7x | Trigram + Full Context Encoder |
| [GLIDE with CaPE](https://arxiv.org/abs/2402.02082) | arXiv 2024 | - | 2.5-2.6x | KV cache reuse + proposal expansion |

### Multimodal Works (2024-2025)

| Paper | Venue | Focus | Speedup |
|-------|-------|-------|---------|
| [SpecVLM: Fast Speculative Decoding in VLMs](https://arxiv.org/abs/2509.11815) | arXiv 2025 | Vision-language models | TBD |
| [ViSpec: Vision-Aware Speculative Decoding](https://arxiv.org/abs/2509.15235) | NeurIPS 2025 | Vision feature reuse | 3.2x |
| [Spec-VLA: VLA Acceleration](https://arxiv.org/abs/2507.22424) | EMNLP 2025 | Vision-language-action | TBD |

---

## Performance Comparison

### Speedup by Approach

| Approach | Typical Speedup | Complexity | Production Ready |
|----------|-----------------|------------|------------------|
| **Token-Level** | 2-3x | Low | ✅ Yes (HuggingFace, vLLM) |
| **Feature-Level (EAGLE)** | 3-5x | Medium | ✅ Yes (GitHub) |
| **Feature-Level (EAGLE-3)** | 5-6.5x | Medium | ⚠️ New (2025) |
| **Hybrid** | 2.5-4x | High | ⚠️ Research |
| **Cascaded** | 3-5x | High | ⚠️ Research |

### Acceptance Rates

| Method | Acceptance Rate | Notes |
|--------|-----------------|-------|
| Same-family token-level | 70-90% | e.g., GPT-2 → GPT-2 XL |
| Cross-family token-level | 40-60% | e.g., LLaMA → GPT |
| Feature-level (EAGLE) | 60-80% | More stable |
| Feature-level (EAGLE-3) | 80-95% | Training-time test helps |

---

## Our Research: EventGPT → VideoLLaVA

### Goal

Accelerate VideoLLaVA-7B/13B inference using EventGPT as a draft model through:

1. **Stage 1: Embedding-Level (Vision)**
   - EventGPT encoder (sparse, fast) → VideoLLaVA encoder (dense, slow)
   - Feature alignment layer
   - Target: 10-50x speedup on vision encoding

2. **Stage 2: Token-Level (Language)**
   - EventGPT LM (1B) → VideoLLaVA LM (7B/13B)
   - Shared vision features (cached)
   - Target: 2-3x speedup on text generation

### Expected Overall Performance

| Metric | Baseline | With Speculative |
|--------|----------|-------------------|
| Total Time | 600 ms | 210 ms |
| Speedup | 1x | **2.9x** |
| VideoQA Accuracy | 75% | 74-75% |

### Roadmap

- [ ] Phase 1: Alignment layer training (Weeks 1-4)
- [ ] Phase 2: Language model training (Weeks 5-8)
- [ ] Phase 3: End-to-end integration (Weeks 9-12)

**See:** [EventGPT → VideoLLaVA Roadmap](EventGPT_VideoLLaVA_roadmap.md)

---

## Code Examples

### Token-Level Speculative Decoding

```python
from transformers import AutoModelForCausalLM

# Load models
draft_model = AutoModelForCausalLM.from_pretrained("gpt2")
target_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Generate with speculative decoding
outputs = target_model.generate(
    input_ids,
    assistant_model=draft_model,  # HuggingFace syntax
    do_sample=True,
    max_new_tokens=100
)
```

### Feature-Level (EAGLE-Style)

```python
# Predict next feature instead of token
features = target_model.get_features(input_ids)

# Draft model predicts next feature
draft_feature = draft_model.predict_feature(features)

# Use target model's LM head to get token
token = target_model.lm_head(draft_feature)

# Verify with target model
accepted = verify_with_target(draft_feature, target_model)
```

---

## Key Equations

### Acceptance Probability

```
α = min(1, p_target / p_draft)

Where:
- p_target: Target model probability
- p_draft: Draft model probability
```

### Expected Speedup

```
Speedup = (k + 1) / (1 + k × (T_draft / T_target))

Where:
- k: Average accepted tokens per cycle
- T_draft: Draft model forward pass time
- T_target: Target model forward pass time
```

### Resampling Distribution (on Rejection)

```
P_resample(t) = max(0, P_target(t) - P_draft(t)) / Z

Where Z normalizes the distribution.
```

---

## Resources

### Libraries and Frameworks

| Resource | Description | Link |
|----------|-------------|------|
| **HuggingFace Transformers** | Assisted Generation API | [Docs](https://huggingface.co/blog/assisted-generation) |
| **vLLM** | Production speculative decoding | [Docs](https://docs.vllm.ai/en/latest/features/spec_decode/) |
| **EAGLE** | Feature-level implementation | [GitHub](https://github.com/SafeAILab/EAGLE) |
| **Medusa** | Multi-head decoding | [GitHub](https://github.com/FasterDecoding/Medusa) |
| **ReDrafter** | Apple + NVIDIA implementation | [GitHub](https://github.com/apple/ml-recurrent-drafter) |
| **Lookahead Decoding** | N-gram + Jacobi iteration | [GitHub](https://github.com/hao-ai-lab/LookaheadDecoding) |

### Survey Papers

1. [A Comprehensive Survey of Speculative Decoding (Xia et al., ACL 2024)](https://aclanthology.org/2024.findings-acl.456.pdf) - 198 citations

2. [Decoding Speculative Decoding (He et al., NAACL 2025)](https://aclanthology.org/2025.naacl-long.328.pdf) - Meta-analysis

3. [NVIDIA: Introduction to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

---

## Latest Research (December 2025 - January 2026)

### Recent Papers

| Paper | arXiv | Focus | Innovation |
|-------|-------|-------|------------|
| Entropy-Aware Speculative Decoding | [2512.23765](https://arxiv.org/abs/2512.23765) | Draft quality | Entropy awareness |
| SpecPV (Long Context) | [2512.02337](https://arxiv.org/abs/2512.02337) | Long-context | Partial verification |
| Sparse Computation | [2512.21911](https://arxiv.org/abs/2512.21911) | Efficiency | Sparse + speculative |
| Adaptive Rejection Sampling | [2512.13194](https://arxiv.org/abs/2512.13194) | Verification | Adaptive rejection |
| Optimal Lower Bounds | [2512.11718](https://arxiv.org/abs/2512.11718) | Theory | Branching random walks |
| Speculative Sampling + RL | [2601.12212](https://arxiv.org/abs/2601.12212) | RL hybrid | Feature reuse |
| Judge Decoding Revisited | [2601.04766](https://arxiv.org/abs/2601.04766) | Theory | First principles |
| Multi-Scale Image Gen | [2601.05149](https://arxiv.org/abs/2601.05149) | Images | Multi-scale local |
| SRT for RL | [2601.09083](https://arxiv.org/abs/2601.09083) | RL | Tree-structured cache |
| Plan, Verify, Fill | [2601.12247](https://arxiv.org/abs/2601.12247) | Diffusion | Structured parallel |

### Emerging Trends

1. **Long-Context Optimization** - SpecPV addresses long-context generation
2. **Cross-Domain Applications** - Image generation, reinforcement learning
3. **Theoretical Foundations** - Optimal lower bounds, first principles analysis
4. **Hybrid Approaches** - Combining with RL, sparse computation
5. **Verification Improvements** - Adaptive rejection sampling, entropy-aware methods

---

## Citation

If you use this research or find these documents helpful, please cite:

```bibtex
@misc{eventgpt_speculative_research,
  title={Speculative Decoding Research: Accelerating LLM and VLM Inference},
  author={Research Team},
  year={2026},
  url={https://github.com/your-repo}
}
```

---

## Contributing

This is an active research project. Contributions welcome:
- Additional paper summaries
- Code implementations
- Benchmark results
- Corrections and improvements

---

## License

Research materials licensed under MIT License.

---

## Acknowledgments

- EAGLE team (Peking University, Microsoft Research)
- Apple ML Research (ReDrafter)
- HuggingFace (Assisted Generation)
- vLLM team
- NVIDIA research team

---

**Last Updated:** January 25, 2026

**Contact:** For questions or collaboration, open an issue or contact the research team.
