# EventGPT Research: File Structure

## Overview

This document describes the organization of research files and benchmark results for the EventGPT → VideoLLaVA cross-modal speculative decoding project.

```
EventGPT/
├── research/                                    # Research documents on speculative decoding
│   ├── README.md                               # Overview of research folder
│   ├── UPDATE_LOG.md                           # Research update log
│   ├── UNIFIED_TEMPLATE.md                     # Template for paper summaries
│   │
│   ├── Core Concept Documents
│   │   ├── token_level_speculative_decoding.md     # Token-level speculative (standard)
│   │   ├── embedding_level_speculative_decoding.md # Feature/hidden-state (EAGLE)
│   │   ├── hybrid_speculative_decoding.md          # Hybrid approaches
│   │   ├── sequential_cascaded_speculative_decoding.md  # Cascade methods
│   │   ├── edge_cloud_speculative_decoding.md      # Edge-cloud deployment
│   │   └── cross_modal_speculative_prefill.md      # NEW: Events→Video prefill analysis
│   │
│   ├── Project-Specific Documents
│   │   └── EventGPT_VideoLLaVA_roadmap.md          # Full roadmap for EventGPT→VideoLLaVA
│   │
│   └── arxiv/                                  # Paper summaries by year
│       ├── README.md
│       ├── TUTORIAL_PAPERS_TRACKER.md
│       ├── 2024/
│       │   └── 2404.08856_multimodal_drafting.md
│       ├── 2025/
│       │   ├── 2505.14260_multimodal_reimagined.md
│       │   ├── 2509.11815_specvlm.md
│       │   └── 2509.15235_vispec.md
│       └── edge_cloud/
│           └── spec_vla.md
│
└── feasible/
    └── benchmark_inference/                    # Benchmark results and analysis
        │
        ├── Benchmark Scripts
        │   ├── benchmark_inference.py              # 3-stage benchmark
        │   ├── benchmark_inference_4stages.py      # 4-stage benchmark with decoupling
        │   ├── benchmark_inference_5stages.py      # 5-stage benchmark
        │   ├── benchmark_inference_properly_decoupled.py
        │   ├── benchmark_with_adapter_S1.py        # Feature alignment benchmark
        │   ├── benchmark_alignment_S1.py
        │   ├── benchmark_comparison_eventgpt_vs_videollava.py
        │   ├── benchmark_egpt_vs_videollava.py
        │   ├── speculative_decoding_S1.py          # Token-level speculative
        │   ├── shared_decoder_speculative_S1.py
        │   ├── verify_videollava_decoupling.py
        │   └── run_all_benchmarks.py
        │
        ├── Analysis Scripts
        │   ├── analyze_stage4.py
        │   ├── analyze_stage4_5_shapes.py
        │   └── analyze_1frame_vs_8frames.py
        │
        ├── Benchmark Results (JSON)
        │   ├── benchmark_results_S1.json           # Stage timing: EventGPT vs VideoLLaVA
        │   ├── benchmark_results_S1_train.json      # Extended training split results
        │   ├── benchmark_adapter_full_S1.json      # Feature alignment: 86.8% acceptance
        │   ├── alignment_metrics_S1.json
        │   ├── speculative_benchmark_1s_results.json  # Token speculative: 1.24%
        │   ├── speculative_results_S1.json
        │   ├── speculative_test_results.json
        │   ├── shared_decoder_results_S1.json
        │   ├── benchmark_1frame_vs_8frames_20260124_175006.json
        │   └── benchmark_1frame_vs_8frames_20260124_193405.json
        │
        ├── Checkpoints
        │   └── shared_decoder_adapter_S1.pt        # Trained alignment adapter
        │
        ├── Analysis Reports (Markdown)
        │   ├── FINAL_BENCHMARK_REPORT.md
        │   ├── BENCHMARK_ANALYSIS_SUMMARY.md
        │   ├── BENCHMARK_1FRAME_VS_8FRAMES_20260124_175006.md
        │   ├── BENCHMARK_1FRAME_VS_8FRAMES_20260124_193405.md
        │   ├── EVENTGPT_1FRAME_VS_VIDEOLLAVA_8FRAMES.md
        │   ├── benchmark_inference_1s_dataset_20260124.md
        │   ├── BENCHMARK_RESULTS_1S_200SAMPLES.md
        │   ├── IMPLEMENTATION_SUMMARY.md
        │   ├── COMPLETE_DECOUPLING_STRATEGY.md
        │   ├── DECOUPLING_SOLUTION_SUMMARY.md
        │   ├── PROPER_STAGE_DECOUPLING.md
        │   ├── STAGE_DECOUPLING_ANALYSIS.md
        │   ├── STAGE_DATA_FORMATS.md
        │   ├── STAGE4_ANALYSIS.md
        │   ├── STAGE4_5_SHAPE_ANALYSIS.md
        │   ├── 4STAGE_DATA_FLOW_ANALYSIS.md
        │   ├── VIDEOLLAVA_DECOUPLING_VERIFICATION.md
        │   ├── egpt_faster_analysis.md
        │   └── PROGRESS_UPDATE.md
        │
        └── Log Files
            ├── benchmark_full_1s.log
            ├── benchmark_run.log
            ├── benchmark_500ms_compare.log
            ├── speculative_benchmark_1s.log
            ├── benchmark_log_20260123.txt
            ├── benchmark_log_20260123_153219.txt
            ├── benchmark_results_200samples_20260123.log
            └── stage4_analysis.log
```

## Document Categories

### 1. Core Speculative Decoding Methods

| File | Description | Key Papers |
|------|-------------|------------|
| `token_level_speculative_decoding.md` | Standard token-level speculative sampling | Chen 2023, Leviathan 2023 |
| `embedding_level_speculative_decoding.md` | Feature/hidden-state level (EAGLE) | EAGLE, EAGLE-2, EAGLE-3 |
| `hybrid_speculative_decoding.md` | Combining token + feature approaches | Chimera, GLIDE+CaPE |
| `sequential_cascaded_speculative_decoding.md` | Multi-stage cascade (A→B→Target) | ReDrafter, Lookahead |
| `edge_cloud_speculative_decoding.md` | Distributed edge-cloud deployment | SLED, DuoDecoding |

### 2. Project-Specific Research

| File | Description | Status |
|------|-------------|--------|
| `cross_modal_speculative_prefill.md` | Events→Video prefill-hidden drafting | NEW |
| `EventGPT_VideoLLaVA_roadmap.md` | Complete implementation roadmap | Active |

### 3. Benchmark Results Summary

| Metric | EventGPT | VideoLLaVA | Speedup |
|--------|----------|------------|---------|
| **Stage 1 (Vision)** | 44-66 ms | 239-568 ms | **5-9x** |
| **Stage 2 (Projector)** | 17-22 ms | 72-104 ms | **4-6x** |
| **Stage 3 (Decode)** | 850-1360 ms | 1040-2040 ms | **1.2-1.5x** |
| **Feature Acceptance** | - | 86.8% @ 0.8 | Alignment adapter |
| **Token Acceptance** | 1.24% | - | Infeasible |

### 4. ArXiv Paper Summaries

| Year | Paper | Venue | Topic |
|------|-------|-------|-------|
| 2024 | 2404.08856 | - | Multimodal drafting |
| 2025 | 2505.14260 | - | Multimodal reimaged |
| 2025 | 2509.11815 | - | SpecVLM |
| 2025 | 2509.15235 | - | ViSpec |

## Key Findings

### Feature-Level Alignment (Viable)
- **Acceptance Rate:** 86.8% at threshold 0.8
- **Adapter:** Lightweight MLP trained on 5,208 samples
- **Gain:** 5-9x speedup on vision encoding

### Token-Level Speculative (Not Viable)
- **Acceptance Rate:** 1.24%
- **Issue:** Different tokenizers, mismatched distributions
- **Conclusion:** Requires extensive retraining

### Overall Speedup
- **Expected:** 1.24x end-to-end (18% latency reduction)
- **Vision-only:** 5-9x speedup
- **Bottleneck:** Language decoding dominates total time

## Quick Reference

### Reading Order for New Researchers

1. `token_level_speculative_decoding.md` - Understand basics
2. `embedding_level_speculative_decoding.md` - Learn EAGLE methods
3. `EventGPT_VideoLLaVA_roadmap.md` - Project overview
4. `cross_modal_speculative_prefill.md` - Our specific approach

### Benchmark Scripts

```bash
# Run 4-stage decoupled benchmark
python feasible/benchmark_inference/benchmark_inference_4stages.py

# Run feature alignment benchmark
python feasible/benchmark_inference/benchmark_with_adapter_S1.py

# Run token-level speculative
python feasible/benchmark_inference/speculative_decoding_S1.py
```

### Key Result Files

| File | Contains |
|------|----------|
| `benchmark_results_S1.json` | Stage timing breakdown |
| `benchmark_adapter_full_S1.json` | Alignment metrics |
| `speculative_benchmark_1s_results.json` | Token acceptance rates |

---

**Last Updated:** January 24, 2026
