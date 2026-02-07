# 5-Stage Benchmark Report: my_egpt_dsec_train_1s

**Date:** 2026-01-27 23:46:48
**Samples:** 0
**Max New Tokens:** 50

## Configuration

| Setting | Value |
|---------|-------|
| Dataset | my_egpt_dsec_train_1s |
| Max Samples | -1 |
| Device | cuda |

## Acceptance Rate Analysis

| Metric | Value |
|--------|-------|
| **Acceptance Rate** | 0.0% ± 0.0% |
| Word Overlap | 0.0% |
| Character Similarity | 0.0% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | 0.0 ± 0.0 ms | 0.0 ± 0.0 ms | 1.00x |
| **Stage 2: Preprocessing** | 0.0 ± 0.0 ms | 0.0 ± 0.0 ms | 1.00x |
| **Stage 3: Vision Encoding** | 0.0 ± 0.0 ms | 0.0 ± 0.0 ms | 1.00x |
| **Stage 4: LLM Prefill** | 0.0 ± 0.0 ms | 0.0 ± 0.0 ms | 1.00x |
| **Stage 5: LLM Decode** | 0.0 ± 0.0 ms | 0.0 ± 0.0 ms | 1.00x |
| **TOTAL** | 0.0 ms | 0.0 ms | 1.00x |

## Stage Breakdown (Percentage)

### EventGPT
| Stage | Time | Percentage |
|-------|------|------------|
| Stage 1: Data Loading | 0.0 ms | 0.0% |
| Stage 2: Preprocessing | 0.0 ms | 0.0% |
| Stage 3: Vision Encoding | 0.0 ms | 0.0% |
| Stage 4: LLM Prefill | 0.0 ms | 0.0% |
| Stage 5: LLM Decode | 0.0 ms | 0.0% |

### Video-LLaVA
| Stage | Time | Percentage |
|-------|------|------------|
| Stage 1: Data Loading | 0.0 ms | 0.0% |
| Stage 2: Preprocessing | 0.0 ms | 0.0% |
| Stage 3: Vision Encoding | 0.0 ms | 0.0% |
| Stage 4: LLM Prefill | 0.0 ms | 0.0% |
| Stage 5: LLM Decode | 0.0 ms | 0.0% |

## Output Comparison

| Model | Output Tokens (avg) |
|-------|---------------------|
| EventGPT | 0.0 |
| Video-LLaVA | 0.0 |

## Parallel Prefill Analysis

| Metric | Value |
|--------|-------|
| Overlap Window | 0.0 ms |
| Theoretical Draft Tokens | 0.0 |
| Actual Draft Tokens | 0.0 |

## Key Findings

1. **EventGPT Bottleneck**: Stage 1 (0.0% of total time)
2. **Video-LLaVA Bottleneck**: Stage 1 (0.0% of total time)
3. **Acceptance Rate**: 0.0% - EventGPT drafts are poorly aligned with Video-LLaVA outputs
4. **Overall Speedup**: EventGPT is 1.00x faster than Video-LLaVA

---

*Generated: 2026-01-27 23:46:48*
*Author: Alice Zhang*
