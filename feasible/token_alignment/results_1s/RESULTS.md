# Token Alignment Results

**Date:** 2026-01-27 21:53:39

## Summary

| Dataset | Samples | Baseline | Model | Top-5 | Improvement |
|---------|---------|----------|-------|-------|-------------|
| Train | 200 | 3.70% | 13.79% | 32.85% | +10.09% |
| Test | 1100 | 1.58% | 26.66% | 43.91% | +25.08% |

## 5-Stage Timing (Test Set, ms)

| Stage | EventGPT | Video-LLaVA |
|-------|----------|-------------|
| stage1 | 1.8 ± 0.1 | 79.3 ± 15.7 |
| stage2 | 3.6 ± 0.1 | 62.3 ± 1.5 |
| stage3 | 8.1 ± 3.5 | 0.0 ± 0.0 |
| stage4 | 83.2 ± 1.8 | 315.5 ± 1.1 |
| stage5 | 472.9 ± 43.9 | 724.0 ± 1.6 |
| total | 569.6 ± 43.8 | 1181.1 ± 16.4 |

## Configuration

- Epochs: 50
- Batch size: 32
- Learning rate: 0.0001
- Model parameters: 45,476,096

## Speedup Analysis

With acceptance rate α = 26.7% and γ = 5 draft tokens:
- Theoretical speedup: 1.36x
