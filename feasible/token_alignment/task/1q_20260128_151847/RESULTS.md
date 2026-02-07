# Token Alignment Results

**Date:** 2026-01-28 15:20:03

## Summary

| Dataset | Samples | Baseline | Model | Top-5 | Improvement |
|---------|---------|----------|-------|-------|-------------|
| Train | 5200 | 1.77% | 27.21% | 51.40% | +25.45% |
| Test | 1100 | 1.58% | 27.90% | 51.64% | +26.32% |

## 5-Stage Timing (Test Set, ms)

| Stage | EventGPT | Video-LLaVA |
|-------|----------|-------------|
| stage1 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| stage2 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| stage3 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| stage4 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| stage5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| total | 0.0 ± 0.0 | 0.0 ± 0.0 |

## Training Curves

![Training Curves](training_curves.png)

- Loss curve: [loss_curve.png](loss_curve.png)
- Accuracy curve: [accuracy_curve.png](accuracy_curve.png)

## Configuration

- Epochs: 50 (early stopping: 10)
- Batch size: 32
- Learning rate: 0.0001
- Model parameters: 45,486,346

## Speedup Analysis

With acceptance rate α = 27.9% and γ = 5 draft tokens:
- Theoretical speedup: 1.39x
