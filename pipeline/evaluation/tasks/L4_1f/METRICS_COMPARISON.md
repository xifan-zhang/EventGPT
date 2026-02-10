# Metrics Comparison: Hidden-State vs Token-Level

Generated: 2026-02-09T06:59:56.382183

## Summary

| Metric | Hidden-State (@0.90) | Token-Level (Top-1) |
|--------|---------------------|---------------------|
| Overall Accept Rate | 27.99% | 27.07% |
| Consecutive Accepts | 8.02 | 5.66 |
| E2E Speedup | 0.95x | 0.84x |

## Cosine Similarity as Token Proxy

| cos_sim threshold | P(token match | cos > thresh) | Count |
|-------------------|-------------------------------|-------|
| > 0.80 | 53.66% | 181861 |
| > 0.85 | 69.13% | 134073 |
| > 0.90 | 80.62% | 105739 |
| > 0.92 | 83.61% | 97601 |
| > 0.95 | 87.82% | 83192 |
| > 0.98 | 91.96% | 59367 |

Pearson r (cos_sim, token_match): **0.6130**

## Per-Position Comparison (First 10)

| Position | Hidden Accept @0.90 | Token Top-1 |
|----------|--------------------:|------------:|
| 0 | 100.00% | 94.30% |
| 1 | 99.89% | 96.88% |
| 2 | 98.93% | 80.35% |
| 3 | 87.49% | 89.21% |
| 4 | 86.80% | 75.55% |
| 5 | 74.91% | 71.84% |
| 6 | 68.44% | 53.78% |
| 7 | 55.65% | 47.77% |
| 8 | 53.77% | 39.04% |
| 9 | 36.68% | 37.54% |

## Two-Phase SD (Token-Level)

- gamma_prefill = 7
- Prefill consecutive accepts: 4.88
- gamma_decode = 5
- Decode consecutive accepts: 3.91
- E2E speedup estimate: 0.84x
