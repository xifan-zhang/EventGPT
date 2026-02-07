# Metrics Comparison: Hidden-State vs Token-Level

Generated: 2026-02-07T01:00:18.127349

## Summary

| Metric | Hidden-State (@0.90) | Token-Level (Top-1) |
|--------|---------------------|---------------------|
| Overall Accept Rate | 16.48% | 16.98% |
| Consecutive Accepts | 5.96 | 3.93 |
| E2E Speedup | 0.91x | 0.78x |

## Cosine Similarity as Token Proxy

| cos_sim threshold | P(token match | cos > thresh) | Count |
|-------------------|-------------------------------|-------|
| > 0.80 | 42.48% | 150796 |
| > 0.85 | 69.16% | 83242 |
| > 0.90 | 79.39% | 66921 |
| > 0.92 | 81.54% | 63182 |
| > 0.95 | 81.71% | 55834 |
| > 0.98 | 82.67% | 35856 |

Pearson r (cos_sim, token_match): **0.5306**

## Per-Position Comparison (First 10)

| Position | Hidden Accept @0.90 | Token Top-1 |
|----------|--------------------:|------------:|
| 0 | 100.00% | 91.07% |
| 1 | 99.96% | 96.05% |
| 2 | 98.75% | 68.41% |
| 3 | 76.11% | 78.35% |
| 4 | 65.80% | 62.51% |
| 5 | 48.47% | 46.73% |
| 6 | 42.11% | 16.10% |
| 7 | 22.29% | 18.78% |
| 8 | 15.67% | 23.88% |
| 9 | 8.32% | 18.25% |

## Two-Phase SD (Token-Level)

- gamma_prefill = 7
- Prefill consecutive accepts: 3.82
- gamma_decode = 5
- Decode consecutive accepts: 3.47
- E2E speedup estimate: 0.78x
