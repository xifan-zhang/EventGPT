# Metrics Comparison: Hidden-State vs Token-Level

Generated: 2026-02-07T01:00:52.220112

## Summary

| Metric | Hidden-State (@0.90) | Token-Level (Top-1) |
|--------|---------------------|---------------------|
| Overall Accept Rate | 21.94% | 22.59% |
| Consecutive Accepts | 7.29 | 5.18 |
| E2E Speedup | 0.94x | 0.84x |

## Cosine Similarity as Token Proxy

| cos_sim threshold | P(token match | cos > thresh) | Count |
|-------------------|-------------------------------|-------|
| > 0.80 | 48.16% | 180152 |
| > 0.85 | 67.09% | 120676 |
| > 0.90 | 80.41% | 89094 |
| > 0.92 | 82.84% | 82280 |
| > 0.95 | 87.35% | 70042 |
| > 0.98 | 92.30% | 49579 |

Pearson r (cos_sim, token_match): **0.5790**

## Per-Position Comparison (First 10)

| Position | Hidden Accept @0.90 | Token Top-1 |
|----------|--------------------:|------------:|
| 0 | 100.00% | 94.09% |
| 1 | 99.98% | 95.63% |
| 2 | 98.84% | 82.75% |
| 3 | 83.05% | 85.35% |
| 4 | 84.93% | 72.31% |
| 5 | 73.75% | 72.59% |
| 6 | 68.48% | 44.69% |
| 7 | 50.85% | 41.01% |
| 8 | 48.17% | 31.10% |
| 9 | 26.64% | 29.85% |

## Two-Phase SD (Token-Level)

- gamma_prefill = 7
- Prefill consecutive accepts: 4.72
- gamma_decode = 5
- Decode consecutive accepts: 3.88
- E2E speedup estimate: 0.84x
