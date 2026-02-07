# Metrics Comparison: Hidden-State vs Token-Level

Generated: 2026-02-07T01:07:21.691995

## Summary

| Metric | Hidden-State (@0.90) | Token-Level (Top-1) |
|--------|---------------------|---------------------|
| Overall Accept Rate | 61.75% | 19.11% |
| Consecutive Accepts | 0.08 | 0.61 |
| E2E Speedup | 0.21x | 0.31x |

## Cosine Similarity as Token Proxy

| cos_sim threshold | P(token match | cos > thresh) | Count |
|-------------------|-------------------------------|-------|
| > 0.80 | 17.94% | 398402 |
| > 0.85 | 16.43% | 367430 |
| > 0.90 | 13.14% | 250777 |
| > 0.92 | 12.63% | 187293 |
| > 0.95 | 16.47% | 52504 |
| > 0.98 | 58.70% | 6404 |

Pearson r (cos_sim, token_match): **-0.2505**

## Per-Position Comparison (First 10)

| Position | Hidden Accept @0.90 | Token Top-1 |
|----------|--------------------:|------------:|
| 0 | 1.96% | 50.95% |
| 1 | 40.36% | 14.89% |
| 2 | 33.95% | 38.60% |
| 3 | 77.58% | 22.65% |
| 4 | 62.49% | 2.99% |
| 5 | 68.05% | 32.35% |
| 6 | 73.95% | 3.92% |
| 7 | 51.65% | 10.30% |
| 8 | 56.64% | 10.03% |
| 9 | 54.03% | 24.43% |

## Two-Phase SD (Token-Level)

- gamma_prefill = 7
- Prefill consecutive accepts: 0.61
- gamma_decode = 5
- Decode consecutive accepts: 0.61
- E2E speedup estimate: 0.31x
