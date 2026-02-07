# Metrics Comparison: Hidden-State vs Token-Level

Generated: 2026-02-07T01:06:03.811199

## Summary

| Metric | Hidden-State (@0.90) | Token-Level (Top-1) |
|--------|---------------------|---------------------|
| Overall Accept Rate | 24.25% | 23.90% |
| Consecutive Accepts | 7.85 | 5.35 |
| E2E Speedup | 0.95x | 0.82x |

## Cosine Similarity as Token Proxy

| cos_sim threshold | P(token match | cos > thresh) | Count |
|-------------------|-------------------------------|-------|
| > 0.80 | 50.30% | 183111 |
| > 0.85 | 67.84% | 127737 |
| > 0.90 | 80.23% | 98496 |
| > 0.92 | 82.55% | 91198 |
| > 0.95 | 86.00% | 78396 |
| > 0.98 | 90.55% | 54821 |

Pearson r (cos_sim, token_match): **0.5916**

## Per-Position Comparison (First 10)

| Position | Hidden Accept @0.90 | Token Top-1 |
|----------|--------------------:|------------:|
| 0 | 100.00% | 94.34% |
| 1 | 99.98% | 95.98% |
| 2 | 98.85% | 74.28% |
| 3 | 88.02% | 89.11% |
| 4 | 86.92% | 74.88% |
| 5 | 72.88% | 73.63% |
| 6 | 68.76% | 49.12% |
| 7 | 55.42% | 51.02% |
| 8 | 54.16% | 35.73% |
| 9 | 32.62% | 34.22% |

## Two-Phase SD (Token-Level)

- gamma_prefill = 7
- Prefill consecutive accepts: 4.59
- gamma_decode = 5
- Decode consecutive accepts: 3.74
- E2E speedup estimate: 0.82x
