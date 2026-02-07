# Metrics Comparison: Hidden-State vs Token-Level

Generated: 2026-02-07T01:05:25.849785

## Summary

| Metric | Hidden-State (@0.90) | Token-Level (Top-1) |
|--------|---------------------|---------------------|
| Overall Accept Rate | 24.11% | 24.39% |
| Consecutive Accepts | 7.62 | 5.48 |
| E2E Speedup | 0.94x | 0.85x |

## Cosine Similarity as Token Proxy

| cos_sim threshold | P(token match | cos > thresh) | Count |
|-------------------|-------------------------------|-------|
| > 0.80 | 50.97% | 183943 |
| > 0.85 | 68.43% | 128410 |
| > 0.90 | 80.91% | 97930 |
| > 0.92 | 83.22% | 90231 |
| > 0.95 | 87.19% | 77339 |
| > 0.98 | 91.35% | 55112 |

Pearson r (cos_sim, token_match): **0.5967**

## Per-Position Comparison (First 10)

| Position | Hidden Accept @0.90 | Token Top-1 |
|----------|--------------------:|------------:|
| 0 | 100.00% | 95.01% |
| 1 | 99.98% | 94.34% |
| 2 | 98.89% | 81.61% |
| 3 | 82.43% | 87.10% |
| 4 | 82.24% | 73.89% |
| 5 | 74.37% | 73.64% |
| 6 | 68.75% | 49.47% |
| 7 | 57.64% | 51.42% |
| 8 | 56.48% | 40.91% |
| 9 | 35.66% | 38.22% |

## Two-Phase SD (Token-Level)

- gamma_prefill = 7
- Prefill consecutive accepts: 4.85
- gamma_decode = 5
- Decode consecutive accepts: 3.94
- E2E speedup estimate: 0.85x
