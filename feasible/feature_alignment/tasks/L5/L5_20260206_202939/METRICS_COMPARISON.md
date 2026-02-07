# Metrics Comparison: Hidden-State vs Token-Level

Generated: 2026-02-07T01:06:42.868493

## Summary

| Metric | Hidden-State (@0.90) | Token-Level (Top-1) |
|--------|---------------------|---------------------|
| Overall Accept Rate | 10.08% | 10.38% |
| Consecutive Accepts | 0.22 | 0.73 |
| E2E Speedup | 0.24x | 0.33x |

## Cosine Similarity as Token Proxy

| cos_sim threshold | P(token match | cos > thresh) | Count |
|-------------------|-------------------------------|-------|
| > 0.80 | 19.97% | 177502 |
| > 0.85 | 25.65% | 108116 |
| > 0.90 | 19.99% | 40925 |
| > 0.92 | 17.45% | 21841 |
| > 0.95 | 22.59% | 1111 |
| > 0.98 | 0.00% | 0 |

Pearson r (cos_sim, token_match): **0.2367**

## Per-Position Comparison (First 10)

| Position | Hidden Accept @0.90 | Token Top-1 |
|----------|--------------------:|------------:|
| 0 | 6.35% | 54.38% |
| 1 | 40.11% | 11.75% |
| 2 | 45.81% | 49.94% |
| 3 | 64.51% | 16.76% |
| 4 | 46.59% | 13.01% |
| 5 | 45.51% | 44.15% |
| 6 | 42.29% | 6.90% |
| 7 | 16.53% | 15.19% |
| 8 | 19.90% | 15.06% |
| 9 | 10.93% | 28.15% |

## Two-Phase SD (Token-Level)

- gamma_prefill = 7
- Prefill consecutive accepts: 0.73
- gamma_decode = 5
- Decode consecutive accepts: 0.73
- E2E speedup estimate: 0.33x
