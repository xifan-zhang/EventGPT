# Metrics Comparison: Hidden-State vs Token-Level

Generated: 2026-02-08T11:46:53.007449

## Summary

| Metric | Hidden-State (@0.90) | Token-Level (Top-1) |
|--------|---------------------|---------------------|
| Overall Accept Rate | 27.77% | 26.51% |
| Consecutive Accepts | 8.30 | 5.84 |
| E2E Speedup | 0.95x | 0.84x |

## Cosine Similarity as Token Proxy

| cos_sim threshold | P(token match | cos > thresh) | Count |
|-------------------|-------------------------------|-------|
| > 0.80 | 53.93% | 190078 |
| > 0.85 | 68.93% | 141313 |
| > 0.90 | 79.85% | 112769 |
| > 0.92 | 82.57% | 103956 |
| > 0.95 | 86.69% | 88781 |
| > 0.98 | 91.39% | 62249 |

Pearson r (cos_sim, token_match): **0.6108**

## Per-Position Comparison (First 10)

| Position | Hidden Accept @0.90 | Token Top-1 |
|----------|--------------------:|------------:|
| 0 | 100.00% | 94.23% |
| 1 | 99.91% | 94.75% |
| 2 | 98.95% | 81.07% |
| 3 | 88.35% | 90.59% |
| 4 | 87.93% | 76.88% |
| 5 | 75.37% | 73.07% |
| 6 | 70.16% | 55.35% |
| 7 | 58.81% | 52.99% |
| 8 | 57.32% | 40.00% |
| 9 | 39.73% | 38.51% |

## Two-Phase SD (Token-Level)

- gamma_prefill = 7
- Prefill consecutive accepts: 4.91
- gamma_decode = 5
- Decode consecutive accepts: 3.89
- E2E speedup estimate: 0.84x
