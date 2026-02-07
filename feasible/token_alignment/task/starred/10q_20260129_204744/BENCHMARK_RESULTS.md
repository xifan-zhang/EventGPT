# Full Benchmark Results: EGPT → Adapter → VL

**Date:** 2026-01-29
**Checkpoint:** `10q_20260129_204744`
**Samples:** 10 samples × 10 questions = 100 evaluations

## Summary

| Metric | Value |
|--------|-------|
| **Consecutive Matches** | 0.0 avg |
| **Total Matches** | 2.5 / 37.6 avg |
| **Match Rate** | 6.2% |

### Consecutive Match Distribution

| Tokens Accepted | Count | Percentage |
|-----------------|-------|------------|
| 0 tokens | 100 | 100.0% |

**Key Finding:** All 100 samples have 0 consecutive matches, meaning speculative decoding would accept 0 tokens in every case.

## Example Outputs

### Sample 0, Question 0
**Q:** What are the key elements in this scene?

| Source | Output Text |
|--------|-------------|
| **EGPT** | "In the scene, there is a car parked on the side of a road. The car has..." |
| **Adapter** | "key elements in this scene include a car driving down a street, a, a, ..." |
| **VL Actual** | "The key elements in this scene include a car driving down a winding ro..." |

- Consecutive: 0
- Total: 2/34 (5.9%)

---

### Sample 0, Question 1
**Q:** Can you elaborate on the elements of the scene provided?

| Source | Output Text |
|--------|-------------|
| **EGPT** | "The scene depicts a vehicle with a visible license plate and headlight..." |
| **Adapter** | "image dep as a a street with a a driving down the road. The. The car i..." |
| **VL Actual** | "The image captures a car driving down a winding road, surrounded by tr..." |

- Consecutive: 0
- Total: 5/49 (10.2%)

---

### Sample 0, Question 2
**Q:** What is this scene about?

| Source | Output Text |
|--------|-------------|
| **EGPT** | "The scene shows a car driving on a road with another car ahead of it. ..." |
| **Adapter** | "image dep about a car driving down a road with a, driving. the The and..." |
| **VL Actual** | "This scene is about a car driving down a winding road, surrounded by t..." |

- Consecutive: 0
- Total: 3/35 (8.6%)

---

### Sample 0, Question 3
**Q:** What is the layout of the road in the scene?

| Source | Output Text |
|--------|-------------|
| **EGPT** | "The road in the scene is a two-lane road with a solid white line divid..." |
| **Adapter** | "image in the scene is a two-way street with a cross in on the.. a row ..." |
| **VL Actual** | "The road in the scene is a two-way road with a car driving down it. Th..." |

- Consecutive: 0
- Total: 3/42 (7.1%)

---

### Sample 0, Question 4
**Q:** What type of vehicles are present in the scene?

| Source | Output Text |
|--------|-------------|
| **EGPT** | "There is a car positioned on the right side of the road, facing away f..." |
| **Adapter** | "the scene, there are cars and a truck present. The cars are positioned..." |
| **VL Actual** | "In the scene, there are two vehicles present: a car and a motorcycle. ..." |

- Consecutive: 0
- Total: 0/22 (0.0%)

---

## Analysis

### Why 0 Consecutive Matches?

The first token always fails to match:

| Sample | Adapter Predicts | VL Outputs |
|--------|------------------|------------|
| Q0 | "key" | "The" |
| Q1 | "image" | "The" |
| Q2 | "image" | "This" |
| Q3 | "image" | "The" |
| Q4 | "the" | "In" |

### Training vs Benchmark Gap

| Evaluation | Match Rate |
|------------|------------|
| **Training** (ground truth tokens) | 21.5% |
| **Benchmark** (autoregressive tokens) | 6.2% |

The gap exists because:
1. **Training:** Uses ground truth EGPT/VL token pairs from the same scene
2. **Benchmark:** Uses autoregressively generated tokens (EGPT from event image, VL from video)

### Root Cause

1. **Different Visual Inputs:**
   - EGPT sees: Event image (sparse motion data)
   - VL sees: Video frames (dense RGB)

2. **Different Description Styles:**
   - EGPT: "In the scene, there is a car **parked** on the side..."
   - VL: "The key elements include a car **driving** down a winding road..."

3. **Adapter Limitation:**
   - Trained to map EGPT style → VL style
   - Cannot predict exact word choices when content differs

## Conclusion

Cross-modal speculative decoding between EventGPT and Video-LLaVA is **not viable** with the current TokenAdapter approach because:

1. The models see different visual inputs
2. They generate semantically different descriptions
3. Even with perfect style transfer, word choices diverge
4. Speculative decoding requires exact consecutive token matches

## Recommendations

1. **For practical speedup:** Use EventGPT alone (2x faster than VL)
2. **For quality:** Use Video-LLaVA alone (better visual understanding)
3. **Alternative research:** Explore embedding-level alignment instead of token-level

---

*Results saved to: `full_benchmark_20260129_225244.json`*
