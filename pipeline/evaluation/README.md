# Stage 3: Evaluation

> Author: Alice Zhang
> Date: 2026-02-07

Offline evaluation of adapter quality using feature-level and token-level metrics. Works directly on extracted hidden states — no GPU inference of full models required.

```bash
export REPO=/home/ps/Documents/code/EventGPT
cd $REPO
DATA=pipeline/feature_extraction/data
ADAPTERS=pipeline/adapter_train/tasks
```

---

## Files

| File | Description |
|------|-------------|
| `measure_feature_acceptance.py` | Feature acceptance metrics (cos_sim, accept rates, speedup estimates) |
| `eval_two_phase.py` | Two-phase pipeline evaluation (prefill hiding L1-L4 + decode SD L5F/B1) |
| `run_all_eval.sh` | Evaluate all adapters (L1-L5, B1, L5F) sequentially |
| `run_two_phase_eval.sh` | Evaluate all two-phase combinations (L1+L5F, ..., B1-only, L5F-only) |

---

## Usage

### Single adapter evaluation (~20min)

```bash
python pipeline/evaluation/measure_feature_acceptance.py \
    --checkpoint $ADAPTERS/L4/L4_*/best_model.pt \
    --test_data  $DATA/chunked_test_1s_4bit \
    --lm_head    $DATA/vl_lm_head.pt
```

Default output: `pipeline/evaluation/tasks/`

### Two-phase evaluation (prefill + decode)

```bash
python pipeline/evaluation/eval_two_phase.py \
    --prefill_checkpoint $ADAPTERS/L4/L4_*/best_model.pt \
    --decode_checkpoint  $ADAPTERS/L5F/L5F_*/best_model.pt \
    --test_data $DATA/chunked_test_1s_4bit \
    --lm_head   $DATA/vl_lm_head.pt
```

Default output: `pipeline/evaluation/tasks/`

### Evaluate all adapters

```bash
bash pipeline/evaluation/run_all_eval.sh
```

### Evaluate all two-phase combos

```bash
bash pipeline/evaluation/run_two_phase_eval.sh
```

---

## CLI Reference

### `measure_feature_acceptance.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Trained adapter `.pt` file |
| `--test_data` | required | Chunked dir or `.pt` test data |
| `--lm_head` | none | VL LM head (enables token-level metrics) |
| `--output_dir` | `pipeline/evaluation/tasks/` | Where to save results |
| `--batch_size` | 64 | Processing batch size |
| `--vlm_only` | off | B1 mode: use `vl_hidden` as input |
| `--gamma_decode` | 5 | Draft tokens per SD iteration |

### `eval_two_phase.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--prefill_checkpoint` | none | L1-L4 checkpoint for prefill phase (skip with `--no_prefill`) |
| `--decode_checkpoint` | required | L5F/B1 checkpoint for decode phase |
| `--test_data` | required | Chunked dir or `.pt` test data |
| `--lm_head` | none | VL LM head |
| `--batch_size` | 64 | Processing batch size |
| `--gamma_decode` | 5 | Draft tokens per SD iteration |
| `--no_prefill` | off | Decode-only baseline (no prefill hiding) |
| `--decode_vlm_only` | off | Use VL hidden states as decode input (for B1) |
| `--label` | auto | Label for output files |

---

## Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| `cos_sim_mean` | Mean cosine similarity (aligned vs target) | > 0.85 |
| `accept_90` | Fraction of positions with cos_sim > 0.90 | > 40% |
| `consecutive_mean_90` | Avg consecutive accepts before first rejection @0.90 | > 5 |
| `speedup_e2e` | Estimated end-to-end speedup | > 1.2x |
| `token_top1_match` | Token prediction accuracy (needs `--lm_head`) | > 30% |

---

## Output

```
pipeline/evaluation/tasks/
├── L4/
│   ├── acceptance_metrics.json   # All metrics
│   ├── position_stats.json       # Per-position acceptance breakdown
│   ├── metrics_summary.png       # 2x2 panel: histogram, per-position, consecutive, speedup
│   ├── stage_timeline.png        # Baseline vs SD timing breakdown
│   ├── token_metrics.png         # Token-level metrics (if --lm_head)
│   └── METRICS_COMPARISON.md     # Hidden vs token comparison table
├── two_phase_results_L4_L5F.json
├── two_phase_results_B1_only.json
└── two_phase_results_L5F_only.json
```

---

## Results (11K test samples)

| Level | cos_sim | Accept@0.90 | Consecutive@0.90 |
|-------|---------|-------------|-------------------|
| L1 | 0.777 | 21.9% | ~3 |
| L2 | 0.779 | 23.2% | ~3 |
| L3 | 0.790 | 24.9% | ~4 |
| L4 | 0.791 | 24.8% | ~4 |
| L5 | 0.759 | 11.2% | ~2 |
| B1 | 0.912 | 61.2% | ~8 |
| L5F | 0.896 | 66.2% | ~9 |
