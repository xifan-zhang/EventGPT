# Stage 4: E2E Wall-Clock Benchmark

> Author: Alice Zhang
> Date: 2026-02-07

Real end-to-end benchmark with both models loaded on GPU, actual inference timing. This is the ground truth for speedup measurement.

```bash
export REPO=/home/ps/Documents/code/EventGPT
cd $REPO
```

---

## Files

| File | Description |
|------|-------------|
| `benchmark_e2e_wallclock.py` | Full pipeline: load both models, run inference, measure wall-clock |

---

## Pipeline Per Sample

```
         EventGPT (GPU 0)                     Video-LLaVA (GPU 1)
  ┌──────────────────────────┐        ┌──────────────────────────┐
  │ Vision encode     ~8ms   │        │ Vision encode     ~0ms   │
  │ LLM prefill      ~83ms  │        │ LLM prefill     ~315ms   │
  │ AR decode       ~298ms   │        │                          │
  │ ─── ~22 free tokens ─── │        │                          │
  └──────────────────────────┘        └──────────────────────────┘
         │                                      │
         │  Adapter (~1.5ms)                    │
         └──── EGPT hidden → VL draft ──────────┤
                                                │
                                    ┌───────────┴──────────┐
                                    │ VL verify batch ~70ms │
                                    │ (accept/reject drafts)│
                                    │ VL AR decode remaining│
                                    └──────────────────────┘
```

---

## Usage

### Quick test (50 samples, ~1h)

```bash
python pipeline/benchmark_e2e/benchmark_e2e_wallclock.py \
    --max_samples 50 --max_new_tokens 50 \
    --configs "vl_baseline,L4+VL"
```

### Full dataset (1100 samples x 10 questions, ~6h per config)

```bash
python pipeline/benchmark_e2e/benchmark_e2e_wallclock.py \
    --max_samples 1100 --max_new_tokens 30 \
    --configs "vl_baseline,L4+VL"
```

Default output: `pipeline/benchmark_e2e/tasks/`

---

## Available Configs

| Config | Description | GPU Setup |
|--------|-------------|-----------|
| `vl_baseline` | Video-LLaVA autoregressive (no SD) | VL on GPU 1 |
| `L1+VL` ... `L4+VL` | Cross-modal SD with prefill hiding | EGPT on GPU 0, VL on GPU 1 |
| `L5_decode` | EAGLE-style decode-only SD | VL on GPU 1 |
| `B1_decode` | VLM-only decode-only SD | VL on GPU 1 |
| `L5F_decode` | Fused EAGLE decode-only SD | VL on GPU 1 |

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_dir` | hardcoded | DSEC dataset with event images + MP4 |
| `--adapter_dir` | `pipeline/adapter_train/tasks/` | Trained adapter checkpoints |
| `--max_samples` | all | Number of scenes to benchmark |
| `--max_new_tokens` | 50 | Output tokens to generate per question |
| `--n_questions` | 10 | Questions per scene |
| `--gamma` | 1 | Draft tokens per SD step |
| `--warmup` | 3 | Warmup scenes (excluded from timing) |
| `--configs` | all | Comma-separated config names to run |
| `--output_dir` | `pipeline/benchmark_e2e/tasks/` | Output directory |

---

## Output

```
pipeline/benchmark_e2e/tasks/
├── e2e_wallclock_YYYYMMDD_HHMMSS.json   # Per-sample results (all timing data)
├── e2e_wallclock_YYYYMMDD_HHMMSS.md     # Summary table (markdown)
├── speedup_comparison_*.png              # Speedup bar chart
├── timing_breakdown_*.png               # Vision + Prefill + Decode stacked bar
├── accept_rate_comparison_*.png         # Acceptance rate per config
└── prefill_hiding_*.png                 # Prefill hiding timeline example
```

---

## Latest Results (10,970 samples, max_new_tokens=30, L4 adapter)

| Config | Prefill (ms) | Decode (ms) | Total (ms) | Accept | Speedup |
|--------|-------------|------------|-----------|--------|---------|
| VL baseline | 317 | 419 | 736 | --- | 1.00x |
| L4+VL SD | 317 | 404 | 721 | 21.2% | 1.03x |

### Speedup Analysis

| Component | Value |
|-----------|-------|
| Mean drafted tokens | 20.9 |
| Mean accepted tokens | 4.4 (21.2%) |
| Tokens saved per sample | 5.4 (accepted + 1 bonus) |
| Value of saved tokens | 5.4 x 14.0 ms = **75 ms** |
| Verify batch cost | **~70 ms** |
| Net saving | **~5 ms** per 736 ms sample |

| Accept% | Speedup |
|---------|---------|
| 21% (current) | 1.01x |
| 50% | 1.14x |
| 70% | 1.25x |
| 100% | 1.47x |

Bottleneck is acceptance rate. Improving adapter quality is the key path to higher speedup.
