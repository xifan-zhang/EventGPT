# Feature Alignment for Cross-Modal Speculative Decoding

> EventGPT (drafter) → Adapter → Video-LLaVA (verifier)

Train lightweight adapters to align EventGPT decoder hidden states to Video-LLaVA hidden state space, enabling cross-modal speculative decoding with prefill hiding.

---

## Pipeline Overview

```
Stage 0          Stage 1              Stage 2             Stage 3              Stage 4
Extract LM Head  Extract Hidden       Train Adapter       Evaluate Adapter     E2E Wall-Clock
(one-time)       States (both models) (L1-L5, B1, L5F)   (offline metrics)    Benchmark

vl_lm_head.pt    chunked_train/       best_model.pt       acceptance_metrics   speedup per sample
                 chunked_test/        history.json        .json, .png          JSON + graphs
```

---

## Stage 0: Extract VL LM Head (one-time)

Extract Video-LLaVA's LM head weights for offline token-level metrics.

```bash
python feasible/feature_alignment/extract_vl_lm_head.py \
    --output_path ./feasible/feature_alignment/vl_lm_head.pt
```

| | |
|---|---|
| **Script** | `extract_vl_lm_head.py` |
| **Input** | HuggingFace `LanguageBind/Video-LLaVA-7B-hf` (auto-downloaded) |
| **Output** | `vl_lm_head.pt` (~256MB, float32 `[32000, 4096]`) |
| **Time** | ~2 min |

---

## Stage 1: Extract Hidden States

Run both EventGPT and Video-LLaVA on the same scenes/questions, save their decoder hidden states.

### Train split (~24h)

```bash
python feasible/feature_alignment/extract_hidden_states.py \
    --split train --chunked --quant 4bit \
    --dataset_dir /home/ps/Documents/code/EventGPT/data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
    --output_dir ./feasible/feature_alignment/data \
    --max_samples -1 --max_questions 10 --max_new_tokens 50 \
    --chunk_size 1000
```

### Test split (~5h)

```bash
python feasible/feature_alignment/extract_hidden_states.py \
    --split test --chunked --quant 4bit \
    --dataset_dir /home/ps/Documents/code/EventGPT/data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \
    --output_dir ./feasible/feature_alignment/data \
    --max_samples -1 --max_questions 10 --max_new_tokens 50 \
    --chunk_size 1000
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_dir` | hardcoded | Path to DSEC dataset with event images + MP4 |
| `--output_dir` | `./data` | Output directory for chunked data |
| `--split` | `train` | `train` or `test` |
| `--chunked` | off | Chunked saving (required for large datasets) |
| `--chunk_size` | 1000 | Samples per chunk (~1.6GB each) |
| `--quant` | `4bit` | Model quantization (`4bit`, `8bit`, `fp16`) |
| `--max_samples` | -1 | Limit samples (-1 = all) |
| `--max_questions` | 10 | Questions per sample |
| `--max_new_tokens` | 50 | Max generated tokens per question |
| `--resume` | off | Resume from last saved chunk |
| `--save_interval` | 1000 | Emergency checkpoint interval |

### Output Structure

```
feasible/feature_alignment/data/
├── chunked_train_1s_4bit/        # 52,000 samples (5,200 scenes x 10 questions)
│   ├── index.json                # Chunk metadata
│   └── chunks/
│       ├── chunk_000000.pt       # samples 0-999, ~1.6GB
│       ├── chunk_001000.pt       # samples 1000-1999
│       └── ... (52 chunks, ~83GB total)
│
└── chunked_test_1s_4bit/         # 11,000 samples (1,100 scenes x 10 questions)
    ├── index.json
    └── chunks/
        ├── chunk_000000.pt
        └── ... (11 chunks, ~18GB total)
```

### Chunk Format

Each `chunk_XXXXXX.pt` contains:

```python
{
    'egpt_hidden': torch.Tensor,  # [n_samples, max_seq, 4096] float32
    'vl_hidden':   torch.Tensor,  # [n_samples, max_seq, 4096] float32
    'seq_lens':    torch.Tensor,  # [n_samples] int64
}
```

`index.json`:
```json
{
    "total_samples": 52000,
    "chunk_size": 1000,
    "chunks": [{"path": "chunk_000000.pt", "start_idx": 0, "n_samples": 1000}, ...],
    "config": {"split": "train", "duration": "1s", "quant": "4bit",
               "max_questions": 10, "max_new_tokens": 50, "hidden_dim": 4096}
}
```

**Data location:** Symlinked from HDD storage. Actual paths:
- Train: `/mnt/hdd/data/egpt/chunked_train_1s_4bit/`
- Test: `/mnt/hdd/data/egpt/hidden_states/chunked_test_1s_4bit/`

---

## Stage 2: Train Adapter

### Quick Reference

```bash
# L1 Bottleneck (2M params, ~1h)
python feasible/feature_alignment/train_hidden_adapter.py \
    --train_data ./feasible/feature_alignment/data/chunked_train_1s_4bit \
    --val_data   ./feasible/feature_alignment/data/chunked_test_1s_4bit \
    --adapter_level 1 --num_epochs 50 --batch_size 64 \
    --output_dir ./feasible/feature_alignment/tasks/L1

# L4 Attention (101M params, ~10h)
python feasible/feature_alignment/train_hidden_adapter.py \
    --train_data ./feasible/feature_alignment/data/chunked_train_1s_4bit \
    --val_data   ./feasible/feature_alignment/data/chunked_test_1s_4bit \
    --adapter_level 4 --num_epochs 300 --batch_size 64 --early_stopping 50 \
    --output_dir ./feasible/feature_alignment/tasks/L4

# L5 EAGLE cross-modal (103M params, ~12h)
python feasible/feature_alignment/train_hidden_adapter.py \
    --train_data ./feasible/feature_alignment/data/chunked_train_1s_4bit \
    --val_data   ./feasible/feature_alignment/data/chunked_test_1s_4bit \
    --adapter_level 5 --num_epochs 100 --batch_size 64 \
    --output_dir ./feasible/feature_alignment/tasks/L5

# L5F Fused EAGLE (170M params, gated h_egpt + h_vl fusion)
python feasible/feature_alignment/train_hidden_adapter.py \
    --train_data ./feasible/feature_alignment/data/chunked_train_1s_4bit \
    --val_data   ./feasible/feature_alignment/data/chunked_test_1s_4bit \
    --adapter_level 6 --num_epochs 100 --batch_size 64 \
    --output_dir ./feasible/feature_alignment/tasks/L5F

# B1 VLM-only baseline (same arch as L5, no cross-modal gap)
python feasible/feature_alignment/train_hidden_adapter.py \
    --train_data ./feasible/feature_alignment/data/chunked_train_1s_4bit \
    --val_data   ./feasible/feature_alignment/data/chunked_test_1s_4bit \
    --adapter_level 5 --vlm_only --num_epochs 50 --batch_size 64 \
    --output_dir ./feasible/feature_alignment/tasks/B1
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--train_data` | required | Chunked dir or single `.pt` file |
| `--val_data` | none | Chunked dir or `.pt` (splits from train if omitted) |
| `--adapter_level` | 1 | 1=Bottleneck, 2=MultiLayer, 3=Wide, 4=Attention, 5=EAGLE, 6=FusedEAGLE |
| `--num_epochs` | 50 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--learning_rate` | 1e-3 | AdamW learning rate |
| `--early_stopping` | 10 | Patience (epochs without val improvement) |
| `--vlm_only` | off | B1 mode: VL→VL (upper bound, no cross-modal gap) |
| `--bottleneck_dim` | 256/1024 | Bottleneck size (L1/L2=256, L3=1024) |
| `--num_heads` | 8 | Attention heads (L4/L5) |
| `--num_layers` | 1 | Transformer layers (L4/L5) |
| `--ffn_dim` | 2048 | FFN dimension (L4/L5) |
| `--output_dir` | `./tasks/` | Output base directory |

### Adapter Architectures

| Level | Name | Architecture | Params | Key Feature |
|-------|------|-------------|--------|-------------|
| **L1** | Bottleneck | `4096→256→4096 + residual` | 2.1M | Simple, fast |
| **L2** | Multi-Layer Bottleneck | `3x (4096→256→4096)` | 6.3M | Stacked nonlinearity |
| **L3** | Wide Bottleneck | `3x (4096→1024→4096)` | 16M | Larger bottleneck |
| **L4** | Attention | `Transformer + FFN + residual` | 101M | Token dependencies |
| **L5** | EAGLE | `CausalAttn + FFN, dual loss` | 103M | Align + predict next h |
| **L5F** | Fused EAGLE | `Gate(h_egpt, h_vl) + L5` | 170M | Both input streams |
| **B1** | VLM-only | Same as L5, `--vlm_only` | 103M | Upper bound (no cross-modal gap) |

### Training Details

- **Loss:** `MSE(aligned, target) + 0.5 * CosineLoss(aligned, target)`
- **Optimizer:** AdamW, weight_decay=0.01
- **Scheduler:** CosineAnnealingLR (T_max = num_epochs, eta_min=1e-6)
- **Gradient clipping:** max_norm=1.0
- **Memory-efficient:** ChunkedTrainLoader streams 1 chunk (~1.6GB) at a time with background prefetching. Total RAM ~3GB instead of 83GB.

### Output

```
feasible/feature_alignment/tasks/L4/L4_20260206_192256/
├── best_model.pt          # Best val loss checkpoint
├── final_model.pt         # Last epoch checkpoint
├── config.json            # Full training configuration
├── history.json           # Per-epoch train/val metrics
└── training_curves.png    # Loss, cosine sim, acceptance rate plots
```

---

## Stage 3: Evaluate Adapter (Offline Metrics)

### Feature-Level Acceptance

```bash
python feasible/feature_alignment/measure_feature_acceptance.py \
    --checkpoint ./feasible/feature_alignment/tasks/L4/L4_20260206_192256/best_model.pt \
    --test_data  ./feasible/feature_alignment/data/chunked_test_1s_4bit \
    --lm_head    ./feasible/feature_alignment/vl_lm_head.pt \
    --output_dir ./feasible/feature_alignment/tasks/L4/L4_20260206_192256
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Trained adapter `.pt` file |
| `--test_data` | required | Chunked dir or `.pt` test data |
| `--lm_head` | none | VL LM head (enables token-level metrics) |
| `--output_dir` | checkpoint parent | Where to save results |
| `--batch_size` | 64 | Processing batch size |
| `--vlm_only` | off | B1 mode: use `vl_hidden` as input |
| `--gamma_decode` | 5 | Draft tokens per SD iteration |
| `--egpt_prefill_ms` | 130.0 | EventGPT prefill latency |
| `--vl_prefill_ms` | 310.0 | Video-LLaVA prefill latency |

### Output

```
tasks/L4/L4_20260206_192256/
├── acceptance_metrics.json   # All metrics (cos_sim, accept rates, speedup estimates)
├── position_stats.json       # Per-position acceptance breakdown
├── metrics_summary.png       # 2x2 panel (histogram, per-position, consecutive, speedup)
├── stage_timeline.png        # Baseline vs SD timing breakdown
├── token_metrics.png         # Token-level metrics (if --lm_head provided)
└── METRICS_COMPARISON.md     # Hidden vs token comparison table
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `cos_sim_mean` | Mean cosine similarity between aligned and target hidden states |
| `accept_90` | Fraction of positions with cos_sim > 0.90 |
| `consecutive_mean_90` | Avg consecutive tokens before first rejection @0.90 |
| `speedup_e2e` | Estimated end-to-end speedup |
| `token_top1_match` | Token prediction accuracy (requires `--lm_head`) |

### Two-Phase Evaluation

Evaluates the prefill hiding + decode SD pipeline:

```bash
python feasible/feature_alignment/eval_two_phase.py \
    --prefill_checkpoint ./feasible/feature_alignment/tasks/L4/L4_20260206_192256/best_model.pt \
    --decode_checkpoint  ./feasible/feature_alignment/tasks/L5F/L5F_20260206_224537/best_model.pt \
    --test_data          ./feasible/feature_alignment/data/chunked_test_1s_4bit \
    --lm_head            ./feasible/feature_alignment/vl_lm_head.pt
```

- **Phase 1 (Prefill Hiding):** L1-L4 adapter drafts tokens during VL prefill gap
- **Phase 2 (Decode SD):** L5F/B1 adapter with gamma=5 standard speculative decoding

---

## Stage 4: E2E Wall-Clock Benchmark

Real end-to-end benchmark with both models loaded, actual inference timing.

### Run Benchmark

```bash
# Quick test (50 samples)
python feasible/feature_alignment/benchmark_e2e_wallclock.py \
    --max_samples 50 --max_new_tokens 50 \
    --output_dir ./feasible/feature_alignment/tasks/benchmark_50tok

# Full dataset (1100 samples, ~6h per config pair)
python feasible/feature_alignment/benchmark_e2e_wallclock.py \
    --max_samples 1100 --max_new_tokens 30 \
    --configs "vl_baseline,L4+VL" \
    --output_dir ./feasible/feature_alignment/tasks/benchmark_30tok_full

# All configs
python feasible/feature_alignment/benchmark_e2e_wallclock.py \
    --max_samples 50 --max_new_tokens 50 \
    --output_dir ./feasible/feature_alignment/tasks/e2e_all
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_dir` | hardcoded | DSEC dataset with event images + MP4 |
| `--adapter_dir` | `./tasks/` | Directory containing trained adapter checkpoints |
| `--max_samples` | all | Number of samples to benchmark |
| `--max_new_tokens` | 50 | Output tokens to generate |
| `--gamma` | 1 | Draft tokens per SD step |
| `--warmup` | 3 | Warmup iterations (excluded from timing) |
| `--configs` | all | Comma-separated config names to run |
| `--output_dir` | `./tasks/` | Output directory |

### Available Configs

| Config | Description |
|--------|-------------|
| `vl_baseline` | Video-LLaVA autoregressive (no SD) |
| `L1+VL` ... `L4+VL` | Cross-modal SD with prefill hiding |
| `L5_decode` | EAGLE-style decode-only |
| `B1_decode` | VLM-only decode-only |
| `L5F_decode` | Fused EAGLE decode-only |

### Pipeline per Sample

1. **EGPT prefill + decode** — event image → vision encode → LLM prefill → AR decode (~400ms)
2. **VL prefill** — video frames → vision encode → LLM prefill (~326ms, runs in parallel on dual GPU)
3. **~22 free tokens** — EGPT decode tokens hidden behind VL prefill latency gap
4. **Adapter** — maps EGPT hidden states → VL draft token predictions (~1.5ms)
5. **VL verify** — batched forward pass accepts/rejects draft tokens
6. **VL AR decode** — continues from last accepted position

### Output

```
tasks/benchmark_50tok/
├── e2e_wallclock_20260207_HHMMSS.json    # Per-sample results
├── e2e_wallclock_20260207_HHMMSS.md      # Summary table
├── speedup_comparison_20260207_HHMMSS.png
├── timing_breakdown_20260207_HHMMSS.png  # Vision + Prefill + Decode stacked bar
├── accept_rate_comparison_20260207_HHMMSS.png
└── prefill_hiding_20260207_HHMMSS.png    # Timeline + example text
```

---

## Results

### All 7 Adapters (Offline Metrics, 11K test samples)

| Level | Params | Val Loss | Accept@0.90 | cos_sim |
|-------|--------|----------|-------------|---------|
| L1 | 2.1M | 1.2798 | 21.9% | 0.777 |
| L2 | 6.3M | 1.2787 | 23.2% | 0.779 |
| L3 | 16M | 1.2499 | 24.9% | 0.790 |
| L4 | 101M | 1.2458 | 24.8% | 0.791 |
| L5 | 103M | 1.3413 | 11.2% | 0.759 |
| **B1** | 103M | 0.6812 | **61.2%** | 0.912 |
| **L5F** | 170M | 0.7282 | **66.2%** | 0.896 |

**Key insight:** B1 >> L5 shows the cross-modal gap is the dominant bottleneck. L5F > B1 shows fused input helps.

### E2E Wall-Clock Benchmark (50 samples, 10 questions, max_new_tokens=50)

| Config | Prefill | Decode | Total | Accept | Speedup | Free Tokens |
|--------|---------|--------|-------|--------|---------|-------------|
| VL baseline | 326ms | 691ms | 1017ms | --- | 1.00x | --- |
| L1+VL | 326ms | 691ms | 1018ms | 15.9% | 1.00x | 21.9 |
| L2+VL | 326ms | 681ms | 1007ms | 19.3% | 1.02x | 21.9 |
| L3+VL | 326ms | 681ms | 1007ms | 19.6% | 1.02x | 21.9 |
| **L4+VL** | 326ms | 679ms | 1005ms | **20.3%** | **1.02x** | 21.9 |

- Per-sample speedup ranges from 1.00x to **1.24x** (best)
- ~22 tokens generated for free, hidden behind VL prefill latency gap (~230ms)
- Full 1100-sample benchmark in progress (30 tokens)

---

## Directory Structure

```
feasible/feature_alignment/
├── README.md                          # This file
├── hidden_adapter.py                  # All adapter architectures (L1-L5, B1, L5F)
├── train_hidden_adapter.py            # Training script
├── extract_hidden_states.py           # Hidden state extraction from both models
├── extract_vl_lm_head.py             # Extract VL LM head weights
├── measure_feature_acceptance.py      # Offline acceptance metrics
├── eval_two_phase.py                  # Two-phase pipeline evaluation
├── benchmark_e2e_wallclock.py         # True E2E wall-clock benchmark
├── retrain_L4_converge.sh             # L4 retraining script (300 epochs)
│
├── data/                              # Symlinks to HDD storage
│   ├── chunked_train_1s_4bit/         # 52K train samples (52 chunks, ~83GB)
│   └── chunked_test_1s_4bit/          # 11K test samples (11 chunks, ~18GB)
│
├── tasks/                             # Training runs and benchmark results
│   ├── L1/L1_20260206_*/              # L1 adapter results
│   ├── L2/L2_20260206_*/
│   ├── L3/L3_20260206_*/
│   ├── L4/L4_20260206_*/
│   ├── L5/L5_20260206_*/
│   ├── L5F/L5F_20260206_*/
│   ├── B1/B1_20260206_*/
│   ├── benchmark_30tok/               # E2E benchmark results
│   ├── benchmark_50tok/
│   └── starred/                       # Best results for quick reference
│
├── vl_lm_head.pt                      # VL LM head weights (256MB)
├── ADAPTER_ARCHITECTURES.md           # Detailed architecture docs
├── METRICS_COMPARISON.md              # Cross-adapter comparison
└── TASKS.md                           # Task tracking
```

---

## Full Pipeline (Copy-Paste)

```bash
cd /home/ps/Documents/code/EventGPT

# === Stage 0: Extract LM Head (one-time, ~2min) ===
python feasible/feature_alignment/extract_vl_lm_head.py \
    --output_path ./feasible/feature_alignment/vl_lm_head.pt

# === Stage 1a: Extract Train Hidden States (~24h) ===
nohup python feasible/feature_alignment/extract_hidden_states.py \
    --split train --chunked --quant 4bit \
    --output_dir ./feasible/feature_alignment/data \
    > extraction_train.log 2>&1 &

# === Stage 1b: Extract Test Hidden States (~5h) ===
nohup python feasible/feature_alignment/extract_hidden_states.py \
    --split test --chunked --quant 4bit \
    --output_dir ./feasible/feature_alignment/data \
    > extraction_test.log 2>&1 &

# === Stage 2: Train Adapter (e.g., L4, ~10h) ===
python feasible/feature_alignment/train_hidden_adapter.py \
    --train_data ./feasible/feature_alignment/data/chunked_train_1s_4bit \
    --val_data   ./feasible/feature_alignment/data/chunked_test_1s_4bit \
    --adapter_level 4 --num_epochs 300 --early_stopping 50 --batch_size 64 \
    --output_dir ./feasible/feature_alignment/tasks/L4

# === Stage 3: Evaluate (offline metrics, ~20min) ===
python feasible/feature_alignment/measure_feature_acceptance.py \
    --checkpoint ./feasible/feature_alignment/tasks/L4/L4_*/best_model.pt \
    --test_data  ./feasible/feature_alignment/data/chunked_test_1s_4bit \
    --lm_head    ./feasible/feature_alignment/vl_lm_head.pt

# === Stage 4: E2E Wall-Clock Benchmark (50 samples, ~1h) ===
python feasible/feature_alignment/benchmark_e2e_wallclock.py \
    --max_samples 50 --max_new_tokens 50 \
    --configs "vl_baseline,L4+VL" \
    --output_dir ./feasible/feature_alignment/tasks/benchmark_50tok
```
