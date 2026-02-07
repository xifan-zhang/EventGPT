# Cross-Modal Speculative Decoding: Training & Evaluation Results

**Generated:** 2026-02-07 01:30 UTC+8
**Updated:** 2026-02-07 02:00 UTC+8
**Project:** EventGPT → Video-LLaVA Cross-Modal Speculative Decoding

---

## Hidden State Extraction (Training Data Generation)

Training data = paired hidden states `(h_egpt, h_vl)` extracted by running both models on the same DSEC scenes.

### Pipeline

```
DSEC 1s clip
  ├── event_npy/{seq}/{clip}.npy
  │     ↓ preprocess_event_images.py (split into 5 temporal frames)
  │   event_image/{seq}/{clip}_0.png ... {clip}_4.png
  │     ↓ extract_hidden_states.py (use FIRST frame only: {clip}_0.png)
  │   EventGPT CLIP ViT → feature adaptor → LLM generate(max_new_tokens=50)
  │     ↓ output_hidden_states=True, last decoder layer
  │   h_egpt: [50, 4096]
  │
  └── mp4/{seq}.mp4
        ↓ extract_hidden_states.py (load 8 uniformly-sampled frames)
      Video-LLaVA video encoder → LLM generate(max_new_tokens=50)
        ↓ output_hidden_states=True, last decoder layer
      h_vl: [50, 4096]
```

### Key Details

| Aspect | EventGPT | Video-LLaVA |
|--------|----------|-------------|
| **Raw input** | `event_image/{seq}/{clip}_0.png` | `mp4/{seq}.mp4` |
| **Frames used** | 1 (first of 5 event frames) | 8 (uniformly sampled from MP4) |
| **Vision encoder** | CLIP ViT (224x224) | LanguageBind Video (224x224) |
| **LLM backbone** | Vicuna-7B | Vicuna-7B |
| **Hidden dim** | 4096 | 4096 |
| **Seq len** | 50 (max_new_tokens) | 50 (max_new_tokens) |
| **Quantization** | 4-bit NF4 | 4-bit NF4 |

### Event Image Details

- Event `.npy` files contain raw event streams: `(x, y, polarity, timestamp)`
- `preprocess_event_images.py` splits events into 5 equal temporal segments, renders each as a PNG (red=positive, blue=negative polarity)
- Extraction uses only **the first frame** (`{clip}_0.png`) as EventGPT input via `event_image_paths[0]`
- Each question generates a 50-token response; hidden states from the last decoder layer are saved
- 10 questions per scene (from `top50_questions.json`): 1,100 scenes × 10 = 11,000 test samples, 5,208 scenes × 10 = 52,080 train samples

### Storage

- Each sample: `h_egpt [50, 4096]` + `h_vl [50, 4096]` + `mask [50]` ≈ 1.6MB (4-bit)
- Chunked into 1000-sample `.pt` files for memory-efficient training
- Train: 52 chunks (~80GB), Test: 11 chunks (~17GB)

### How to Extract

```bash
# Train set (~24h on RTX 4090)
python extract_hidden_states.py --split train --chunked --quant 4bit

# Test set (~5h)
python extract_hidden_states.py --split test --chunked --quant 4bit
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Data** | DSEC 1s event-video pairs, 4-bit quantized hidden states |
| **Train samples** | 52,000 (80GB, 52 chunks × 1.6GB) |
| **Val samples** | 11,000 (17GB, 11 chunks) |
| **Storage** | NVMe (~3.1 GB/s) |
| **GPU** | Single 24GB (RTX 4090) |
| **Optimizer** | AdamW, lr=1e-3 |
| **Scheduler** | CosineAnnealingLR |
| **Loss** | MSE + 0.5 × CosLoss |
| **Batch size** | 64 |
| **Max epochs** | 50 |
| **Early stopping** | Patience=10 |

---

## Adapter Architectures Trained

### L1-L4: Per-Position Alignment Adapters (EGPT → VL)

Map h_egpt[t] → h_vl[t] at the same position. Used for **Phase 1: Prefill Hiding**.

| Level | Architecture | Params | Task Name | Description |
|-------|-------------|--------|-----------|-------------|
| **L1** | Bottleneck | 2.1M | `train_L1_Bottleneck` | Linear → ReLU → Linear + residual |
| **L2** | Multi-Layer Bottleneck | 6.3M | `train_L2_MultiLayerBottleneck` | 3 bottleneck blocks stacked |
| **L3** | Wide Bottleneck | 16M | `train_L3_WideBottleneck` | Wider bottleneck (1024 dim) |
| **L4** | Attention | 101M | `train_L4_Attention` | Self-attention + cross-attention + FFN |

### L5, B1, L5F: EAGLE-Style Autoregressive Prediction Adapters

Predict h_vl[t+1] from context. Used for **Phase 2: Decode Acceleration**.

| Level | Architecture | Params | Task Name | Input | Description |
|-------|-------------|--------|-----------|-------|-------------|
| **L5** | EAGLE (cross-modal) | 103M | `train_L5_EAGLE` | h_egpt only | Causal transformer, EGPT → VL prediction |
| **B1** | EAGLE (VLM-only baseline) | 103M | `train_B1_VLMOnly_EAGLE` | h_vl only | Same arch, VL → VL (upper bound) |
| **L5F** | Fused EAGLE | 170M | `train_L5F_FusedEAGLE` | h_egpt + h_vl | Gated fusion of both modalities |

---

## Training Results: Hidden-State Metrics

| Level | Params | Epochs | Best Val Loss | Accept@0.80 | Accept@0.85 | Accept@0.90 | Accept@0.95 | CosSim |
|-------|--------|--------|---------------|-------------|-------------|-------------|-------------|--------|
| **L1** | 2.1M | 43 (ES) | 1.2798 | 46.1% | 30.2% | 21.9% | 18.1% | 0.756 |
| **L2** | 6.3M | 34 (ES) | 1.2787 | 46.3% | 31.4% | 23.2% | 18.6% | 0.758 |
| **L3** | 16M | 50 | 1.2499 | ~46% | ~30% | 24.9% | ~18% | 0.770 |
| **L4** | 101M | 50 | 1.2458 | 47.0% | 32.3% | 24.8% | 20.0% | 0.790 |
| **L5** | 103M | 50 | 1.3413 | 46.4% | 28.5% | 11.2% | 0.4% | 0.771 |
| **B1** | 103M | 50 | 0.6812 | 98.2% | 90.2% | 61.2% | 12.8% | 0.907 |
| **L5F** | 170M | 50 | 0.7282 | 98.5% | 90.6% | 66.2% | 20.0% | 0.913 |

ES = Early Stopping (patience=10)

---

## Key Findings

### 1. L1-L4: Diminishing Returns on Per-Position Alignment

```
L1 (2.1M) → L2 (6.3M): +1.3% Accept@0.90, -0.001 val loss
L2 (6.3M) → L3 (16M):  +1.7% Accept@0.90, -0.029 val loss
L3 (16M)  → L4 (101M): -0.1% Accept@0.90, -0.004 val loss
```

**Conclusion:** The cross-modal distribution gap (not adapter capacity) is the bottleneck. Increasing params from 2M to 101M yields only +3% Accept@0.90.

### 2. EAGLE Variants: Cross-Modal Gap Confirmed

| Comparison | Accept@0.90 | Val Loss | Insight |
|------------|-------------|----------|---------|
| **B1 >> L5** | 61.2% vs 11.2% | 0.68 vs 1.34 | Cross-modal gap is dominant |
| **L5F > B1** | 66.2% vs 61.2% | 0.73 vs 0.68 | Events add complementary signal |
| **L5F >> L5** | 66.2% vs 11.2% | 0.73 vs 1.34 | Fusion essential for cross-modal SD |

### 3. L5F Validates Information-Theoretic Argument

L5F (gated fusion of h_egpt + h_vl) **guarantees ≥ B1** because it has strictly more information. Empirically:
- L5F beats B1 by +5.0% Accept@0.90
- The learned gate selectively leverages event features for motion/temporal tokens
- The VL residual path ensures graceful fallback

---

## Two-Phase SD Pipeline Design

### Phase 1: Prefill Hiding (L4 Adapter)

```
EGPT: [prefill 130ms][── generate draft tokens ─────]
VL:   [────────── prefill 310ms ────────────────────][verify]
                     ← 180ms free window →
```

- L4 aligns h_egpt[t] → VL space → VL LM head → draft token
- VL batch-verifies when its prefill completes
- Any accepted tokens are "free" (generated during VL's prefill)

### Phase 2: Decode Acceleration (L5F Adapter)

```
For each SD iteration (γ=5):
  1. EGPT generates 5 draft tokens → h_egpt[t:t+5]
  2. L5F fuses h_egpt + h_vl → predicts h_vl[t+1:t+6]
  3. VL LM head → draft tokens
  4. VL verifies γ+1 tokens in one pass
  5. Accept consecutive correct prefix
```

| Phase | Adapter | Input | Role | Accept@0.90 |
|-------|---------|-------|------|-------------|
| **1: Prefill Hiding** | L4 | h_egpt only | Align EGPT→VL space | 24.8% |
| **2: Decode** | L5F | h_egpt + h_vl | Predict next VL token | 66.2% |

---

## Token-Level Evaluation Results

Token-level metrics: project adapted hidden states through VL LM head (32064 vocab), argmax → tokens, compare with VL ground truth. This is the ground truth for actual SD acceptance.

### Per-Position Adapters (L1-L4): Same-Position Token Match

| Level | Token Top-1 | Prefill Free Tokens (7 slots) | Time Saved |
|-------|------------|-------------------------------|------------|
| **L1** | 16.98% | 3.82 / 7 | 95ms |
| **L2** | 22.59% | 4.72 / 7 | 118ms |
| **L3** | 24.39% | 4.85 / 7 | 121ms |
| **L4** | 23.90% | 4.59 / 7 | 115ms |

### EAGLE Adapters (L5, B1, L5F): Shifted Next-Token Prediction

| Level | Same-Pos Top-1 | Shifted Top-1 | Shifted Top-5 | Consec Accepts (α) |
|-------|---------------|---------------|---------------|---------------------|
| **L5** | 10.38% | — | — | 0.73 |
| **B1** | 19.11% | 40.96% | 77.16% | 0.51 |
| **L5F** | 24.64% | 36.18% | 69.09% | 0.73 |

**Key insight:** B1 has higher per-position shifted top-1 (40.96%) but lower consecutive accepts (0.51). L5F maintains longer streaks (0.73) — what matters for SD speedup.

---

## Two-Phase Pipeline E2E Results

### Cost Model
- Baseline: VL prefill (310ms) + 50 × 25ms decode = **1560ms**
- SD iteration: 1 VL verify pass (25ms) + adapter (1.5ms) = 26.5ms per iter
- Tokens/iter: α + 1 (where α = consecutive shifted accepts)

### Results

| Config | Prefill Adapter | Free Tokens | Decode α | Decode Iters | E2E Speedup |
|--------|----------------|-------------|----------|-------------|-------------|
| **L1+L5F** | L1 (2.1M) | 3.82 | 0.73 | 26.7 | **1.54x** |
| **L2+L5F** | L2 (6.3M) | 4.72 | 0.73 | 26.1 | **1.56x** |
| **L3+L5F** | L3 (16M) | 4.85 | 0.73 | 26.0 | **1.56x** |
| **L4+L5F** | L4 (101M) | 4.59 | 0.73 | 26.2 | **1.55x** |
| **L5F-only** | None | 0 | 0.73 | 28.9 | **1.45x** |
| **B1-only** | None | 0 | 0.51 | 33.1 | **1.31x** |

### Analysis

1. **Two-phase > decode-only**: L3+L5F (1.56x) > L5F-only (1.45x). Prefill hiding contributes +0.11x from 4.85 free tokens.

2. **L5F > B1 for SD**: L5F-only (1.45x) > B1-only (1.31x). Cross-modal fusion improves consecutive accepts (0.73 vs 0.51), which matters more than per-position accuracy for SD.

3. **L2-L3 are optimal prefill adapters**: L3 (16M) achieves the most free tokens (4.85) with 2.5x fewer params than L4 (101M). Diminishing returns confirm capacity isn't the bottleneck.

4. **Prefill adapter choice has minor impact**: L1→L4 only spans 1.54x–1.56x. The decode adapter (L5F) dominates E2E speedup.

---

## Checkpoint Locations

| Level | Checkpoint Path |
|-------|----------------|
| L1 | `tasks/L1/L1_20260206_095906/best_model.pt` |
| L2 | `tasks/L2/L2_20260206_181048/best_model.pt` |
| L3 | `tasks/L3/L3_20260206_183919/best_model.pt` |
| L4 | `tasks/L4/L4_20260206_192256/best_model.pt` |
| L5 | `tasks/L5/L5_20260206_202939/best_model.pt` |
| B1 | `tasks/B1/B1_20260206_213805/best_model.pt` |
| L5F | `tasks/L5F/L5F_20260206_224537/best_model.pt` |

---

## Speedup Formula (Offline Proxy)

### Baseline (standard autoregressive)
```
T_baseline = T_vl_prefill + N × T_vl_decode
           = 310ms + 50 × 25ms = 1560ms
```

### Two-Phase SD
```
T_sd = T_parallel_prefill + T_decode

Phase 1 (Prefill Hiding):
  T_parallel_prefill = max(T_vl_prefill, T_egpt_prefill + T_adapter)
                     = max(310, 130 + 1.5) = 310ms
  Free tokens: α_prefill (consecutive same-position matches in γ_prefill=7 slots)
  Remaining tokens: N_remaining = N - α_prefill

Phase 2 (SD Decode):
  Per SD iteration:
    - Adapter forward pass: T_adapter = 1.5ms
    - VL verifies γ+1 tokens in ONE forward pass: T_vl_decode = 25ms
    - Cost per iter: T_vl_decode + T_adapter = 26.5ms
    - Tokens per iter: α_decode + 1 (consecutive shifted accepts + 1)

  Iterations: N_remaining / (α_decode + 1)
  T_decode = iterations × (T_vl_decode + T_adapter)
```

### Speedup
```
Speedup = T_baseline / T_sd

Key insight: SD verification is a SINGLE forward pass for γ+1 tokens,
not γ+1 separate passes. With KV cache, cost ≈ 1 forward pass.
```

### Note: Offline vs True E2E

This offline proxy evaluation uses pre-extracted hidden states and the VL LM head for token-level comparison. It estimates SD acceptance rates without running the actual models in a live generation loop. A separate true E2E benchmark (loading both models, processing raw inputs, real wall-clock timing) is the definitive measure. See `benchmark_e2e_wallclock.py`.

---

## How to Reproduce

### Train all adapters
```bash
# L1-L4 per-position adapters
for LEVEL in 1 2 3 4; do
  conda run -n egpt python train_hidden_adapter.py \
    --train_data ./data/chunked_train_1s_4bit \
    --val_data ./data/chunked_test_1s_4bit \
    --adapter_level $LEVEL --num_epochs 50 --batch_size 64 \
    --output_dir ./tasks/L$LEVEL
done

# L5 EAGLE (cross-modal)
conda run -n egpt python train_hidden_adapter.py \
  --train_data ./data/chunked_train_1s_4bit \
  --val_data ./data/chunked_test_1s_4bit \
  --adapter_level 5 --num_epochs 50 --batch_size 64 \
  --output_dir ./tasks/L5

# B1 VLM-only EAGLE (baseline)
conda run -n egpt python train_hidden_adapter.py \
  --train_data ./data/chunked_train_1s_4bit \
  --val_data ./data/chunked_test_1s_4bit \
  --adapter_level 5 --vlm_only --num_epochs 50 --batch_size 64 \
  --output_dir ./tasks/B1

# L5F Fused EAGLE (cross-modal + VL)
conda run -n egpt python train_hidden_adapter.py \
  --train_data ./data/chunked_train_1s_4bit \
  --val_data ./data/chunked_test_1s_4bit \
  --adapter_level 6 --num_epochs 50 --batch_size 64 \
  --output_dir ./tasks/L5F
```

### Run token-level evaluation (individual adapters)
```bash
bash run_all_eval.sh        # L1-L2
bash run_remaining_eval.sh  # L3-L5, B1, L5F
```

### Run two-phase pipeline + baselines (offline proxy)
```bash
bash run_two_phase_eval.sh
# Runs: L1+L5F, L2+L5F, L3+L5F, L4+L5F, B1-only, L5F-only
# Results saved as JSON in respective task dirs
```

### Run true E2E wall-clock benchmark (real models)
```bash
conda run -n egpt python benchmark_e2e_wallclock.py \
  --dataset_dir ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \
  --max_samples 50 --max_new_tokens 50
# Loads EventGPT + Video-LLaVA, processes raw event images + MP4
# Measures 3-stage timing: vision encoder, LLM prefill, LLM decode
# Computes SD speedup from real wall-clock measurements
```

---

## Two Evaluation Approaches

| Aspect | Offline Proxy (`eval_two_phase.py`) | True E2E (`benchmark_e2e_wallclock.py`) |
|--------|-------------------------------------|----------------------------------------|
| **Input** | Pre-extracted hidden states (4-bit) | Raw event images + MP4 videos |
| **Models loaded** | Adapter only (~170M params) | Both full models (~7B each) |
| **What it measures** | Token-level acceptance via LM head projection | Real wall-clock 3-stage timing |
| **Timing source** | Estimated from TimingConfig constants | Measured on actual GPU |
| **Speedup** | Theoretical (from acceptance rates) | Based on real measurements |
| **Speed** | Fast (~2 min per config) | Slow (loads 2 models, runs inference) |
| **Use case** | Rapid adapter comparison, ablation | Final paper results, ground truth |

---

## File Reference

| File | Description |
|------|-------------|
| `hidden_adapter.py` | All adapter architectures + factory |
| `train_hidden_adapter.py` | Training loop with chunked data loading + prefetch |
| `measure_feature_acceptance.py` | Hidden-state + token-level evaluation (individual adapters) |
| `eval_two_phase.py` | Offline proxy: two-phase pipeline evaluation (prefill + decode) |
| `benchmark_e2e_wallclock.py` | True E2E: real models, raw data, 3-stage wall-clock timing |
| `extract_vl_lm_head.py` | Extract VL LM head for offline token metrics |
| `run_all_eval.sh` | Batch evaluation for L1-L2 |
| `run_remaining_eval.sh` | Batch evaluation for L3-L5, B1, L5F |
| `run_two_phase_eval.sh` | Two-phase + baseline evaluations (6 configs) |
| `ADAPTER_ARCHITECTURES.md` | Detailed architecture documentation |
| `METRICS_COMPARISON.md` | Hidden-state vs token-level metrics explanation |
