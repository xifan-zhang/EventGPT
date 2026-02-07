# Token Alignment Workflow for Speculative Decoding

**Date:** 2026-01-28
**Purpose:** Train a lightweight TokenAdapter to improve token-level acceptance rate between EventGPT (draft model) and Video-LLaVA (target model) for speculative decoding.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SPECULATIVE DECODING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   EventGPT (Draft)          TokenAdapter           Video-LLaVA (Target)     │
│   ┌─────────────┐          ┌───────────┐          ┌─────────────┐          │
│   │ Event Image │ ──────►  │  Predict  │ ──────►  │   Verify    │          │
│   │  (1 frame)  │  draft   │  target   │ aligned  │   tokens    │          │
│   │  ~50ms      │  tokens  │  tokens   │  tokens  │  (8 frames) │          │
│   └─────────────┘          └───────────┘          └─────────────┘          │
│         │                        │                       │                  │
│         ▼                        ▼                       ▼                  │
│   Fast prefill            Low overhead            Accept/Reject             │
│   (event camera)           (~1ms)                  verified                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Goal

| Metric | Value |
|--------|-------|
| Baseline acceptance rate | ~3.4% (direct token matching) |
| Target acceptance rate | 50%+ with TokenAdapter |
| Previous best | 26.66% (200 training samples) |
| Current workflow | Full dataset training (5208+ samples) |

---

## Directory Structure

```
feasible/token_alignment/
├── WORKFLOW.md                    # This document
├── extract_tokens_train.py        # Step 1: Extract tokens from both models
├── train_and_evaluate.py          # Step 2: Train TokenAdapter and evaluate
├── run_full_workflow.sh           # Automated workflow script
├── token_adapter.py               # TokenAdapter model architecture
├── base.py                        # Base classes and utilities
│
├── task/                          # Task folders (auto-created)
│   ├── 1s_20260128_020530/        # 1s dataset, timestamped
│   │   ├── best_model.pt          # Best model checkpoint
│   │   ├── results.json           # Full metrics & training history
│   │   ├── RESULTS.md             # Human-readable summary
│   │   ├── training_curves.png    # Combined loss & accuracy plot
│   │   ├── loss_curve.png         # Training loss curve
│   │   └── accuracy_curve.png     # Acceptance rate curves
│   │
│   ├── 500ms_20260128_043000/     # 500ms dataset, timestamped
│   │   └── (same structure)
│   └── ...
│
├── train_tokens_full.json         # Extracted 1s training tokens
├── train_tokens_500ms.json        # Extracted 500ms training tokens
└── test_tokens_500ms.json         # Extracted 500ms test tokens
```

---

## Quick Start

### Run Complete Workflow (1s + 500ms)

```bash
cd /home/ps/Documents/code/EventGPT

# Run in background
nohup ./feasible/token_alignment/run_full_workflow.sh > /tmp/full_workflow.log 2>&1 &

# Monitor progress
tail -f /tmp/full_workflow.log
```

### Run Single Dataset

```bash
# Step 1: Extract tokens
python3 feasible/token_alignment/extract_tokens_train.py \
    --dataset_dir /path/to/dataset \
    --output_file ./feasible/token_alignment/train_tokens.json \
    --max_samples -1

# Step 2: Train (auto-creates task folder with timestamp)
python3 feasible/token_alignment/train_and_evaluate.py \
    --train_benchmark ./feasible/token_alignment/train_tokens.json \
    --test_benchmark ./feasible/token_alignment/test_tokens.json \
    --task_name 1s \
    --num_epochs 50
```

---

## Step-by-Step Workflow

### Step 1: Token Extraction

Extract output tokens from both EventGPT and Video-LLaVA for the same inputs.

**Script:** `extract_tokens_train.py`

**Input:**
| Model | Input Type | Source |
|-------|------------|--------|
| EventGPT | event_image (PNG) | `dataset/event_image/*.png` |
| Video-LLaVA | mp4 (8 frames) | `dataset/mp4/*.mp4` |

**Process:**
```
1. Load EventGPT (4-bit quantized, ~4GB VRAM)
2. For each sample:
   - Load event_image PNG
   - Process through EventGPT visual encoder
   - Generate 50 tokens with query: "What are the key elements in this scene?"
3. Unload EventGPT, clear GPU memory

4. Load Video-LLaVA (4-bit quantized, ~4GB VRAM)
5. For each sample (where EventGPT succeeded):
   - Load 8 frames from MP4 (uniform sampling)
   - Process through Video-LLaVA
   - Generate 50 tokens with same query
6. Save paired tokens to JSON
```

**Command:**
```bash
python3 feasible/token_alignment/extract_tokens_train.py \
    --dataset_dir /home/ps/Documents/code/EventGPT/data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
    --output_file ./feasible/token_alignment/train_tokens_full.json \
    --max_samples -1 \
    --max_new_tokens 50
```

**Output Format:**
```json
{
  "timestamp": "2026-01-28 01:30:00",
  "dataset_dir": "/path/to/dataset",
  "total_samples": 5208,
  "eventgpt_success": 5200,
  "videollava_success": 5150,
  "results": [
    {
      "sample_idx": 0,
      "sample_id": "sample_001",
      "query": "What are the key elements in this scene?",
      "egpt_tokens": [1, 450, 338, ...],
      "egpt_text": "The scene shows a street with...",
      "vl_tokens": [1, 450, 338, ...],
      "vl_text": "The video shows a urban street..."
    }
  ]
}
```

**Timing Estimates:**
| Dataset | Samples | EventGPT | Video-LLaVA | Total |
|---------|---------|----------|-------------|-------|
| 1s train | 5,208 | ~50 min | ~50 min | ~100 min |
| 500ms train | 10,475 | ~100 min | ~100 min | ~200 min |
| 500ms test | 2,220 | ~20 min | ~20 min | ~40 min |

---

### Step 2: Training

Train the TokenAdapter to predict Video-LLaVA tokens from EventGPT tokens.

**Script:** `train_and_evaluate.py`

**Model Architecture:**
```
┌────────────────────────────────────────────────────────────┐
│                    TokenAdapter (~50M params)              │
├────────────────────────────────────────────────────────────┤
│  Input: EventGPT token sequence [t1, t2, ..., tn]         │
│                        │                                   │
│                        ▼                                   │
│  ┌──────────────────────────────────────────┐             │
│  │ Token Embedding (32000 → 512)            │             │
│  │ + Position Embedding (128 → 512)         │             │
│  └──────────────────────────────────────────┘             │
│                        │                                   │
│                        ▼                                   │
│  ┌──────────────────────────────────────────┐             │
│  │ Transformer Encoder (4 layers)           │             │
│  │ - 8 attention heads                      │             │
│  │ - 2048 FFN dim                           │             │
│  │ - GELU activation                        │             │
│  │ - Causal masking                         │             │
│  └──────────────────────────────────────────┘             │
│                        │                                   │
│                        ▼                                   │
│  ┌──────────────────────────────────────────┐             │
│  │ LayerNorm + Linear (512 → 32000)         │             │
│  └──────────────────────────────────────────┘             │
│                        │                                   │
│                        ▼                                   │
│  Output: Predicted Video-LLaVA logits                     │
└────────────────────────────────────────────────────────────┘
```

**Training Configuration:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 50 | Maximum training epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `early_stopping` | 10 | Patience for early stopping |
| `max_seq_len` | 128 | Maximum sequence length |
| `embed_dim` | 512 | Embedding dimension |
| `num_layers` | 4 | Transformer layers |
| `num_heads` | 8 | Attention heads |

**Command:**
```bash
# Auto-creates: ./feasible/token_alignment/task/1s_YYYYMMDD_HHMMSS/
python3 feasible/token_alignment/train_and_evaluate.py \
    --train_benchmark ./feasible/token_alignment/train_tokens_full.json \
    --test_benchmark ./feasible/benchmark_parallel_prefill/results/parallel_prefill_5stages_20260127_160820.json \
    --task_name 1s \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --early_stopping 10
```

**Task Folder Naming:**
- Auto-detected from path: `1s`, `500ms`, `2s`, etc.
- Manual override: `--task_name myexperiment`
- Format: `{task_name}_{YYYYMMDD}_{HHMMSS}`

---

### Step 3: Outputs

Each training run creates a timestamped task folder with:

| File | Description |
|------|-------------|
| `best_model.pt` | Best model checkpoint (by validation accuracy) |
| `results.json` | Full metrics, config, and training history |
| `RESULTS.md` | Human-readable summary with tables |
| `training_curves.png` | Combined loss and accuracy plot |
| `loss_curve.png` | Training loss over epochs |
| `accuracy_curve.png` | Acceptance rates (train, val, top-5) with baseline |

**Training Curves Include:**
- Training loss curve
- Training accuracy
- Validation accuracy (Top-1)
- Validation accuracy (Top-5)
- Baseline reference lines

---

## Datasets

### Available Datasets

| Duration | Train Path | Train Samples | Test Path | Test Samples |
|----------|------------|---------------|-----------|--------------|
| 500ms | `my_egpt_dsec_train_500ms` | 10,475 | `my_egpt_dsec_seq_500ms` | 2,220 |
| 1s | `my_egpt_dsec_train_1s` | 5,208 | `my_egpt_dsec_seq_1s` | 1,100 |
| 2s | `my_egpt_dsec_train_2s` | 2,604 | `my_egpt_dsec_seq_2s` | 550 |
| 4s | `my_egpt_dsec_train_4s` | 1,302 | `my_egpt_dsec_seq_4s` | 275 |

**Base paths:**
- Train: `/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_train/`
- Test: `/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_test/`

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Acceptance Rate (Top-1)** | % of tokens correctly predicted |
| **Top-5 Rate** | % of target tokens in top-5 predictions |
| **Baseline** | Direct token matching without adapter |
| **Improvement** | Model accuracy - Baseline |

**Expected Results:**
| Dataset | Baseline | With Adapter | Top-5 | Improvement |
|---------|----------|--------------|-------|-------------|
| Train | ~3% | ~40-50% | ~60-70% | +37-47% |
| Test | ~3% | ~30-40% | ~50-60% | +27-37% |

---

## Theoretical Background

### Speculative Decoding Formula

With acceptance rate α and draft length γ:
```
Expected accepted tokens = (1 - α^(γ+1)) / (1 - α)
```

| α (acceptance) | γ=5 tokens | Speedup |
|----------------|------------|---------|
| 10% | 1.1 tokens | 1.1x |
| 30% | 1.8 tokens | 1.8x |
| 50% | 2.5 tokens | 2.5x |
| 70% | 3.5 tokens | 3.5x |

### Why Token Alignment Helps

```
Without Adapter:
  EventGPT:    "A parked car on the street"
  Video-LLaVA: "A vehicle moving on the road"
  Match: ~3% (only "the", "a")

With Adapter:
  Learned: "parked" → "moving"/"stationary"
  Learned: "car" → "vehicle"/"car"
  Learned: "street" → "road"/"street"
  Match: ~30-40%
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 16  # or 8
```

### Slow Extraction
- Normal speed: ~1.7 samples/sec per model
- Uses 4-bit quantization by default

### Low Acceptance Rate
- Ensure same query for both models
- Check token alignment in output JSON
- Try increasing model capacity: `--num_layers 6`

### Format Errors
- Script auto-detects: benchmark, extraction, inference formats
- Check JSON structure matches expected format

---

## Full Workflow Script

The `run_full_workflow.sh` script automates:

```
PHASE 1: 1s Dataset
├── Wait for extraction (if running)
├── Train TokenAdapter
└── Evaluate on test set

PHASE 2: 500ms Dataset
├── Extract train tokens (10,475 samples)
├── Extract test tokens (2,220 samples)
├── Train TokenAdapter
└── Evaluate on test set

SUMMARY: Print results from both datasets
```

**Run:**
```bash
nohup ./feasible/token_alignment/run_full_workflow.sh > /tmp/full_workflow.log 2>&1 &
tail -f /tmp/full_workflow.log
```

---

## Current Status

- [x] Token extraction script (event_image + mp4)
- [x] Training script with auto task folders
- [x] Training curves visualization
- [x] Full workflow automation script
- [~] 1s dataset extraction (in progress)
- [ ] 1s dataset training
- [ ] 500ms dataset extraction
- [ ] 500ms dataset training

**Monitor:**
```bash
# Full workflow
tail -f /tmp/full_workflow.log

# Extraction only
tail -f /tmp/extract_train_tokens.log
```

**List task folders:**
```bash
ls -la ./feasible/token_alignment/task/
```
