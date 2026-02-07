# Token Alignment for Speculative Decoding

Train a lightweight TokenAdapter to improve token-level acceptance rate between EventGPT (draft) and Video-LLaVA (target) for speculative decoding.

## Results Summary

| Metric | Baseline | Achieved | Status |
|--------|----------|----------|--------|
| Token Acceptance (Training) | 8% | 21.5% | ✓ Improved |
| Token Acceptance (Benchmark) | 8% | 6.2% total, **0% consecutive** | ✗ Failed |
| Speedup | 1.0x | **1.0x (no speedup)** | ✗ Not viable |

**Conclusion:** Cross-modal token-level speculative decoding between EventGPT and Video-LLaVA is **not viable** due to fundamental content differences between the two models.

## Quick Start

### Option 1: Single Question (Fast, ~3 hours)

```bash
cd /home/ps/Documents/code/EventGPT

# Run full workflow: extract → train → evaluate
nohup ./feasible/token_alignment/run_1q_workflow.sh > /tmp/workflow_1q.log 2>&1 &

# Monitor progress
tail -f /tmp/workflow_1q.log
```

### Option 2: 10 Questions (Balanced, ~20 hours)

```bash
cd /home/ps/Documents/code/EventGPT

# Run 10-question workflow: extract → train 50 epochs → evaluate
nohup ./feasible/token_alignment/run_10q_workflow.sh > /tmp/workflow_10q.log 2>&1 &

# Monitor progress
tail -f /tmp/workflow_10q.log
```

### Option 3: 50 Questions (Comprehensive, ~72 hours)

```bash
cd /home/ps/Documents/code/EventGPT

# Run full workflow with all 50 questions
nohup ./feasible/token_alignment/run_full_workflow.sh > /tmp/full_workflow.log 2>&1 &

# Monitor progress
tail -f /tmp/full_workflow.log
```

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOKEN ALIGNMENT PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. EXTRACT TOKENS                                               │
│     ┌──────────────┐         ┌──────────────┐                   │
│     │   EventGPT   │         │ Video-LLaVA  │                   │
│     │  (1 frame)   │         │  (8 frames)  │                   │
│     └──────┬───────┘         └──────┬───────┘                   │
│            │                        │                            │
│            ▼                        ▼                            │
│     egpt_tokens (50)         vl_tokens (50)                     │
│                                                                  │
│  2. TRAIN TOKENADAPTER                                           │
│     ┌──────────────────────────────────────┐                    │
│     │  TokenAdapter (4-layer transformer)  │                    │
│     │  Input: EventGPT tokens              │                    │
│     │  Output: Predicted Video-LLaVA tokens│                    │
│     └──────────────────────────────────────┘                    │
│                                                                  │
│  3. EVALUATE                                                     │
│     - Acceptance rate (Top-1, Top-5)                            │
│     - Compare with baseline                                      │
│     - Save training curves                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Usage

### Step 1: Extract Tokens

Two extraction modes available:

#### Sequential Extraction (Lower Memory, ~4GB)
Loads one model at a time. Slower but works on smaller GPUs.

```bash
# Extract from training set (5208 samples)
python3 feasible/token_alignment/extract_tokens_train.py \
    --dataset_dir /path/to/my_egpt_dsec_train_1s \
    --output_file ./feasible/token_alignment/train_tokens.json \
    --single_question  # Use only top question (faster)
    # Remove --single_question to use all 50 questions
```

#### Parallel Extraction (Faster, ~8GB) - Recommended
Loads both models simultaneously. ~1.5-2x faster.

```bash
# Extract with top 10 questions (~15 hours)
python3 feasible/token_alignment/extract_tokens_parallel.py \
    --dataset_dir /path/to/my_egpt_dsec_train_1s \
    --output_file ./feasible/token_alignment/train_tokens_10q.json \
    --max_questions 10

# Extract with all 50 questions (~70 hours)
python3 feasible/token_alignment/extract_tokens_parallel.py \
    --dataset_dir /path/to/my_egpt_dsec_train_1s \
    --output_file ./feasible/token_alignment/train_tokens_50q.json
```

**Sequential Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_dir` | 1s train set | Path to DSEC dataset |
| `--output_file` | train_tokens_full.json | Output JSON path |
| `--questions_file` | top50_questions.json | Questions to use |
| `--single_question` | False | Use only top 1 question |
| `--max_samples` | -1 (all) | Limit number of samples |
| `--max_new_tokens` | 50 | Tokens to generate |

**Parallel Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_dir` | 1s train set | Path to DSEC dataset |
| `--output_file` | train_tokens_parallel.json | Output JSON path |
| `--questions_file` | top50_questions.json | Questions to use |
| `--max_questions` | -1 (all) | Limit questions (e.g., 10) |
| `--max_samples` | -1 (all) | Limit number of samples |
| `--max_new_tokens` | 50 | Tokens to generate |

### Step 2: Train TokenAdapter

```bash
python3 feasible/token_alignment/train_and_evaluate.py \
    --train_benchmark ./feasible/token_alignment/train_tokens.json \
    --test_benchmark ./feasible/token_alignment/test_tokens.json \
    --task_name 1q \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --early_stopping 10
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--train_benchmark` | required | Training tokens JSON |
| `--test_benchmark` | required | Test tokens JSON |
| `--task_name` | auto | Name for task folder |
| `--num_epochs` | 50 | Max training epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 1e-4 | Initial LR |
| `--early_stopping` | 10 | Patience for early stop |

### Step 3: View Results

Results are saved in timestamped task folders:

```
task/
└── 1q_20260128_143000/
    ├── best_model.pt          # Best model checkpoint
    ├── results.json           # Full metrics & history
    ├── RESULTS.md             # Human-readable summary
    ├── training_curves.png    # Loss & accuracy plots
    ├── loss_curve.png         # Training loss
    └── accuracy_curve.png     # Acceptance rates
```

## Scripts

### Training & Evaluation

| Script | Description |
|--------|-------------|
| `extract_tokens_train.py` | Sequential token extraction (lower memory) |
| `extract_tokens_parallel.py` | Parallel token extraction (faster) |
| `train_and_evaluate.py` | Train TokenAdapter and evaluate |
| `token_adapter.py` | TokenAdapter model architecture |

### Workflow Scripts

| Script | Description |
|--------|-------------|
| `run_1q_workflow.sh` | Single question workflow (~3 hours) |
| `run_10q_workflow.sh` | 10 questions workflow (~20 hours) |
| `run_full_workflow.sh` | 50 questions workflow (~72 hours) |

### Evaluation Scripts

| Script | Description |
|--------|-------------|
| `evaluate_speculative.py` | Evaluate speculative decoding with TokenAdapter |
| `debug_tokens.py` | Debug token alignment issues |

## Paths

### Output Paths

| Type | Path |
|------|------|
| **Token Data** | `feasible/token_alignment/train_tokens_*.json` |
| **Task Results** | `feasible/token_alignment/task/<name>_YYYYMMDD_HHMMSS/` |
| **Starred Checkpoints** | `feasible/token_alignment/task/starred/` |

### Checkpoint Structure

Each task folder contains:
```
task/<name>_YYYYMMDD_HHMMSS/
├── best_model.pt           # Trained TokenAdapter weights (173MB)
├── results.json            # Full metrics & training history
├── RESULTS.md              # Human-readable summary
├── training_curves.png     # Combined loss & accuracy plots
├── loss_curve.png          # Training loss curve
└── accuracy_curve.png      # Acceptance rate curves
```

## Starred Checkpoints

Two trained checkpoints are available in `task/starred/`:

### 1. Single Question: `1q_20260128_151847`

| Metric | Train | Test |
|--------|-------|------|
| Samples | 5,200 | 1,100 |
| Baseline | 1.77% | 1.58% |
| **With Adapter** | **27.21%** | **27.90%** |
| Top-5 | 51.40% | 51.64% |
| Improvement | +25.45% | +26.32% |

**Checkpoint:** `task/starred/1q_20260128_151847/best_model.pt`

### 2. Ten Questions: `10q_20260129_204744`

| Metric | Train | Test |
|--------|-------|------|
| Samples | 52,000 | 11,000 |
| Baseline | 8.03% | 8.05% |
| **With Adapter** | **23.24%** | **21.48%** |
| Top-5 | 53.80% | 51.52% |
| Improvement | +15.21% | +13.44% |

**Checkpoint:** `task/starred/10q_20260129_204744/best_model.pt`

**Additional Files:**
- `TOKEN_ALIGNMENT_ANALYSIS.md` - Token-by-token comparison
- `BENCHMARK_RESULTS.md` - Full benchmark with 100 evaluations
- `full_benchmark_20260129_225244.json` - Raw benchmark data

### Loading Checkpoints

```python
import torch
from feasible.token_alignment.token_adapter import TokenAdapter

# Load 10-question checkpoint (recommended)
adapter = TokenAdapter()
checkpoint = torch.load(
    'feasible/token_alignment/task/starred/10q_20260129_204744/best_model.pt'
)
adapter.load_state_dict(checkpoint['model_state_dict'])
adapter.eval()

# Use adapter
egpt_tokens = torch.tensor([[1, 2, 3, 4, 5]])  # EventGPT tokens
vl_logits = adapter(egpt_tokens)  # Predicted Video-LLaVA logits
vl_tokens = vl_logits.argmax(dim=-1)  # Predicted tokens
```

## File Structure

```
token_alignment/
├── README.md                    # This file
├── WORKFLOW.md                  # Detailed workflow documentation
│
├── # Extraction & Training
├── extract_tokens_train.py      # Sequential token extraction
├── extract_tokens_parallel.py   # Parallel token extraction (recommended)
├── train_and_evaluate.py        # Train TokenAdapter & evaluate
├── token_adapter.py             # TokenAdapter model architecture
├── base.py                      # Base classes & utilities
│
├── # Workflow Scripts
├── run_1q_workflow.sh           # Single question workflow (~3 hours)
├── run_10q_workflow.sh          # 10 questions workflow (~20 hours)
├── run_full_workflow.sh         # 50 questions workflow (~72 hours)
│
├── # Data
├── top50_questions.json         # Top 50 DSEC questions
├── train_tokens_*.json          # Extracted token pairs
│
├── # Results
└── task/
    ├── starred/                 # Best checkpoints (symlinked)
    │   ├── 1q_20260128_151847/  # Single question (27.9% accuracy)
    │   └── 10q_20260129_204744/ # 10 questions (21.5% accuracy)
    └── <name>_YYYYMMDD_HHMMSS/  # Other task folders
```

## Datasets

| Duration | Train Samples | Test Samples | Train Path |
|----------|---------------|--------------|------------|
| 500ms | 10,475 | 2,220 | my_egpt_dsec_train_500ms |
| 1s | 5,208 | 1,100 | my_egpt_dsec_train_1s |
| 2s | 2,604 | 550 | my_egpt_dsec_train_2s |
| 4s | 1,302 | 275 | my_egpt_dsec_train_4s |

**Base paths:**
- Train: `/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_train/`
- Test: `/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_test/`

## Architecture & Pipeline

### Full Speculative Decoding Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SPECULATIVE DECODING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────┐                                                            │
│  │   Event Image   │ [H, W, C]                                                  │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         STAGE 1: DRAFT (EventGPT)                        │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │   │
│  │  │ Vision Enc.  │───▶│   Prefill    │───▶│   AR Decode (γ tokens)   │   │   │
│  │  │  [1, 576, D] │    │  [1, L, D]   │    │   [1, γ] token IDs       │   │   │
│  │  └──────────────┘    └──────────────┘    └────────────┬─────────────┘   │   │
│  └───────────────────────────────────────────────────────┼─────────────────┘   │
│                                                          │                      │
│                                                          ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      STAGE 2: ADAPT (TokenAdapter)                       │   │
│  │                                                                          │   │
│  │   Input: EGPT tokens [1, γ]    ───▶    Output: VL logits [1, γ, 32010]  │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                          │                      │
│                                                          ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     STAGE 3: VERIFY (Video-LLaVA)                        │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │   │
│  │  │ Vision Enc.  │───▶│   Prefill    │───▶│  Batch Verify γ tokens   │   │   │
│  │  │  [8, 576, D] │    │  [1, L, D]   │    │   Compare distributions  │   │   │
│  │  └──────────────┘    └──────────────┘    └────────────┬─────────────┘   │   │
│  └───────────────────────────────────────────────────────┼─────────────────┘   │
│                                                          │                      │
│                                                          ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         STAGE 4: ACCEPT/REJECT                           │   │
│  │                                                                          │   │
│  │   For each position i = 0, 1, ..., γ-1:                                  │   │
│  │     if adapter_pred[i] == vl_sample[i]:  ACCEPT                         │   │
│  │     else:                                 REJECT (stop here)             │   │
│  │                                                                          │   │
│  │   Result: Accept k consecutive tokens (0 ≤ k ≤ γ)                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### TokenAdapter Architecture (45M params)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TokenAdapter Internal Structure                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUT: draft_tokens                                                            │
│   Shape: [batch, seq_len]        Example: [32, 50] = 32 samples × 50 tokens     │
│                                                                                  │
│          │                                                                       │
│          ▼                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Token Embedding (nn.Embedding)                                          │   │
│   │  ─────────────────────────────                                           │   │
│   │  vocab_size: 32,010  →  embed_dim: 512                                   │   │
│   │  Parameters: 32,010 × 512 = 16.4M                                        │   │
│   │                                                                          │   │
│   │  Input:  [batch, seq_len]           →  [32, 50]                          │   │
│   │  Output: [batch, seq_len, 512]      →  [32, 50, 512]                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          ▼                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Position Embedding (nn.Embedding)                                       │   │
│   │  ──────────────────────────────────                                      │   │
│   │  max_positions: 64  →  embed_dim: 512                                    │   │
│   │  Parameters: 64 × 512 = 32.8K                                            │   │
│   │                                                                          │   │
│   │  Input:  positions [batch, seq_len] →  [32, 50]                          │   │
│   │  Output: [batch, seq_len, 512]      →  [32, 50, 512]                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          ▼  (element-wise addition)                                              │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Combined: token_emb + pos_emb                                           │   │
│   │  Shape: [batch, seq_len, 512]       →  [32, 50, 512]                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          ▼                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Transformer Encoder (4 layers × TransformerEncoderLayer)                │   │
│   │  ─────────────────────────────────────────────────────────────────────   │   │
│   │                                                                          │   │
│   │  Each layer:                                                             │   │
│   │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│   │  │  Multi-Head Self-Attention                                       │    │   │
│   │  │  • num_heads: 8                                                  │    │   │
│   │  │  • head_dim: 512 / 8 = 64                                        │    │   │
│   │  │  • Causal mask: upper triangular [seq, seq]                      │    │   │
│   │  │  Input:  [32, 50, 512]  →  Output: [32, 50, 512]                 │    │   │
│   │  ├─────────────────────────────────────────────────────────────────┤    │   │
│   │  │  Feed-Forward Network                                            │    │   │
│   │  │  • Linear(512 → 2048) + GELU + Linear(2048 → 512)               │    │   │
│   │  │  • Dropout: 0.1                                                  │    │   │
│   │  │  Input:  [32, 50, 512]  →  Output: [32, 50, 512]                 │    │   │
│   │  └─────────────────────────────────────────────────────────────────┘    │   │
│   │                                                                          │   │
│   │  Parameters per layer: ~2.1M                                             │   │
│   │  Total (4 layers): ~8.4M                                                 │   │
│   │                                                                          │   │
│   │  Input:  [batch, seq_len, 512]      →  [32, 50, 512]                     │   │
│   │  Output: [batch, seq_len, 512]      →  [32, 50, 512]                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          ▼                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Output Projection                                                       │   │
│   │  ─────────────────                                                       │   │
│   │  LayerNorm(512) + Linear(512 → 32,010)                                   │   │
│   │  Parameters: 512 + 512×32,010 = 16.4M                                    │   │
│   │                                                                          │   │
│   │  Input:  [batch, seq_len, 512]      →  [32, 50, 512]                     │   │
│   │  Output: [batch, seq_len, 32010]    →  [32, 50, 32010]                   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          ▼  (+ token_bias [32010])                                               │
│                                                                                  │
│   OUTPUT: logits                                                                 │
│   Shape: [batch, seq_len, 32010]        Example: [32, 50, 32010]                │
│                                                                                  │
│   To get predicted tokens: logits.argmax(dim=-1) → [32, 50]                     │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PARAMETER SUMMARY                                                               │
│  ─────────────────                                                               │
│  Token Embedding:     16,389,120  (32,010 × 512)                                │
│  Position Embedding:      32,768  (64 × 512)                                    │
│  Transformer (4L):     8,397,824  (4 × ~2.1M)                                   │
│  Output Projection:   16,389,624  (512 + 512 × 32,010)                          │
│  Token Bias:              32,010                                                 │
│  ─────────────────────────────────────────                                       │
│  TOTAL:               45,486,346  (~173 MB)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Example

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CONCRETE EXAMPLE: Single Sample                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  EGPT generates: "In the scene, there is a car parked on the side of a"         │
│  EGPT tokens:    [512, 278, 9088, 29892, 727, 338, 263, 1559, 14089, ...]       │
│                                                                                  │
│          │                                                                       │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          TokenAdapter                                     │   │
│  │                                                                          │   │
│  │  Input:  [512, 278, 9088, 29892, 727, 338, 263, 1559, 14089, ...]       │   │
│  │          "In"  "the" "scene" ","  "there" "is" "a"  "car" "park"        │   │
│  │                                                                          │   │
│  │  Output: [1820, 3161, 297, 445, 9088, 3160, 263, 1559, 19500, ...]      │   │
│  │          "key" "elements" "in" "this" "scene" "include" "a" "car" "driving" │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     Video-LLaVA Verification                              │   │
│  │                                                                          │   │
│  │  VL Actual: [1820, 3161, 297, 445, 9088, 3160, 263, 1559, 19500, ...]   │   │
│  │             "key" "elements" "in" "this" "scene" "include" "a" "car" "driving" │   │
│  │                                                                          │   │
│  │  Position:    0      1      2     3      4       5      6    7     8     │   │
│  │  Match:       ✓      ✓      ✓     ✓      ✓       ✓      ✓    ✓     ✓     │   │
│  │                                                                          │   │
│  │  → Accept 9 consecutive tokens!                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  BUT IN PRACTICE (autoregressive benchmark):                                    │
│                                                                                  │
│  Adapter predicts: "key"  but VL outputs: "The"                                 │
│  Position 0 fails → Accept 0 tokens                                             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Why It Fails: Visual Input Mismatch

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ROOT CAUSE: DIFFERENT INPUTS                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────┐     ┌─────────────────────────────┐           │
│  │       EventGPT Input        │     │     Video-LLaVA Input       │           │
│  │                             │     │                             │           │
│  │  ┌───────────────────────┐ │     │  ┌───────────────────────┐ │           │
│  │  │    Event Image        │ │     │  │    8 Video Frames     │ │           │
│  │  │  (sparse motion data) │ │     │  │   (dense RGB data)    │ │           │
│  │  │                       │ │     │  │  ┌──┐┌──┐┌──┐┌──┐    │ │           │
│  │  │   . .   . . .         │ │     │  │  │F1││F2││F3││F4│    │ │           │
│  │  │  . . . . . .          │ │     │  │  └──┘└──┘└──┘└──┘    │ │           │
│  │  │    .   . .  .         │ │     │  │  ┌──┐┌──┐┌──┐┌──┐    │ │           │
│  │  │  . . .   . . .        │ │     │  │  │F5││F6││F7││F8│    │ │           │
│  │  └───────────────────────┘ │     │  │  └──┘└──┘└──┘└──┘    │ │           │
│  │                             │     │  └───────────────────────┘ │           │
│  │  Sees: Static snapshot      │     │  Sees: Motion over time    │           │
│  │  Output: "car parked"       │     │  Output: "car driving"     │           │
│  └─────────────────────────────┘     └─────────────────────────────┘           │
│                                                                                  │
│  Same scene, different interpretations → Different token sequences              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Actual Results

### Training Metrics (Token-Level)

| Dataset | Baseline | With TokenAdapter | Top-5 |
|---------|----------|-------------------|-------|
| Train | 8.0% | 21.5% | 51.5% |
| Test | 8.0% | 21.5% | 51.5% |

**Improvement:** +13.5% over baseline (8% → 21.5%)

### Benchmark Results (Autoregressive Generation)

| Metric | Value |
|--------|-------|
| **Consecutive Matches** | 0.0 avg |
| **Total Matches** | 2.5 / 37.6 avg |
| **Match Rate** | 6.2% |

**Critical Finding:** 100% of samples have 0 consecutive matches, meaning speculative decoding would accept 0 tokens in every case.

### Training vs Benchmark Gap

| Evaluation | Match Rate |
|------------|------------|
| **Training** (ground truth tokens) | 21.5% |
| **Benchmark** (autoregressive tokens) | 6.2% |

The gap exists because:
1. **Training:** Uses ground truth EGPT/VL token pairs from the same scene
2. **Benchmark:** Uses autoregressively generated tokens (EGPT from event image, VL from video)

## Why Cross-Modal SD Fails

### Root Cause Analysis

| Factor | Impact |
|--------|--------|
| **Different Visual Inputs** | EGPT sees event image (sparse motion), VL sees video frames (dense RGB) |
| **Different Descriptions** | EGPT: "car **parked** on the side", VL: "car **driving** down winding road" |
| **First Token Mismatch** | Adapter predicts "key"/"image", VL outputs "The"/"This" |
| **Consecutive Requirement** | SD only accepts tokens consecutively from position 0 |

### Example First Token Failures

| Sample | Adapter Predicts | VL Outputs |
|--------|------------------|------------|
| Q0 | "key" | "The" |
| Q1 | "image" | "The" |
| Q2 | "image" | "This" |
| Q3 | "image" | "The" |
| Q4 | "the" | "In" |

## Speculative Decoding Speedup

**Theoretical:** With acceptance rate α and draft length γ:
```
Expected accepted = (1 - α^(γ+1)) / (1 - α)
```

| Acceptance (α) | γ=5 tokens | Speedup |
|----------------|------------|---------|
| 3% (baseline) | 1.2 tokens | 1.0x |
| 30% | 1.8 tokens | 1.4x |
| 50% | 2.5 tokens | 1.8x |
| 70% | 3.5 tokens | 2.5x |

**Actual (Cross-Modal):** 0 consecutive tokens accepted → **No speedup possible**

## Conclusion

**Cross-modal speculative decoding between EventGPT and Video-LLaVA is not viable** with the TokenAdapter approach because:

1. The models see different visual inputs (event image vs video frames)
2. They generate semantically different descriptions of the same scene
3. Even with successful style transfer, word choices diverge
4. Speculative decoding requires exact consecutive token matches from position 0

### Recommendations

| Goal | Recommendation |
|------|----------------|
| **Latency-critical apps** | Use EventGPT alone (2x faster prefill than VL) |
| **Quality-critical apps** | Use Video-LLaVA alone (better visual understanding) |
| **Future research** | Explore embedding-level alignment instead of token-level |

### Detailed Analysis

See `task/starred/10q_20260129_204744/` for:
- `TOKEN_ALIGNMENT_ANALYSIS.md` - Token-by-token comparison showing adapter behavior
- `BENCHMARK_RESULTS.md` - Full 100-sample benchmark with output examples

---

## End-to-End Benchmark Integration

The TokenAdapter is integrated with the benchmark scripts for end-to-end evaluation. You can compare **baseline** (original model) vs **aligned** (with TokenAdapter) acceptance rates.

### Running Benchmarks with TokenAdapter

```bash
# Parallel prefill benchmark with TokenAdapter
python feasible/benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py \
  --use_token_adapter \
  --max_samples 100

# Specify a specific checkpoint
python feasible/benchmark_parallel_prefill/benchmark_parallel_prefill_5stages.py \
  --use_token_adapter \
  --token_adapter_path ./feasible/token_alignment/task/10q_*/best_model.pt \
  --max_samples 100

# Inference benchmark with TokenAdapter
python feasible/benchmark_inference/benchmark_inference_5stages.py \
  --use_token_adapter \
  --max_samples 100
```

### Benchmark Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_token_adapter` | False | Enable TokenAdapter for aligned evaluation |
| `--token_adapter_path` | Auto-detect | Path to trained checkpoint (auto-detects latest) |

### Output Metrics

The benchmark reports both baseline and aligned metrics:

| Metric | Description |
|--------|-------------|
| `acceptance_rate_avg` | Baseline acceptance (direct token matching) |
| `aligned_acceptance_rate_avg` | Aligned acceptance (with TokenAdapter) |
| `aligned_top5_rate_avg` | Top-5 aligned acceptance |
| `aligned_improvement` | Improvement over baseline |

Example output:
```
Baseline Acceptance Rate: 2.5%
Aligned Acceptance Rate:  27.9% (with TokenAdapter)
Aligned Top-5 Rate:       51.6%
Improvement:              +25.4%
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 16  # or 8
```

### CUDA Token ID Error
If you see `nll_loss_forward_reduce_cuda_kernel` error:
- Video-LLaVA may use token IDs > 32000
- Fixed in current code (vocab_size=32010)

### Low Validation Accuracy
- Ensure train and test use same extraction format
- Check token lengths match (should be 50 each)
- Try increasing model capacity: `--num_layers 6`

## Next: Feature-Level Alignment

**Status:** Recommended path forward after token-level alignment proved insufficient.

Token-level alignment achieved 21.5% on training data but **0% consecutive matches** on autoregressive generation, making speculative decoding impossible.

### Why Feature-Level?

| Approach | Training Accuracy | Benchmark (AR) | Issue |
|----------|-------------------|----------------|-------|
| **Token-level** | 21.5% | 0% consecutive | Content mismatch |
| **Feature-level** | TBD | TBD | May capture semantic similarity |

Feature-level alignment can potentially succeed because:
- Hidden states capture **semantic meaning** before token discretization
- Even if "street" vs "winding road", semantic vectors may be similar
- Enables soft matching instead of exact token matching

### Implementation Path

```bash
# Extract hidden states for feature alignment
python3 feasible/token_alignment/extract_tokens_parallel.py \
    --dataset_dir /path/to/dataset \
    --output_file ./train_tokens_10q.json \
    --max_questions 10 \
    --extract_hidden_states  # Extract hidden states

# Train feature alignment adapter
python3 feasible/feature_alignment/train_feature_adapter.py \
    --hidden_states ./train_hidden_states_10q.pt
```

See `feasible/feature_alignment/` for implementation details.

---

## Target Metrics for Viable Speculative Decoding

For speculative decoding to provide meaningful speedup, the following metrics must be achieved:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TARGET METRICS FOR VIABILITY                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  METRIC                    │ CURRENT (Token)  │ TARGET         │ FOR SPEEDUP    │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Consecutive Match Rate    │ 0%               │ ≥50%           │ 1.5x+          │
│  Avg Tokens Accepted       │ 0.0              │ ≥3.0           │ 1.5x+          │
│  First Token Accuracy      │ 0%               │ ≥70%           │ Required       │
│  End-to-End Speedup        │ 1.0x             │ ≥1.5x          │ Viable         │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  SPEEDUP FORMULA                                                                 │
│  ───────────────                                                                 │
│  With acceptance rate α and draft length γ:                                      │
│                                                                                  │
│  Expected Tokens = (1 - α^(γ+1)) / (1 - α)                                      │
│                                                                                  │
│  │ α (Accept Rate) │ γ=5 drafts │ Tokens Accepted │ Speedup │                   │
│  │─────────────────│────────────│─────────────────│─────────│                   │
│  │ 30%             │ 5          │ 1.4             │ 1.2x    │                   │
│  │ 50%             │ 5          │ 2.0             │ 1.5x    │  ← Minimum viable │
│  │ 70%             │ 5          │ 3.3             │ 2.0x    │  ← Good           │
│  │ 85%             │ 5          │ 5.0             │ 2.5x    │  ← Excellent      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Current Status vs Target

| Metric | Token-Level (Current) | Target | Gap |
|--------|----------------------|--------|-----|
| Training Accuracy | 21.5% | N/A | N/A |
| **Benchmark Consecutive** | **0%** | **≥50%** | **-50%** |
| First Token Match | 0% | ≥70% | -70% |
| Avg Tokens Accepted | 0.0 | ≥3.0 | -3.0 |
| Speedup | 1.0x | ≥1.5x | -0.5x |

**Conclusion:** Token-level approach cannot bridge this gap due to fundamental input mismatch.

---

## Further Directions

**Token-level alignment has been evaluated and found insufficient for cross-modal speculative decoding.** The fundamental issue is that EGPT and VL see different visual inputs and generate semantically different content.

### Why Token-Level Failed

| Approach | Result | Issue |
|----------|--------|-------|
| Direct token matching | 8% | Vocabulary mismatch |
| TokenAdapter | 21.5% training, 0% consecutive | Content mismatch |

### Recommended Next Steps

### 1. Feature-Level Speculative Decoding (Most Promising)

Use hidden state alignment instead of token-level.

| Metric | Token-Level | Feature-Level (Expected) |
|--------|-------------|--------------------------|
| Alignment Target | Exact token match | Hidden state similarity |
| Tolerance | 0 (exact) | Cosine sim ≥ 0.9 |
| Expected Accuracy | 21.5% | 40-60% |
| Consecutive Match | 0% | 30-50% |

- **Pro:** Bypasses tokenizer mismatch, captures semantic similarity
- **Con:** More complex implementation
- **See:** `feasible/feature_alignment/` for implementation

### 2. Same-Model Speculative Decoding

Use a smaller/quantized version of Video-LLaVA as draft.

| Metric | Cross-Modal | Same-Model (Expected) |
|--------|-------------|----------------------|
| Draft Model | EventGPT (7B) | VL-4bit or VL-small |
| Acceptance Rate | 0% | 70-85% |
| Speedup | 1.0x | 1.8-2.5x |

- **Pro:** Guaranteed output distribution alignment
- **Con:** Doesn't leverage EventGPT's speed advantage

### 3. Parallel Prefill Only (No SD)

Exploit EventGPT's fast prefill without speculative decoding.

| Metric | Value |
|--------|-------|
| EGPT Prefill | 112ms |
| VL Prefill | 383ms |
| Prefill Speedup | 3.4x |
| Decode Speedup | 1.0x (none) |

- **Pro:** 3.4x prefill speedup, simple implementation
- **Con:** No decode speedup
- **See:** `feasible/egpt_prefill_only/` for implementation

### 4. Use EventGPT Standalone

For latency-critical applications, use EventGPT alone.

| Metric | EventGPT | Video-LLaVA |
|--------|----------|-------------|
| Prefill | 112ms | 383ms |
| Per-token | ~15ms | ~30ms |
| Total (50 tokens) | ~860ms | ~1880ms |
| Speedup | 2.2x | 1.0x |

- **Pro:** 2x faster than VL, works with event cameras
- **Con:** Cannot leverage VL's richer visual understanding

### Approaches Not Recommended

| Approach | Why Not Viable |
|----------|----------------|
| Fine-tune EventGPT LM Head | Would need to see video frames, defeats purpose |
| Prompt Engineering | Cannot fix fundamental input difference |
| Draft-Target Fine-tuning | Expensive, still has input mismatch |
| Medusa-style Multi-head | Still requires output distribution alignment |

---

## Research Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RESEARCH ROADMAP                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  COMPLETED                                                                       │
│  ─────────                                                                       │
│  ✓ Token-level alignment (TokenAdapter)                                         │
│    - Training: 21.5% accuracy                                                    │
│    - Benchmark: 0% consecutive matches                                           │
│    - Result: NOT VIABLE for cross-modal SD                                       │
│                                                                                  │
│  IN PROGRESS                                                                     │
│  ───────────                                                                     │
│  □ Feature-level alignment                                                       │
│    - Target: 40-60% hidden state similarity                                      │
│    - Target: ≥30% consecutive acceptance                                         │
│    - See: feasible/feature_alignment/                                            │
│                                                                                  │
│  FUTURE                                                                          │
│  ──────                                                                          │
│  □ EAGLE-style speculative decoding                                              │
│    - Use EGPT hidden states to predict VL next-token                             │
│    - Target: ≥50% acceptance rate                                                │
│                                                                                  │
│  □ Cascaded speculative decoding                                                 │
│    - EGPT → Feature Adapter → VL                                                 │
│    - Combined prefill + decode speedup                                           │
│    - Target: 1.5-2.0x end-to-end speedup                                         │
│                                                                                  │
│  □ Hybrid approach (parallel prefill + SD)                                       │
│    - Prefill: 3.4x speedup (parallel)                                            │
│    - Decode: 1.3-1.5x speedup (feature SD)                                       │
│    - Target: 2.0x+ end-to-end                                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## References

1. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", 2023
2. Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", 2024
3. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework", 2024
4. Hinton et al., "Distilling the Knowledge in a Neural Network", 2015

## Changelog

### 2026-01-29 (Evening)
- **Full Benchmark Evaluation**: Ran 100 evaluations (10 samples × 10 questions)
  - Found 0% consecutive matches across all samples
  - 6.2% total match rate vs 21.5% training accuracy
  - Root cause: EGPT and VL generate semantically different descriptions
- **Conclusion**: Cross-modal speculative decoding is **not viable** with TokenAdapter
  - First token always fails to match
  - Even perfect style transfer cannot fix content differences
- **New Analysis Files**: `task/starred/10q_20260129_204744/`
  - `TOKEN_ALIGNMENT_ANALYSIS.md` - Detailed token comparison
  - `BENCHMARK_RESULTS.md` - Full benchmark with output examples

### 2026-01-29 (Morning)
- **Benchmark Integration**: Added `--use_token_adapter` flag to benchmark scripts
  - `benchmark_parallel_prefill_5stages.py` now supports TokenAdapter evaluation
  - `benchmark_inference_5stages.py` now supports TokenAdapter evaluation
  - Auto-detects latest trained checkpoint if path not specified
  - Reports both baseline and aligned acceptance rates
- **10-Question Workflow**: Added `run_10q_workflow.sh`
  - Balanced option between 1q and 50q
  - Trains for full 50 epochs (no early stopping)
  - Estimated ~20 hours total
- **Training Configuration**: Added `--early_stopping 999` option to disable early stopping

### 2026-01-28
- Added multi-question extraction (top 50 questions)
- Added task folder system with timestamps
- Added training curves visualization
- Fixed token extraction to save only generated tokens
- Updated workflow scripts for automated pipeline

### 2026-01-27
- Initial TokenAdapter implementation
- Basic extraction and training pipeline
