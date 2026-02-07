# Stage 1: Feature Extraction

> Author: Alice Zhang
> Date: 2026-02-07

Extract decoder hidden states from both EventGPT and Video-LLaVA on the same DSEC scenes/questions. These paired hidden states are the training data for all adapters.

```bash
export REPO=/home/ps/Documents/code/EventGPT
cd $REPO
```

---

## Files

| File | Description |
|------|-------------|
| `extract_vl_lm_head.py` | Extract VL LM head weights for token-level metrics (one-time) |
| `extract_hidden_states.py` | Run both models, save paired hidden states as chunked `.pt` files |
| `monitor_extraction.sh` | Monitor long-running extraction progress |

---

## Stage 0: Extract VL LM Head (one-time)

```bash
python pipeline/feature_extraction/extract_vl_lm_head.py
```

| | |
|---|---|
| **Input** | HuggingFace `LanguageBind/Video-LLaVA-7B-hf` (auto-downloaded) |
| **Output** | `pipeline/feature_extraction/data/vl_lm_head.pt` (~256MB, float32 `[32000, 4096]`) |
| **Time** | ~2 min |

---

## Stage 1: Extract Hidden States

### Train split (~24h)

```bash
python pipeline/feature_extraction/extract_hidden_states.py \
    --split train --chunked --quant 4bit \
    --dataset_dir $REPO/data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
    --max_samples -1 --max_questions 10 --max_new_tokens 50 --chunk_size 1000
```

### Test split (~5h)

```bash
python pipeline/feature_extraction/extract_hidden_states.py \
    --split test --chunked --quant 4bit \
    --dataset_dir $REPO/data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \
    --max_samples -1 --max_questions 10 --max_new_tokens 50 --chunk_size 1000
```

### Resume interrupted extraction

```bash
python pipeline/feature_extraction/extract_hidden_states.py \
    --split train --chunked --quant 4bit --resume
```

Default output: `pipeline/feature_extraction/data/`

---

## CLI Reference (`extract_hidden_states.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_dir` | hardcoded | Path to DSEC dataset with event images + MP4 |
| `--output_dir` | `pipeline/feature_extraction/data/` | Output directory for chunked data |
| `--split` | `train` | `train` or `test` |
| `--chunked` | off | Chunked saving (required for large datasets) |
| `--chunk_size` | 1000 | Samples per chunk (~1.6GB each) |
| `--quant` | `4bit` | Model quantization (`4bit`, `8bit`, `fp16`) |
| `--max_samples` | -1 | Limit samples (-1 = all) |
| `--max_questions` | 10 | Questions per sample |
| `--max_new_tokens` | 50 | Max generated tokens per question |
| `--resume` | off | Resume from last saved chunk |
| `--save_interval` | 1000 | Emergency checkpoint interval |

---

## Output

```
pipeline/feature_extraction/data/
├── vl_lm_head.pt                  # VL LM head (~256MB)
├── chunked_train_1s_4bit/         # 52K samples (5,200 scenes x 10 questions), ~83GB
│   ├── index.json                 # Chunk metadata
│   └── chunks/
│       ├── chunk_000000.pt        # samples 0-999, ~1.6GB each
│       └── ...                    # 52 chunks total
└── chunked_test_1s_4bit/          # 11K samples (1,100 scenes x 10 questions), ~18GB
    ├── index.json
    └── chunks/                    # 11 chunks total
```

### Chunk format

Each `chunk_XXXXXX.pt` contains:

```python
{
    'egpt_hidden': torch.Tensor,  # [N, max_seq, 4096] float32 — EventGPT decoder hidden states
    'vl_hidden':   torch.Tensor,  # [N, max_seq, 4096] float32 — Video-LLaVA decoder hidden states
    'seq_lens':    torch.Tensor,  # [N] int64 — actual sequence lengths (rest is padding)
}
```
