# IMU-Based MLLMs for Speculative Decoding

Research on IMU-based Multimodal Large Language Models as draft models for EventGPT speculative decoding.

## Directory Structure

```
imu_mllm/
├── README.md                    # This file
├── IMU_MLLM_RESEARCH.md         # Detailed research notes
├── LLaSA/                       # LLaMA + LIMU-BERT (BEST MATCH)
│   └── checkpoints/
│       └── LLaSA-7B/            # Downloading from HuggingFace
├── OneLLM/                      # LLaMA2 + Universal Encoder
│   └── checkpoints/
│       └── OneLLM-7B/           # Downloading from HuggingFace
└── SensorLLM/                   # Chronos + LLaMA (training framework)
    └── checkpoints/
        └── chronos-t5-large/    # Time-series encoder
```

## Model Comparison

| Model | Encoder | Decoder | Tokenizer | EventGPT Compatible |
|-------|---------|---------|-----------|---------------------|
| **LLaSA** | LIMU-BERT (62K) | Vicuna-7B/13B | LLaMA | ✅ YES |
| **OneLLM** | 1D Conv + CLIP-ViT | LLaMA2-7B | LLaMA2 | ✅ YES |
| **SensorLLM** | Chronos-t5-large | LLaMA (any) | LLaMA | ✅ YES |

## Quick Start

### LLaSA (Recommended)
```bash
cd LLaSA
# Model: checkpoints/LLaSA-7B/
# Uses Vicuna tokenizer (LLaMA compatible)
```

### OneLLM
```bash
cd OneLLM
pip install -r requirements.txt

# Run demo
python demos/cli.py \
    --image_path ${IMAGE_PATH} \
    --gpu_ids 0 \
    --tokenizer_path config/llama2/tokenizer.model \
    --llama_config config/llama2/7B.json \
    --pretrained_path checkpoints/OneLLM-7B/consolidated.00-of-01.pth
```

### SensorLLM
```bash
cd SensorLLM
pip install -r requirements.txt

# Train alignment stage
torchrun --nproc_per_node=1 sensorllm/train/train_mem.py \
    --model_name_or_path [LLaMA_PATH] \
    --pt_encoder_backbone_ckpt checkpoints/chronos-t5-large \
    --dataset mhealth \
    ...
```

## Architecture for Speculative Decoding

```
                    ┌─────────────────────┐
                    │   IMU Data (6-ch)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────┐ ┌──────────────┐
    │ LIMU-BERT (62K) │ │ 1D Conv+ViT │ │ Chronos-t5   │
    │    (LLaSA)      │ │  (OneLLM)   │ │ (SensorLLM)  │
    └────────┬────────┘ └──────┬──────┘ └──────┬───────┘
             │                 │               │
             ▼                 ▼               ▼
    ┌─────────────────────────────────────────────────┐
    │              MLP Projection Layer               │
    └─────────────────────┬───────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────┐
    │     LLaMA Decoder (Vicuna/LLaMA2/LLaMA3)        │
    │           (Same tokenizer as EventGPT)          │
    └─────────────────────┬───────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────┐
    │                 Draft Tokens                     │
    └─────────────────────┬───────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────┐
    │     EventGPT / VideoLLaVA Verification          │
    └─────────────────────────────────────────────────┘
```

## Datasets

| Dataset | Source | IMU | Video | Narrations |
|---------|--------|-----|-------|------------|
| Ego4D | ego4d-data.org | ✅ | ✅ | ✅ Dense |
| Ego-Exo4D | ego-exo4d-data.org | ✅ (2x) | ✅ | ✅ |
| SensorCaps | LLaSA repo | ✅ | - | ✅ 35,960 |
| OpenSQA | LLaSA repo | ✅ | - | ✅ 179,727 Q&A |

## Download Status

Check download progress:
```bash
# LLaSA
ls -la LLaSA/checkpoints/LLaSA-7B/

# OneLLM
ls -la OneLLM/checkpoints/OneLLM-7B/

# SensorLLM (Chronos encoder)
ls -la SensorLLM/checkpoints/chronos-t5-large/
```

## References

- [LLaSA Paper](https://arxiv.org/abs/2406.14498) | [GitHub](https://github.com/BASHLab/LLaSA)
- [OneLLM Paper](https://arxiv.org/abs/2312.03700) | [GitHub](https://github.com/csuhan/OneLLM)
- [SensorLLM Paper](https://arxiv.org/abs/2410.10624) | [GitHub](https://github.com/cruiseresearchgroup/SensorLLM)
- [LIMU-BERT](https://github.com/dapowan/LIMU-BERT-Public)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
