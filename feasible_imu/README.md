# IMU MLLM Benchmarking

5-stage wall-clock time benchmarking for IMU-based Multimodal LLMs.

## Models

### Primary: OneLLM-7B (Trained on Ego4D)

| Component | Details |
|-----------|---------|
| IMU Tokenizer | 1D Conv layer |
| Universal Encoder | CLIP-ViT (shared) |
| UPM | Dynamic routing |
| LLM | LLaMA2-7B |
| Training Data | **Ego4D IMU** (2K samples) |
| Tokenizer | LLaMA2 (compatible with EventGPT) |

```bash
# Benchmark OneLLM
python benchmark_onellm_5stages.py --max_samples 100
```

### Alternative: LLaSA-7B

| Component | Size | Notes |
|-----------|------|-------|
| LIMU-BERT encoder | 62K params | Hidden size: 72 |
| MLP projector | 2-layer | 72 → 4096 |
| Vicuna-7B decoder | 7B params | LLaMA tokenizer |
| IMU tokens | ~120 | vs ~577 for CLIP images |

## 5-Stage Pipeline

```
Stage 1: Load IMU data from disk
Stage 2: Preprocess (normalization)
Stage 3: LIMU-BERT encoding (62K params - very fast)
Stage 4: LLM Prefill (build KV cache)
Stage 5: LLM Decode (autoregressive)
```

## Usage

```bash
python benchmark_inference_5stages.py --max_samples 100
```

---

## Dataset Analysis

### IMU-Only Datasets (LLaSA Training Data)

| Dataset | Pairs | Content | Video | Download |
|---------|-------|---------|-------|----------|
| **SensorCaps** | 35,960 | IMU + captions | No | `huggingface-cli download BASH-Lab/SensorCaps` |
| **OpenSQA** | 179,727 | IMU + Q&A | No | `huggingface-cli download BASH-Lab/OpenSQA` |

**Source HAR datasets:** HHAR, UCI-HAR, MotionSense, Shoaib

### IMU + Video Paired Datasets

| Dataset | IMU | Video | Text | Size | Access |
|---------|-----|-------|------|------|--------|
| **Ego4D** | ✓ 200Hz | ✓ egocentric | ✓ dense narrations | 3,025h | License required |
| **Ego-Exo4D** | ✓ dual (800Hz, 1000Hz) | ✓ ego + exo | ✓ narrations | 800+ participants | License required |
| **VIDIMU** | ✓ 5 sensors | ✓ commodity camera | labels only | 54 subjects | Public |
| **MuJo** | synthetic (IMUTube) | ✓ | ✓ GPT-generated | Large | Check paper |
| **IMHD** | ✓ preprocessed | ✓ 32 views | ✓ | CVPR 2024 | HuggingFace |
| **EgoVid-5M** | ✓ poses | ✓ egocentric | ✓ descriptions | 5M clips | GitHub |

### Recommendations

#### For Benchmarking LLaSA
- **SensorCaps + OpenSQA**: Direct compatibility, IMU format matches training
- Download: `python download_datasets.py --dataset sensorcaps --output_dir /mnt/hdd/data`

#### For IMU-Video Alignment Research
1. **Ego4D/MMG-Ego4D** (Best): Dense narrations, large scale
   - Register at https://ego4d-data.org/
   - Use ego4d CLI: `ego4d --datasets full_scale --modalities imu video`

2. **IMHD** (Easiest): Ready on HuggingFace
   - `huggingface-cli download AfterJourney00/IMHD-Dataset --repo-type dataset`

3. **VIDIMU** (Smallest): Good for quick experiments
   - 13 daily activities, 54 subjects

### Dataset Comparison for Speculative Decoding

For EventGPT → LLaSA speculative decoding, need:
- Paired event camera + IMU data (rare)
- Or: Video + IMU (use video frames as proxy for events)

| Approach | Dataset | Feasibility |
|----------|---------|-------------|
| Direct pairing | DSEC (events only) | Need to add IMU |
| Video proxy | Ego4D (video + IMU) | Convert video → synthetic events |
| Synthetic | Any video | Generate events + IMU |

---

## Download Commands

```bash
# OneLLM-7B model (Ego4D trained) - RECOMMENDED
huggingface-cli download csuhan/OneLLM-7B --local-dir /mnt/hdd/data/OneLLM-7B

# LLaSA model (~13.5GB)
huggingface-cli download BASH-Lab/LLaSA-7B --local-dir /mnt/hdd/data/LLaSA-7B

# SensorCaps dataset
huggingface-cli download BASH-Lab/SensorCaps --repo-type dataset --local-dir /mnt/hdd/data/SensorCaps

# OpenSQA dataset
huggingface-cli download BASH-Lab/OpenSQA --repo-type dataset --local-dir /mnt/hdd/data/OpenSQA

# IMHD dataset (IMU + Video)
huggingface-cli download AfterJourney00/IMHD-Dataset --repo-type dataset --local-dir /mnt/hdd/data/IMHD

# Ego4D (requires license)
pip install ego4d -i https://mirrors.aliyun.com/pypi/simple/
ego4d --output_directory=/mnt/hdd/data/ego4d --datasets full_scale --modalities imu
```

## Benchmark Scripts

| Script | Model | Dataset |
|--------|-------|---------|
| `benchmark_onellm_5stages.py` | OneLLM-7B | Ego4D IMU / Synthetic |
| `benchmark_inference_5stages.py` | LLaSA-7B | SensorCaps / Synthetic |

## Model Comparison for Speculative Decoding

| Model | Encoder | Decoder | Ego4D Trained | LLaMA Tokenizer |
|-------|---------|---------|---------------|-----------------|
| **OneLLM** | 1D Conv + ViT | LLaMA2-7B | ✓ | ✓ |
| **LLaSA** | LIMU-BERT (62K) | Vicuna-7B | ✗ | ✓ |
| **IMU2CLIP** | CNN+RNN | - | ✓ | - |

**Recommendation:** Use **OneLLM** for speculative decoding with EventGPT:
- Both use LLaMA-compatible tokenizers
- OneLLM trained on Ego4D (real IMU data)
- Shared encoder architecture potential

## References

- [OneLLM Paper (CVPR 2024)](https://arxiv.org/abs/2312.03700)
- [OneLLM GitHub](https://github.com/csuhan/OneLLM)
- [IMU2CLIP](https://github.com/facebookresearch/imu2clip)
- [LLaSA Paper](https://arxiv.org/abs/2406.14498)
- [LLaSA GitHub](https://github.com/BASHLab/LLaSA)
- [LIMU-BERT](https://github.com/dapowan/LIMU-BERT-Public)
- [Ego4D](https://ego4d-data.org/)
- [IMHD Dataset](https://github.com/AfterJourney00/IMHD-Dataset)
