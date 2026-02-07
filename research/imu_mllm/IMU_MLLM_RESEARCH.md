# IMU-Based MLLMs for Action Description

## Goal
Research IMU-based Multimodal Large Language Models (MLLMs) as potential **draft models** to substitute EventGPT in speculative decoding pipelines. Key requirements:
- Lighter encoder than EventGPT's event camera encoder
- Ability to describe sequences of actions
- Compatible with speculative decoding (fast inference)
- **Tokenizer compatible with EventGPT and LLaMA**

---

## 0. EventGPT Architecture Reference (Target Model)

**Source:** `model/EventChatModel.py`

```python
# EventGPT uses LLaMA-based architecture
from transformers import LlamaForCausalLM, LlamaConfig, LlamaModel
from transformers import LlamaTokenizer, AutoTokenizer

class EventChatModel(LlamaForCausalLM):
    # Decoder: LlamaForCausalLM
    # Hidden size: 4096
    # Text hidden size: 1024 (CLIP encoder output)
    # Visual encoder: CLIP (CLIPVisionModel)
    # Projector: 2-layer MLP (1024 → 4096)
```

**Key Specs:**
| Component | Specification |
|-----------|--------------|
| **Decoder** | LlamaForCausalLM |
| **Tokenizer** | LlamaTokenizer (SentencePiece BPE) |
| **Hidden size** | 4096 |
| **Vocab size** | ~32000 (LLaMA base) |
| **Vision encoder** | CLIP ViT (1024-dim output) |
| **Context length** | 2048 tokens |

**Tokenizer Compatibility Requirement:**
For speculative decoding to work, the draft model MUST use the same tokenizer as the target model (EventGPT/VideoLLaVA). Both use **LLaMA tokenizer**.

---

## 1. Complete IMU-Based MLLM Models (with Decoder)

### 1.1 PandaGPT (2023)
**Paper:** [PandaGPT: One Model To Instruction-Follow Them All](https://ar5iv.labs.arxiv.org/html/2305.16355)

**Architecture:**
- Combines **ImageBind** (multimodal encoder) + **Vicuna** (LLM)
- Supports 6 modalities: text, image/video, audio, depth, thermal, **IMU**
- **Lightweight design**: Only trains a linear projection matrix + LoRA weights
- Training: 8×A100 40G GPUs for 7 hours (only image-language data)

**IMU Support:**
- Zero-shot cross-modal capability via ImageBind's shared embedding space
- No explicit IMU training required (emergent capability)

**Pros for Draft Model:**
- Lightweight projection layer
- Already supports IMU modality
- Fast inference potential

**Cons:**
- IMU is zero-shot only (not fine-tuned)
- Relies on heavy ImageBind encoder (630M params for ViT-H)

---

### 1.2 OneLLM (CVPR 2024) ⭐ COMPATIBLE WITH EVENTGPT
**Paper:** [OneLLM: One Framework to Align All Modalities](https://arxiv.org/abs/2312.03700)
**GitHub:** https://github.com/csuhan/OneLLM
**HuggingFace:** https://huggingface.co/csuhan/OneLLM-7B

**Complete Architecture:**
```
IMU Data (or other modality)
    ↓
Modality Tokenizer (1D Conv for IMU)
    - Single convolution layer per modality
    ↓
Universal Encoder (pretrained CLIP-ViT)
    ↓
Universal Projection Module (UPM)
    - Dynamic routing between modality projectors
    - Multiple image projection modules combined
    ↓
LLaMA2-7B Decoder  ← LLaMA2-based!
    - Uses LLaMA2 tokenizer
    ↓
Text Output
```

**Decoder:** **LLaMA2-7B**
**Tokenizer:** **LLaMA2 tokenizer** (`config/llama2/tokenizer.model`) ✓ COMPATIBLE

**Key Numbers:**
| Component | Specification |
|-----------|--------------|
| IMU tokenizer | 1D Conv layer |
| Universal encoder | CLIP-ViT |
| UPM | Dynamic routing |
| LLM | LLaMA2-7B |
| Modalities | 8 (image, audio, video, point, depth, normal, IMU, fMRI) |
| Training data | 2M multimodal instruction items |

**Progressive Training:**
- Stage I: Image-text pretraining
- Stage II: Video-audio-point-text pretraining
- Stage III: **Depth-normal-IMU-fMRI-text pretraining**

**Pretrained Model:**
- Preview 7B: https://huggingface.co/csuhan/OneLLM-7B

**Why Good for EventGPT:**
1. ✅ Uses LLaMA2 tokenizer (compatible with LLaMA)
2. ✅ Explicitly trained on IMU modality
3. ✅ Unified encoder reduces overhead
4. ✅ Pretrained weights available
5. ⚠️ Full 7B model (may need distillation for draft)

---

### 1.3 SensorLLM (EMNLP 2025)
**Paper:** [SensorLLM: Aligning Large Language Models with Motion Sensors](https://aclanthology.org/2025.emnlp-main.19.pdf)

**Architecture:**
- Two-stage framework:
  1. **Pretrained LLM**
  2. **Pretrained time-series (TS) embedder**
  3. **Lightweight MLP** (projection)

**Key Features:**
- Aligns wearable sensor data with descriptive text
- High-precision Q&A pairs created **without human annotations**
- Interprets and reasons over time-series signals

**Pros for Draft Model:**
- **Lightweight MLP** connector
- Designed for wearable sensor data
- Language grounding for action description

---

### 1.4 LLaSA (2024) ⭐ BEST MATCH FOR EVENTGPT
**Paper:** [LLaSA: Large Language and Sensor Assistant](https://arxiv.org/html/2406.14498)
**GitHub:** https://github.com/BASHLab/LLaSA
**HuggingFace:** https://huggingface.co/BASH-Lab/

**Complete Architecture:**
```
Raw IMU Data (6-channel: accel + gyro)
    ↓
LIMU-BERT Encoder (62K params, hidden_size=72)
    - Multi-head attention encoder-decoder
    - GELU activation
    - Pretrained on HHAR, UCI-HAR, MotionSense, Shoaib
    ↓
2-layer MLP Projector (72 → LLM hidden)
    ↓
Vicuna 1.5 LLM (7B or 13B)  ← LLaMA2-based!
    - Uses LLaMA tokenizer
    - LoRA fine-tuning
    ↓
Text Output
```

**Decoder:** Vicuna-7B-1.5 / Vicuna-13B-1.5-16K (LLaMA2-based)
**Tokenizer:** **LLaMA tokenizer** ✓ COMPATIBLE WITH EVENTGPT

**Key Numbers:**
| Component | Size |
|-----------|------|
| LIMU-BERT encoder | **62K params** |
| MLP projector | ~2 layers |
| Full model (7B) | 7B params |
| Full model (13B) | 13B params |
| IMU tokens | **194 tokens** (vs 2175 raw) |
| Context | 2048 tokens |

**Training:**
- LoRA applied to Vicuna
- 7B: 1× A100 GPU
- 13B: 2× A100 GPUs
- Follows LLaVA instruction-tuning

**Datasets Created:**
- **SensorCaps**: 35,960 IMU-caption pairs
- **OpenSQA**: 179,727 Q&A pairs for sensor reasoning

**Why Best for EventGPT:**
1. ✅ Uses LLaMA tokenizer (same as EventGPT)
2. ✅ Extremely lightweight encoder (62K vs CLIP's millions)
3. ✅ Direct IMU → text generation
4. ✅ Pretrained weights available
5. ✅ LLaVA-style architecture (similar to VideoLLaVA)

---

### 1.5 IMU2CLIP (EMNLP 2023)
**Paper:** [IMU2CLIP: Multimodal Contrastive Learning for IMU Motion Sensors](https://arxiv.org/abs/2210.14395)

**Architecture:**
- Translates IMU motions → textual descriptions and videos
- Uses CLIP embedding space for alignment
- Enables **motion-based media search** and **LM-based multimodal reasoning**

**Key Applications:**
- Motion-based media search
- LLM reasoning with motion sensor data
- Text as grounding platform

**Pros for Draft Model:**
- Directly produces text from IMU
- CLIP alignment enables cross-modal transfer
- Lightweight compared to full MLLMs

---

### 1.6 MobiDiary (arXiv 2025)
**Paper:** [MobiDiary: Autoregressive Action Captioning with Wearable Devices](https://arxiv.org/html/2601.08204)

**Architecture:**
- **Unified sensor encoder** with local temporal semantic representations
- Text Encoder + Cross-attention fusion
- Language Generation Network

**Key Features:**
- Identifies action boundaries
- Captures discriminative motion patterns
- Designed for **wearable devices** (edge deployment)

**Pros for Draft Model:**
- **Edge-optimized design**
- Action boundary detection (useful for temporal reasoning)
- Autoregressive captioning

---

### 1.7 PRIMUS (NeurIPS 2024 Workshop / ICASSP 2025)
**Paper:** [PRIMUS: Pretraining IMU Encoders with Multimodal Self-Supervision](https://arxiv.org/abs/2411.15127)
**GitHub:** https://github.com/Nokia-Bell-Labs/pretrained-imu-encoders

**Architecture:**
- **IMU Encoder: ~1.4M parameters** (extremely lightweight!)
  - Stacked RNN: Conv + GroupNorm + MaxPool + GRU
  - Based on IMU2CLIP architecture
- Two MLP heads for multimodal and unimodal loss

**Key Features:**
- Combines self-supervision + multimodal + nearest-neighbor supervision
- **Up to 15% accuracy improvement** with <500 labeled samples
- Designed for **mobile and wearable deployment**

**Pros for Draft Model:**
- **Only 1.4M parameters** - excellent for draft model!
- Efficient deployment on mobile/wearable
- State-of-the-art pretrained encoder available
- Checkpoint: https://zenodo.org/records/15147513

**Note:** Encoder only - needs to be paired with LLM decoder

---

### 1.8 Mojito (arXiv 2025) ⚠️ NOT COMPATIBLE
**Paper:** [Mojito: LLM-Aided Motion Instructor with Jitter-Reduced Inertial Tokens](https://arxiv.org/abs/2502.16175)
**GitHub:** https://github.com/koyui/mojito
**Project:** https://koyui.github.io/mojito/

**Complete Architecture:**
```
Jittery IMU Signals
    ↓
IMU Tokenizer (VQ-VAE based)
    - Distribution matching (Zipf's law alignment)
    - Jitter reduction through quantization
    ↓
Motion Decoder (reconstruct motion)
    ↓
Projection Module → Language semantic space
    ↓
Qwen2 LLM  ← NOT LLaMA!
    - LoRA adapters for domain adaptation
    ↓
Text Output
```

**Decoder:** **Qwen2** ❌ NOT COMPATIBLE WITH EVENTGPT
**Tokenizer:** Qwen2 tokenizer ❌ Different from LLaMA

**Key Features:**
- Novel jitter-reduced inertial tokens
- VQ-VAE for discrete IMU representation
- Zipf's law distribution matching
- Real-time motion capture under noisy conditions

**Checkpoint:** `checkpoints/mojito_imu_tokenizer.pth` at HuggingFace `Cunkaixin/mojito`

**Why NOT for EventGPT:**
- ❌ Uses Qwen2 tokenizer (incompatible with LLaMA)
- ❌ Different vocabulary and token IDs
- ✅ Good IMU tokenizer design (could be adapted)

---

### 1.9 MotionLLaMA (2024) - MoCap Focus
**Paper:** [MotionLLaMA: A Unified Framework for Motion Synthesis and Comprehension](https://arxiv.org/html/2411.17335v1)
**GitHub:** https://github.com/ZeyuLing/MotionLLaMA

**Complete Architecture:**
```
Motion Data (SMPL-H skeletal)
    ↓
HoMi Tokenizer (Holistic Motion)
    - VQ-VAE with single codebook
    - FFT for temporal frequency
    - Body + hand positions separately
    ↓
Motion Vocabulary (extended lexicon)
    ↓
LLaMA3 + LoRA  ← LLaMA3-based
    - LLaMA3 tokenizer for text
    - Motion tokens as extended vocabulary
    ↓
Text/Motion Output
```

**Decoder:** **LLaMA3** (LLaMA3.2-Instruct)
**Tokenizer:** LLaMA3 tokenizer ⚠️ Partially compatible

**Key Features:**
- Supports 9 motion-related tasks
- Single-person and dual-person motions
- Outperforms 6-codebook tokenizers

**Why Partially Suitable:**
- ⚠️ LLaMA3 tokenizer (different vocab from LLaMA1/2)
- ❌ MoCap data, not raw IMU
- ✅ Good motion tokenizer design

---

## 2. Complete Model Comparison

### 2.1 Tokenizer Compatibility with EventGPT

| Model | Decoder | Tokenizer | Compatible? | Notes |
|-------|---------|-----------|-------------|-------|
| **LLaSA** | Vicuna 1.5 (LLaMA2) | LLaMA | ✅ **YES** | Same tokenizer family |
| **OneLLM** | LLaMA2-7B | LLaMA2 | ✅ **YES** | Same tokenizer |
| **PandaGPT** | Vicuna | LLaMA | ✅ **YES** | Same tokenizer |
| Mojito | Qwen2 | Qwen2 | ❌ NO | Different tokenizer |
| MotionLLaMA | LLaMA3 | LLaMA3 | ⚠️ Partial | Different vocab size |
| SensorLLM | Various | Depends | ⚠️ Check | Implementation varies |

### 2.2 Encoder Size Comparison

| Model | IMU Encoder Size | Full Model Size | Decoder |
|-------|-----------------|-----------------|---------|
| **LLaSA (LIMU-BERT)** | **62K params** | 7B/13B | Vicuna |
| **PRIMUS** | **1.4M params** | 1.4M (encoder only) | None |
| IMU2CLIP | ~1.4M | Requires CLIP | None |
| OneLLM | 1D Conv | 7B+ | LLaMA2 |
| ImageBind IMU | CNN+Transformer | 630M (ViT-H) | None |
| Mojito | VQ-VAE | Unknown | Qwen2 |

### 2.3 Recommendation for Draft Model

**Primary Recommendation: LLaSA-7B**
```
✅ LLaMA tokenizer compatible
✅ Extremely lightweight encoder (62K params)
✅ Complete MLLM (encoder + decoder)
✅ Pretrained weights available
✅ LLaVA-style architecture
```

**Alternative: OneLLM-7B**
```
✅ LLaMA2 tokenizer compatible
✅ Explicitly trained on IMU
✅ Pretrained weights available
⚠️ Full 7B model (needs distillation)
```

**Encoder-Only Option: PRIMUS + Custom LLM**
```
✅ Only 1.4M encoder params
✅ State-of-the-art pretrained
⚠️ Need to add LLaMA decoder
⚠️ Need alignment training
```

---

## 3. IMU-Video Paired Datasets

### 3.1 Ego4D / MMG-Ego4D
**Website:** https://ego4d-data.org/

**Statistics:**
- 3,025 hours of egocentric video
- 855 camera wearers, 74 locations, 9 countries
- **Modalities**: Video, Audio, IMU, Gaze, 3D scans

**Annotations:**
- Dense timestamped narrations in English
- Largest aligned language-video repository

**MMG-Ego4D Subset:**
- Specifically constructed for **Multimodal Generalization**
- Contains **video, audio, and IMU** modalities
- Studies generalization when modalities are missing

---

### 3.2 Ego-Exo4D
**Website:** https://ego-exo4d-data.org/

**Statistics:**
- 800+ participants, 13 cities
- 131 different scene contexts
- Synchronized egocentric + exocentric video

**IMU Data:**
- **Two IMUs**: 800 Hz and 1000 Hz
- Barometer: 50 fps
- Magnetometer: 10 fps

**Annotations:**
- First-person narrations by camera wearers
- Time-indexed video-language resources

---

### 3.3 VIDIMU
**Paper:** [Multimodal video and IMU kinematic dataset](https://www.nature.com/articles/s41597-023-02554-9)

**Statistics:**
- 13 daily activities
- 54 subjects (16 with simultaneous IMU)
- Commodity camera + 5 inertial sensors

**Focus:**
- Remote daily life activity recognition
- Kinematic analysis
- Affordable patient tracking

---

### 3.4 MuJo Dataset
**Paper:** [MuJo: Multimodal Joint Feature Space Learning](https://arxiv.org/html/2406.03857v1)

**Modalities:**
- Video
- Video-derived poses
- Synthetic IMU (via IMUTube)
- **Textual descriptions** (labels, captions, GPT-3.5 generated)

**Key Feature:**
- Large fitness dataset with parallel modalities
- Explicit language descriptions

---

### 3.5 ActionNet
**Paper:** [ActionNet: A Multimodal Dataset](https://cdfg.mit.edu/assets/images/actionnet.pdf)

**Focus:**
- Human activities using wearable sensors
- Multimodal sensor fusion

---

### 3.6 MMAct Dataset
**Modalities:**
- External cameras
- First-person video
- Watch IMU
- Smartphone: IMU + barometer + WiFi

**Activities:**
- Daily activities with multimodal capture

---

### 3.7 MotionGaze Dataset
**Statistics:**
- 10 hours of data
- 481K+ images
- 800K+ IMU readings

**Postures:**
- Sitting, standing, lying, walking
- Various motion conditions

---

### 3.8 EgoVid-5M
**GitHub:** https://github.com/JeffWang987/EgoVid

**Statistics:**
- 5 million egocentric video clips
- Fine-grained kinematic control
- High-level text descriptions

**IMU Data:**
- IMU poses and camera extrinsics
- SfM poses and Kalman Filter fused poses

---

## 4. Dataset Comparison for Scene Description

| Dataset | IMU | Video | Text/Narration | Scene Description | Size |
|---------|-----|-------|----------------|-------------------|------|
| **Ego4D** | ✓ | ✓ | ✓ Dense narrations | ✓ | 3,025h |
| **Ego-Exo4D** | ✓ (2 IMUs) | ✓ | ✓ First-person narrations | ✓ | 800+ participants |
| **MuJo** | ✓ (synthetic) | ✓ | ✓ GPT-generated | Fitness focus | Large |
| VIDIMU | ✓ | ✓ | Activity labels | Limited | 54 subjects |
| MMAct | ✓ | ✓ | Activity labels | Limited | - |
| EgoVid-5M | ✓ | ✓ | ✓ Text descriptions | ✓ | 5M clips |

**Recommendation:**
1. **Ego4D / MMG-Ego4D** - Best for scene description with dense narrations
2. **Ego-Exo4D** - High-quality IMU (dual IMUs, high Hz)
3. **EgoVid-5M** - Large scale with text descriptions

---

## 5. Proposed Architecture for Draft Model (LLaMA-Compatible)

### Option A: LLaSA as Draft Model ⭐ RECOMMENDED
```
IMU Data (6-channel, sampled)
    ↓
LIMU-BERT Encoder (62K params)
    - Multi-head attention
    - Pretrained on HAR datasets
    ↓
2-layer MLP Projector
    ↓
Vicuna-7B (LLaMA tokenizer)  ← Same tokenizer as EventGPT!
    ↓
Draft Tokens
    ↓
EventGPT/VideoLLaVA Verification
```

**Pros:**
- ✅ Same tokenizer = direct token verification
- ✅ Complete pretrained model available
- ✅ Lightweight encoder (62K)
- ✅ Already generates action descriptions

**Cons:**
- 7B decoder still large
- May need to quantize or distill

---

### Option B: LIMU-BERT + TinyLLaMA (Custom Build)
```
IMU Data
    ↓
LIMU-BERT Encoder (62K)
    - Download from github.com/dapowan/LIMU-BERT-Public
    ↓
Custom MLP Projection (72 → 2048)
    ↓
TinyLLaMA-1.1B (LLaMA tokenizer)
    - HuggingFace: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    ↓
Draft Tokens
```

**Pros:**
- ✅ Only 1.1B decoder (much faster)
- ✅ Same LLaMA tokenizer
- ✅ LIMU-BERT is proven

**Cons:**
- Need to train projection layer
- Need action description fine-tuning

---

### Option C: PRIMUS + LLaMA-based LLM
```
IMU Data (400Hz, 6-channel)
    ↓
PRIMUS Encoder (1.4M params)
    - 1D Conv + GroupNorm + MaxPool + GRU
    - Checkpoint: zenodo.org/records/15147513
    ↓
Projection Layer (train this)
    ↓
TinyLLaMA-1.1B or LLaMA-2-7B
    ↓
Draft Tokens
```

**Pros:**
- ✅ State-of-the-art IMU encoder
- ✅ Very lightweight (1.4M)

**Cons:**
- Need to train full pipeline
- No pretrained MLLM

---

### Option D: OneLLM (If 7B is acceptable)
```
IMU Data
    ↓
OneLLM 1D Conv Tokenizer
    ↓
Universal Encoder (CLIP-ViT)
    ↓
UPM (Universal Projection Module)
    ↓
LLaMA2-7B  ← Same tokenizer family!
    ↓
Draft Tokens
```

**Pros:**
- ✅ Complete pretrained model
- ✅ Explicitly trained on IMU
- ✅ LLaMA2 tokenizer compatible

**Cons:**
- Full 7B model
- May be slow for draft model

---

### Architecture Comparison for Speculative Decoding

| Option | Encoder | Decoder | Tokenizer Match | Ready to Use |
|--------|---------|---------|-----------------|--------------|
| **A: LLaSA** | 62K | 7B | ✅ LLaMA | ✅ Yes |
| **B: LIMU-BERT+TinyLLaMA** | 62K | 1.1B | ✅ LLaMA | ⚠️ Need training |
| **C: PRIMUS+LLaMA** | 1.4M | 1.1B-7B | ✅ LLaMA | ⚠️ Need training |
| **D: OneLLM** | 1D Conv | 7B | ✅ LLaMA2 | ✅ Yes |

---

## 6. Speculative Decoding Compatibility

### Requirements for IMU Draft Model:
1. **Fast prefill**: IMU encoder should be faster than event camera encoder
2. **Token alignment**: IMU embeddings should align with target model's embedding space
3. **Acceptance rate**: Need sufficient semantic overlap for good acceptance

### Advantages of IMU over Event Camera:
- Simpler sensor (6 channels vs. event stream)
- More temporally structured data
- Established pretrained encoders (PRIMUS, IMU2CLIP)
- Lower computational cost

### Challenges:
- Different modality semantics (motion vs. vision)
- May have lower acceptance rate initially
- Need paired event-IMU data for alignment training

---

## 7. Next Steps

1. **Evaluate PRIMUS encoder** on action description task
2. **Download Ego4D/MMG-Ego4D** for training data
3. **Prototype draft model** with PRIMUS + TinyLLaMA
4. **Measure acceptance rate** vs. EventGPT/VideoLLaVA
5. **Fine-tune alignment** if acceptance rate is low

---

## 8. Key References

### LLaMA-Compatible IMU MLLMs (Recommended):
- **[LLaSA](https://github.com/BASHLab/LLaSA)** - Best match for EventGPT
  - Paper: https://arxiv.org/abs/2406.14498
  - HuggingFace: https://huggingface.co/BASH-Lab/
- **[OneLLM (CVPR 2024)](https://github.com/csuhan/OneLLM)**
  - Paper: https://arxiv.org/abs/2312.03700
  - HuggingFace: https://huggingface.co/csuhan/OneLLM-7B
- **[LIMU-BERT](https://github.com/dapowan/LIMU-BERT-Public)** - Lightweight IMU encoder
  - Paper: https://tanrui.github.io/pub/LIMU_BERT.pdf

### IMU Encoders (Need LLM pairing):
- **[PRIMUS (NeurIPS 2024)](https://github.com/Nokia-Bell-Labs/pretrained-imu-encoders)**
  - Paper: https://arxiv.org/abs/2411.15127
  - Checkpoint: https://zenodo.org/records/15147513
- [IMU2CLIP (EMNLP 2023)](https://arxiv.org/abs/2210.14395)
- [ImageBind](https://arxiv.org/abs/2305.05665)

### Other IMU MLLMs (Not LLaMA-compatible):
- [Mojito (Qwen2)](https://github.com/koyui/mojito) - Good IMU tokenizer design
- [MotionLLaMA (LLaMA3)](https://github.com/ZeyuLing/MotionLLaMA) - MoCap focus

### Other Models:
- [PandaGPT](https://panda-gpt.github.io/)
- [SensorLLM (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.19.pdf)
- [MobiDiary (2025)](https://arxiv.org/html/2601.08204)

### LLaMA-based LLMs for Custom Build:
- [TinyLLaMA-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- [LLaMA-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)

### Datasets:
- [Ego4D](https://ego4d-data.org/) - IMU + Video + Narrations
- [Ego-Exo4D](https://ego-exo4d-data.org/) - Dual IMUs, high Hz
- [VIDIMU](https://www.nature.com/articles/s41597-023-02554-9)
- [MuJo](https://arxiv.org/html/2406.03857v1)
- [EgoVid-5M](https://github.com/JeffWang987/EgoVid)
- [SensorCaps](https://github.com/BASHLab/LLaSA) - 35,960 IMU-caption pairs
- [OpenSQA](https://github.com/BASHLab/LLaSA) - 179,727 IMU Q&A pairs
- [Awesome-IMU-Sensing (GitHub)](https://github.com/rh20624/Awesome-IMU-Sensing)

### Surveys:
- [Efficient MLLMs Survey](https://link.springer.com/article/10.1007/s44267-025-00099-6)
- [Omni-MLLM Survey](https://arxiv.org/html/2412.11694v1)
- [LLMs for Wearable HAR Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC11314694/)

---

## 9. Quick Start Guide

### To use LLaSA (Recommended):
```bash
# Clone repository
git clone https://github.com/BASHLab/LLaSA.git
cd LLaSA

# Download model from HuggingFace
# Models: BASH-Lab/LLaSA-7B, BASH-Lab/LLaSA-13B

# The tokenizer is Vicuna/LLaMA compatible
```

### To use OneLLM:
```bash
# Clone repository
git clone https://github.com/csuhan/OneLLM.git
cd OneLLM

# Download model
# HuggingFace: csuhan/OneLLM-7B

# Tokenizer: config/llama2/tokenizer.model
```

### To build custom (LIMU-BERT + TinyLLaMA):
```bash
# Get LIMU-BERT encoder
git clone https://github.com/dapowan/LIMU-BERT-Public.git

# Get TinyLLaMA
pip install transformers
# Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Train projection layer on Ego4D/SensorCaps data
```
