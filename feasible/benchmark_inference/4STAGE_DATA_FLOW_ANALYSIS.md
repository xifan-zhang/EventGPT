# 4-Stage Data Flow Analysis for EventGPT

Analysis Date: 2026-01-24 01:01:40
Samples Analyzed: 10

## Overview

EventGPT inference pipeline consists of 4 stages:

| Stage | Operation | Input | Output |
|-------|-----------|-------|--------|
| 1 | Data Loading | Event image paths | Preprocessed event tensors |
| 2 | Tokenization | Text prompt | Token IDs |
| 3 | Vision Encoding | Event tensors | Vision features |
| 4 | LLM Decoding | Tokens + Features | Generated tokens |

## Stage Breakdown

### Stage 1: Data Loading
- **Purpose**: Load event images from disk
- **Input**: Event image file paths (list of strings)
- **Process**:
  - Load PNG images using PIL
  - Convert to numpy arrays
  - Extract image size (H, W)
- **Output**: Event tensors with shape `[T, 3, H, W]` where T=number of images
- **Performance**: ~0.8% of total time (minimal bottleneck)

### Stage 2: Tokenization & Prompt Preparation
- **Purpose**: Prepare text input for LLM
- **Input**: Text query string
- **Process**:
  - Format prompt with special tokens (e.g., <event_start>, <event_end>)
  - Tokenize using LLM tokenizer
  - Create input_ids tensor
- **Output**: `input_ids` tensor with shape `[1, seq_len]`
- **Performance**: ~1.7% of total time (negligible)

### Stage 3: Vision Encoding (DECOUPLED)
- **Purpose**: Extract visual features from event images
- **Input**: Event tensor with shape `[1, T, 3, H, W]`
- **Process**:
  - Pass through CLIP vision tower
  - Extract patch embeddings
  - Project to LLM embedding dimension
- **Output**: Vision features with shape `[1, num_patches, embed_dim]`
- **Key**: Features are **cached** and reused in Stage 4 (no re-encoding)
- **Performance**: ~0.7% of total time (well-optimized)

### Stage 4: LLM Decoding (DECOUPLED)
- **Purpose**: Generate text output using LLM
- **Input**:
  - `input_ids`: Tokenized prompt `[1, seq_len]`
  - `event_features`: Pre-computed from Stage 3 (cached)
  - `event_image_sizes`: Image resolution for layout
- **Process**:
  - Skip vision encoding (use cached features)
  - Prepare multimodal input embeddings
  - Run LLM forward pass token-by-token
  - Sample/decode output tokens
- **Output**: Generated token IDs
- **Performance**: ~96.8% of total time (MAJOR BOTTLENECK)

## Data Flow Details

### Sample 0
- Event images: 5
- Image size: [480, 640]
- Input tokens: 60
- Output tokens: 45
- Vision features shape: torch.Size([1, 577, 4096])
- Output: *In the scene, there is a road with a car driving ahead. The road is bordered by hedges and trees on ...*

### Sample 1
- Event images: 5
- Image size: [480, 640]
- Input tokens: 60
- Output tokens: 36
- Vision features shape: torch.Size([1, 577, 4096])
- Output: *The scene depicts a road with a car driving away from the viewer. The road is lined with trees and h...*

### Sample 2
- Event images: 5
- Image size: [480, 640]
- Input tokens: 60
- Output tokens: 48
- Vision features shape: torch.Size([1, 577, 4096])
- Output: *In the scene, there is a road with a car driving ahead. The road curves to the right and is bordered...*

### Sample 3
- Event images: 5
- Image size: [480, 640]
- Input tokens: 60
- Output tokens: 30
- Vision features shape: torch.Size([1, 577, 4096])
- Output: *In the scene, there is a road with a car driving away. The road is bordered by trees and dense veget...*

### Sample 4
- Event images: 5
- Image size: [480, 640]
- Input tokens: 60
- Output tokens: 56
- Vision features shape: torch.Size([1, 577, 4096])
- Output: *In the scene, there is a road with a car driving away from the viewer. The road is bordered by a gua...*

### Sample 5
- Event images: 5
- Image size: [480, 640]
- Input tokens: 60
- Output tokens: 34
- Vision features shape: torch.Size([1, 577, 4096])
- Output: *In the scene, there is a road with vehicles driving on it. The road is bordered by trees and vegetat...*

### Sample 6
- Event images: 5
- Image size: [480, 640]
- Input tokens: 60
- Output tokens: 55
- Vision features shape: torch.Size([1, 577, 4096])
- Output: *In the scene, there is a vehicle on a road with a curve to the right. The road is bordered by a guar...*

### Sample 7
- Event images: 5
- Image size: [480, 640]
- Input tokens: 60
- Output tokens: 56
- Vision features shape: torch.Size([1, 577, 4096])
- Output: *In the scene, there is a road with a car driving on it. The road is marked with white lines and has ...*

### Sample 8
- Event images: 5
- Image size: [480, 640]
- Input tokens: 60
- Output tokens: 56
- Vision features shape: torch.Size([1, 577, 4096])
- Output: *In the scene, there is a road with a car driving away from the viewer. The road is bordered by a gua...*

### Sample 9
- Event images: 5
- Image size: [480, 640]
- Input tokens: 60
- Output tokens: 41
- Vision features shape: torch.Size([1, 577, 4096])
- Output: *In the scene, there is a car on a road with a guardrail on the right side. The road curves to the le...*

## Key Insights

### 1. Proper Stage 3+4 Decoupling
- Vision features are extracted **once** in Stage 3
- Features are **cached** and passed to generate() in Stage 4
- No redundant re-encoding happens
- This enables accurate timing measurement of each stage

### 2. LLM is the Bottleneck
- Stage 4 (LLM) accounts for **96.8%** of total time
- Stages 1-3 combined are only **3.2%** of time
- LLM is **135x slower** than vision encoding

### 3. Optimization Potential
- Focus optimization efforts on Stage 4 (LLM)
- Stage 3 (Vision) is already well-optimized
- Possible Stage 4 optimizations:
  - Speculative decoding (2-3x speedup)
  - Token pruning (10-20% reduction)
  - Quantization (1.5-2x speedup)
  - Batch inference (2-4x speedup)

### 4. Data Shape Summary
- Input event images: Shape varies (typically `[T, H, W, 3]` for T images)
- After CLIP preprocessing: `[T, 3, 336, 336]` (bfloat16)
- Vision features: `[1, num_patches, 4096]` (after projection)
- Input tokens: `[1, prompt_len]` (typically 10-50 tokens)
- Output tokens: `[1, prompt_len + generated_len]` (up to 512 new tokens)

## Files Modified

- `model/EventChatModel.py`: Added `event_features` parameter to `generate()`
- `model/EventChatModel.py`: Added `visval_encode()` method for vision encoding
- `feasible/benchmark_inference/benchmark_inference_4stages.py`: Unified benchmark

