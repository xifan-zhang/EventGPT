# Stage 4 (LLM Decoding) Analysis: EventGPT vs Video-LLaVA

## Overview

Stage 4 is the LLM decoding phase where tokens are generated autoregressively. This stage dominates inference time (>95% for both models).

## Performance Comparison

| Metric | EventGPT | Video-LLaVA | Ratio |
|--------|----------|-------------|-------|
| **Stage 4 Time** | ~1.0s | ~1.9s | 1.9x faster |
| **Stage 4 Percentage** | 96.7% | 99.1% | - |
| **Avg Tokens Generated** | 44.8 | 90.7 | 2x more |
| **Tokens/Second** | ~45 | ~47 | Similar |

## Input Format Analysis

### EventGPT Stage 4 Inputs

```python
# Input IDs (prompt tokens)
input_ids: torch.Tensor  # Shape: [1, ~50-100]
                         # dtype: torch.long
                         # Contains: text tokens + EVENT_TOKEN placeholders

# Pre-computed vision features (from Stage 3)
event_features: torch.Tensor  # Shape: [1, num_patches, hidden_dim]
                              # dtype: torch.bfloat16
                              # Contains: Encoded event image features

# Image size for positional encoding
event_image_sizes: List[int]  # [height, width]

# Generation parameters
do_sample: bool = False
max_new_tokens: int = 100
use_cache: bool = True
```

### Video-LLaVA Stage 4 Inputs

```python
# Input IDs (prompt tokens with image placeholders)
input_ids: torch.Tensor  # Shape: [1, ~600-700]
                         # dtype: torch.long
                         # Contains: text tokens + expanded image tokens (576 per image)

# Pixel values (raw image data - processed during forward pass)
pixel_values: torch.Tensor  # Shape: [1, 3, 336, 336]
                            # dtype: torch.float16
                            # Contains: Preprocessed RGB image

# Attention mask
attention_mask: torch.Tensor  # Shape: [1, seq_len]

# Generation parameters
max_new_tokens: int = 100
do_sample: bool = False
```

## Output Format Analysis

### EventGPT Stage 4 Outputs

```python
# Generated token IDs
output_ids: torch.Tensor  # Shape: [1, input_len + generated_len]
                          # dtype: torch.long

# Typical output: ~45 tokens
# Format: Concise scene descriptions
```

### Video-LLaVA Stage 4 Outputs

```python
# Generated token IDs
output_ids: torch.Tensor  # Shape: [1, input_len + generated_len]
                          # dtype: torch.long

# Typical output: ~90 tokens
# Format: More verbose, detailed descriptions
```

## Why EventGPT is Faster

### 1. Feature Caching (Primary Reason)

**EventGPT:**
- Vision features pre-computed in Stage 3: `event_features = model.visval_encode(event_tensor)`
- Stage 4 receives cached features directly
- No vision tower forward pass during generation
- **Savings: ~100-200ms per sample**

**Video-LLaVA:**
- `pixel_values` passed to `generate()`
- Vision tower runs during first forward pass
- Image tokens expanded (1 image → 576 tokens)
- Extra computation during generation

### 2. Context Length

| Model | Input Tokens | Image Tokens | Total Context |
|-------|--------------|--------------|---------------|
| EventGPT | ~50 | ~256 (patches) | ~306 |
| Video-LLaVA | ~50 | ~576 (expanded) | ~626 |

- EventGPT has **~50% shorter context**
- Shorter context = faster attention computation
- KV-cache updates are smaller

### 3. Data Type

| Model | Dtype | Memory | Speed |
|-------|-------|--------|-------|
| EventGPT | bfloat16 | Lower | Faster |
| Video-LLaVA | float16 | Higher | Slower |

- bfloat16 has same range as float32 but less precision
- Better optimized on modern GPUs (A100, RTX 3090/4090)

### 4. Output Length

- EventGPT: ~45 tokens average (concise)
- Video-LLaVA: ~90 tokens average (verbose)
- **2x fewer tokens to generate**

### 5. Architecture Differences

**EventGPT:**
```
Event Image → CLIP Encoder → Event Projector → LLM (LLaMA)
              (Stage 3)      (Stage 3)         (Stage 4)
```

**Video-LLaVA:**
```
Image → CLIP Vision Tower → Multi-modal Projector → LLM (Vicuna)
        (during generate)   (during generate)        (Stage 4)
```

## Optimization Opportunities

### For EventGPT (Already Fast)
1. **Quantization**: INT8/INT4 for further speedup
2. **Speculative decoding**: With smaller draft model
3. **Batch inference**: Process multiple samples together
4. **Flash Attention**: If not already enabled

### For Video-LLaVA
1. **Feature caching**: Pre-compute vision features like EventGPT
2. **Image token pruning**: Reduce 576 → fewer tokens
3. **Early exit**: Stop generation when confident
4. **Model distillation**: Smaller but equally capable model

## Speculative Decoding Feasibility

### Current Status
- Token acceptance rate: **4.2%** (very low)
- Reason: Different tokenizers, different output styles

### Requirements for Success
1. **Shared tokenizer** or vocabulary alignment
2. **Output style alignment** (similar verbosity)
3. **Feature alignment** module between vision encoders

### Recommendation
Direct speculative decoding not recommended without:
- Training EventGPT to match Video-LLaVA outputs
- Implementing feature alignment adapter
- Using shared decoder heads

## Summary

EventGPT is **1.86x faster** than Video-LLaVA in Stage 4 primarily due to:

1. **Feature caching** - Vision features pre-computed, not re-encoded
2. **Shorter context** - ~50% fewer tokens in attention
3. **Concise outputs** - ~50% fewer tokens generated
4. **bfloat16 dtype** - Better GPU utilization

The architectural advantage of decoupling vision encoding (Stage 3) from LLM decoding (Stage 4) is the key innovation that enables EventGPT's speed advantage.
