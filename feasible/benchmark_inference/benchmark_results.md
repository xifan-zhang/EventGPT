# EventGPT vs LLaVA Benchmark Results

**Last Updated:** 2026-01-22

## Overview

Benchmark comparing EventGPT (draft model) against LLaVA 1.5-7B (target model) for event-based video understanding tasks. Tests performed on DSEC dataset.

### Configuration
- **Dataset**: my_egpt_dsec_seq_5s (100 samples)
- **GPU**: 24GB VRAM
- **EventGPT**: Uses 5 preprocessed event images per sequence
- **Target Model**: LLaVA 1.5 (default) or Video-LLaVA (experimental with `--use_video_llava`)
  - **LLaVA 1.5**: `llava-hf/llava-1.5-7b-hf` - 8 frames as 2x4 grid image (recommended)
  - **Video-LLaVA**: `LanguageBind/Video-LLaVA-7B-hf` - native 8-frame video input (experimental)
- **Speculative Decoding**: gamma (γ) = 5
- **Query**: "What are the key elements in this scene?"

### Target Model Options
1. **LLaVA 1.5 (default, recommended)**: Uses 8 frames as 2x4 grid image. Produces good quality outputs.
2. **Video-LLaVA**: Use `--use_video_llava` flag. Requires `transformers>=4.40`.
   - ✅ **Fixed (2026-01-22)**: `model.generate()` bug in transformers 4.44.0 now has workaround
   - Uses custom generation function `custom_video_llava_generate()` instead of `model.generate()`
   - Supports both MP4 files (preferred) and image folders as input
   - See "Bug Fixes" section below for details

### LLaVA 1.5 Video Input Fix (Backup Mode)
LLaVA 1.5 doesn't handle multiple separate images well (produces repetitive garbage output).
**Solution**: Combine 8 video frames into a single 2x4 grid image before processing.

## Latest Benchmark Results (5s Dataset, 100 samples)

### Timing Summary

| Model | Total Time | S1: Load | S2: Preprocess | S3: Generate |
|-------|------------|----------|----------------|--------------|
| EventGPT | 0.986s | 0.008s | 0.027s | 0.950s |
| LLaVA | 3.497s | 1.614s | 0.014s | 1.867s |

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **c (Total Ratio)** | 0.282 | EventGPT_Total / LLaVA_Total |
| **Acceptance Rate (α)** | 3.4% | 139/4146 tokens matched |
| **Theoretical Speedup** | 0.43x | With speculative decoding (γ=5) |

### GPU Memory Usage

| Model | Model Load | Inference Peak |
|-------|------------|----------------|
| EventGPT | 13,576 MB | 14,309 MB |
| LLaVA | 13,536 MB | 14,286 MB |

## Speculative Decoding Analysis

With γ=5 (5 draft tokens per verification step):

- **c = 0.282**: EventGPT is ~3.5x faster than LLaVA (total time)
- **α = 3.4%**: Low acceptance rate due to different output styles
  - EventGPT: "In the scene, there is a road with a car..."
  - LLaVA: "The image features a car driving down a winding road..."

The low acceptance rate is expected because both models describe the same scene but use different vocabulary and sentence structures.

### Theoretical Speedup Formula

```
Speedup = E[accepted] / (c * γ + 1)
where E[accepted] = (1 - α^(γ+1)) / (1 - α)
```

## Multi-Dataset Benchmark Results (Historical)

| Dataset | EventGPT Total | EventGPT Gen | LLaVA Total | LLaVA Gen | c | Speedup |
|---------|----------------|--------------|-------------|-----------|---|---------|
| 500ms | 1.498s | 1.016s | 18.994s | 15.161s | 0.079 | ~13x |
| 1s | 1.584s | 1.001s | 22.294s | 15.127s | 0.071 | ~14x |
| 2s | 1.545s | 0.978s | 21.522s | 14.396s | 0.072 | ~14x |
| 4s | 1.556s | 0.946s | 22.478s | 14.678s | 0.069 | ~14x |
| 5s | 1.586s | 0.935s | 23.202s | 15.369s | 0.068 | ~15x |
| 10s | 1.420s | 0.885s | 24.270s | 15.373s | 0.058 | ~17x |
| 20s | 1.702s | 0.891s | 23.779s | 14.712s | 0.072 | ~14x |

**Note**: Historical results used 8 separate images for LLaVA (before grid fix).

## Input Specifications

| Model | Input Type | Shape/Size | Resolution |
|-------|------------|------------|------------|
| EventGPT | 5 event images | [5, 3, 336, 336] | 640x480 → 336x336 |
| LLaVA | 8 frames grid | [1, 3, 336, 336] | 2880x2160 → 336x336 |

### EventGPT Input Format

**Data Loading:**
```python
# Load preprocessed event images
event_image_paths = ["interlaken_00_a/000000_0.png", ...]  # 5 images
for img_path in event_image_paths:
    img = load_image(os.path.join(dataset_dir, "event_image", img_path))
    img_array = np.array(img)  # shape: [H, W, 3], e.g., [480, 640, 3]
```

**CLIP Preprocessing:**
```python
# event_processor is CLIPImageProcessor
event = event_processor(img_array, return_tensors='pt')['pixel_values'][0]
event = event.to(device, dtype=torch.bfloat16)  # shape: [3, 336, 336]
event_list.append(event)  # List of 5 tensors
```

**Prompt Format:**
```python
conv_mode = 'eventgpt_v1'
prompt = prepare_event_prompt(query, conv_mode)
# Result: "A chat between a curious human and an AI assistant. The AI gives helpful answers.
#          USER: <event>\n{query}\nASSISTANT:"

input_ids = tokenizer_event_token(prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt')
input_ids = input_ids.unsqueeze(0).to(device)  # shape: [1, seq_len]
```

**model.generate():**
```python
output_ids = model.generate(
    input_ids,                    # [1, seq_len] - tokenized prompt
    event_tensors=event_list,     # List of 5 tensors, each [3, 336, 336]
    event_image_sizes=[H, W],     # Original image size [480, 640]
    do_sample=True,               # True if temperature > 0
    temperature=0.6,
    top_p=1.0,
    num_beams=1,
    max_new_tokens=512,
    use_cache=True
)
```

### Video-LLaVA Input Format (Default Target Model)

**Model**: `LanguageBind/Video-LLaVA-7B-hf`

Video-LLaVA is trained on video data and handles 8 frames natively via `<video>` token.

**Data Loading:**
```python
# Load 8 video frames evenly sampled from folder
images = load_images_from_folder(video_path, max_frames=8)  # List of 8 PIL Images
# Convert to numpy array: (num_frames, height, width, 3)
video_frames = np.stack([np.array(img) for img in images])
```

**Prompt Format:**
```python
prompt = f"USER: <video>\n{query}\nASSISTANT:"
# Example: "USER: <video>\nWhat are the key elements in this scene?\nASSISTANT:"
```

**Processor Preprocessing:**
```python
# processor is VideoLlavaProcessor
inputs = processor(
    text=prompt,
    videos=video_frames,      # numpy array (8, H, W, 3)
    return_tensors="pt",
    padding=True
).to(device, torch.float16)

# inputs contains:
#   - input_ids: [1, seq_len]
#   - attention_mask: [1, seq_len]
#   - pixel_values_videos: [1, 8, 3, 224, 224]
```

**model.generate():**
```python
generate_ids = model.generate(
    **inputs,                 # input_ids, attention_mask, pixel_values_videos
    max_new_tokens=512,
    do_sample=False,          # Greedy decoding
    num_beams=1
)
```

### LLaVA 1.5 Input Format (Backup - use with `--use_llava15`)

**Model**: `llava-hf/llava-1.5-7b-hf`

LLaVA 1.5 doesn't handle multiple images well, so we combine 8 frames into a 2x4 grid image.

**Data Loading:**
```python
# Load 8 video frames evenly sampled from folder
images = load_images_from_folder(video_path, max_frames=8)  # List of 8 PIL Images

# Create 2x4 grid image (LLaVA 1.5 doesn't handle multiple images well)
grid_image = create_image_grid(images, cols=4)  # Single PIL Image, size: (4*W, 2*H)
```

**Prompt Format:**
```python
prompt = f"USER: <image>\nThis image shows {len(images)} video frames arranged in a grid. {query}\nASSISTANT:"
# Example: "USER: <image>\nThis image shows 8 video frames arranged in a grid. What are the key elements in this scene?\nASSISTANT:"
```

**Processor Preprocessing:**
```python
# processor is LlavaProcessor (CLIPImageProcessor + tokenizer)
inputs = processor(
    images=[grid_image],      # Single grid image
    text=prompt,
    return_tensors="pt",
    padding=True
).to(device, torch.float16)

# inputs contains:
#   - input_ids: [1, seq_len]
#   - attention_mask: [1, seq_len]
#   - pixel_values: [1, 3, 336, 336]
```

**model.generate():**
```python
generate_ids = model.generate(
    **inputs,                 # input_ids, attention_mask, pixel_values
    max_new_tokens=512,
    do_sample=False,          # Greedy decoding
    repetition_penalty=1.1,   # Prevent repetitive output
    num_beams=1
)
```

### Key Differences

| Aspect | EventGPT | Video-LLaVA | LLaVA 1.5 (backup) |
|--------|----------|-------------|-------------------|
| **Model** | EventGPT-7b | Video-LLaVA-7B-hf | llava-1.5-7b-hf |
| **Dtype** | bfloat16 | float16 | float16 |
| **Image Input** | List of 5 tensors | numpy array (8, H, W, 3) | Single grid image |
| **Special Token** | `<event>` | `<video>` | `<image>` |
| **Sampling** | `do_sample=True, temp=0.6` | `do_sample=False` | `do_sample=False` |
| **Processor Input** | `event_tensors` | `videos=` | `images=` |

## Key Findings

1. **Total Time Speedup**: EventGPT is **~3.5x faster** than LLaVA (c = 0.282)

2. **Low Acceptance Rate**: α ≈ 3% because models produce semantically similar but textually different outputs

3. **Grid Image Fix**: LLaVA 1.5 requires video frames to be combined into a single grid image for quality output

4. **Memory Efficiency**: Both models use similar peak memory (~14 GB) with grid image approach

5. **Speculative Decoding**: With low acceptance rate (3.4%), theoretical speedup is limited

## Metrics Definition

- **c (Draft/Target Ratio)**: `EventGPT_Total / LLaVA_Total` - lower is better
- **α (Acceptance Rate)**: Fraction of draft tokens accepted by target model
- **γ (Gamma)**: Number of draft tokens per verification step
- **Theoretical Speedup**: Expected speedup with speculative decoding

---

## Bug Fixes

### Video-LLaVA `model.generate()` Bug Fix

**Date:** 2026-01-22

**Problem:**
`model.generate()` in transformers 4.44.0 produces garbage output for Video-LLaVA (e.g., `"1 1 1 1 1..."` instead of coherent text).

**Root Cause:**
The bug occurs during the generation loop in transformers 4.44.0, not in the forward pass. Manual step-by-step generation works correctly, but `model.generate()` fails. The forward pass produces correct logits (top predictions: "This", "The", "It"), but the generation loop corrupts the output.

**Solution:**
Upgrade transformers to version **4.47.0 or later**:

```bash
pip install transformers==4.47.0 -i https://mirrors.aliyun.com/pypi/simple/
```

**Additional Requirement:**
Set environment variable for protobuf compatibility:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

Or run the benchmark with:
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python benchmark_inference.py ...
```

**Verified Working Configuration:**
- transformers: 4.47.0
- torch: 2.1.2+cu121
- tokenizers: 0.21.4

**Test Results:**
- Official HuggingFace test video: `"This is funny because the baby is playing with a Wii remote while sitting on the floor, which is an unusual and amusing sight..."` ✅
- DSEC driving scenes: `"The image captures a car driving down a winding road, surrounded by trees and mountains..."` ✅

**Legacy Workaround (for transformers < 4.47.0):**
If you cannot upgrade transformers, a `custom_video_llava_generate()` function is available in the codebase that performs manual autoregressive decoding as a workaround.
