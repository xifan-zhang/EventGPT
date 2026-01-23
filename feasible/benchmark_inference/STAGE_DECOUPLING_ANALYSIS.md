# Stage Decoupling Analysis: Why Stages 3+4 Cannot be Separated

**Date:** 2026-01-23
**Issue:** Original 4-stage timing implementation measured vision encoding in Stage 3 but didn't actually use the extracted features in Stage 4, causing double-counting of vision encoding work.
**Resolution:** Reverted to effectively 3-stage timing (Stage 1, 2, combined 3+4)

---

## Problem Statement

The goal was to split inference into 4 stages:
- **Stage 1:** Data loading
- **Stage 2:** Preprocessing
- **Stage 3:** Vision encoding (CLIP feature extraction)
- **Stage 4:** LLM decoding (token generation)

However, the implementations fail to properly decouple:

### EventGPT
```python
# Stage 3 - Extract features
vision_tower = model.get_visual_tower()
image_features = [vision_tower(event) for event in event_tensor]

# Stage 4 - But model.generate() re-encodes!
output_ids = model.generate(
    input_ids,
    event_tensors=event_tensor,  # ← Passes ORIGINAL tensors, not cached features
    ...
)
```

**Result:** Vision encoding happens in BOTH Stage 3 and Stage 4. Features extracted in Stage 3 are discarded.

### Video-LLaVA
```python
# Stage 3 - Extract features
video_features = model.vision_tower(pixel_values_videos)

# Stage 4 - But model.generate() re-encodes!
generate_ids = model.generate(
    **inputs,  # ← Contains original pixel_values_videos, not cached features
    ...
)
```

**Result:** Same problem - vision encoding happens twice.

### LLaVA 1.5
```python
# Stage 3 - Extract features
image_features = model.vision_tower(pixel_values)

# Stage 4 - But model.generate() re-encodes!
generate_ids = model.generate(
    **inputs,  # ← Contains original pixel_values, not cached features
    ...
)
```

**Result:** Same problem - vision encoding happens twice.

---

## Root Cause: Monolithic model.generate()

The core issue is that **HuggingFace model.generate() is a monolithic function** that handles both vision encoding and LLM decoding internally:

```python
def generate(...):
    # Internally:
    # 1. Encode input_ids (tokens) via embedding layer
    # 2. Encode visual features (images/video) via vision tower
    # 3. Generate tokens using LLM decoder with both inputs
    # (All fused together, cannot extract intermediate results)
```

---

## Architecture Limitations

### EventGPT
**Location:** `model.EventChatModel.generate()`

```
Input: [input_ids, event_tensors]
  ↓
[Inside model.generate() - MONOLITHIC]
  ├─ Text encoding: embed(input_ids)
  ├─ Vision encoding: visual_tower(event_tensors)  ← Hidden inside
  ├─ Feature projection: visual_projector(features)  ← Hidden inside
  ├─ KV cache management  ← Hidden inside
  └─ Token generation loop  ← Mixed with vision
  ↓
Output: [generated_tokens]
```

**Problem:** Cannot intercept vision encoding without modifying model internals.

### Video-LLaVA
**Location:** `transformers.VideoLlavaForConditionalGeneration.generate()`

```
Input: [input_ids, pixel_values_videos]
  ↓
[Inside model.generate() - MONOLITHIC]
  ├─ Text encoding: embed(input_ids)
  ├─ Video encoding: video_encoder(pixel_values_videos)  ← Hidden inside
  ├─ Temporal pooling  ← Hidden inside
  ├─ Feature projection  ← Hidden inside
  ├─ KV cache management  ← Hidden inside
  └─ Token generation loop  ← Mixed with vision
  ↓
Output: [generated_tokens]
```

**Problem:** Same as EventGPT - cannot intercept vision encoding.

---

## What Would Be Required to Properly Decouple

### Option 1: Modify Model Forward Pass (Invasive)

Would need to modify each model's `generate()` method to accept pre-computed vision features:

```python
# Modified model.generate() signature:
def generate(input_ids, vision_features=None, **kwargs):
    if vision_features is None:
        # Original behavior: encode vision
        vision_features = self.vision_tower(kwargs['event_tensors'])
    else:
        # New behavior: skip vision encoding, use cached features
        pass

    # Generate tokens using cached features
    return self._generate_with_cached_vision(input_ids, vision_features, **kwargs)
```

**Pros:** Properly decouples vision and generation
**Cons:**
- Requires modifying HuggingFace transformers
- Would not work with model updates
- Not practical for benchmarking

### Option 2: Custom Generation Loop (Moderate)

Implement a custom generation function that manually manages KV cache:

```python
def generate_with_cached_vision(model, input_ids, vision_features, max_new_tokens=512):
    """
    Custom generation that accepts pre-computed vision features
    """
    # Prepare initial inputs
    embed_output = model.embed_tokens(input_ids)

    # Project vision features
    vision_projections = model.visual_projector(vision_features)

    # Concatenate text embeddings with vision features
    combined_input_embeds = torch.cat([embed_output, vision_projections], dim=1)

    # Manually loop through token generation with KV cache
    past_key_values = None
    generated_ids = input_ids.clone()

    for step in range(max_new_tokens):
        outputs = model.transformer(
            inputs_embeds=combined_input_embeds[:, -1:, :],  # Only current position
            past_key_values=past_key_values,
            use_cache=True
        )
        past_key_values = outputs.past_key_values

        # Sample next token
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)

        if next_token == eos_token:
            break

    return generated_ids
```

**Pros:** Properly decouples stages
**Cons:**
- Requires understanding of model internals
- Difficult to maintain across model updates
- High risk of bugs/correctness issues
- Different for each model (EventGPT, Video-LLaVA, LLaVA 1.5)

### Option 3: Use PyTorch Hooks (Intermediate)

Instrument model with hooks to measure vision encoding without modifying architecture:

```python
vision_encoding_time = 0

def vision_tower_hook(module, input, output):
    global vision_encoding_time
    vision_encoding_time = ...  # Measure from hook timing
    return output

model.vision_tower.register_forward_hook(vision_tower_hook)

# Then run full generate() and vision_tower_hook intercepts timing
```

**Pros:** Non-invasive, works with existing models
**Cons:**
- Only provides timing, not true decoupling
- Overhead from hook execution
- Doesn't prevent re-encoding

---

## Current Solution: Accept Monolithic Stages

The current fix acknowledges that proper decoupling is not feasible for these models:

```python
stage3_time = 0.0  # Fused into Stage 4 (cannot decouple)
stage4_time = full_inference_time  # Includes vision encoding + token generation
```

### Why This Is Correct

1. **Honest Measurement:** Reflects actual model behavior
2. **No Double-Counting:** Vision encoding measured once, in Stage 4
3. **Simple:** No complex custom loops to maintain
4. **Portable:** Works with model updates

### Limitation

Cannot separately measure:
- Vision encoding time
- LLM decoding time
- Per-token generation speed

However, this information can be obtained through:
- **Profiling:** `torch.profiler.profile()`
- **Model instrumentation:** Forward hooks for specific layers
- **Custom implementation:** For research/optimization (not in production code)

---

## Comparison: What Each Model Supports

| Feature | EventGPT | Video-LLaVA | LLaVA 1.5 |
|---------|----------|-------------|----------|
| Pre-computed vision features | ❌ No | ❌ No | ❌ No |
| KV cache support | ✅ Yes | ✅ Yes | ✅ Yes |
| Modifiable generate() | ❌ Fixed | ❌ Fixed | ❌ Fixed |
| Vision tower direct call | ✅ Yes (for measurement only) | ✅ Yes (for measurement only) | ✅ Yes (for measurement only) |

---

## Benchmark Implications

### Actual Timing (Corrected)

```
EventGPT (200 samples):
  Stage 1 (Load):        0.008s ✓ Decoupled
  Stage 2 (Preprocess):  0.017s ✓ Decoupled
  Stage 3+4 (Inference): 0.976s ✗ Monolithic (vision + generation fused)
  ─────────────────────────
  Total:                 1.001s

Video-LLaVA (200 samples):
  Stage 1 (Load):        0.784s ✓ Decoupled
  Stage 2 (Preprocess):  0.067s ✓ Decoupled
  Stage 3+4 (Inference): 2.451s ✗ Monolithic (vision + generation fused)
  ─────────────────────────
  Total:                 3.302s
```

### Impact on Analysis

**Before (incorrect):**
- Claimed Stage 3 (vision) = 0.028s
- Claimed Stage 4 (generation) = 0.948s
- **But vision was encoded twice** (invalid)

**After (correct):**
- Stage 3+4 (fused) = 0.976s
- Cannot separately measure vision vs generation
- Reflects actual model behavior

---

## Recommendations for Future Work

### If Vision/Generation Split Is Needed

1. **Use profiling tools:**
   ```python
   import torch.profiler

   with torch.profiler.profile(...) as prof:
       model.generate(...)
   prof.key_averages().table(sort_by="cpu_time")
   ```

2. **Profile specific layers:**
   - Measure vision tower directly (separate from generation)
   - Register forward hooks on generation components

3. **Implement custom generation (if critical):**
   - Accept complexity/maintenance burden
   - Test thoroughly for correctness
   - Document internal model structure dependencies

### Best Practice Going Forward

- **Document model capabilities** (pre-computed feature support)
- **Use profiling for internal measurements** (not for stage timing)
- **Measure only what models support** (honest benchmarking)
- **Compare models fairly** (same measurement methodology)

---

## References

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [EventGPT Model Code](https://github.com/YourRepo/EventGPT)
- [Video-LLaVA Model Card](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)
- [PyTorch Profiler Guide](https://pytorch.org/tutorials/recipes/recipes/profiling.html)

---

**Conclusion:** The original 4-stage timing was fundamentally flawed due to model architecture constraints. The corrected 3-stage timing (Stage 1, 2, combined 3+4) honestly reflects what these models can support.

**Last Updated:** 2026-01-23 23:42 UTC
