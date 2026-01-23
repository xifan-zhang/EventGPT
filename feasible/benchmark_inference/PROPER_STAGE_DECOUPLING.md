# Proper Stage 3+4 Decoupling: Vision Encoding then Decoding

**Date:** 2026-01-23
**Status:** Analysis + Implementation Plan
**Finding:** YES, proper decoupling IS possible for EventGPT! Partial for Video-LLaVA.

---

## EventGPT: FULL DECOUPLING POSSIBLE ✅

### Discovery: `visval_encode()` Method

EventGPT has a dedicated vision encoding method:

```python
def visval_encode(self, event_tensor):
    """Stage 3: Vision encoding (CLIP feature extraction)"""
    with torch.no_grad():
        outputs = self.get_model().visual_tower.visual_tower(event_tensor)
    events_feature = outputs.last_hidden_state
    events_feature = events_feature.detach().requires_grad_(True)
    events_feature = self.get_model().visual_projector(events_feature)
    return events_feature
```

Location: `model/EventChatModel.py:185`

### Implementation Strategy

**Stage 3 (Vision Encoding - Separated)**
```python
# Stage 3: Extract vision features ONCE
with torch.inference_mode():
    with torch.cuda.nvtx.range("Stage3_VisionEncoding"):
        event_features = model.visval_encode(event_tensor)
        # Returns: Tensor[batch, num_patches, hidden_dim]
```

**Stage 4 (LLM Decoding - with cached features)**
```python
# Stage 4: Use pre-computed features for generation
with torch.inference_mode():
    with torch.cuda.nvtx.range("Stage4_LLMDecoding"):
        # Modified generate call that accepts pre-computed features
        output_ids = model.generate_with_cached_features(
            input_ids,
            event_features=event_features,  # Pre-computed!
            max_new_tokens=512
        )
```

### How It Works

In `prepare_inputs_labels_for_multimodal()`, the model:
1. Calls `visval_encode()` to get event features (Stage 3)
2. Embeds text tokens with `embed_tokens()` (Stage 2)
3. Interleaves text and event tokens based on EVENT_TOKEN_INDEX
4. Calls `super().generate()` with `inputs_embeds` (Stage 4)

**Key insight:** We can extract steps 1-2 as preprocessing, then Stage 4 is pure generation.

### Code Changes Required

Modify EventGPT's generate method to accept pre-computed features:

```python
def generate_with_cached_features(
    self,
    input_ids: torch.Tensor,
    event_features: torch.Tensor,  # NEW: pre-computed features
    event_positions: List[int] = None,  # Where to interleave features
    **kwargs,
) -> torch.LongTensor:
    """
    Stage 4: Generate with cached vision features

    Args:
        input_ids: Text token IDs [batch, seq_len]
        event_features: Pre-computed vision features [batch, patches, hidden]
        event_positions: Positions where to insert event features
    """
    # Instead of re-encoding event_tensors, use cached event_features
    inputs_embeds = self.prepare_inputs_embeds_with_cached_features(
        input_ids,
        event_features,
        event_positions
    )

    return super().generate(
        inputs_embeds=inputs_embeds,
        **kwargs
    )
```

---

## Video-LLaVA: PARTIAL DECOUPLING (with workarounds)

### Challenge

Video-LLaVA is a HuggingFace model where generate() is more tightly integrated. However, we can still measure vision vs generation:

### Approach 1: Forward Hooks (Non-invasive)

```python
vision_encoding_time = {"start": 0, "end": 0}

def vision_forward_hook(module, input, output):
    vision_encoding_time["end"] = time.time()
    return output

def register_vision_hooks(model):
    """Register hooks to measure vision tower latency"""
    # Hook on video_tower
    model.vision_tower.register_forward_hook(vision_forward_hook)
    # Alternative: hook on language_model
    model.language_model.register_forward_hook(...)
```

Then:
```python
# Stage 3: Start hook timer
vision_encoding_time["start"] = time.time()

# Stage 4: Call generate (hooks will capture timing)
generate_ids = model.generate(**inputs)

stage3_time = vision_encoding_time["end"] - vision_encoding_time["start"]
stage4_time = total_time - stage3_time
```

### Approach 2: Custom Generation Loop (More control)

```python
def generate_with_timing(
    model,
    processor,
    inputs,
    max_new_tokens=512
):
    """
    Custom generation that separates vision encoding from LLM decoding
    """
    input_ids = inputs['input_ids']
    pixel_values_videos = inputs.get('pixel_values_videos')
    attention_mask = inputs.get('attention_mask')

    # Stage 3: Encode video frames
    with torch.inference_mode():
        video_features = model.vision_tower(pixel_values_videos)
        # Returns: Tensor[batch, num_frames, patches, hidden]

    # Stage 4: Generate tokens using cached features
    with torch.inference_mode():
        # Manually implement generation loop
        generated_ids = input_ids.clone()
        past_key_values = None

        for step in range(max_new_tokens):
            outputs = model.language_model(
                input_ids=generated_ids if step == 0 else next_token.unsqueeze(0),
                video_features=video_features if step == 0 else None,
                attention_mask=...,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)

            if next_token == eos_token:
                break

    return generated_ids
```

---

## LLaVA 1.5: PARTIAL DECOUPLING (similar to Video-LLaVA)

Same approaches as Video-LLaVA apply here.

---

## Summary Table

| Model | Stage 3 Decoupling | Stage 4 Decoupling | Method | Difficulty |
|-------|--------------------|--------------------|--------|------------|
| **EventGPT** | ✅ Full | ✅ Full | Modify `generate()` | Low |
| **Video-LLaVA** | ⚠️ Hooks | ⚠️ Custom loop | Forward hooks or custom loop | Medium |
| **LLaVA 1.5** | ⚠️ Hooks | ⚠️ Custom loop | Forward hooks or custom loop | Medium |

---

## Recommended Implementation Path

### Priority 1: EventGPT (High Payoff, Low Effort)
1. Extract `visval_encode()` call as Stage 3
2. Modify `generate()` to accept `event_features` parameter
3. Simple, minimal changes to model code

### Priority 2: Video-LLaVA (Medium Payoff, Medium Effort)
1. Use forward hooks to measure vision encoding
2. Keep existing generate() (no model modification)
3. Reliable timing with minimal invasiveness

### Priority 3: LLaVA 1.5 (Similar to Video-LLaVA)
1. Same hook approach as Video-LLaVA
2. Consistent methodology

---

## Implementation Details for EventGPT

### Step 1: Add cached features parameter to generate()

```python
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    event_tensors: Optional[torch.Tensor] = None,
    event_features: Optional[torch.Tensor] = None,  # NEW
    event_image_sizes: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)

    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    # If event_features provided, use them directly (Stage 4 only)
    if event_features is not None:
        inputs, position_ids, attention_mask, _, inputs_embeds, _ = \
            self.prepare_inputs_with_cached_features(
                inputs,
                position_ids,
                attention_mask,
                event_features,
                event_image_sizes
            )
    # Otherwise encode from event_tensors (original behavior)
    elif event_tensors is not None:
        inputs, position_ids, attention_mask, _, inputs_embeds, _ = \
            self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None,
                event_tensors, event_image_sizes
            )
    else:
        raise NotImplementedError("please input Event")

    return super().generate(
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        **kwargs
    )
```

### Step 2: Add feature preparation method

```python
def prepare_inputs_with_cached_features(
    self, input_ids, position_ids, attention_mask,
    event_features, event_image_sizes
):
    """Prepare inputs using pre-computed vision features (Stage 4 only)"""
    # Similar to prepare_inputs_labels_for_multimodal,
    # but skips visval_encode() and uses event_features directly

    # Embed text tokens
    cur_input_embeds = self.get_model().embed_tokens(input_ids)

    # Interleave with cached event features
    new_input_embeds = []
    event_idx = 0
    for batch_idx, cur_input_id in enumerate(input_ids):
        event_token_indices = [-1] + torch.where(cur_input_id == EVENT_TOKEN_INDEX)[0].tolist() + [cur_input_id.shape[0]]

        # Split text and insert event features
        cur_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_embeds_no_im = torch.split(cur_embeds, split_sizes)

        cur_new_embeds = []
        for i in range(num_events + 1):
            cur_new_embeds.append(cur_embeds_no_im[i])
            if i < num_events:
                # Use cached feature instead of encoding
                cur_new_embeds.append(event_features[event_idx])
                event_idx += 1

        new_input_embeds.append(torch.cat(cur_new_embeds))

    # Pad and return
    max_len = max(x.shape[0] for x in new_input_embeds)
    ...
    return input_ids, position_ids, attention_mask, None, inputs_embeds_padded, None
```

---

## Testing Decoupling

```python
# Stage 3: Vision Encoding
with torch.inference_mode():
    event_features = model.visval_encode(event_tensor)

# Stage 4: LLM Decoding (using cached features)
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        event_features=event_features,  # Pre-computed
        max_new_tokens=512
    )

# Verify: Should produce same output as regular generate()
output_ids_regular = model.generate(
    input_ids,
    event_tensors=event_tensor,  # Original path
    max_new_tokens=512
)

assert torch.equal(output_ids, output_ids_regular), "Decoupling verification failed!"
```

---

## Conclusion

**YES, proper Stage 3+4 decoupling IS POSSIBLE:**

1. **EventGPT:** Can be fully decoupled with minor modifications (~100 lines)
2. **Video-LLaVA:** Can use hooks for timing measurement (~50 lines)
3. **LLaVA 1.5:** Same hook approach as Video-LLaVA

**Next steps:**
1. Implement EventGPT cached features support
2. Add forward hooks for Video-LLaVA timing
3. Re-run benchmarks with proper stage separation
4. Measure actual vision vs generation bottlenecks

---

**Status:** Ready for Implementation
**Estimated effort:** 2-3 hours
**Payoff:** Accurate Stage 3 vs Stage 4 timing breakdown
