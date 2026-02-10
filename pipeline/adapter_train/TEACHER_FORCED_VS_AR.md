# Teacher-Forced Forward Pass vs Autoregressive Generate

> Author: Alice Zhang
> Date: 2026-02-07

This document compares the two inference modes used in EventGPT's cross-modal speculative decoding pipeline.

---

## Overview

| | Teacher-Forced Forward Pass | Autoregressive Generate |
|---|---|---|
| **Purpose** | Training (hidden state alignment) | Inference (text generation) |
| **Input** | Full prompt + all generation tokens at once | Prompt only; tokens generated one at a time |
| **Calls** | `model.model(inputs_embeds=...)` | `model.generate(input_ids=...)` |
| **Gradient** | Yes (backprop through LoRA params) | No (`@torch.no_grad()`) |
| **Speed** | 1 forward pass for entire sequence | N forward passes for N tokens |
| **KV Cache** | Not used (`use_cache=False`) | Used (`use_cache=True`) |
| **Output** | Hidden states at every position | Token IDs + optionally hidden states |

---

## Autoregressive Generate (AR)

Standard text generation. Used in `extract_hidden_states.py` and at inference time.

```
Step 0: [prompt tokens]           → predict token_0
Step 1: [prompt tokens, token_0]  → predict token_1
Step 2: [prompt tokens, token_0, token_1] → predict token_2
...
Step N: [prompt tokens, token_0, ..., token_{N-1}] → predict token_N
```

Each step:
1. Forward pass through the full decoder (or just last token with KV cache)
2. Get logits at last position
3. Sample/argmax → next token
4. Append to sequence, repeat

```python
# From extract_hidden_states.py
with torch.inference_mode():
    outputs = model.generate(
        input_ids,
        event_features=event_features,
        event_image_sizes=[event_image_size],
        do_sample=False,
        max_new_tokens=50,
        use_cache=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

# Collect hidden states from each step
for step_hidden in outputs.hidden_states:
    last_layer = step_hidden[-1]       # last decoder layer
    last_pos = last_layer[0, -1, :]    # last position → [4096]
    all_hidden.append(last_pos)
```

**Key properties:**
- `@torch.no_grad()` decorator on `EventChatModel.generate()` — cannot backprop
- Each step only sees tokens generated so far (causal)
- KV cache avoids recomputing past tokens
- N forward passes for N tokens → slow for training

---

## Teacher-Forced Forward Pass (TF)

Feed the entire sequence (prompt + known generation tokens) in a single forward pass. The causal attention mask ensures each position only sees previous positions — mathematically equivalent to AR generation.

```
Single pass: [prompt tokens, token_0, token_1, ..., token_N]
                                ↓         ↓              ↓
                            hidden_0   hidden_1       hidden_N
```

Each position `t` in the decoder:
- Attends only to positions `0..t` (causal mask)
- Produces the same hidden state as AR step `t`
- All positions computed in parallel via matrix operations

```python
# From train_lora_adapter.py
def teacher_forced_forward(self, input_data, gen_token_ids):
    # Concatenate prompt + generation tokens
    full_input_ids = torch.cat([input_ids, gen_ids], dim=1)

    # Prepare multimodal inputs (replace event tokens with visual features)
    _, position_ids, attention_mask, _, inputs_embeds, _ = \
        model.prepare_inputs_labels_for_multimodal(
            full_input_ids, None, None, None, None,
            event_features=event_features,
            event_image_sizes=[event_image_size],
        )

    # Single forward pass with gradients
    outputs = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False,
    )

    # Extract generation positions (last gen_len positions)
    gen_hidden = outputs.last_hidden_state[0, -gen_len:, :]
    return gen_hidden  # [gen_len, 4096]
```

**Key properties:**
- Gradients flow through decoder → can train LoRA weights
- Single forward pass for entire sequence → O(1) passes vs O(N)
- Causal mask ensures mathematical equivalence to AR
- No KV cache needed (full sequence processed at once)
- `use_cache=False` since we don't need cached keys/values

---

## Why They Produce the Same Hidden States

The causal attention mask is the key:

```
Attention mask for sequence [A, B, C, D]:

     A  B  C  D
A  [ 1  0  0  0 ]    A sees only A
B  [ 1  1  0  0 ]    B sees A, B
C  [ 1  1  1  0 ]    C sees A, B, C
D  [ 1  1  1  1 ]    D sees A, B, C, D
```

Position D's hidden state is computed using attention over {A, B, C, D} — identical to what AR generation would compute at step 3 after generating tokens A, B, C.

This equivalence holds because:
1. Transformer self-attention with causal mask = processing each position independently of future tokens
2. No information leaks from future positions
3. The same weights, same inputs, same attention pattern → same output

---

## When to Use Each

| Scenario | Method | Why |
|----------|--------|-----|
| **Hidden state extraction** (L1-L5 data) | AR Generate | Need actual generated text + hidden states; no gradients needed |
| **L6 LoRA training** | Teacher-Forced | Need gradients for backprop; use known EGPT tokens as generation targets |
| **E2E inference** | AR Generate | Real-time generation; no known future tokens |
| **L6 validation** | Teacher-Forced | Evaluate alignment quality using known tokens |

---

## Speed Comparison (Approximate)

For a 50-token generation with EventGPT-7B (4-bit):

| | AR Generate | Teacher-Forced |
|---|---|---|
| Forward passes | 50 (1 per token) | 1 (all tokens at once) |
| Wall time | ~2.5 sec | ~0.3 sec |
| GPU memory peak | Lower (KV cache) | Higher (full sequence activations) |
| Supports backprop | No | Yes |

Teacher-forced is ~8x faster per sample for getting hidden states, making it practical for training loops over 50K+ samples.

---

## In Our Pipeline

```
L1-L5 workflow:
  extract_hidden_states.py (AR generate) → .pt chunks → train_hidden_adapter.py (no model needed)

L6 workflow:
  train_lora_adapter.py (teacher-forced forward, live EGPT model with LoRA, gradients)
```

L1-L5 extract hidden states once, then train a separate adapter on the pre-extracted data.
L6 runs live forward passes through the model during training — teacher-forced makes this feasible by avoiding N sequential generate steps per sample.
