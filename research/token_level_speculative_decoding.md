# Token-Level Speculative Decoding (Standard/Vanilla)

## Overview

**Token-level speculative decoding** (also called **standard** or **vanilla** speculative decoding) is the foundational approach to LLM inference acceleration. It uses a smaller **draft model** to predict tokens, which are then verified by a larger **target model**.

This is the original speculative decoding approach introduced concurrently by:
- **Chen et al. (2023)** - "Accelerating Large Language Model Decoding with Speculative Sampling"
- **Leviathan et al. (2023)** - "Fast Inference from Transformers via Speculative Decoding"

---

## Core Algorithm

### Standard Speculative Sampling

```
Algorithm: Vanilla Speculative Sampling

Input: Context C, Draft Model M_d, Target Model M_t, Draft Length k
Output: Generated tokens

1: procedure SPECULATIVE_SAMPLE(C, M_d, M_t, k):
2:     # Draft Stage: Generate k tokens with small model
3:     for i = 1 to k:
4:         t_i ← M_d.generate(C, temperature=τ)
5:         q_i ← M_d.get_probability(t_i | C)
6:         C ← C + t_i
7:     end for
8:
9:     # Verification Stage: Verify all k tokens with large model (in parallel)
10:    for i = 1 to k:
11:        p_i ← M_t.get_probability(t_i | C_original)
12:        α_i ← min(1, p_i / q_i)  # Acceptance probability
13:
14:        if random() < α_i:
15:            accept(t_i)  # Draft token accepted
16:        else:
17:            # Rejection: Resample from adjusted distribution
18:            p_adjusted ← clamp(p_i - q_i, min=0)
19:            p_adjusted ← p_adjusted / sum(p_adjusted)
20:            t_new ← sample(p_adjusted)
21:            accept(t_new)
22:            break  # Stop accepting further drafts
23:    return accepted_tokens
```

### Key Mathematical Equations

**Draft Probability:**
```
q_i = P_M_d(t_i | C, t_1, ..., t_{i-1})
```

**Target Probability:**
```
p_i = P_M_t(t_i | C, t_1, ..., t_{i-1})
```

**Acceptance Probability:**
```
α_i = min(1, p_i / q_i)
```

**Resampling Distribution (on rejection):**
```
P_resample(t) = max(0, P_M_t(t) - P_M_d(t)) / Z

Where Z = Σ_t max(0, P_M_t(t) - P_M_d(t))  # Normalization constant
```

**Expected Speedup:**
```
Speedup = (k + 1) / (1 + k * (T_d / T_t))

Where:
- k: Average number of accepted draft tokens per cycle
- T_d: Time per forward pass of draft model
- T_t: Time per forward pass of target model

Optimal when: T_d << T_t and acceptance_rate >> 0
```

---

## Key Papers

### 1. Chen et al. (2023)
**Paper:** "Accelerating Large Language Model Decoding with Speculative Sampling"
- **arXiv:** [2302.01318](https://arxiv.org/abs/2302.01318)
- **Authors:** Charlie Chen, Geoffrey Irving, Jean-Baptiste Lespiau, et al.
- **Published:** February 2023
- **Citations:** 642+

**Key Contributions:**
- Introduced speculative sampling as an inference acceleration technique
- Proved that the method preserves the target model's distribution
- Showed 2-3x speedup on language modeling tasks

### 2. Leviathan et al. (2023)
**Paper:** "Fast Inference from Transformers via Speculative Decoding"
- **arXiv:** [2211.17192](https://arxiv.org/abs/2211.17192)
- **Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias
- **Venue:** ICML 2023
- **Published:** November 2022 (preprint), May 2023 (ICML)
- **Citations:** 1,122+

**Key Contributions:**
- Concurrent discovery of speculative decoding
- Focus on transformer architectures specifically
- Demonstrated effectiveness on GPT-style models

---

## Implementation

### Sample Code (PyTorch)

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple

class TokenLevelSpeculativeDecoding:
    """
    Standard/Vanilla Token-Level Speculative Decoding

    Uses a small draft model to predict tokens, verified by a larger target model.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        draft_length: int = 5,
        temperature: float = 1.0
    ):
        """
        Args:
            draft_model: Smaller model for fast drafting
            target_model: Larger model for verification
            draft_length: Number of draft tokens to generate
            temperature: Sampling temperature
        """
        self.draft_model = draft_model
        self.target_model = target_model
        self.draft_length = draft_length
        self.temperature = temperature

    @torch.no_grad()
    def generate_drafts(
        self,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate k draft tokens using draft model

        Args:
            context: Input context (batch, seq_len)

        Returns:
            draft_tokens: (batch, k) draft token IDs
            draft_probs: (batch, k, vocab_size) draft probabilities
        """
        draft_tokens = []
        draft_probs = []

        current_context = context.clone()

        for _ in range(self.draft_length):
            # Forward pass
            outputs = self.draft_model(current_context)
            logits = outputs.logits[:, -1, :] / self.temperature
            probs = torch.softmax(logits, dim=-1)

            # Sample token
            next_token = torch.multinomial(probs, num_samples=1)

            # Store
            draft_tokens.append(next_token)
            draft_probs.append(probs)

            # Append to context
            current_context = torch.cat([current_context, next_token], dim=-1)

        draft_tokens = torch.cat(draft_tokens, dim=-1)  # (batch, k)
        draft_probs = torch.stack(draft_probs, dim=1)  # (batch, k, vocab_size)

        return draft_tokens, draft_probs

    @torch.no_grad()
    def verify_drafts(
        self,
        context: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Verify draft tokens using target model

        Args:
            context: Original input context (batch, seq_len)
            draft_tokens: (batch, k) draft token IDs
            draft_probs: (batch, k, vocab_size) draft probabilities

        Returns:
            accepted_tokens: (batch, n_accepted) accepted token IDs
        """
        batch_size = context.shape[0]
        device = context.device
        vocab_size = draft_probs.shape[-1]

        # Create full sequence for target model
        full_sequence = torch.cat([context, draft_tokens], dim=-1)

        # Get target model probabilities
        target_outputs = self.target_model(full_sequence)
        target_logits = target_outputs.logits[:, context.shape[1]-1:-1, :] / self.temperature
        target_probs = torch.softmax(target_logits, dim=-1)

        accepted = []

        for i in range(self.draft_length):
            # Get probabilities for this position
            q_i = draft_probs[:, i, :]  # (batch, vocab_size)
            p_i = target_probs[:, i, :]  # (batch, vocab_size)

            # Get draft token
            t_i = draft_tokens[:, i:i+1]  # (batch, 1)

            # Get probability of draft token
            q_t = q_i.gather(1, t_i).squeeze(1)  # (batch,)
            p_t = p_i.gather(1, t_i).squeeze(1)  # (batch,)

            # Acceptance probability
            alpha = torch.clamp(p_t / q_t, max=1.0)  # (batch,)

            # Accept or reject
            random_draws = torch.rand(batch_size, device=device)
            accept_mask = (random_draws < alpha).long()

            accepted_batch = []

            for b in range(batch_size):
                if accept_mask[b] == 1:
                    # Accept draft token
                    accepted_batch.append(t_i[b:b+1])
                else:
                    # Reject: resample from adjusted distribution
                    p_adjusted = torch.clamp(p_i[b] - q_i[b], min=0)
                    p_adjusted = p_adjusted / (p_adjusted.sum() + 1e-10)

                    sampled = torch.multinomial(p_adjusted, num_samples=1)
                    accepted_batch.append(sampled)

                    # Stop processing this batch element
                    break

            if accepted_batch:
                accepted.append(torch.cat(accepted_batch, dim=0).unsqueeze(1))

        if not accepted:
            # Fallback: generate one token with target model
            outputs = self.target_model(context)
            logits = outputs.logits[:, -1, :] / self.temperature
            probs = torch.softmax(logits, dim=-1)
            fallback = torch.multinomial(probs, num_samples=1)
            return fallback

        return torch.cat(accepted, dim=-1).unsqueeze(0)

    def generate(
        self,
        prompt: torch.Tensor,
        max_tokens: int = 100,
    ) -> torch.Tensor:
        """
        Generate text with token-level speculative decoding

        Args:
            prompt: Input prompt (batch, seq_len)
            max_tokens: Maximum tokens to generate

        Returns:
            generated: Full generated sequence
        """
        context = prompt.clone()

        for _ in range(max_tokens):
            # Generate drafts
            draft_tokens, draft_probs = self.generate_drafts(context)

            # Verify drafts
            accepted = self.verify_drafts(context, draft_tokens, draft_probs)

            # Append accepted tokens
            context = torch.cat([context, accepted], dim=-1)

            # Check for EOS
            if context[0, -1].item() == self.target_model.config.eos_token_id:
                break

        return context


# Usage example
def example_usage():
    """Example of using token-level speculative decoding"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load models
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Draft model (smaller)
    draft_model = AutoModelForCausalLM.from_pretrained("gpt2")
    draft_model.eval()

    # Target model (larger, could be gpt2-xl for actual use)
    target_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    target_model.eval()

    # Initialize speculative decoder
    decoder = TokenLevelSpeculativeDecoding(
        draft_model=draft_model,
        target_model=target_model,
        draft_length=5,
        temperature=1.0
    )

    # Generate
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt")

    output = decoder.generate(inputs["input_ids"], max_tokens=50)

    # Decode
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    example_usage()
```

---

## Assisted Generation (HuggingFace)

HuggingFace implements token-level speculative decoding as **"Assisted Generation"**:

### Official Implementation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
assistant_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")
target_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

# Generate with assisted generation
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = target_model.generate(
    **inputs,
    assistant_model=assistant_model,  # Draft model
    do_sample=True,
    temperature=0.7,
    max_new_tokens=100
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Universal Assisted Generation (2024)

HuggingFace's **Universal Assisted Generation** (October 2024) extends this to work with models that have **different tokenizers**:

```python
# Draft and target can now have different tokenizers
draft_tokenizer = AutoTokenizer.from_pretrained("gpt2")
target_tokenizer = AutoTokenizer.from_pretrained("llama-2-7b")

# The library handles token alignment automatically
outputs = target_model.generate(
    **target_inputs,
    assistant_model=assistant_model,
    assistant_tokenizer=draft_tokenizer,  # Different tokenizer!
    do_sample=True,
    max_new_tokens=100
)
```

---

## Performance Characteristics

### Speedup Formula

```
Speedup = (Time_vanilla) / (Time_speculative)

Time_vanilla = N * T_target

Time_speculative = N_cycles * (T_draft_gen + T_target_verify)

Where:
- N_cycles = N / (k * acceptance_rate)
- k: Draft length
- acceptance_rate: Average acceptance probability

Theoretical max speedup (perfect acceptance):
Speedup_max ≈ (k + 1) / (1 + k * (T_draft / T_target))

For T_draft = 0.1 * T_target, k = 5:
Speedup_max ≈ 6 / (1 + 0.5) = 4x
```

### Acceptance Rate Factors

The acceptance rate depends on:
1. **Draft Model Quality** - Better models → higher acceptance
2. **Task Difficulty** - Easy tasks (code, templated text) → higher acceptance
3. **Temperature** - Lower temperature → higher acceptance
4. **Model Similarity** - Draft and target from same family → higher acceptance

Typical acceptance rates:
- Same-family models (e.g., GPT-2 → GPT-2 XL): 70-90%
- Cross-family (e.g., LLaMA → GPT): 40-60%

---

## Limitations of Token-Level Approach

1. **Distribution Mismatch**
   - Draft and target models have different distributions
   - Causes low acceptance rates when models are dissimilar

2. **Sequential Draft Generation**
   - Drafts must be generated sequentially (autoregressive)
   - No parallelism in drafting stage

3. **Error Propagation**
   - Early rejections cause all later drafts to be discarded
   - Wastes computation on later drafts

4. **Requires Paired Models**
   - Need both draft and target models
   - Original requirement: same tokenizer (relaxed in 2024)

---

## Comparison with Feature-Level

| Aspect | Token-Level | Feature-Level |
|--------|-------------|---------------|
| **Prediction Target** | Next token distribution | Next hidden state |
| **Draft Generation** | Sequential (autoregressive) | Can be parallel |
| **Acceptance Rate** | 40-90% (model-dependent) | 60-95% (more stable) |
| **Speedup** | 2-3x typical | 3-6x typical |
| **Complexity** | Low | Medium (feature extraction) |
| **Model Requirements** | Two separate LLMs | Target model + draft head |

---

## Related Methods

### 1. Medusa (2024)
**Paper:** [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- **Authors:** Cai et al.
- **GitHub:** [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)
- **Speedup:** ~2x

**Idea:** Multiple decoding heads on the **same model**, predicting multiple tokens in parallel. No separate draft model needed.

### 2. Hydra (2024)
**Paper:** [arXiv:2402.05109](https://arxiv.org/abs/2402.05109)
- **Authors:** Ankner et al.
- **Speedup:** +1.31x over Medusa

**Idea:** Sequentially-dependent draft heads improve accuracy.

### 3. SpecTr (2023)
**Paper:** [arXiv:2310.15141](https://arxiv.org/abs/2310.15141)
- **Authors:** Sun et al. (NeurIPS 2023)

**Idea:** Reformulates speculative decoding as an **optimal transport problem** for better draft-target matching.

---

## References

### Foundational Papers
- [Accelerating LLM Decoding with Speculative Sampling (Chen et al., 2023)](https://arxiv.org/abs/2302.01318) - 642 citations
- [Fast Inference via Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192) - ICML 2023, 1,122 citations

### Related Works
- [Medusa: Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) - Cai et al., 2024
- [Hydra: Sequentially-Dependent Heads](https://arxiv.org/abs/2402.05109) - Ankner et al., 2024
- [SpecTr: Optimal Transport for Speculative Decoding](https://arxiv.org/abs/2310.15141) - Sun et al., NeurIPS 2023

### Resources
- [HuggingFace Assisted Generation Blog](https://huggingface.co/blog/assisted-generation) - May 2023
- [Universal Assisted Generation Blog](https://huggingface.co/blog/universal_assisted_generation) - October 2024
- [vLLM Speculative Decoding Docs](https://docs.vllm.ai/en/latest/features/spec_decode/)

### Survey Papers
- [A Comprehensive Survey of Speculative Decoding (Xia et al., ACL Findings 2024)](https://aclanthology.org/2024.findings-acl.456.pdf) - 198 citations
- [Decoding Speculative Decoding (He et al., NAACL 2025)](https://aclanthology.org/2025.naacl-long.328.pdf)

---

## Summary

**Token-level speculative decoding** is the standard approach that:
1. Uses a small draft model to predict tokens sequentially
2. Verifies all draft tokens in parallel with the target model
3. Achieves 2-3x speedup with good draft-target alignment
4. Is simple to implement and widely supported (HuggingFace, vLLM)

**Key advantage:** Simplicity - just two models running standard generation.

**Key limitation:** Distribution mismatch between draft and target reduces acceptance rate.

This approach works best when:
- Draft and target models are from the same family
- You have a good quality draft model available
- You want a simple, production-ready solution

For higher speedups (3-6x), see **feature-level speculative decoding** and **hybrid approaches**.

---

**Last Updated:** January 2026
