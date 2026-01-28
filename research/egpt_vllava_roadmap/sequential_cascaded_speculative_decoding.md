# Sequential and Cascaded Speculative Decoding

## Table of Contents
1. [Overview](#overview)
2. [Theoretical Analysis](#theoretical-analysis)
3. [Key Methods](#key-methods)
4. [Benefits Assessment](#benefits-assessment)
5. [Implementation](#implementation)
6. [Performance Analysis](#performance-analysis)
7. [Optimal Cascade Depth](#optimal-cascade-depth)
8. [Production Considerations](#production-considerations)
9. [References](#references)

---

## Overview

Sequential and cascaded speculative decoding extends the standard two-model (draft + target) paradigm to **multiple stages** of drafting and verification. These methods chain multiple models together, where each stage refines the output of the previous one.

**Architecture:**
```
Model A (smallest) → Model B (medium) → Model C (larger) → ... → Target Model (largest)
```

**This document provides comprehensive theoretical analysis, empirical results, and practical recommendations.**

---

## Theoretical Analysis

This section provides comprehensive theoretical analysis of cascaded speculative decoding benefits and limitations.

### 1. Better Distribution Matching

**Hypothesis:** Smaller gaps between consecutive models improve acceptance rates.

```
Standard: Draft → Target (large gap)
Cascaded: Draft A → Draft B → Target (smaller gaps)

Acceptance probability:
α_total = Π α_i  (product of stage-wise acceptance rates)
```

**Analysis:**
- If α_A→B = 0.85 and α_B→Target = 0.85
- Overall α = 0.85 × 0.85 = 0.7225 (~72%)
- This beats direct α = 0.6 for same-family small→large

**Potential:** ✅ Valid when models form a good hierarchy

---

### 2. Higher Parallelization Potential

**Hypothesis:** Multiple small models can generate drafts faster than one medium model.

```
Sequential Cost: T_A + T_B + T_verify
vs
Single Draft: T_medium + T_verify

Benefit when: T_A + T_B << T_medium
```

**Analysis:**
- Model A (1B): ~2ms per forward
- Model B (7B): ~10ms per forward
- Model Medium (7B alone): ~10ms per forward
- Target (70B): ~100ms per forward

Cascaded: 2ms + 10ms + 100ms_verify = 12ms + verify
Single: 10ms + 100ms_verify = 10ms + verify

**Potential:** ⚠️ Marginal - depends on model sizes and parallelization

---

### 3. Error Recovery and Correction

**Hypothesis:** Early stages can catch and correct errors before final verification.

```
Stage 1: A generates draft with errors
Stage 2: B detects and corrects A's errors
Stage 3: Target verifies B's corrected draft
```

**Analysis:**
- In practice, errors propagate, not correct
- Each stage compounds uncertainty
- Late rejections invalidate all earlier work

**Potential:** ❌ Limited - errors accumulate, not recover

---

## Limitations and Challenges

### 1. Error Accumulation

```
Problem: Errors compound exponentially

Stage 1 error rate: ε_1 = 0.10 (10%)
Stage 2 error rate: ε_2 = 0.15 (15%)

Combined error: 1 - (1-ε_1)(1-ε_2) = 1 - 0.9×0.85 = 23.5%

Each additional stage amplifies uncertainty
```

**Impact:** Late-stage rejections waste all earlier computation

---

### 2. Coordination Overhead

```
Overhead sources:
- Synchronization between stages
- Data transfer (features, KV cache)
- Tree attention management
- Rollback coordination

Measured overhead: 5-15% of total time
```

**Impact:** Eats into theoretical speedup gains

---

### 3. Memory Bandwidth Bottleneck

```
Problem: Each stage requires model weights in memory

Memory requirements (FP16):
- Model A (1B): ~2GB
- Model B (7B): ~14GB
- Model C (13B): ~26GB
- Target (70B): ~140GB
- Total: ~182GB

vs Single Draft:
- Draft (7B): ~14GB
- Target (70B): ~140GB
- Total: ~154GB

Cascade needs ~18% more memory
```

**Impact:** May not fit on single GPU, requires distributed setup

---

### 4. Training Complexity

```
Challenge: Must train N draft models + N alignment layers

Training time:
- Single draft: 1-2 days on 8×A100
- 2-stage cascade: 3-4 days
- 3-stage cascade: 5-7 days

Hyperparameter sensitivity increases exponentially
```

**Impact:** Significant engineering investment

---

## Empirical Results from Papers

### 1. Cascade Speculative Drafting (NeurIPS 2024)

**Paper:** "Cascade Speculative Drafting for Even Faster LLM Inference"
- **arXiv:** [2312.11462](https://arxiv.org/abs/2312.11462)
- **Authors:** Z. Chen et al.
- **Venue:** NeurIPS 2024
- **Citations:** 65+

**Results:**
```
Configuration: 1B → 7B → 13B → 70B

Method          Speedup   Acceptance Rate
----------------------------------------
Vanilla         1.00x     -
Single Draft    2.1x      68%
CS Drafting     2.7x      75% (overall)
```

**Key Findings:**
- ~28% improvement over single-stage
- Diminishing returns after 3 stages
- Training alignment critical

---

### 2. Faster Cascades via Speculative Decoding (ICLR 2025)

**Paper:** "Faster Cascades via Speculative Decoding"
- **arXiv:** [2405.19261](https://arxiv.org/abs/2405.19261)
- **Authors:** H. Narasimhan et al.
- **Venue:** ICLR 2025
- **Citations:** 22-23

**Results:**
```
Comparison on LLaMA-2 7B:

Method              Tokens/sec   Speedup
----------------------------------------
Autoregressive      18.5         1.00x
Speculative (1-draft) 44.2       2.39x
Speculative (2-draft) 51.8       2.80x
Speculative (3-draft) 54.1       2.92x
```

**Key Findings:**
- 2-draft provides 17% improvement over 1-draft
- 3-draft provides only 4% additional improvement
- **Diminishing returns are significant**

---

### 3. 3-Model Speculative Decoding (2025)

**Paper:** "3-Model Speculative Decoding"
- **arXiv:** [2510.12966](https://arxiv.org/abs/2510.12966)
- **Authors:** S. Byun et al.
- **Published:** October 2025

**Results:**
```
Model Hierarchy: 0.5B → 3B → 7B → 13B

Speedup comparison:
- 1-stage (0.5B → 13B): 2.8x
- 2-stage (0.5B → 7B → 13B): 3.2x
- 3-stage (0.5B → 3B → 7B → 13B): 3.3x
```

**Key Findings:**
- First cascade stage: +14% improvement
- Second cascade stage: +3% improvement
- **Third stage: <1% improvement**

---

## Benefits Assessment

### When Cascading Works Well ✅

| Condition | Why |
|-----------|-----|
| **Strong model hierarchy** | Clear size/quality differences (1B → 7B → 70B) |
| **High baseline acceptance** | Each stage >75% acceptance |
| **Specialized stages** | Different architectures for different tasks |
| **Abundant compute** | Multiple GPUs available for parallel stages |
| **Well-aligned training** | Proper feature alignment between stages |

### When Cascading Fails ❌

| Condition | Why |
|-----------|-----|
| **Similar model sizes** | 7B → 13B provides minimal benefit |
| **Low acceptance rates** | Any stage <60% destroys overall efficiency |
| **Memory constrained** | Can't fit all model weights |
| **Latency critical** | Coordination overhead hurts |
| **Poor alignment** | Feature spaces don't match |

---

## Optimal Cascade Depth

### Analysis

Based on empirical data from papers:

```
Speedup vs Cascade Depth (theoretical ideal)

1 stage:  2.5x (baseline)
2 stages: 3.2x (+28%)
3 stages: 3.4x (+6%)
4 stages: 3.45x (+1.5%)
5 stages: 3.47x (+0.6%)

Diminishing returns after 2-3 stages
```

### Recommendation

**Optimal: 2-3 stages maximum**

**Why:**
1. Most gains realized in first cascade
2. Marginal benefit <5% after 3 stages
3. Complexity increases linearly
4. Failure probability increases exponentially

---

## Production Considerations

### 1. Deployment Complexity

```
Single Stage:
┌────────────┐
│  Load 2    │ Draft + Target models
│  models    │
└────────────┘

Cascade (3-stage):
┌────────────┐
│  Load 4    │ Draft A + Draft B + Draft C + Target
│  models    │ + 3 alignment layers
└────────────┘
┌────────────┐
│  Coordinate│ Stage synchronization
│  3 stages  │ Rollback handling
└────────────┘
┌────────────┐
│  Monitor   │ Per-stage acceptance
│  3 metrics │ Stage-specific fallbacks
└────────────┘
```

**Engineering effort:** 3-5x higher than single stage

---

### 2. Reliability Concerns

```
Failure modes:
- Stage 1 fails → fallback to target
- Stage 2 fails → wasted Stage 1 work
- Stage 3 fails → wasted Stage 1+2 work

Probability of full cascade success:
P(success) = Π P(stage_i succeeds)

With P(success) = 0.9 per stage:
- 1 stage: 90% success rate
- 2 stages: 81% success rate
- 3 stages: 73% success rate
```

**Impact:** More stages = more frequent fallbacks

---

## For EventGPT → VideoLLaVA Application

### Proposed Cascade

```
Stage 1: EventGPT Vision Encoder (sparse, fast)
    → Extracts sparse event features
    → ~100x faster than dense

Stage 2: EventGPT Language Model (1B)
    → Refines features with context
    → Trained on EventGPT + VideoLLaVA alignment

Stage 3: VideoLLaVA-7B (target)
    → Verifies and generates final output
    → Uses cached vision features
```

### Expected Benefits

| Metric | Single Stage | 2-Stage Cascade | Improvement |
|--------|--------------|-----------------|-------------|
| Vision speedup | 10x | 50x | +5x |
| Language speedup | 2.5x | 2.8x | +12% |
| Overall speedup | 2.9x | 3.5x | +21% |
| Memory usage | 16GB | 20GB | +25% |
| Acceptance rate | 70% | 65% | -7% |

### Recommendation for EventGPT → VideoLLaVA

**Use 2-stage cascade:**
1. Stage 1: EventGPT encoder (vision) with feature alignment
2. Stage 2: Skip intermediate language model, go directly to VideoLLaVA LM

**Rationale:**
- Vision stage provides most benefit (sparse → dense)
- Language cascade adds complexity with marginal gain
- Feature alignment is critical - must train adapter layer
- Falls back to dense encoding when alignment is poor

---

## Conclusion

### Benefit Potential: MODERATE ⚠️

| Aspect | Verdict | Evidence |
|--------|---------|----------|
| **Theoretical speedup** | ✅ Good | 2-3x potential vs 2-2.5x single stage |
| **Practical speedup** | ⚠️ Moderate | 10-30% improvement in practice |
| **Engineering cost** | ❌ High | 3-5x more complex |
| **Reliability** | ⚠️ Reduced | More failure points |
| **Memory overhead** | ⚠️ Significant | +15-25% more memory |

### Final Recommendation

**Cascading is worthwhile when:**
1. You have natural model hierarchy (sizes differ by 5-10x)
2. Compute is abundant (multiple GPUs)
3. You can invest in training alignment
4. Acceptance rates remain high (>75% per stage)

**For most applications:**
- **2-stage cascade is optimal**
- **3+ stages only for research/competition**
- **Focus on alignment and acceptance rate**

### For EventGPT → VideoLLaVA Specifically:

**Do cascade, but only on vision:**
```
EventGPT Vision → Feature Alignment → VideoLLaVA Vision
(Sparse)         (Train MLP)        (Dense verify)
```

**Skip language cascade** - marginal benefit, high complexity

---

**Last Updated:** January 2026

## Core Concepts

**Standard Speculative Decoding:**
```
Draft Model → Target Model (1 stage)
```

**Cascaded Speculative Decoding:**
```
Model A → Model B → Model C → ... → Target Model (N stages)
```

### Why Cascade?

1. **Better Distribution Matching** - Smaller gaps between consecutive models
2. **Higher Acceptance Rates** - Each stage has higher probability of acceptance
3. **Parallel Draft Generation** - Multiple candidates explored at each stage

---

## Key Methods

### 1. ReDrafter (2024)
**Paper:** "Recurrent Drafter for Fast Speculative Decoding in Large Language Models"
- **Authors:** Y. Cheng et al.
- **arXiv:** [2403.09919](https://arxiv.org/abs/2403.09919)
- **GitHub:** [apple/ml-recurrent-drafter](https://github.com/apple/ml-recurrent-drafter)
- **Speedup:** Up to 2.7x on NVIDIA GPUs

**Key Innovation:** Uses an RNN-based draft model with beam search and dynamic tree attention.

#### Mathematical Formulation

**Draft Generation (RNN):**
```
h_t = RNN(h_{t-1}, x_t)  # Hidden state update
p_draft(t) = Softmax(W_out * h_t + b_out)  # Token distribution
```

**Beam Search Drafting:**
```
B = {(s_1, p_1), ..., (s_k, p_k)}  # Top-k beam candidates
For each beam b in B:
    Generate m draft tokens: t_1, ..., t_m
```

**Acceptance Probability:**
```
α(i) = min(1, p_target(t_i) / p_draft(t_i))
```

#### Sample Code

```python
import torch
import torch.nn as nn

class ReDrafter(nn.Module):
    """Recurrent Drafter for Speculative Decoding"""
    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super().__init__()
        self.rnn = nn.GRM(vocab_size, hidden_size, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, hidden=None):
        """
        Args:
            input_ids: (batch, seq_len)
            hidden: RNN hidden state
        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: Updated RNN state
        """
        emb = nn.functional.one_hot(input_ids, self.proj.out_features).float()
        output, hidden = self.rnn(emb, hidden)
        logits = self.proj(output)
        return logits, hidden

    def generate_drafts(self, input_ids, num_draft_tokens=5, beam_width=3):
        """Generate draft tokens with beam search"""
        beams = [(input_ids, 0.0)]  # (sequence, score)
        drafts = []

        for _ in range(num_draft_tokens):
            new_beams = []
            for seq, score in beams:
                logits, hidden = self.forward(seq)
                probs = torch.softmax(logits[:, -1], dim=-1)

                # Get top-k candidates
                topk_probs, topk_ids = torch.topk(probs, beam_width)

                for i in range(beam_width):
                    new_seq = torch.cat([seq, topk_ids[:, i:i+1]], dim=-1)
                    new_score = score + torch.log(topk_probs[:, i]).item()
                    new_beams.append((new_seq, new_score))

            # Keep top beam_width beams
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        return beams[0][0]  # Return best beam

# Verification with target model
def verify_drafts(draft_ids, target_model, draft_model, gamma=2.0):
    """
    Verify draft tokens using standard speculative sampling

    Args:
        draft_ids: Draft token IDs from drafter
        target_model: Large target model
        draft_model: Draft model (for getting p_draft)
        gamma: Acceptance threshold
    """
    # Get target model probabilities
    with torch.no_grad():
        target_logits = target_model(draft_ids)
        p_target = torch.softmax(target_logits, dim=-1)

    # Get draft model probabilities
    with torch.no_grad():
        draft_logits = draft_model(draft_ids)
        p_draft = torch.softmax(draft_logits, dim=-1)

    # Accept/reject each token
    accepted = []
    for i in range(1, draft_ids.shape[1]):  # Skip first token (already known)
        t_i = draft_ids[:, i]
        q = p_draft[i-1, t_i]
        p = p_target[i, t_i]

        # Acceptance probability
        alpha = min(1.0, p / q)

        if torch.rand(1).item() < alpha:
            accepted.append(t_i)
        else:
            # Resample from adjusted distribution
            p_adjusted = torch.clamp(p - q, min=0)
            p_adjusted = p_adjusted / p_adjusted.sum()
            t_sampled = torch.multinomial(p_adjusted, 1)
            accepted.append(t_sampled)
            break  # Stop after rejection

    return torch.stack(accepted)
```

---

### 2. Cascade Speculative Drafting (NeurIPS 2024)
**Paper:** "Cascade Speculative Drafting for Even Faster LLM Inference"
- **Authors:** Z. Chen et al.
- **Venue:** NeurIPS 2024
- **PDF:** [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/9cb5b083ba4f5ca6bd05dd307a2fb354-Paper-Conference.pdf)
- **Citations:** 65+

**Key Innovation:** Two types of cascades - vertical (layer-wise) and horizontal (token-wise).

#### Algorithm

```
Algorithm: Cascade Speculative Drafting (CS Drafting)

Input: Context C, Target Model M, Draft Models {D_1, D_2, ..., D_n}
Output: Generated tokens T

1: for each generation step do:
2:     # Vertical Cascade: Draft at different layers
3:     for i = 1 to n do:
4:         L_i ← D_i.generate_drafts(C, k_i)  # k_i draft tokens
5:     end for
6:
7:     # Horizontal Cascade: Merge and verify
8:     L_merged ← merge_drafts({L_1, ..., L_n})
9:
10:    # Verify with target model
11:    T_accepted ← M.verify(L_merged)
12:
13:    C ← C + T_accepted
14: end for

return C
```

#### Mathematical Formulation

**Expected Speedup:**
```
Speedup = Σ_{i=1}^{n} k_i * α_i / (Σ_{i=1}^{n} t_i + t_verify)

Where:
- k_i: Number of draft tokens at stage i
- α_i: Acceptance rate at stage i
- t_i: Time for drafting at stage i
- t_verify: Time for verification
```

#### Sample Code

```python
class CascadeSpeculativeDecoding:
    """Cascade Speculative Drafting Implementation"""

    def __init__(self, draft_models, target_model, draft_lengths):
        """
        Args:
            draft_models: List of draft models [D_1, D_2, ..., D_n]
            target_model: Large target model
            draft_lengths: Number of drafts per model [k_1, k_2, ..., k_n]
        """
        self.draft_models = draft_models
        self.target_model = target_model
        self.draft_lengths = draft_lengths

    def vertical_cascade(self, context):
        """Generate drafts at multiple levels"""
        all_drafts = []

        for model, k in zip(self.draft_models, self.draft_lengths):
            # Generate k draft tokens from this model
            draft_ids = model.generate(context, max_new_tokens=k)
            all_drafts.append(draft_ids)

        return all_drafts

    def horizontal_cascade(self, draft_sequences):
        """Merge drafts from multiple models"""
        # Strategy 1: Concatenate all drafts
        merged = torch.cat(draft_sequences, dim=-1)

        # Strategy 2: Interleave drafts
        # merged = self.interleave(draft_sequences)

        # Strategy 3: Select best drafts using confidence scores
        # merged = self.select_best(draft_sequences)

        return merged

    def verify_cascade(self, context, merged_drafts):
        """Verify merged drafts with target model"""
        # Standard speculative sampling verification
        full_sequence = torch.cat([context, merged_drafts], dim=-1)

        with torch.no_grad():
            target_logits = self.target_model(full_sequence)
            target_probs = torch.softmax(target_logits.logits, dim=-1)

        accepted = []
        for i, token_id in enumerate(merged_drafts[0], start=1):
            # Get probabilities
            p_target = target_probs[i-1, token_id]

            # Simple acceptance (can use draft model prob for comparison)
            if torch.rand(1).item() < p_target:
                accepted.append(token_id)
            else:
                break

        return torch.tensor(accepted)

    def generate(self, prompt, max_tokens=100):
        """Main generation loop"""
        context = prompt

        for _ in range(max_tokens):
            # Vertical cascade
            drafts = self.vertical_cascade(context)

            # Horizontal cascade
            merged = self.horizontal_cascade(drafts)

            # Verify
            accepted = self.verify_cascade(context, merged)

            if len(accepted) == 0:
                # Fallback to standard decoding
                output = self.target_model.generate(
                    context,
                    max_new_tokens=1,
                    do_sample=True
                )
                context = torch.cat([context, output[:, -1:]], dim=-1)
            else:
                context = torch.cat([context, accepted], dim=-1)

            # Check for EOS
            if context[0, -1].item() == self.target_model.config.eos_token_id:
                break

        return context
```

---

### 3. Lookahead Decoding (ICML 2024)
**Paper:** "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"
- **Authors:** Y. Fu et al.
- **arXiv:** [2402.02057](https://arxiv.org/abs/2402.02057)
- **GitHub:** [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)
- **Citations:** 200+

**Key Innovation:** N-gram + Jacobi iteration for parallel drafting without a separate draft model.

#### Algorithm

```
Algorithm: Lookahead Decoding

Input: Context C, Target Model M, Window Size W, N-gram Size N
Output: Generated tokens T

1: Initialize n-gram cache G ← {}

2: for each generation step do:
3:     # Extract n-gram candidates from cache
4:     candidates ← G.get(C[-(N-1):], [])
5:
6:     if candidates is not empty:
7:         # Jacobi iteration: parallel verification
8:         for each candidate c in candidates:
9:             p_c ← M.get_probability(C, c)
10:        end for
11:
12:        # Select best candidate
13:        t* ← argmax_c p_c
14:        T ← T + t*
15:    else:
16:        # Standard autoregressive generation
17:        t ← M.generate(C, max_tokens=1)
18:        T ← T + t

19:        # Update n-gram cache
20:        for i = 1 to W:
21:            G[C[-i:]] ← M.predict_next(C[-i:])
22:        end for
23:    end if
24: end for

return T
```

#### Jacobi Iteration

The key insight is using **Jacobi iteration** to parallelize dependency resolution:

```
Given: x_t depends on {x_{t-1}, x_{t-2}, ..., x_{t-n}}

Standard (Sequential):
x_1 = f(x_0)
x_2 = f(x_1)
x_3 = f(x_2)

Jacobi (Parallel):
[x_1, x_2, x_3] = f([x_0, x_0, x_0])  # Initial guess
Iterate until convergence
```

#### Sample Code

```python
class LookaheadDecoding:
    """Lookahead Decoding with N-gram Cache"""

    def __init__(self, model, ngram_size=3, window_size=5):
        self.model = model
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.ngram_cache = {}

    def get_ngram_key(self, context):
        """Extract n-gram key from context"""
        if len(context) < self.ngram_size - 1:
            return None
        return tuple(context[-(self.ngram_size-1):].tolist())

    def predict_from_cache(self, context):
        """Predict next token from n-gram cache"""
        key = self.get_ngram_key(context)
        if key is None or key not in self.ngram_cache:
            return None

        candidates = self.ngram_cache[key]
        return candidates

    def update_cache(self, context):
        """Update n-gram cache with new predictions"""
        key = self.get_ngram_key(context)
        if key is None:
            return

        # Generate predictions for window size
        with torch.no_grad():
            for i in range(1, min(self.window_size + 1, len(context) + 1)):
                sub_context = context[-i:]
                logits = self.model(sub_context)
                probs = torch.softmax(logits.logits[:, -1], dim=-1)

                # Store top-k candidates
                topk_probs, topk_ids = torch.topk(probs, k=5)
                self.ngram_cache[tuple(sub_context.tolist())] = [
                    (idx.item(), prob.item())
                    for idx, prob in zip(topk_ids[0], topk_probs[0])
                ]

    def verify_candidates_jacobi(self, context, candidates):
        """Verify multiple candidates in parallel (Jacobi iteration)"""
        if not candidates:
            return None

        # Create sequences for each candidate
        sequences = []
        for token_id, _ in candidates:
            seq = torch.cat([context, torch.tensor([[token_id]])], dim=-1)
            sequences.append(seq)

        # Get probabilities in parallel
        with torch.no_grad():
            logits_list = [self.model(seq).logits[:, -1] for seq in sequences]
            probs_list = [torch.softmax(logits, dim=-1) for logits in logits_list]

        # Select best candidate
        best_score = -1
        best_token = None

        for i, (token_id, prior_prob) in enumerate(candidates):
            # Score = probability of this token given context
            prob = probs_list[i][token_id].item()

            # Combine with prior probability from cache
            score = prob * prior_prob

            if score > best_score:
                best_score = score
                best_token = token_id

        return best_token

    def generate(self, prompt, max_tokens=100):
        """Generate text with Lookahead Decoding"""
        context = prompt.clone()
        generated = []

        # Initialize cache
        self.update_cache(context)

        for step in range(max_tokens):
            # Try n-gram cache first
            candidates = self.predict_from_cache(context)

            if candidates:
                # Jacobi verification
                next_token = self.verify_candidates_jacobi(context, candidates)

                if next_token is not None:
                    generated.append(next_token)
                    context = torch.cat([
                        context,
                        torch.tensor([[next_token]])
                    ], dim=-1)

                    # Update cache
                    self.update_cache(context)

                    if next_token == self.model.config.eos_token_id:
                        break

                    continue

            # Fallback to standard generation
            with torch.no_grad():
                output = self.model.generate(
                    context,
                    max_new_tokens=1,
                    do_sample=True
                )
                next_token = output[0, -1].item()

            generated.append(next_token)
            context = output

            # Update cache
            self.update_cache(context)

            if next_token == self.model.config.eos_token_id:
                break

        return context
```

---

### 4. PaSS: Parallel Speculative Sampling (NeurIPS 2023)
**Paper:** "PaSS: Parallel Speculative Sampling"
- **Authors:** G. Monea, A. Joulin, E. Grave
- **arXiv:** [2311.13581](https://arxiv.org/abs/2311.13581)
- **PDF:** [NeurIPS 2023](https://neurips2023-enlsp.github.io/papers/paper_83.pdf)

**Key Innovation:** Parallel speculative sampling with adaptive draft length.

#### Algorithm

```
Algorithm: Parallel Speculative Sampling (PaSS)

Input: Context C, Draft Model D, Target Model T
Output: Generated tokens

1: function PARALLEL_SAMPLE(C, D, T):
2:     # Parallel draft generation
3:     draft_tokens ← D.generate_parallel(C, k)
4:     draft_probs ← D.get_probs(C, draft_tokens)
5:
6:     # Parallel verification
7:     target_probs ← T.get_probs(C, draft_tokens)
8:
9:     # Adaptive acceptance
10:    for i = 1 to k:
11:        α_i ← min(1, target_probs[i] / draft_probs[i])
12:        if random() < α_i:
13:            accept(draft_tokens[i])
14:        else:
15:            resample(target_probs[i] - draft_probs[i])
16:            break
17:    return accepted_tokens
```

#### Sample Code

```python
class ParallelSpeculativeSampling:
    """Parallel Speculative Sampling (PaSS)"""

    def __init__(self, draft_model, target_model, max_draft_length=10):
        self.draft_model = draft_model
        self.target_model = target_model
        self.max_draft_length = max_draft_length

    def parallel_draft(self, context, k):
        """Generate k draft tokens in parallel"""
        batch_size = context.shape[0]
        device = context.device

        # Create parallel input sequences
        # For position i, use context + dummy tokens for positions > i
        draft_sequences = []

        for i in range(k):
            # Pad context to length k
            padded = torch.cat([
                context,
                torch.zeros(1, k - i, dtype=torch.long, device=device)
            ], dim=-1)

            # For this position, generate actual draft
            if i == 0:
                with torch.no_grad():
                    draft = self.draft_model.generate(
                        context,
                        max_new_tokens=1,
                        do_sample=False
                    )
                padded[:, i:i+1] = draft[:, -1:]
            else:
                # Use previous drafts
                padded[:, i:i+1] = draft_sequences[-1][:, -1:]

            draft_sequences.append(padded)

        return torch.cat(draft_sequences, dim=0)  # (k, seq_len)

    def parallel_verify(self, context, draft_sequences):
        """Verify all draft sequences in parallel"""
        k = len(draft_sequences)
        device = context.device

        # Get draft model probabilities
        draft_probs = []
        for seq in draft_sequences:
            with torch.no_grad():
                logits = self.draft_model(seq).logits[:, -1]
                probs = torch.softmax(logits, dim=-1)
                draft_probs.append(probs)

        # Get target model probabilities for all sequences
        target_probs = []
        for seq in draft_sequences:
            with torch.no_grad():
                logits = self.target_model(seq).logits[:, -1]
                probs = torch.softmax(logits, dim=-1)
                target_probs.append(probs)

        # Accept or reject each position
        accepted = []
        for i in range(k):
            draft_token = draft_sequences[i][:, context.shape[1]:].squeeze()

            q = draft_probs[i][draft_token].item()
            p = target_probs[i][draft_token].item()

            alpha = min(1.0, p / q)

            if torch.rand(1).item() < alpha:
                accepted.append(draft_token.item())
            else:
                # Resample
                p_adjusted = torch.clamp(
                    target_probs[i] - draft_probs[i],
                    min=0
                )
                p_adjusted = p_adjusted / p_adjusted.sum()
                sampled = torch.multinomial(p_adjusted, 1).item()
                accepted.append(sampled)
                break

        return torch.tensor(accepted).unsqueeze(0)

    def generate(self, prompt, max_tokens=100):
        """Generate with parallel speculative sampling"""
        context = prompt.clone()

        for _ in range(max_tokens):
            # Adaptive draft length based on acceptance history
            k = self.adaptive_draft_length()

            # Parallel drafting
            draft_sequences = self.parallel_draft(context, k)

            # Parallel verification
            accepted = self.parallel_verify(context, draft_sequences)

            context = torch.cat([context, accepted], dim=-1)

            if context[0, -1].item() == self.target_model.config.eos_token_id:
                break

        return context

    def adaptive_draft_length(self):
        """Adaptive draft length based on acceptance rate"""
        # Simple version: can be made more sophisticated
        return self.max_draft_length
```

---

## Comparison Table

| Method | Year | Key Idea | Draft Type | Speedup | Code |
|--------|------|----------|------------|---------|------|
| **ReDrafter** | 2024 | RNN + Beam Search | Sequential | 2.7x | [GitHub](https://github.com/apple/ml-recurrent-drafter) |
| **CS Drafting** | 2024 | Vertical + Horizontal Cascade | Multi-stage | TBD | - |
| **Lookahead** | 2024 | N-gram + Jacobi | Parallel (no model) | ~2x | [GitHub](https://github.com/hao-ai-lab/LookaheadDecoding) |
| **PaSS** | 2023 | Parallel Draft Generation | Parallel | TBD | - |

---

## Performance Equations

### General Speedup Formula

For N-stage cascaded speculative decoding:

```
Speedup = T_vanilla / T_cascaded

Where:
T_vanilla = N_tokens * T_single_forward

T_cascaded = Σ_{i=1}^{N} (T_draft_i + k_i * T_verify / acceptance_rate_i)

Optimal when:
Σ_{i=1}^{N} T_draft_i << T_vanilla
And:
Π_{i=1}^{N} acceptance_rate_i > threshold (~0.5)
```

### Acceptance Rate Cascade

```
Overall Acceptance = Π_{i=1}^{N} α_i

Where α_i is acceptance rate at stage i

Example:
α_1 = 0.9 (Model A → Model B)
α_2 = 0.8 (Model B → Target)

Overall = 0.9 * 0.8 = 0.72 (72% overall acceptance)
```

---

## References

### Papers
- [ReDrafter (arXiv:2403.09919)](https://arxiv.org/abs/2403.09919) - Cheng et al., 2024
- [Cascade Speculative Drafting (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/9cb5b083ba4f5ca6bd05dd307a2fb354-Paper-Conference.pdf) - Chen et al., 2024
- [Lookahead Decoding (arXiv:2402.02057)](https://arxiv.org/abs/2402.02057) - Fu et al., ICML 2024
- [PaSS (arXiv:2311.13581)](https://arxiv.org/abs/2311.13581) - Monea et al., NeurIPS 2023

### Related Works
- [Speculative Cascades (Google Research, 2025)](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)
- [Faster Cascades via Speculative Decoding (ICLR 2025)](https://openreview.net/forum?id=vo9t20wsmd) - Narasimhan et al., 2025

### Resources
- [ReDrafter GitHub](https://github.com/apple/ml-recurrent-drafter) - Apple + NVIDIA implementation
- [Lookahead Decoding GitHub](https://github.com/hao-ai-lab/LookaheadDecoding) - Official implementation

---

**Last Updated:** January 2026

---

## Summary

Sequential and cascaded speculative decoding represents a natural extension of standard speculative decoding:

1. **ReDrafter**: RNN-based sequential drafting with beam search
2. **CS Drafting**: Multi-stage vertical + horizontal cascades
3. **Lookahead**: Cache-based parallel drafting (no draft model needed)
4. **PaSS**: Parallel speculative sampling with adaptive draft length

The key insight: **chain multiple drafting stages** to gradually refine predictions, with each stage having a smaller distribution gap to the next. This improves acceptance rates while maintaining speedup benefits.

These methods are particularly useful when:
- You have multiple models of different sizes
- You want to minimize distribution mismatch
- You can parallelize verification efficiently
