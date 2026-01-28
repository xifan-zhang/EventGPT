# PipeSpec: Breaking Stage Dependencies in Hierarchical LLM Decoding

**Paper**: arXiv:2505.01572v1
**Authors**: Bradley McDanel (Franklin and Marshall College), Sai Qian Zhang, Yunhai Hu (NYU), Zining Liu (UPenn)
**PDF**: `../cascaded_SD/pdf/PipeSpec_hierarchical_LLM_decoding.pdf`

---

## Core Problem with Standard SD

Traditional speculative decoding has two key limitations:

1. **Synchronous Execution**: Draft and verify stages operate in strict lockstep - draft model must wait for verification before generating new tokens
2. **Misprediction Penalty**: When a token is rejected, all subsequent draft tokens are invalidated and must be discarded

```
Standard SD: M0 drafts → waits → M2 verifies → M0 drafts → waits...
             [======]  [idle]  [=========]   [======]  [idle]

Problem: Alternating idle periods waste hardware resources
```

---

## PipeSpec Solution: Asynchronous k-Model Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPESPEC k-MODEL PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   M0 (smallest)     M1 (medium)      ...      Mk (target)      │
│   10 tok/unit       5 tok/unit               1 tok/unit        │
│                                                                 │
│   ┌─────┐          ┌─────┐                  ┌─────┐            │
│   │Draft│ ──────►  │Verify│ ──────► ... ►  │Verify│            │
│   │     │          │+Draft│                 │Final │            │
│   └─────┘          └─────┘                  └─────┘            │
│      │                │                        │                │
│      ▼                ▼                        ▼                │
│   Continuous       Continuous              Continuous           │
│   Generation       Generation              Verification         │
│                                                                 │
│   ◄──────────── Rollback on rejection ────────────►            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Innovations

1. **Asynchronous Execution**: Models run independently in producer-consumer pairs
2. **Optimistic Generation**: Draft models assume downstream acceptance, continue generating
3. **Rollback Cascade**: Rejection triggers targeted rollback to earlier stages
4. **Hierarchical Refinement**: Intermediate models filter low-quality drafts

---

## How PipeSpec Works

### Comparison of Decoding Approaches (Figure 2)

| Method | Configuration | Throughput | Mechanism |
|--------|---------------|------------|-----------|
| Autoregressive | M2 only | 1.0 tok/unit | Sequential, one token at a time |
| Standard SD | {M0, M2} | 1.5 tok/unit | Draft-verify lockstep, idle periods |
| PipeSpec (2 models) | {M0, M2} | 1.75 tok/unit | Async eliminates waiting |
| PipeSpec (3 models) | {M0, M1, M2} | **2.25 tok/unit** | Hierarchical + async |

### Why PipeSpec is Faster

**Standard SD**:
```
Time 0: M0 drafts tokens 0-3
Time 1: M2 verifies 0-3, M0 IDLE (waiting)
Time 2: M0 drafts tokens 4-7
Time 3: M2 verifies 4-7, M0 IDLE (waiting)
```

**PipeSpec**:
```
Time 0: M0 drafts 0-3
Time 1: M0 drafts 4-7  |  M2 verifies 0-3 (parallel!)
Time 2: M0 drafts 8-11 |  M2 verifies 4-7 (parallel!)
        No idle periods - continuous execution
```

---

## Algorithm

```python
def pipespec_decode(models, input_prompt):
    """
    Algorithm 1: Pipelined Speculative Decoding

    models: [M0, M1, ..., MK] ordered by increasing size
    """
    # Each model maintains its own output buffer
    O = [[] for _ in range(K+1)]

    while not finished:
        # All models run in parallel
        for i in range(K+1):
            # Check for rejection from downstream
            if received_rejection_from_stage(j > i):
                rollback(O[i], to_match=O[j])

            if i == 0:
                # First model: generate drafts
                token = M[0].generate_next()
                O[0].append(token)
            else:
                # Other models: verify and generate
                draft_tokens = get_from(O[i-1])
                predictions = M[i].generate()

                # Accept matching tokens
                for token in draft_tokens:
                    if token matches predictions:
                        O[i].append(token)
                    else:
                        signal_rejection_to_earlier_stages()
                        break

        if end_token in O[K]:
            break

    return O[K]
```

---

## Theoretical Framework

### Token Generation Rate

For model Mi with acceptance rate α and window size γ:

**Expected tokens per decoding step**:
```
E(N(Mi)) = (1 - ρi) × 1 + ρi × (1 - α^(γi+1)_(i-1,i)) / (1 - α_(i-1,i))

Where:
- ρi = probability Mi performs verification (vs single token generation)
- α_(i-1,i) = acceptance rate between Mi-1 and Mi
- γi = token window size for Mi
```

### Steady-State Verification Probability

```
ρi = α_(i-1,i) / (1 - α^(γi+1)_(i-1,i) + α_(i-1,i))
```

This converges as n → ∞, representing the long-run probability of verification.

### Theorem 1: Guaranteed Improvement

**For any 0 < α < 1 and γ > 0**:
```
PipeSpec(P) = (1 - ρk) × 1 + ρk × (1 - α^(γk+1)_(k-1,k)) / (1 - α_(k-1,k)) > 1
```

**Proof**: Since α > 0 and γ > 0, the expected tokens per step is always > 1.

PipeSpec is **guaranteed better than autoregressive** for any non-zero acceptance rate.

### PipeSpec vs Standard SD

**Standard SD throughput**:
```
SD(Pa) = (1 - α^(γt+1)_(d,t)) / ((1 - α_(d,t)) × (γt/c_(d,t) + 1))

Note: Includes waiting time γt/c_(d,t) for draft generation
```

**PipeSpec throughput** (no waiting):
```
PipeSpec(Pa) ≈ (1 - α^(γt+1)_(d,t)) / (1 - α_(d,t))
```

**Key insight**: PipeSpec eliminates the draft waiting term, giving strictly better performance.

---

## Experimental Results

### Setup

- **Hardware**: 4× NVIDIA A100-40GB, NVLink interconnect
- **Models**: LLaMA-2 (7B, 13B), LLaMA-3 (1B, 8B, 70B)
- **Benchmarks**: CNN/DM, XSum, HumanEval
- **Settings**: Greedy decoding (temp=0), max 512 tokens

### Main Results (Table 2)

| Dataset | Method | Models | Time (ms/tok) | Speedup |
|---------|--------|--------|---------------|---------|
| **CNN/DM** | AR Baseline | LLaMA2-7B | 20.44 | 1.00× |
| | Speculative | 68M, 7B | 15.12 | 1.35× |
| | LayerSkip | LLaMA2-7B | – | 1.86× |
| | **PipeSpec** | 68M, 7B | 14.56 | 1.40× |
| **CNN/DM** | AR Baseline | LLaMA2-13B | 30.02 | 1.00× |
| | Speculative | 68M, 7B, 13B | 20.66 | 1.45× |
| | LayerSkip | LLaMA2-13B | – | 1.81× |
| | **PipeSpec** | 68M, 7B, 13B | 15.54 | **1.93×** |
| **XSum** | AR Baseline | LLaMA2-13B | 30.26 | 1.00× |
| | Speculative | 68M, 7B, 13B | 21.32 | 1.42× |
| | **PipeSpec** | 68M, 7B, 13B | 15.13 | **2.00×** |
| **HumanEval** | AR Baseline | LLaMA3.1-70B | 123.69 | 1.00× |
| | Speculative | 8B, 70B | 93.42 | 1.32× |
| | Speculative | 1B, 8B, 70B | 90.14 | 1.37× |
| | PipeSpec | 8B, 70B | 54.52 | 2.27× |
| | **PipeSpec** | 1B, 8B, 70B | 48.76 | **2.54×** |

### Ablation Study (Table 1)

Impact of async execution vs hierarchical models on LLaMA3.1-70B:

| Execution Mode | Single Draft | Multi-Draft |
|----------------|--------------|-------------|
| Synchronous | 1.32× | 1.37× |
| **Asynchronous** | 2.27× | **2.54×** |

**Key finding**:
- Async contributes +0.95× improvement (1.32× → 2.27×)
- Multi-draft contributes +0.27× improvement (2.27× → 2.54×)
- **Asynchronous execution is the primary speedup driver!**

---

## Token Acceptance Analysis (Figure 3)

### Distribution Patterns

**Standard SD**:
- Fixed spike at 8 tokens (rigid window size)
- Must alternate between drafting and verifying batches of 8

**PipeSpec**:
- Long-tail distribution extending to 30+ tokens
- Flexible acceptance based on actual agreement
- Spike at 6 tokens in 3-model setup (pipeline stage optimization)

### Implications

PipeSpec can:
1. **Capitalize on "easy" sequences** where models agree → larger batches
2. **Recover quickly** through pipeline when predictions diverge
3. **Adapt naturally** to varying prediction quality

---

## Lookahead Window Analysis (Figure 4)

| Lookahead Size | SD Performance | PipeSpec Performance |
|----------------|----------------|----------------------|
| 1-5 tokens | Poor (insufficient batching) | Good (continuous pipeline) |
| 5-10 tokens | Optimal | Degrading |
| 10+ tokens | Poor (speculation waste) | Poor (verification wait) |

**Optimal settings**:
- **Standard SD**: Lookahead = 8 tokens
- **PipeSpec**: Lookahead = 0 (minimal)

---

## Resource Utilization (Figure 5)

| Method | Avg GPU Utilization | Avg Power | Energy/Token |
|--------|---------------------|-----------|--------------|
| Autoregressive | 37.2% | 134.7W | 16.5 J |
| Standard SD | 23.0% | 113.5W | 9.6 J |
| **PipeSpec** | **39.7%** | 139.6W | **5.8 J** |

**Key insights**:
- Standard SD has **lower** utilization than AR due to idle periods
- PipeSpec achieves **highest** utilization through continuous execution
- PipeSpec is **3× more energy efficient** than autoregressive

---

## Application to EventGPT → VideoLLaVA

### Direct Mapping

```
PipeSpec LLM Pipeline:
    M0 (1B)      →    M1 (8B)     →    M2 (70B)
    fast, draft       medium           slow, target
    10 tok/unit       5 tok/unit       1 tok/unit

Cross-Modal Pipeline:
    EventGPT     →    FrameEnc    →    VideoLLaVA
    sparse, fast      medium           dense, slow
    ~2ms              ~10ms            ~50ms
```

### Why This Mapping Works

| PipeSpec Concept | Cross-Modal Application |
|------------------|-------------------------|
| Asynchronous execution | Event encoding runs while video verifies |
| Producer-consumer pairs | Event→Frame, Frame→Video relationships |
| Rollback on rejection | Discard event drafts when video rejects |
| Intermediate filtering | Frame encoder catches obvious mismatches |
| Progressive refinement | Sparse → Medium → Dense information |

### Expected Benefits

1. **Hide encoding latency**: Event encoder runs continuously
2. **Early rejection**: Frame encoder filters bad drafts cheaply (~10ms vs ~50ms)
3. **Higher throughput**: No idle periods in vision pipeline
4. **Real-time capable**: Continuous execution suits event camera streams

### Proposed Architecture

```python
class CrossModalPipeSpec:
    """
    Asynchronous pipeline for event → frame → video.
    """
    def __init__(self):
        self.event_encoder = EventGPT()      # M0: fast draft
        self.frame_encoder = FrameEncoder()   # M1: intermediate
        self.video_llava = VideoLLaVA()       # M2: target

        # Output buffers for each stage
        self.O_event = Queue()
        self.O_frame = Queue()
        self.O_video = Queue()

    def run_pipeline(self, events, frames, video):
        """
        All three stages run asynchronously.
        """
        # Stage 0: Event drafting (continuous)
        async def event_stage():
            while not done:
                if received_rejection():
                    rollback(self.O_event)
                features = self.event_encoder(events)
                self.O_event.put(features)

        # Stage 1: Frame verification + drafting (continuous)
        async def frame_stage():
            while not done:
                if received_rejection():
                    rollback(self.O_frame)
                event_features = self.O_event.get()
                frame_features = self.frame_encoder(frames)
                if aligned(event_features, frame_features):
                    self.O_frame.put(frame_features)
                else:
                    signal_rejection_to_event_stage()

        # Stage 2: Video verification (continuous)
        async def video_stage():
            while not done:
                frame_features = self.O_frame.get()
                video_features = self.video_llava.encode(video)
                if aligned(frame_features, video_features):
                    self.O_video.put(video_features)
                else:
                    signal_rejection_to_frame_stage()

        # Run all stages in parallel
        await asyncio.gather(event_stage(), frame_stage(), video_stage())
        return self.O_video
```

### Expected Performance

| Stage | Latency | Throughput (standalone) | Throughput (pipelined) |
|-------|---------|-------------------------|------------------------|
| Event | 2ms | 500 feat/s | 500 feat/s |
| Frame | 10ms | 100 feat/s | ~100 feat/s |
| Video | 50ms | 20 feat/s | **~45 feat/s** |

**Pipelined speedup**: ~2.25× on video verification stage

---

## Key Takeaways

### For Research

1. **Asynchronous execution is the primary speedup driver** - more important than model count
2. **Pipeline depth correlates with efficiency** - 3 models > 2 models
3. **Optimal lookahead differs**: SD needs 8 tokens, PipeSpec needs 0
4. **GPU utilization improves** with pipelining (39.7% vs 23.0%)
5. **Energy efficiency**: 3× better than autoregressive

### For EventGPT Application

1. **Async pipelining** naturally fits event camera's continuous stream
2. **Frame encoder as intermediate** provides cheap early rejection
3. **Rollback mechanism** handles cross-modal misalignment
4. **No idle periods** - all encoders run continuously
5. **Scalable** - can add more intermediate stages if needed

---

## Limitations

1. **Static configuration**: Fixed pipeline depth, no dynamic adaptation
2. **Rollback cost**: Frequent mispredictions trigger expensive cascades
3. **Memory requirements**: All models must fit in available GPU memory
4. **Higher power draw**: Continuous parallel execution increases power consumption

---

## Comparison with Other Cascaded SD Methods

| Method | Models | Async | Training | Speedup | Key Idea |
|--------|--------|-------|----------|---------|----------|
| Standard SD | 2 | No | None | 1.3-1.5× | Draft-verify |
| PyramidSD | 3 | No | None | 1.91× | Entropy gradient |
| HiSpec | 2 + EE | No | Partial | 2.01× | Early-exit verification |
| **PipeSpec** | k | **Yes** | None | **2.54×** | Async pipeline |

**PipeSpec's unique contribution**: Breaking synchronization barriers through true asynchronous execution.

---

## References

- Leviathan et al., 2023 - Fast Inference via Speculative Decoding
- EAGLE (Li et al., 2024) - Feature uncertainty for speculation
- LayerSkip (Elhoushi et al., 2024) - Early-exit + self-speculative
- Draft&Verify (Zhang et al., 2023) - Self-speculative decoding
- TRIFORCE (Sun et al., 2024) - Hierarchical SD for long sequences

---

**Document Created**: January 28, 2026
**Source**: arXiv:2505.01572v1
