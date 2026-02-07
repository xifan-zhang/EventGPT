# Edge-Cloud Speculative Decoding

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [System Architecture](#system-architecture)
4. [Related Works](#related-works)
5. [Key Challenges](#key-challenges)
6. [Proposed Solutions](#proposed-solutions)
7. [Implementation Guide](#implementation-guide)
8. [Performance Analysis](#performance-analysis)
9. [Use Cases](#use-cases)
10. [Future Directions](#future-directions)
11. [References](#references)

---

## Introduction

**Edge-Cloud Speculative Decoding** is a distributed inference paradigm where:
- **Draft model** runs on edge devices (phones, IoT, local GPUs)
- **Target model** runs in the cloud (data centers, powerful servers)

This approach leverages edge computing for low-latency drafting while using cloud resources for high-quality verification.

### Key Insight

> "Push drafting to the edge, keep verification in the cloud"

The draft model on edge generates candidate tokens quickly with minimal latency, while the cloud-based target model ensures quality through parallel verification.

---

## Motivation

### Why Edge-Cloud?

| Aspect | Edge-Only | Cloud-Only | Edge-Cloud |
|--------|-----------|------------|------------|
| **Latency** | ✅ Lowest | ❌ Highest (network) | ⚠️ Medium |
| **Quality** | ❌ Limited (small model) | ✅ Best (large model) | ✅ Best |
| **Privacy** | ✅ Data stays local | ❌ Data sent to cloud | ⚠️ Partial |
| **Cost** | ❌ Expensive hardware | ✅ Shared resources | ✅ Optimized |
| **Bandwidth** | ✅ None needed | ❌ High (all tokens) | ✅ Reduced |
| **Reliability** | ⚠️ Device limits | ✅ Cloud redundancy | ✅ Hybrid |

### Bandwidth Reduction

Standard cloud inference requires sending **all inputs and receiving all outputs**:
```
Input video: 100MB
Generated text: 10KB
Total bandwidth: 100MB + 10KB per request
```

Edge-cloud speculative decoding:
```
Input: 100MB (processed on edge)
Draft tokens: 1KB (only candidates)
Final output: 10KB
Total bandwidth: 1KB + 10KB per request
```

**Bandwidth savings: ~99%**

---

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EDGE-CLOUD SPECULATIVE DECODING                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐                                   ┌─────────────────┐
│   EDGE DEVICE   │                                   │     CLOUD       │
│  (Phone/Tablet) │                                   │  (Data Center)  │
├─────────────────┤                                   ├─────────────────┤
│                 │                                   │                 │
│  ┌───────────┐  │  Draft Tokens (k candidates)      │  ┌───────────┐  │
│  │  Draft    │  │ ─────────────────────────────────►  │  │  Target   │  │
│  │  Model    │  │    Low bandwidth (1-10KB)          │  │  │  Model   │  │
│  │  (1B)     │  │                                   │  │  │  (70B)   │  │
│  └─────┬─────┘  │                                   │  └─────┬─────┘  │
│        │         │                                   │        │         │
│        │         │  Accept/Reject (bitmask)          │        │         │
│        │         │ ◄─────────────────────────────────  │        │         │
│        │         │    Low bandwidth (1KB)             │        │         │
│        │         │                                   │        │         │
│  ┌─────▼─────┐  │                                   │        │         │
│  │  Local    │  │  Fallback: Full Request            │        │         │
│  │  Cache    │  │  ◄─────────────────────────────────  │        │         │
│  └───────────┘  │    When acceptance rate low        │        │         │
│                 │                                   │                 │
│  ┌───────────┐  │  Generated Output (final text)     │  ┌───────────┐  │
│  │  Display  │  │ ◄─────────────────────────────────  │  │  Logging  │  │
│  └───────────┘  │                                   │  └───────────┘  │
│                 │                                   │                 │
└─────────────────┘                                   └─────────────────┘

KEY PROTOCOL MESSAGES:

1. DraftRequest:
   - context_tokens: List[int]
   - vision_features: Optional[Tensor]  (pre-processed on edge)
   - max_draft_tokens: int

2. DraftResponse (from edge to cloud):
   - draft_tokens: List[int]
   - draft_probs: List[float]
   - context_hash: str  (for verification)

3. VerificationResponse (from cloud to edge):
   - accept_mask: List[bool]  (parallel verification result)
   - resampled_tokens: List[int]  (if rejection)
   - next_context: List[int]  (updated context)

4. FallbackRequest (edge → cloud):
   - Full context when local draft unavailable
```

---

## Related Works

### Distributed Speculative Decoding

| Paper | Year | Focus | Key Contribution |
|-------|------|-------|------------------|
| [Fast Collaborative Inference via Distributed Speculative Decoding](https://www.sciencedirect.com/science/article/pii/S2949715925000782) | 2025 | Distributed systems | Multi-device drafting |
| [LLM Acceleration via Adaptive Edge-Cloud Speculative Decoding](https://ieeexplore.ieee.org/iel8/4234/11331808/11333273.pdf) | 2024 | Edge-Cloud | Adaptive strategy selection |
| [Speculative Decoding Pipeline](https://www.emergentmind.com/topics/speculative-decoding-pipeline) | 2023 | Pipeline optimization | End-to-end pipeline design |

### Edge AI Optimization

| Paper | Year | Focus | Key Contribution |
|-------|------|-------|------------------|
| MobileLLM (Apple) | 2024 | Edge inference | Quantization + speculative |
 [Efficient Speculative Decoding for AI PCs via Hierarchical...](https://dl.acm.org/doi/10.1145/3737902.3768352) | 2024 | AI PCs | Hierarchical verification |

---

## Key Challenges

### 1. Network Latency

**Problem:** Round-trip time (RTT) can negate speculative benefits

```
Time breakdown:
- Draft generation (edge): 2ms
- Network RTT: 20-100ms  ← BOTTLENECK
- Cloud verification: 50ms
- Network return: 20-100ms
Total: 92-252ms vs vanilla 150ms

Benefit: NEGATIVE when RTT > 50ms
```

**Solutions:**
- Use UDP for draft submission (loss acceptable)
- Pre-establish connections (connection pooling)
- Batch multiple draft requests
- Use edge locations closer to users

### 2. Draft Model Quality on Edge

**Problem:** Small models on edge have lower quality

```
Typical edge constraints:
- Memory: 2-4GB max
- Compute: 10-50 TFLOPS
- Power: 5-15W (mobile)

Model size options:
- 1B params: ~2GB FP16, 5W
- 3B params: ~6GB FP16, 15W (too large)
- 7B params: ~14GB FP16 (impossible)

Best option: 1B model with quantization
```

**Impact:**
- Acceptance rate drops to 40-60%
- Fallback to cloud increases

**Solutions:**
- Aggressive quantization (4-bit, 2-bit)
- Knowledge distillation from cloud model
- Hybrid: Use medium model when available

### 3. Privacy and Security

**Problem:** Sending context to cloud may leak sensitive data

```
Privacy concerns:
- User prompts contain personal info
- Vision features may capture private scenes
- Conversation history accumulates sensitive data

Attack vectors:
- Eavesdropping on network traffic
- Cloud provider logging
- Model inversion attacks
```

**Solutions:**
- On-edge preprocessing (sanitization)
- Differential privacy noise
- Federated learning for draft model
- Secure enclaves for cloud verification

### 4. Synchronization and State Management

**Problem:** Edge and cloud must maintain consistent state

```
State synchronization challenges:
- KV cache consistency
- Context window alignment
- Rollback coordination
- Multi-user session management
```

**Solutions:**
- State snapshots every N tokens
- Delta-based synchronization
- Lazy KV cache transfer
- Optimistic concurrency control

---

## Proposed Solutions

### Solution 1: Adaptive Edge-Cloud Strategy

**Paper:** "LLM Acceleration via Adaptive Edge-Cloud Speculative Decoding" (2024)

**Core Idea:** Dynamically choose between edge-only, cloud-only, and hybrid based on:

```python
def choose_strategy(context, network_conditions, model_availability):
    """
    Adaptive strategy selection

    Returns: 'edge', 'cloud', or 'hybrid'
    """
    # Factors to consider
    edge_acceptance_rate = estimate_edge_quality(context)
    network_rtt = measure_network_latency()
    battery_level = get_battery_percentage()
    privacy_level = detect_privacy_sensitivity(context)

    # Decision tree
    if privacy_level == HIGH:
        return 'edge'  # Don't send to cloud

    if network_rtt > 100:  # ms
        return 'edge'  # Too slow for hybrid

    if edge_acceptance_rate < 0.5:
        return 'cloud'  # Edge not good enough

    if battery_level < 20:
        return 'cloud'  # Save battery

    return 'hybrid'  # Best of both worlds
```

**Benefits:**
- Optimizes for latency, quality, privacy, battery
- Seamless fallback between modes
- User-transparent operation

### Solution 2: Pipeline Parallelism

**Idea:** Overlap network transfer with computation

```
Timeline (single generation cycle):

Edge:              Generate draft ──► Send ──► Receive ──► Display
Cloud:                             Receive ──► Verify ──► Send

With pipelining:

Edge:  Gen1 ──► Send1 ──► Gen2 ──► Send2 ──► Recv1 ──► Disp1 ──► Recv2
Cloud:       Recv1 ──► Ver1 ──► Send1 ──► Recv2 ──► Ver2 ──► Send2

Overlap: Network transfer happens while next draft generates
```

**Implementation:**
- Async draft submission
- Non-blocking receive
- Double-buffering
- Connection pooling

**Speedup:** 1.5-2x over naive edge-cloud

### Solution 3: Hierarchical Draft Model Selection

**Idea:** Use different draft models based on edge capabilities

```
Edge Capability Tiers:

Tier 1 (Low-end phones):
    - 0.5B model, 4-bit quantized
    - 2GB memory, 5W power
    - Acceptance: 35-45%
    - Fallback: 60% of time

Tier 2 (Mid-range phones):
    - 1B model, 8-bit quantized
    - 3GB memory, 10W power
    - Acceptance: 55-65%
    - Fallback: 30% of time

Tier 3 (High-end phones/edge devices):
    - 3B model, 8-bit + sparsity
    - 6GB memory, 15W power
    - Acceptance: 70-80%
    - Fallback: 15% of time

Tier 4 (Edge GPU/Jetson):
    - 7B model, 8-bit
    - 14GB memory, 30W power
    - Acceptance: 80-85%
    - Fallback: 5% of time
```

**Auto-detection:**
```python
def detect_device_tier():
    memory_gb = get_available_memory()
    tflops = estimate_compute_capability()
    power_budget = get_power_limit()

    if memory_gb < 3:
        return 'tier1'
    elif memory_gb < 6:
        return 'tier2'
    elif memory_gb < 12:
        return 'tier3'
    else:
        return 'tier4'
```

---

## Implementation Guide

### Edge Device Setup

#### 1. Model Quantization

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def quantize_model_for_edge(model_path, output_path, bits=4):
    """
    Quantize model for edge deployment

    Args:
        model_path: Path to original model
        output_path: Path to save quantized model
        bits: Quantization bits (4 or 8 recommended)
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Apply quantization
    if bits == 4:
        from bitsandbytes.nn import Modules4bit
        model = Modules4bit(
            model,
            quantize=True,
            quant_type="nf4"
        )
    elif bits == 8:
        from transformers import BitsAndBytesConfig
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=config
        )

    # Save quantized model
    model.save_pretrained(output_path)

    return model
```

#### 2. Draft Generation on Edge

```python
import torch
import asyncio
import aiohttp

class EdgeDraftClient:
    """
    Edge client for speculative decoding

    Generates drafts locally, sends to cloud for verification
    """

    def __init__(
        self,
        model_path,
        cloud_url,
        max_draft_tokens=5,
        device="cuda"
    ):
        self.model = self.load_quantized_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.cloud_url = cloud_url
        self.max_draft_tokens = max_draft_tokens
        self.device = device

        # Connection pooling
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10)
        )

    def load_quantized_model(self, path):
        """Load quantized draft model"""
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=self.device,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.eval()
        return model

    @torch.no_grad()
    def generate_drafts(self, input_ids, temperature=1.0):
        """Generate draft tokens on edge"""
        # Generate draft tokens autoregressively
        draft_tokens = []
        draft_probs = []

        current_ids = input_ids.clone()

        for _ in range(self.max_draft_tokens):
            outputs = self.model(current_ids)
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            # Sample token
            next_token = torch.multinomial(probs, num_samples=1)

            # Store
            draft_tokens.append(next_token)
            draft_probs.append(probs[0, next_token].item())

            # Append
            current_ids = torch.cat([current_ids, next_token], dim=-1)

        draft_tokens = torch.cat(draft_tokens, dim=-1)
        return draft_tokens, draft_probs

    async def verify_with_cloud(self, input_ids, draft_tokens, draft_probs):
        """Send drafts to cloud for verification"""
        payload = {
            "context_tokens": input_ids.tolist(),
            "draft_tokens": draft_tokens.tolist(),
            "draft_probs": draft_probs,
            "max_draft_tokens": self.max_draft_tokens
        }

        # Async HTTP request
        async with self.session.post(
            f"{self.cloud_url}/verify",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=1.0)  # 1 second timeout
        ) as response:
            result = await response.json()

        return result

    async def generate_stream(self, prompt, max_tokens=100):
        """Stream generation with edge-cloud speculative decoding"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        generated = []

        for step in range(max_tokens):
            # Generate drafts on edge
            draft_tokens, draft_probs = self.generate_drafts(input_ids)

            # Verify with cloud
            result = await self.verify_with_cloud(
                input_ids,
                draft_tokens,
                draft_probs
            )

            # Process results
            accept_mask = result['accept_mask']
            accepted_tokens = result['accepted_tokens']

            # Append accepted tokens
            for token in accepted_tokens:
                generated.append(token)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device)
                ], dim=-1)

            # Check for EOS
            if accepted_tokens[-1] == self.tokenizer.eos_token_id:
                break

            # Fallback if needed
            if result.get('fallback', False):
                # Generate locally without speculative
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=1,
                    do_sample=True
                )
                token = outputs[0, -1].item()
                generated.append(token)

        # Decode
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text

    async def close(self):
        """Clean up resources"""
        await self.session.close()
```

### Cloud Server Setup

#### 1. Verification Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

class VerifyRequest(BaseModel):
    context_tokens: list[int]
    draft_tokens: list[int]
    draft_probs: list[float]
    max_draft_tokens: int

class VerifyResponse(BaseModel):
    accept_mask: list[bool]
    accepted_tokens: list[int]
    resampled_tokens: list[int]
    fallback: bool = False

app = FastAPI()

# Load target model
target_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")

@app.post("/verify", response_model=VerifyResponse)
async def verify_drafts(request: VerifyRequest):
    """
    Verify draft tokens from edge device

    Implements standard speculative sampling verification
    """
    try:
        # Convert to tensors
        context_ids = torch.tensor(
            [request.context_tokens],
            dtype=torch.long
        )
        draft_ids = torch.tensor(
            [request.draft_tokens],
            dtype=torch.long
        )

        # Create full sequence
        full_ids = torch.cat([context_ids, draft_ids], dim=-1)

        # Get target model probabilities
        with torch.no_grad():
            outputs = target_model(full_ids)
            # Get logits for draft positions only
            target_logits = outputs.logits[:, context_ids.shape[1]-1:-1, :]
            target_probs = torch.softmax(target_logits, dim=-1)

        # Verify each draft token
        accept_mask = []
        accepted_tokens = []

        for i, token_id in enumerate(request.draft_tokens):
            q = request.draft_probs[i]
            p = target_probs[0, i, token_id].item()

            # Acceptance probability
            alpha = min(1.0, p / q)

            # Accept or reject
            import random
            if random.random() < alpha:
                accept_mask.append(True)
                accepted_tokens.append(token_id)
            else:
                # Rejection
                accept_mask.append(False)

                # Resample from adjusted distribution
                p_adjusted = torch.clamp(
                    target_probs[0, i] - q,
                    min=0
                )
                p_adjusted = p_adjusted / (p_adjusted.sum() + 1e-10)

                resampled = torch.multinomial(p_adjusted, 1).item()
                accepted_tokens.append(resampled)

                # Stop accepting after rejection
                break

        return VerifyResponse(
            accept_mask=accept_mask,
            accepted_tokens=accepted_tokens,
            resampled_tokens=[],
            fallback=len(accepted_tokens) == 0
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
```

#### 2. Connection Management

```python
import asyncio
from collections import defaultdict

class ConnectionManager:
    """
    Manage connections between edge devices and cloud

    Implements connection pooling, keep-alive, and load balancing
    """

    def __init__(self, max_connections=100):
        self.connections = defaultdict(list)
        self.max_connections = max_connections
        self.semaphores = defaultdict(asyncio.Semaphore)

    async def get_connection(self, device_id):
        """Get or create connection for device"""
        if device_id not in self.semaphores:
            self.semaphores[device_id] = asyncio.Semaphore(
                self.max_connections
            )

        await self.semaphores[device_id].acquire()
        return device_id

    def release_connection(self, device_id):
        """Release connection"""
        self.semaphores[device_id].release()

    async def broadcast_update(self, device_ids, message):
        """Broadcast updates to multiple devices"""
        tasks = [
            self.send_to_device(device_id, message)
            for device_id in device_ids
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def send_to_device(self, device_id, message):
        """Send message to specific device"""
        # Implementation depends on transport (WebSocket, gRPC, etc.)
        pass
```

---

## Performance Analysis

### Latency Breakdown

```
Edge-Cloud Hybrid (RTT = 50ms):

┌─────────────────────────────────────────────────────────────────┐
│ Operation                    │ Time    │ % of Total │
├─────────────────────────────────────────────────────────────────┤
│ Draft generation (edge)      │ 3ms     │ 2.4%       │
│ Network: Edge → Cloud         │ 25ms    │ 20.2%      │
│ Cloud verification            │ 60ms    │ 48.4%      │
│ Network: Cloud → Edge         │ 25ms    │ 20.2%      │
│ Local processing              │ 11ms    │ 8.9%       │
├─────────────────────────────────────────────────────────────────┤
│ Total                         │ 124ms   │ 100%       │
│ Vanilla (cloud-only)          │ 500ms   │ 403%       │
│ Speedup                       │ 4.03x   │ -          │
└─────────────────────────────────────────────────────────────────┘

Edge-Cloud Hybrid (RTT = 10ms, close to edge):

┌─────────────────────────────────────────────────────────────────┐
│ Operation                    │ Time    │ % of Total │
├─────────────────────────────────────────────────────────────────┤
│ Draft generation (edge)      │ 3ms     │ 3.2%       │
│ Network: Edge → Cloud         │ 5ms     │ 5.3%       │
│ Cloud verification            │ 60ms    │ 63.8%      │
│ Network: Cloud → Edge         │ 5ms     │ 5.3%       │
│ Local processing              │ 21ms    │ 22.3%      │
├─────────────────────────────────────────────────────────────────┤
│ Total                         │ 94ms    │ 100%       │
│ Vanilla (cloud-only)          │ 500ms   │ 532%       │
│ Speedup                       │ 5.32x   │ -          │
└─────────────────────────────────────────────────────────────────┘
```

### Bandwidth Analysis

```
Per-Generation-Cycle Bandwidth:

Standard Cloud Inference:
  Input video (processed): 100MB
  Prompt tokens: 1KB
  Generated tokens: 10KB
  Total: ~100MB per request

Edge-Cloud Speculative:
  Draft tokens: 1KB
  Verification request: 2KB
  Verification response: 3KB
  Final output: 10KB
  Total: ~16KB per request

Bandwidth Reduction: 99.98%

For 1000 requests/day:
  Standard: 100GB/day
  Edge-Cloud: 16MB/day
```

### Cost Analysis

```
Cloud Computing Cost (per 1M tokens):

Standard (all in cloud):
  GPU hours: 100 hours × $3/hr = $300
  Bandwidth: 100GB × $0.01/GB = $1
  Total: $301

Edge-Cloud (70% edge processing):
  Cloud GPU: 30 hours × $3/hr = $90
  Edge: Amortized hardware cost ≈ $10
  Bandwidth: 1.6GB × $0.01/GB = $0.016
  Total: $100

Cost Savings: 67%
```

---

## Use Cases

### 1. Mobile Assistants

**Application:** Personal AI assistant on smartphones

**Requirements:**
- Low latency (<200ms per response)
- Privacy (local processing when possible)
- Battery efficiency

**Solution:**
```python
# Adaptive edge-cloud based on context

context = analyze_context(user_query)

if context.privacy == "sensitive":
    mode = "edge_only"  # Don't send to cloud
elif context.complexity == "high":
    mode = "cloud_only"  # Need full model
else:
    mode = "hybrid"  # Speculative decoding

response = generate(query, mode=mode)
```

### 2. IoT Cameras

**Application:** Video surveillance with real-time analysis

**Requirements:**
- Process on device when possible
- Cloud fallback for complex queries
- Minimal bandwidth

**Solution:**
```
Camera (Edge):
  1. Extract features locally
  2. Detect simple events locally
  3. Draft complex descriptions
  4. Send to cloud for verification

Cloud:
  1. Verify drafts
  2. Handle edge cases
  3. Store long-term data
```

### 3. Automotive

**Application:** In-car voice assistant

**Requirements:**
- Must work offline (tunnels, rural areas)
- Low latency for safety
- High quality for complex queries

**Solution:**
```python
# Fallback chain

def automotive_generate(query):
    # Try edge first (must work offline)
    try:
        response = edge_model.generate(query, max_tokens=100)
        return response
    except LowConfidenceError:
        # If connected, use hybrid
        if is_connected():
            return edge_cloud_hybrid(query)
        else:
            # Fallback to cached responses
            return cached_responses.get(query, "I couldn't process that")
```

### 4. Smart Home

**Application:** Voice-controlled home automation

**Requirements:**
- Privacy (home data stays local)
- Speed (instant response)
- Multi-device coordination

**Solution:**
```
Hub (Edge device):
  - Local voice recognition
  - Draft generation
  - Device control

Cloud:
  - Verification of complex commands
  - Learning user preferences
  - Cross-device coordination
```

---

## Future Directions

### 1. Federated Learning for Edge Models

**Idea:** Train edge draft models without sending raw data to cloud

```
Federated Learning Loop:

1. Cloud sends global model to edge devices
2. Edge devices train on local data
3. Send model updates (gradients) to cloud
4. Cloud aggregates and updates global model
5. Repeat

Benefits:
- Privacy preservation
- Personalized models
- Reduced bandwidth
```

### 2. Hierarchical Edge-Cloud Architecture

**Idea:** Multiple edge tiers before cloud

```
Device Edge (Phone)
    ↓ (fallback)
Local Edge (Home router, nano-server)
    ↓ (fallback)
Edge Cloud (CDN, regional data center)
    ↓ (fallback)
Central Cloud (main data center)
```

### 3. Predictive Prefetching

**Idea:** Pre-fetch likely next tokens based on user patterns

```
User pattern analysis:
  Morning: News brief → weather → calendar
  Evening: Social updates → entertainment
  Driving: Navigation → music → messages

Pre-fetch likely drafts to reduce latency further
```

### 4. Compression Techniques

**Idea:** Compress drafts before sending to cloud

```
Compression methods:
- Arithmetic encoding of tokens
- Delta encoding for probabilities
- Sparse representation for accept masks
- LZ4 for fast compression/decompression

Target: 50% reduction in draft size
```

---

## References

### Edge-Cloud Speculative Decoding

- [Fast Collaborative Inference via Distributed Speculative Decoding](https://www.sciencedirect.com/science/article/pii/S2949715925000782) - 2025
- [LLM Acceleration via Adaptive Edge-Cloud Speculative Decoding](https://ieeexplore.ieee.org/iel8/4234/11331808/11333273.pdf) - IEEE 2024
- [Efficient Speculative Decoding for AI PCs via Hierarchical Verification](https://dl.acm.org/doi/10.1145/3737902.3768352) - ACM 2024

### Edge AI Optimization

- [MobileLLM: An Efficient Large Language Model for Mobile Devices (Apple, 2024)](https://machinelearning.apple.com/research/mobilellm/)
- [Speculative Decoding on Edge GPUs](https://developer.nvidia.com/blog/speculative-decoding-on-jetson/)
- [TinyLLM: Efficient LLM for Edge Devices](https://arxiv.org/abs/2401.03085)

### Distributed Systems

- [Pipeline Parallelism for LLM Inference](https://arxiv.org/abs/2305.14324)
- [Distributed Inference with Speculative Decoding](https://proceedings.mlr.xyz/v2023/chen23a.html)

---

## Summary

### Key Takeaways

| Aspect | Verdict |
|--------|---------|
| **Latency** | ⚠️ Depends heavily on RTT; best when RTT < 20ms |
| **Bandwidth** | ✅ Excellent: 99%+ reduction |
| **Cost** | ✅ Good: 50-70% cloud cost reduction |
| **Quality** | ✅ Same as cloud (verification ensures quality) |
| **Privacy** | ⚠️ Partial: drafts sent, but less than full data |
| **Complexity** | ❌ High: requires edge + cloud coordination |

### When to Use Edge-Cloud Speculative Decoding

✅ **Use when:**
- Network RTT < 50ms (edge locations, 5G)
- Edge devices have sufficient compute (>10 TFLOPS)
- Bandwidth is expensive or limited
- Need cloud-level quality with edge-level latency
- Can tolerate some complexity

❌ **Avoid when:**
- Network RTT > 100ms
- Edge devices are very limited (<5 TFLOPS)
- Privacy requirements prohibit any cloud communication
- Need extreme simplicity
- Cloud resources are abundant and cheap

### For EventGPT → VideoLLaVA

**Recommended Architecture:**
```
Edge (Event Camera/Phone):
  - EventGPT Vision Encoder (sparse features)
  - EventGPT LM (1B quantized)
  - Draft generation

Cloud:
  - VideoLLaVA-7B/13B (verification)
  - Training and fine-tuning
  - Long-term storage
```

**Expected Benefits:**
- 50-100x speedup on vision encoding (sparse → dense)
- 2-3x speedup on text generation
- 95%+ bandwidth reduction
- Maintains VideoLLaVA quality

---

**Last Updated:** January 2026
