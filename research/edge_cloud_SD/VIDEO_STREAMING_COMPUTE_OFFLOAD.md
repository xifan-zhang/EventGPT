# Video Streaming + Token-Only Verification: Reversing Edge-Cloud Paradigm

**Date:** 2026-01-28
**Author:** Alice Zhang
**Status:** New Idea / Proposal

---

## Core Insight

> **Traditional Approach:** Process video on edge, send tokens to cloud for verification
> **New Approach:** Stream raw video to cloud, only send draft tokens back to edge for display

### Why This Makes Sense

| Factor | Video Streaming | Vision Encoding |
|--------|----------------|-----------------|
| **Bandwidth Cost** | $0.01-0.05/GB | N/A (compute) |
| **Latency (5G)** | <1ms network | 50-500ms compute |
| **Infrastructure** | Highly optimized CDNs | GPU-intensive |
| **Scalability** | Elastic, pay-per-use | Limited by GPUs |
| **Edge Requirement** | Camera + network | GPU + memory |

**Key Realization:** Video streaming infrastructure is mature, cheap, and fast. Vision encoding is expensive and GPU-bound. Flip the paradigm.

---

## Architecture: Stream-to-Cloud Speculative Decoding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              STREAM-TO-CLOUD SPECULATIVE DECODING                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐                                   ┌─────────────────────┐
│   EDGE DEVICE   │                                   │        CLOUD        │
│  (Camera/Phone) │                                   │   (Data Center)     │
├─────────────────┤                                   ├─────────────────────┤
│                 │                                   │                     │
│  ┌───────────┐  │   Raw Video Stream (H.264/H.265)  │  ┌───────────────┐  │
│  │  Camera   │──┼──────────────────────────────────►│  │ Vision Encoder│  │
│  └───────────┘  │   5G: 100Mbps, <1ms latency       │  │ (CLIP/SigLIP) │  │
│                 │   Cost: ~$0.02/GB                 │  └───────┬───────┘  │
│                 │                                   │          │          │
│                 │                                   │          ▼          │
│  ┌───────────┐  │   Draft Tokens (lightweight)      │  ┌───────────────┐  │
│  │ Small LLM │◄─┼──────────────────────────────────│  │  Target LLM   │  │
│  │ (0.5-1B)  │  │   ~1KB per generation cycle      │  │  (7B-70B)     │  │
│  └─────┬─────┘  │                                   │  └───────────────┘  │
│        │        │                                   │                     │
│        ▼        │   Verification Request            │                     │
│  ┌───────────┐  │   (Edge drafts → Cloud verify)    │                     │
│  │  Display  │──┼──────────────────────────────────►│                     │
│  └───────────┘  │                                   │                     │
│                 │                                   │                     │
└─────────────────┘                                   └─────────────────────┘

BANDWIDTH ANALYSIS:
─────────────────────────────────────────────────────────────────────────────
Direction          │ Data                      │ Size/Request │ Cost
─────────────────────────────────────────────────────────────────────────────
Edge → Cloud       │ Video stream (1080p30)    │ 5-10 MB/s    │ $0.02/GB
Edge → Cloud       │ Draft tokens for verify   │ 1-2 KB       │ negligible
Cloud → Edge       │ Verified tokens + next    │ 2-5 KB       │ negligible
─────────────────────────────────────────────────────────────────────────────
```

---

## Research Support

### 1. Disaggregated Inference is Industry Standard (2025)

> "Eighteen months ago, the Hao AI Lab introduced DistServe with a simple bet: split LLM inference into prefill and decode, and scale them independently on separate compute pools. Today, almost every production-grade LLM serving framework – NVIDIA Dynamo, llm-d, Ray Serve LLM SGLang, vLLM, LMCache, MoonCake – runs on disaggregation."
> — [Hao AI Lab, 2025](https://hao-ai-lab.github.io/blogs/distserve-retro/)

**Implication:** If prefill and decode can be separated across GPUs, vision encoding and text generation can also be separated across edge/cloud.

### 2. Prefill is Compute-Bound, Decode is Memory-Bound

> "P/D disaggregated architectures physically or logically separate the prefill and decode stages. The fundamental rationale is the orthogonality of resource bottlenecks: prefill is compute-FLOPs bound (requiring peak GPU arithmetic throughput), whereas decode is usually memory bandwidth and working set bound."
> — [Prefill-Decode Disaggregation](https://www.emergentmind.com/topics/prefill-decode-p-d-disaggregated-architectures)

**Implication:** Vision encoding (part of prefill) is compute-heavy → put it in the cloud. Draft generation (decode) is memory-light → run on edge.

### 3. 5G Makes Video Streaming Near-Free

> "5G offers massive network capacity, giving operators an opportunity to provide more bandwidth (>100Mbps) per user... Network latency with 5G can drop as low as one millisecond."
> — [Harmonic: 5G Mobile Streaming](https://www.harmonicinc.com/insights/blog/mobile-streaming/)

> "By the end of 2025, global 5G subscriptions are expected to reach about 2.9 billion, accounting for roughly one-third of all mobile subscriptions."
> — [Dacast: 5G Streaming](https://www.dacast.com/blog/5g-streaming/)

**Bandwidth Costs (2025):**
| Provider | Cost/GB | Notes |
|----------|---------|-------|
| CacheFly | $0.01-0.05 | Volume discounts |
| Cloudflare | $0.01 | At scale |
| AWS CloudFront | $0.02-0.085 | Region-dependent |

### 4. Vision Encoding is the Bottleneck

From our EventGPT benchmarks:
```
Vision + Prefill: 66ms (EventGPT) vs 568ms (Video-LLaVA)
LLM Decode: ~20ms/token (both models)

Vision encoding = 25-50% of total inference time
```

**Implication:** Offloading vision to cloud saves significant edge compute.

### 5. SLED Framework Validates Edge-Cloud Separation

> "SLED is a framework that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server verifies the tokens utilizing a more precise target model."
> — [SLED: Speculative LLM Decoding for Edge Serving](https://arxiv.org/html/2506.09397v3)

---

## Comparison: Traditional vs Stream-to-Cloud

| Aspect | Traditional Edge-Cloud | Stream-to-Cloud (New) |
|--------|----------------------|---------------------|
| **Edge Compute** | Vision encoder + Draft LLM | Draft LLM only |
| **Edge Memory** | 4-8 GB (vision + LLM) | 1-2 GB (small LLM) |
| **Edge Power** | 10-30W | 3-5W |
| **Network Upload** | Tokens only (1KB) | Video + Tokens (5-10MB/s) |
| **Network Download** | Verification (2KB) | Draft tokens (1KB) |
| **Cloud Compute** | Target LLM only | Vision + Target LLM |
| **Latency** | Edge vision + network RTT | Network RTT + cloud vision |
| **Privacy** | Raw data stays on edge | Raw video sent to cloud |

### When Stream-to-Cloud Wins

1. **Edge device is compute-limited** (IoT, cheap phones)
2. **5G/fiber network available** (<10ms RTT)
3. **Privacy is not critical** (public cameras, non-sensitive content)
4. **Cloud GPUs are underutilized** (can handle vision load)
5. **Battery life is critical** (mobile devices)

### When Traditional Edge-Cloud Wins

1. **Privacy-sensitive applications** (medical, personal)
2. **High-latency networks** (>50ms RTT)
3. **Offline operation required** (automotive, rural)
4. **Edge has powerful GPU** (Jetson, dedicated hardware)

---

## Implementation for EventGPT → Video-LLaVA

### Stream-to-Cloud Architecture

```python
class StreamToCloudSpeculative:
    """
    Stream video to cloud, run vision + target LLM there.
    Edge only runs lightweight draft LLM.
    """

    def __init__(self):
        # Edge: Ultra-lightweight draft model
        self.edge_draft = TinyLLM(size="0.5B", quantized=True)  # 500MB

        # Cloud: Full vision + language pipeline
        self.cloud_vision = VideoLLaVAVisionEncoder()  # Cloud only
        self.cloud_llm = VideoLLaVA7B()  # Cloud only

        # Video streaming
        self.video_streamer = H265Streamer(bitrate="5Mbps")

    async def generate(self, video_stream, query):
        # 1. Stream video to cloud (async, non-blocking)
        cloud_task = asyncio.create_task(
            self.stream_and_encode(video_stream, query)
        )

        # 2. Edge: Generate draft tokens using cached/prior context
        # (First iteration uses generic priors, later uses cloud feedback)
        draft_tokens = self.edge_draft.generate(
            context=self.last_cloud_context,
            max_tokens=5
        )

        # 3. Cloud: Vision encode + verify drafts + continue generation
        cloud_result = await cloud_task
        verified, next_context = cloud_result

        # 4. Update edge context for next iteration
        self.last_cloud_context = next_context

        return verified

    async def stream_and_encode(self, video_stream, query):
        """Cloud-side: receive video, encode, generate."""
        # Receive streamed video
        frames = await self.receive_video_stream(video_stream)

        # Vision encoding (expensive, but on cloud GPU)
        vision_features = self.cloud_vision.encode(frames)  # 500ms

        # Target LLM generation
        output = self.cloud_llm.generate(
            vision_features=vision_features,
            query=query,
            max_tokens=50
        )

        return output
```

### Bandwidth Budget

```
1080p30 H.265 video stream: 5 Mbps = 37.5 MB/min

For 1-minute video query:
- Upload: 37.5 MB video + 10 KB tokens = 37.51 MB
- Download: 5 KB verified tokens
- Total: ~38 MB

Cost at $0.02/GB: $0.00076 per query

Compare to running Video-LLaVA on edge:
- GPU: $0.50/hr for RTX 4090 equivalent
- 1 query = ~2 seconds = $0.00028 compute
- But requires $500+ edge hardware

Break-even: ~1800 queries/day justifies edge GPU
```

---

## Hybrid Strategy: Adaptive Offload

```python
def choose_strategy(network_quality, edge_capability, privacy_level):
    """
    Dynamically choose between strategies based on conditions.
    """

    # Privacy override
    if privacy_level == "HIGH":
        return "EDGE_ONLY"  # Never send raw video

    # Network quality check
    if network_quality.rtt > 50:  # ms
        return "EDGE_VISION"  # Too slow for streaming

    if network_quality.bandwidth < 10:  # Mbps
        return "EDGE_VISION"  # Can't stream video

    # Edge capability check
    if edge_capability.gpu_tflops > 50:  # Powerful edge
        return "EDGE_VISION"  # Has compute to spare

    if edge_capability.memory_gb < 4:  # Limited edge
        return "STREAM_TO_CLOUD"  # Can't fit vision encoder

    # Default: stream to cloud (saves edge compute)
    return "STREAM_TO_CLOUD"


# Strategy implementations
STRATEGIES = {
    "EDGE_ONLY": {
        "edge": ["vision", "draft_llm", "target_llm"],
        "cloud": [],
        "network": "none"
    },
    "EDGE_VISION": {
        "edge": ["vision", "draft_llm"],
        "cloud": ["target_llm"],
        "network": "tokens_only"
    },
    "STREAM_TO_CLOUD": {
        "edge": ["draft_llm"],
        "cloud": ["vision", "target_llm"],
        "network": "video_stream"
    },
    "CLOUD_ONLY": {
        "edge": [],
        "cloud": ["vision", "draft_llm", "target_llm"],
        "network": "video_stream"
    }
}
```

---

## EventGPT Advantage: Event Streams are Smaller

For event cameras, the stream-to-cloud approach is even more attractive:

```
Event camera data rate: 10-100 KB/s (sparse events)
vs RGB video: 5-10 MB/s (dense pixels)

50-500x smaller data to stream!
```

**Proposed EventGPT Stream-to-Cloud:**
1. Edge: Stream sparse events to cloud (10-100 KB/s)
2. Cloud: EventGPT vision encoder (66ms) + draft generation
3. Cloud: Video-LLaVA for verification (if needed)
4. Edge: Display results

**Benefits:**
- Minimal bandwidth (event streams are tiny)
- Full EventGPT quality (cloud has resources)
- Battery-efficient edge (just capture + stream)

---

## Research Opportunities

### 1. Optimal Video Compression for VLM

**Question:** What video codec settings maximize VLM accuracy while minimizing bandwidth?

**Hypothesis:** VLMs are robust to compression artifacts; aggressive compression (H.265, 1-5 Mbps) may have <1% accuracy loss.

### 2. Predictive Video Prefetching

**Question:** Can we predict which video frames the VLM needs and stream only those?

**Approach:** Use edge saliency detection to prioritize important regions/frames.

### 3. Event-Stream Compression

**Question:** What is the optimal compression for event camera streams?

**Current:** Raw events are already sparse; additional compression may hurt more than help.

### 4. Privacy-Preserving Video Streaming

**Question:** Can we stream video to cloud while preserving privacy?

**Approaches:**
- Federated vision encoders
- Differential privacy on video
- Secure enclaves for cloud processing

---

## References

### Disaggregated Inference
- [Disaggregated Inference: 18 Months Later | Hao AI Lab](https://hao-ai-lab.github.io/blogs/distserve-retro/)
- [NVIDIA Dynamo for Distributed Inference](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [Prefill-Decode Disaggregation](https://www.emergentmind.com/topics/prefill-decode-p-d-disaggregated-architectures)
- [Disaggregated Inference with vLLM](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/)

### Edge LLM & Speculative Decoding
- [SLED: Speculative LLM Decoding for Edge](https://arxiv.org/html/2506.09397v3)
- [CoSense-LLM: Edge-First Multimodal](https://arxiv.org/html/2510.19670v1)
- [Edge LLM Survey](https://www.sciencedirect.com/science/article/abs/pii/S1574013725000310)

### Video Streaming & 5G
- [5G Streaming Revolution (Dacast)](https://www.dacast.com/blog/5g-streaming/)
- [5G Mobile Streaming (Harmonic)](https://www.harmonicinc.com/insights/blog/mobile-streaming/)
- [Low-Latency Streaming Solutions](https://www.dacast.com/blog/best-low-latency-video-streaming-solution/)

### LLM Inference Optimization
- [Memory vs Compute Bottleneck](https://arxiv.org/html/2507.14397v1)
- [LLM Inference Best Practices (Databricks)](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [Google: Network Latency Trumps Compute](https://www.sdxcentral.com/news/ai-inference-crisis-google-engineers-on-why-network-latency-and-memory-trump-compute/)

---

## Summary

| Aspect | Key Point |
|--------|-----------|
| **Core Idea** | Stream video to cloud, draft tokens on edge |
| **Why It Works** | Video streaming is cheap/fast; vision encoding is expensive |
| **Best For** | Compute-limited edge, 5G networks, non-private data |
| **Avoid When** | High privacy, poor network, powerful edge GPU |
| **EventGPT Advantage** | Event streams are 50-500x smaller than video |
| **Expected Benefit** | 3-5x edge power savings, same inference quality |

---

**Last Updated:** 2026-01-28
