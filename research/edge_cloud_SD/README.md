# Edge-Cloud Speculative Decoding Research

**Created:** 2026-01-28

Research on distributed speculative decoding architectures for VLMs, focusing on edge-cloud separation strategies.

---

## Documents

| Document | Description | Date |
|----------|-------------|------|
| [edge_cloud_speculative_decoding.md](edge_cloud_speculative_decoding.md) | Original edge-cloud SD survey: architecture, challenges, solutions | 2026-01 |
| [VIDEO_STREAMING_COMPUTE_OFFLOAD.md](VIDEO_STREAMING_COMPUTE_OFFLOAD.md) | **NEW** Stream video to cloud, draft on edge paradigm | 2026-01-28 |

---

## Key Ideas

### Traditional Edge-Cloud SD
- Edge: Vision encoder + Draft LLM
- Cloud: Target LLM for verification
- Network: Tokens only

### Stream-to-Cloud SD (New)
- Edge: Draft LLM only (minimal compute)
- Cloud: Vision encoder + Target LLM
- Network: Video stream + tokens

**Insight:** Video streaming is cheap (5G, CDN). Vision encoding is expensive. Stream raw video to cloud, save edge compute.

---

## Relevance to EventGPT

Event camera data is 50-500x smaller than RGB video:
- RGB: 5-10 MB/s
- Events: 10-100 KB/s

This makes stream-to-cloud especially attractive for EventGPT:
1. Stream sparse events to cloud (minimal bandwidth)
2. Cloud runs EventGPT vision encoder
3. Cloud runs Video-LLaVA for verification
4. Edge only displays results

---

## Research Support

- [Disaggregated Inference is Industry Standard (2025)](https://hao-ai-lab.github.io/blogs/distserve-retro/)
- [SLED: Edge Speculative Decoding](https://arxiv.org/html/2506.09397v3)
- [5G: 100Mbps, <1ms latency](https://www.dacast.com/blog/5g-streaming/)
- [Video bandwidth: $0.01-0.05/GB](https://www.vpsbenchmarks.com/hosters/bandwidth_prices)

---

**Last Updated:** 2026-01-28
