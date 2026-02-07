# Paper Abstract Drafts

**Working Title:** Cross-Modal Speculative Decoding: Hiding Event Camera Prefill Inside Video VLM Inference

---

## Abstract (Option A - Technical Focus)

Speculative decoding accelerates large language model inference by using a fast draft model to propose tokens that a slower target model verifies in parallel. While recent work has explored parallel execution of draft and target models during decoding, all existing approaches operate within the same input modality. We introduce **cross-modal speculative decoding**, a novel paradigm that exploits the physical properties of different visual sensors. Using an event camera-based vision-language model (EventGPT) as the draft for a frame-based video VLM (Video-LLaVA), we leverage the inherent sparsity of event cameras to achieve 8.59× faster prefill (66ms vs. 568ms for 8 frames). This timing asymmetry enables **prefill hiding**: the draft model completes its entire prefill and generates approximately 27 tokens entirely within the target's prefill window, achieving effectively free speculation. On the DSEC driving dataset, our approach generates 61% of the typical response length at zero additional latency cost, yielding 1.3–1.7× end-to-end speedup depending on acceptance rate. Our work opens a new research direction—sensor-level speculation—where draft model efficiency derives from physical sensor properties rather than architectural compression alone.

---

## Abstract (Option B - Concise)

Speculative decoding accelerates inference by having a fast draft model propose tokens for a slower target to verify. We introduce **cross-modal speculative decoding**, the first approach to use fundamentally different sensor modalities for draft and target models. By employing an event camera-based VLM (EventGPT) as draft for a video VLM (Video-LLaVA), we exploit the 8.59× prefill speed advantage of sparse event representations over dense video frames. This asymmetry enables **prefill hiding**: generating ~27 draft tokens entirely within the target's 568ms prefill window at zero additional latency. Our method achieves 1.3–1.7× speedup while opening a new research direction in sensor-aware speculative inference.

---

## Abstract (Option C - Emphasis on Novelty)

All existing speculative decoding methods use the same input modality for draft and target models—smaller text models draft for larger ones, compressed visual tokens draft for full representations. We challenge this assumption with **cross-modal speculative decoding**, demonstrating that a fundamentally different sensor modality can serve as an effective draft source. Our key insight is that event cameras, which capture sparse asynchronous brightness changes, encode temporal dynamics in a constant ~577 tokens regardless of video length, while frame-based models scale linearly (576 tokens × frames). Using EventGPT (event camera) as draft for Video-LLaVA (RGB video), we achieve 8.59× faster prefill, enabling **prefill hiding**—where the draft's entire prefill plus ~27 token generations complete within the target's prefill window. This "free" speculation yields 1.3–1.7× end-to-end speedup. Beyond acceleration, our work establishes sensor-level speculation as a new paradigm, where physical sensor properties—not just model architecture—determine drafting efficiency.

---

## Abstract (Option D - Application-Oriented)

Real-time video understanding with vision-language models (VLMs) remains challenging due to the quadratic scaling of attention with visual token count. We present **EventSpec**, a cross-modal speculative decoding framework that accelerates video VLM inference by using event camera inputs as a draft modality. Event cameras capture scene dynamics as sparse, asynchronous events rather than dense frames, producing constant-length representations (~577 tokens) regardless of temporal span—compared to 576×N tokens for N-frame video. This fundamental efficiency enables our core contribution: **prefill hiding**, where the event-based draft model (EventGPT) completes its 66ms prefill and generates ~27 speculative tokens entirely within the target video model's (Video-LLaVA) 568ms prefill window. The result is effectively "free" draft speculation. On driving scene benchmarks, EventSpec achieves 1.3–1.7× inference speedup while maintaining output quality through standard rejection sampling. Our approach demonstrates that heterogeneous sensor systems—combining event cameras with RGB video—can unlock inference efficiency beyond what single-modality optimization achieves.

---

## Recommended: Option A or C

**Option A** is best for venues emphasizing technical contribution (NeurIPS, ICML, ICLR).

**Option C** is best for venues emphasizing novelty and new directions (CVPR, ECCV).

---

## Key Claims to Support in Paper

1. **First cross-modal speculative decoding** using different sensor modalities
2. **8.59× prefill speedup** from event vs. video encoding (66ms vs. 568ms)
3. **~27 free draft tokens** generated during target prefill (502ms window)
4. **1.3–1.7× end-to-end speedup** depending on acceptance rate
5. **Constant token scaling** for events vs. linear for video frames
6. **New research direction**: sensor-level speculation

---

## Suggested Paper Structure

1. **Introduction**: Speculative decoding background, limitation of same-modality assumption
2. **Related Work**: DSI, PEARL, SSD, HiViS, SpecVLM (all same-modality)
3. **Method**:
   - Cross-modal speculative decoding formulation
   - Prefill hiding mechanism
   - Feature alignment for acceptance rate improvement
4. **Experiments**:
   - Prefill timing comparison (EventGPT vs Video-LLaVA)
   - Acceptance rate analysis (static vs dynamic scenes)
   - End-to-end speedup measurements
   - Ablation: event draft vs single-frame RGB draft
5. **Analysis**:
   - When does cross-modal speculation work best?
   - Acceptance rate vs scene dynamics
   - Memory considerations
6. **Conclusion**: New paradigm of sensor-level speculation

---

**Document Created:** 2026-01-26
