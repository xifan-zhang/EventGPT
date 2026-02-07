#!/usr/bin/env python3
"""
================================================================================
        PARALLEL PREFILL BENCHMARK: EventGPT vs Video-LLaVA (4-bit)
        Token Hiding Analysis with Full 5-Stage Timing
================================================================================

This script benchmarks EventGPT and Video-LLaVA with 4-bit quantization:
- All 5 stages measured separately for each model
- Wall-clock time for parallel execution simulation
- "Hidden tokens" analysis: tokens generated during Video-LLaVA's slower prefill
- Full token output and text from both models
- Memory-efficient 4-bit quantization for both 7B models

5 STAGES:
  Stage 1: Data Loading (images/video from disk)
  Stage 2: Preprocessing (tokenization, image transforms)
  Stage 3: Vision Encoding (CLIP/ViT forward pass)
  Stage 4: LLM Prefill (processing input context, build KV cache)
  Stage 5: LLM Decode (autoregressive token generation)

PARALLEL PREFILL CONCEPT:
  - EventGPT prefill is faster (~75ms) than Video-LLaVA (~336ms)
  - Overlap window = VL_prefill - EGPT_prefill (~261ms)
  - During this window, EventGPT can generate "free" draft tokens
  - These tokens are "hidden" within Video-LLaVA's prefill latency

USAGE:
  python benchmark_parallel_prefill_5stages.py --dataset_dir DATASET_DIR --max_samples N

OPTIONS:
  --dataset_dir    Path to dataset (default: ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s)
  --max_samples    Number of samples to benchmark (-1 for all, default: -1)
  --max_new_tokens Maximum tokens to generate (default: 50)
  --output_dir     Output directory for results (default: ./results/)
  --device         Device to use (default: cuda)

OUTPUT:
  JSON file: parallel_prefill_5stages_{timestamp}.json
  Markdown report: parallel_prefill_5stages_{timestamp}.md

AUTHOR: Alice Zhang
DATE: 2026-01-27
================================================================================
"""

import os
import sys
import json
import argparse
import torch
import time
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

# Fix protobuf compatibility issue with sentencepiece
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import AutoProcessor, VideoLlavaForConditionalGeneration, VideoLlavaProcessor, AutoTokenizer, BitsAndBytesConfig
from common.common import load_image

# TokenAdapter import (optional - for aligned evaluation)
TokenAdapter = None
TokenAdapterConfig = None
try:
    from feasible.token_alignment.token_adapter import TokenAdapter, TokenAdapterConfig
except ImportError:
    pass  # TokenAdapter not available

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    print("Warning: PyAV not found. Install with: pip install av")


def get_gpu_memory_mb() -> Dict[str, float]:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def reset_gpu_memory_stats():
    """Reset GPU memory statistics for fresh measurement."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def load_video_frames_from_mp4(video_path: str, num_frames: int = 8) -> List:
    """Load frames from MP4 video, sampled uniformly using PyAV."""
    if not HAS_PYAV:
        raise ImportError("PyAV is required for MP4 video loading. Install with: pip install av")

    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames

    if total_frames == 0:
        total_frames = sum(1 for _ in container.decode(stream))
        container.seek(0)

    if total_frames >= num_frames:
        indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int))
    else:
        indices = set(range(total_frames))

    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            pil_img = frame.to_image()
            frames.append(pil_img)

    container.close()
    return frames


def compute_aligned_acceptance_rate(
    draft_tokens: List[int],
    target_tokens: List[int],
    token_adapter: 'TokenAdapter',
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute acceptance rate using TokenAdapter for alignment.

    The TokenAdapter predicts Video-LLaVA tokens from EventGPT tokens,
    improving the acceptance rate for speculative decoding.

    Args:
        draft_tokens: EventGPT generated token IDs
        target_tokens: Video-LLaVA generated token IDs
        token_adapter: Trained TokenAdapter model
        device: Device to use

    Returns:
        Dict with acceptance metrics for aligned predictions
    """
    try:
        if token_adapter is None:
            return {
                "aligned_acceptance_rate": 0.0,
                "aligned_matched": 0,
                "aligned_total": 0,
                "method": "no_adapter"
            }

        # Convert to tensor
        draft_tensor = torch.tensor([draft_tokens], device=device)

        # Get aligned predictions from TokenAdapter
        with torch.inference_mode():
            logits = token_adapter(draft_tensor)  # [1, seq, vocab]
            aligned_tokens = logits.argmax(dim=-1)[0].tolist()  # [seq]

        # Compare aligned tokens vs target tokens
        min_len = min(len(aligned_tokens), len(target_tokens))
        if min_len == 0:
            return {
                "aligned_acceptance_rate": 0.0,
                "aligned_matched": 0,
                "aligned_total": max(len(aligned_tokens), len(target_tokens)),
                "method": "length_mismatch"
            }

        # Count matches
        matched = sum(1 for i in range(min_len)
                     if aligned_tokens[i] == target_tokens[i])

        acceptance_rate = matched / len(target_tokens) if target_tokens else 0.0

        # Also compute top-5 acceptance
        top5_preds = logits.topk(5, dim=-1).indices[0]  # [seq, 5]
        top5_matched = 0
        for i in range(min(len(top5_preds), len(target_tokens))):
            if target_tokens[i] in top5_preds[i].tolist():
                top5_matched += 1
        top5_rate = top5_matched / len(target_tokens) if target_tokens else 0.0

        return {
            "aligned_acceptance_rate": float(acceptance_rate),
            "aligned_matched": int(matched),
            "aligned_total": int(len(target_tokens)),
            "aligned_top5_rate": float(top5_rate),
            "aligned_top5_matched": int(top5_matched),
            "method": "token_adapter"
        }

    except Exception as e:
        return {
            "aligned_acceptance_rate": 0.0,
            "aligned_matched": 0,
            "aligned_total": 0,
            "error": str(e),
            "method": "error"
        }


def compute_acceptance_rate(draft_tokens: List[int], target_tokens: List[int],
                            draft_text: str, target_text: str,
                            egpt_tokenizer=None, vl_tokenizer=None) -> Dict[str, float]:
    """
    Compute token-level acceptance rate from draft (EventGPT) to target (Video-LLaVA).

    In speculative decoding, draft tokens are accepted when they match target tokens.
    Since models use different tokenizers, we:
    1. Decode draft tokens using EventGPT tokenizer
    2. Re-tokenize with Video-LLaVA tokenizer
    3. Compare token sequences
    4. Calculate match percentage
    """
    try:
        if vl_tokenizer is None:
            # Fallback: use text-level comparison if tokenizers not provided
            # This shouldn't happen in practice but provides graceful degradation
            if not draft_text or not target_text:
                return {"acceptance_rate": 0.0, "matched_tokens": 0, "total_tokens": 0}

            # Simple lexical overlap as fallback
            draft_words = set(draft_text.lower().split())
            target_words = set(target_text.lower().split())
            matched = len(draft_words & target_words)
            total = max(len(draft_words), len(target_words), 1)
            acceptance_rate = matched / total if total > 0 else 0.0

            return {
                "acceptance_rate": float(acceptance_rate),
                "matched_tokens": matched,
                "total_tokens": total,
                "method": "fallback_text_overlap"
            }

        # Token-level acceptance: re-tokenize with target tokenizer
        # Re-tokenize draft output with Video-LLaVA tokenizer
        draft_tokens_vl = vl_tokenizer(draft_text, return_tensors=None,
                                       padding=False, truncation=False)['input_ids']

        # Tokenize target with Video-LLaVA tokenizer
        target_tokens_vl = vl_tokenizer(target_text, return_tensors=None,
                                        padding=False, truncation=False)['input_ids']

        # Compare token sequences at position level
        min_length = min(len(draft_tokens_vl), len(target_tokens_vl))
        if min_length == 0:
            return {
                "acceptance_rate": 0.0,
                "matched_tokens": 0,
                "total_tokens": max(len(draft_tokens_vl), len(target_tokens_vl)),
                "method": "token_level_mismatch"
            }

        # Count exact token matches
        matched_tokens = sum(1 for i in range(min_length)
                            if draft_tokens_vl[i] == target_tokens_vl[i])

        # Acceptance rate is based on matches relative to target length
        # (or we could use draft length - both are reasonable)
        acceptance_rate = matched_tokens / len(target_tokens_vl) if target_tokens_vl else 0.0

        return {
            "acceptance_rate": float(acceptance_rate),
            "matched_tokens": int(matched_tokens),
            "total_tokens": int(len(target_tokens_vl)),
            "draft_tokens": int(len(draft_tokens_vl)),
            "method": "token_level_matching"
        }

    except Exception as e:
        # If tokenization fails, return zero acceptance
        return {
            "acceptance_rate": 0.0,
            "matched_tokens": 0,
            "total_tokens": 0,
            "error": str(e),
            "method": "error"
        }


def benchmark_eventgpt_5stages(
    model, tokenizer, processor,
    sample: Dict, dataset_dir: str, query: str,
    max_new_tokens: int = 50
) -> Optional[Dict]:
    """Benchmark EventGPT with all 5 stages timed separately."""
    from model.EventChatModel import get_spatio_temporal_features
    from dataset.conversation import prepare_event_prompt
    from dataset.constants import EVENT_TOKEN_INDEX
    from common.common import tokenizer_event_token

    device = "cuda"
    result = {"model": "EventGPT"}

    # Track GPU memory before inference
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024 * 1024)

    try:
        if "event_image" not in sample or not sample["event_image"]:
            return None

        event_image_paths = sample["event_image"]

        # ========== STAGE 1: Data Loading ==========
        torch.cuda.synchronize()
        stage1_start = time.time()
        img_path = os.path.join(dataset_dir, "event_image", event_image_paths[0])
        img = load_image(img_path)
        img_array = np.array(img)
        event_image_size = list(img_array.shape[:2])
        torch.cuda.synchronize()
        result["stage1_time"] = time.time() - stage1_start

        # ========== STAGE 2: Preprocessing ==========
        torch.cuda.synchronize()
        stage2_start = time.time()
        event = processor(img_array, return_tensors='pt')['pixel_values'][0]
        event = event.to(device, dtype=torch.bfloat16)

        conv_mode = 'eventgpt_v1'
        prompt = prepare_event_prompt(query, conv_mode)
        input_ids = tokenizer_event_token(
            prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)
        torch.cuda.synchronize()
        result["stage2_time"] = time.time() - stage2_start

        # ========== STAGE 3: Vision Encoding ==========
        torch.cuda.synchronize()
        stage3_start = time.time()
        with torch.inference_mode():
            feature = model.visval_encode(event.unsqueeze(0))
            feature = model.get_model().feature_adaptor(feature)
            feature = feature.squeeze(0)
            event_features = get_spatio_temporal_features([feature])
            event_features = event_features.unsqueeze(0)
        torch.cuda.synchronize()
        result["stage3_time"] = time.time() - stage3_start

        # ========== STAGE 4: LLM Prefill ==========
        torch.cuda.synchronize()
        stage4_start = time.time()
        with torch.inference_mode():
            (
                _,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                torch.ones_like(input_ids, dtype=torch.bool),
                None,
                None,
                event_tensors=None,
                event_image_sizes=event_image_size,
                event_features=event_features,
            )

            if attention_mask is None:
                attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.bool, device=device)
            if position_ids is None:
                position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=device).unsqueeze(0)

            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=True,
            )

            logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
        torch.cuda.synchronize()
        result["stage4_time"] = time.time() - stage4_start
        result["prefill_length"] = inputs_embeds.shape[1]

        # ========== STAGE 5: LLM Decode ==========
        torch.cuda.synchronize()
        stage5_start = time.time()
        output_token_ids = []
        token_times = []  # Track per-token generation time

        with torch.inference_mode():
            cur_pos = result["prefill_length"]
            kv_cache = outputs.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            output_token_ids.append(next_token.item())

            for _ in range(max_new_tokens - 1):
                token_start = time.time()
                cur_embed = model.get_model().embed_tokens(next_token)
                new_attention_mask = torch.ones((1, cur_pos + 1), dtype=torch.bool, device=device)

                outputs = model.model(
                    inputs_embeds=cur_embed,
                    attention_mask=new_attention_mask,
                    position_ids=torch.tensor([[cur_pos]], device=device),
                    past_key_values=kv_cache,
                    use_cache=True,
                )

                logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
                kv_cache = outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                token_id = next_token.item()
                output_token_ids.append(token_id)
                cur_pos += 1
                torch.cuda.synchronize()
                token_times.append(time.time() - token_start)

                if token_id == tokenizer.eos_token_id:
                    break

        torch.cuda.synchronize()
        result["stage5_time"] = time.time() - stage5_start
        result["output_tokens"] = output_token_ids
        result["output_text"] = tokenizer.decode(output_token_ids, skip_special_tokens=True)
        result["num_output_tokens"] = len(output_token_ids)
        result["tokens_per_sec"] = len(output_token_ids) / result["stage5_time"] if result["stage5_time"] > 0 else 0
        result["avg_token_time"] = np.mean(token_times) if token_times else 0
        result["total_time"] = sum([result["stage1_time"], result["stage2_time"],
                                    result["stage3_time"], result["stage4_time"], result["stage5_time"]])
        result["prefill_complete_time"] = sum([result["stage1_time"], result["stage2_time"],
                                               result["stage3_time"], result["stage4_time"]])

        # Track GPU memory after inference
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            result["gpu_memory_mb"] = {
                "before": mem_before,
                "after": mem_after,
                "peak": peak_mem,
                "inference_delta": mem_after - mem_before,
            }

        return result

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()[-500:]}


def benchmark_videollava_5stages(
    model, processor,
    sample: Dict, dataset_dir: str, query: str,
    max_new_tokens: int = 50
) -> Optional[Dict]:
    """Benchmark Video-LLaVA with all 5 stages timed separately."""
    device = "cuda"
    result = {"model": "Video-LLaVA"}

    # Track GPU memory before inference
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024 * 1024)

    try:
        video_data = sample.get("video_data")

        # ========== STAGE 1: Data Loading ==========
        torch.cuda.synchronize()
        stage1_start = time.time()
        if video_data:
            video_path = os.path.join(dataset_dir, "mp4", video_data + ".mp4")
            if not os.path.exists(video_path):
                return None
            rgb_images = load_video_frames_from_mp4(video_path, num_frames=8)
        else:
            if "event_image" not in sample or not sample["event_image"]:
                return None
            event_image_paths = sample["event_image"]
            rgb_images = []
            for img_path in event_image_paths[:8]:
                full_path = os.path.join(dataset_dir, "event_image", img_path)
                img = load_image(full_path)
                rgb_images.append(img)
            while len(rgb_images) < 8:
                rgb_images.append(rgb_images[-1])

        if len(rgb_images) == 0:
            return None
        torch.cuda.synchronize()
        result["stage1_time"] = time.time() - stage1_start
        result["num_frames"] = len(rgb_images)

        # ========== STAGE 2: Preprocessing ==========
        torch.cuda.synchronize()
        stage2_start = time.time()
        prompt = f"USER: <video>\n{query}\nASSISTANT:"
        inputs = processor(text=prompt, videos=rgb_images, return_tensors="pt")

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask')
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(device)

        pixel_values_videos = inputs.get('pixel_values_videos')
        torch.cuda.synchronize()
        result["stage2_time"] = time.time() - stage2_start

        if pixel_values_videos is None:
            return {"error": f"No pixel_values_videos found. Keys: {list(inputs.keys())}"}

        # ========== STAGE 3: Vision Encoding ==========
        # Note: For Video-LLaVA with device_map="auto", vision encoding happens in Stage 4
        # We measure a minimal time here for consistency
        torch.cuda.synchronize()
        stage3_start = time.time()
        # Vision tower runs as part of the forward pass in Stage 4
        torch.cuda.synchronize()
        result["stage3_time"] = time.time() - stage3_start

        # ========== STAGE 4: LLM Prefill ==========
        torch.cuda.synchronize()
        stage4_start = time.time()
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos.to(device),
                past_key_values=None,
                use_cache=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        torch.cuda.synchronize()
        result["stage4_time"] = time.time() - stage4_start
        result["prefill_length"] = input_ids.shape[1]
        if pixel_values_videos is not None:
            # Video tokens: 8 frames * 256 tokens per frame = 2048
            result["prefill_length"] += pixel_values_videos.shape[1] * 256

        result["prefill_complete_time"] = sum([result["stage1_time"], result["stage2_time"],
                                               result["stage3_time"], result["stage4_time"]])

        # ========== STAGE 5: LLM Decode ==========
        torch.cuda.synchronize()
        stage5_start = time.time()
        output_token_ids = [next_token.item()]
        token_times = []

        with torch.inference_mode():
            cur_token = next_token
            # Determine attention mask dtype safely
            attn_dtype = attention_mask.dtype if isinstance(attention_mask, torch.Tensor) else torch.long
            cur_attention_mask = torch.ones(
                (1, past_key_values[0][0].shape[2] + 1),
                dtype=attn_dtype,
                device=device
            )

            for _ in range(max_new_tokens - 1):
                token_start = time.time()
                outputs = model(
                    input_ids=cur_token,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits = outputs.logits
                past_key_values = outputs.past_key_values

                next_token_logits = logits[:, -1, :]
                cur_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                token_id = cur_token.item()
                output_token_ids.append(token_id)

                cur_attention_mask = torch.ones(
                    (1, cur_attention_mask.shape[1] + 1),
                    dtype=attn_dtype, device=device
                )

                torch.cuda.synchronize()
                token_times.append(time.time() - token_start)

                if token_id == processor.tokenizer.eos_token_id:
                    break

        torch.cuda.synchronize()
        result["stage5_time"] = time.time() - stage5_start
        result["output_tokens"] = output_token_ids
        result["output_text"] = processor.tokenizer.decode(output_token_ids, skip_special_tokens=True)
        result["num_output_tokens"] = len(output_token_ids)
        result["tokens_per_sec"] = len(output_token_ids) / result["stage5_time"] if result["stage5_time"] > 0 else 0
        result["avg_token_time"] = np.mean(token_times) if token_times else 0
        result["total_time"] = sum([result["stage1_time"], result["stage2_time"],
                                    result["stage3_time"], result["stage4_time"], result["stage5_time"]])

        # Track GPU memory after inference
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            result["gpu_memory_mb"] = {
                "before": mem_before,
                "after": mem_after,
                "peak": peak_mem,
                "inference_delta": mem_after - mem_before,
            }

        return result

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()[-500:]}


def compute_parallel_metrics(egpt_result: Dict, vl_result: Dict) -> Dict:
    """
    Compute parallel execution metrics:
    - Overlap window: time available for free draft generation
    - Hidden tokens: tokens that can be generated during VL's prefill
    - Wall-clock time: actual parallel execution time
    """
    egpt_prefill = egpt_result.get("prefill_complete_time", 0)
    vl_prefill = vl_result.get("prefill_complete_time", 0)

    # Overlap window: VL prefill is slower, so EGPT can generate tokens during the gap
    overlap_window = max(0, vl_prefill - egpt_prefill)

    # Calculate how many tokens EGPT can generate in the overlap window
    egpt_token_rate = egpt_result.get("tokens_per_sec", 0)
    hidden_tokens = int(overlap_window * egpt_token_rate) if egpt_token_rate > 0 else 0
    hidden_tokens = min(hidden_tokens, egpt_result.get("num_output_tokens", 0))

    # Wall-clock time for parallel execution
    # In parallel: start both, VL finishes prefill, then both decode
    # Wall time = max(EGPT_prefill, VL_prefill) + max(EGPT_decode - hidden_time, VL_decode)
    egpt_decode_time = egpt_result.get("stage5_time", 0)
    vl_decode_time = vl_result.get("stage5_time", 0)

    # Time EGPT spends decoding that's "hidden" in VL prefill
    hidden_decode_time = min(overlap_window, egpt_decode_time)
    egpt_remaining_decode = max(0, egpt_decode_time - hidden_decode_time)

    # Wall-clock: parallel prefill phase + parallel decode phase
    wall_clock_prefill = max(egpt_prefill, vl_prefill)
    wall_clock_decode = max(egpt_remaining_decode, vl_decode_time)
    wall_clock_total = wall_clock_prefill + wall_clock_decode

    # Sequential baseline (if we ran both models one after another)
    sequential_total = egpt_result.get("total_time", 0) + vl_result.get("total_time", 0)

    # Parallel efficiency
    parallel_speedup = sequential_total / wall_clock_total if wall_clock_total > 0 else 0

    return {
        "egpt_prefill_complete": egpt_prefill,
        "vl_prefill_complete": vl_prefill,
        "overlap_window": overlap_window,
        "hidden_tokens": hidden_tokens,
        "hidden_decode_time": hidden_decode_time,
        "wall_clock_prefill": wall_clock_prefill,
        "wall_clock_decode": wall_clock_decode,
        "wall_clock_total": wall_clock_total,
        "sequential_total": sequential_total,
        "parallel_speedup": parallel_speedup,
        "egpt_tokens_generated": egpt_result.get("num_output_tokens", 0),
        "vl_tokens_generated": vl_result.get("num_output_tokens", 0),
    }


def run_5stage_benchmark(
    eventgpt_model, egpt_tokenizer, egpt_processor,
    videollava_model, vl_processor,
    dataset: List, dataset_dir: str,
    max_samples: int = -1,
    max_new_tokens: int = 50,
    token_adapter=None,
) -> Dict:
    """Run 5-stage benchmark on dataset with parallel timing analysis.

    Args:
        eventgpt_model: EventGPT model
        egpt_tokenizer: EventGPT tokenizer
        egpt_processor: EventGPT processor
        videollava_model: Video-LLaVA model
        vl_processor: Video-LLaVA processor
        dataset: Dataset list
        dataset_dir: Dataset directory path
        max_samples: Max samples to benchmark (-1 for all)
        max_new_tokens: Max tokens to generate
        token_adapter: Optional TokenAdapter for aligned evaluation
    """
    print("\n" + "=" * 80)
    print(f"Running 5-Stage Parallel Prefill Benchmark")
    print(f"Max samples: {max_samples if max_samples != -1 else 'all'}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Token Adapter: {'Enabled' if token_adapter is not None else 'Disabled'}")
    print("=" * 80)

    query = "What are the key elements in this scene?"
    results = []
    errors = []

    samples_to_process = dataset if max_samples == -1 else dataset[:max_samples]

    for sample_idx, sample in enumerate(tqdm(samples_to_process, desc="5-Stage Benchmark")):
        try:
            # Run EventGPT benchmark
            egpt_result = benchmark_eventgpt_5stages(
                eventgpt_model, egpt_tokenizer, egpt_processor,
                sample, dataset_dir, query, max_new_tokens
            )

            # Run Video-LLaVA benchmark
            vl_result = benchmark_videollava_5stages(
                videollava_model, vl_processor,
                sample, dataset_dir, query, max_new_tokens
            )

            if egpt_result and "error" not in egpt_result and vl_result and "error" not in vl_result:
                # Compute baseline acceptance rate (direct token matching)
                acceptance = compute_acceptance_rate(
                    egpt_result["output_tokens"],
                    vl_result["output_tokens"],
                    egpt_result["output_text"],
                    vl_result["output_text"],
                    egpt_tokenizer=egpt_tokenizer,
                    vl_tokenizer=vl_processor.tokenizer,
                )

                # Compute aligned acceptance rate (with TokenAdapter)
                aligned_acceptance = {}
                if token_adapter is not None:
                    aligned_acceptance = compute_aligned_acceptance_rate(
                        egpt_result["output_tokens"],
                        vl_result["output_tokens"],
                        token_adapter,
                        device="cuda",
                    )

                # Compute parallel execution metrics
                parallel = compute_parallel_metrics(egpt_result, vl_result)

                results.append({
                    "sample_idx": sample_idx,
                    "eventgpt": egpt_result,
                    "videollava": vl_result,
                    "acceptance": acceptance,
                    "aligned_acceptance": aligned_acceptance,
                    "parallel": parallel,
                })
            else:
                errors.append({
                    "sample_idx": sample_idx,
                    "egpt_error": egpt_result.get("error") if egpt_result else "No result",
                    "vl_error": vl_result.get("error") if vl_result else "No result",
                })

        except Exception as e:
            errors.append({
                "sample_idx": sample_idx,
                "error": str(e),
            })

    return {"results": results, "errors": errors}


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute aggregate statistics from benchmark results."""
    if not results:
        return {}

    def safe_avg(values):
        return float(np.mean(values)) if values else 0.0

    def safe_std(values):
        return float(np.std(values)) if values else 0.0

    # Extract values
    acceptance_rates = [r.get("acceptance", {}).get("acceptance_rate", 0) for r in results]

    egpt_stats = {
        "stage1_time_avg": safe_avg([r["eventgpt"]["stage1_time"] for r in results]),
        "stage1_time_std": safe_std([r["eventgpt"]["stage1_time"] for r in results]),
        "stage2_time_avg": safe_avg([r["eventgpt"]["stage2_time"] for r in results]),
        "stage2_time_std": safe_std([r["eventgpt"]["stage2_time"] for r in results]),
        "stage3_time_avg": safe_avg([r["eventgpt"]["stage3_time"] for r in results]),
        "stage3_time_std": safe_std([r["eventgpt"]["stage3_time"] for r in results]),
        "stage4_time_avg": safe_avg([r["eventgpt"]["stage4_time"] for r in results]),
        "stage4_time_std": safe_std([r["eventgpt"]["stage4_time"] for r in results]),
        "stage5_time_avg": safe_avg([r["eventgpt"]["stage5_time"] for r in results]),
        "stage5_time_std": safe_std([r["eventgpt"]["stage5_time"] for r in results]),
        "prefill_length_avg": safe_avg([r["eventgpt"]["prefill_length"] for r in results]),
        "output_tokens_avg": safe_avg([r["eventgpt"]["num_output_tokens"] for r in results]),
        "tokens_per_sec_avg": safe_avg([r["eventgpt"]["tokens_per_sec"] for r in results]),
        "total_time_avg": safe_avg([r["eventgpt"]["total_time"] for r in results]),
        "prefill_complete_time_avg": safe_avg([r["eventgpt"]["prefill_complete_time"] for r in results]),
        "gpu_peak_memory_avg": safe_avg([r["eventgpt"].get("gpu_memory_mb", {}).get("peak", 0) for r in results]),
    }

    vl_stats = {
        "stage1_time_avg": safe_avg([r["videollava"]["stage1_time"] for r in results]),
        "stage1_time_std": safe_std([r["videollava"]["stage1_time"] for r in results]),
        "stage2_time_avg": safe_avg([r["videollava"]["stage2_time"] for r in results]),
        "stage2_time_std": safe_std([r["videollava"]["stage2_time"] for r in results]),
        "stage3_time_avg": safe_avg([r["videollava"]["stage3_time"] for r in results]),
        "stage3_time_std": safe_std([r["videollava"]["stage3_time"] for r in results]),
        "stage4_time_avg": safe_avg([r["videollava"]["stage4_time"] for r in results]),
        "stage4_time_std": safe_std([r["videollava"]["stage4_time"] for r in results]),
        "stage5_time_avg": safe_avg([r["videollava"]["stage5_time"] for r in results]),
        "stage5_time_std": safe_std([r["videollava"]["stage5_time"] for r in results]),
        "prefill_length_avg": safe_avg([r["videollava"]["prefill_length"] for r in results]),
        "output_tokens_avg": safe_avg([r["videollava"]["num_output_tokens"] for r in results]),
        "tokens_per_sec_avg": safe_avg([r["videollava"]["tokens_per_sec"] for r in results]),
        "total_time_avg": safe_avg([r["videollava"]["total_time"] for r in results]),
        "prefill_complete_time_avg": safe_avg([r["videollava"]["prefill_complete_time"] for r in results]),
        "gpu_peak_memory_avg": safe_avg([r["videollava"].get("gpu_memory_mb", {}).get("peak", 0) for r in results]),
    }

    parallel_stats = {
        "overlap_window_avg": safe_avg([r["parallel"]["overlap_window"] for r in results]),
        "hidden_tokens_avg": safe_avg([r["parallel"]["hidden_tokens"] for r in results]),
        "wall_clock_total_avg": safe_avg([r["parallel"]["wall_clock_total"] for r in results]),
        "sequential_total_avg": safe_avg([r["parallel"]["sequential_total"] for r in results]),
        "parallel_speedup_avg": safe_avg([r["parallel"]["parallel_speedup"] for r in results]),
    }

    # Additional acceptance metrics
    matched_tokens_per_result = [r.get("acceptance", {}).get("matched_tokens", 0) for r in results]
    total_tokens_per_result = [r.get("acceptance", {}).get("total_tokens", 0) for r in results]

    # Aligned acceptance metrics (with TokenAdapter)
    aligned_acceptance_rates = [r.get("aligned_acceptance", {}).get("aligned_acceptance_rate", 0) for r in results]
    aligned_top5_rates = [r.get("aligned_acceptance", {}).get("aligned_top5_rate", 0) for r in results]
    has_aligned = any(r.get("aligned_acceptance", {}).get("method") == "token_adapter" for r in results)

    stats_dict = {
        "n_samples": len(results),
        "acceptance_rate_avg": safe_avg(acceptance_rates),
        "acceptance_rate_std": safe_std(acceptance_rates),
        "matched_tokens_total": sum(matched_tokens_per_result),
        "total_tokens_compared": sum(total_tokens_per_result),
        "eventgpt": egpt_stats,
        "videollava": vl_stats,
        "parallel": parallel_stats,
    }

    # Add aligned metrics if TokenAdapter was used
    if has_aligned:
        stats_dict["aligned_acceptance_rate_avg"] = safe_avg(aligned_acceptance_rates)
        stats_dict["aligned_acceptance_rate_std"] = safe_std(aligned_acceptance_rates)
        stats_dict["aligned_top5_rate_avg"] = safe_avg(aligned_top5_rates)
        stats_dict["aligned_improvement"] = safe_avg(aligned_acceptance_rates) - safe_avg(acceptance_rates)

    return stats_dict


def generate_markdown_report(stats: Dict, results: List[Dict], dataset_name: str,
                            output_path: str, config: Dict, memory_stats: Dict = None):
    """Generate comprehensive markdown report."""
    egpt = stats.get("eventgpt", {})
    vl = stats.get("videollava", {})
    par = stats.get("parallel", {})
    mem = memory_stats or {}

    # Sample outputs for verification
    sample_outputs = []
    for r in results[:3]:  # First 3 samples
        sample_outputs.append({
            "idx": r["sample_idx"],
            "egpt_text": r["eventgpt"]["output_text"][:200] + "..." if len(r["eventgpt"]["output_text"]) > 200 else r["eventgpt"]["output_text"],
            "egpt_tokens": r["eventgpt"]["num_output_tokens"],
            "vl_text": r["videollava"]["output_text"][:200] + "..." if len(r["videollava"]["output_text"]) > 200 else r["videollava"]["output_text"],
            "vl_tokens": r["videollava"]["num_output_tokens"],
        })

    report = f"""# Parallel Prefill Benchmark Report: {dataset_name}

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Samples:** {stats.get('n_samples', 0)}
**Max New Tokens:** {config.get('max_new_tokens', 50)}

## Executive Summary

This benchmark measures how many EventGPT draft tokens can be "hidden" within Video-LLaVA's
slower prefill phase. EventGPT's faster prefill creates an overlap window for free token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | {par.get('overlap_window_avg', 0)*1000:.1f} ms |
| **Hidden Tokens** | {par.get('hidden_tokens_avg', 0):.1f} tokens |
| **Parallel Speedup** | {par.get('parallel_speedup_avg', 0):.2f}x |
| **Acceptance Rate** | {stats.get('acceptance_rate_avg', 0)*100:.1f}% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | {egpt.get('stage1_time_avg', 0)*1000:.1f} ± {egpt.get('stage1_time_std', 0)*1000:.1f} ms | {vl.get('stage1_time_avg', 0)*1000:.1f} ± {vl.get('stage1_time_std', 0)*1000:.1f} ms | {vl.get('stage1_time_avg', 0.0001)/max(egpt.get('stage1_time_avg', 0.0001), 0.0001):.2f}x |
| **Stage 2: Preprocessing** | {egpt.get('stage2_time_avg', 0)*1000:.1f} ± {egpt.get('stage2_time_std', 0)*1000:.1f} ms | {vl.get('stage2_time_avg', 0)*1000:.1f} ± {vl.get('stage2_time_std', 0)*1000:.1f} ms | {vl.get('stage2_time_avg', 0.0001)/max(egpt.get('stage2_time_avg', 0.0001), 0.0001):.2f}x |
| **Stage 3: Vision Encoding** | {egpt.get('stage3_time_avg', 0)*1000:.1f} ± {egpt.get('stage3_time_std', 0)*1000:.1f} ms | {vl.get('stage3_time_avg', 0)*1000:.1f} ± {vl.get('stage3_time_std', 0)*1000:.1f} ms | {vl.get('stage3_time_avg', 0.0001)/max(egpt.get('stage3_time_avg', 0.0001), 0.0001):.2f}x |
| **Stage 4: LLM Prefill** | {egpt.get('stage4_time_avg', 0)*1000:.1f} ± {egpt.get('stage4_time_std', 0)*1000:.1f} ms | {vl.get('stage4_time_avg', 0)*1000:.1f} ± {vl.get('stage4_time_std', 0)*1000:.1f} ms | {vl.get('stage4_time_avg', 0.0001)/max(egpt.get('stage4_time_avg', 0.0001), 0.0001):.2f}x |
| **Stage 5: LLM Decode** | {egpt.get('stage5_time_avg', 0)*1000:.1f} ± {egpt.get('stage5_time_std', 0)*1000:.1f} ms | {vl.get('stage5_time_avg', 0)*1000:.1f} ± {vl.get('stage5_time_std', 0)*1000:.1f} ms | {vl.get('stage5_time_avg', 0.0001)/max(egpt.get('stage5_time_avg', 0.0001), 0.0001):.2f}x |
| **TOTAL** | {egpt.get('total_time_avg', 0)*1000:.1f} ms | {vl.get('total_time_avg', 0)*1000:.1f} ms | {vl.get('total_time_avg', 0.0001)/max(egpt.get('total_time_avg', 0.0001), 0.0001):.2f}x |

## Prefill Time Comparison

| Model | Prefill Complete (S1-S4) | Prefill Length | Throughput |
|-------|--------------------------|----------------|------------|
| EventGPT | {egpt.get('prefill_complete_time_avg', 0)*1000:.1f} ms | {egpt.get('prefill_length_avg', 0):.0f} tokens | {egpt.get('prefill_length_avg', 0)/max(egpt.get('prefill_complete_time_avg', 0.0001), 0.0001):.0f} tok/s |
| Video-LLaVA | {vl.get('prefill_complete_time_avg', 0)*1000:.1f} ms | {vl.get('prefill_length_avg', 0):.0f} tokens | {vl.get('prefill_length_avg', 0)/max(vl.get('prefill_complete_time_avg', 0.0001), 0.0001):.0f} tok/s |

## Token Generation Statistics

| Model | Output Tokens (avg) | Tokens/sec | Decode Time |
|-------|---------------------|------------|-------------|
| EventGPT | {egpt.get('output_tokens_avg', 0):.1f} | {egpt.get('tokens_per_sec_avg', 0):.1f} | {egpt.get('stage5_time_avg', 0)*1000:.1f} ms |
| Video-LLaVA | {vl.get('output_tokens_avg', 0):.1f} | {vl.get('tokens_per_sec_avg', 0):.1f} | {vl.get('stage5_time_avg', 0)*1000:.1f} ms |

## GPU Memory Usage (4-bit Quantization)

| Model | Model Size | Peak Inference Memory |
|-------|------------|----------------------|
| EventGPT | {mem.get('eventgpt_model_mb', 0):.0f} MB | {egpt.get('gpu_peak_memory_avg', 0):.0f} MB |
| Video-LLaVA | {mem.get('videollava_model_mb', 0):.0f} MB | {vl.get('gpu_peak_memory_avg', 0):.0f} MB |
| **Total (both models)** | {mem.get('total_models_mb', 0):.0f} MB | - |

## Parallel Execution Analysis

### Token Hiding Opportunity

```
Timeline (Parallel Execution):

  EventGPT:  |--Prefill--|---Decode (hidden)---|---Decode (visible)---|
                         ^                      ^
  Video-LLaVA: |--------Prefill (slow)---------|--------Decode--------|

              |<------ Overlap Window -------->|
                     ({par.get('overlap_window_avg', 0)*1000:.0f} ms)
                     ({par.get('hidden_tokens_avg', 0):.0f} tokens hidden)
```

| Metric | Value |
|--------|-------|
| EventGPT Prefill Complete | {egpt.get('prefill_complete_time_avg', 0)*1000:.1f} ms |
| Video-LLaVA Prefill Complete | {vl.get('prefill_complete_time_avg', 0)*1000:.1f} ms |
| **Overlap Window** | {par.get('overlap_window_avg', 0)*1000:.1f} ms |
| **Hidden Tokens** | {par.get('hidden_tokens_avg', 0):.1f} tokens |
| Wall-Clock Time (parallel) | {par.get('wall_clock_total_avg', 0)*1000:.1f} ms |
| Sequential Time (baseline) | {par.get('sequential_total_avg', 0)*1000:.1f} ms |
| **Parallel Speedup** | {par.get('parallel_speedup_avg', 0):.2f}x |

## Sample Output Verification

"""

    for s in sample_outputs:
        report += f"""### Sample {s['idx']}

**EventGPT** ({s['egpt_tokens']} tokens):
> {s['egpt_text']}

**Video-LLaVA** ({s['vl_tokens']} tokens):
> {s['vl_text']}

---

"""

    report += f"""
## Key Findings

1. **Prefill Speedup**: EventGPT prefill is {vl.get('prefill_complete_time_avg', 0.0001)/max(egpt.get('prefill_complete_time_avg', 0.0001), 0.0001):.2f}x faster than Video-LLaVA
2. **Overlap Window**: {par.get('overlap_window_avg', 0)*1000:.1f}ms available for free draft token generation
3. **Hidden Tokens**: ~{par.get('hidden_tokens_avg', 0):.0f} EventGPT tokens can be generated "for free"
4. **Acceptance Rate**: {stats.get('acceptance_rate_avg', 0)*100:.1f}% - drafts are {"well" if stats.get('acceptance_rate_avg', 0) > 0.5 else "partially"} aligned

## Implications for Speculative Decoding

With {par.get('hidden_tokens_avg', 0):.0f} hidden tokens and {stats.get('acceptance_rate_avg', 0)*100:.1f}% acceptance rate:
- Expected accepted tokens per batch: {par.get('hidden_tokens_avg', 0) * stats.get('acceptance_rate_avg', 0):.1f}
- Effective speedup potential: {1 + par.get('hidden_tokens_avg', 0) * stats.get('acceptance_rate_avg', 0):.2f}x

---

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*Author: Alice Zhang*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"✓ Markdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="5-stage parallel prefill benchmark")
    parser.add_argument("--dataset_dir", type=str,
                        default="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Number of samples to benchmark (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum tokens to generate per sample")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    # Token Adapter arguments
    parser.add_argument("--use_token_adapter", action="store_true",
                        help="Use TokenAdapter for aligned evaluation")
    parser.add_argument("--token_adapter_path", type=str, default=None,
                        help="Path to trained TokenAdapter checkpoint (default: auto-detect latest)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    json_path = os.path.join(args.dataset_dir, "EventGPT_Instruction_Subset.json")
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))

    # Load models with 4-bit quantization
    print("\n" + "=" * 80)
    print("Loading Models (4-bit Quantization)")
    print("=" * 80)

    from model.EventChatModel import EventChatModel

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Track GPU memory usage
    memory_stats = {}

    print("\n[1/2] Loading EventGPT (4-bit)...")
    reset_gpu_memory_stats()
    eventgpt_path = "./checkpoints/EventGPT-7b"
    eventgpt_model = EventChatModel.from_pretrained(
        eventgpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    eventgpt_model.eval()
    egpt_tokenizer = AutoTokenizer.from_pretrained(eventgpt_path, use_fast=True)
    egpt_processor = eventgpt_model.get_visual_tower().event_processor
    memory_stats["eventgpt_load"] = get_gpu_memory_mb()
    print(f"  GPU Memory: {memory_stats['eventgpt_load']['allocated_mb']:.0f} MB allocated")

    print("\n[2/2] Loading Video-LLaVA (4-bit)...")
    mem_before_vl = get_gpu_memory_mb()
    videollava_model_id = "LanguageBind/Video-LLaVA-7B-hf"
    videollava_model = VideoLlavaForConditionalGeneration.from_pretrained(
        videollava_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    vl_processor = VideoLlavaProcessor.from_pretrained(videollava_model_id)
    videollava_model.eval()
    memory_stats["videollava_load"] = get_gpu_memory_mb()
    memory_stats["videollava_model_mb"] = memory_stats["videollava_load"]["allocated_mb"] - mem_before_vl["allocated_mb"]
    memory_stats["eventgpt_model_mb"] = mem_before_vl["allocated_mb"]
    memory_stats["total_models_mb"] = memory_stats["videollava_load"]["allocated_mb"]
    print(f"  GPU Memory: {memory_stats['videollava_model_mb']:.0f} MB (Video-LLaVA)")
    print(f"  Total GPU Memory: {memory_stats['total_models_mb']:.0f} MB (both models)")

    print("\n✓ Both models loaded successfully (4-bit quantized)")

    # Load TokenAdapter if requested
    token_adapter = None
    if args.use_token_adapter:
        if TokenAdapter is None:
            print("\n⚠️  TokenAdapter not available. Install from feasible/token_alignment/")
        else:
            # Auto-detect latest checkpoint if not specified
            adapter_path = args.token_adapter_path
            if adapter_path is None:
                # Look for latest task folder with best_model.pt
                task_dirs = sorted(Path("./feasible/token_alignment/task").glob("*_*"))
                for task_dir in reversed(task_dirs):
                    checkpoint = task_dir / "best_model.pt"
                    if checkpoint.exists():
                        adapter_path = str(checkpoint)
                        break

            if adapter_path and os.path.exists(adapter_path):
                print(f"\n[3/3] Loading TokenAdapter from {adapter_path}...")
                config = TokenAdapterConfig(
                    draft_vocab_size=32000,
                    target_vocab_size=32000,
                    embed_dim=512,
                    num_layers=4,
                    num_heads=8,
                    ffn_dim=2048,
                )
                token_adapter = TokenAdapter(config)
                checkpoint = torch.load(adapter_path, map_location="cuda")
                token_adapter.load_state_dict(checkpoint['model_state_dict'])
                token_adapter.to("cuda")
                token_adapter.eval()
                print(f"✓ TokenAdapter loaded ({sum(p.numel() for p in token_adapter.parameters()):,} params)")
            else:
                print(f"\n⚠️  TokenAdapter checkpoint not found at {adapter_path}")
                print("   Train one first with: python feasible/token_alignment/train_and_evaluate.py")

    # Run benchmark
    output = run_5stage_benchmark(
        eventgpt_model, egpt_tokenizer, egpt_processor,
        videollava_model, vl_processor,
        dataset, args.dataset_dir,
        args.max_samples,
        args.max_new_tokens,
        token_adapter=token_adapter,
    )

    # Compute statistics
    stats = compute_statistics(output["results"])

    # Save results
    results_data = {
        "timestamp": timestamp,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name,
        "config": {
            "quantization": "4bit",
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "use_token_adapter": args.use_token_adapter,
            "token_adapter_path": args.token_adapter_path if token_adapter is not None else None,
        },
        "gpu_memory": {
            "eventgpt_model_mb": memory_stats.get("eventgpt_model_mb", 0),
            "videollava_model_mb": memory_stats.get("videollava_model_mb", 0),
            "total_models_mb": memory_stats.get("total_models_mb", 0),
        },
        "statistics": stats,
        "results": output["results"],
        "errors": output["errors"],
    }

    json_path = os.path.join(args.output_dir, f"parallel_prefill_5stages_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n✓ Results saved to {json_path}")

    # Generate markdown report
    markdown_path = os.path.join(args.output_dir, f"parallel_prefill_5stages_{timestamp}.md")
    generate_markdown_report(stats, output["results"], dataset_name, markdown_path,
                             {"max_samples": args.max_samples, "max_new_tokens": args.max_new_tokens,
                              "device": args.device}, memory_stats)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    if stats:
        print(f"\nSamples processed: {stats['n_samples']}")
        print(f"Baseline Acceptance Rate: {stats['acceptance_rate_avg']*100:.1f}%")

        # Print aligned metrics if available
        if 'aligned_acceptance_rate_avg' in stats:
            print(f"Aligned Acceptance Rate:  {stats['aligned_acceptance_rate_avg']*100:.1f}% (with TokenAdapter)")
            print(f"Aligned Top-5 Rate:       {stats['aligned_top5_rate_avg']*100:.1f}%")
            print(f"Improvement:              +{stats['aligned_improvement']*100:.1f}%")

        print(f"\n{'─'*40}")
        print("EventGPT 5-Stage Breakdown:")
        print(f"{'─'*40}")
        print(f"  Stage 1 (Data Load):   {stats['eventgpt']['stage1_time_avg']*1000:.1f} ms")
        print(f"  Stage 2 (Preprocess):  {stats['eventgpt']['stage2_time_avg']*1000:.1f} ms")
        print(f"  Stage 3 (Vision):      {stats['eventgpt']['stage3_time_avg']*1000:.1f} ms")
        print(f"  Stage 4 (Prefill):     {stats['eventgpt']['stage4_time_avg']*1000:.1f} ms")
        print(f"  Stage 5 (Decode):      {stats['eventgpt']['stage5_time_avg']*1000:.1f} ms")
        print(f"  TOTAL:                 {stats['eventgpt']['total_time_avg']*1000:.1f} ms")
        print(f"  Output tokens:         {stats['eventgpt']['output_tokens_avg']:.1f} avg")

        print(f"\n{'─'*40}")
        print("Video-LLaVA 5-Stage Breakdown:")
        print(f"{'─'*40}")
        print(f"  Stage 1 (Data Load):   {stats['videollava']['stage1_time_avg']*1000:.1f} ms")
        print(f"  Stage 2 (Preprocess):  {stats['videollava']['stage2_time_avg']*1000:.1f} ms")
        print(f"  Stage 3 (Vision):      {stats['videollava']['stage3_time_avg']*1000:.1f} ms")
        print(f"  Stage 4 (Prefill):     {stats['videollava']['stage4_time_avg']*1000:.1f} ms")
        print(f"  Stage 5 (Decode):      {stats['videollava']['stage5_time_avg']*1000:.1f} ms")
        print(f"  TOTAL:                 {stats['videollava']['total_time_avg']*1000:.1f} ms")
        print(f"  Output tokens:         {stats['videollava']['output_tokens_avg']:.1f} avg")

        print(f"\n{'─'*40}")
        print("Parallel Execution Analysis:")
        print(f"{'─'*40}")
        print(f"  Overlap Window:        {stats['parallel']['overlap_window_avg']*1000:.1f} ms")
        print(f"  Hidden Tokens:         {stats['parallel']['hidden_tokens_avg']:.1f} tokens")
        print(f"  Wall-Clock Time:       {stats['parallel']['wall_clock_total_avg']*1000:.1f} ms")
        print(f"  Sequential Time:       {stats['parallel']['sequential_total_avg']*1000:.1f} ms")
        print(f"  Parallel Speedup:      {stats['parallel']['parallel_speedup_avg']:.2f}x")

        print(f"\n{'─'*40}")
        print("GPU Memory Usage (4-bit):")
        print(f"{'─'*40}")
        print(f"  EventGPT model:        {memory_stats.get('eventgpt_model_mb', 0):.0f} MB")
        print(f"  Video-LLaVA model:     {memory_stats.get('videollava_model_mb', 0):.0f} MB")
        print(f"  Total (both models):   {memory_stats.get('total_models_mb', 0):.0f} MB")
        print(f"  EGPT peak inference:   {stats['eventgpt'].get('gpu_peak_memory_avg', 0):.0f} MB")
        print(f"  VL peak inference:     {stats['videollava'].get('gpu_peak_memory_avg', 0):.0f} MB")

        print(f"\n{'─'*40}")
        print(f"Overall Speedup (EGPT vs VL): {stats['videollava']['total_time_avg']/max(stats['eventgpt']['total_time_avg'], 0.0001):.2f}x")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
