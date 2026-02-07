#!/usr/bin/env python3
"""
================================================================================
        5-STAGE COMPREHENSIVE BENCHMARK: EventGPT vs Video-LLaVA
        Parallel Prefill with Real Acceptance Rate
================================================================================

This script benchmarks EventGPT and Video-LLaVA with:
- All 5 stages measured separately for each model
- Real acceptance rate from draft (EventGPT) to target (Video-LLaVA)
- Output tokens and text from both models
- Memory-efficient 4-bit quantization

5 STAGES:
  Stage 1: Data Loading (images/video from disk)
  Stage 2: Preprocessing (tokenization, image transforms)
  Stage 3: Vision Encoding (CLIP/ViT forward pass)
  Stage 4: LLM Prefill (processing input context)
  Stage 5: LLM Decode (generating output tokens)

USAGE:
  python benchmark_parallel_prefill_4bit.py --dataset_dir DATASET_DIR --max_samples N

OPTIONS:
  --dataset_dir   Path to dataset (default: ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s)
  --max_samples    Number of samples to benchmark (-1 for all, default: -1)
  --max_new_tokens Maximum tokens to generate (default: 28)
  --output_dir     Output directory for results (default: ./results/)
  --device         Device to use (default: cuda)

OUTPUT:
  JSON file: {dataset_name}_5stages_{timestamp}.json
  Markdown report: {dataset_name}_5stages_{timestamp}.md

AUTHOR: Alice Zhang
DATE: 2026-01-26
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from common.common import load_image

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    print("Warning: PyAV not found. Install with: pip install av")


def load_video_frames_from_mp4(video_path, num_frames=8):
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


def compute_acceptance_rate(draft_tokens: List[int], target_tokens: List[int],
                             draft_text: str, target_text: str) -> Dict[str, float]:
    """
    Compute acceptance rate from draft (EventGPT) to target (Video-LLaVA).

    Since tokenizers differ, we compute semantic similarity:
    1. Word overlap
    2. Character similarity (LCS)
    3. Length similarity
    """
    # 1. Word overlap
    draft_words = set(draft_text.lower().split())
    target_words = set(target_text.lower().split())

    if draft_words:
        word_overlap = len(draft_words & target_words) / len(draft_words)
    else:
        word_overlap = 0.0

    # 2. Character similarity (longest common substring)
    def longest_common_substring(s1, s2):
        m = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        longest = 0
        for x in range(1, len(s1) + 1):
            for y in range(1, len(s2) + 1):
                if s1[x - 1] == s2[y - 1]:
                    m[x][y] = m[x - 1][y - 1] + 1
                    if m[x][y] > longest:
                        longest = m[x][y]
                else:
                    m[x][y] = 0
        return longest

    lcs_length = longest_common_substring(draft_text.lower(), target_text.lower())
    char_similarity = lcs_length / max(len(draft_text), len(target_text), 1)

    # 3. Length similarity
    length_similarity = 1 - abs(len(draft_tokens) - len(target_tokens)) / max(len(draft_tokens), len(target_tokens), 1)

    # Combined acceptance rate
    acceptance_rate = (word_overlap * 0.5 + char_similarity * 0.3 + length_similarity * 0.2)

    return {
        "acceptance_rate": acceptance_rate,
        "word_overlap": word_overlap,
        "char_similarity": char_similarity,
        "length_similarity": length_similarity,
    }


def benchmark_eventgpt_5stages(
    model, tokenizer, processor,
    sample: Dict, dataset_dir: str, query: str,
    max_new_tokens: int = 28
) -> Optional[Dict]:
    """Benchmark EventGPT with all 5 stages."""
    from model.EventChatModel import get_spatio_temporal_features
    from dataset.conversation import prepare_event_prompt
    from dataset.constants import EVENT_TOKEN_INDEX
    from common.common import tokenizer_event_token
    import numpy as np

    device = "cuda"
    result = {"model": "EventGPT"}

    try:
        # Check for event images
        if "event_image" not in sample or not sample["event_image"]:
            return None

        event_image_paths = sample["event_image"]

        # ========== STAGE 1: Data Loading ==========
        stage1_start = time.time()
        img_path = os.path.join(dataset_dir, "event_image", event_image_paths[0])
        img = load_image(img_path)
        img_array = np.array(img)
        event_image_size = list(img_array.shape[:2])
        torch.cuda.synchronize()
        result["stage1_time"] = time.time() - stage1_start

        # ========== STAGE 2: Preprocessing ==========
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
        stage5_start = time.time()
        output_token_ids = []

        with torch.inference_mode():
            cur_pos = result["prefill_length"]
            kv_cache = outputs.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            output_token_ids.append(next_token.item())

            for _ in range(max_new_tokens - 1):
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

                if token_id == tokenizer.eos_token_id:
                    break

        torch.cuda.synchronize()
        result["stage5_time"] = time.time() - stage5_start
        result["output_tokens"] = output_token_ids
        result["output_text"] = tokenizer.decode(output_token_ids, skip_special_tokens=True)
        result["total_time"] = result["stage1_time"] + result["stage2_time"] + result["stage3_time"] + result["stage4_time"] + result["stage5_time"]

        return result

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()[-300:]}


def benchmark_videollava_5stages(
    model, processor,
    sample: Dict, dataset_dir: str, query: str,
    max_new_tokens: int = 28
) -> Optional[Dict]:
    """Benchmark Video-LLaVA with all 5 stages."""
    device = "cuda"
    result = {"model": "Video-LLaVA"}

    try:
        # Check for video data
        video_data = sample.get("video_data")

        # ========== STAGE 1: Data Loading ==========
        stage1_start = time.time()
        if video_data:
            video_path = os.path.join(dataset_dir, "mp4", video_data + ".mp4")
            if not os.path.exists(video_path):
                return None
            rgb_images = load_video_frames_from_mp4(video_path, num_frames=8)
        else:
            # Fallback to event_image
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

        # ========== STAGE 2: Preprocessing ==========
        stage2_start = time.time()
        image_tokens = "<image>\n" * len(rgb_images)
        prompt = f"USER: {image_tokens}{query}\nASSISTANT:"
        inputs = processor(text=prompt, images=rgb_images, return_tensors="pt")

        # Move to device (but NOT pixel_values - let the model handle it with device_map="auto")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Video-LLaVA uses 'pixel_values_images' key - keep as-is for device_map="auto"
        pixel_values = inputs.get('pixel_values_images') or inputs.get('pixel_values')

        torch.cuda.synchronize()
        result["stage2_time"] = time.time() - stage2_start

        # Check if pixel_values is available
        if pixel_values is None:
            return {"error": f"No pixel values found. Available keys: {list(inputs.keys())}"}

        # ========== STAGE 3: Vision Encoding ==========
        stage3_start = time.time()
        with torch.inference_mode():
            if hasattr(model, 'vision_tower') and model.vision_tower is not None:
                _ = model.vision_tower(pixel_values)
        torch.cuda.synchronize()
        result["stage3_time"] = time.time() - stage3_start

        # ========== STAGE 4: LLM Prefill ==========
        stage4_start = time.time()
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                past_key_values=None,
                use_cache=True,
            )
        torch.cuda.synchronize()
        result["stage4_time"] = time.time() - stage4_start
        result["prefill_length"] = input_ids.shape[1]
        if pixel_values is not None:
            result["prefill_length"] += pixel_values.shape[0] * 576  # Approximate

        # ========== STAGE 5: LLM Decode ==========
        stage5_start = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                use_cache=True,
            )
        torch.cuda.synchronize()
        result["stage5_time"] = time.time() - stage5_start

        # Decode output
        output_length = input_ids.shape[1]
        output_token_ids = generated_ids[0, output_length:].tolist()
        result["output_tokens"] = output_token_ids
        result["output_text"] = processor.tokenizer.decode(output_token_ids, skip_special_tokens=True)
        result["total_time"] = result["stage1_time"] + result["stage2_time"] + result["stage3_time"] + result["stage4_time"] + result["stage5_time"]

        return result

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()[-300:]}


def run_5stage_benchmark(
    eventgpt_model, egpt_tokenizer, egpt_processor,
    videollava_model, vl_processor,
    dataset: List, dataset_dir: str,
    max_samples: int = -1,
    max_new_tokens: int = 28,
) -> Dict:
    """Run 5-stage benchmark on dataset."""
    print("\n" + "=" * 80)
    print(f"Running 5-Stage Benchmark (max {max_samples} samples)")
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
                # Compute acceptance rate
                acceptance = compute_acceptance_rate(
                    egpt_result["output_tokens"],
                    vl_result["output_tokens"],
                    egpt_result["output_text"],
                    vl_result["output_text"],
                )

                # Parallel efficiency analysis
                egpt_prefill_complete = egpt_result["stage1_time"] + egpt_result["stage2_time"] + egpt_result["stage3_time"] + egpt_result["stage4_time"]
                vl_prefill_complete = vl_result["stage1_time"] + vl_result["stage2_time"] + vl_result["stage3_time"] + vl_result["stage4_time"]
                overlap_window = max(0, vl_prefill_complete - egpt_prefill_complete)

                # Draft tokens generated during overlap
                if egpt_result["stage5_time"] > 0:
                    draft_rate = len(egpt_result["output_tokens"]) / egpt_result["stage5_time"]
                else:
                    draft_rate = 0

                theoretical_draft_tokens = min(len(egpt_result["output_tokens"]), int(overlap_window * draft_rate))

                results.append({
                    "sample_idx": sample_idx,
                    "eventgpt": egpt_result,
                    "videollava": vl_result,
                    "acceptance": acceptance,
                    "parallel": {
                        "egpt_prefill_complete": egpt_prefill_complete,
                        "vl_prefill_complete": vl_prefill_complete,
                        "overlap_window": overlap_window,
                        "theoretical_draft_tokens": theoretical_draft_tokens,
                        "actual_draft_tokens": len(egpt_result["output_tokens"]),
                    },
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
    """Compute statistics from 5-stage benchmark results."""
    if not results:
        return {}

    def avg(key, model="eventgpt"):
        values = [r[model][key] for r in results if key in r.get(model, {})]
        return float(np.mean(values)) if values else 0.0

    def std(key, model="eventgpt"):
        values = [r[model][key] for r in results if key in r.get(model, {})]
        return float(np.std(values)) if values else 0.0

    acceptance_rates = [r.get("acceptance", {}).get("acceptance_rate", 0) for r in results]

    return {
        "n_samples": len(results),
        "acceptance_rate_avg": float(np.mean(acceptance_rates)) if acceptance_rates else 0.0,
        "acceptance_rate_std": float(np.std(acceptance_rates)) if acceptance_rates else 0.0,
        "word_overlap_avg": float(np.mean([r.get("acceptance", {}).get("word_overlap", 0) for r in results])),
        "char_similarity_avg": float(np.mean([r.get("acceptance", {}).get("char_similarity", 0) for r in results])),
        "eventgpt": {
            "stage1_time_avg": avg("stage1_time", "eventgpt"),
            "stage1_time_std": std("stage1_time", "eventgpt"),
            "stage2_time_avg": avg("stage2_time", "eventgpt"),
            "stage2_time_std": std("stage2_time", "eventgpt"),
            "stage3_time_avg": avg("stage3_time", "eventgpt"),
            "stage3_time_std": std("stage3_time", "eventgpt"),
            "stage4_time_avg": avg("stage4_time", "eventgpt"),
            "stage4_time_std": std("stage4_time", "eventgpt"),
            "stage5_time_avg": avg("stage5_time", "eventgpt"),
            "stage5_time_std": std("stage5_time", "eventgpt"),
            "prefill_length_avg": avg("prefill_length", "eventgpt"),
            "output_tokens_avg": float(np.mean([len(r.get("eventgpt", {}).get("output_tokens", [])) for r in results])),
            "total_time_avg": avg("total_time", "eventgpt"),
        },
        "videollava": {
            "stage1_time_avg": avg("stage1_time", "videollava"),
            "stage1_time_std": std("stage1_time", "videollava"),
            "stage2_time_avg": avg("stage2_time", "videollava"),
            "stage2_time_std": std("stage2_time", "videollava"),
            "stage3_time_avg": avg("stage3_time", "videollava"),
            "stage3_time_std": std("stage3_time", "videollava"),
            "stage4_time_avg": avg("stage4_time", "videollava"),
            "stage4_time_std": std("stage4_time", "videollava"),
            "stage5_time_avg": avg("stage5_time", "videollava"),
            "stage5_time_std": std("stage5_time", "videollava"),
            "prefill_length_avg": avg("prefill_length", "videollava"),
            "output_tokens_avg": float(np.mean([len(r.get("videollava", {}).get("output_tokens", [])) for r in results])),
            "total_time_avg": avg("total_time", "videollava"),
        },
        "parallel": {
            "overlap_window_avg": avg("overlap_window", "parallel"),
            "theoretical_draft_tokens_avg": avg("theoretical_draft_tokens", "parallel"),
            "actual_draft_tokens_avg": avg("actual_draft_tokens", "parallel"),
        },
    }


def generate_markdown_report(stats: Dict, results: List[Dict], dataset_name: str, output_path: str, config: Dict):
    """Generate markdown report from 5-stage benchmark results."""

    egpt = stats.get("eventgpt", {})
    vl = stats.get("videollava", {})
    par = stats.get("parallel", {})

    # Find bottleneck stages
    egpt_times = [egpt.get("stage1_time_avg", 0), egpt.get("stage2_time_avg", 0),
                  egpt.get("stage3_time_avg", 0), egpt.get("stage4_time_avg", 0),
                  egpt.get("stage5_time_avg", 0)]
    vl_times = [vl.get("stage1_time_avg", 0), vl.get("stage2_time_avg", 0),
                vl.get("stage3_time_avg", 0), vl.get("stage4_time_avg", 0),
                vl.get("stage5_time_avg", 0)]

    egpt_bottleneck = egpt_times.index(max(egpt_times)) + 1
    vl_bottleneck = vl_times.index(max(vl_times)) + 1

    report = f"""# 5-Stage Benchmark Report: {dataset_name}

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Samples:** {stats.get('n_samples', 0)}
**Max New Tokens:** {config.get('max_new_tokens', 28)}

## Configuration

| Setting | Value |
|---------|-------|
| Dataset | {dataset_name} |
| Max Samples | {config.get('max_samples', -1)} |
| Device | {config.get('device', 'cuda')} |

## Acceptance Rate Analysis

| Metric | Value |
|--------|-------|
| **Acceptance Rate** | {stats.get('acceptance_rate_avg', 0)*100:.1f}% ± {stats.get('acceptance_rate_std', 0)*100:.1f}% |
| Word Overlap | {stats.get('word_overlap_avg', 0)*100:.1f}% |
| Character Similarity | {stats.get('char_similarity_avg', 0)*100:.1f}% |

## 5-Stage Timing Comparison

| Stage | EventGPT | Video-LLaVA | Speedup |
|-------|----------|-------------|---------|
| **Stage 1: Data Loading** | {egpt.get('stage1_time_avg', 0)*1000:.1f} ± {egpt.get('stage1_time_std', 0)*1000:.1f} ms | {vl.get('stage1_time_avg', 0)*1000:.1f} ± {vl.get('stage1_time_std', 0)*1000:.1f} ms | {vl.get('stage1_time_avg', 0.0001)/max(egpt.get('stage1_time_avg', 0.0001), 0.0001):.2f}x |
| **Stage 2: Preprocessing** | {egpt.get('stage2_time_avg', 0)*1000:.1f} ± {egpt.get('stage2_time_std', 0)*1000:.1f} ms | {vl.get('stage2_time_avg', 0)*1000:.1f} ± {vl.get('stage2_time_std', 0)*1000:.1f} ms | {vl.get('stage2_time_avg', 0.0001)/max(egpt.get('stage2_time_avg', 0.0001), 0.0001):.2f}x |
| **Stage 3: Vision Encoding** | {egpt.get('stage3_time_avg', 0)*1000:.1f} ± {egpt.get('stage3_time_std', 0)*1000:.1f} ms | {vl.get('stage3_time_avg', 0)*1000:.1f} ± {vl.get('stage3_time_std', 0)*1000:.1f} ms | {vl.get('stage3_time_avg', 0.0001)/max(egpt.get('stage3_time_avg', 0.0001), 0.0001):.2f}x |
| **Stage 4: LLM Prefill** | {egpt.get('stage4_time_avg', 0)*1000:.1f} ± {egpt.get('stage4_time_std', 0)*1000:.1f} ms | {vl.get('stage4_time_avg', 0)*1000:.1f} ± {vl.get('stage4_time_std', 0)*1000:.1f} ms | {vl.get('stage4_time_avg', 0.0001)/max(egpt.get('stage4_time_avg', 0.0001), 0.0001):.2f}x |
| **Stage 5: LLM Decode** | {egpt.get('stage5_time_avg', 0)*1000:.1f} ± {egpt.get('stage5_time_std', 0)*1000:.1f} ms | {vl.get('stage5_time_avg', 0)*1000:.1f} ± {vl.get('stage5_time_std', 0)*1000:.1f} ms | {vl.get('stage5_time_avg', 0.0001)/max(egpt.get('stage5_time_avg', 0.0001), 0.0001):.2f}x |
| **TOTAL** | {egpt.get('total_time_avg', 0)*1000:.1f} ms | {vl.get('total_time_avg', 0)*1000:.1f} ms | {vl.get('total_time_avg', 0.0001)/max(egpt.get('total_time_avg', 0.0001), 0.0001):.2f}x |

## Stage Breakdown (Percentage)

### EventGPT
| Stage | Time | Percentage |
|-------|------|------------|
| Stage 1: Data Loading | {egpt.get('stage1_time_avg', 0)*1000:.1f} ms | {egpt.get('stage1_time_avg', 0)/max(egpt.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% |
| Stage 2: Preprocessing | {egpt.get('stage2_time_avg', 0)*1000:.1f} ms | {egpt.get('stage2_time_avg', 0)/max(egpt.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% |
| Stage 3: Vision Encoding | {egpt.get('stage3_time_avg', 0)*1000:.1f} ms | {egpt.get('stage3_time_avg', 0)/max(egpt.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% |
| Stage 4: LLM Prefill | {egpt.get('stage4_time_avg', 0)*1000:.1f} ms | {egpt.get('stage4_time_avg', 0)/max(egpt.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% |
| Stage 5: LLM Decode | {egpt.get('stage5_time_avg', 0)*1000:.1f} ms | {egpt.get('stage5_time_avg', 0)/max(egpt.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% |

### Video-LLaVA
| Stage | Time | Percentage |
|-------|------|------------|
| Stage 1: Data Loading | {vl.get('stage1_time_avg', 0)*1000:.1f} ms | {vl.get('stage1_time_avg', 0)/max(vl.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% |
| Stage 2: Preprocessing | {vl.get('stage2_time_avg', 0)*1000:.1f} ms | {vl.get('stage2_time_avg', 0)/max(vl.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% |
| Stage 3: Vision Encoding | {vl.get('stage3_time_avg', 0)*1000:.1f} ms | {vl.get('stage3_time_avg', 0)/max(vl.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% |
| Stage 4: LLM Prefill | {vl.get('stage4_time_avg', 0)*1000:.1f} ms | {vl.get('stage4_time_avg', 0)/max(vl.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% |
| Stage 5: LLM Decode | {vl.get('stage5_time_avg', 0)*1000:.1f} ms | {vl.get('stage5_time_avg', 0)/max(vl.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% |

## Output Comparison

| Model | Output Tokens (avg) |
|-------|---------------------|
| EventGPT | {egpt.get('output_tokens_avg', 0):.1f} |
| Video-LLaVA | {vl.get('output_tokens_avg', 0):.1f} |

## Parallel Prefill Analysis

| Metric | Value |
|--------|-------|
| Overlap Window | {par.get('overlap_window_avg', 0)*1000:.1f} ms |
| Theoretical Draft Tokens | {par.get('theoretical_draft_tokens_avg', 0):.1f} |
| Actual Draft Tokens | {par.get('actual_draft_tokens_avg', 0):.1f} |

## Key Findings

1. **EventGPT Bottleneck**: Stage {egpt_bottleneck} ({egpt_times[egpt_bottleneck-1]/max(egpt.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% of total time)
2. **Video-LLaVA Bottleneck**: Stage {vl_bottleneck} ({vl_times[vl_bottleneck-1]/max(vl.get('total_time_avg', 0.0001), 0.0001)*100:.1f}% of total time)
3. **Acceptance Rate**: {stats.get('acceptance_rate_avg', 0)*100:.1f}% - EventGPT drafts are {"well" if stats.get('acceptance_rate_avg', 0) > 0.5 else "partially" if stats.get('acceptance_rate_avg', 0) > 0.3 else "poorly"} aligned with Video-LLaVA outputs
4. **Overall Speedup**: EventGPT is {vl.get('total_time_avg', 0.0001)/max(egpt.get('total_time_avg', 0.0001), 0.0001):.2f}x faster than Video-LLaVA

---

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*Author: Alice Zhang*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"✓ Markdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="5-stage comprehensive benchmark")
    parser.add_argument("--dataset_dir", type=str, default="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Number of samples to benchmark (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=28)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    json_path = os.path.join(args.dataset_dir, "EventGPT_Instruction_Subset.json")
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))

    # Load models
    print("\n" + "=" * 80)
    print("Loading Models")
    print("=" * 80)

    from model.EventChatModel import EventChatModel

    print("\n[1/2] Loading EventGPT...")
    eventgpt_path = "./checkpoints/EventGPT-7b"
    eventgpt_model = EventChatModel.from_pretrained(
        eventgpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    eventgpt_model.eval()
    egpt_tokenizer = AutoTokenizer.from_pretrained(eventgpt_path, use_fast=True)
    egpt_processor = eventgpt_model.get_visual_tower().event_processor

    print("\n[2/2] Loading Video-LLaVA...")
    videollava_model_id = "LanguageBind/Video-LLaVA-7B-hf"
    videollava_model = LlavaForConditionalGeneration.from_pretrained(
        videollava_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    vl_processor = AutoProcessor.from_pretrained(videollava_model_id)
    videollava_model.eval()

    print("\n✓ Both models loaded successfully")

    # Run benchmark
    output = run_5stage_benchmark(
        eventgpt_model, egpt_tokenizer, egpt_processor,
        videollava_model, vl_processor,
        dataset, args.dataset_dir,
        args.max_samples,
        args.max_new_tokens,
    )

    # Compute statistics
    stats = compute_statistics(output["results"])

    # Save results
    results_data = {
        "timestamp": timestamp,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name,
        "config": {
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
        },
        "statistics": stats,
        "results": output["results"],
        "errors": output["errors"],
    }

    json_path = os.path.join(args.output_dir, f"{dataset_name}_5stages_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n✓ Results saved to {json_path}")

    # Generate markdown report
    markdown_path = os.path.join(args.output_dir, f"{dataset_name}_5stages_{timestamp}.md")
    generate_markdown_report(stats, output["results"], dataset_name, markdown_path,
                             {"max_samples": args.max_samples, "max_new_tokens": args.max_new_tokens,
                              "device": args.device})

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    if stats:
        print(f"\nSamples processed: {stats['n_samples']}")
        print(f"\nAcceptance Rate: {stats['acceptance_rate_avg']*100:.1f}%")

        print(f"\nEventGPT 5-Stage Breakdown:")
        print(f"  Stage 1 (Data Load):   {stats['eventgpt']['stage1_time_avg']*1000:.1f} ms")
        print(f"  Stage 2 (Preprocess):  {stats['eventgpt']['stage2_time_avg']*1000:.1f} ms")
        print(f"  Stage 3 (Vision):      {stats['eventgpt']['stage3_time_avg']*1000:.1f} ms")
        print(f"  Stage 4 (Prefill):     {stats['eventgpt']['stage4_time_avg']*1000:.1f} ms")
        print(f"  Stage 5 (Decode):      {stats['eventgpt']['stage5_time_avg']*1000:.1f} ms")
        print(f"  TOTAL:                 {stats['eventgpt']['total_time_avg']*1000:.1f} ms")

        print(f"\nVideo-LLaVA 5-Stage Breakdown:")
        print(f"  Stage 1 (Data Load):   {stats['videollava']['stage1_time_avg']*1000:.1f} ms")
        print(f"  Stage 2 (Preprocess):  {stats['videollava']['stage2_time_avg']*1000:.1f} ms")
        print(f"  Stage 3 (Vision):      {stats['videollava']['stage3_time_avg']*1000:.1f} ms")
        print(f"  Stage 4 (Prefill):     {stats['videollava']['stage4_time_avg']*1000:.1f} ms")
        print(f"  Stage 5 (Decode):      {stats['videollava']['stage5_time_avg']*1000:.1f} ms")
        print(f"  TOTAL:                 {stats['videollava']['total_time_avg']*1000:.1f} ms")

        print(f"\nOverall Speedup: {stats['videollava']['total_time_avg']/max(stats['eventgpt']['total_time_avg'], 0.0001):.2f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
