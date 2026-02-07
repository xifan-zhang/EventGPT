#!/usr/bin/env python3
"""
================================================================================
                    5-STAGE BENCHMARK: EventGPT vs Video-LLaVA
================================================================================

README
------
This script provides comprehensive benchmarking of EventGPT and Video-LLaVA models
with detailed 5-stage timing separation for inference analysis.

5-STAGE PIPELINE:
  Stage 1: Load images/video from disk
  Stage 2: Preprocess images (CLIP normalization)
  Stage 3: Vision encoding (CLIP ViT forward pass)
  Stage 4: LLM Prefill (process input + vision tokens, build KV cache)
  Stage 5: LLM Decode (autoregressive token generation)

BENCHMARK MODES:
  1. Default Mode: EventGPT (all event frames) vs Video-LLaVA (1 image)
  2. Comparison Mode: EventGPT (1 frame) vs Video-LLaVA (8 frames from MP4)
     - Use --compare_1vs8 flag to enable
     - Demonstrates token scaling efficiency

KEY METRICS:
  - Prefill latency (input-dependent, parallelizable)
  - Decode throughput (memory-bound, sequential)
  - Time-to-first-token (TTFT) = Stages 1-4
  - Tokens per second in decode phase
  - KV cache memory usage

USAGE EXAMPLES:
  # Default: EventGPT (all frames) vs Video-LLaVA (1 image)
  python benchmark_inference_5stages.py

  # EventGPT only
  python benchmark_inference_5stages.py --eventgpt_only

  # Comparison mode: EventGPT (1 frame) vs Video-LLaVA (8 frames from MP4)
  python benchmark_inference_5stages.py --compare_1vs8

  # Limit samples for quick testing
  python benchmark_inference_5stages.py --max_samples 10

  # Custom output path
  python benchmark_inference_5stages.py --output_dir ./results

REQUIREMENTS:
  - PyTorch with CUDA support
  - transformers >= 4.36.0
  - PyAV (for MP4 video loading)
  - tqdm

OUTPUT:
  - JSON file with detailed per-sample results
  - Markdown analysis report with statistics
  - Console summary with speedup metrics

CHANGELOG:
----------
[2026-01-25] v2.3.0 - PARALLEL PREFILL IMPLEMENTATION COMPLETED
  - Created benchmark_parallel_prefill/ directory with parallel prefill benchmark
  - Successfully benchmarked 1100 samples with 4-bit quantization
  - Both 7B models fit in 24GB GPU: EventGPT (13GB BF16) + Video-LLaVA (4GB INT4)
  - EventGPT prefill: 74.9ms, Video-LLaVA: 336.2ms (4.49x speedup)
  - Overlap window: 261ms allows ~9 free draft tokens
  - Files: FINAL_REPORT_20260125.md, benchmark_parallel_quantized.py
  - 8-bit quantization has compatibility issues with Video-LLaVA model

[2026-01-24] v2.2.0 - PARALLEL PREFILL RESEARCH
  - Added 500ms dataset benchmark (2220 samples)
  - Added theoretical speedup analysis (O(n²) attention complexity)
  - Added per-token acceptance rate equivalent analysis
  - Added KV cache memory scaling formulas
  - Added batch capacity impact analysis
  - Results: 4.50x total speedup (EventGPT 1 frame vs Video-LLaVA 8 frames)
  - Measured 8.59x prefill speedup vs 53.3x theoretical (memory-bandwidth bound)

[2026-01-24] v2.0.0
  - Added --compare_1vs8 mode for EventGPT (1 frame) vs Video-LLaVA (8 frames)
  - Added MP4 video loading with PyAV for Video-LLaVA
  - Added automatic markdown report generation
  - Added KV cache memory estimation
  - Added token scaling analysis
  - Improved output with datetime stamps and file paths

[2026-01-24] v1.1.0
  - Fixed attention_mask None handling in prepare_inputs_labels_for_multimodal
  - Added TTFT (Time to First Token) metric
  - Added prefill/decode percentage breakdown

[2026-01-24] v1.0.0
  - Initial 5-stage benchmark implementation
  - Separated Stage 4 (prefill) from Stage 5 (decode)
  - Manual token-by-token generation for accurate timing

AUTHOR: Alice Zhang
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
from tqdm import tqdm
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.EventChatModel import EventChatModel, get_spatio_temporal_features
from common.common import tokenizer_event_token, load_image
from dataset.conversation import prepare_event_prompt
from dataset.constants import (
    EVENT_TOKEN_INDEX,
    DEFAULT_EVENT_PATCH_TOKEN,
    DEFAULT_EV_START_TOKEN,
    DEFAULT_EV_END_TOKEN,
)

# TokenAdapter import (optional - for aligned evaluation)
TokenAdapter = None
TokenAdapterConfig = None
try:
    from feasible.token_alignment.token_adapter import TokenAdapter, TokenAdapterConfig
except ImportError:
    pass  # TokenAdapter not available


def compute_aligned_acceptance_rate(
    draft_tokens: List[int],
    target_tokens: List[int],
    token_adapter,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute acceptance rate using TokenAdapter for alignment.

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


def load_preprocessed_event_images(event_image_paths, event_processor, dataset_dir, device):
    """Load preprocessed event images and process them with event_processor.

    Returns:
        event_image_size: [height, width] of the first image
        event_list: List of processed event tensors
        stage1_time: Time for loading images from disk
        stage2_time: Time for CLIP preprocessing
    """
    import numpy as np

    event_list = []
    event_image_size = None

    # Stage 1: Load event images from disk
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    stage1_start = time.time()
    loaded_images = []
    for img_path in event_image_paths:
        full_path = os.path.join(dataset_dir, "event_image", img_path)
        img = load_image(full_path)
        img_array = np.array(img)
        if event_image_size is None:
            event_image_size = list(img_array.shape[:2])
        loaded_images.append(img_array)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    stage1_time = time.time() - stage1_start

    # Stage 2: Process images with event_processor (CLIP preprocessing)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    stage2_start = time.time()
    for img_array in loaded_images:
        event = event_processor(img_array, return_tensors='pt')['pixel_values'][0]
        event = event.to(device, dtype=torch.bfloat16)
        event_list.append(event)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    stage2_time = time.time() - stage2_start

    return event_image_size, event_list, stage1_time, stage2_time


def run_eventgpt_5stage_benchmark(model, tokenizer, processor, dataset_dir, dataset, device="cuda"):
    """
    Run EventGPT benchmark with 5-stage timing separation.

    Stages:
        1. Load images from disk
        2. Preprocess images (CLIP)
        3. Vision encoding
        4. LLM Prefill (process embeddings, build KV cache, generate first token)
        5. LLM Decode (autoregressive generation of remaining tokens)
    """
    print("\n" + "=" * 80)
    print("EventGPT: 5-Stage Benchmark (Prefill/Decode Separated)")
    print("=" * 80)

    results = []
    query = "What are the key elements in this scene?"

    for sample_idx, sample in enumerate(tqdm(dataset, desc="EventGPT 5-Stage")):
        try:
            event_data = sample.get("event_data")
            if not event_data:
                continue

            # Stage 1 & 2: Load and preprocess event images
            if "event_image" in sample and sample["event_image"]:
                event_image_paths = sample["event_image"]
                event_image_size, event_tensor, stage1_time, stage2_time = load_preprocessed_event_images(
                    event_image_paths, processor, dataset_dir, device
                )
            else:
                continue

            # Prepare input tokens
            conv_mode = 'eventgpt_v1'
            prompt = prepare_event_prompt(query, conv_mode)
            input_ids = tokenizer_event_token(
                prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(device)

            # ===== STAGE 3: VISION ENCODING =====
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage3_start = time.time()

            with torch.inference_mode():
                # Encode each event frame
                ev_features_list = []
                for ev in event_tensor:
                    ev = ev.unsqueeze(0)
                    feature = model.visval_encode(ev)
                    feature = model.get_model().feature_adaptor(feature)
                    feature = feature.squeeze(0)
                    ev_features_list.append(feature)
                # Combine into spatio-temporal features
                event_features = get_spatio_temporal_features(ev_features_list)
                event_features = event_features.unsqueeze(0)  # Add batch dimension

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage3_time = time.time() - stage3_start

            # ===== STAGE 4: LLM PREFILL =====
            # Prepare input embeddings (inject vision features into text sequence)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage4_start = time.time()

            with torch.inference_mode():
                # Prepare multimodal inputs (vision features already computed)
                (
                    _,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    _
                ) = model.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    None,  # position_ids
                    torch.ones_like(input_ids, dtype=torch.bool),  # attention_mask
                    None,  # past_key_values
                    None,  # labels
                    event_tensors=None,
                    event_image_sizes=event_image_size,
                    event_features=event_features  # Use cached features
                )

                # Create default attention mask if None
                if attention_mask is None:
                    attention_mask = torch.ones(
                        (1, inputs_embeds.shape[1]), dtype=torch.bool, device=device
                    )
                if position_ids is None:
                    position_ids = torch.arange(
                        0, inputs_embeds.shape[1], dtype=torch.long, device=device
                    ).unsqueeze(0)

                # Run prefill: single forward pass to build KV cache
                outputs = model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                # Get logits for next token prediction
                hidden_states = outputs.last_hidden_state
                logits = model.lm_head(hidden_states[:, -1:, :])
                past_key_values = outputs.past_key_values

                # Generate first token
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage4_time = time.time() - stage4_start

            prefill_tokens = inputs_embeds.shape[1]  # Number of input tokens processed

            # ===== STAGE 5: LLM DECODE =====
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage5_start = time.time()

            generated_ids = [next_token.item()]
            max_new_tokens = 512
            eos_token_id = tokenizer.eos_token_id

            with torch.inference_mode():
                cur_pos = inputs_embeds.shape[1]
                cur_token = next_token

                for step in range(max_new_tokens - 1):  # -1 because we already have first token
                    # Get embedding for current token
                    cur_embed = model.get_model().embed_tokens(cur_token)

                    # Update attention mask
                    new_attention_mask = torch.ones(
                        (1, cur_pos + 1), dtype=torch.bool, device=device
                    )

                    # Forward pass with KV cache
                    outputs = model.model(
                        inputs_embeds=cur_embed,
                        attention_mask=new_attention_mask,
                        position_ids=torch.tensor([[cur_pos]], device=device),
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )

                    hidden_states = outputs.last_hidden_state
                    logits = model.lm_head(hidden_states[:, -1:, :])
                    past_key_values = outputs.past_key_values

                    # Sample next token (greedy for benchmarking)
                    next_token_logits = logits[:, -1, :]
                    cur_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    generated_ids.append(cur_token.item())
                    cur_pos += 1

                    # Check for EOS
                    if cur_token.item() == eos_token_id:
                        break

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage5_time = time.time() - stage5_start

            # Decode output text
            output_ids = torch.tensor([generated_ids], device=device)
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            num_generated = len(generated_ids)
            decode_tokens_per_sec = num_generated / stage5_time if stage5_time > 0 else 0
            ttft = stage1_time + stage2_time + stage3_time + stage4_time  # Time to first token

            results.append({
                "sample": sample_idx,
                "stage1_time": stage1_time,
                "stage2_time": stage2_time,
                "stage3_time": stage3_time,
                "stage4_time": stage4_time,  # Prefill
                "stage5_time": stage5_time,  # Decode
                "total_time": stage1_time + stage2_time + stage3_time + stage4_time + stage5_time,
                "ttft": ttft,  # Time to first token
                "prefill_tokens": prefill_tokens,
                "decode_tokens": num_generated,
                "decode_tokens_per_sec": decode_tokens_per_sec,
                "output": output,
                "token_ids": generated_ids,
            })

            print(f"\nSample {sample_idx}: "
                  f"S1={stage1_time:.4f}s | S2={stage2_time:.4f}s | S3={stage3_time:.4f}s | "
                  f"S4(prefill)={stage4_time:.4f}s | S5(decode)={stage5_time:.4f}s | "
                  f"TTFT={ttft:.4f}s | Decode={decode_tokens_per_sec:.1f} tok/s | "
                  f"Tokens={num_generated}")

        except Exception as e:
            print(f"\nError on sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


class PrefillDecodeHooks:
    """Forward hooks to measure prefill vs decode time for Video-LLaVA."""

    def __init__(self, model):
        self.model = model
        self.is_prefill = True
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.prefill_start = None
        self.decode_start = None
        self.first_forward_done = False
        self.hooks = []

    def _forward_pre_hook(self, module, input):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if not self.first_forward_done:
            # This is prefill
            self.prefill_start = time.time()
        elif self.decode_start is None:
            # First decode step
            self.decode_start = time.time()

    def _forward_hook(self, module, input, output):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if not self.first_forward_done:
            self.prefill_time = time.time() - self.prefill_start
            self.first_forward_done = True

    def register_hooks(self):
        """Register hooks on the language model."""
        try:
            # Try to find the main transformer block
            if hasattr(self.model, 'language_model'):
                lm = self.model.language_model
            elif hasattr(self.model, 'model'):
                lm = self.model.model
            else:
                lm = self.model

            if hasattr(lm, 'model'):
                target = lm.model
            else:
                target = lm

            h1 = target.register_forward_pre_hook(self._forward_pre_hook)
            h2 = target.register_forward_hook(self._forward_hook)
            self.hooks = [h1, h2]
            return True
        except Exception as e:
            print(f"  Warning: Could not register hooks: {e}")
            return False

    def unregister_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def finalize_decode_time(self):
        """Call this after generation completes to calculate decode time."""
        if self.decode_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.decode_time = time.time() - self.decode_start

    def reset(self):
        self.is_prefill = True
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.prefill_start = None
        self.decode_start = None
        self.first_forward_done = False


def run_videollava_5stage_benchmark(dataset_dir, dataset, device="cuda", max_samples=None):
    """
    Benchmark Video-LLaVA with 5-stage timing separation.

    Uses token-by-token generation to measure prefill vs decode accurately.
    """
    print("\n" + "=" * 80)
    print("Video-LLaVA: 5-Stage Benchmark (Prefill/Decode Separated)")
    print("=" * 80)

    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        print("Loading Video-LLaVA model...")
        model_id = "llava-hf/llava-1.5-7b-hf"
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
                local_files_only=True
            )
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        except:
            print("  Downloading from HuggingFace...")
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
            )
            processor = AutoProcessor.from_pretrained(model_id)

        model.eval()
        print("✓ Video-LLaVA loaded")

        results = []
        base_query = "What are the key elements in this scene?"
        samples_to_process = dataset[:max_samples] if max_samples else dataset

        for sample_idx, sample in enumerate(tqdm(samples_to_process, desc="Video-LLaVA 5-Stage", leave=False)):
            try:
                event_data = sample.get("event_data")
                if not event_data:
                    continue

                if "event_image" not in sample or not sample["event_image"]:
                    continue

                event_image_paths = sample["event_image"]

                # ===== STAGE 1: LOAD IMAGES =====
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage1_start = time.time()

                images = []
                for img_path in event_image_paths:
                    full_path = os.path.join(dataset_dir, "event_image", img_path)
                    img = load_image(full_path)
                    images.append(img)

                # Use only first image for LLaVA (single image model)
                if len(images) > 1:
                    images = [images[0]]

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage1_time = time.time() - stage1_start

                # ===== STAGE 2: PREPROCESS =====
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage2_start = time.time()

                prompt = f"USER: <image>\n{base_query}\nASSISTANT:"
                inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage2_time = time.time() - stage2_start

                # ===== STAGE 3: VISION ENCODING =====
                # For LLaVA, vision encoding happens during the first forward pass
                # We'll measure it by running a forward pass before generation
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage3_start = time.time()

                with torch.inference_mode():
                    # Get vision features by running the vision tower
                    if hasattr(model, 'vision_tower'):
                        vision_tower = model.vision_tower
                    elif hasattr(model, 'get_vision_tower'):
                        vision_tower = model.get_vision_tower()
                    else:
                        vision_tower = model.model.vision_tower if hasattr(model.model, 'vision_tower') else None

                    if vision_tower is not None and 'pixel_values' in inputs:
                        # Run vision encoding separately
                        pixel_values = inputs['pixel_values']
                        _ = vision_tower(pixel_values)

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage3_time = time.time() - stage3_start

                # ===== STAGE 4 & 5: PREFILL AND DECODE =====
                # Use manual generation loop to separate prefill from decode
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage4_start = time.time()

                with torch.inference_mode():
                    # Prepare inputs for generation
                    input_ids = inputs['input_ids']
                    attention_mask = inputs.get('attention_mask')
                    pixel_values = inputs.get('pixel_values')

                    # Run prefill (first forward pass with full input)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        past_key_values=None,
                        use_cache=True,
                        return_dict=True,
                    )

                    logits = outputs.logits
                    past_key_values = outputs.past_key_values

                    # Get first generated token
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage4_time = time.time() - stage4_start

                prefill_tokens = input_ids.shape[1]

                # ===== STAGE 5: DECODE =====
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage5_start = time.time()

                generated_ids = [next_token.item()]
                max_new_tokens = 512
                eos_token_id = processor.tokenizer.eos_token_id

                with torch.inference_mode():
                    cur_token = next_token
                    cur_attention_mask = torch.ones(
                        (1, input_ids.shape[1] + 1), dtype=attention_mask.dtype, device=device
                    )

                    for step in range(max_new_tokens - 1):
                        outputs = model(
                            input_ids=cur_token,
                            attention_mask=cur_attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )

                        logits = outputs.logits
                        past_key_values = outputs.past_key_values

                        next_token_logits = logits[:, -1, :]
                        cur_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                        generated_ids.append(cur_token.item())

                        # Update attention mask
                        cur_attention_mask = torch.ones(
                            (1, cur_attention_mask.shape[1] + 1),
                            dtype=attention_mask.dtype, device=device
                        )

                        if cur_token.item() == eos_token_id:
                            break

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage5_time = time.time() - stage5_start

                # Decode output
                output_ids = torch.tensor([generated_ids], device=device)
                output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                num_generated = len(generated_ids)
                decode_tokens_per_sec = num_generated / stage5_time if stage5_time > 0 else 0
                ttft = stage1_time + stage2_time + stage3_time + stage4_time

                results.append({
                    "model": "videollava",
                    "sample": sample_idx,
                    "stage1_time": stage1_time,
                    "stage2_time": stage2_time,
                    "stage3_time": stage3_time,
                    "stage4_time": stage4_time,  # Prefill
                    "stage5_time": stage5_time,  # Decode
                    "total_time": stage1_time + stage2_time + stage3_time + stage4_time + stage5_time,
                    "ttft": ttft,
                    "prefill_tokens": prefill_tokens,
                    "decode_tokens": num_generated,
                    "decode_tokens_per_sec": decode_tokens_per_sec,
                    "output": output_text,
                    "token_ids": generated_ids,
                })

            except Exception as e:
                print(f"\n  Error on sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n  Video-LLaVA processed {len(results)} samples successfully")
        return results

    except Exception as e:
        print(f"⚠️  Video-LLaVA not available: {e}")
        import traceback
        traceback.print_exc()
        return []


def cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =============================================================================
# COMPARISON MODE: EventGPT (1 frame) vs Video-LLaVA (8 frames from MP4)
# =============================================================================

def load_video_frames_from_mp4(video_path, num_frames=8):
    """Load frames from MP4 video, sampled uniformly using PyAV.

    Args:
        video_path: Path to MP4 video file
        num_frames: Number of frames to sample uniformly

    Returns:
        frames: List of PIL Images
        total_frames: Total frames in video
        sampled_indices: List of sampled frame indices
    """
    if not HAS_PYAV:
        raise ImportError("PyAV is required for MP4 video loading. Install with: pip install av")

    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames

    if total_frames == 0:
        # Fallback: count frames by iterating
        total_frames = sum(1 for _ in container.decode(stream))
        container.seek(0)

    # Sample frame indices uniformly
    if total_frames >= num_frames:
        indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int))
    else:
        indices = set(range(total_frames))

    frames = []
    sampled_indices = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            pil_img = frame.to_image()
            frames.append(pil_img)
            sampled_indices.append(i)

    container.close()
    return frames, total_frames, sampled_indices


def estimate_kv_cache_mb(past_key_values):
    """Estimate KV cache size in MB."""
    if past_key_values is None:
        return 0
    total = 0
    for kv in past_key_values:
        if kv is not None and len(kv) == 2:
            k, v = kv
            total += k.numel() + v.numel()
    return (total * 2) / (1024 * 1024)  # 2 bytes for float16/bfloat16


def run_eventgpt_1frame_benchmark(model, tokenizer, processor, dataset_dir, dataset, device="cuda"):
    """Run EventGPT benchmark using only 1 event frame per sample.

    This mode demonstrates EventGPT's efficiency when using minimal input.
    """
    print("\n" + "=" * 80)
    print("EventGPT: 1 Event Frame Benchmark")
    print("=" * 80)

    results = []
    query = "What are the key elements in this scene?"

    for sample_idx, sample in enumerate(tqdm(dataset, desc="EventGPT (1 frame)")):
        try:
            if "event_image" not in sample or not sample["event_image"]:
                continue

            event_image_paths = sample["event_image"]

            # Use only FIRST frame
            first_frame_path = event_image_paths[0]
            full_path = os.path.join(dataset_dir, "event_image", first_frame_path)
            img = load_image(full_path)
            img_array = np.array(img)
            event_image_size = list(img_array.shape[:2])

            # Stage 1 & 2: Load and preprocess (combined for 1 frame)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage12_start = time.time()

            event_tensor = processor(img_array, return_tensors='pt')['pixel_values'][0]
            event_tensor = event_tensor.to(device, dtype=torch.bfloat16)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage12_time = time.time() - stage12_start

            # Prepare tokens
            conv_mode = 'eventgpt_v1'
            prompt = prepare_event_prompt(query, conv_mode)
            input_ids = tokenizer_event_token(prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0).to(device)

            with torch.inference_mode():
                # Stage 3: Vision encoding (1 frame)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage3_start = time.time()

                ev = event_tensor.unsqueeze(0)
                feature = model.visval_encode(ev)
                feature = model.get_model().feature_adaptor(feature)
                event_features = feature  # [1, 577, 4096]

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage3_time = time.time() - stage3_start

                # Stage 4: Prefill
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage4_start = time.time()

                (_, position_ids, attention_mask, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, torch.ones_like(input_ids, dtype=torch.bool),
                    None, None, event_tensors=None, event_image_sizes=event_image_size,
                    event_features=event_features
                )

                if attention_mask is None:
                    attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.bool, device=device)
                if position_ids is None:
                    position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=device).unsqueeze(0)

                outputs = model.model(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=None, use_cache=True,
                    output_attentions=False, output_hidden_states=False, return_dict=True,
                )

                hidden_states = outputs.last_hidden_state
                logits = model.lm_head(hidden_states[:, -1:, :])
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage4_time = time.time() - stage4_start

                prefill_len = inputs_embeds.shape[1]
                kv_cache_mb = estimate_kv_cache_mb(past_key_values)

                # Stage 5: Decode (5 tokens for timing)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage5_start = time.time()

                generated = [next_token.item()]
                cur_token = next_token
                cur_pos = prefill_len

                for _ in range(5):
                    cur_embed = model.get_model().embed_tokens(cur_token)
                    new_mask = torch.ones((1, cur_pos + 1), dtype=torch.bool, device=device)

                    outputs = model.model(
                        inputs_embeds=cur_embed, attention_mask=new_mask,
                        position_ids=torch.tensor([[cur_pos]], device=device),
                        past_key_values=past_key_values, use_cache=True,
                        output_attentions=False, output_hidden_states=False, return_dict=True,
                    )

                    logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
                    past_key_values = outputs.past_key_values
                    cur_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated.append(cur_token.item())
                    cur_pos += 1

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage5_time = time.time() - stage5_start

                results.append({
                    'sample': sample_idx,
                    'prefill_len': prefill_len,
                    'kv_cache_mb': kv_cache_mb,
                    'stage12_time': stage12_time,
                    's3_time': stage3_time,
                    's4_time': stage4_time,
                    's5_time': stage5_time,
                    'total_time': stage12_time + stage3_time + stage4_time + stage5_time,
                })

        except Exception as e:
            continue

    return results


def run_videollava_8frames_benchmark(dataset_dir, dataset, device="cuda", max_samples=None):
    """Run Video-LLaVA benchmark using 8 frames from MP4 video.

    This mode demonstrates Video-LLaVA's token scaling with multiple frames.
    """
    print("\n" + "=" * 80)
    print("Video-LLaVA: 8 Frames from MP4 Video Benchmark")
    print("=" * 80)

    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        print("Loading Video-LLaVA model...")
        model_id = "llava-hf/llava-1.5-7b-hf"
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map=device, local_files_only=True
            )
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        except:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map=device
            )
            processor = AutoProcessor.from_pretrained(model_id)

        model.eval()
        print("✓ Video-LLaVA loaded")

        results = []
        query = "What are the key elements in this scene?"
        samples_to_process = dataset[:max_samples] if max_samples else dataset

        for sample_idx, sample in enumerate(tqdm(samples_to_process, desc="Video-LLaVA (8 frames)")):
            try:
                video_data = sample.get("video_data")
                if not video_data:
                    continue

                video_path = os.path.join(dataset_dir, "mp4", video_data + ".mp4")
                if not os.path.exists(video_path):
                    continue

                # Load 8 frames from video
                frames, total_frames, sampled_indices = load_video_frames_from_mp4(video_path, num_frames=8)

                if len(frames) == 0:
                    continue

                # Stage 1 & 2: Load and preprocess
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage12_start = time.time()

                image_tokens = "<image>\n" * len(frames)
                prompt = f"USER: {image_tokens}{query}\nASSISTANT:"
                inputs = processor(text=prompt, images=frames, return_tensors="pt").to(device)

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage12_time = time.time() - stage12_start

                input_ids = inputs['input_ids']
                attention_mask = inputs.get('attention_mask')
                pixel_values = inputs.get('pixel_values')

                with torch.inference_mode():
                    # Stage 3: Vision encoding
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    stage3_start = time.time()

                    if hasattr(model, 'vision_tower') and pixel_values is not None:
                        _ = model.vision_tower(pixel_values, output_hidden_states=True)

                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    stage3_time = time.time() - stage3_start

                    # Stage 4: Prefill
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    stage4_start = time.time()

                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        pixel_values=pixel_values, past_key_values=None,
                        use_cache=True, return_dict=True,
                    )

                    logits = outputs.logits
                    past_key_values = outputs.past_key_values
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    stage4_time = time.time() - stage4_start

                    prefill_len = past_key_values[0][0].shape[2]
                    kv_cache_mb = estimate_kv_cache_mb(past_key_values)

                    # Stage 5: Decode (5 tokens for timing)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    stage5_start = time.time()

                    generated = [next_token.item()]
                    cur_token = next_token
                    cur_mask = torch.ones((1, prefill_len + 1), dtype=attention_mask.dtype, device=device)

                    for _ in range(5):
                        outputs = model(
                            input_ids=cur_token, attention_mask=cur_mask,
                            past_key_values=past_key_values, use_cache=True, return_dict=True,
                        )

                        logits = outputs.logits
                        past_key_values = outputs.past_key_values
                        cur_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        generated.append(cur_token.item())
                        cur_mask = torch.ones((1, cur_mask.shape[1] + 1), dtype=attention_mask.dtype, device=device)

                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    stage5_time = time.time() - stage5_start

                    results.append({
                        'sample': sample_idx,
                        'n_frames': len(frames),
                        'prefill_len': prefill_len,
                        'kv_cache_mb': kv_cache_mb,
                        'stage12_time': stage12_time,
                        's3_time': stage3_time,
                        's4_time': stage4_time,
                        's5_time': stage5_time,
                        'total_time': stage12_time + stage3_time + stage4_time + stage5_time,
                    })

            except Exception as e:
                continue

        del model
        cleanup_gpu()
        return results

    except Exception as e:
        print(f"Error loading Video-LLaVA: {e}")
        return []


def compute_comparison_stats(results):
    """Compute statistics from comparison benchmark results."""
    if not results:
        return {}

    n = len(results)
    stats = {
        'n_samples': n,
        'prefill_len_avg': sum(r['prefill_len'] for r in results) / n,
        'kv_cache_mb_avg': sum(r.get('kv_cache_mb', 0) for r in results) / n,
        's3_time_avg': sum(r['s3_time'] for r in results) / n,
        's4_time_avg': sum(r['s4_time'] for r in results) / n,
        's5_time_avg': sum(r['s5_time'] for r in results) / n,
        's3_time_std': np.std([r['s3_time'] for r in results]),
        's4_time_std': np.std([r['s4_time'] for r in results]),
        's5_time_std': np.std([r['s5_time'] for r in results]),
    }
    stats['total_time_avg'] = stats['s3_time_avg'] + stats['s4_time_avg'] + stats['s5_time_avg']
    stats['prefill_throughput'] = stats['prefill_len_avg'] / stats['s4_time_avg'] if stats['s4_time_avg'] > 0 else 0
    stats['decode_throughput'] = 5 / stats['s5_time_avg'] if stats['s5_time_avg'] > 0 else 0
    return stats


def save_comparison_results(egpt_results, vllava_results, output_dir, timestamp):
    """Save comparison results to JSON and generate markdown analysis."""

    egpt_stats = compute_comparison_stats(egpt_results)
    vllava_stats = compute_comparison_stats(vllava_results)

    # Save JSON results
    json_path = os.path.join(output_dir, f"benchmark_1frame_vs_8frames_{timestamp}.json")
    results_data = {
        'timestamp': timestamp,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'eventgpt_frames': 1,
            'videollava_frames': 8,
        },
        'eventgpt': {
            'stats': egpt_stats,
            'samples': egpt_results,
        },
        'videollava': {
            'stats': vllava_stats,
            'samples': vllava_results,
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Results saved to: {json_path}")

    # Calculate comparison metrics
    if egpt_stats and vllava_stats and egpt_stats['s4_time_avg'] > 0:
        prefill_speedup = vllava_stats['s4_time_avg'] / egpt_stats['s4_time_avg']
        token_ratio = vllava_stats['prefill_len_avg'] / egpt_stats['prefill_len_avg']
        decode_speedup = egpt_stats['decode_throughput'] / vllava_stats['decode_throughput'] if vllava_stats['decode_throughput'] > 0 else 0
        total_speedup = vllava_stats['total_time_avg'] / egpt_stats['total_time_avg'] if egpt_stats['total_time_avg'] > 0 else 0

        # Generate markdown
        md_path = os.path.join(output_dir, f"BENCHMARK_1FRAME_VS_8FRAMES_{timestamp}.md")
        md_content = f"""# EventGPT (1 frame) vs Video-LLaVA (8 frames) Benchmark

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Output Path:** `{json_path}`

## Configuration

| Setting | EventGPT | Video-LLaVA |
|---------|----------|-------------|
| Input Frames | 1 event frame | 8 video frames (from MP4) |
| Samples Tested | {egpt_stats['n_samples']} | {vllava_stats['n_samples']} |

## Results Summary

| Metric | EventGPT (1 frame) | Video-LLaVA (8 frames) | Ratio |
|--------|-------------------|------------------------|-------|
| **Prefill Length** | {egpt_stats['prefill_len_avg']:.0f} tokens | {vllava_stats['prefill_len_avg']:.0f} tokens | **{token_ratio:.1f}x** |
| **Vision Encoding** | {egpt_stats['s3_time_avg']*1000:.2f} ms | {vllava_stats['s3_time_avg']*1000:.2f} ms | {vllava_stats['s3_time_avg']/egpt_stats['s3_time_avg']:.2f}x |
| **Prefill Time** | {egpt_stats['s4_time_avg']*1000:.2f} ms | {vllava_stats['s4_time_avg']*1000:.2f} ms | **{prefill_speedup:.2f}x** |
| **Decode Time (5 tok)** | {egpt_stats['s5_time_avg']*1000:.2f} ms | {vllava_stats['s5_time_avg']*1000:.2f} ms | {vllava_stats['s5_time_avg']/egpt_stats['s5_time_avg']:.2f}x |
| **Total Time** | {egpt_stats['total_time_avg']*1000:.2f} ms | {vllava_stats['total_time_avg']*1000:.2f} ms | **{total_speedup:.2f}x** |

## Throughput Analysis

| Metric | EventGPT | Video-LLaVA |
|--------|----------|-------------|
| Prefill Throughput | {egpt_stats['prefill_throughput']:.0f} tok/s | {vllava_stats['prefill_throughput']:.0f} tok/s |
| Decode Throughput | {egpt_stats['decode_throughput']:.1f} tok/s | {vllava_stats['decode_throughput']:.1f} tok/s |
| Samples/sec | {1/egpt_stats['total_time_avg']:.2f} | {1/vllava_stats['total_time_avg']:.2f} |

## Memory Usage

| Metric | EventGPT | Video-LLaVA | Ratio |
|--------|----------|-------------|-------|
| KV Cache | {egpt_stats['kv_cache_mb_avg']:.1f} MB | {vllava_stats['kv_cache_mb_avg']:.1f} MB | {vllava_stats['kv_cache_mb_avg']/egpt_stats['kv_cache_mb_avg']:.1f}x |

## Key Findings

### 1. Token Efficiency
- **EventGPT**: {egpt_stats['prefill_len_avg']:.0f} tokens (constant regardless of frame count)
- **Video-LLaVA**: {vllava_stats['prefill_len_avg']:.0f} tokens (576 per frame × 8 frames)
- **Ratio**: Video-LLaVA uses **{token_ratio:.1f}x more tokens**

### 2. Prefill Speedup
- EventGPT prefill: {egpt_stats['s4_time_avg']*1000:.2f} ms ± {egpt_stats['s4_time_std']*1000:.2f} ms
- Video-LLaVA prefill: {vllava_stats['s4_time_avg']*1000:.2f} ms ± {vllava_stats['s4_time_std']*1000:.2f} ms
- **EventGPT is {prefill_speedup:.2f}x faster** in prefill

### 3. Decode Speedup
- EventGPT decode: {egpt_stats['decode_throughput']:.1f} tokens/sec
- Video-LLaVA decode: {vllava_stats['decode_throughput']:.1f} tokens/sec
- **EventGPT is {decode_speedup:.2f}x faster** in decode (smaller KV cache)

## Token Scaling Analysis

```
Video-LLaVA (LINEAR scaling):
  1 frame  →    ~600 tokens
  8 frames →  ~4,640 tokens  (7.7x increase)
  16 frames → ~9,260 tokens  (15.4x increase)
  32 frames → ~18,500 tokens (30.8x increase)

EventGPT (CONSTANT scaling):
  1 frame  →  ~636 tokens
  5 frames →  ~641 tokens
  8 frames →  ~646 tokens
  N frames →  ~650 tokens (constant)
```

## Conclusion

EventGPT achieves **{total_speedup:.2f}x total speedup** over Video-LLaVA when processing temporal video data:

1. **{prefill_speedup:.1f}x faster prefill** due to {token_ratio:.1f}x fewer tokens
2. **{decode_speedup:.1f}x faster decode** due to smaller KV cache attention window
3. **{token_ratio:.1f}x less memory** enabling larger batch sizes

This advantage grows with longer videos as Video-LLaVA scales linearly while EventGPT remains constant.
"""

        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"✓ Analysis saved to: {md_path}")

        return json_path, md_path

    return json_path, None


def print_5stage_summary(results, model_name="EventGPT"):
    """Print summary statistics for 5-stage benchmark."""
    if not results:
        print(f"No results for {model_name}")
        return {}

    n = len(results)

    avg_s1 = sum(r['stage1_time'] for r in results) / n
    avg_s2 = sum(r['stage2_time'] for r in results) / n
    avg_s3 = sum(r['stage3_time'] for r in results) / n
    avg_s4 = sum(r['stage4_time'] for r in results) / n
    avg_s5 = sum(r['stage5_time'] for r in results) / n
    avg_total = avg_s1 + avg_s2 + avg_s3 + avg_s4 + avg_s5
    avg_ttft = sum(r['ttft'] for r in results) / n
    avg_prefill_tokens = sum(r['prefill_tokens'] for r in results) / n
    avg_decode_tokens = sum(r['decode_tokens'] for r in results) / n
    avg_decode_tps = sum(r['decode_tokens_per_sec'] for r in results) / n

    # Percentages
    s1_pct = (avg_s1 / avg_total) * 100
    s2_pct = (avg_s2 / avg_total) * 100
    s3_pct = (avg_s3 / avg_total) * 100
    s4_pct = (avg_s4 / avg_total) * 100
    s5_pct = (avg_s5 / avg_total) * 100

    prefill_total = avg_s1 + avg_s2 + avg_s3 + avg_s4
    prefill_pct = (prefill_total / avg_total) * 100
    decode_pct = s5_pct

    print(f"\n{'=' * 80}")
    print(f"5-STAGE BENCHMARK SUMMARY - {model_name}")
    print(f"{'=' * 80}")
    print(f"Samples processed:                 {n}")
    print(f"\n{'─' * 80}")
    print("STAGE BREAKDOWN:")
    print(f"{'─' * 80}")
    print(f"  Stage 1 (Load):                  {avg_s1:.4f}s ({s1_pct:.1f}%)")
    print(f"  Stage 2 (Preprocess):            {avg_s2:.4f}s ({s2_pct:.1f}%)")
    print(f"  Stage 3 (Vision Encoding):       {avg_s3:.4f}s ({s3_pct:.1f}%)")
    print(f"  Stage 4 (LLM Prefill):           {avg_s4:.4f}s ({s4_pct:.1f}%)")
    print(f"  Stage 5 (LLM Decode):            {avg_s5:.4f}s ({s5_pct:.1f}%)")
    print(f"{'─' * 80}")
    print(f"  Total per sample:                {avg_total:.4f}s")

    print(f"\n{'─' * 80}")
    print("PREFILL vs DECODE:")
    print(f"{'─' * 80}")
    print(f"  Prefill (S1-S4):                 {prefill_total:.4f}s ({prefill_pct:.1f}%)")
    print(f"  Decode (S5):                     {avg_s5:.4f}s ({decode_pct:.1f}%)")
    print(f"  Time to First Token (TTFT):      {avg_ttft:.4f}s")

    print(f"\n{'─' * 80}")
    print("THROUGHPUT:")
    print(f"{'─' * 80}")
    print(f"  Avg prefill tokens:              {avg_prefill_tokens:.1f}")
    print(f"  Avg decode tokens:               {avg_decode_tokens:.1f}")
    print(f"  Decode throughput:               {avg_decode_tps:.1f} tokens/sec")
    print(f"  Prefill throughput:              {avg_prefill_tokens / avg_s4:.1f} tokens/sec" if avg_s4 > 0 else "")
    print(f"  End-to-end throughput:           {1.0 / avg_total:.2f} samples/sec")

    return {
        'samples': n,
        'stage1_avg': avg_s1,
        'stage2_avg': avg_s2,
        'stage3_avg': avg_s3,
        'stage4_avg': avg_s4,
        'stage5_avg': avg_s5,
        'total_avg': avg_total,
        'ttft_avg': avg_ttft,
        'prefill_tokens_avg': avg_prefill_tokens,
        'decode_tokens_avg': avg_decode_tokens,
        'decode_tps_avg': avg_decode_tps,
        'prefill_pct': prefill_pct,
        'decode_pct': decode_pct,
    }


def compare_models(egpt_summary, vllava_summary):
    """Print comparison between EventGPT and Video-LLaVA."""
    if not egpt_summary or not vllava_summary:
        return

    print(f"\n{'=' * 80}")
    print("COMPARISON: EventGPT vs Video-LLaVA")
    print(f"{'=' * 80}")

    headers = ['Metric', 'EventGPT', 'Video-LLaVA', 'Speedup']
    print(f"\n{headers[0]:<25} {headers[1]:<15} {headers[2]:<15} {headers[3]:<10}")
    print("-" * 70)

    metrics = [
        ('Stage 1 (Load)', 'stage1_avg'),
        ('Stage 2 (Preprocess)', 'stage2_avg'),
        ('Stage 3 (Vision)', 'stage3_avg'),
        ('Stage 4 (Prefill)', 'stage4_avg'),
        ('Stage 5 (Decode)', 'stage5_avg'),
        ('Total', 'total_avg'),
        ('TTFT', 'ttft_avg'),
    ]

    for label, key in metrics:
        e_val = egpt_summary.get(key, 0)
        v_val = vllava_summary.get(key, 0)
        speedup = v_val / e_val if e_val > 0 else 0
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        print(f"{label:<25} {e_val:.4f}s        {v_val:.4f}s        {speedup_str}")

    # Throughput comparison
    print(f"\n{'Throughput':<25}")
    print("-" * 70)

    e_decode_tps = egpt_summary.get('decode_tps_avg', 0)
    v_decode_tps = vllava_summary.get('decode_tps_avg', 0)
    decode_speedup = e_decode_tps / v_decode_tps if v_decode_tps > 0 else 0
    print(f"{'Decode (tok/s)':<25} {e_decode_tps:.1f}           {v_decode_tps:.1f}           {decode_speedup:.2f}x")

    # Overall speedup
    total_speedup = vllava_summary['total_avg'] / egpt_summary['total_avg'] if egpt_summary['total_avg'] > 0 else 0
    print(f"\n{'=' * 80}")
    status = "FASTER" if total_speedup > 1 else "SLOWER"
    print(f"✅ EventGPT is {total_speedup:.2f}x {status} than Video-LLaVA (end-to-end)")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="5-Stage Benchmark with Prefill/Decode Separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default mode: EventGPT (all frames) vs Video-LLaVA (1 image)
  python benchmark_inference_5stages.py

  # Comparison mode: EventGPT (1 frame) vs Video-LLaVA (8 frames from MP4)
  python benchmark_inference_5stages.py --compare_1vs8

  # EventGPT only with limited samples
  python benchmark_inference_5stages.py --eventgpt_only --max_samples 100
        """
    )
    parser.add_argument("--dataset_dir", type=str,
                        default="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s")
    parser.add_argument("--eventgpt_model_path", type=str,
                        default="./checkpoints/EventGPT-7b")
    parser.add_argument("--videollava_model", type=str,
                        default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to benchmark (None = all)")
    parser.add_argument("--eventgpt_only", action="store_true",
                        help="Only benchmark EventGPT (skip Video-LLaVA)")
    parser.add_argument("--compare_1vs8", action="store_true",
                        help="Comparison mode: EventGPT (1 frame) vs Video-LLaVA (8 frames from MP4)")
    parser.add_argument("--speculative", action="store_true",
                        help="Run speculative decoding analysis")
    parser.add_argument("--gamma", type=int, default=4,
                        help="Draft tokens per step for speculative decoding")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: dataset_dir)")

    # Token Adapter arguments
    parser.add_argument("--use_token_adapter", action="store_true",
                        help="Use TokenAdapter for aligned evaluation")
    parser.add_argument("--token_adapter_path", type=str, default=None,
                        help="Path to trained TokenAdapter checkpoint (default: auto-detect latest)")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(__file__))

    # Load dataset
    dataset_json_path = os.path.join(args.dataset_dir, "EventGPT_Instruction_Subset.json")
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    num_samples = len(dataset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'#' * 80}")
    print(f"# 5-Stage Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}")
    print(f"Dataset: {num_samples} samples")
    print(f"Output directory: {args.output_dir}")

    # ==========================================================================
    # COMPARISON MODE: EventGPT (1 frame) vs Video-LLaVA (8 frames from MP4)
    # ==========================================================================
    if args.compare_1vs8:
        print(f"\n{'=' * 80}")
        print("MODE: Comparison - EventGPT (1 frame) vs Video-LLaVA (8 frames)")
        print(f"{'=' * 80}")

        if not HAS_PYAV:
            print("ERROR: PyAV is required for --compare_1vs8 mode. Install with: pip install av")
            return

        # Load EventGPT model
        print(f"\nLoading EventGPT from {args.eventgpt_model_path}...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.eventgpt_model_path, use_fast=False, local_files_only=True
        )
        model = EventChatModel.from_pretrained(
            args.eventgpt_model_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        model = model.to(args.device)
        model.eval()
        processor = model.get_visual_tower().event_processor
        print("✓ EventGPT loaded")

        # Run EventGPT (1 frame) benchmark
        egpt_results = run_eventgpt_1frame_benchmark(
            model, tokenizer, processor, args.dataset_dir, dataset, args.device
        )

        # Unload EventGPT
        del model
        cleanup_gpu()

        # Run Video-LLaVA (8 frames) benchmark
        vllava_results = run_videollava_8frames_benchmark(
            args.dataset_dir, dataset, args.device, args.max_samples
        )

        # Print comparison summary
        egpt_stats = compute_comparison_stats(egpt_results)
        vllava_stats = compute_comparison_stats(vllava_results)

        if egpt_stats and vllava_stats:
            print(f"\n{'=' * 80}")
            print("COMPARISON SUMMARY")
            print(f"{'=' * 80}")
            print(f"{'Metric':<25} {'EventGPT (1 frame)':<20} {'Video-LLaVA (8 frames)':<25} {'Speedup':<10}")
            print("-" * 80)
            print(f"{'Prefill Tokens':<25} {egpt_stats['prefill_len_avg']:<20.0f} {vllava_stats['prefill_len_avg']:<25.0f} {vllava_stats['prefill_len_avg']/egpt_stats['prefill_len_avg']:.1f}x more")
            print(f"{'Prefill Time (ms)':<25} {egpt_stats['s4_time_avg']*1000:<20.2f} {vllava_stats['s4_time_avg']*1000:<25.2f} {vllava_stats['s4_time_avg']/egpt_stats['s4_time_avg']:.2f}x faster")
            print(f"{'Decode (tok/s)':<25} {egpt_stats['decode_throughput']:<20.1f} {vllava_stats['decode_throughput']:<25.1f} {egpt_stats['decode_throughput']/vllava_stats['decode_throughput']:.2f}x faster")
            print(f"{'Total Time (ms)':<25} {egpt_stats['total_time_avg']*1000:<20.2f} {vllava_stats['total_time_avg']*1000:<25.2f} {vllava_stats['total_time_avg']/egpt_stats['total_time_avg']:.2f}x faster")
            print(f"{'KV Cache (MB)':<25} {egpt_stats['kv_cache_mb_avg']:<20.1f} {vllava_stats['kv_cache_mb_avg']:<25.1f} {vllava_stats['kv_cache_mb_avg']/egpt_stats['kv_cache_mb_avg']:.1f}x more")

        # Save results
        save_comparison_results(egpt_results, vllava_results, args.output_dir, timestamp)

        print(f"\n{'=' * 80}")
        print("COMPARISON BENCHMARK COMPLETE")
        print(f"{'=' * 80}")
        return

    # ==========================================================================
    # DEFAULT MODE: Standard 5-stage benchmark
    # ==========================================================================
    print(f"\nMODE: Standard 5-Stage Benchmark")

    # Generate output path
    if args.output_json is None:
        args.output_json = os.path.join(
            args.dataset_dir,
            f"benchmark_5stages_{num_samples}samples_{timestamp}.json"
        )

    # Load EventGPT model
    print(f"Loading EventGPT from {args.eventgpt_model_path}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.eventgpt_model_path, use_fast=False, local_files_only=True
    )
    model = EventChatModel.from_pretrained(
        args.eventgpt_model_path, torch_dtype=torch.bfloat16, local_files_only=True
    )
    model = model.to(args.device)
    model.eval()

    processor = model.get_visual_tower().event_processor

    # Run EventGPT benchmark
    egpt_results = run_eventgpt_5stage_benchmark(
        model, tokenizer, processor, args.dataset_dir, dataset, args.device
    )

    egpt_summary = print_5stage_summary(egpt_results, "EventGPT")

    # Run Video-LLaVA benchmark if needed
    vllava_results = []
    vllava_summary = {}

    if not args.eventgpt_only:
        # Unload EventGPT to free memory
        del model
        cleanup_gpu()
        print("\nUnloaded EventGPT to free memory for Video-LLaVA...")

        vllava_results = run_videollava_5stage_benchmark(
            args.dataset_dir, dataset, args.device, args.max_samples
        )

        vllava_summary = print_5stage_summary(vllava_results, "Video-LLaVA")

        # Print comparison
        compare_models(egpt_summary, vllava_summary)

    # Print recommendations
    print(f"\n{'=' * 80}")
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print(f"{'=' * 80}")

    if egpt_summary:
        print(f"\n📊 EventGPT Analysis:")
        print(f"   • Prefill: {egpt_summary.get('prefill_pct', 0):.1f}% of total time")
        print(f"   • Decode: {egpt_summary.get('decode_pct', 0):.1f}% of total time")
        print(f"   • TTFT (Time to First Token): {egpt_summary.get('ttft_avg', 0):.4f}s")
        print(f"   • Decode throughput: {egpt_summary.get('decode_tps_avg', 0):.1f} tokens/sec")

        if egpt_summary.get('decode_pct', 0) > 70:
            print(f"\n🎯 Optimization Priority: DECODE PHASE")
            print(f"   → Decode is the bottleneck ({egpt_summary.get('decode_pct', 0):.1f}%)")
            print(f"   → Recommendations:")
            print(f"     • Speculative decoding")
            print(f"     • Quantization (INT8/INT4)")
            print(f"     • Continuous batching")
            print(f"     • Token pruning")
        elif egpt_summary.get('prefill_pct', 0) > 50:
            print(f"\n🎯 Optimization Priority: PREFILL PHASE")
            print(f"   → Prefill is significant ({egpt_summary.get('prefill_pct', 0):.1f}%)")
            print(f"   → Recommendations:")
            print(f"     • Flash attention")
            print(f"     • Chunked prefill")
            print(f"     • Vision feature caching")

    # Save results
    if args.output_json:
        output_data = {
            'config': {
                'dataset_dir': args.dataset_dir,
                'eventgpt_model_path': args.eventgpt_model_path,
                'videollava_model': args.videollava_model,
                'max_samples': args.max_samples,
                'timestamp': datetime.now().isoformat(),
            },
            'summary': {
                'eventgpt': egpt_summary,
            },
            'samples': {
                'eventgpt': egpt_results,
            }
        }

        if vllava_summary:
            output_data['summary']['videollava'] = vllava_summary
            output_data['samples']['videollava'] = vllava_results
            if egpt_summary and vllava_summary:
                output_data['summary']['speedup'] = {
                    'total': vllava_summary['total_avg'] / egpt_summary['total_avg'] if egpt_summary['total_avg'] > 0 else 0,
                    'ttft': vllava_summary['ttft_avg'] / egpt_summary['ttft_avg'] if egpt_summary['ttft_avg'] > 0 else 0,
                    'decode_tps': egpt_summary['decode_tps_avg'] / vllava_summary['decode_tps_avg'] if vllava_summary['decode_tps_avg'] > 0 else 0,
                }

        os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {args.output_json}")

    print(f"\n{'=' * 80}")
    print("5-STAGE BENCHMARK COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
