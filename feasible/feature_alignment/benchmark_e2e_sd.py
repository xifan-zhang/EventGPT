#!/usr/bin/env python3
"""
End-to-End Speculative Decoding Benchmark
==========================================

Real wall-clock measurement of speculative decoding with:
  - EventGPT as drafter (fast prefill)
  - Video-LLaVA as verifier (target model)
  - Hidden state adapter for draft token generation

Two-phase SD pipeline:
  Phase 1 (Prefill Hiding): EGPT prefills while VL prefills in parallel.
    Draft tokens generated during VL's slower prefill are "free".
  Phase 2 (Decode): Standard SD with gamma=5 draft tokens per iteration.

Usage:
    python benchmark_e2e_sd.py \\
        --adapter_checkpoint ./tasks/L1/.../best_model.pt \\
        --dataset_dir ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \\
        --max_samples 100 --gamma_decode 5

Requires GPU. Both models loaded in 4-bit (~8GB VRAM total).
"""

import os
import sys
import json
import time
import gc
import argparse
import threading
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass, asdict

# Fix protobuf compatibility
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from feasible.feature_alignment.hidden_adapter import load_any_adapter


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class BenchmarkConfig:
    adapter_checkpoint: str = ""
    dataset_dir: str = "./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s"
    max_samples: int = 100
    max_new_tokens: int = 50
    gamma_decode: int = 5
    warmup_samples: int = 3
    output_dir: str = ""
    device: str = "cuda"


# =========================================================================
# Model loading
# =========================================================================

def load_models(device: str = "cuda"):
    """Load EventGPT + Video-LLaVA in 4-bit."""
    from transformers import (
        AutoTokenizer, BitsAndBytesConfig,
        VideoLlavaForConditionalGeneration, VideoLlavaProcessor,
    )
    from model.EventChatModel import EventChatModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("\n[1/2] Loading EventGPT (4-bit)...")
    eventgpt_path = "./checkpoints/EventGPT-7b"
    egpt_model = EventChatModel.from_pretrained(
        eventgpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    egpt_model.eval()
    egpt_tokenizer = AutoTokenizer.from_pretrained(eventgpt_path, use_fast=True)
    egpt_processor = egpt_model.get_visual_tower().event_processor
    mem_egpt = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"  GPU Memory: {mem_egpt:.0f} MB")

    print("\n[2/2] Loading Video-LLaVA (4-bit)...")
    vl_model_id = "LanguageBind/Video-LLaVA-7B-hf"
    vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
        vl_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    vl_processor = VideoLlavaProcessor.from_pretrained(vl_model_id)
    vl_model.eval()
    mem_total = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"  GPU Memory: {mem_total:.0f} MB (total both models)")

    return egpt_model, egpt_tokenizer, egpt_processor, vl_model, vl_processor


# =========================================================================
# EventGPT inference helpers
# =========================================================================

def egpt_prefill_and_get_hidden(
    model, tokenizer, processor,
    sample: Dict, dataset_dir: str, query: str,
    device: str = "cuda",
) -> Optional[Dict]:
    """
    Run EventGPT prefill: vision encoding + LLM prefill.

    Returns dict with:
        kv_cache, first_token, hidden_states (from prefill last layer),
        prefill_time_ms
    """
    from model.EventChatModel import get_spatio_temporal_features
    from common.common import load_image
    from dataset.conversation import prepare_event_prompt
    from dataset.constants import EVENT_TOKEN_INDEX

    event_image_paths = sample.get("event_image", [])
    if not event_image_paths:
        return None

    # Load + preprocess first event frame
    img_path = os.path.join(dataset_dir, "event_image", event_image_paths[0])
    img = load_image(img_path)
    img_array = np.array(img)
    event_image_size = list(img_array.shape[:2])

    event = processor(img_array, return_tensors='pt')['pixel_values'][0]
    event = event.to(device, dtype=torch.bfloat16)

    # Prepare text input
    conv_mode = 'eventgpt_v1'
    prompt = prepare_event_prompt(query, conv_mode)
    from common.common import tokenizer_event_token
    input_ids = tokenizer_event_token(
        prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    torch.cuda.synchronize()
    t_start = time.time()

    with torch.inference_mode():
        # Vision encoding
        ev = event.unsqueeze(0)
        feature = model.visval_encode(ev)
        feature = model.get_model().feature_adaptor(feature)
        feature = feature.squeeze(0)
        event_features = get_spatio_temporal_features([feature])
        event_features = event_features.unsqueeze(0)

        # Prepare multimodal inputs
        (
            _, position_ids, attention_mask, past_key_values,
            inputs_embeds, _
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None,
            torch.ones_like(input_ids, dtype=torch.bool),
            None, None,
            event_tensors=None,
            event_image_sizes=event_image_size,
            event_features=event_features,
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                (1, inputs_embeds.shape[1]), dtype=torch.bool, device=device
            )
        if position_ids is None:
            position_ids = torch.arange(
                0, inputs_embeds.shape[1], dtype=torch.long, device=device
            ).unsqueeze(0)

        # Prefill forward pass (with hidden states for adapter)
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        kv_cache = outputs.past_key_values
        hidden_states = outputs.last_hidden_state  # [1, seq, 4096]
        logits = model.lm_head(hidden_states[:, -1:, :])
        first_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    torch.cuda.synchronize()
    prefill_ms = (time.time() - t_start) * 1000

    return {
        "kv_cache": kv_cache,
        "first_token": first_token,
        "hidden_states": hidden_states,
        "prefill_ms": prefill_ms,
        "prefill_length": inputs_embeds.shape[1],
        "attention_mask": attention_mask,
    }


def egpt_decode_tokens(
    model, kv_cache, first_token, attention_mask,
    prefill_length: int, num_tokens: int,
    device: str = "cuda",
) -> Tuple[List[int], float]:
    """
    Generate tokens autoregressively from EventGPT (for drafting during overlap).

    Returns (token_ids, decode_time_ms).
    """
    torch.cuda.synchronize()
    t_start = time.time()

    generated = [first_token.item()]
    cur_token = first_token
    cur_pos = prefill_length

    with torch.inference_mode():
        for _ in range(num_tokens - 1):
            cur_embed = model.get_model().embed_tokens(cur_token)
            new_mask = torch.ones(
                (1, cur_pos + 1), dtype=torch.bool, device=device
            )
            outputs = model.model(
                inputs_embeds=cur_embed,
                attention_mask=new_mask,
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=kv_cache,
                use_cache=True,
            )
            logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
            kv_cache = outputs.past_key_values
            cur_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(cur_token.item())
            cur_pos += 1

    torch.cuda.synchronize()
    decode_ms = (time.time() - t_start) * 1000

    return generated, decode_ms


# =========================================================================
# Video-LLaVA inference helpers
# =========================================================================

def vl_prefill(
    model, processor, sample: Dict, dataset_dir: str, query: str,
    device: str = "cuda",
) -> Optional[Dict]:
    """
    Run Video-LLaVA full prefill.

    Returns dict with kv_cache, first_token, logits, prefill_time_ms.
    """
    from common.common import load_image

    event_image_paths = sample.get("event_image", [])
    if not event_image_paths:
        return None

    # Load RGB image(s) for Video-LLaVA
    rgb_images = []
    for img_path in event_image_paths:
        full_path = os.path.join(dataset_dir, "event_image", img_path)
        try:
            rgb_images.append(load_image(full_path))
        except Exception:
            continue

    if not rgb_images:
        return None

    # Pad to 8 frames for video mode
    while len(rgb_images) < 8:
        rgb_images.append(rgb_images[-1])

    # Preprocess
    prompt = f"USER: <video>\n{query}\nASSISTANT:"
    inputs = processor(text=prompt, videos=rgb_images, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs.get('attention_mask')
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(device)
    pixel_values_videos = inputs.get('pixel_values_videos')
    if pixel_values_videos is None:
        return None

    torch.cuda.synchronize()
    t_start = time.time()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos.to(device),
            past_key_values=None,
            use_cache=True,
        )
        logits = outputs.logits
        kv_cache = outputs.past_key_values
        first_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    torch.cuda.synchronize()
    prefill_ms = (time.time() - t_start) * 1000

    return {
        "kv_cache": kv_cache,
        "first_token": first_token,
        "prefill_ms": prefill_ms,
        "prefill_length": input_ids.shape[1],
        "attention_mask": attention_mask,
    }


def vl_decode_baseline(
    model, kv_cache, first_token, attention_mask,
    prefill_length: int, max_new_tokens: int,
    eos_token_id: int, device: str = "cuda",
) -> Tuple[List[int], float]:
    """
    Standard autoregressive decode from Video-LLaVA (baseline).

    Returns (token_ids, decode_time_ms).
    """
    torch.cuda.synchronize()
    t_start = time.time()

    generated = [first_token.item()]
    cur_token = first_token
    attn_dtype = attention_mask.dtype if isinstance(attention_mask, torch.Tensor) else torch.long
    cur_attention_mask = torch.ones(
        (1, kv_cache[0][0].shape[2] + 1), dtype=attn_dtype, device=device
    )

    with torch.inference_mode():
        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=cur_token,
                attention_mask=cur_attention_mask,
                past_key_values=kv_cache,
                use_cache=True,
            )
            logits = outputs.logits
            kv_cache = outputs.past_key_values
            cur_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(cur_token.item())

            cur_attention_mask = torch.ones(
                (1, cur_attention_mask.shape[1] + 1), dtype=attn_dtype, device=device
            )

            if cur_token.item() == eos_token_id:
                break

    torch.cuda.synchronize()
    decode_ms = (time.time() - t_start) * 1000

    return generated, decode_ms


def vl_verify_token(
    model, draft_token_id: int, kv_cache, kv_len: int,
    attention_mask_dtype, device: str = "cuda",
) -> Tuple[int, object]:
    """
    Run a single VL forward pass to verify a draft token.

    Returns (target_token_id, updated_kv_cache).
    """
    cur_token = torch.tensor([[draft_token_id]], device=device)
    attn = torch.ones((1, kv_len + 1), dtype=attention_mask_dtype, device=device)

    with torch.inference_mode():
        outputs = model(
            input_ids=cur_token,
            attention_mask=attn,
            past_key_values=kv_cache,
            use_cache=True,
        )
    logits = outputs.logits
    target_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    return target_id, outputs.past_key_values


# =========================================================================
# Parallel prefill (threading + CUDA streams)
# =========================================================================

def parallel_prefill(
    egpt_model, egpt_tokenizer, egpt_processor,
    vl_model, vl_processor,
    sample: Dict, dataset_dir: str, query: str,
    device: str = "cuda",
) -> Optional[Dict]:
    """
    Run EGPT and VL prefills in parallel using two threads + CUDA streams.

    On a single GPU, the CUDA scheduler interleaves kernels from both streams.
    We measure real wall-clock overlap.

    Returns dict with both prefill results + timing breakdown, or None on error.
    """
    egpt_stream = torch.cuda.Stream(device=device)
    vl_stream = torch.cuda.Stream(device=device)

    egpt_result_box = [None]
    vl_result_box = [None]
    egpt_error_box = [None]
    vl_error_box = [None]

    def run_egpt():
        try:
            with torch.cuda.stream(egpt_stream):
                egpt_result_box[0] = egpt_prefill_and_get_hidden(
                    egpt_model, egpt_tokenizer, egpt_processor,
                    sample, dataset_dir, query, device,
                )
        except Exception as e:
            egpt_error_box[0] = e

    def run_vl():
        try:
            with torch.cuda.stream(vl_stream):
                vl_result_box[0] = vl_prefill(
                    vl_model, vl_processor, sample, dataset_dir, query, device,
                )
        except Exception as e:
            vl_error_box[0] = e

    # Synchronize before launch to get clean timing
    torch.cuda.synchronize()
    wall_start = time.time()

    t_egpt = threading.Thread(target=run_egpt)
    t_vl = threading.Thread(target=run_vl)

    t_egpt.start()
    t_vl.start()

    t_egpt.join()
    t_vl.join()

    # Wait for both streams to finish
    egpt_stream.synchronize()
    vl_stream.synchronize()
    torch.cuda.synchronize()

    wall_ms = (time.time() - wall_start) * 1000

    if egpt_error_box[0]:
        raise egpt_error_box[0]
    if vl_error_box[0]:
        raise vl_error_box[0]

    egpt_result = egpt_result_box[0]
    vl_result = vl_result_box[0]

    if egpt_result is None or vl_result is None:
        return None

    # Compute overlap
    egpt_ms = egpt_result["prefill_ms"]
    vl_ms = vl_result["prefill_ms"]
    sequential_ms = egpt_ms + vl_ms
    overlap_ms = sequential_ms - wall_ms  # Positive = real overlap achieved

    return {
        "egpt": egpt_result,
        "vl": vl_result,
        "wall_ms": wall_ms,
        "egpt_ms": egpt_ms,
        "vl_ms": vl_ms,
        "sequential_ms": sequential_ms,
        "overlap_ms": max(0, overlap_ms),
        "overlap_ratio": max(0, overlap_ms) / sequential_ms if sequential_ms > 0 else 0,
    }


# =========================================================================
# Speculative Decoding loop
# =========================================================================

def run_sd_decode(
    vl_model, adapter, vl_lm_head_weight,
    egpt_hidden_states: torch.Tensor,
    vl_kv_cache, vl_first_token: int,
    attention_mask_dtype,
    kv_len_start: int,
    gamma: int = 5,
    max_new_tokens: int = 50,
    eos_token_id: int = 2,
    device: str = "cuda",
) -> Tuple[List[int], float, Dict]:
    """
    Speculative decoding loop using adapter-projected draft tokens.

    Phase 2 (Decode):
        1. Generate gamma draft tokens via adapter(EGPT_hidden) + VL LM head
        2. Verify each draft token with VL forward pass
        3. Accept consecutive prefix, reject rest
        4. Continue from rejection point

    Returns (generated_tokens, decode_time_ms, stats_dict).
    """
    W = vl_lm_head_weight.to(device)  # [V, D]

    torch.cuda.synchronize()
    t_start = time.time()

    generated = [vl_first_token]
    kv_cache = vl_kv_cache
    kv_len = kv_len_start
    total_drafted = 0
    total_accepted = 0
    iterations = 0

    with torch.inference_mode():
        while len(generated) < max_new_tokens:
            iterations += 1

            # --- Draft phase: project adapter(EGPT_hidden) → tokens ---
            # Use positions starting from current decode position
            draft_start = len(generated) - 1
            draft_end = min(draft_start + gamma, egpt_hidden_states.shape[1])
            n_draft = draft_end - draft_start

            if n_draft <= 0:
                # No more EGPT hidden states to draft from -- fall back to
                # standard autoregressive
                cur_token_t = torch.tensor([[generated[-1]]], device=device)
                attn = torch.ones((1, kv_len + 1), dtype=attention_mask_dtype, device=device)
                outputs = vl_model(
                    input_ids=cur_token_t,
                    attention_mask=attn,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                kv_cache = outputs.past_key_values
                kv_len += 1
                next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
                generated.append(next_id)
                if next_id == eos_token_id:
                    break
                continue

            egpt_chunk = egpt_hidden_states[:, draft_start:draft_end, :]  # [1, n_draft, D]
            aligned = adapter(egpt_chunk)  # [1, n_draft, D]

            # Project to tokens
            draft_logits = torch.matmul(aligned, W.T)  # [1, n_draft, V]
            draft_tokens = draft_logits.argmax(dim=-1)[0].tolist()  # [n_draft]
            total_drafted += n_draft

            # --- Verify phase: VL forward passes ---
            accepted = 0
            for i, draft_id in enumerate(draft_tokens):
                target_id, kv_cache = vl_verify_token(
                    vl_model, draft_id, kv_cache, kv_len,
                    attention_mask_dtype, device,
                )
                kv_len += 1

                if draft_id == target_id:
                    accepted += 1
                    generated.append(draft_id)
                    if draft_id == eos_token_id:
                        break
                else:
                    # Reject: use target token instead
                    generated.append(target_id)
                    break

            total_accepted += accepted

            if generated[-1] == eos_token_id:
                break

    torch.cuda.synchronize()
    decode_ms = (time.time() - t_start) * 1000

    stats = {
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "iterations": iterations,
        "accept_rate": total_accepted / total_drafted if total_drafted > 0 else 0,
        "tokens_per_iter": len(generated) / iterations if iterations > 0 else 0,
    }

    return generated, decode_ms, stats


# =========================================================================
# Benchmark runner
# =========================================================================

def run_benchmark(config: BenchmarkConfig):
    """Run full E2E speculative decoding benchmark."""
    device = config.device
    print("=" * 80)
    print("E2E SPECULATIVE DECODING BENCHMARK")
    print(f"  Adapter:       {config.adapter_checkpoint}")
    print(f"  Dataset:       {config.dataset_dir}")
    print(f"  Max samples:   {config.max_samples}")
    print(f"  gamma_decode:  {config.gamma_decode}")
    print(f"  Max new tokens:{config.max_new_tokens}")
    print("=" * 80)

    # Load models
    egpt_model, egpt_tokenizer, egpt_processor, vl_model, vl_processor = load_models(device)

    # Load adapter
    print(f"\nLoading adapter: {config.adapter_checkpoint}")
    adapter, ckpt = load_any_adapter(config.adapter_checkpoint, device)
    adapter = adapter.to(device)
    adapter.eval()
    print(f"  Type: {type(adapter).__name__}")
    print(f"  Params: {adapter.get_num_parameters():,}")

    # Get VL LM head (not quantized)
    vl_lm_head_weight = vl_model.language_model.lm_head.weight.data.float()
    print(f"  VL LM head: {list(vl_lm_head_weight.shape)}")

    eos_token_id = vl_processor.tokenizer.eos_token_id

    # Load dataset
    json_path = os.path.join(config.dataset_dir, "EventGPT_Instruction_Subset.json")
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    if config.max_samples > 0:
        dataset = dataset[:config.max_samples + config.warmup_samples]

    query = "What are the key elements in this scene?"

    # Results
    results = []

    for idx, sample in enumerate(tqdm(dataset, desc="E2E SD Benchmark")):
        is_warmup = idx < config.warmup_samples

        try:
            # =============================================================
            # BASELINE: Video-LLaVA standalone
            # =============================================================
            vl_result = vl_prefill(
                vl_model, vl_processor, sample, config.dataset_dir, query, device,
            )
            if vl_result is None:
                continue

            baseline_tokens, baseline_decode_ms = vl_decode_baseline(
                vl_model, vl_result["kv_cache"], vl_result["first_token"],
                vl_result["attention_mask"], vl_result["prefill_length"],
                config.max_new_tokens, eos_token_id, device,
            )
            baseline_total_ms = vl_result["prefill_ms"] + baseline_decode_ms

            # Free VL KV cache
            del vl_result["kv_cache"]
            torch.cuda.empty_cache()

            # =============================================================
            # SD RUN: Parallel prefill + SD decode
            # =============================================================

            # Step 1+2: EGPT and VL prefill in parallel (real wall-clock)
            par = parallel_prefill(
                egpt_model, egpt_tokenizer, egpt_processor,
                vl_model, vl_processor,
                sample, config.dataset_dir, query, device,
            )
            if par is None:
                continue

            egpt_result = par["egpt"]
            vl_result2 = par["vl"]
            parallel_prefill_ms = par["wall_ms"]  # Real measured wall-clock

            # Step 3: Adapter alignment + SD decode
            egpt_hidden = egpt_result["hidden_states"]  # [1, seq, 4096]

            attn_dtype = (
                vl_result2["attention_mask"].dtype
                if isinstance(vl_result2["attention_mask"], torch.Tensor)
                else torch.long
            )
            kv_len = vl_result2["kv_cache"][0][0].shape[2]

            sd_tokens, sd_decode_ms, sd_stats = run_sd_decode(
                vl_model, adapter, vl_lm_head_weight,
                egpt_hidden,
                vl_result2["kv_cache"], vl_result2["first_token"].item(),
                attn_dtype, kv_len,
                gamma=config.gamma_decode,
                max_new_tokens=config.max_new_tokens,
                eos_token_id=eos_token_id,
                device=device,
            )

            sd_total_ms = parallel_prefill_ms + sd_decode_ms
            speedup = baseline_total_ms / sd_total_ms if sd_total_ms > 0 else 1.0

            # Capture timing before cleanup
            sd_egpt_prefill_ms = par["egpt_ms"]
            sd_vl_prefill_ms = par["vl_ms"]
            overlap_ms = par["overlap_ms"]
            overlap_ratio = par["overlap_ratio"]

            # Cleanup
            del egpt_result, vl_result2, par
            torch.cuda.empty_cache()

            if is_warmup:
                continue

            result = {
                "sample": idx,
                # Baseline
                "baseline_prefill_ms": vl_result["prefill_ms"],
                "baseline_decode_ms": baseline_decode_ms,
                "baseline_total_ms": baseline_total_ms,
                "baseline_tokens": len(baseline_tokens),
                # SD - parallel prefill (real wall-clock)
                "sd_egpt_prefill_ms": sd_egpt_prefill_ms,
                "sd_vl_prefill_ms": sd_vl_prefill_ms,
                "sd_parallel_prefill_ms": parallel_prefill_ms,
                "sd_overlap_ms": overlap_ms,
                "sd_overlap_ratio": overlap_ratio,
                # SD - decode
                "sd_decode_ms": sd_decode_ms,
                "sd_total_ms": sd_total_ms,
                "sd_tokens": len(sd_tokens),
                "sd_drafted": sd_stats["total_drafted"],
                "sd_accepted": sd_stats["total_accepted"],
                "sd_accept_rate": sd_stats["accept_rate"],
                "sd_iterations": sd_stats["iterations"],
                "sd_tokens_per_iter": sd_stats["tokens_per_iter"],
                # Speedup
                "speedup": speedup,
            }
            results.append(result)

            tqdm.write(
                f"  Sample {idx}: baseline={baseline_total_ms:.0f}ms  "
                f"SD={sd_total_ms:.0f}ms  speedup={speedup:.2f}x  "
                f"accept={sd_stats['accept_rate']:.1%}  "
                f"overlap={overlap_ms:.0f}ms({overlap_ratio:.0%})"
            )

        except Exception as e:
            import traceback
            tqdm.write(f"  Error sample {idx}: {e}")
            traceback.print_exc()
            continue

    # =====================================================================
    # Aggregate results
    # =====================================================================
    if not results:
        print("\nNo results collected!")
        return

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    n = len(results)

    def avg(key):
        return np.mean([r[key] for r in results])

    def std(key):
        return np.std([r[key] for r in results])

    print(f"\n  Samples: {n}")
    print(f"\n  BASELINE (Video-LLaVA standalone):")
    print(f"    Prefill:  {avg('baseline_prefill_ms'):7.1f} +/- {std('baseline_prefill_ms'):5.1f} ms")
    print(f"    Decode:   {avg('baseline_decode_ms'):7.1f} +/- {std('baseline_decode_ms'):5.1f} ms")
    print(f"    Total:    {avg('baseline_total_ms'):7.1f} +/- {std('baseline_total_ms'):5.1f} ms")
    print(f"    Tokens:   {avg('baseline_tokens'):7.1f}")

    print(f"\n  SPECULATIVE DECODING (parallel prefill):")
    print(f"    EGPT prefill:     {avg('sd_egpt_prefill_ms'):7.1f} +/- {std('sd_egpt_prefill_ms'):5.1f} ms")
    print(f"    VL prefill:       {avg('sd_vl_prefill_ms'):7.1f} +/- {std('sd_vl_prefill_ms'):5.1f} ms")
    print(f"    Parallel (wall):  {avg('sd_parallel_prefill_ms'):7.1f} +/- {std('sd_parallel_prefill_ms'):5.1f} ms")
    print(f"    Overlap:          {avg('sd_overlap_ms'):7.1f} ms ({avg('sd_overlap_ratio'):5.1%})")
    print(f"    SD decode:        {avg('sd_decode_ms'):7.1f} +/- {std('sd_decode_ms'):5.1f} ms")
    print(f"    Total:            {avg('sd_total_ms'):7.1f} +/- {std('sd_total_ms'):5.1f} ms")
    print(f"    Tokens:           {avg('sd_tokens'):7.1f}")
    print(f"    Accept rate:      {avg('sd_accept_rate'):7.1%}")
    print(f"    Tokens/iter:      {avg('sd_tokens_per_iter'):7.2f}")

    speedup_val = avg('speedup')
    print(f"\n  SPEEDUP: {speedup_val:.2f}x")
    print("=" * 80)

    # =====================================================================
    # Save results
    # =====================================================================
    if config.output_dir:
        output_dir = Path(config.output_dir)
    else:
        output_dir = Path(config.adapter_checkpoint).parent

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON results
    output_data = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "num_samples": n,
        "aggregate": {
            "baseline_prefill_ms": {"mean": avg('baseline_prefill_ms'), "std": std('baseline_prefill_ms')},
            "baseline_decode_ms": {"mean": avg('baseline_decode_ms'), "std": std('baseline_decode_ms')},
            "baseline_total_ms": {"mean": avg('baseline_total_ms'), "std": std('baseline_total_ms')},
            "sd_egpt_prefill_ms": {"mean": avg('sd_egpt_prefill_ms'), "std": std('sd_egpt_prefill_ms')},
            "sd_vl_prefill_ms": {"mean": avg('sd_vl_prefill_ms'), "std": std('sd_vl_prefill_ms')},
            "sd_parallel_prefill_ms": {"mean": avg('sd_parallel_prefill_ms'), "std": std('sd_parallel_prefill_ms')},
            "sd_overlap_ms": {"mean": avg('sd_overlap_ms'), "std": std('sd_overlap_ms')},
            "sd_overlap_ratio": {"mean": avg('sd_overlap_ratio'), "std": std('sd_overlap_ratio')},
            "sd_decode_ms": {"mean": avg('sd_decode_ms'), "std": std('sd_decode_ms')},
            "sd_total_ms": {"mean": avg('sd_total_ms'), "std": std('sd_total_ms')},
            "sd_accept_rate": {"mean": avg('sd_accept_rate'), "std": std('sd_accept_rate')},
            "sd_tokens_per_iter": {"mean": avg('sd_tokens_per_iter'), "std": std('sd_tokens_per_iter')},
            "speedup": {"mean": speedup_val, "std": std('speedup')},
        },
        "per_sample": results,
    }

    json_path = output_dir / f"benchmark_e2e_sd_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Markdown report
    md_path = output_dir / f"benchmark_e2e_sd_{timestamp}.md"
    md_lines = [
        "# E2E Speculative Decoding Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Configuration",
        f"- Adapter: `{config.adapter_checkpoint}`",
        f"- Dataset: `{config.dataset_dir}`",
        f"- Samples: {n}",
        f"- gamma_decode: {config.gamma_decode}",
        f"- max_new_tokens: {config.max_new_tokens}",
        "",
        "## Results",
        "",
        "| Metric | Baseline | SD | Speedup |",
        "|--------|---------|----|---------:|",
        f"| Prefill (ms) | {avg('baseline_prefill_ms'):.1f} | {avg('sd_parallel_prefill_ms'):.1f} (parallel) | {avg('baseline_prefill_ms')/avg('sd_parallel_prefill_ms'):.2f}x |",
        f"| Decode (ms) | {avg('baseline_decode_ms'):.1f} | {avg('sd_decode_ms'):.1f} | {avg('baseline_decode_ms')/avg('sd_decode_ms'):.2f}x |" if avg('sd_decode_ms') > 0 else "| Decode (ms) | - | - | - |",
        f"| Total (ms) | {avg('baseline_total_ms'):.1f} | {avg('sd_total_ms'):.1f} | **{speedup_val:.2f}x** |",
        "",
        "## Parallel Prefill",
        f"- EGPT prefill: {avg('sd_egpt_prefill_ms'):.1f} ms",
        f"- VL prefill: {avg('sd_vl_prefill_ms'):.1f} ms",
        f"- Sequential would be: {avg('sd_egpt_prefill_ms') + avg('sd_vl_prefill_ms'):.1f} ms",
        f"- Parallel wall-clock: {avg('sd_parallel_prefill_ms'):.1f} ms",
        f"- Overlap achieved: {avg('sd_overlap_ms'):.1f} ms ({avg('sd_overlap_ratio'):.1%})",
        "",
        "## SD Statistics",
        f"- Accept rate: {avg('sd_accept_rate'):.1%}",
        f"- Tokens per iteration: {avg('sd_tokens_per_iter'):.2f}",
        f"- Avg drafted per sample: {avg('sd_drafted'):.1f}",
        f"- Avg accepted per sample: {avg('sd_accepted'):.1f}",
        "",
    ]
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {md_path}")

    return output_data


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="E2E Speculative Decoding Benchmark")
    parser.add_argument('--adapter_checkpoint', type=str, required=True,
                        help='Path to trained hidden adapter checkpoint')
    parser.add_argument('--dataset_dir', type=str,
                        default='./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s',
                        help='Path to test dataset')
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--gamma_decode', type=int, default=5)
    parser.add_argument('--warmup_samples', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = BenchmarkConfig(
        adapter_checkpoint=args.adapter_checkpoint,
        dataset_dir=args.dataset_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        gamma_decode=args.gamma_decode,
        warmup_samples=args.warmup_samples,
        output_dir=args.output_dir or "",
        device=args.device,
    )

    run_benchmark(config)


if __name__ == "__main__":
    main()
