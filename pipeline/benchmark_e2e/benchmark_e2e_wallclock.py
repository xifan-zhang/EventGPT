#!/usr/bin/env python3
"""
True End-to-End Wall-Clock Benchmark: All Methods & Baselines
=============================================================

Author: Alice Zhang
Date: 2026-02-07

Strictly from raw data. Both models loaded simultaneously in 4-bit (~8GB).
Speculative decoding runs two models in parallel.

Pipeline (per sample, per adapter config):
  1. VL Baseline: VL prefill + VL autoregressive decode (standalone)
  2. SD Run: Parallel prefill (EGPT + VL via threading + CUDA streams)
     → SD decode (adapter drafts → VL verifies)
  3. Repeat for all adapter configs

Adapter configs benchmarked:
  - VL baseline (no SD)
  - L1-only through L5-only (decode-only SD, full gamma)
  - B1-only, L5F-only (decode-only SD, gamma=1 real hidden states)
  - L1+L5F through L4+L5F (two-phase: prefill hiding + L5F gamma=1 SD)

Outputs: JSON results + matplotlib graphs + markdown report.

Usage:
  conda run -n egpt python benchmark_e2e_wallclock.py \
    --max_samples 50 --max_new_tokens 50
"""

import os
import sys
import json
import time
import gc
import threading
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass, asdict

if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ADAPTER_BASE = Path(__file__).resolve().parent / 'tasks'


# =========================================================================
# Vision Timing Hooks (for Video-LLaVA 3-stage)
# =========================================================================

class VisionTimingHooks:
    def __init__(self, model):
        self.model = model
        self.vision_time = 0.0
        self._start = None
        self.hooks = []

    def _pre_hook(self, module, input):
        torch.cuda.synchronize()
        self._start = time.time()

    def _post_hook(self, module, input, output):
        torch.cuda.synchronize()
        if self._start is not None:
            self.vision_time = time.time() - self._start

    def register(self):
        try:
            vt = self.model.get_vision_tower()
            if vt is not None:
                self.hooks.append(vt.register_forward_pre_hook(self._pre_hook))
                self.hooks.append(vt.register_forward_hook(self._post_hook))
                return True
        except Exception:
            pass
        return False

    def unregister(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset(self):
        self.vision_time = 0.0
        self._start = None


# =========================================================================
# 3-stage timing: EventGPT
# =========================================================================

def time_egpt_3stage(model, tokenizer, processor, sample, dataset_dir,
                     query, device, max_new_tokens, event_image_key='event_image'):
    """Measure EGPT vision/prefill/decode separately. Returns dict or None."""
    from model.EventChatModel import get_spatio_temporal_features
    from common.common import load_image, tokenizer_event_token
    from dataset.conversation import prepare_event_prompt
    from dataset.constants import EVENT_TOKEN_INDEX

    event_paths = sample.get(event_image_key, [])
    if not event_paths:
        return None

    img_path = os.path.join(dataset_dir, event_image_key, event_paths[0])
    try:
        img = load_image(img_path)
    except Exception:
        return None
    img_array = np.array(img)
    event_image_size = list(img_array.shape[:2])
    event = processor(img_array, return_tensors='pt')['pixel_values'][0]
    event = event.to(device, dtype=torch.bfloat16)

    prompt = prepare_event_prompt(query, 'eventgpt_v1')
    input_ids = tokenizer_event_token(
        prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    # Stage 1: Vision Encoder
    torch.cuda.synchronize()
    t1s = time.time()
    with torch.inference_mode():
        feature = model.visval_encode(event.unsqueeze(0))
    torch.cuda.synchronize()
    t1 = (time.time() - t1s) * 1000

    # Stage 2: LLM Prefill
    torch.cuda.synchronize()
    t2s = time.time()
    with torch.inference_mode():
        feature = model.get_model().feature_adaptor(feature).squeeze(0)
        event_features = get_spatio_temporal_features([feature]).unsqueeze(0)
        _, position_ids, attention_mask, _, inputs_embeds, _ = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, torch.ones_like(input_ids, dtype=torch.bool),
                None, None, event_tensors=None,
                event_image_sizes=event_image_size,
                event_features=event_features)
        if attention_mask is None:
            attention_mask = torch.ones(
                (1, inputs_embeds.shape[1]), dtype=torch.bool, device=device)
        if position_ids is None:
            position_ids = torch.arange(
                0, inputs_embeds.shape[1], dtype=torch.long, device=device
            ).unsqueeze(0)
        outputs = model.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=None, use_cache=True)
        logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
    torch.cuda.synchronize()
    t2 = (time.time() - t2s) * 1000

    # Stage 3: LLM Decode
    torch.cuda.synchronize()
    t3s = time.time()
    n_tokens = 0
    with torch.inference_mode():
        kv = outputs.past_key_values
        cur_pos = inputs_embeds.shape[1]
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        n_tokens += 1
        for _ in range(max_new_tokens - 1):
            cur_embed = model.get_model().embed_tokens(next_tok)
            new_attn = torch.ones(
                (1, cur_pos + 1), dtype=torch.bool, device=device)
            outputs = model.model(
                inputs_embeds=cur_embed, attention_mask=new_attn,
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=kv, use_cache=True)
            logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
            kv = outputs.past_key_values
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            n_tokens += 1
            cur_pos += 1
            if next_tok.item() == tokenizer.eos_token_id:
                break
    torch.cuda.synchronize()
    t3 = (time.time() - t3s) * 1000

    del kv, outputs, logits
    return {'vision_ms': t1, 'prefill_ms': t2, 'decode_ms': t3,
            'total_ms': t1 + t2 + t3, 'num_tokens': n_tokens}


# =========================================================================
# 3-stage timing: Video-LLaVA
# =========================================================================

def time_vl_3stage(model, processor, hooks, sample, dataset_dir,
                   query, device, max_new_tokens):
    """Measure VL vision/prefill/decode separately. Returns dict or None."""
    video_data = sample.get('video_data')
    if not video_data:
        return None
    video_path = os.path.join(dataset_dir, 'mp4', video_data + '.mp4')
    frames = load_video_frames(video_path)
    if frames is None:
        return None

    prompt = f'USER: <video>\n{query}\nASSISTANT:'
    inputs = processor(text=prompt, videos=frames, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs.get('attention_mask')
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(device)
    pvv = inputs.get('pixel_values_videos')
    if pvv is None:
        return None

    # Stage 1+2: Vision + Prefill (hooks separate vision time)
    hooks.reset()
    torch.cuda.synchronize()
    ps = time.time()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values_videos=pvv.to(device),
            past_key_values=None, use_cache=True)
        next_tok = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize()
    prefill_total = (time.time() - ps) * 1000
    t1 = hooks.vision_time * 1000
    t2 = prefill_total - t1

    # Stage 3: Decode
    torch.cuda.synchronize()
    t3s = time.time()
    n_tokens = 1
    with torch.inference_mode():
        past_kv = outputs.past_key_values
        cur_tok = next_tok
        attn_len = past_kv[0][0].shape[2] + 1
        attn_dtype = (attention_mask.dtype
                      if isinstance(attention_mask, torch.Tensor) else torch.long)
        for _ in range(max_new_tokens - 1):
            cur_attn = torch.ones(
                (1, attn_len), dtype=attn_dtype, device=device)
            outputs = model(
                input_ids=cur_tok, attention_mask=cur_attn,
                past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            cur_tok = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True)
            n_tokens += 1
            attn_len += 1
            if cur_tok.item() == processor.tokenizer.eos_token_id:
                break
    torch.cuda.synchronize()
    t3 = (time.time() - t3s) * 1000

    del past_kv, outputs
    return {'vision_ms': t1, 'prefill_ms': t2, 'decode_ms': t3,
            'total_ms': t1 + t2 + t3, 'num_tokens': n_tokens}


# =========================================================================
# Video frame loader
# =========================================================================

def load_video_frames(video_path, num_frames=8):
    import av
    try:
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
            if len(frames) >= num_frames:
                break
        container.close()
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                return None
        return frames
    except Exception:
        return None


# =========================================================================
# Model loading — both models simultaneously
# =========================================================================

def load_all_models(device='cuda'):
    """Load EventGPT + Video-LLaVA in 4-bit simultaneously."""
    from transformers import (
        AutoTokenizer, BitsAndBytesConfig,
        VideoLlavaForConditionalGeneration, VideoLlavaProcessor,
    )
    from model.EventChatModel import EventChatModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("\n[1/2] Loading EventGPT (4-bit)...")
    egpt_path = str(ROOT / "checkpoints" / "EventGPT-7b")
    egpt_model = EventChatModel.from_pretrained(
        egpt_path, torch_dtype=torch.bfloat16,
        device_map='auto', quantization_config=bnb_config)
    egpt_model.eval()
    egpt_tokenizer = AutoTokenizer.from_pretrained(egpt_path, use_fast=True)
    egpt_processor = egpt_model.get_visual_tower().event_processor
    mem_egpt = torch.cuda.memory_allocated() / 1024**2
    print(f"  GPU: {mem_egpt:.0f} MB")

    print("\n[2/2] Loading Video-LLaVA (4-bit)...")
    vl_model_id = "LanguageBind/Video-LLaVA-7B-hf"
    vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
        vl_model_id, torch_dtype=torch.float16,
        device_map='auto', quantization_config=bnb_config)
    vl_model.eval()
    vl_processor = VideoLlavaProcessor.from_pretrained(vl_model_id)
    mem_total = torch.cuda.memory_allocated() / 1024**2
    print(f"  GPU: {mem_total:.0f} MB (both models)")

    return (egpt_model, egpt_tokenizer, egpt_processor,
            vl_model, vl_processor)


# =========================================================================
# EventGPT prefill — returns hidden states + KV cache
# =========================================================================

def egpt_prefill(model, tokenizer, processor, sample, dataset_dir, query,
                 device='cuda', event_image_key='event_image'):
    """EventGPT vision + prefill. Returns dict or None."""
    from model.EventChatModel import get_spatio_temporal_features
    from common.common import load_image, tokenizer_event_token
    from dataset.conversation import prepare_event_prompt
    from dataset.constants import EVENT_TOKEN_INDEX

    event_paths = sample.get(event_image_key, [])
    if not event_paths:
        return None

    img_path = os.path.join(dataset_dir, event_image_key, event_paths[0])
    img = load_image(img_path)
    img_array = np.array(img)
    event_image_size = list(img_array.shape[:2])

    event = processor(img_array, return_tensors='pt')['pixel_values'][0]
    event = event.to(device, dtype=torch.bfloat16)

    prompt = prepare_event_prompt(query, 'eventgpt_v1')
    input_ids = tokenizer_event_token(
        prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    # Use current_stream().synchronize() so that when called from a
    # dedicated CUDA stream (parallel context), we only wait for that
    # stream, not all streams globally.
    torch.cuda.current_stream().synchronize()
    t_start = time.time()

    with torch.inference_mode():
        feature = model.visval_encode(event.unsqueeze(0))
        feature = model.get_model().feature_adaptor(feature)
        feature = feature.squeeze(0)
        event_features = get_spatio_temporal_features([feature])
        event_features = event_features.unsqueeze(0)

        _, position_ids, attention_mask, _, inputs_embeds, _ = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, torch.ones_like(input_ids, dtype=torch.bool),
                None, None, event_tensors=None,
                event_image_sizes=event_image_size,
                event_features=event_features,
            )
        if attention_mask is None:
            attention_mask = torch.ones(
                (1, inputs_embeds.shape[1]), dtype=torch.bool, device=device)
        if position_ids is None:
            position_ids = torch.arange(
                0, inputs_embeds.shape[1], dtype=torch.long, device=device
            ).unsqueeze(0)

        outputs = model.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=None,
            use_cache=True, output_hidden_states=True, return_dict=True)

        kv_cache = outputs.past_key_values
        hidden_states = outputs.last_hidden_state  # [1, seq, 4096]
        logits = model.lm_head(hidden_states[:, -1:, :])
        first_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    torch.cuda.current_stream().synchronize()
    prefill_ms = (time.time() - t_start) * 1000

    return {
        'kv_cache': kv_cache,
        'first_token': first_token,
        'hidden_states': hidden_states,
        'last_hidden': hidden_states[:, -1:, :],  # [1, 1, D] for h[0]
        'prefill_ms': prefill_ms,
        'prefill_length': inputs_embeds.shape[1],
        'attention_mask': attention_mask,
    }


# =========================================================================
# EventGPT autoregressive decode — collect per-step hidden states
# =========================================================================

def egpt_decode_collect_hidden(model, kv_cache, first_token, prefill_length,
                                prefill_last_hidden, max_new_tokens,
                                eos_token_id, device='cuda'):
    """
    EGPT autoregressive decode, collecting hidden state at each step.

    Combines prefill's last hidden state (h[0]) with decode hidden states
    (h[1..N-1]) to form the full sequence of decode hidden states that
    matches what the adapters were trained on.

    Returns:
        tokens: list of generated token ids (len = N)
        hidden_states: [1, N, 4096] tensor of per-step decode hidden states
        decode_ms: time in ms
    """
    torch.cuda.synchronize()
    t_start = time.time()

    generated = [first_token.item()]
    step_hidden = [prefill_last_hidden.detach()]  # h[0] from prefill
    cur_token = first_token
    cur_pos = prefill_length

    with torch.inference_mode():
        for _ in range(max_new_tokens - 1):
            cur_embed = model.get_model().embed_tokens(cur_token)
            new_mask = torch.ones(
                (1, cur_pos + 1), dtype=torch.bool, device=device)
            outputs = model.model(
                inputs_embeds=cur_embed,
                attention_mask=new_mask,
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=kv_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            kv_cache = outputs.past_key_values
            hidden = outputs.last_hidden_state  # [1, 1, D]
            step_hidden.append(hidden.detach())

            logits = model.lm_head(hidden)
            cur_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(cur_token.item())
            cur_pos += 1

            if cur_token.item() == eos_token_id:
                break

    torch.cuda.synchronize()
    decode_ms = (time.time() - t_start) * 1000

    # [1, N, D] — matches training data format
    all_hidden = torch.cat(step_hidden, dim=1)

    return generated, all_hidden, decode_ms


# =========================================================================
# Video-LLaVA prefill
# =========================================================================

def vl_prefill_fn(model, processor, sample, dataset_dir, query,
                  device='cuda'):
    """Video-LLaVA vision + prefill from raw MP4. Returns dict or None."""
    video_data = sample.get('video_data')
    if not video_data:
        return None

    video_path = os.path.join(dataset_dir, 'mp4', video_data + '.mp4')
    frames = load_video_frames(video_path)
    if frames is None:
        return None

    prompt = f'USER: <video>\n{query}\nASSISTANT:'
    inputs = processor(text=prompt, videos=frames, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs.get('attention_mask')
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(device)
    pixel_values_videos = inputs.get('pixel_values_videos')
    if pixel_values_videos is None:
        return None

    torch.cuda.current_stream().synchronize()
    t_start = time.time()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos.to(device),
            past_key_values=None, use_cache=True,
            output_hidden_states=True)
        logits = outputs.logits
        kv_cache = outputs.past_key_values
        first_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        # Extract last-layer hidden states (needed for B1 and L5F)
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None

    torch.cuda.current_stream().synchronize()
    prefill_ms = (time.time() - t_start) * 1000

    return {
        'kv_cache': kv_cache,
        'first_token': first_token,
        'hidden_states': hidden_states,
        'prefill_ms': prefill_ms,
        'prefill_length': input_ids.shape[1],
        'attention_mask': attention_mask,
    }


# =========================================================================
# VL baseline decode (autoregressive, no SD)
# =========================================================================

def vl_decode_baseline(model, kv_cache, first_token, attention_mask,
                       max_new_tokens, eos_token_id, device='cuda'):
    """Standard autoregressive VL decode. Returns (tokens, decode_ms)."""
    torch.cuda.synchronize()
    t_start = time.time()

    generated = [first_token.item()]
    cur_token = first_token
    attn_dtype = (attention_mask.dtype
                  if isinstance(attention_mask, torch.Tensor) else torch.long)
    cur_attn = torch.ones(
        (1, kv_cache[0][0].shape[2] + 1), dtype=attn_dtype, device=device)

    with torch.inference_mode():
        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=cur_token, attention_mask=cur_attn,
                past_key_values=kv_cache, use_cache=True)
            kv_cache = outputs.past_key_values
            cur_token = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(cur_token.item())
            cur_attn = torch.ones(
                (1, cur_attn.shape[1] + 1), dtype=attn_dtype, device=device)
            if cur_token.item() == eos_token_id:
                break

    torch.cuda.synchronize()
    decode_ms = (time.time() - t_start) * 1000
    return generated, decode_ms


# =========================================================================
# VL batched verification — single forward pass for all draft tokens
# =========================================================================

def vl_verify_batch(model, last_accepted_id, draft_token_ids, kv_cache, kv_len,
                    attn_dtype, device='cuda', output_hidden=False):
    """Single VL forward pass to verify all draft tokens at once.

    Feeds [last_accepted, d[0], d[1], ..., d[gamma-1]] in one forward pass.
    Compares VL's predictions at each position with draft tokens.

    Returns:
        n_accepted: number of accepted draft tokens
        accepted_tokens: list of accepted tokens + bonus/corrected token
        kv_cache: truncated KV cache (only accepted positions kept)
        new_kv_len: updated KV length
        hidden_list: list of [1,1,D] hidden states for accepted positions
                     (only if output_hidden=True)
    """
    # Build input: [last_accepted, d[0], d[1], ..., d[gamma-1]]
    input_ids = torch.tensor(
        [[last_accepted_id] + list(draft_token_ids)], device=device)
    seq_len = input_ids.shape[1]
    attn = torch.ones((1, kv_len + seq_len), dtype=attn_dtype, device=device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids, attention_mask=attn,
            past_key_values=kv_cache, use_cache=True,
            output_hidden_states=output_hidden)

    # logits[i] = VL's prediction for what follows input[i]
    # logits[0] should predict d[0], logits[1] should predict d[1], etc.
    logits = outputs.logits[0]  # [seq_len, vocab]
    vl_predictions = logits.argmax(dim=-1).tolist()  # [seq_len]

    # Find first mismatch
    n_accepted = 0
    for i, draft_id in enumerate(draft_token_ids):
        if vl_predictions[i] == draft_id:
            n_accepted += 1
        else:
            break

    # Accepted tokens + bonus/corrected token
    accepted_tokens = list(draft_token_ids[:n_accepted])
    bonus_or_correction = vl_predictions[n_accepted]  # VL's token at mismatch
    accepted_tokens.append(bonus_or_correction)

    # Truncate KV cache: keep kv_len_start + n_accepted + 1 entries
    # (+1 for the last_accepted token that was fed)
    new_kv_len = kv_len + n_accepted + 1
    full_kv = outputs.past_key_values
    if n_accepted < len(draft_token_ids):
        # Need to truncate — remove unaccepted draft positions
        truncated_kv = tuple(
            (k[:, :, :new_kv_len, :], v[:, :, :new_kv_len, :])
            for k, v in full_kv)
    else:
        # All accepted — keep full KV cache
        truncated_kv = full_kv
        new_kv_len = kv_len + seq_len

    # Extract hidden states for accepted positions (B1/L5F)
    hidden_list = []
    if output_hidden and outputs.hidden_states:
        hs = outputs.hidden_states[-1]  # [1, seq_len, D]
        # Position 0 = after last_accepted, position i = after d[i-1]
        # We want hidden states for positions 0..n_accepted (inclusive)
        for j in range(n_accepted + 1):
            hidden_list.append(hs[:, j:j+1, :].detach())

    return n_accepted, accepted_tokens, truncated_kv, new_kv_len, hidden_list


# =========================================================================
# Parallel prefill (threading + CUDA streams)
# =========================================================================

def parallel_prefill(egpt_model, egpt_tokenizer, egpt_processor,
                     vl_model, vl_processor,
                     sample, dataset_dir, query, device='cuda',
                     event_image_key='event_image'):
    """Run EGPT and VL prefills in parallel. Returns dict or None."""
    egpt_stream = torch.cuda.Stream(device=device)
    vl_stream = torch.cuda.Stream(device=device)

    egpt_box = [None]
    vl_box = [None]
    egpt_err = [None]
    vl_err = [None]

    def run_egpt():
        try:
            with torch.cuda.stream(egpt_stream):
                egpt_box[0] = egpt_prefill(
                    egpt_model, egpt_tokenizer, egpt_processor,
                    sample, dataset_dir, query, device,
                    event_image_key=event_image_key)
        except Exception as e:
            egpt_err[0] = e

    def run_vl():
        try:
            with torch.cuda.stream(vl_stream):
                vl_box[0] = vl_prefill_fn(
                    vl_model, vl_processor,
                    sample, dataset_dir, query, device)
        except Exception as e:
            vl_err[0] = e

    torch.cuda.synchronize()
    wall_start = time.time()

    t_egpt = threading.Thread(target=run_egpt)
    t_vl = threading.Thread(target=run_vl)
    t_egpt.start()
    t_vl.start()
    t_egpt.join()
    t_vl.join()

    egpt_stream.synchronize()
    vl_stream.synchronize()
    torch.cuda.synchronize()
    wall_ms = (time.time() - wall_start) * 1000

    if egpt_err[0]:
        raise egpt_err[0]
    if vl_err[0]:
        raise vl_err[0]

    egpt_result = egpt_box[0]
    vl_result = vl_box[0]
    if egpt_result is None or vl_result is None:
        return None

    egpt_ms = egpt_result['prefill_ms']
    vl_ms = vl_result['prefill_ms']
    sequential_ms = egpt_ms + vl_ms
    overlap_ms = max(0, sequential_ms - wall_ms)

    return {
        'egpt': egpt_result,
        'vl': vl_result,
        'wall_ms': wall_ms,
        'egpt_ms': egpt_ms,
        'vl_ms': vl_ms,
        'sequential_ms': sequential_ms,
        'overlap_ms': overlap_ms,
        'overlap_ratio': overlap_ms / sequential_ms if sequential_ms > 0 else 0,
    }


# =========================================================================
# Parallel prefill + EGPT decode (for L1+VL through L4+VL)
# =========================================================================

def sequential_egpt_vl_prefill(
        egpt_model, egpt_tokenizer, egpt_processor,
        vl_model, vl_processor,
        sample, dataset_dir, query, gamma,
        max_new_tokens, device='cuda',
        event_image_key='event_image'):
    """EGPT prefill+decode then VL prefill — sequential, no GPU contention.

    Records per-token timestamps during EGPT decode so we can precisely
    determine which tokens are "free" (generated before VL prefill would
    finish on a separate accelerator).

    Pipeline (assuming dual-GPU / true overlap):
      t=0:               EGPT prefill starts, VL prefill starts (parallel)
      t=egpt_prefill:    EGPT prefill done, EGPT decode starts
      t=egpt_prefill+k*dt: k-th EGPT decode token ready
      t=vl_prefill:      VL prefill done — EGPT stops, adapter drafts
      gamma_prefill = tokens generated by EGPT before VL prefill finishes

    Returns dict with per-token timestamps and computed gamma_prefill.
    """
    # Phase 1: EGPT prefill (alone, accurate timing)
    egpt_result = egpt_prefill(
        egpt_model, egpt_tokenizer, egpt_processor,
        sample, dataset_dir, query, device,
        event_image_key=event_image_key)
    if egpt_result is None:
        return None

    # Phase 2: EGPT decode — record per-token timestamps
    torch.cuda.synchronize()
    t_dec_start = time.time()

    generated = [egpt_result['first_token'].item()]
    step_hidden = [egpt_result['last_hidden'].detach()]
    # token_timestamps[i] = cumulative ms from decode start when token i+1
    # (0-indexed decode step) finished generating
    token_timestamps_ms = []
    cur_token = egpt_result['first_token']
    cur_pos = egpt_result['prefill_length']
    kv = egpt_result['kv_cache']
    eos = egpt_tokenizer.eos_token_id

    with torch.inference_mode():
        for _ in range(max_new_tokens - 1):
            cur_embed = egpt_model.get_model().embed_tokens(cur_token)
            new_mask = torch.ones(
                (1, cur_pos + 1), dtype=torch.bool, device=device)
            outputs = egpt_model.model(
                inputs_embeds=cur_embed,
                attention_mask=new_mask,
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=kv,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True)
            kv = outputs.past_key_values
            hidden = outputs.last_hidden_state
            step_hidden.append(hidden.detach())
            logits = egpt_model.lm_head(hidden)
            cur_token = torch.argmax(
                logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(cur_token.item())
            cur_pos += 1

            # Record actual timestamp for this token
            torch.cuda.synchronize()
            token_timestamps_ms.append(
                (time.time() - t_dec_start) * 1000)

            if cur_token.item() == eos:
                break

    torch.cuda.synchronize()
    egpt_decode_ms = (time.time() - t_dec_start) * 1000
    all_hidden = torch.cat(step_hidden, dim=1)
    egpt_decode_result = (generated, all_hidden, egpt_decode_ms)

    # Free EGPT KV cache before VL prefill
    del kv, egpt_result['kv_cache']
    torch.cuda.empty_cache()

    # Phase 3: VL prefill (alone, accurate timing)
    vl_result = vl_prefill_fn(
        vl_model, vl_processor, sample, dataset_dir, query, device)
    if vl_result is None:
        return None

    # Phase 4: Compute gamma_prefill from actual timestamps
    egpt_prefill_ms = egpt_result['prefill_ms']
    vl_prefill_ms = vl_result['prefill_ms']
    gap_ms = max(0, vl_prefill_ms - egpt_prefill_ms)

    # Count tokens whose cumulative decode time fits within the gap
    gamma_prefill_decode = 0
    last_free_decode_ms = 0.0
    for i, t in enumerate(token_timestamps_ms):
        if t <= gap_ms:
            gamma_prefill_decode = i + 1
            last_free_decode_ms = t
        else:
            break

    # +1 for h[0] from prefill (always free — generated during prefill)
    gamma_prefill_hidden = gamma_prefill_decode + 1
    gamma_prefill_hidden = min(gamma_prefill_hidden, all_hidden.shape[1])

    # wall_ms = when the last free token was generated
    # = egpt_prefill_ms + cumulative decode time for gamma tokens
    wall_ms = egpt_prefill_ms + last_free_decode_ms
    # But VL prefill is the bottleneck, so wall >= vl_prefill
    wall_ms = max(wall_ms, vl_prefill_ms)

    n_decode = len(token_timestamps_ms)
    egpt_per_token_ms = (egpt_decode_ms / n_decode if n_decode > 0 else 10.0)

    return {
        'egpt': egpt_result,
        'vl': vl_result,
        'egpt_decode': egpt_decode_result,
        'token_timestamps_ms': token_timestamps_ms,
        'wall_ms': wall_ms,
        'egpt_ms': egpt_prefill_ms,
        'vl_ms': vl_prefill_ms,
        'gamma_prefill': gamma_prefill_hidden,
        'egpt_per_token_ms': egpt_per_token_ms,
        'egpt_decode_ms': egpt_decode_ms,
        'gap_ms': gap_ms,
        'overlap_ms': egpt_prefill_ms,
        'overlap_ratio': egpt_prefill_ms / (egpt_prefill_ms + vl_prefill_ms)
            if (egpt_prefill_ms + vl_prefill_ms) > 0 else 0,
    }


# =========================================================================
# SD decode loop
# =========================================================================

def run_sd_decode(vl_model, adapter, vl_lm_head_weight,
                  vl_kv_cache, vl_first_token,
                  attn_dtype, kv_len_start,
                  gamma=5, max_new_tokens=50, eos_token_id=2,
                  is_fused=False, is_vlm_only=False,
                  egpt_decode_hidden=None,
                  vl_prefill_last_hidden=None,
                  device='cuda'):
    """
    SD decode: adapter drafts tokens, VL verifies.

    Hidden state sources (matching training data extraction):
      - L1-L4, L5: egpt_decode_hidden  (pre-generated EGPT decode hidden states)
      - B1 (vlm_only): VL hidden states accumulated during verification
      - L5F (fused): egpt_decode_hidden + VL hidden states from verification

    For B1 and L5F, VL verification uses output_hidden_states=True so we
    collect VL hidden states to feed back to the adapter.
    """
    W = vl_lm_head_weight.to(device)
    need_vl_hidden = is_vlm_only or is_fused

    # Determine adapter dtype for casting inputs
    adapter_dtype = next(adapter.parameters()).dtype

    torch.cuda.synchronize()
    t_start = time.time()

    generated = [vl_first_token]
    kv_cache = vl_kv_cache
    kv_len = kv_len_start
    total_drafted = 0
    total_accepted = 0
    iterations = 0

    # Accumulate VL hidden states for B1/L5F
    vl_hidden_buffer = []
    if need_vl_hidden and vl_prefill_last_hidden is not None:
        vl_hidden_buffer.append(vl_prefill_last_hidden)  # h_vl[0]

    with torch.inference_mode():
        while len(generated) < max_new_tokens:
            iterations += 1

            # --- Draft phase ---
            # L1-L5: one-shot draft using pre-computed EGPT hidden states (full gamma)
            # B1/L5F: gamma=1 draft using real VL hidden states only
            draft_start = len(generated) - 1

            if is_vlm_only or is_fused:
                # B1/L5F: gamma=1 draft using real VL hidden states only.
                # No autoregressive rollout — avoids training/inference
                # mismatch. No SD speedup alone; useful for prefill hiding
                # in two-phase pipeline.
                if not vl_hidden_buffer:
                    # No VL hidden states yet — fallback to VL AR decode
                    cur_t = torch.tensor(
                        [[generated[-1]]], device=device)
                    attn = torch.ones(
                        (1, kv_len + 1), dtype=attn_dtype, device=device)
                    outputs = vl_model(
                        input_ids=cur_t, attention_mask=attn,
                        past_key_values=kv_cache, use_cache=True,
                        output_hidden_states=need_vl_hidden)
                    kv_cache = outputs.past_key_values
                    kv_len += 1
                    next_id = torch.argmax(
                        outputs.logits[:, -1, :], dim=-1).item()
                    generated.append(next_id)
                    if need_vl_hidden and outputs.hidden_states:
                        vl_hidden_buffer.append(
                            outputs.hidden_states[-1].detach())
                    if next_id == eos_token_id:
                        break
                    continue

                seed_h = vl_hidden_buffer[-1].to(adapter_dtype)  # [1,1,D]

                if is_vlm_only:
                    # B1: VL hidden → predict next VL hidden
                    aligned = adapter(seed_h)[:, -1:, :]
                else:
                    # L5F: (EGPT hidden, VL hidden) → predict next
                    egpt_n = (egpt_decode_hidden.shape[1]
                              if egpt_decode_hidden is not None else 0)
                    pos = draft_start
                    if pos >= egpt_n:
                        # No more EGPT hidden — fallback to VL AR
                        cur_t = torch.tensor(
                            [[generated[-1]]], device=device)
                        attn = torch.ones(
                            (1, kv_len + 1), dtype=attn_dtype, device=device)
                        outputs = vl_model(
                            input_ids=cur_t, attention_mask=attn,
                            past_key_values=kv_cache, use_cache=True,
                            output_hidden_states=need_vl_hidden)
                        kv_cache = outputs.past_key_values
                        kv_len += 1
                        next_id = torch.argmax(
                            outputs.logits[:, -1, :], dim=-1).item()
                        generated.append(next_id)
                        if need_vl_hidden and outputs.hidden_states:
                            vl_hidden_buffer.append(
                                outputs.hidden_states[-1].detach())
                        if next_id == eos_token_id:
                            break
                        continue
                    egpt_h = egpt_decode_hidden[:, pos:pos+1, :].to(
                        adapter_dtype)
                    aligned = adapter(egpt_h, seed_h)[:, -1:, :]

                n_draft = 1
            else:
                # L1-L5: one-shot draft from pre-computed EGPT hidden
                egpt_n = (egpt_decode_hidden.shape[1]
                          if egpt_decode_hidden is not None else 0)
                draft_end = min(draft_start + gamma, egpt_n)
                n_draft = draft_end - draft_start
                if n_draft <= 0:
                    # Fallback to autoregressive VL decode
                    cur_t = torch.tensor(
                        [[generated[-1]]], device=device)
                    attn = torch.ones(
                        (1, kv_len + 1), dtype=attn_dtype, device=device)
                    outputs = vl_model(
                        input_ids=cur_t, attention_mask=attn,
                        past_key_values=kv_cache, use_cache=True)
                    kv_cache = outputs.past_key_values
                    kv_len += 1
                    next_id = torch.argmax(
                        outputs.logits[:, -1, :], dim=-1).item()
                    generated.append(next_id)
                    if next_id == eos_token_id:
                        break
                    continue

                egpt_chunk = egpt_decode_hidden[
                    :, draft_start:draft_end, :].to(adapter_dtype)
                aligned = adapter(egpt_chunk)

            draft_logits = torch.matmul(aligned.float(), W.T)
            draft_tokens = draft_logits.argmax(dim=-1)[0].tolist()
            total_drafted += n_draft

            # --- Verify phase (single batched VL forward) ---
            accepted, new_tokens, kv_cache, kv_len, hidden_list = \
                vl_verify_batch(
                    vl_model, generated[-1], draft_tokens,
                    kv_cache, kv_len, attn_dtype, device,
                    output_hidden=need_vl_hidden)

            generated.extend(new_tokens)
            total_accepted += accepted

            if need_vl_hidden:
                vl_hidden_buffer.extend(hidden_list)

            if generated[-1] == eos_token_id:
                break

    torch.cuda.synchronize()
    decode_ms = (time.time() - t_start) * 1000

    stats = {
        'total_drafted': total_drafted,
        'total_accepted': total_accepted,
        'iterations': iterations,
        'accept_rate': (total_accepted / total_drafted
                        if total_drafted > 0 else 0),
        'tokens_per_iter': (len(generated) / iterations
                            if iterations > 0 else 0),
    }
    return generated, decode_ms, stats


# =========================================================================
# Find all adapter checkpoints
# =========================================================================

def find_adapter_checkpoints(base_dir):
    """Find best_model.pt for each adapter level.

    Uses the specific checkpoints from TRAINING_RESULTS_20260207.md.
    Falls back to auto-discovery (latest timestamped dir) if not found.
    """
    # Documented checkpoints from training results
    KNOWN_CHECKPOINTS = {
        'L1': 'L1/L1_20260206_095906/best_model.pt',
        'L2': 'L2/L2_20260206_181048/best_model.pt',
        'L3': 'L3/L3_20260206_183919/best_model.pt',
        'L4': 'L4/L4_20260206_192256/best_model.pt',
        'L5': 'L5/L5_20260206_202939/best_model.pt',
        'B1': 'B1/B1_20260206_213805/best_model.pt',
        'L5F': 'L5F/L5F_20260206_224537/best_model.pt',
    }

    configs = {}
    for level in ['L1', 'L2', 'L3', 'L4', 'L5', 'B1', 'L5F']:
        # Try documented path first
        known = base_dir / KNOWN_CHECKPOINTS.get(level, '')
        if known.exists():
            configs[level] = str(known)
            continue

        # Fallback: find latest best_model.pt under level dir
        level_dir = base_dir / level
        if not level_dir.exists():
            continue
        candidates = sorted(level_dir.rglob('best_model.pt'),
                            key=lambda p: p.parent.name, reverse=True)
        if candidates:
            configs[level] = str(candidates[0])

    # Flat layout: adapter_dir itself or its subdirs contain best_model.pt
    # with names like L4_20260209_011428/best_model.pt
    if not configs:
        # Check if best_model.pt is directly in base_dir
        direct = base_dir / 'best_model.pt'
        if direct.exists():
            # Infer level from dir name (e.g. L4_20260209_011428 → L4)
            level = base_dir.name.split('_')[0]
            if level in ['L1', 'L2', 'L3', 'L4', 'L5', 'B1', 'L5F']:
                configs[level] = str(direct)
        else:
            # Check subdirs named like L4_timestamp/best_model.pt
            for subdir in sorted(base_dir.iterdir(), reverse=True):
                if not subdir.is_dir():
                    continue
                ckpt = subdir / 'best_model.pt'
                if ckpt.exists():
                    level = subdir.name.split('_')[0]
                    if level in ['L1', 'L2', 'L3', 'L4', 'L5', 'B1', 'L5F']:
                        configs.setdefault(level, str(ckpt))

    return configs


# =========================================================================
# Graph generation
# =========================================================================

def generate_graphs(all_results, output_dir, timestamp,
                    stage_timing=None, example=None):
    """Generate matplotlib graphs from benchmark results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available, skipping graphs")
        return

    configs = list(all_results.keys())
    if not configs:
        return

    # --- 1. Speedup bar chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    speedups = [all_results[c]['speedup_mean'] for c in configs]
    colors = []
    for c in configs:
        if 'baseline' in c:
            colors.append('#888888')
        elif '+' in c:
            colors.append('#2196F3')
        else:
            colors.append('#4CAF50')

    bars = ax.bar(range(len(configs)), speedups, color=colors, edgecolor='black',
                  linewidth=0.5)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Speedup vs VL Baseline', fontsize=11)
    ax.set_title('E2E Speculative Decoding Speedup — All Configs', fontsize=13)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_ylim(0, max(speedups) * 1.15)

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    plt.tight_layout()
    path1 = os.path.join(output_dir, f'speedup_comparison_{timestamp}.png')
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path1}")

    # --- 2. Timing breakdown (stacked bar: Vision + Prefill + Decode) ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get vision encoder times from stage_timing
    vl_vision_ms = 0
    egpt_vision_ms = 0
    if stage_timing:
        vl_vision_ms = stage_timing.get('videollava', {}).get('vision_ms', 0)
        egpt_vision_ms = stage_timing.get('eventgpt', {}).get('vision_ms', 0)

    vision_times = []
    prefill_times = []
    decode_times = []
    for c in configs:
        if 'baseline' in c:
            vision_times.append(vl_vision_ms)
        else:
            # +VL configs: EGPT vision is hidden, use VL vision (= 0 for VL)
            vision_times.append(0)
        prefill_times.append(all_results[c].get('prefill_ms_mean', 0))
        decode_times.append(all_results[c].get('decode_ms_mean', 0))

    x = range(len(configs))
    ax.bar(x, vision_times, label='Vision Encoder', color='#4CAF50',
           edgecolor='black', linewidth=0.5)
    ax.bar(x, prefill_times, bottom=vision_times, label='LLM Prefill',
           color='#FF9800', edgecolor='black', linewidth=0.5)
    bottoms = [v + p for v, p in zip(vision_times, prefill_times)]
    ax.bar(x, decode_times, bottom=bottoms, label='LLM Decode',
           color='#2196F3', edgecolor='black', linewidth=0.5)

    # Add speedup labels on top of each bar
    for i, c in enumerate(configs):
        total = vision_times[i] + prefill_times[i] + decode_times[i]
        spd = all_results[c]['speedup_mean']
        ax.text(i, total + 5, f'{spd:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title('Timing Breakdown: Vision + Prefill + Decode', fontsize=13)
    ax.legend(fontsize=10)
    ymax = max(v + p + d for v, p, d in
               zip(vision_times, prefill_times, decode_times))
    ax.set_ylim(0, ymax * 1.12)
    plt.tight_layout()
    path2 = os.path.join(output_dir, f'timing_breakdown_{timestamp}.png')
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path2}")

    # --- 3. Accept rate comparison ---
    sd_configs = [c for c in configs if 'baseline' not in c]
    if sd_configs:
        fig, ax = plt.subplots(figsize=(10, 5))
        accept_rates = [all_results[c].get('accept_rate_mean', 0) * 100
                        for c in sd_configs]
        bars = ax.bar(range(len(sd_configs)), accept_rates, color='#9C27B0',
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(sd_configs)))
        ax.set_xticklabels(sd_configs, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Accept Rate (%)', fontsize=11)
        ax.set_title('SD Token Accept Rate by Config', fontsize=13)
        ax.set_ylim(0, 100)

        for bar, val, c in zip(bars, accept_rates, sd_configs):
            spd = all_results[c]['speedup_mean']
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{val:.1f}% ({spd:.2f}x)', ha='center', va='bottom',
                    fontsize=9)

        plt.tight_layout()
        path3 = os.path.join(output_dir,
                             f'accept_rate_comparison_{timestamp}.png')
        fig.savefig(path3, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path3}")

    # --- 4. Prefill hiding timeline + example text ---
    if example and stage_timing:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                 gridspec_kw={'height_ratios': [1.2, 1]})

        # --- 4a. Timeline: prefill hiding phase ---
        ax = axes[0]
        egpt_v = stage_timing['eventgpt']['vision_ms']
        egpt_p = stage_timing['eventgpt']['prefill_ms']
        egpt_d = example.get('egpt_decode_ms', 0)
        vl_p = example.get('vl_prefill_ms', 0)
        gap = example.get('gap_ms', 0)
        n_free = int(example.get('num_free_tok', 0))

        y_egpt = 1.5
        y_vl = 0.5
        h = 0.6

        # EGPT timeline: vision → prefill → decode
        ax.barh(y_egpt, egpt_v, left=0, height=h,
                color='#4CAF50', edgecolor='black', linewidth=0.5)
        ax.barh(y_egpt, egpt_p, left=egpt_v, height=h,
                color='#FF9800', edgecolor='black', linewidth=0.5)
        ax.barh(y_egpt, egpt_d, left=egpt_v + egpt_p, height=h,
                color='#2196F3', edgecolor='black', linewidth=0.5)
        # Labels inside bars
        ax.text(egpt_v / 2, y_egpt, f'Vision\n{egpt_v:.0f}ms',
                ha='center', va='center', fontsize=7, fontweight='bold')
        ax.text(egpt_v + egpt_p / 2, y_egpt, f'Prefill\n{egpt_p:.0f}ms',
                ha='center', va='center', fontsize=7, fontweight='bold')
        ax.text(egpt_v + egpt_p + egpt_d / 2, y_egpt,
                f'Decode ({n_free} free tok)\n{egpt_d:.0f}ms',
                ha='center', va='center', fontsize=7, fontweight='bold')

        # VL timeline: prefill (starts at 0, parallel with EGPT)
        ax.barh(y_vl, vl_p, left=0, height=h,
                color='#FF9800', edgecolor='black', linewidth=0.5,
                alpha=0.7, hatch='//')
        ax.text(vl_p / 2, y_vl, f'VL Prefill\n{vl_p:.0f}ms',
                ha='center', va='center', fontsize=7, fontweight='bold')

        # Gap annotation
        egpt_prefill_end = egpt_v + egpt_p
        if gap > 0:
            ax.annotate('', xy=(vl_p, y_vl + h / 2 + 0.05),
                        xytext=(egpt_prefill_end, y_vl + h / 2 + 0.05),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
            ax.text((egpt_prefill_end + vl_p) / 2, y_vl + h / 2 + 0.15,
                    f'gap={gap:.0f}ms\n→ {n_free} free tokens',
                    ha='center', va='bottom', fontsize=8, color='red',
                    fontweight='bold')

        ax.set_yticks([y_vl, y_egpt])
        ax.set_yticklabels(['Video-LLaVA\n(verifier)', 'EventGPT\n(drafter)'],
                           fontsize=9)
        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_title('Prefill Hiding: EGPT decode tokens are FREE '
                     '(hidden behind VL prefill)', fontsize=12)
        ax.set_xlim(0, max(egpt_v + egpt_p + egpt_d, vl_p) * 1.05)
        ax.set_ylim(0, 2.3)

        # Legend
        legend_patches = [
            mpatches.Patch(color='#4CAF50', label='Vision Encoder'),
            mpatches.Patch(color='#FF9800', label='LLM Prefill'),
            mpatches.Patch(color='#2196F3', label='LLM Decode (free tokens)'),
        ]
        ax.legend(handles=legend_patches, fontsize=8, loc='upper right')

        # --- 4b. Example text with accepted/rejected tokens ---
        ax2 = axes[1]
        ax2.axis('off')

        bl_text = example.get('bl_text', '')[:120]
        sd_text = example.get('sd_text', '')[:120]
        n_acc = example.get('accepted', 0)
        n_draft = example.get('drafted', 0)
        cfg_name = example.get('config_name', '')
        spd = example.get('speedup', 1.0)

        # Find common prefix between bl and sd text (accepted tokens)
        bl_words = bl_text.split()
        sd_words = sd_text.split()
        common_len = 0
        for a, b in zip(bl_words, sd_words):
            if a == b:
                common_len += 1
            else:
                break
        accepted_part = ' '.join(sd_words[:common_len])
        rest_part = ' '.join(sd_words[common_len:])

        y = 0.85
        ax2.text(0.02, y, f'{cfg_name} ({spd:.2f}x speedup, '
                 f'{n_acc}/{n_draft} tokens accepted):',
                 fontsize=10, fontweight='bold', transform=ax2.transAxes,
                 va='top')
        y -= 0.18
        # Accepted tokens in green, rest in blue
        ax2.text(0.02, y, 'SD output: ', fontsize=9,
                 transform=ax2.transAxes, va='top')
        ax2.text(0.12, y, accepted_part + ' ',
                 fontsize=9, color='green', fontweight='bold',
                 transform=ax2.transAxes, va='top')
        # Calculate approximate x position for rest
        approx_x = 0.12 + len(accepted_part) * 0.006
        if approx_x < 0.95:
            ax2.text(approx_x, y, rest_part,
                     fontsize=9, color='#666666',
                     transform=ax2.transAxes, va='top')

        y -= 0.18
        ax2.text(0.02, y, 'VL baseline: ', fontsize=9,
                 transform=ax2.transAxes, va='top')
        ax2.text(0.12, y, bl_text,
                 fontsize=9, color='black',
                 transform=ax2.transAxes, va='top')

        y -= 0.22
        ax2.text(0.02, y,
                 f'Green = accepted draft tokens (free, no VL compute needed)  |  '
                 f'Gray = VL autoregressive decode after rejection',
                 fontsize=8, color='#444444', style='italic',
                 transform=ax2.transAxes, va='top')

        plt.tight_layout()
        path4 = os.path.join(output_dir,
                             f'prefill_hiding_example_{timestamp}.png')
        fig.savefig(path4, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path4}")


# =========================================================================
# Main benchmark
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="True E2E Wall-Clock SD Benchmark (all methods, raw data)")
    parser.add_argument('--dataset_dir', type=str,
                        default='/mnt/hdd/data/my_egpt_dsec_test/my_egpt_dsec_seq_1s')
    parser.add_argument('--adapter_dir', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=50)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--gamma', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--questions_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--configs', type=str, default=None,
                        help='Comma-separated config names to run (default: all)')
    parser.add_argument('--event_image_key', type=str, default='event_image',
                        help='JSON key + subfolder for event images (default: event_image)')
    args = parser.parse_args()

    device = 'cuda'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent / 'tasks')
    os.makedirs(args.output_dir, exist_ok=True)

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else ADAPTER_BASE

    # Load dataset
    dataset_json = os.path.join(
        args.dataset_dir, 'EventGPT_Instruction_Subset.json')
    with open(dataset_json) as f:
        dataset = json.load(f)
    if args.max_samples:
        dataset = dataset[:args.max_samples + args.warmup]

    # Load questions (top 10 from dataset)
    if args.questions_file:
        with open(args.questions_file) as f:
            questions = [q['question'] for q in json.load(f)]
    else:
        q_path = (Path(ROOT) / 'feasible' / 'token_alignment'
                  / 'top50_questions.json')
        if q_path.exists():
            with open(q_path) as f:
                questions = [q['question'] for q in json.load(f)[:10]]
        else:
            questions = ["What are the key elements in this scene?"]

    total_runs = len(dataset) * len(questions)

    print("=" * 70)
    print("TRUE E2E WALL-CLOCK BENCHMARK — ALL METHODS & BASELINES")
    print(f"Date: {timestamp}")
    print(f"Samples: {len(dataset)} ({args.warmup} warmup + "
          f"{len(dataset) - args.warmup} measured)")
    print(f"Questions: {len(questions)}, Total runs: {total_runs}")
    print(f"Max tokens: {args.max_new_tokens}, gamma: {args.gamma}")
    print(f"Adapter dir: {adapter_dir}")
    print("=" * 70)

    # ==================================================================
    # Load both models simultaneously
    # ==================================================================
    (egpt_model, egpt_tokenizer, egpt_processor,
     vl_model, vl_processor) = load_all_models(device)

    # Get VL LM head
    vl_lm_head_weight = vl_model.language_model.lm_head.weight.data.float()
    eos_token_id = vl_processor.tokenizer.eos_token_id
    print(f"  VL LM head: {list(vl_lm_head_weight.shape)}")

    # ==================================================================
    # 3-stage timing for both models (from raw data)
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("3-STAGE TIMING (from raw data, both models loaded)")
    print(f"{'=' * 70}")

    vl_hooks = VisionTimingHooks(vl_model)
    vl_hooks.register()

    egpt_timings = []
    vl_timings = []
    timing_query = questions[0]
    n_timing = min(len(dataset), args.warmup + 20)  # warmup + 20 measured

    for i in range(n_timing):
        sample = dataset[i]
        et = time_egpt_3stage(
            egpt_model, egpt_tokenizer, egpt_processor,
            sample, args.dataset_dir, timing_query, device,
            args.max_new_tokens, event_image_key=args.event_image_key)
        if et is not None:
            egpt_timings.append(et)
        torch.cuda.empty_cache()

        vt = time_vl_3stage(
            vl_model, vl_processor, vl_hooks,
            sample, args.dataset_dir, timing_query, device,
            args.max_new_tokens)
        if vt is not None:
            vl_timings.append(vt)
        torch.cuda.empty_cache()

    vl_hooks.unregister()

    # Aggregate (skip warmup)
    w = args.warmup

    def timing_stats(results, key):
        vals = ([r[key] for r in results[w:]] if len(results) > w
                else [r[key] for r in results])
        return np.mean(vals) if vals else 0, np.std(vals) if vals else 0

    egpt_vision, _ = timing_stats(egpt_timings, 'vision_ms')
    egpt_prefill_t, _ = timing_stats(egpt_timings, 'prefill_ms')
    egpt_decode_t, _ = timing_stats(egpt_timings, 'decode_ms')
    egpt_total_t, _ = timing_stats(egpt_timings, 'total_ms')
    egpt_tokens_t, _ = timing_stats(egpt_timings, 'num_tokens')
    egpt_per_token = egpt_decode_t / max(egpt_tokens_t, 1)

    vl_vision, _ = timing_stats(vl_timings, 'vision_ms')
    vl_prefill_t, _ = timing_stats(vl_timings, 'prefill_ms')
    vl_decode_t, _ = timing_stats(vl_timings, 'decode_ms')
    vl_total_t, _ = timing_stats(vl_timings, 'total_ms')
    vl_tokens_t, _ = timing_stats(vl_timings, 'num_tokens')
    vl_per_token = vl_decode_t / max(vl_tokens_t, 1)

    stage_timing = {
        'eventgpt': {
            'vision_ms': egpt_vision, 'prefill_ms': egpt_prefill_t,
            'decode_ms': egpt_decode_t, 'total_ms': egpt_total_t,
            'per_token_ms': egpt_per_token, 'avg_tokens': egpt_tokens_t,
        },
        'videollava': {
            'vision_ms': vl_vision, 'prefill_ms': vl_prefill_t,
            'decode_ms': vl_decode_t, 'total_ms': vl_total_t,
            'per_token_ms': vl_per_token, 'avg_tokens': vl_tokens_t,
        },
    }

    n_e = max(len(egpt_timings) - w, len(egpt_timings))
    n_v = max(len(vl_timings) - w, len(vl_timings))
    print(f"\n  EventGPT ({n_e} samples):")
    print(f"    Vision:   {egpt_vision:>8.1f} ms")
    print(f"    Prefill:  {egpt_prefill_t:>8.1f} ms")
    print(f"    Decode:   {egpt_decode_t:>8.1f} ms  "
          f"({egpt_per_token:.1f} ms/token)")
    print(f"    Total:    {egpt_total_t:>8.1f} ms")

    print(f"\n  Video-LLaVA ({n_v} samples):")
    print(f"    Vision:   {vl_vision:>8.1f} ms")
    print(f"    Prefill:  {vl_prefill_t:>8.1f} ms")
    print(f"    Decode:   {vl_decode_t:>8.1f} ms  "
          f"({vl_per_token:.1f} ms/token)")
    print(f"    Total:    {vl_total_t:>8.1f} ms")

    prefill_speedup = (vl_vision + vl_prefill_t) / max(
        egpt_vision + egpt_prefill_t, 0.01)
    print(f"\n  Prefill speedup (VL/EGPT): {prefill_speedup:.2f}x")

    # Find all adapter checkpoints
    adapter_ckpts = find_adapter_checkpoints(adapter_dir)
    print(f"\n  Adapters found: {list(adapter_ckpts.keys())}")

    # Load all adapters upfront
    from feasible.feature_alignment.hidden_adapter import load_any_adapter

    adapters = {}
    for level, ckpt_path in sorted(adapter_ckpts.items()):
        model, checkpoint = load_any_adapter(ckpt_path, device)
        model = model.to(device)
        model.eval()
        adapter_class = type(model).__name__
        is_fused = adapter_class == 'FusedEAGLEAdapter'
        is_vlm_only = ('B1' in level or
                       checkpoint.get('adapter_type', '').startswith('B1'))
        adapters[level] = {
            'model': model,
            'is_fused': is_fused,
            'is_vlm_only': is_vlm_only,
            'class': adapter_class,
        }
        print(f"  {level}: {adapter_class} (fused={is_fused}, "
              f"vlm_only={is_vlm_only})")

    # Build config list
    # Decode-only with full gamma: L1-L5 (one-shot from EGPT hidden states)
    # Decode-only with gamma=1: B1, L5F (real VL hidden states only)
    # Two-phase (gamma=1 L5F decode): L1+L5F through L4+L5F
    sd_configs = []

    # L1-L5: one-shot decode from pre-computed EGPT hidden states (full gamma)
    for level in ['L1', 'L2', 'L3', 'L4', 'L5']:
        if level in adapters:
            sd_configs.append({
                'name': f'{level}-only',
                'decode_adapter': level,
                'prefill_adapter': None,
            })

    # B1, L5F: gamma=1 decode (real VL hidden states only)
    for level in ['B1', 'L5F']:
        if level in adapters:
            sd_configs.append({
                'name': f'{level}-only',
                'decode_adapter': level,
                'prefill_adapter': None,
            })

    # Two-phase: parallel prefill + L5F decode at gamma=1
    if 'L5F' in adapters:
        for pl in ['L1', 'L2', 'L3', 'L4']:
            if pl in adapters:
                sd_configs.append({
                    'name': f'{pl}+L5F',
                    'decode_adapter': 'L5F',
                    'prefill_adapter': pl,
                })

    # L1+VL through L4+VL: adapter drafts from EGPT hidden states, VL verifies
    for level in ['L1', 'L2', 'L3', 'L4']:
        if level in adapters:
            sd_configs.append({
                'name': f'{level}+VL',
                'decode_adapter': level,
                'prefill_adapter': None,
            })

    # Filter configs if --configs specified
    if args.configs:
        wanted = set(c.strip() for c in args.configs.split(','))
        sd_configs = [c for c in sd_configs if c['name'] in wanted]

    print(f"\n  SD configs: {[c['name'] for c in sd_configs]}")

    # ==================================================================
    # Per-sample, per-question benchmark
    # ==================================================================
    baseline_results = []
    config_results = {c['name']: [] for c in sd_configs}

    pbar = tqdm(total=total_runs, desc="E2E Benchmark")
    run_idx = 0

    for idx, sample in enumerate(dataset):
        is_warmup = idx < args.warmup

        for qi, query in enumerate(questions):
            try:
                # ---- VL Baseline: prefill + decode (from raw MP4) ----
                vl_result = vl_prefill_fn(
                    vl_model, vl_processor, sample,
                    args.dataset_dir, query, device)
                if vl_result is None:
                    pbar.update(1)
                    run_idx += 1
                    continue

                bl_tokens, bl_decode_ms = vl_decode_baseline(
                    vl_model, vl_result['kv_cache'],
                    vl_result['first_token'],
                    vl_result['attention_mask'],
                    args.max_new_tokens, eos_token_id, device)
                bl_total = vl_result['prefill_ms'] + bl_decode_ms

                bl_text = vl_processor.tokenizer.decode(
                    bl_tokens, skip_special_tokens=True)

                del vl_result['kv_cache']
                torch.cuda.empty_cache()

                if not is_warmup:
                    baseline_results.append({
                        'sample': idx,
                        'question': query,
                        'prefill_ms': vl_result['prefill_ms'],
                        'decode_ms': bl_decode_ms,
                        'total_ms': bl_total,
                        'n_tokens': len(bl_tokens),
                        'text': bl_text,
                    })

                # ---- SD: parallel prefill + decode per config ----
                # Check if we have any non-VL-decode configs
                has_sd_configs = any(
                    not c['name'].endswith('+VL') for c in sd_configs)
                has_vl_configs = any(
                    c['name'].endswith('+VL') for c in sd_configs)

                # For +VL configs: EGPT prefill + gamma decode overlapped
                # with VL prefill (EGPT decode hidden behind VL prefill)
                par_vl = None
                if has_vl_configs:
                    par_vl = sequential_egpt_vl_prefill(
                        egpt_model, egpt_tokenizer, egpt_processor,
                        vl_model, vl_processor,
                        sample, args.dataset_dir, query, args.gamma,
                        args.max_new_tokens, device,
                        event_image_key=args.event_image_key)

                # For SD configs (B1-only, L5F-only, etc.): standard
                # parallel prefill + full EGPT decode
                par = None
                egpt_decode_hidden = None
                egpt_decode_ms = 0
                if has_sd_configs:
                    par = parallel_prefill(
                        egpt_model, egpt_tokenizer, egpt_processor,
                        vl_model, vl_processor,
                        sample, args.dataset_dir, query, device,
                        event_image_key=args.event_image_key)
                    if par is not None:
                        egpt_tokens, egpt_decode_hidden, egpt_decode_ms = \
                            egpt_decode_collect_hidden(
                                egpt_model,
                                par['egpt']['kv_cache'],
                                par['egpt']['first_token'],
                                par['egpt']['prefill_length'],
                                par['egpt']['last_hidden'],
                                args.max_new_tokens,
                                egpt_tokenizer.eos_token_id,
                                device)

                if par is None and par_vl is None:
                    pbar.update(1)
                    run_idx += 1
                    continue

                # VL prefill last hidden state (h_vl[0] for B1/L5F)
                vl_src = par if par is not None else par_vl
                vl_prefill_last_h = vl_src['vl'].get('hidden_states')
                if vl_prefill_last_h is not None:
                    vl_prefill_last_h = vl_prefill_last_h[:, -1:, :]

                for cfg in sd_configs:
                    is_vl_decode = cfg['name'].endswith('+VL')
                    # Pick the right parallel result source
                    src = par_vl if is_vl_decode else par
                    if src is None:
                        continue

                    vl_kv = tuple(
                        (k.clone(), v.clone())
                        for k, v in src['vl']['kv_cache']
                    )
                    kv_len = vl_kv[0][0].shape[2]
                    attn_dtype = (
                        src['vl']['attention_mask'].dtype
                        if isinstance(src['vl']['attention_mask'],
                                      torch.Tensor)
                        else torch.long
                    )

                    decode_level = cfg['decode_adapter']
                    adapter_info = adapters[decode_level]
                    adapter_model = adapter_info['model']
                    is_decode_only = cfg['prefill_adapter'] is None

                    if is_vl_decode:
                        # L1+VL through L4+VL:
                        #   1. Adapter maps free EGPT hidden → draft tokens
                        #   2. VL verifies all draft tokens in one pass
                        #   3. VL AR decodes remaining tokens
                        adapter_dtype = next(adapter_model.parameters()).dtype

                        vl_first = par_vl['vl']['first_token'].item()
                        generated = [vl_first]
                        sd_drafted = 0
                        sd_accepted = 0

                        # EGPT hidden states from decode
                        vl_egpt_hidden = par_vl['egpt_decode'][1] \
                            if par_vl['egpt_decode'] is not None else None

                        # Only use "free" hidden states (within gap)
                        gamma_free = par_vl.get('gamma_prefill', 0)
                        egpt_n = (vl_egpt_hidden.shape[1]
                                  if vl_egpt_hidden is not None else 0)
                        n_draft = min(gamma_free, egpt_n)

                        # --- Adapter draft (timed separately) ---
                        adapter_ms = 0.0
                        draft_tokens = []
                        with torch.inference_mode():
                            if n_draft > 0:
                                torch.cuda.synchronize()
                                t_adapter = time.time()
                                egpt_chunk = vl_egpt_hidden[
                                    :, :n_draft, :].to(adapter_dtype)
                                aligned = adapter_model(egpt_chunk)
                                W = vl_lm_head_weight.to(device)
                                draft_logits = torch.matmul(
                                    aligned.float(), W.T)
                                draft_tokens = draft_logits.argmax(
                                    dim=-1)[0].tolist()
                                torch.cuda.synchronize()
                                adapter_ms = (
                                    time.time() - t_adapter) * 1000

                        # draft_tokens[0] predicts the FIRST decode token
                        # (from h[0] = prefill last hidden state).
                        # vl_first already IS the first decode token.
                        # So draft_tokens[0] duplicates vl_first — skip it.
                        # Verify draft_tokens[1:] against VL.
                        verify_tokens = draft_tokens[1:] if draft_tokens \
                            else []
                        sd_drafted = len(verify_tokens)

                        # --- VL verify + AR decode (timed together) ---
                        torch.cuda.synchronize()
                        t_vl_decode = time.time()

                        with torch.inference_mode():
                            if sd_drafted > 0:
                                n_acc, new_toks, vl_kv, kv_len, _ = \
                                    vl_verify_batch(
                                        vl_model, generated[-1],
                                        verify_tokens, vl_kv, kv_len,
                                        attn_dtype, device,
                                        output_hidden=False)
                                generated.extend(new_toks)
                                sd_accepted = n_acc

                            # Pure VL AR decode for remaining tokens
                            remaining = args.max_new_tokens - len(generated)
                            if remaining > 0 and generated[-1] != eos_token_id:
                                cur_token = torch.tensor(
                                    [[generated[-1]]], device=device)
                                for _ in range(remaining):
                                    cur_attn = torch.ones(
                                        (1, kv_len + 1), dtype=attn_dtype,
                                        device=device)
                                    outputs = vl_model(
                                        input_ids=cur_token,
                                        attention_mask=cur_attn,
                                        past_key_values=vl_kv,
                                        use_cache=True)
                                    vl_kv = outputs.past_key_values
                                    kv_len += 1
                                    cur_token = torch.argmax(
                                        outputs.logits[:, -1, :],
                                        dim=-1, keepdim=True)
                                    generated.append(cur_token.item())
                                    if cur_token.item() == eos_token_id:
                                        break

                        torch.cuda.synchronize()
                        vl_decode_ms = (
                            time.time() - t_vl_decode) * 1000
                        # sd_decode_ms = adapter + VL verify + VL AR
                        sd_decode_ms = adapter_ms + vl_decode_ms

                        sd_tokens = generated
                        num_free_tok = gamma_free

                        # Log draft vs VL text for first question
                        if qi == 0:
                            draft_text = vl_processor.tokenizer.decode(
                                draft_tokens,
                                skip_special_tokens=True
                            ) if draft_tokens else "(none)"
                            verify_text = vl_processor.tokenizer.decode(
                                verify_tokens,
                                skip_special_tokens=True
                            ) if verify_tokens else "(none)"
                            sd_full_text = vl_processor.tokenizer.decode(
                                generated, skip_special_tokens=True)
                            d0_text = vl_processor.tokenizer.decode(
                                [draft_tokens[0]]) if draft_tokens else "?"
                            vf_text = vl_processor.tokenizer.decode(
                                [vl_first])
                            tqdm.write(
                                f"    {cfg['name']} d[0]='{d0_text}' "
                                f"vl_first='{vf_text}'")
                            tqdm.write(
                                f"    {cfg['name']} verify({sd_drafted}): "
                                f"{verify_text[:80]}")
                            tqdm.write(
                                f"    {cfg['name']} final({len(generated)}): "
                                f"{sd_full_text[:80]}")
                            tqdm.write(
                                f"    VL baseline({len(bl_tokens)}): "
                                f"{bl_text[:80]}")
                            tqdm.write(
                                f"    accept={sd_accepted}/{sd_drafted} "
                                f"adapter={adapter_ms:.1f}ms "
                                f"vl_decode={vl_decode_ms:.1f}ms")

                        sd_stats = {
                            'total_drafted': sd_drafted,
                            'total_accepted': sd_accepted,
                            'accept_rate': (sd_accepted / sd_drafted
                                            if sd_drafted > 0 else 0),
                            'iterations': 1,
                            'tokens_per_iter': len(generated),
                            'num_free_tok': num_free_tok,
                            'adapter_ms': adapter_ms,
                            'vl_decode_ms': vl_decode_ms,
                        }
                    else:
                        # Full SD decode loop (B1-only, L5F-only, L1+L5F, etc.)
                        sd_tokens, sd_decode_ms, sd_stats = run_sd_decode(
                            vl_model, adapter_model, vl_lm_head_weight,
                            vl_kv,
                            par['vl']['first_token'].item(),
                            attn_dtype, kv_len,
                            gamma=args.gamma,
                            max_new_tokens=args.max_new_tokens,
                            eos_token_id=eos_token_id,
                            is_fused=adapter_info['is_fused'],
                            is_vlm_only=adapter_info['is_vlm_only'],
                            egpt_decode_hidden=egpt_decode_hidden,
                            vl_prefill_last_hidden=vl_prefill_last_h,
                            device=device)

                    # E2E total: prefill + decode
                    if is_vl_decode:
                        # L1+VL: EGPT decode overlapped with VL prefill
                        # wall_ms already includes both prefills + EGPT
                        # decode gamma steps. sd_decode_ms = adapter draft
                        # + VL verify + VL AR decode.
                        sd_total = par_vl['wall_ms'] + sd_decode_ms
                    elif is_decode_only:
                        if adapter_info['is_vlm_only']:
                            sd_total = (par['vl_ms'] + sd_decode_ms)
                        else:
                            sd_total = (par['wall_ms'] + egpt_decode_ms
                                        + sd_decode_ms)
                    else:
                        sd_total = (par['wall_ms'] + egpt_decode_ms
                                    + sd_decode_ms)

                    speedup = bl_total / sd_total if sd_total > 0 else 1.0

                    sd_text = vl_processor.tokenizer.decode(
                        sd_tokens, skip_special_tokens=True)

                    del vl_kv
                    torch.cuda.empty_cache()

                    if not is_warmup:
                        p = par_vl if is_vl_decode else par
                        result_entry = {
                            'sample': idx,
                            'question': query,
                            'bl_total_ms': bl_total,
                            'parallel_prefill_ms': p['wall_ms'],
                            'egpt_prefill_ms': p['egpt_ms'],
                            'vl_prefill_ms': p['vl_ms'],
                            'egpt_decode_ms': (
                                par_vl['egpt_decode'][2]
                                if is_vl_decode and par_vl['egpt_decode']
                                else egpt_decode_ms),
                            'overlap_ms': p['overlap_ms'],
                            'overlap_ratio': p['overlap_ratio'],
                            'sd_decode_ms': sd_decode_ms,
                            'sd_total_ms': sd_total,
                            'speedup': speedup,
                            'n_tokens': len(sd_tokens),
                            'drafted': sd_stats['total_drafted'],
                            'accepted': sd_stats['total_accepted'],
                            'accept_rate': sd_stats['accept_rate'],
                            'tokens_per_iter': sd_stats['tokens_per_iter'],
                            'text': sd_text,
                        }
                        if is_vl_decode:
                            result_entry['num_free_tok'] = sd_stats.get(
                                'num_free_tok', 0)
                            result_entry['gap_ms'] = par_vl.get('gap_ms', 0)
                            result_entry['egpt_per_token_ms'] = par_vl.get(
                                'egpt_per_token_ms', 0)
                            result_entry['adapter_ms'] = sd_stats.get(
                                'adapter_ms', 0)
                            result_entry['vl_decode_ms'] = sd_stats.get(
                                'vl_decode_ms', 0)
                        config_results[cfg['name']].append(result_entry)

                del par, egpt_decode_hidden
                torch.cuda.empty_cache()

                if not is_warmup and qi == 0:
                    best_cfg = max(
                        [(c['name'],
                          config_results[c['name']][-1]['speedup'])
                         for c in sd_configs
                         if config_results[c['name']]],
                        key=lambda x: x[1], default=('?', 0))
                    free_tok_info = ""
                    if par_vl is not None:
                        gp = par_vl.get('gamma_prefill', 0)
                        gap = par_vl.get('gap_ms', 0)
                        wms = par_vl.get('wall_ms', 0)
                        free_tok_info = (
                            f"  free_tok={gp} gap={gap:.0f}ms "
                            f"wall={wms:.0f}ms")
                    tqdm.write(
                        f"  #{idx}: BL={bl_total:.0f}ms  "
                        f"best={best_cfg[0]} {best_cfg[1]:.2f}x"
                        f"{free_tok_info}")

            except Exception as e:
                import traceback
                tqdm.write(f"  Error #{idx}q{qi}: {e}")
                traceback.print_exc()

            pbar.update(1)
            run_idx += 1

    pbar.close()

    # Cleanup models
    del egpt_model, egpt_tokenizer, egpt_processor
    for a in adapters.values():
        del a['model']
    del vl_lm_head_weight
    del vl_model, vl_processor
    torch.cuda.empty_cache()
    gc.collect()

    # ==================================================================
    # Aggregate results
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")

    def avg(lst, key):
        vals = [r[key] for r in lst]
        return np.mean(vals) if vals else 0

    def std(lst, key):
        vals = [r[key] for r in lst]
        return np.std(vals) if vals else 0

    n = len(baseline_results)
    print(f"\n  Samples: {n}")
    bl_total_mean = avg(baseline_results, 'total_ms')
    bl_prefill_mean = avg(baseline_results, 'prefill_ms')
    bl_decode_mean = avg(baseline_results, 'decode_ms')

    print(f"\n  VL Baseline:")
    print(f"    Prefill:  {bl_prefill_mean:>7.1f} +/- "
          f"{std(baseline_results, 'prefill_ms'):>5.1f} ms")
    print(f"    Decode:   {bl_decode_mean:>7.1f} +/- "
          f"{std(baseline_results, 'decode_ms'):>5.1f} ms")
    print(f"    Total:    {bl_total_mean:>7.1f} +/- "
          f"{std(baseline_results, 'total_ms'):>5.1f} ms")

    # Aggregate per config
    all_agg = {}
    all_agg['VL baseline'] = {
        'speedup_mean': 1.0,
        'prefill_ms_mean': bl_prefill_mean,
        'decode_ms_mean': bl_decode_mean,
        'accept_rate_mean': 0,
    }

    print(f"\n  {'Config':<18} | {'Prefill':>8} | {'Decode':>8} | "
          f"{'Total':>8} | {'Accept':>7} | {'Speedup':>7} | {'FreeTok':>7}")
    print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    print(f"  {'VL baseline':<18} | {bl_prefill_mean:>7.0f}ms | "
          f"{bl_decode_mean:>7.0f}ms | {bl_total_mean:>7.0f}ms | "
          f"{'---':>7} | {'1.00x':>7} | {'---':>7}")

    for cfg in sd_configs:
        name = cfg['name']
        results = config_results[name]
        if not results:
            continue

        prefill_mean = avg(results, 'parallel_prefill_ms')
        decode_mean = avg(results, 'sd_decode_ms')
        total_mean = avg(results, 'sd_total_ms')
        speedup_mean = avg(results, 'speedup')
        accept_mean = avg(results, 'accept_rate')
        overlap_mean = avg(results, 'overlap_ms')

        # num_free_tok only for +VL configs
        is_vl = name.endswith('+VL')
        free_tok_vals = [r.get('num_free_tok', 0) for r in results
                         if 'num_free_tok' in r]
        free_tok_mean = np.mean(free_tok_vals) if free_tok_vals else 0
        free_str = f"{free_tok_mean:>5.1f}" if is_vl else "---"

        print(f"  {name:<18} | {prefill_mean:>7.0f}ms | "
              f"{decode_mean:>7.0f}ms | {total_mean:>7.0f}ms | "
              f"{accept_mean:>6.1%} | {speedup_mean:>6.2f}x | "
              f"{free_str:>7}")

        agg_entry = {
            'speedup_mean': speedup_mean,
            'prefill_ms_mean': prefill_mean,
            'decode_ms_mean': decode_mean,
            'total_ms_mean': total_mean,
            'accept_rate_mean': accept_mean,
            'overlap_ms_mean': overlap_mean,
            'n_samples': len(results),
        }
        if is_vl and free_tok_vals:
            agg_entry['num_free_tok_mean'] = free_tok_mean
        all_agg[name] = agg_entry

    # ==================================================================
    # Generate graphs
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("GENERATING GRAPHS")
    print(f"{'=' * 70}")
    # Collect one example for the graph (first non-warmup +VL result)
    example_data = None
    for cfg in sd_configs:
        name = cfg['name']
        if name.endswith('+VL') and config_results.get(name):
            r = config_results[name][0]  # first sample
            # Find matching baseline
            bl_match = None
            for br in baseline_results:
                if br['sample'] == r['sample'] and br['question'] == r['question']:
                    bl_match = br
                    break
            if bl_match:
                example_data = {
                    'config_name': name,
                    'bl_text': bl_match.get('text', ''),
                    'sd_text': r.get('text', ''),
                    'drafted': r.get('drafted', 0),
                    'accepted': r.get('accepted', 0),
                    'num_free_tok': r.get('num_free_tok', 0),
                    'gap_ms': r.get('gap_ms', 0),
                    'egpt_prefill_ms': r.get('egpt_prefill_ms', 0),
                    'vl_prefill_ms': r.get('vl_prefill_ms', 0),
                    'egpt_decode_ms': r.get('egpt_decode_ms', 0),
                    'speedup': r.get('speedup', 1.0),
                }
                break
    generate_graphs(all_agg, args.output_dir, timestamp,
                    stage_timing=stage_timing, example=example_data)

    # ==================================================================
    # Save JSON
    # ==================================================================
    output = {
        'timestamp': timestamp,
        'config': {
            'dataset_dir': args.dataset_dir,
            'max_samples': args.max_samples,
            'max_new_tokens': args.max_new_tokens,
            'gamma': args.gamma,
            'warmup': args.warmup,
            'n_questions': len(questions),
            'questions': questions,
        },
        'stage_timing': stage_timing,
        'aggregate': all_agg,
        'baseline_per_sample': baseline_results,
        'sd_per_sample': {k: v for k, v in config_results.items() if v},
    }

    json_path = os.path.join(
        args.output_dir, f'e2e_wallclock_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ==================================================================
    # Markdown report
    # ==================================================================
    md_lines = [
        "# E2E Wall-Clock Benchmark — All Methods",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Configuration",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Samples: {n} (after {args.warmup} warmup)",
        f"- Questions: {len(questions)}",
        f"- Max tokens: {args.max_new_tokens}, gamma: {args.gamma}",
        "",
        "## Results",
        "",
        "| Config | Prefill (ms) | Decode (ms) | Total (ms) | Accept | Speedup | FreeTok |",
        "|--------|-------------|------------|-----------|--------|---------|---------|",
        f"| VL baseline | {bl_prefill_mean:.0f} | {bl_decode_mean:.0f} | "
        f"{bl_total_mean:.0f} | --- | 1.00x | --- |",
    ]

    for cfg in sd_configs:
        name = cfg['name']
        if name not in all_agg:
            continue
        a = all_agg[name]
        ft = a.get('num_free_tok_mean')
        ft_str = f"{ft:.1f}" if ft is not None else "---"
        md_lines.append(
            f"| {name} | {a['prefill_ms_mean']:.0f} | "
            f"{a['decode_ms_mean']:.0f} | {a['total_ms_mean']:.0f} | "
            f"{a['accept_rate_mean']:.1%} | **{a['speedup_mean']:.2f}x** | "
            f"{ft_str} |")

    # 3-stage timing table
    md_lines += [
        "",
        "## 3-Stage Timing (Both Models)",
        "",
        "| Model | Vision (ms) | Prefill (ms) | Decode (ms) | Total (ms) | ms/token |",
        "|-------|------------|-------------|------------|-----------|----------|",
    ]
    et = stage_timing['eventgpt']
    vt = stage_timing['videollava']
    md_lines.append(
        f"| EventGPT | {et['vision_ms']:.1f} | {et['prefill_ms']:.1f} | "
        f"{et['decode_ms']:.1f} | {et['total_ms']:.1f} | "
        f"{et['per_token_ms']:.1f} |")
    md_lines.append(
        f"| Video-LLaVA | {vt['vision_ms']:.1f} | {vt['prefill_ms']:.1f} | "
        f"{vt['decode_ms']:.1f} | {vt['total_ms']:.1f} | "
        f"{vt['per_token_ms']:.1f} |")

    md_lines += [
        "",
        "## Graphs",
        "",
        f"![Speedup](speedup_comparison_{timestamp}.png)",
        f"![Timing](timing_breakdown_{timestamp}.png)",
        f"![Accept](accept_rate_comparison_{timestamp}.png)",
        "",
    ]

    md_path = os.path.join(
        args.output_dir, f'e2e_wallclock_{timestamp}.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {md_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
