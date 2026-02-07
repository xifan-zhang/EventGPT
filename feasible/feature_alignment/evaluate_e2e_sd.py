#!/usr/bin/env python3
"""
End-to-End Speculative Decoding Evaluation
===========================================

Evaluates cross-modal speculative decoding quality at three levels:

A. Hidden-State Level Metrics (cosine sim, accept rates, consecutive accepts)
B. Token Level Metrics (project through VL LM head, compare argmax tokens)
C. Wall-Clock Speedup (two-phase: prefill hiding + decode acceleration)

Key Insight - L1-L4 vs L5 Phase Impact:
    L1-L4: Speedup ONLY from prefill hiding (EGPT ~25ms/tok same as VL)
    L5:    Speedup from BOTH prefill hiding + cheaper decode drafting (~3ms/tok)

Modes:
    live:   Loads both models, generates fresh hidden states + timing
    cached: Uses pre-extracted hidden states, loads VL for LM head only

Usage:
    # Live mode (loads both models, generates, times)
    python feasible/feature_alignment/evaluate_e2e_sd.py \\
        --adapter_checkpoint ./feasible/feature_alignment/tasks/L1/.../best_model.pt \\
        --dataset_dir ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \\
        --max_samples 50 --max_questions 10 --gamma_decode 5

    # Cached mode (uses pre-extracted hidden states, loads VL for LM head only)
    python feasible/feature_alignment/evaluate_e2e_sd.py \\
        --mode cached \\
        --adapter_checkpoint ./feasible/feature_alignment/tasks/L1/.../best_model.pt \\
        --cached_hidden_states /mnt/hdd/data/egpt/hidden_states/chunked_test_1s_4bit

Updated: 2026-02-06
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Fix protobuf compatibility
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from feasible.feature_alignment.hidden_adapter import load_any_adapter


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TimingConfig:
    """Default timing configuration (ms). Overridden by live measurements."""
    egpt_prefill: float = 130.0
    egpt_per_token: float = 25.0
    vl_prefill: float = 310.0
    vl_per_token: float = 25.0
    adapter_latency: float = 1.5
    l5_draft_per_token: float = 3.0  # L5 EAGLE adapter cost per token


# =============================================================================
# 1. load_models() - Load both models in 4-bit
# =============================================================================

def load_models(load_vl_only: bool = False) -> Dict:
    """
    Load models for evaluation.

    Args:
        load_vl_only: If True, only load Video-LLaVA (for cached mode, LM head access)

    Returns:
        Dict with model references and utilities
    """
    from transformers import (
        AutoTokenizer, BitsAndBytesConfig,
        VideoLlavaForConditionalGeneration, VideoLlavaProcessor,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    result = {}

    if not load_vl_only:
        from model.EventChatModel import EventChatModel

        print("\n[1/2] Loading EventGPT (4-bit)...")
        eventgpt_path = "/home/ps/Documents/code/EventGPT/checkpoints/EventGPT-7b"
        egpt_model = EventChatModel.from_pretrained(
            eventgpt_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )
        egpt_model.eval()
        egpt_tokenizer = AutoTokenizer.from_pretrained(eventgpt_path, use_fast=True)
        egpt_processor = egpt_model.get_visual_tower().event_processor
        print(f"  EventGPT loaded. GPU: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

        result['egpt_model'] = egpt_model
        result['egpt_tokenizer'] = egpt_tokenizer
        result['egpt_processor'] = egpt_processor
        step = "[2/2]"
    else:
        step = "[1/1]"

    print(f"\n{step} Loading Video-LLaVA (4-bit)...")
    videollava_model_id = "LanguageBind/Video-LLaVA-7B-hf"
    vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
        videollava_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    vl_processor = VideoLlavaProcessor.from_pretrained(videollava_model_id)
    vl_model.eval()
    print(f"  Video-LLaVA loaded. GPU: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

    # Get VL LM head (important: use language_model.lm_head, NOT vl_model.lm_head)
    vl_lm_head = vl_model.language_model.lm_head

    result['vl_model'] = vl_model
    result['vl_processor'] = vl_processor
    result['vl_lm_head'] = vl_lm_head
    result['bnb_config'] = bnb_config

    return result


# =============================================================================
# 2. extract_with_timing() - Run both models, get hidden states + timing
# =============================================================================

def extract_with_timing(
    models: Dict,
    sample: Dict,
    dataset_dir: Path,
    query: str,
    max_new_tokens: int = 50,
) -> Optional[Dict]:
    """
    Run both models on one sample. Extract hidden states, tokens, and timing.

    Returns dict with:
        egpt_hidden, vl_hidden, egpt_tokens, vl_tokens,
        egpt_text, vl_text, timing (dict of ms values)
    """
    from model.EventChatModel import get_spatio_temporal_features
    from dataset.conversation import prepare_event_prompt
    from dataset.constants import EVENT_TOKEN_INDEX
    from common.common import tokenizer_event_token, load_image

    device = "cuda"

    try:
        import av
    except ImportError:
        raise ImportError("PyAV required. Install with: pip install av")

    # --- EventGPT ---
    try:
        if "event_image" not in sample or not sample["event_image"]:
            return None

        event_image_paths = sample["event_image"]
        img_path = dataset_dir / "event_image" / event_image_paths[0]
        if not img_path.exists():
            return None

        img = load_image(str(img_path))
        img_array = np.array(img)
        event_image_size = list(img_array.shape[:2])

        event = models['egpt_processor'](img_array, return_tensors='pt')['pixel_values'][0]
        event = event.to(device, dtype=torch.bfloat16)

        conv_mode = 'eventgpt_v1'
        prompt = prepare_event_prompt(query, conv_mode)
        input_ids = tokenizer_event_token(
            prompt, models['egpt_tokenizer'], EVENT_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)

        # Warm up
        torch.cuda.synchronize()

        # EGPT: vision encode + prefill + decode with timing
        torch.cuda.synchronize()
        t0_egpt = torch.cuda.Event(enable_timing=True)
        t1_egpt = torch.cuda.Event(enable_timing=True)

        t0_egpt.record()
        with torch.inference_mode():
            feature = models['egpt_model'].visval_encode(event.unsqueeze(0))
            feature = models['egpt_model'].get_model().feature_adaptor(feature)
            feature = feature.squeeze(0)
            event_features = get_spatio_temporal_features([feature])
            event_features = event_features.unsqueeze(0)

        torch.cuda.synchronize()
        t_egpt_vision = torch.cuda.Event(enable_timing=True)
        t_egpt_vision.record()

        with torch.inference_mode():
            egpt_outputs = models['egpt_model'].generate(
                input_ids,
                event_features=event_features,
                event_image_sizes=[event_image_size],
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        t1_egpt.record()
        torch.cuda.synchronize()

        egpt_prefill_ms = t0_egpt.elapsed_time(t_egpt_vision)
        egpt_total_ms = t0_egpt.elapsed_time(t1_egpt)

        # Extract EGPT hidden states + tokens
        egpt_generated_ids = egpt_outputs.sequences[0]
        egpt_text = models['egpt_tokenizer'].decode(egpt_generated_ids, skip_special_tokens=True)
        if "ASSISTANT:" in egpt_text:
            egpt_text = egpt_text.split("ASSISTANT:")[-1].strip()

        egpt_all_hidden = []
        for step_hidden in egpt_outputs.hidden_states:
            last_layer = step_hidden[-1]
            last_pos = last_layer[0, -1, :]
            egpt_all_hidden.append(last_pos)
        egpt_hidden = torch.stack(egpt_all_hidden, dim=0)

        num_egpt_tokens = len(egpt_all_hidden)
        egpt_decode_ms = egpt_total_ms - egpt_prefill_ms
        egpt_per_token_ms = egpt_decode_ms / max(num_egpt_tokens, 1)

    except Exception as e:
        print(f"  EGPT error: {e}")
        return None

    # --- Video-LLaVA ---
    try:
        video_data = sample.get("video_data")
        if not video_data:
            return None

        mp4_path = dataset_dir / "mp4" / f"{video_data}.mp4"
        if not mp4_path.exists():
            return None

        # Load video frames
        container = av.open(str(mp4_path))
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames == 0:
            total_frames = sum(1 for _ in container.decode(stream))
            container.seek(0)

        num_frames = 8
        if total_frames >= num_frames:
            indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int))
        else:
            indices = set(range(total_frames))

        frames = []
        for i, frame in enumerate(container.decode(stream)):
            if i in indices:
                frames.append(frame.to_image())
        container.close()

        if not frames:
            return None

        vl_prompt = f"USER: <video>\n{query} ASSISTANT:"
        inputs = models['vl_processor'](
            text=vl_prompt, videos=frames, return_tensors="pt", padding=True,
        )

        vl_input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        pixel_values_videos = inputs["pixel_values_videos"].to(device, dtype=torch.float16)

        torch.cuda.synchronize()
        t0_vl = torch.cuda.Event(enable_timing=True)
        t1_vl = torch.cuda.Event(enable_timing=True)

        t0_vl.record()
        with torch.inference_mode():
            vl_outputs = models['vl_model'].generate(
                input_ids=vl_input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        t1_vl.record()
        torch.cuda.synchronize()

        vl_total_ms = t0_vl.elapsed_time(t1_vl)

        # Extract VL hidden states + tokens
        vl_generated_ids = vl_outputs.sequences[0][vl_input_ids.shape[1]:]
        vl_text = models['vl_processor'].tokenizer.decode(vl_generated_ids, skip_special_tokens=True)

        vl_all_hidden = []
        for step_hidden in vl_outputs.hidden_states:
            last_layer = step_hidden[-1]
            last_pos = last_layer[0, -1, :]
            vl_all_hidden.append(last_pos)
        vl_hidden = torch.stack(vl_all_hidden, dim=0)

        num_vl_tokens = len(vl_all_hidden)
        # Estimate VL prefill as fraction of total
        # First token generation includes prefill; remaining are decode
        if num_vl_tokens > 1:
            vl_per_token_ms = vl_total_ms / num_vl_tokens
            # Rough prefill estimate: total - (num_tokens * avg_decode)
            # Better: first step is prefill+decode, rest are pure decode
            vl_prefill_ms = vl_total_ms * 0.3  # ~30% is prefill typically
        else:
            vl_per_token_ms = vl_total_ms
            vl_prefill_ms = vl_total_ms

    except Exception as e:
        print(f"  VL error: {e}")
        return None

    # Align lengths
    min_len = min(len(egpt_hidden), len(vl_hidden))
    if min_len < 5:
        return None

    egpt_hidden = egpt_hidden[:min_len]
    vl_hidden = vl_hidden[:min_len]

    return {
        'egpt_hidden': egpt_hidden.cpu(),
        'vl_hidden': vl_hidden.cpu(),
        'egpt_tokens': egpt_generated_ids.tolist(),
        'vl_tokens': vl_generated_ids.tolist(),
        'egpt_text': egpt_text,
        'vl_text': vl_text,
        'seq_len': min_len,
        'timing': {
            'egpt_prefill_ms': egpt_prefill_ms,
            'egpt_total_ms': egpt_total_ms,
            'egpt_per_token_ms': egpt_per_token_ms,
            'egpt_num_tokens': num_egpt_tokens,
            'vl_total_ms': vl_total_ms,
            'vl_prefill_ms': vl_prefill_ms,
            'vl_per_token_ms': vl_per_token_ms,
            'vl_num_tokens': num_vl_tokens,
        },
    }


# =============================================================================
# 3. compute_hidden_state_metrics() - cos sim, accept rates, consecutive
# =============================================================================

def compute_hidden_state_metrics(
    aligned_hidden: torch.Tensor,
    vl_hidden: torch.Tensor,
    mask: torch.Tensor,
    thresholds: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
) -> Dict:
    """
    Compute hidden-state level metrics (vectorized).

    Reuses logic from measure_feature_acceptance.py:compute_all_metrics_parallel().

    Args:
        aligned_hidden: [batch, seq, hidden] aligned EGPT hidden states
        vl_hidden: [batch, seq, hidden] VL hidden states
        mask: [batch, seq] valid positions
        thresholds: cosine similarity thresholds

    Returns:
        Dict with cos_sim stats, accept rates, consecutive accepts, per-position
    """
    batch_size, max_seq, hidden_dim = aligned_hidden.shape

    # Cosine similarity
    aligned_norm = F.normalize(aligned_hidden, dim=-1)
    vl_norm_t = F.normalize(vl_hidden, dim=-1)
    cos_sim = (aligned_norm * vl_norm_t).sum(dim=-1)  # [batch, seq]

    valid_sims = cos_sim[mask.bool()]

    metrics = {
        'cos_sim_mean': valid_sims.mean().item(),
        'cos_sim_std': valid_sims.std().item(),
        'cos_sim_min': valid_sims.min().item(),
        'cos_sim_max': valid_sims.max().item(),
        'cos_sim_median': valid_sims.median().item(),
        'total_valid_positions': int(mask.sum().item()),
    }

    valid_count = mask.sum()

    # Accept rates
    for thresh in thresholds:
        accept_mask = (cos_sim > thresh) & mask.bool()
        metrics[f'accept_{int(thresh*100)}'] = (accept_mask.float().sum() / valid_count).item()

    # Consecutive accepts
    for thresh in thresholds:
        thresh_key = int(thresh * 100)
        accept_int = (cos_sim > thresh).int()
        cumprod = accept_int.cumprod(dim=1)
        consecutive_per_sample = cumprod.sum(dim=1).float()
        seq_lens = mask.sum(dim=1)
        consecutive_per_sample = torch.minimum(consecutive_per_sample, seq_lens)

        metrics[f'consecutive_mean_{thresh_key}'] = consecutive_per_sample.mean().item()
        metrics[f'consecutive_std_{thresh_key}'] = consecutive_per_sample.std().item()
        metrics[f'consecutive_max_{thresh_key}'] = consecutive_per_sample.max().item()
        metrics[f'consecutive_min_{thresh_key}'] = consecutive_per_sample.min().item()

    # Per-position acceptance (first 20 positions)
    position_accept_90 = []
    position_mean_sim = []
    for pos in range(min(max_seq, 20)):
        valid_at_pos = mask[:, pos].bool()
        if valid_at_pos.sum() > 0:
            sims_at_pos = cos_sim[valid_at_pos, pos]
            position_accept_90.append((sims_at_pos > 0.90).float().mean().item())
            position_mean_sim.append(sims_at_pos.mean().item())
        else:
            break

    metrics['position_accept_90'] = position_accept_90
    metrics['position_mean_sim'] = position_mean_sim

    return metrics


# =============================================================================
# 4. compute_token_level_metrics() - Project through VL LM head
# =============================================================================

def compute_token_level_metrics(
    egpt_hidden: torch.Tensor,
    vl_hidden: torch.Tensor,
    adapter: torch.nn.Module,
    vl_lm_head: torch.nn.Module,
    mask: torch.Tensor,
    gamma_prefill: int = None,
    gamma_decode: int = 5,
) -> Dict:
    """
    Compute token-level metrics by projecting hidden states through VL LM head.

    draft_token = argmax(VL_LM_head(adapter(h_egpt)))
    target_token = argmax(VL_LM_head(h_vl))

    Args:
        egpt_hidden: [batch, seq, hidden] raw EGPT hidden states
        vl_hidden: [batch, seq, hidden] VL hidden states (target)
        adapter: trained adapter module
        vl_lm_head: VL language model's LM head (nn.Linear)
        mask: [batch, seq] valid positions
        gamma_prefill: number of prefill draft tokens (auto-computed if None)
        gamma_decode: number of decode draft tokens per iteration

    Returns:
        Dict with token match rates, top-5 match, consecutive, per-position
    """
    device = egpt_hidden.device
    batch_size, max_seq, hidden_dim = egpt_hidden.shape

    with torch.no_grad():
        # Align EGPT hidden states
        aligned = adapter(egpt_hidden)

        # Project through VL LM head
        draft_logits = vl_lm_head(aligned)     # [batch, seq, vocab]
        target_logits = vl_lm_head(vl_hidden)  # [batch, seq, vocab]

        # Argmax tokens
        draft_tokens = draft_logits.argmax(dim=-1)   # [batch, seq]
        target_tokens = target_logits.argmax(dim=-1)  # [batch, seq]

        # Top-5 from target
        target_top5 = target_logits.topk(5, dim=-1).indices  # [batch, seq, 5]

    # Token match (argmax)
    token_match = (draft_tokens == target_tokens) & mask.bool()
    valid_count = mask.sum()
    token_match_rate = token_match.float().sum() / valid_count

    # Top-5 match: is draft_token in target's top-5?
    draft_expanded = draft_tokens.unsqueeze(-1).expand_as(target_top5)
    top5_match = ((draft_expanded == target_top5).any(dim=-1)) & mask.bool()
    top5_match_rate = top5_match.float().sum() / valid_count

    # Consecutive token matches
    match_int = token_match.int()
    cumprod = match_int.cumprod(dim=1)
    consecutive_per_sample = cumprod.sum(dim=1).float()
    seq_lens = mask.sum(dim=1)
    consecutive_per_sample = torch.minimum(consecutive_per_sample, seq_lens)

    metrics = {
        'token_match_rate': token_match_rate.item(),
        'top5_match_rate': top5_match_rate.item(),
        'consecutive_token_mean': consecutive_per_sample.mean().item(),
        'consecutive_token_std': consecutive_per_sample.std().item(),
        'consecutive_token_max': consecutive_per_sample.max().item(),
        'consecutive_token_min': consecutive_per_sample.min().item(),
    }

    # Per-position token match (first 20 positions)
    position_token_match = []
    for pos in range(min(max_seq, 20)):
        valid_at_pos = mask[:, pos].bool()
        if valid_at_pos.sum() > 0:
            matches = token_match[valid_at_pos, pos]
            position_token_match.append(matches.float().mean().item())
        else:
            break
    metrics['position_token_match'] = position_token_match

    # Prefill acceptance (gamma_prefill tokens)
    if gamma_prefill is None:
        gamma_prefill = int(seq_lens.float().mean().item())

    gamma_p = min(gamma_prefill, max_seq)
    if gamma_p > 0:
        prefill_match = token_match[:, :gamma_p]
        prefill_mask = mask[:, :gamma_p]
        if prefill_mask.sum() > 0:
            prefill_match_rate = prefill_match.float().sum() / prefill_mask.sum()
            # Consecutive in prefill region
            prefill_cumprod = prefill_match.int().cumprod(dim=1)
            prefill_consec = prefill_cumprod.sum(dim=1).float()
            metrics['prefill_token_match_rate'] = prefill_match_rate.item()
            metrics['prefill_consecutive_mean'] = prefill_consec.mean().item()
        else:
            metrics['prefill_token_match_rate'] = 0.0
            metrics['prefill_consecutive_mean'] = 0.0
    metrics['gamma_prefill_used'] = gamma_p

    # Decode acceptance (gamma_decode tokens, for L5 EAGLE)
    gamma_d = min(gamma_decode, max_seq)
    if gamma_d > 0:
        decode_match = token_match[:, :gamma_d]
        decode_mask = mask[:, :gamma_d]
        if decode_mask.sum() > 0:
            decode_match_rate = decode_match.float().sum() / decode_mask.sum()
            decode_cumprod = decode_match.int().cumprod(dim=1)
            decode_consec = decode_cumprod.sum(dim=1).float()
            metrics['decode_token_match_rate'] = decode_match_rate.item()
            metrics['decode_consecutive_mean'] = decode_consec.mean().item()
        else:
            metrics['decode_token_match_rate'] = 0.0
            metrics['decode_consecutive_mean'] = 0.0
    metrics['gamma_decode_used'] = gamma_d

    return metrics


# =============================================================================
# 5. simulate_two_phase_sd() - Speedup calculation
# =============================================================================

def simulate_two_phase_sd(
    timing: TimingConfig,
    hidden_metrics: Dict,
    token_metrics: Dict,
    adapter_level: int,
    num_output_tokens: int = 50,
    gamma_decode: int = 5,
) -> Dict:
    """
    Simulate two-phase speculative decoding speedup.

    Phase 1 (Prefill Hiding): Both models run in parallel, EGPT drafts
        gamma_prefill = floor((VL_prefill - EGPT_prefill) / EGPT_per_token)
    Phase 2 (Decode):
        L1-L4: NO decode acceleration (draft cost = EGPT full decode ~ VL decode)
        L5: Decode acceleration (draft cost = adapter only, ~3ms/token)

    Args:
        timing: TimingConfig with measured or default timings
        hidden_metrics: from compute_hidden_state_metrics
        token_metrics: from compute_token_level_metrics
        adapter_level: 1-5
        num_output_tokens: assumed output length
        gamma_decode: draft length per decode iteration (L5 only)

    Returns:
        Dict with baseline_time, sd_time, speedup, phase breakdowns
    """
    t = timing
    is_l5 = (adapter_level == 5)

    # ---- Phase 1: Prefill Hiding ----
    # During VL prefill, EGPT can run vision + generate draft tokens
    free_time = max(0, t.vl_prefill - t.egpt_prefill)
    gamma_prefill = int(free_time / t.egpt_per_token) if t.egpt_per_token > 0 else 0

    # Accepted prefill tokens (use consecutive hidden-state accepts @0.90)
    consec_hs_90 = hidden_metrics.get('consecutive_mean_90', 0)
    # Also check token-level acceptance for prefill
    prefill_consec_token = token_metrics.get('prefill_consecutive_mean', 0)

    # Use the more conservative (token-level) for actual acceptance
    prefill_accepted = min(prefill_consec_token, gamma_prefill)
    # Fall back to hidden-state metric if token data unavailable
    if prefill_consec_token == 0 and consec_hs_90 > 0:
        prefill_accepted = min(consec_hs_90, gamma_prefill)

    # Prefill phase wall-clock
    prefill_time = max(t.vl_prefill, t.egpt_prefill + gamma_prefill * t.egpt_per_token)

    # Tokens remaining after prefill
    tokens_remaining = max(0, num_output_tokens - prefill_accepted)

    # ---- Phase 2: Decode ----
    if is_l5:
        # L5: EAGLE-style decode drafting at ~3ms/token
        decode_consec = token_metrics.get('decode_consecutive_mean', 0)
        avg_accepted_per_iter = max(decode_consec, 0)

        if avg_accepted_per_iter > 0:
            tokens_per_iter = avg_accepted_per_iter + 1  # accepted + 1 bonus
            iterations = tokens_remaining / tokens_per_iter
            # Each iteration: draft gamma tokens + VL verify
            time_per_iter = (gamma_decode * t.l5_draft_per_token + t.vl_per_token * (gamma_decode + 1))
            decode_time = iterations * time_per_iter
        else:
            # No acceptance - fall back to standard decode
            decode_time = tokens_remaining * t.vl_per_token
    else:
        # L1-L4: NO decode acceleration
        # Draft cost ~ EGPT full decode ~ VL decode, so no speedup
        decode_time = tokens_remaining * t.vl_per_token

    # ---- E2E ----
    sd_total_time = prefill_time + decode_time
    baseline_time = t.vl_prefill + num_output_tokens * t.vl_per_token
    speedup = baseline_time / sd_total_time if sd_total_time > 0 else 1.0

    result = {
        # Phase 1: Prefill
        'phase1_prefill_hiding': {
            'vl_prefill_ms': t.vl_prefill,
            'egpt_prefill_ms': t.egpt_prefill,
            'free_time_ms': free_time,
            'gamma_prefill': gamma_prefill,
            'prefill_accepted_tokens': prefill_accepted,
            'prefill_wall_clock_ms': prefill_time,
        },
        # Phase 2: Decode
        'phase2_decode': {
            'is_l5': is_l5,
            'tokens_remaining': tokens_remaining,
            'decode_wall_clock_ms': decode_time,
        },
        # E2E
        'e2e': {
            'baseline_time_ms': baseline_time,
            'sd_total_time_ms': sd_total_time,
            'speedup': speedup,
            'num_output_tokens': num_output_tokens,
            'adapter_level': adapter_level,
        },
    }

    if is_l5:
        result['phase2_decode'].update({
            'gamma_decode': gamma_decode,
            'avg_accepted_per_iter': avg_accepted_per_iter,
            'l5_draft_cost_per_token_ms': t.l5_draft_per_token,
        })

    return result


# =============================================================================
# 6. generate_report() - Markdown report
# =============================================================================

def generate_report(
    hidden_metrics: Dict,
    token_metrics: Dict,
    sd_result: Dict,
    timing: TimingConfig,
    adapter_level: int,
    adapter_path: str,
    num_samples: int,
    output_path: Path,
) -> str:
    """Generate formatted markdown report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"# End-to-End Speculative Decoding Evaluation Report")
    lines.append(f"")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Adapter:** L{adapter_level} (`{adapter_path}`)")
    lines.append(f"**Samples:** {num_samples}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # A. Hidden-State Level
    lines.append(f"## A. Hidden-State Level Metrics")
    lines.append(f"")
    lines.append(f"### Cosine Similarity")
    lines.append(f"| Statistic | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Mean | {hidden_metrics['cos_sim_mean']:.4f} |")
    lines.append(f"| Std | {hidden_metrics['cos_sim_std']:.4f} |")
    lines.append(f"| Median | {hidden_metrics['cos_sim_median']:.4f} |")
    lines.append(f"| Min / Max | {hidden_metrics['cos_sim_min']:.4f} / {hidden_metrics['cos_sim_max']:.4f} |")
    lines.append(f"")

    lines.append(f"### Acceptance Rates")
    lines.append(f"| Threshold | Accept Rate | Consecutive (mean) |")
    lines.append(f"|-----------|------------|---------------------|")
    for thresh in [80, 85, 90, 95]:
        rate = hidden_metrics.get(f'accept_{thresh}', 0)
        consec = hidden_metrics.get(f'consecutive_mean_{thresh}', 0)
        lines.append(f"| @0.{thresh} | {rate:.2%} | {consec:.2f} tokens |")
    lines.append(f"")

    # B. Token Level
    lines.append(f"## B. Token Level Metrics (VL LM Head Projection)")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Token Match (argmax) | {token_metrics['token_match_rate']:.2%} |")
    lines.append(f"| Top-5 Match | {token_metrics['top5_match_rate']:.2%} |")
    lines.append(f"| Consecutive Token Matches | {token_metrics['consecutive_token_mean']:.2f} tokens |")
    lines.append(f"| Prefill Token Match | {token_metrics.get('prefill_token_match_rate', 0):.2%} |")
    lines.append(f"| Prefill Consecutive | {token_metrics.get('prefill_consecutive_mean', 0):.2f} tokens |")
    lines.append(f"| Decode Token Match (gamma={token_metrics.get('gamma_decode_used', 5)}) | {token_metrics.get('decode_token_match_rate', 0):.2%} |")
    lines.append(f"| Decode Consecutive | {token_metrics.get('decode_consecutive_mean', 0):.2f} tokens |")
    lines.append(f"")

    # Per-Position comparison
    hs_pos = hidden_metrics.get('position_accept_90', [])
    tk_pos = token_metrics.get('position_token_match', [])
    if hs_pos or tk_pos:
        max_pos = max(len(hs_pos), len(tk_pos))
        lines.append(f"### Per-Position Acceptance")
        lines.append(f"| Position | Hidden-State @0.90 | Token Match |")
        lines.append(f"|----------|-------------------|-------------|")
        for pos in range(min(max_pos, 15)):
            hs_val = f"{hs_pos[pos]:.2%}" if pos < len(hs_pos) else "N/A"
            tk_val = f"{tk_pos[pos]:.2%}" if pos < len(tk_pos) else "N/A"
            lines.append(f"| {pos} | {hs_val} | {tk_val} |")
        lines.append(f"")

    # C. Two-Phase SD Speedup
    p1 = sd_result['phase1_prefill_hiding']
    p2 = sd_result['phase2_decode']
    e2e = sd_result['e2e']

    lines.append(f"## C. Two-Phase SD Speedup")
    lines.append(f"")
    lines.append(f"### Phase 1: Prefill Hiding")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| VL Prefill | {p1['vl_prefill_ms']:.1f} ms |")
    lines.append(f"| EGPT Prefill | {p1['egpt_prefill_ms']:.1f} ms |")
    lines.append(f"| Free overlap time | {p1['free_time_ms']:.1f} ms |")
    lines.append(f"| gamma_prefill | {p1['gamma_prefill']} tokens |")
    lines.append(f"| Prefill accepted | {p1['prefill_accepted_tokens']:.1f} tokens |")
    lines.append(f"| Prefill wall-clock | {p1['prefill_wall_clock_ms']:.1f} ms |")
    lines.append(f"")

    lines.append(f"### Phase 2: Decode")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Adapter level | L{e2e['adapter_level']} ({'EAGLE decode' if p2['is_l5'] else 'No decode accel'}) |")
    lines.append(f"| Tokens remaining | {p2['tokens_remaining']:.0f} |")
    if p2['is_l5']:
        lines.append(f"| gamma_decode | {p2.get('gamma_decode', 5)} |")
        lines.append(f"| Accepted/iteration | {p2.get('avg_accepted_per_iter', 0):.2f} tokens |")
        lines.append(f"| L5 draft cost | {p2.get('l5_draft_cost_per_token_ms', 3.0):.1f} ms/token |")
    lines.append(f"| Decode wall-clock | {p2['decode_wall_clock_ms']:.1f} ms |")
    lines.append(f"")

    lines.append(f"### E2E Speedup")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Baseline (VL only) | {e2e['baseline_time_ms']:.1f} ms |")
    lines.append(f"| SD total | {e2e['sd_total_time_ms']:.1f} ms |")
    lines.append(f"| **Speedup** | **{e2e['speedup']:.2f}x** |")
    lines.append(f"| Output tokens | {e2e['num_output_tokens']} |")
    lines.append(f"")

    # Timing config
    lines.append(f"---")
    lines.append(f"### Timing Configuration Used")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| EGPT prefill | {timing.egpt_prefill:.1f} ms |")
    lines.append(f"| EGPT per-token | {timing.egpt_per_token:.1f} ms |")
    lines.append(f"| VL prefill | {timing.vl_prefill:.1f} ms |")
    lines.append(f"| VL per-token | {timing.vl_per_token:.1f} ms |")
    lines.append(f"| Adapter latency | {timing.adapter_latency:.1f} ms |")
    lines.append(f"| L5 draft/token | {timing.l5_draft_per_token:.1f} ms |")
    lines.append(f"")

    report = "\n".join(lines)

    report_path = output_path / f"e2e_sd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved: {report_path}")

    return report


# =============================================================================
# Data Loading Helpers
# =============================================================================

def load_cached_hidden_states(data_path: str, device: str = 'cuda'):
    """Load hidden states from chunked or single file format."""
    data_path = Path(data_path)

    if data_path.is_dir():
        # Chunked format
        index_path = data_path / 'index.json'
        if not index_path.exists():
            raise ValueError(f"No index.json found in {data_path}")

        with open(index_path) as f:
            index = json.load(f)

        chunks_dir = data_path / 'chunks'
        all_egpt = []
        all_vl = []
        all_lens = []

        print(f"Loading {len(index['chunks'])} chunks...")
        for chunk_info in tqdm(index['chunks']):
            chunk_path = chunks_dir / chunk_info['path']
            chunk = torch.load(chunk_path, map_location='cpu')
            all_egpt.append(chunk['egpt_hidden'])
            all_vl.append(chunk['vl_hidden'])
            all_lens.append(chunk['seq_lens'])

        # Pad to same max_seq before concatenation
        max_seq = max(t.shape[1] for t in all_egpt)
        hidden_dim = all_egpt[0].shape[2]

        padded_egpt = []
        padded_vl = []
        for e, v in zip(all_egpt, all_vl):
            n, s, h = e.shape
            if s < max_seq:
                pad_e = torch.zeros(n, max_seq, h)
                pad_e[:, :s, :] = e
                padded_egpt.append(pad_e)
                pad_v = torch.zeros(n, max_seq, h)
                pad_v[:, :s, :] = v
                padded_vl.append(pad_v)
            else:
                padded_egpt.append(e)
                padded_vl.append(v)

        egpt_hidden = torch.cat(padded_egpt, dim=0).to(device).float()
        vl_hidden = torch.cat(padded_vl, dim=0).to(device).float()
        seq_lens = torch.cat(all_lens, dim=0)
    else:
        # Single file format
        data = torch.load(data_path, map_location='cpu')
        egpt_hidden = data['egpt_hidden'].to(device).float()
        vl_hidden = data['vl_hidden'].to(device).float()
        seq_lens = data['seq_lens']

    # Create mask
    batch_size, max_seq = egpt_hidden.shape[:2]
    mask = torch.zeros(batch_size, max_seq, device=device)
    for i, seq_len in enumerate(seq_lens):
        mask[i, :seq_len] = 1

    return egpt_hidden, vl_hidden, mask


def detect_adapter_level(checkpoint_path: str) -> int:
    """Detect adapter level from checkpoint or path."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    adapter_type = checkpoint.get('adapter_type', 'L1_Bottleneck')

    level_map = {
        'L1_Bottleneck': 1,
        'L2_MultiLayerBottleneck': 2,
        'L3_WideBottleneck': 3,
        'L4_Attention': 4,
        'L5_EAGLE': 5,
    }
    level = level_map.get(adapter_type, 1)

    # Also try path-based detection
    path_str = str(checkpoint_path)
    for lv in range(5, 0, -1):
        if f'/L{lv}/' in path_str or f'/L{lv}_' in path_str:
            level = lv
            break

    return level


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="End-to-End Speculative Decoding Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live mode
  python evaluate_e2e_sd.py \\
      --adapter_checkpoint ./tasks/L1/.../best_model.pt \\
      --dataset_dir ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s

  # Cached mode (faster, uses pre-extracted hidden states)
  python evaluate_e2e_sd.py \\
      --mode cached \\
      --adapter_checkpoint ./tasks/L1/.../best_model.pt \\
      --cached_hidden_states /mnt/hdd/data/egpt/hidden_states/chunked_test_1s_4bit
        """,
    )

    parser.add_argument('--mode', type=str, default='cached', choices=['live', 'cached'],
                        help='Evaluation mode: live (loads both models) or cached (uses pre-extracted)')
    parser.add_argument('--adapter_checkpoint', type=str, required=True,
                        help='Path to trained adapter checkpoint (.pt)')
    parser.add_argument('--adapter_level', type=int, default=None,
                        help='Adapter level (1-5). Auto-detected if not specified.')

    # Live mode args
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_test/my_egpt_dsec_seq_1s',
                        help='Dataset directory (live mode)')
    parser.add_argument('--questions_file', type=str,
                        default='./feasible/token_alignment/top50_questions.json',
                        help='Questions file (live mode)')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Max samples to evaluate (live mode)')
    parser.add_argument('--max_questions', type=int, default=10,
                        help='Max questions per sample (live mode)')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='Max tokens to generate (live mode)')

    # Cached mode args
    parser.add_argument('--cached_hidden_states', type=str,
                        default='/mnt/hdd/data/egpt/hidden_states/chunked_test_1s_4bit',
                        help='Path to cached hidden states (cached mode)')
    parser.add_argument('--max_cached_samples', type=int, default=-1,
                        help='Limit cached samples (-1 for all)')

    # SD config
    parser.add_argument('--gamma_decode', type=int, default=5,
                        help='Draft tokens per decode iteration (L5)')
    parser.add_argument('--num_output_tokens', type=int, default=50,
                        help='Assumed output length for speedup estimation')

    # Timing overrides
    parser.add_argument('--egpt_prefill_ms', type=float, default=130.0)
    parser.add_argument('--egpt_per_token_ms', type=float, default=25.0)
    parser.add_argument('--vl_prefill_ms', type=float, default=310.0)
    parser.add_argument('--vl_per_token_ms', type=float, default=25.0)
    parser.add_argument('--adapter_latency_ms', type=float, default=1.5)
    parser.add_argument('--l5_draft_per_token_ms', type=float, default=3.0)

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: adapter checkpoint dir)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for cached mode processing')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # --- Load adapter ---
    print(f"\nLoading adapter: {args.adapter_checkpoint}")
    adapter, checkpoint = load_any_adapter(args.adapter_checkpoint, device)
    adapter = adapter.to(device)
    adapter.eval()
    print(f"  Type: {type(adapter).__name__}")
    print(f"  Parameters: {adapter.get_num_parameters():,}")

    # Detect adapter level
    if args.adapter_level is not None:
        adapter_level = args.adapter_level
    else:
        adapter_level = detect_adapter_level(args.adapter_checkpoint)
    print(f"  Level: L{adapter_level}")

    # --- Timing config ---
    timing = TimingConfig(
        egpt_prefill=args.egpt_prefill_ms,
        egpt_per_token=args.egpt_per_token_ms,
        vl_prefill=args.vl_prefill_ms,
        vl_per_token=args.vl_per_token_ms,
        adapter_latency=args.adapter_latency_ms,
        l5_draft_per_token=args.l5_draft_per_token_ms,
    )

    # --- Output dir ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.adapter_checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Mode: CACHED
    # ==================================================================
    if args.mode == 'cached':
        print(f"\n--- Cached Mode ---")

        # Load cached hidden states
        print(f"Loading cached hidden states: {args.cached_hidden_states}")
        egpt_hidden, vl_hidden, mask = load_cached_hidden_states(
            args.cached_hidden_states, device
        )

        if args.max_cached_samples > 0 and args.max_cached_samples < len(egpt_hidden):
            egpt_hidden = egpt_hidden[:args.max_cached_samples]
            vl_hidden = vl_hidden[:args.max_cached_samples]
            mask = mask[:args.max_cached_samples]

        num_samples = len(egpt_hidden)
        print(f"  Samples: {num_samples}")
        print(f"  Max seq len: {egpt_hidden.shape[1]}")
        print(f"  Hidden dim: {egpt_hidden.shape[-1]}")

        # A. Hidden-state metrics
        print("\nComputing hidden-state metrics...")
        with torch.no_grad():
            # Process in batches
            if num_samples > args.batch_size:
                aligned_list = []
                for i in tqdm(range(0, num_samples, args.batch_size), desc="Adapter forward"):
                    batch = egpt_hidden[i:i+args.batch_size]
                    aligned_list.append(adapter(batch))
                aligned_hidden = torch.cat(aligned_list, dim=0)
            else:
                aligned_hidden = adapter(egpt_hidden)

        hidden_metrics = compute_hidden_state_metrics(aligned_hidden, vl_hidden, mask)
        print(f"  cos_sim_mean: {hidden_metrics['cos_sim_mean']:.4f}")
        print(f"  Accept@0.90: {hidden_metrics.get('accept_90', 0):.2%}")
        print(f"  Consecutive@0.90: {hidden_metrics.get('consecutive_mean_90', 0):.2f}")

        # B. Token-level metrics (need VL LM head)
        print("\nLoading Video-LLaVA for LM head projection...")
        models = load_models(load_vl_only=True)
        vl_lm_head = models['vl_lm_head']

        print("Computing token-level metrics...")
        # Move LM head weights to same device, process in batches
        token_metrics_list = []
        batch_sz = min(args.batch_size, 32)  # Smaller batches for LM head (large vocab)

        for i in tqdm(range(0, num_samples, batch_sz), desc="Token metrics"):
            end = min(i + batch_sz, num_samples)
            batch_egpt = egpt_hidden[i:end]
            batch_vl = vl_hidden[i:end]
            batch_mask = mask[i:end]

            batch_metrics = compute_token_level_metrics(
                batch_egpt, batch_vl, adapter, vl_lm_head, batch_mask,
                gamma_decode=args.gamma_decode,
            )
            token_metrics_list.append(batch_metrics)

        # Aggregate token metrics (weighted average by batch size)
        token_metrics = {}
        total_samples_processed = 0
        for bm in token_metrics_list:
            weight = 1  # Each batch has equal weight for simplicity
            for k, v in bm.items():
                if isinstance(v, (int, float)):
                    if k not in token_metrics:
                        token_metrics[k] = 0.0
                    token_metrics[k] += v * weight
                elif isinstance(v, list):
                    if k not in token_metrics:
                        token_metrics[k] = [0.0] * len(v)
                    for j, val in enumerate(v):
                        if j < len(token_metrics[k]):
                            token_metrics[k][j] += val * weight
            total_samples_processed += weight

        # Average
        for k, v in token_metrics.items():
            if isinstance(v, (int, float)):
                token_metrics[k] = v / total_samples_processed
            elif isinstance(v, list):
                token_metrics[k] = [x / total_samples_processed for x in v]

        print(f"  Token match rate: {token_metrics['token_match_rate']:.2%}")
        print(f"  Top-5 match rate: {token_metrics['top5_match_rate']:.2%}")
        print(f"  Consecutive token matches: {token_metrics['consecutive_token_mean']:.2f}")

        # Update timing from live measurements if available (here we use defaults)
        live_timing = timing

    # ==================================================================
    # Mode: LIVE
    # ==================================================================
    elif args.mode == 'live':
        print(f"\n--- Live Mode ---")

        # Load both models
        models = load_models(load_vl_only=False)
        vl_lm_head = models['vl_lm_head']

        dataset_dir = Path(args.dataset_dir)
        json_path = dataset_dir / "EventGPT_Instruction_Subset.json"
        with open(json_path) as f:
            dataset = json.load(f)

        if args.max_samples > 0:
            dataset = dataset[:args.max_samples]

        # Load questions
        questions_file = Path(args.questions_file)
        if questions_file.exists():
            with open(questions_file) as f:
                questions_data = json.load(f)
            questions = [q['question'] for q in questions_data][:args.max_questions]
        else:
            questions = ["What are the key elements in this scene?"]

        print(f"  Dataset: {len(dataset)} samples x {len(questions)} questions")

        # Extract hidden states with timing
        all_egpt_hidden = []
        all_vl_hidden = []
        all_timings = []
        errors = 0

        total = len(dataset) * len(questions)
        with tqdm(total=total, desc="Extracting") as pbar:
            for sample in dataset:
                for query in questions:
                    result = extract_with_timing(
                        models, sample, dataset_dir, query,
                        max_new_tokens=args.max_new_tokens,
                    )
                    if result is not None:
                        all_egpt_hidden.append(result['egpt_hidden'])
                        all_vl_hidden.append(result['vl_hidden'])
                        all_timings.append(result['timing'])
                    else:
                        errors += 1
                    pbar.update(1)
                    pbar.set_postfix(extracted=len(all_egpt_hidden), errors=errors)

        if not all_egpt_hidden:
            print("No samples extracted successfully!")
            return

        num_samples = len(all_egpt_hidden)
        print(f"\n  Extracted: {num_samples} samples, Errors: {errors}")

        # Stack with padding
        max_seq = max(h.shape[0] for h in all_egpt_hidden)
        hidden_dim = all_egpt_hidden[0].shape[1]

        egpt_hidden = torch.zeros(num_samples, max_seq, hidden_dim, device=device)
        vl_hidden = torch.zeros(num_samples, max_seq, hidden_dim, device=device)
        mask = torch.zeros(num_samples, max_seq, device=device)

        for i, (eh, vh) in enumerate(zip(all_egpt_hidden, all_vl_hidden)):
            seq_len = eh.shape[0]
            egpt_hidden[i, :seq_len] = eh.to(device).float()
            vl_hidden[i, :seq_len] = vh.to(device).float()
            mask[i, :seq_len] = 1

        # Compute average timing
        avg_timing = {}
        for key in all_timings[0]:
            vals = [t[key] for t in all_timings]
            avg_timing[key] = sum(vals) / len(vals)

        print(f"  Avg EGPT prefill: {avg_timing['egpt_prefill_ms']:.1f} ms")
        print(f"  Avg EGPT per-token: {avg_timing['egpt_per_token_ms']:.1f} ms")
        print(f"  Avg VL total: {avg_timing['vl_total_ms']:.1f} ms")
        print(f"  Avg VL per-token: {avg_timing['vl_per_token_ms']:.1f} ms")

        # Use live timing
        live_timing = TimingConfig(
            egpt_prefill=avg_timing['egpt_prefill_ms'],
            egpt_per_token=avg_timing['egpt_per_token_ms'],
            vl_prefill=avg_timing.get('vl_prefill_ms', timing.vl_prefill),
            vl_per_token=avg_timing['vl_per_token_ms'],
            adapter_latency=timing.adapter_latency,
            l5_draft_per_token=timing.l5_draft_per_token,
        )

        # A. Hidden-state metrics
        print("\nComputing hidden-state metrics...")
        with torch.no_grad():
            aligned_hidden = adapter(egpt_hidden)

        hidden_metrics = compute_hidden_state_metrics(aligned_hidden, vl_hidden, mask)
        print(f"  cos_sim_mean: {hidden_metrics['cos_sim_mean']:.4f}")
        print(f"  Accept@0.90: {hidden_metrics.get('accept_90', 0):.2%}")

        # B. Token-level metrics
        print("Computing token-level metrics...")
        token_metrics = compute_token_level_metrics(
            egpt_hidden, vl_hidden, adapter, vl_lm_head, mask,
            gamma_decode=args.gamma_decode,
        )
        print(f"  Token match rate: {token_metrics['token_match_rate']:.2%}")
        print(f"  Top-5 match rate: {token_metrics['top5_match_rate']:.2%}")

    # ==================================================================
    # C. Two-Phase SD Speedup (both modes)
    # ==================================================================
    print("\nSimulating two-phase speculative decoding...")
    sd_result = simulate_two_phase_sd(
        timing=live_timing,
        hidden_metrics=hidden_metrics,
        token_metrics=token_metrics,
        adapter_level=adapter_level,
        num_output_tokens=args.num_output_tokens,
        gamma_decode=args.gamma_decode,
    )

    e2e = sd_result['e2e']
    p1 = sd_result['phase1_prefill_hiding']
    p2 = sd_result['phase2_decode']

    print(f"\n{'='*60}")
    print(f"  Phase 1 (Prefill Hiding):")
    print(f"    gamma_prefill = {p1['gamma_prefill']} tokens")
    print(f"    Accepted = {p1['prefill_accepted_tokens']:.1f} tokens")
    print(f"    Wall-clock = {p1['prefill_wall_clock_ms']:.1f} ms")
    print(f"  Phase 2 (Decode):")
    print(f"    L{adapter_level} {'EAGLE' if p2['is_l5'] else 'no decode accel'}")
    print(f"    Remaining = {p2['tokens_remaining']:.0f} tokens")
    print(f"    Wall-clock = {p2['decode_wall_clock_ms']:.1f} ms")
    print(f"  E2E:")
    print(f"    Baseline = {e2e['baseline_time_ms']:.1f} ms")
    print(f"    SD total = {e2e['sd_total_time_ms']:.1f} ms")
    print(f"    Speedup  = {e2e['speedup']:.2f}x")
    print(f"{'='*60}")

    # ==================================================================
    # Save results
    # ==================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # JSON results
    results = {
        'timestamp': datetime.now().isoformat(),
        'mode': args.mode,
        'adapter_checkpoint': args.adapter_checkpoint,
        'adapter_level': adapter_level,
        'num_samples': num_samples,
        'hidden_state_metrics': hidden_metrics,
        'token_level_metrics': token_metrics,
        'sd_simulation': sd_result,
        'timing_config': asdict(live_timing),
    }

    # Ensure JSON serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    results = make_serializable(results)

    json_path = output_dir / f"e2e_sd_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results: {json_path}")

    # Markdown report
    generate_report(
        hidden_metrics=hidden_metrics,
        token_metrics=token_metrics,
        sd_result=sd_result,
        timing=live_timing,
        adapter_level=adapter_level,
        adapter_path=args.adapter_checkpoint,
        num_samples=num_samples,
        output_path=output_dir,
    )

    print(f"\nAll results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    main()
