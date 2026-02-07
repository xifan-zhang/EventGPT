#!/usr/bin/env python3
"""
Compute Prefill Stage Speedup and End-to-End Speedup

This script measures:
1. Prefill stage speedup: EGPT prefill vs VL prefill
2. End-to-end speedup: With SD verification vs VL baseline

Key insight: The speedup comes from:
- Prefill: EGPT is 3.4x faster (112ms vs 383ms)
- Free drafts: ~26 tokens generated during VL prefill overlap
- Accepted tokens: Save VL decode time for each accepted token
"""

import sys
import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET_DIR = './data/my_egpt_dsec_test/my_egpt_dsec_seq_1s'


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
    except:
        return None


def measure_prefill_speedup(samples, max_samples=10):
    """Measure prefill times for both models separately."""
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from common.common import load_image

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    results = {'egpt_prefill': [], 'vl_prefill': []}

    # === EGPT Prefill ===
    print("\n[1/2] Measuring EventGPT prefill...")
    from model.EventChatModel import EventChatModel, get_spatio_temporal_features
    from dataset.conversation import prepare_event_prompt
    from dataset.constants import EVENT_TOKEN_INDEX
    from common.common import tokenizer_event_token

    egpt_model = EventChatModel.from_pretrained(
        './checkpoints/EventGPT-7b',
        torch_dtype=torch.bfloat16,
        device_map='auto',
        quantization_config=bnb_config,
    )
    egpt_model.eval()
    egpt_tokenizer = AutoTokenizer.from_pretrained('./checkpoints/EventGPT-7b', use_fast=True)
    egpt_processor = egpt_model.get_visual_tower().event_processor

    query = "What is happening in this scene?"

    for sample in tqdm(samples[:max_samples], desc='EGPT prefill'):
        event_paths = sample.get('event_image', [])
        if not event_paths:
            continue

        img_path = os.path.join(DATASET_DIR, 'event_image', event_paths[0])
        try:
            img = load_image(img_path)
            img_array = np.array(img)
            event_image_size = list(img_array.shape[:2])
            event = egpt_processor(img_array, return_tensors='pt')['pixel_values'][0]
            event = event.to('cuda', dtype=torch.bfloat16)
        except:
            continue

        torch.cuda.synchronize()
        start = time.time()

        prompt = prepare_event_prompt(query, 'eventgpt_v1')
        input_ids = tokenizer_event_token(prompt, egpt_tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cuda')

        with torch.inference_mode():
            feature = egpt_model.visval_encode(event.unsqueeze(0))
            feature = egpt_model.get_model().feature_adaptor(feature)
            feature = feature.squeeze(0)
            event_features = get_spatio_temporal_features([feature])
            event_features = event_features.unsqueeze(0)

            _, position_ids, attention_mask, _, inputs_embeds, _ = egpt_model.prepare_inputs_labels_for_multimodal(
                input_ids, None, torch.ones_like(input_ids, dtype=torch.bool), None, None,
                event_tensors=None, event_image_sizes=event_image_size, event_features=event_features,
            )

            if attention_mask is None:
                attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.bool, device='cuda')
            if position_ids is None:
                position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device='cuda').unsqueeze(0)

            outputs = egpt_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                       position_ids=position_ids, past_key_values=None, use_cache=True)
            _ = egpt_model.lm_head(outputs.last_hidden_state[:, -1:, :])

        torch.cuda.synchronize()
        results['egpt_prefill'].append(time.time() - start)

    del egpt_model, egpt_tokenizer, egpt_processor
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # === VL Prefill ===
    print("\n[2/2] Measuring Video-LLaVA prefill...")
    from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

    vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
        'LanguageBind/Video-LLaVA-7B-hf',
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=bnb_config,
    )
    vl_model.eval()
    vl_processor = VideoLlavaProcessor.from_pretrained('LanguageBind/Video-LLaVA-7B-hf')

    for sample in tqdm(samples[:max_samples], desc='VL prefill'):
        video_data = sample.get('video_data')
        if not video_data:
            continue

        video_path = os.path.join(DATASET_DIR, 'mp4', video_data + '.mp4')
        frames = load_video_frames(video_path)
        if frames is None:
            continue

        torch.cuda.synchronize()
        start = time.time()

        prompt = f'USER: <video>\n{query}\nASSISTANT:'
        inputs = vl_processor(text=prompt, videos=frames, return_tensors='pt')
        input_ids = inputs['input_ids'].to('cuda')
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda')
        pixel_values_videos = inputs.get('pixel_values_videos').to('cuda')

        with torch.inference_mode():
            outputs = vl_model(input_ids=input_ids, attention_mask=attention_mask,
                              pixel_values_videos=pixel_values_videos,
                              past_key_values=None, use_cache=True)

        torch.cuda.synchronize()
        results['vl_prefill'].append(time.time() - start)

    del vl_model, vl_processor
    torch.cuda.empty_cache()
    gc.collect()

    return results


def compute_theoretical_speedup(egpt_prefill_ms, vl_prefill_ms, vl_token_time_ms=14.5,
                                 max_new_tokens=50, acceptance_rate=0.279):
    """
    Compute theoretical speedup from parallel prefill + SD.

    Timeline:
    - EGPT prefill: 0 → egpt_prefill_ms
    - EGPT draft:   egpt_prefill_ms → vl_prefill_ms (overlap window)
    - VL prefill:   0 → vl_prefill_ms
    - VL verify:    vl_prefill_ms → vl_prefill_ms + 50ms
    - VL decode:    Continue AR for remaining tokens
    """

    # Overlap window where EGPT generates free drafts
    overlap_ms = vl_prefill_ms - egpt_prefill_ms

    # Free draft tokens (at ~10ms/token for EGPT)
    egpt_token_time_ms = 10.0
    free_drafts = int(overlap_ms / egpt_token_time_ms)

    # Accepted tokens
    accepted = int(free_drafts * acceptance_rate)

    # Time saved by accepting tokens
    time_saved_ms = accepted * vl_token_time_ms

    # VL baseline (sequential)
    vl_baseline_ms = vl_prefill_ms + max_new_tokens * vl_token_time_ms

    # With SD (parallel prefill + batch verify + remaining AR)
    verify_time_ms = 50  # Single forward pass for all drafts
    remaining_tokens = max_new_tokens - accepted
    sd_time_ms = vl_prefill_ms + verify_time_ms + remaining_tokens * vl_token_time_ms

    return {
        'egpt_prefill_ms': egpt_prefill_ms,
        'vl_prefill_ms': vl_prefill_ms,
        'overlap_ms': overlap_ms,
        'free_drafts': free_drafts,
        'accepted': accepted,
        'time_saved_ms': time_saved_ms,
        'vl_baseline_ms': vl_baseline_ms,
        'sd_time_ms': sd_time_ms,
        'prefill_speedup': vl_prefill_ms / egpt_prefill_ms,
        'e2e_speedup': vl_baseline_ms / sd_time_ms,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=10)
    parser.add_argument('--acceptance_rate', type=float, default=0.279)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--measure', action='store_true', help='Actually run measurements')
    args = parser.parse_args()

    print('='*70)
    print('PREFILL & END-TO-END SPEEDUP ANALYSIS')
    print('='*70)

    if args.measure:
        # Load dataset
        with open(os.path.join(DATASET_DIR, 'EventGPT_Instruction_Subset.json')) as f:
            samples = json.load(f)

        # Measure prefill times
        results = measure_prefill_speedup(samples, args.max_samples)

        egpt_prefill_ms = np.mean(results['egpt_prefill']) * 1000
        vl_prefill_ms = np.mean(results['vl_prefill']) * 1000
    else:
        # Use benchmark data
        egpt_prefill_ms = 112.5
        vl_prefill_ms = 382.8

    # Compute speedups
    analysis = compute_theoretical_speedup(
        egpt_prefill_ms=egpt_prefill_ms,
        vl_prefill_ms=vl_prefill_ms,
        acceptance_rate=args.acceptance_rate,
        max_new_tokens=args.max_new_tokens,
    )

    print(f'\n{"─"*70}')
    print('PREFILL STAGE SPEEDUP')
    print(f'{"─"*70}')
    print(f'  EventGPT prefill:    {analysis["egpt_prefill_ms"]:>8.1f} ms')
    print(f'  Video-LLaVA prefill: {analysis["vl_prefill_ms"]:>8.1f} ms')
    print(f'  Prefill Speedup:     {analysis["prefill_speedup"]:>8.2f}x')

    print(f'\n{"─"*70}')
    print('PARALLEL PREFILL BENEFIT')
    print(f'{"─"*70}')
    print(f'  Overlap window:      {analysis["overlap_ms"]:>8.1f} ms')
    print(f'  Free draft tokens:   {analysis["free_drafts"]:>8} tokens')
    print(f'  Acceptance rate:     {args.acceptance_rate*100:>7.1f}%')
    print(f'  Accepted tokens:     {analysis["accepted"]:>8} tokens')
    print(f'  Time saved:          {analysis["time_saved_ms"]:>8.1f} ms')

    print(f'\n{"─"*70}')
    print('END-TO-END SPEEDUP')
    print(f'{"─"*70}')
    print(f'  VL baseline:         {analysis["vl_baseline_ms"]:>8.1f} ms')
    print(f'  With SD:             {analysis["sd_time_ms"]:>8.1f} ms')
    print(f'  End-to-End Speedup:  {analysis["e2e_speedup"]:>8.2f}x')

    print(f'\n{"─"*70}')
    print('SPEEDUP AT DIFFERENT ACCEPTANCE RATES')
    print(f'{"─"*70}')

    for ar in [0.0, 0.10, 0.279, 0.50, 0.70, 1.0]:
        a = compute_theoretical_speedup(
            egpt_prefill_ms=analysis['egpt_prefill_ms'],
            vl_prefill_ms=analysis['vl_prefill_ms'],
            acceptance_rate=ar,
            max_new_tokens=args.max_new_tokens,
        )
        label = "(baseline)" if ar == 0 else "(current)" if ar == 0.279 else ""
        print(f'  α={ar*100:>5.1f}%: {a["e2e_speedup"]:.2f}x  ({a["accepted"]:>2} tokens accepted) {label}')

    print('='*70)

    # Summary
    print(f'\n*** SUMMARY ***')
    print(f'  Prefill Stage Speedup:  {analysis["prefill_speedup"]:.1f}x (EGPT is {analysis["prefill_speedup"]:.1f}x faster)')
    print(f'  End-to-End Speedup:     {analysis["e2e_speedup"]:.2f}x (with {args.acceptance_rate*100:.0f}% acceptance)')
    print(f'  Free Tokens:            {analysis["free_drafts"]} tokens at ZERO cost')


if __name__ == '__main__':
    main()
