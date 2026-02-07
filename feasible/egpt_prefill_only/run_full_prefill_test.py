#!/usr/bin/env python3
"""
Full Prefill Acceleration Test
- Full test set (1100 samples)
- Top 10 questions
- Sequential measurement, compute parallel benefit
"""

import sys
import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Fix protobuf
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Config
DATASET_DIR = './data/my_egpt_dsec_test/my_egpt_dsec_seq_1s'
OUTPUT_DIR = './feasible/token_alignment/task/starred'
MAX_QUESTIONS = 10
MAX_NEW_TOKENS = 50

print('='*70)
print('FULL PREFILL ACCELERATION TEST')
print(f'Dataset: {DATASET_DIR}')
print(f'Questions: Top {MAX_QUESTIONS}')
print(f'Max tokens: {MAX_NEW_TOKENS}')
print('='*70)

# Load dataset
with open(os.path.join(DATASET_DIR, 'EventGPT_Instruction_Subset.json')) as f:
    dataset = json.load(f)

# Load questions
with open('./feasible/token_alignment/top50_questions.json') as f:
    questions = [q['question'] for q in json.load(f)[:MAX_QUESTIONS]]

print(f'\nSamples: {len(dataset)}')
print(f'Questions: {len(questions)}')
print(f'Total inferences: {len(dataset) * len(questions)}')

# Video loader
import av
def load_video_frames(video_path, num_frames=8):
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

from common.common import load_image
from transformers import AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ============ PHASE 1: EVENTGPT ============
print('\n' + '='*70)
print('[PHASE 1] EventGPT: Prefill + Decode')
print('='*70)

print('\nLoading EventGPT (4-bit)...')
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

print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

egpt_results = []
total_egpt = len(dataset) * len(questions)

print(f'\nRunning {total_egpt} inferences...')

pbar = tqdm(total=total_egpt, desc='EventGPT')
for sample_idx, sample in enumerate(dataset):
    event_paths = sample.get('event_image', [])
    if not event_paths:
        pbar.update(len(questions))
        continue

    img_path = os.path.join(DATASET_DIR, 'event_image', event_paths[0])
    try:
        img = load_image(img_path)
        img_array = np.array(img)
        event_image_size = list(img_array.shape[:2])
        event = egpt_processor(img_array, return_tensors='pt')['pixel_values'][0]
        event = event.to('cuda', dtype=torch.bfloat16)
    except:
        pbar.update(len(questions))
        continue

    for q_idx, query in enumerate(questions):
        try:
            # Prefill
            torch.cuda.synchronize()
            prefill_start = time.time()

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
                logits = egpt_model.lm_head(outputs.last_hidden_state[:, -1:, :])

            torch.cuda.synchronize()
            prefill_time = time.time() - prefill_start

            # Decode
            torch.cuda.synchronize()
            decode_start = time.time()
            output_tokens = []

            with torch.inference_mode():
                cur_pos = inputs_embeds.shape[1]
                kv_cache = outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                output_tokens.append(next_token.item())

                for _ in range(MAX_NEW_TOKENS - 1):
                    cur_embed = egpt_model.get_model().embed_tokens(next_token)
                    new_attn = torch.ones((1, cur_pos + 1), dtype=torch.bool, device='cuda')
                    outputs = egpt_model.model(inputs_embeds=cur_embed, attention_mask=new_attn,
                                              position_ids=torch.tensor([[cur_pos]], device='cuda'),
                                              past_key_values=kv_cache, use_cache=True)
                    logits = egpt_model.lm_head(outputs.last_hidden_state[:, -1:, :])
                    kv_cache = outputs.past_key_values
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    output_tokens.append(next_token.item())
                    cur_pos += 1
                    if next_token.item() == egpt_tokenizer.eos_token_id:
                        break

            torch.cuda.synchronize()
            decode_time = time.time() - decode_start

            egpt_results.append({
                'sample_idx': sample_idx,
                'question_idx': q_idx,
                'prefill_time': prefill_time,
                'decode_time': decode_time,
                'num_tokens': len(output_tokens),
                'tokens': output_tokens,
            })

        except Exception as e:
            pass

        pbar.update(1)

pbar.close()

# Save EGPT results
egpt_summary = {
    'n_samples': len(egpt_results),
    'prefill_avg': np.mean([r['prefill_time'] for r in egpt_results]),
    'prefill_std': np.std([r['prefill_time'] for r in egpt_results]),
    'decode_avg': np.mean([r['decode_time'] for r in egpt_results]),
    'decode_std': np.std([r['decode_time'] for r in egpt_results]),
    'tokens_avg': np.mean([r['num_tokens'] for r in egpt_results]),
}

print(f'\nEventGPT Summary:')
print(f'  Samples: {egpt_summary["n_samples"]}')
print(f'  Prefill: {egpt_summary["prefill_avg"]*1000:.1f} ± {egpt_summary["prefill_std"]*1000:.1f} ms')
print(f'  Decode: {egpt_summary["decode_avg"]*1000:.1f} ± {egpt_summary["decode_std"]*1000:.1f} ms')
print(f'  Tokens: {egpt_summary["tokens_avg"]:.1f} avg')

# Free memory
del egpt_model, egpt_tokenizer, egpt_processor
torch.cuda.empty_cache()
import gc
gc.collect()

# ============ PHASE 2: VIDEO-LLAVA ============
print('\n' + '='*70)
print('[PHASE 2] Video-LLaVA: Prefill + Decode')
print('='*70)

print('\nLoading Video-LLaVA (4-bit)...')
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
    'LanguageBind/Video-LLaVA-7B-hf',
    torch_dtype=torch.float16,
    device_map='auto',
    quantization_config=bnb_config,
)
vl_model.eval()
vl_processor = VideoLlavaProcessor.from_pretrained('LanguageBind/Video-LLaVA-7B-hf')

print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

vl_results = []
total_vl = len(dataset) * len(questions)

print(f'\nRunning {total_vl} inferences...')

pbar = tqdm(total=total_vl, desc='Video-LLaVA')
for sample_idx, sample in enumerate(dataset):
    video_data = sample.get('video_data')
    if not video_data:
        pbar.update(len(questions))
        continue

    video_path = os.path.join(DATASET_DIR, 'mp4', video_data + '.mp4')
    frames = load_video_frames(video_path)
    if frames is None:
        pbar.update(len(questions))
        continue

    for q_idx, query in enumerate(questions):
        try:
            # Prefill
            torch.cuda.synchronize()
            prefill_start = time.time()

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
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            torch.cuda.synchronize()
            prefill_time = time.time() - prefill_start

            # Decode
            torch.cuda.synchronize()
            decode_start = time.time()
            output_tokens = [next_token.item()]

            with torch.inference_mode():
                past_kv = outputs.past_key_values
                cur_token = next_token
                attn_dtype = attention_mask.dtype if attention_mask is not None else torch.long
                cur_attn = torch.ones((1, past_kv[0][0].shape[2] + 1), dtype=attn_dtype, device='cuda')

                for _ in range(MAX_NEW_TOKENS - 1):
                    outputs = vl_model(input_ids=cur_token, attention_mask=cur_attn,
                                      past_key_values=past_kv, use_cache=True)
                    past_kv = outputs.past_key_values
                    cur_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                    output_tokens.append(cur_token.item())
                    cur_attn = torch.ones((1, cur_attn.shape[1] + 1), dtype=attn_dtype, device='cuda')
                    if cur_token.item() == vl_processor.tokenizer.eos_token_id:
                        break

            torch.cuda.synchronize()
            decode_time = time.time() - decode_start

            vl_results.append({
                'sample_idx': sample_idx,
                'question_idx': q_idx,
                'prefill_time': prefill_time,
                'decode_time': decode_time,
                'num_tokens': len(output_tokens),
                'tokens': output_tokens,
            })

        except Exception as e:
            pass

        pbar.update(1)

pbar.close()

# VL Summary
vl_summary = {
    'n_samples': len(vl_results),
    'prefill_avg': np.mean([r['prefill_time'] for r in vl_results]),
    'prefill_std': np.std([r['prefill_time'] for r in vl_results]),
    'decode_avg': np.mean([r['decode_time'] for r in vl_results]),
    'decode_std': np.std([r['decode_time'] for r in vl_results]),
    'tokens_avg': np.mean([r['num_tokens'] for r in vl_results]),
}

print(f'\nVideo-LLaVA Summary:')
print(f'  Samples: {vl_summary["n_samples"]}')
print(f'  Prefill: {vl_summary["prefill_avg"]*1000:.1f} ± {vl_summary["prefill_std"]*1000:.1f} ms')
print(f'  Decode: {vl_summary["decode_avg"]*1000:.1f} ± {vl_summary["decode_std"]*1000:.1f} ms')
print(f'  Tokens: {vl_summary["tokens_avg"]:.1f} avg')

# ============ ANALYSIS ============
print('\n' + '='*70)
print('PREFILL ACCELERATION ANALYSIS')
print('='*70)

egpt_prefill = egpt_summary['prefill_avg']
egpt_decode = egpt_summary['decode_avg']
vl_prefill = vl_summary['prefill_avg']
vl_decode = vl_summary['decode_avg']

overlap_window = max(0, vl_prefill - egpt_prefill)
egpt_tokens = egpt_summary['tokens_avg']
egpt_token_rate = egpt_tokens / egpt_decode
free_tokens = int(overlap_window * egpt_token_rate)

vl_tokens = vl_summary['tokens_avg']
vl_token_time = vl_decode / vl_tokens

vl_total = vl_prefill + vl_decode

# With SD at different acceptance rates
acceptance_rates = [0.0158, 0.279, 0.50, 0.70]

print(f'\nSamples: EGPT={egpt_summary["n_samples"]}, VL={vl_summary["n_samples"]}')
print(f'\n{"─"*50}')
print('TIMING COMPARISON:')
print(f'  EventGPT prefill:    {egpt_prefill*1000:>8.1f} ± {egpt_summary["prefill_std"]*1000:.1f} ms')
print(f'  EventGPT decode:     {egpt_decode*1000:>8.1f} ± {egpt_summary["decode_std"]*1000:.1f} ms')
print(f'  Video-LLaVA prefill: {vl_prefill*1000:>8.1f} ± {vl_summary["prefill_std"]*1000:.1f} ms')
print(f'  Video-LLaVA decode:  {vl_decode*1000:>8.1f} ± {vl_summary["decode_std"]*1000:.1f} ms')

print(f'\n{"─"*50}')
print('PARALLEL PREFILL BENEFIT:')
print(f'  Overlap window:      {overlap_window*1000:>8.1f} ms')
print(f'  Free draft tokens:   {free_tokens:>8} tokens')
print(f'  EGPT token rate:     {egpt_token_rate:>8.1f} tok/s')
print(f'  VL token time:       {vl_token_time*1000:>8.1f} ms/token')

print(f'\n{"─"*50}')
print('SPEEDUP AT DIFFERENT ACCEPTANCE RATES:')
print(f'  VL baseline:         {vl_total*1000:>8.1f} ms  (1.00x)')

for ar in acceptance_rates:
    accepted = int(free_tokens * ar)
    time_saved = accepted * vl_token_time
    parallel_time = vl_prefill + 0.05 + max(0, vl_decode - time_saved)
    speedup = vl_total / parallel_time
    label = "baseline" if ar < 0.02 else ("current" if ar < 0.30 else "target")
    print(f'  α={ar*100:>5.1f}% ({label:>8}): {parallel_time*1000:>8.1f} ms  ({speedup:.2f}x) [{accepted} tokens]')

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results = {
    'timestamp': timestamp,
    'config': {
        'dataset': DATASET_DIR,
        'n_questions': MAX_QUESTIONS,
        'max_tokens': MAX_NEW_TOKENS,
    },
    'eventgpt': egpt_summary,
    'videollava': vl_summary,
    'analysis': {
        'overlap_window_ms': overlap_window * 1000,
        'free_draft_tokens': free_tokens,
        'egpt_token_rate': egpt_token_rate,
        'vl_token_time_ms': vl_token_time * 1000,
        'vl_baseline_ms': vl_total * 1000,
    },
    'questions': questions,
}

output_path = os.path.join(OUTPUT_DIR, f'prefill_acceleration_full_{timestamp}.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n{"─"*50}')
print(f'Results saved to: {output_path}')
print('='*70)
