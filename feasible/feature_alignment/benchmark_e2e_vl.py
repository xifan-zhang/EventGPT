#!/usr/bin/env python3
"""
Video-LLaVA Standalone Benchmark: 3-Stage Timing + Hidden State Extraction
==========================================================================

Loads Video-LLaVA from scratch, processes raw MP4 files, and measures:
  - Stage 1: Vision Encoder timing
  - Stage 2: LLM Prefill timing
  - Stage 3: LLM Decode timing + hidden state extraction

Also extracts VL LM head weights for adapter evaluation.

Usage:
  conda run -n egpt python benchmark_e2e_vl.py \
    --dataset_dir ./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \
    --max_samples 50 --max_new_tokens 50
"""

import os
import sys
import json
import time
import gc
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ============================================================
# Vision Timing Hooks
# ============================================================
class VisionTimingHooks:
    """Forward hooks to measure Video-LLaVA vision encoder time separately."""

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


# ============================================================
# Video frame loader
# ============================================================
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


# ============================================================
# 3-Stage Benchmark + Hidden State Extraction
# ============================================================
def benchmark_videollava(model, processor, hooks, dataset_dir, samples,
                         questions, device, max_new_tokens):
    """
    Process raw MP4 files through Video-LLaVA.

    Returns:
        timing_results: list of per-sample timing dicts
        all_hidden: list of [seq_len, hidden_dim] tensors (CPU)
    """
    timing_results = []
    all_hidden = []
    pbar = tqdm(total=len(samples) * len(questions), desc="Video-LLaVA")

    for sample in samples:
        video_data = sample.get('video_data')
        if not video_data:
            pbar.update(len(questions))
            continue

        video_path = os.path.join(dataset_dir, 'mp4', video_data + '.mp4')
        frames = load_video_frames(video_path)
        if frames is None:
            pbar.update(len(questions))
            continue

        for query in questions:
            try:
                prompt = f'USER: <video>\n{query}\nASSISTANT:'
                inputs = processor(text=prompt, videos=frames, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                pixel_values_videos = inputs.get('pixel_values_videos')
                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.to(device)

                # ---- Stage 1+2: Vision + LLM Prefill ----
                hooks.reset()
                torch.cuda.synchronize()
                prefill_start = time.time()

                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos,
                        past_key_values=None, use_cache=True,
                        output_hidden_states=True)
                    next_token = torch.argmax(
                        outputs.logits[:, -1, :], dim=-1, keepdim=True)

                torch.cuda.synchronize()
                prefill_total = time.time() - prefill_start
                t1 = hooks.vision_time * 1000
                t2 = prefill_total * 1000 - t1

                # ---- Stage 3: LLM Decode + hidden state extraction ----
                torch.cuda.synchronize()
                t3_start = time.time()
                output_tokens = [next_token.item()]
                step_hidden = []

                # Prefill last hidden state
                if outputs.hidden_states is not None:
                    step_hidden.append(
                        outputs.hidden_states[-1][:, -1, :].cpu())

                with torch.inference_mode():
                    past_kv = outputs.past_key_values
                    cur_token = next_token
                    attn_len = past_kv[0][0].shape[2] + 1
                    attn_dtype = (attention_mask.dtype
                                  if attention_mask is not None else torch.long)

                    for _ in range(max_new_tokens - 1):
                        cur_attn = torch.ones(
                            (1, attn_len), dtype=attn_dtype, device=device)
                        outputs = model(
                            input_ids=cur_token, attention_mask=cur_attn,
                            past_key_values=past_kv, use_cache=True,
                            output_hidden_states=True)
                        past_kv = outputs.past_key_values
                        cur_token = torch.argmax(
                            outputs.logits[:, -1, :], dim=-1, keepdim=True)
                        output_tokens.append(cur_token.item())
                        if outputs.hidden_states is not None:
                            step_hidden.append(
                                outputs.hidden_states[-1][:, -1, :].cpu())
                        attn_len += 1
                        if cur_token.item() == processor.tokenizer.eos_token_id:
                            break

                torch.cuda.synchronize()
                t3 = (time.time() - t3_start) * 1000

                h = (torch.cat(step_hidden, dim=0) if step_hidden
                     else torch.zeros(0, 4096))
                all_hidden.append(h)

                # Decode tokens to text
                text = processor.tokenizer.decode(
                    output_tokens, skip_special_tokens=True)

                timing_results.append({
                    'vision_ms': t1,
                    'prefill_ms': t2,
                    'decode_ms': t3,
                    'total_ms': t1 + t2 + t3,
                    'num_tokens': len(output_tokens),
                    'text': text,
                    'query': query,
                })

            except Exception as e:
                tqdm.write(f"  Error: {e}")
            pbar.update(1)

    pbar.close()
    return timing_results, all_hidden


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Video-LLaVA Standalone Benchmark (3-stage timing + hidden extraction)")
    parser.add_argument('--dataset_dir', type=str,
                        default='./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s')
    parser.add_argument('--vl_model', type=str,
                        default='LanguageBind/Video-LLaVA-7B-hf')
    parser.add_argument('--max_samples', type=int, default=50)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--questions_file', type=str, default=None)
    parser.add_argument('--use_4bit', action='store_true', default=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_hidden', action='store_true', default=False,
                        help='Save extracted hidden states to disk')
    args = parser.parse_args()

    device = 'cuda'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent / 'tasks')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset_json = os.path.join(
        args.dataset_dir, 'EventGPT_Instruction_Subset.json')
    with open(dataset_json) as f:
        dataset = json.load(f)
    if args.max_samples:
        dataset = dataset[:args.max_samples]

    # Load questions
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

    print("=" * 70)
    print("VIDEO-LLAVA STANDALONE BENCHMARK")
    print(f"Date: {timestamp}")
    print(f"Samples: {len(dataset)}, Questions: {len(questions)}")
    print(f"Max tokens: {args.max_new_tokens}, Warmup: {args.warmup}")
    print("=" * 70)

    from transformers import (
        BitsAndBytesConfig,
        VideoLlavaForConditionalGeneration,
        VideoLlavaProcessor,
    )

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load model
    print("\nLoading Video-LLaVA...")
    vl_load_kwargs = dict(torch_dtype=torch.float16, device_map='auto')
    if bnb_config:
        vl_load_kwargs['quantization_config'] = bnb_config

    vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
        args.vl_model, **vl_load_kwargs)
    vl_model.eval()
    vl_processor = VideoLlavaProcessor.from_pretrained(args.vl_model)
    print(f"  GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Register vision timing hooks
    hooks = VisionTimingHooks(vl_model)
    hooks.register()

    # Run benchmark
    timing_results, all_hidden = benchmark_videollava(
        vl_model, vl_processor, hooks,
        args.dataset_dir, dataset, questions, device, args.max_new_tokens
    )

    # Extract LM head weights
    lm_head_weight = vl_model.language_model.lm_head.weight.data.float().cpu()
    print(f"  VL LM head: {lm_head_weight.shape}")

    hooks.unregister()
    del vl_model, vl_processor
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Analysis ----
    w = args.warmup

    def stats(key):
        vals = ([r[key] for r in timing_results[w:]]
                if len(timing_results) > w
                else [r[key] for r in timing_results])
        return np.mean(vals), np.std(vals)

    vision_avg, vision_std = stats('vision_ms')
    prefill_avg, prefill_std = stats('prefill_ms')
    decode_avg, decode_std = stats('decode_ms')
    total_avg, total_std = stats('total_ms')
    tokens_avg, _ = stats('num_tokens')
    per_token = decode_avg / max(tokens_avg, 1)

    n = max(len(timing_results) - w, len(timing_results))
    print(f"\n{'=' * 70}")
    print(f"3-STAGE TIMING RESULTS ({n} samples after warmup)")
    print(f"{'=' * 70}")
    print(f"  Vision:   {vision_avg:>8.1f} +/- {vision_std:>5.1f} ms")
    print(f"  Prefill:  {prefill_avg:>8.1f} +/- {prefill_std:>5.1f} ms")
    print(f"  Decode:   {decode_avg:>8.1f} +/- {decode_std:>5.1f} ms"
          f"  ({per_token:.1f} ms/token)")
    print(f"  Total:    {total_avg:>8.1f} +/- {total_std:>5.1f} ms")
    print(f"  Tokens:   {tokens_avg:.1f} avg")
    print(f"  Hidden:   {len(all_hidden)} sequences extracted")

    # ---- Save ----
    output = {
        'timestamp': timestamp,
        'config': {
            'vl_model': args.vl_model,
            'dataset_dir': args.dataset_dir,
            'max_samples': args.max_samples,
            'max_new_tokens': args.max_new_tokens,
            'use_4bit': args.use_4bit,
            'warmup': args.warmup,
            'n_questions': len(questions),
        },
        'timing': {
            'vision_ms': {'mean': vision_avg, 'std': vision_std},
            'prefill_ms': {'mean': prefill_avg, 'std': prefill_std},
            'decode_ms': {'mean': decode_avg, 'std': decode_std},
            'total_ms': {'mean': total_avg, 'std': total_std},
            'per_token_ms': per_token,
            'avg_tokens': tokens_avg,
        },
        'lm_head_shape': list(lm_head_weight.shape),
        'n_hidden_sequences': len(all_hidden),
        'per_sample': timing_results,
    }

    json_path = os.path.join(
        args.output_dir, f'benchmark_vl_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved timing: {json_path}")

    # Optionally save hidden states + LM head
    if args.save_hidden:
        hidden_path = os.path.join(
            args.output_dir, f'vl_hidden_{timestamp}.pt')
        torch.save({
            'hidden_states': all_hidden,
            'lm_head_weight': lm_head_weight,
        }, hidden_path)
        print(f"Saved hidden: {hidden_path}")

    print("=" * 70)


if __name__ == '__main__':
    main()
