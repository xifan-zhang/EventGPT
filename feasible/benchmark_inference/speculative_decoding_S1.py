#!/usr/bin/env python3
"""
Speculative Decoding: EventGPT (Draft) + Video-LLaVA (Target)
=============================================================

Measures token-level acceptance rate between:
- EventGPT (draft model) - processes event images
- Video-LLaVA (target model) - processes video frames

Models are loaded sequentially to fit in 24GB VRAM.

Usage:
    conda activate egpt
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python speculative_decoding_S1.py \
        --dataset_dir ./data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
        --max_samples 10
"""

import os
import sys
import json
import argparse
import torch
import time
import gc
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# Add paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'feasible', 'feature_alignment'))
sys.path.insert(0, os.path.join(ROOT, 'feasible'))

from transformers import (
    AutoConfig,
    AutoTokenizer,
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
)

from feature_alignment import LightweightAlignmentModule
from model.EventChatModel import EventChatModel
from common.common import tokenizer_event_token
from dataset.conversation import prepare_event_prompt
from dataset.constants import (
    EVENT_TOKEN_INDEX,
    DEFAULT_EVENT_PATCH_TOKEN,
    DEFAULT_EV_START_TOKEN,
    DEFAULT_EV_END_TOKEN,
)


def cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def read_video_frames(video_paths, max_frames=8):
    """Read video frames from image paths."""
    frames = []
    for img_path in video_paths[:max_frames]:
        img = Image.open(img_path).convert('RGB')
        frames.append(np.array(img))
    return np.stack(frames) if frames else None


def run_eventgpt_phase(
    samples: list,
    dataset_dir: str,
    model_path: str,
    query: str,
    max_new_tokens: int,
    device: str,
) -> list:
    """Run EventGPT inference on all samples."""
    print("\n" + "="*60)
    print("PHASE 1: EventGPT (Draft Model) Inference")
    print("="*60)

    print(f"Loading EventGPT from {model_path}...")

    config = AutoConfig.from_pretrained(model_path)

    # Fix CLIP path
    if hasattr(config, 'mm_visual_tower'):
        visual_tower = config.mm_visual_tower
        if not os.path.exists(visual_tower):
            local_clip = os.path.join(ROOT, "checkpoints", "clip-vit-large-patch14-336")
            if os.path.exists(local_clip):
                config.mm_visual_tower = local_clip
            else:
                config.mm_visual_tower = "openai/clip-vit-large-patch14-336"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = EventChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        config=config,
    )

    # Add special tokens
    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_EVENT_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens([DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN], special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_visual_tower()
    event_processor = vision_tower.event_processor

    model.to(device)
    model.eval()
    print("EventGPT loaded.")

    results = []

    for sample in tqdm(samples, desc="EventGPT"):
        try:
            event_images = sample.get('event_image', [])
            if not event_images:
                continue

            event_paths = [
                os.path.join(dataset_dir, 'event_image', p)
                for p in event_images
            ]

            # Process event images
            event_tensors = []
            for img_path in event_paths:
                img = np.array(Image.open(img_path).convert('RGB'))
                processed = event_processor(img, return_tensors='pt')['pixel_values'][0]
                processed = processed.to(device, dtype=torch.bfloat16)
                event_tensors.append(processed)

            event_size = list(np.array(Image.open(event_paths[0])).shape[:2])

            # Prepare prompt
            prompt = prepare_event_prompt(query, 'eventgpt_v1')
            input_ids = tokenizer_event_token(
                prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(device)

            # Generate
            torch.cuda.synchronize()
            start = time.time()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    event_tensors=event_tensors,
                    event_image_sizes=event_size,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )

            torch.cuda.synchronize()
            elapsed = time.time() - start

            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Extract generated tokens
            if output_ids.shape[1] > input_ids.shape[1]:
                gen_tokens = output_ids[0, input_ids.shape[1]:].tolist()
            else:
                gen_tokens = output_ids[0].tolist()

            results.append({
                'id': sample.get('id', 'unknown'),
                'eventgpt_output': output,
                'eventgpt_tokens': gen_tokens,
                'eventgpt_time': elapsed,
            })

        except Exception as e:
            print(f"\nError on {sample.get('id')}: {e}")
            continue

    # Cleanup
    del model
    del tokenizer
    cleanup_gpu()
    print(f"EventGPT phase complete. Processed {len(results)} samples.")

    return results


def run_videollava_phase(
    samples: list,
    eventgpt_results: list,
    dataset_dir: str,
    model_path: str,
    query: str,
    max_new_tokens: int,
    device: str,
) -> list:
    """Run Video-LLaVA inference on all samples."""
    print("\n" + "="*60)
    print("PHASE 2: Video-LLaVA (Target Model) Inference")
    print("="*60)

    print(f"Loading Video-LLaVA from {model_path}...")

    processor = VideoLlavaProcessor.from_pretrained(model_path)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    model.to(device)
    model.eval()
    print("Video-LLaVA loaded.")

    # Create lookup for eventgpt results
    eventgpt_lookup = {r['id']: r for r in eventgpt_results}

    results = []

    for sample in tqdm(samples, desc="Video-LLaVA"):
        sample_id = sample.get('id', 'unknown')

        if sample_id not in eventgpt_lookup:
            continue

        try:
            video_data = sample.get('video_data', '')
            video_dir = os.path.join(dataset_dir, 'video', video_data)
            video_paths = sorted(Path(video_dir).glob('*.png'))[:8]

            if not video_paths:
                continue

            # Load video frames
            video_frames = read_video_frames([str(p) for p in video_paths], max_frames=8)

            # Prepare prompt
            prompt = f"USER: <video>\n{query} ASSISTANT:"

            # Process inputs
            inputs = processor(
                text=prompt,
                videos=video_frames,
                return_tensors="pt",
                padding=True
            ).to(device, torch.float16)

            # Generate
            torch.cuda.synchronize()
            start = time.time()

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            torch.cuda.synchronize()
            elapsed = time.time() - start

            output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            if "ASSISTANT:" in output:
                output = output.split("ASSISTANT:")[-1].strip()

            # Extract generated tokens
            input_len = inputs['input_ids'].shape[1]
            gen_tokens = output_ids[0, input_len:].tolist()

            # Merge with EventGPT results
            result = eventgpt_lookup[sample_id].copy()
            result['videollava_output'] = output
            result['videollava_tokens'] = gen_tokens
            result['videollava_time'] = elapsed

            # Compute acceptance
            egpt_tokens = result['eventgpt_tokens']
            vllava_tokens = gen_tokens
            min_len = min(len(egpt_tokens), len(vllava_tokens))

            if min_len > 0:
                matches = sum(1 for i in range(min_len)
                             if egpt_tokens[i] == vllava_tokens[i])
                result['accepted_tokens'] = matches
                result['total_compared'] = min_len
                result['acceptance_rate'] = matches / min_len
            else:
                result['accepted_tokens'] = 0
                result['total_compared'] = 0
                result['acceptance_rate'] = 0.0

            # Compute c ratio
            result['c_ratio'] = (result['eventgpt_time'] / result['videollava_time']
                                if result['videollava_time'] > 0 else 0)

            results.append(result)

        except Exception as e:
            print(f"\nError on {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup
    del model
    del processor
    cleanup_gpu()
    print(f"Video-LLaVA phase complete. Processed {len(results)} samples.")

    return results


def main():
    parser = argparse.ArgumentParser(description='Speculative Decoding: EventGPT + Video-LLaVA')
    parser.add_argument('--dataset_dir', type=str,
                        default='./data/my_egpt_dsec_train/my_egpt_dsec_train_1s')
    parser.add_argument('--eventgpt_path', type=str,
                        default='./checkpoints/EventGPT-7b')
    parser.add_argument('--videollava_path', type=str,
                        default='LanguageBind/Video-LLaVA-7B-hf')
    parser.add_argument('--adapter_path', type=str,
                        default='./feasible/feature_alignment/checkpoints/alignment_1s/lightweight_alignment.pt')
    parser.add_argument('--output_json', type=str,
                        default='./feasible/benchmark_inference/speculative_results_S1.json')
    parser.add_argument('--max_samples', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--query', type=str,
                        default='What are the key elements in this scene?')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("SPECULATIVE DECODING: EventGPT (Draft) + Video-LLaVA (Target)")
    print("="*70)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Draft tokens per step (γ): {args.gamma}")
    print(f"Max samples: {args.max_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("="*70)
    print("\nNote: Models loaded sequentially to fit in 24GB VRAM")

    # Load dataset
    json_path = os.path.join(args.dataset_dir, "EventGPT_Instruction_Subset.json")
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    # Phase 1: EventGPT
    eventgpt_results = run_eventgpt_phase(
        samples=dataset,
        dataset_dir=args.dataset_dir,
        model_path=args.eventgpt_path,
        query=args.query,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    # Phase 2: Video-LLaVA
    results = run_videollava_phase(
        samples=dataset,
        eventgpt_results=eventgpt_results,
        dataset_dir=args.dataset_dir,
        model_path=args.videollava_path,
        query=args.query,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    # Compute summary
    if results:
        total_accepted = sum(r['accepted_tokens'] for r in results)
        total_compared = sum(r['total_compared'] for r in results)
        overall_acceptance = total_accepted / total_compared if total_compared > 0 else 0

        avg_eventgpt_time = np.mean([r['eventgpt_time'] for r in results])
        avg_videollava_time = np.mean([r['videollava_time'] for r in results])
        avg_c_ratio = avg_eventgpt_time / avg_videollava_time if avg_videollava_time > 0 else 0

        # Theoretical speedup
        alpha = overall_acceptance
        gamma = args.gamma
        c = avg_c_ratio

        if 0 < alpha < 1:
            expected_accepted = (1 - alpha**(gamma + 1)) / (1 - alpha)
        else:
            expected_accepted = 1 if alpha == 0 else gamma + 1

        cost = c * gamma + 1
        theoretical_speedup = expected_accepted / cost if cost > 0 else 0
    else:
        total_accepted = total_compared = 0
        overall_acceptance = avg_eventgpt_time = avg_videollava_time = 0
        avg_c_ratio = theoretical_speedup = 0

    summary = {
        'config': {
            'gamma': args.gamma,
            'max_new_tokens': args.max_new_tokens,
            'query': args.query,
            'eventgpt_path': args.eventgpt_path,
            'videollava_path': args.videollava_path,
        },
        'results': {
            'total_samples': len(results),
            'overall_token_acceptance_rate': overall_acceptance,
            'total_accepted_tokens': total_accepted,
            'total_compared_tokens': total_compared,
            'avg_eventgpt_time': avg_eventgpt_time,
            'avg_videollava_time': avg_videollava_time,
            'avg_c_ratio': avg_c_ratio,
            'theoretical_speedup': theoretical_speedup,
        },
        'samples': results,
    }

    # Save
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {args.output_json}")

    # Print summary
    print("\n" + "="*70)
    print("SPECULATIVE DECODING RESULTS")
    print("="*70)

    print(f"\nSamples processed: {len(results)}")
    print(f"γ (draft tokens per step): {args.gamma}")

    print(f"\n{'─'*70}")
    print("TOKEN-LEVEL ACCEPTANCE RATE")
    print(f"{'─'*70}")
    print(f"  EventGPT vs Video-LLaVA: {overall_acceptance:.2%}")
    print(f"  Matched tokens: {total_accepted}/{total_compared}")

    print(f"\n{'─'*70}")
    print("TIMING")
    print(f"{'─'*70}")
    print(f"  Avg EventGPT (draft) time:     {avg_eventgpt_time:.3f}s")
    print(f"  Avg Video-LLaVA (target) time: {avg_videollava_time:.3f}s")
    print(f"  c ratio (draft/target):        {avg_c_ratio:.3f}")

    print(f"\n{'─'*70}")
    print("THEORETICAL SPEEDUP")
    print(f"{'─'*70}")
    print(f"  With γ={args.gamma}, α={overall_acceptance:.3f}:")
    print(f"  Speedup: {theoretical_speedup:.2f}x")

    if overall_acceptance < 0.05:
        print(f"\n  ⚠ Very low token acceptance (~{overall_acceptance:.1%})")
        print(f"    → EventGPT and Video-LLaVA have different tokenizers/decoders")
        print(f"    → For speculative decoding: need shared decoder or distillation")
    elif overall_acceptance > 0.3:
        print(f"\n  ✓ Moderate acceptance - speculative decoding beneficial")

    print("="*70)


if __name__ == '__main__':
    main()
