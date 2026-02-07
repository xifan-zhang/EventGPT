#!/usr/bin/env python3
"""
Unified Benchmark: EventGPT vs Video-LLaVA
===========================================

Benchmarks both models on the same dataset with proper Stage 3+4 decoupling.

EventGPT: Proper decoupling via cached features (model.generate(event_features=...))
Video-LLaVA: Timing measurement via forward hooks (non-invasive)
"""

import os
import sys
import json
import argparse
import torch
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional, List

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.EventChatModel import EventChatModel
from common.common import tokenizer_event_token, load_image
from dataset.conversation import prepare_event_prompt
from dataset.constants import EVENT_TOKEN_INDEX


class VisionTimingHooks:
    """Forward hooks to measure vision encoding time"""

    def __init__(self, model):
        self.model = model
        self.vision_encoding_time = 0.0
        self.vision_encode_start = None
        self.hooks = []

    def _vision_forward_pre_hook(self, module, input):
        self.vision_encode_start = time.time()

    def _vision_forward_hook(self, module, input, output):
        if self.vision_encode_start is not None:
            self.vision_encoding_time = time.time() - self.vision_encode_start

    def register_hooks(self):
        try:
            vision_tower = self.model.get_vision_tower()
            if vision_tower is not None:
                h1 = vision_tower.register_forward_pre_hook(self._vision_forward_pre_hook)
                h2 = vision_tower.register_forward_hook(self._vision_forward_hook)
                self.hooks = [h1, h2]
                return True
        except:
            pass
        return False

    def unregister_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_vision_time(self):
        return self.vision_encoding_time

    def reset(self):
        self.vision_encoding_time = 0.0
        self.vision_encode_start = None


def load_preprocessed_event_images(event_image_paths, event_processor, dataset_dir, device):
    """Load preprocessed event images"""
    event_list = []
    event_image_size = None

    if device.startswith("cuda"): torch.cuda.synchronize()
    stage1_start = time.time()
    loaded_images = []
    for img_path in event_image_paths:
        full_path = os.path.join(dataset_dir, "event_image", img_path)
        img = load_image(full_path)
        img_array = np.array(img)
        if event_image_size is None:
            event_image_size = list(img_array.shape[:2])
        loaded_images.append(img_array)
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage1_time = time.time() - stage1_start

    if device.startswith("cuda"): torch.cuda.synchronize()
    stage2_start = time.time()
    for img_array in loaded_images:
        event = event_processor(img_array, return_tensors='pt')['pixel_values'][0]
        event = event.to(device, dtype=torch.bfloat16)
        event_list.append(event)
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage2_time = time.time() - stage2_start

    return event_image_size, event_list, stage1_time, stage2_time


def benchmark_eventgpt(model, tokenizer, processor, dataset_dir, dataset, device, max_samples=None):
    """Benchmark EventGPT with proper Stage 3+4 decoupling"""
    print("\n" + "="*80)
    print("EventGPT: Properly Decoupled Stages 3 & 4")
    print("="*80)

    results = []
    query = "What are the key elements in this scene?"
    samples_to_process = dataset[:max_samples] if max_samples else dataset

    for sample_idx, sample in enumerate(tqdm(samples_to_process, desc="EventGPT", leave=False)):
        try:
            event_data = sample.get("event_data")
            if not event_data:
                continue

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

            # Stage 3: Vision Encoding
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage3_start = time.time()

            with torch.inference_mode():
                event_features = model.visval_encode(event_tensor[0].unsqueeze(0))

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage3_time = time.time() - stage3_start

            # Stage 4: LLM Decoding
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage4_start = time.time()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    event_features=event_features,
                    event_image_sizes=event_image_size,
                    do_sample=True,
                    temperature=0.6,
                    top_p=1.0,
                    num_beams=1,
                    max_new_tokens=512,
                    use_cache=True
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage4_time = time.time() - stage4_start

            generated_ids = output_ids[0, input_ids.shape[1]:].tolist() if output_ids.shape[1] > input_ids.shape[1] else output_ids[0].tolist()

            results.append({
                "model": "eventgpt",
                "sample": sample_idx,
                "stage1_time": stage1_time,
                "stage2_time": stage2_time,
                "stage3_time": stage3_time,
                "stage4_time": stage4_time,
                "total_time": stage1_time + stage2_time + stage3_time + stage4_time,
                "tokens": len(generated_ids),
            })

        except Exception as e:
            continue

    return results


def benchmark_videollava_with_hooks(tokenizer, processor, dataset_dir, dataset, device, max_samples=None):
    """Benchmark Video-LLaVA using forward hooks for timing measurement"""
    print("\n" + "="*80)
    print("Video-LLaVA: Timing via Forward Hooks")
    print("="*80)

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM

        # Load Video-LLaVA model
        print("Loading Video-LLaVA model...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                torch_dtype=torch.float16,
                device_map=device,
                local_files_only=True
            )
        except:
            # Try loading from HuggingFace
            model = AutoModelForCausalLM.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                torch_dtype=torch.float16,
                device_map=device,
            )

        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model.eval()
        print("✓ Video-LLaVA loaded")

        results = []
        query = "What are the key elements in this scene?"
        samples_to_process = dataset[:max_samples] if max_samples else dataset

        hooks = VisionTimingHooks(model)
        hooks.register_hooks()

        for sample_idx, sample in enumerate(tqdm(samples_to_process, desc="Video-LLaVA", leave=False)):
            try:
                # For demo: use event images instead of video
                # In production, would load from MP4
                event_data = sample.get("event_data")
                if not event_data:
                    continue

                if "event_image" in sample and sample["event_image"]:
                    event_image_paths = sample["event_image"]

                    # Load images for Video-LLaVA (simplified)
                    if device.startswith("cuda"): torch.cuda.synchronize()
                    stage1_start = time.time()

                    images = []
                    for img_path in event_image_paths:
                        full_path = os.path.join(dataset_dir, "event_image", img_path)
                        img = load_image(full_path)
                        images.append(img)

                    if device.startswith("cuda"): torch.cuda.synchronize()
                    stage1_time = time.time() - stage1_start

                    # Prepare inputs
                    if device.startswith("cuda"): torch.cuda.synchronize()
                    stage2_start = time.time()

                    inputs = processor(text=query, images=images, return_tensors="pt").to(device)

                    if device.startswith("cuda"): torch.cuda.synchronize()
                    stage2_time = time.time() - stage2_start

                    # Reset hooks for this sample
                    hooks.reset()

                    # Stage 3+4: Vision + LLM (measured via hooks)
                    if device.startswith("cuda"): torch.cuda.synchronize()
                    total_start = time.time()

                    with torch.inference_mode():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.6,
                            top_p=1.0,
                        )

                    if device.startswith("cuda"): torch.cuda.synchronize()
                    total_time = time.time() - total_start

                    stage3_time = hooks.get_vision_time()
                    stage4_time = total_time - stage3_time

                    generated_ids = output_ids[0].tolist()

                    results.append({
                        "model": "videollava",
                        "sample": sample_idx,
                        "stage1_time": stage1_time,
                        "stage2_time": stage2_time,
                        "stage3_time": stage3_time,
                        "stage4_time": stage4_time,
                        "total_time": stage1_time + stage2_time + stage3_time + stage4_time,
                        "tokens": len(generated_ids),
                    })

                except Exception as e:
                    continue

        hooks.unregister_hooks()
        return results

    except Exception as e:
        print(f"⚠️  Video-LLaVA loading failed: {e}")
        print("   Proceeding with EventGPT only")
        return []


def print_comparison(eventgpt_results, videollava_results):
    """Print side-by-side comparison"""
    print("\n" + "="*100)
    print("COMPARISON: EventGPT vs Video-LLaVA")
    print("="*100)

    models = []

    if eventgpt_results:
        egpt_avg_s1 = np.mean([r['stage1_time'] for r in eventgpt_results])
        egpt_avg_s2 = np.mean([r['stage2_time'] for r in eventgpt_results])
        egpt_avg_s3 = np.mean([r['stage3_time'] for r in eventgpt_results])
        egpt_avg_s4 = np.mean([r['stage4_time'] for r in eventgpt_results])
        egpt_total = egpt_avg_s1 + egpt_avg_s2 + egpt_avg_s3 + egpt_avg_s4
        egpt_tokens = np.mean([r['tokens'] for r in eventgpt_results])

        models.append({
            "name": "EventGPT",
            "samples": len(eventgpt_results),
            "s1": egpt_avg_s1,
            "s2": egpt_avg_s2,
            "s3": egpt_avg_s3,
            "s4": egpt_avg_s4,
            "total": egpt_total,
            "tokens": egpt_tokens,
        })

    if videollava_results:
        vllava_avg_s1 = np.mean([r['stage1_time'] for r in videollava_results])
        vllava_avg_s2 = np.mean([r['stage2_time'] for r in videollava_results])
        vllava_avg_s3 = np.mean([r['stage3_time'] for r in videollava_results])
        vllava_avg_s4 = np.mean([r['stage4_time'] for r in videollava_results])
        vllava_total = vllava_avg_s1 + vllava_avg_s2 + vllava_avg_s3 + vllava_avg_s4
        vllava_tokens = np.mean([r['tokens'] for r in videollava_results])

        models.append({
            "name": "Video-LLaVA",
            "samples": len(videollava_results),
            "s1": vllava_avg_s1,
            "s2": vllava_avg_s2,
            "s3": vllava_avg_s3,
            "s4": vllava_avg_s4,
            "total": vllava_total,
            "tokens": vllava_tokens,
        })

    # Print comparison table
    print(f"\n{'Model':<15} {'Samples':<10} {'S1 Load':<10} {'S2 Prep':<10} {'S3 Vision':<12} {'S4 LLM':<12} {'Total':<10} {'Tokens':<8}")
    print("-" * 100)

    for model in models:
        print(f"{model['name']:<15} {model['samples']:<10} {model['s1']:.4f}s{'':<3} {model['s2']:.4f}s{'':<3} {model['s3']:.4f}s{'':<5} {model['s4']:.4f}s{'':<5} {model['total']:.4f}s {model['tokens']:.1f}")

    # Print percentages
    print("\n" + "-" * 100)
    print(f"{'PERCENTAGES':<15}")
    print("-" * 100)

    for model in models:
        s1_pct = (model['s1'] / model['total']) * 100
        s2_pct = (model['s2'] / model['total']) * 100
        s3_pct = (model['s3'] / model['total']) * 100
        s4_pct = (model['s4'] / model['total']) * 100
        print(f"{model['name']:<15} {'S1: ' + f'{s1_pct:.1f}%':<10} {'S2: ' + f'{s2_pct:.1f}%':<10} {'S3: ' + f'{s3_pct:.1f}%':<12} {'S4: ' + f'{s4_pct:.1f}%':<12}")

    # Print speedup
    if len(models) == 2:
        speedup = models[1]['total'] / models[0]['total']
        print("\n" + "-" * 100)
        print(f"\nSpeedup: {models[0]['name']} is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than {models[1]['name']}")

    # Print bottleneck analysis
    print("\n" + "-" * 100)
    print("\nBOTTLENECK ANALYSIS:")
    print("-" * 100)

    for model in models:
        s1_pct = (model['s1'] / model['total']) * 100
        s2_pct = (model['s2'] / model['total']) * 100
        s3_pct = (model['s3'] / model['total']) * 100
        s4_pct = (model['s4'] / model['total']) * 100

        bottlenecks = [
            ("Load", s1_pct),
            ("Preprocess", s2_pct),
            ("Vision", s3_pct),
            ("LLM", s4_pct),
        ]
        bottlenecks.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{model['name']}:")
        for stage, pct in bottlenecks:
            marker = "🔴 BOTTLENECK" if pct > 70 else "⚠️  SIGNIFICANT" if pct > 20 else "✅ Optimized"
            print(f"  {stage:<12} {pct:>6.1f}% {marker}")


def main():
    parser = argparse.ArgumentParser(description="EventGPT vs Video-LLaVA Benchmark")
    parser.add_argument("--dataset_dir", type=str, default="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s")
    parser.add_argument("--eventgpt_model", type=str, default="./checkpoints/EventGPT-7b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_videollava", action="store_true", help="Skip Video-LLaVA benchmark")

    args = parser.parse_args()

    # Load dataset
    dataset_json = os.path.join(args.dataset_dir, "EventGPT_Instruction_Subset.json")
    if not os.path.exists(dataset_json):
        print(f"Dataset not found: {dataset_json}")
        return

    with open(dataset_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    print(f"\n{'='*100}")
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"{'='*100}")

    # Benchmark EventGPT
    print("\nLoading EventGPT...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.eventgpt_model, use_fast=False, local_files_only=True)
        model = EventChatModel.from_pretrained(
            args.eventgpt_model, torch_dtype=torch.bfloat16, local_files_only=True
        )
        model = model.to(args.device)
        model.eval()
        processor = model.get_visual_tower().event_processor
        print("✓ EventGPT loaded")

        eventgpt_results = benchmark_eventgpt(
            model, tokenizer, processor, args.dataset_dir, dataset, args.device, args.max_samples
        )
        print(f"✓ EventGPT: {len(eventgpt_results)} samples benchmarked")

    except Exception as e:
        print(f"✗ EventGPT failed: {e}")
        eventgpt_results = []

    # Benchmark Video-LLaVA
    videollava_results = []
    if not args.skip_videollava:
        print("\nLoading Video-LLaVA...")
        videollava_results = benchmark_videollava_with_hooks(
            tokenizer, processor, args.dataset_dir, dataset, args.device, args.max_samples
        )
        if videollava_results:
            print(f"✓ Video-LLaVA: {len(videollava_results)} samples benchmarked")

    # Print comparison
    if eventgpt_results or videollava_results:
        print_comparison(eventgpt_results, videollava_results)

    print("\n" + "="*100)
    print("BENCHMARK COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
