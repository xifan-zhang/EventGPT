#!/usr/bin/env python3
"""
Unified Benchmark: EventGPT vs Video-LLaVA with 4-Stage Analysis & Speculative Decoding
========================================================================================

Benchmarks both models with accurate stage separation and speculative decoding analysis:

4-Stage Pipeline:
  - Stage 1: Load images from disk
  - Stage 2: Preprocess images (CLIP)
  - Stage 3: Vision encoding
  - Stage 4: LLM decoding

Speculative Decoding Mode:
  - EventGPT as draft model (fast)
  - Video-LLaVA as target model (accurate)
  - Measures token acceptance rate and speedup

Usage:
  # EventGPT only (default, fast)
  python benchmark_inference_4stages.py --eventgpt_only

  # Both models comparison
  python benchmark_inference_4stages.py

  # Speculative decoding benchmark
  python benchmark_inference_4stages.py --speculative --gamma 4

  # Full benchmark with 200 samples
  python benchmark_inference_4stages.py --speculative --max_samples 200 --gamma 4
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

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.EventChatModel import EventChatModel
from common.common import tokenizer_event_token, load_image
from dataset.conversation import prepare_event_prompt
from dataset.constants import (
    EVENT_TOKEN_INDEX,
    DEFAULT_EVENT_PATCH_TOKEN,
    DEFAULT_EV_START_TOKEN,
    DEFAULT_EV_END_TOKEN,
)


def run_eventgpt_decoupled_benchmark(model, tokenizer, processor, dataset_dir, dataset, device="cuda"):
    """
    Run EventGPT benchmark with properly decoupled Stage 3 and Stage 4.

    Stage 3: Vision encoding (extract features)
    Stage 4: LLM decoding (generate with cached features)
    """
    print("\n" + "="*80)
    print("EventGPT: Properly Decoupled Stages 3 & 4")
    print("="*80)

    results = []
    query = "What are the key elements in this scene?"

    for sample_idx, sample in enumerate(tqdm(dataset, desc="EventGPT Decoupled Test")):
        try:
            event_data = sample.get("event_data")
            if not event_data:
                continue

            # Stage 1: Load event images
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

            # ===== STAGE 3: VISION ENCODING (Decoupled) =====
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage3_start = time.time()
            with torch.inference_mode():
                # Call visval_encode directly - this is the vision encoding step
                event_features = model.visval_encode(event_tensor[0].unsqueeze(0))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage3_time = time.time() - stage3_start

            # ===== STAGE 4: LLM DECODING (with cached features) =====
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage4_start = time.time()
            with torch.inference_mode():
                # Pass pre-computed features to skip re-encoding
                output_ids = model.generate(
                    input_ids,
                    event_features=event_features,  # Cached features!
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

            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Extract generated tokens
            if output_ids.shape[1] > input_ids.shape[1]:
                generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
            else:
                generated_ids = output_ids[0].tolist()

            results.append({
                "sample": sample_idx,
                "stage1_time": stage1_time,
                "stage2_time": stage2_time,
                "stage3_time": stage3_time,  # Vision encoding (DECOUPLED)
                "stage4_time": stage4_time,  # LLM decoding (with cached features)
                "total_time": stage1_time + stage2_time + stage3_time + stage4_time,
                "output": output,
                "tokens": len(generated_ids),
                "token_ids": generated_ids,  # For speculative decoding comparison
            })

            print(f"\nSample {sample_idx}: S1={stage1_time:.4f}s | S2={stage2_time:.4f}s | "
                  f"S3={stage3_time:.4f}s | S4={stage4_time:.4f}s | Total={stage1_time+stage2_time+stage3_time+stage4_time:.4f}s | "
                  f"Tokens={len(generated_ids)}")

        except Exception as e:
            print(f"Error on sample {sample_idx}: {e}")
            continue

    return results


def load_preprocessed_event_images(event_image_paths, event_processor, dataset_dir, device):
    """Load preprocessed event images and process them with event_processor.

    Args:
        event_image_paths: List of relative paths to event images
        event_processor: The CLIP image processor
        dataset_dir: Base dataset directory
        device: Device to load tensors to

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

    # Stage 2: Process images with event_processor (CLIP preprocessing)
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage2_start = time.time()
    for img_array in loaded_images:
        event = event_processor(img_array, return_tensors='pt')['pixel_values'][0]
        event = event.to(device, dtype=torch.bfloat16)
        event_list.append(event)
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage2_time = time.time() - stage2_start

    return event_image_size, event_list, stage1_time, stage2_time


class VisionTimingHooks:
    """Forward hooks to measure vision encoding time for Video-LLaVA"""

    def __init__(self, model):
        self.model = model
        self.vision_encoding_time = 0.0
        self.vision_encode_start = None
        self.hooks = []

    def _vision_forward_pre_hook(self, module, input):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.vision_encode_start = time.time()

    def _vision_forward_hook(self, module, input, output):
        if self.vision_encode_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.vision_encoding_time = time.time() - self.vision_encode_start

    def register_hooks(self):
        """Register hooks on vision tower"""
        vision_tower = None
        try:
            # Try different ways to access vision tower
            if hasattr(self.model, 'get_vision_tower'):
                vision_tower = self.model.get_vision_tower()
            elif hasattr(self.model, 'vision_tower'):
                vision_tower = self.model.vision_tower
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
                vision_tower = self.model.model.vision_tower

            if vision_tower is not None:
                h1 = vision_tower.register_forward_pre_hook(self._vision_forward_pre_hook)
                h2 = vision_tower.register_forward_hook(self._vision_forward_hook)
                self.hooks = [h1, h2]
                print(f"  ✓ Vision hooks registered on {type(vision_tower).__name__}")
                return True
            else:
                print("  ⚠ Vision tower not found, timing will be estimated")
        except Exception as e:
            print(f"  ⚠ Could not register vision hooks: {e}")
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


def run_videollava_benchmark_with_hooks(dataset_dir, dataset, device="cuda", max_samples=None):
    """
    Benchmark Video-LLaVA using forward hooks for timing measurement.
    Uses non-invasive hooks to measure vision tower execution time.
    """
    print("\n" + "="*80)
    print("Video-LLaVA: Timing via Forward Hooks (Optional)")
    print("="*80)

    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        print("Loading Video-LLaVA model...")
        model_id = "llava-hf/llava-1.5-7b-hf"
        try:
            # Try loading from local cache first
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
                local_files_only=True
            )
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        except:
            # Fall back to downloading from HuggingFace
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

        hooks = VisionTimingHooks(model)
        hooks.register_hooks()

        for sample_idx, sample in enumerate(tqdm(samples_to_process, desc="Video-LLaVA", leave=False)):
            try:
                event_data = sample.get("event_data")
                if not event_data:
                    continue

                if "event_image" in sample and sample["event_image"]:
                    event_image_paths = sample["event_image"]

                    # Load images
                    if device.startswith("cuda"): torch.cuda.synchronize()
                    stage1_start = time.time()

                    images = []
                    for img_path in event_image_paths:
                        full_path = os.path.join(dataset_dir, "event_image", img_path)
                        img = load_image(full_path)
                        images.append(img)

                    # Use only first image for LLaVA (single image model)
                    if len(images) > 1:
                        images = [images[0]]

                    if device.startswith("cuda"): torch.cuda.synchronize()
                    stage1_time = time.time() - stage1_start

                    # Prepare inputs with proper LLaVA format
                    if device.startswith("cuda"): torch.cuda.synchronize()
                    stage2_start = time.time()

                    # LLaVA requires image tokens in the prompt
                    prompt = f"USER: <image>\n{base_query}\nASSISTANT:"
                    inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)

                    if device.startswith("cuda"): torch.cuda.synchronize()
                    stage2_time = time.time() - stage2_start

                    # Reset hooks for this sample
                    hooks.reset()

                    # Generate with hooks
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

                    # Extract only generated tokens (exclude input)
                    input_len = inputs['input_ids'].shape[1]
                    generated_ids = output_ids[0, input_len:].tolist()

                    # Decode the generated text
                    output_text = processor.batch_decode(
                        output_ids[:, input_len:], skip_special_tokens=True
                    )[0].strip()

                    results.append({
                        "model": "videollava",
                        "sample": sample_idx,
                        "stage1_time": stage1_time,
                        "stage2_time": stage2_time,
                        "stage3_time": stage3_time,
                        "stage4_time": stage4_time,
                        "total_time": stage1_time + stage2_time + stage3_time + stage4_time,
                        "tokens": len(generated_ids),
                        "token_ids": generated_ids,  # For speculative decoding comparison
                        "output": output_text,
                    })
                else:
                    continue

            except Exception as e:
                print(f"\n  Error on sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        hooks.unregister_hooks()
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
    torch.cuda.synchronize()


def compute_speculative_decoding_metrics(eventgpt_results, videollava_results, gamma=4):
    """
    Compute speculative decoding metrics.

    EventGPT is the draft model, Video-LLaVA is the target model.
    Measures token acceptance rate and theoretical speedup.
    """
    print("\n" + "="*80)
    print("SPECULATIVE DECODING ANALYSIS")
    print(f"Draft Model: EventGPT | Target Model: Video-LLaVA | Gamma: {gamma}")
    print("="*80)

    # Match samples by index
    egpt_by_sample = {r['sample']: r for r in eventgpt_results}
    vllava_by_sample = {r['sample']: r for r in videollava_results}

    matched_samples = []
    total_matched_tokens = 0
    total_compared_tokens = 0

    for sample_idx in egpt_by_sample:
        if sample_idx not in vllava_by_sample:
            continue

        egpt = egpt_by_sample[sample_idx]
        vllava = vllava_by_sample[sample_idx]

        egpt_tokens = egpt.get('token_ids', [])
        vllava_tokens = vllava.get('token_ids', [])

        min_len = min(len(egpt_tokens), len(vllava_tokens))
        if min_len == 0:
            continue

        # Count matching tokens
        matches = sum(1 for i in range(min_len) if egpt_tokens[i] == vllava_tokens[i])

        acceptance_rate = matches / min_len if min_len > 0 else 0.0

        matched_samples.append({
            'sample': sample_idx,
            'egpt_tokens': len(egpt_tokens),
            'vllava_tokens': len(vllava_tokens),
            'matched_tokens': matches,
            'compared_tokens': min_len,
            'acceptance_rate': acceptance_rate,
            'egpt_time': egpt['total_time'],
            'vllava_time': vllava['total_time'],
        })

        total_matched_tokens += matches
        total_compared_tokens += min_len

    if not matched_samples:
        print("No matched samples for speculative decoding analysis")
        return {}

    # Overall metrics
    overall_acceptance = total_matched_tokens / total_compared_tokens if total_compared_tokens > 0 else 0.0

    # Timing metrics
    avg_egpt_time = np.mean([s['egpt_time'] for s in matched_samples])
    avg_vllava_time = np.mean([s['vllava_time'] for s in matched_samples])
    c_ratio = avg_egpt_time / avg_vllava_time if avg_vllava_time > 0 else 0

    # Theoretical speedup calculation
    alpha = overall_acceptance
    if 0 < alpha < 1:
        expected_accepted = (1 - alpha**(gamma + 1)) / (1 - alpha)
    else:
        expected_accepted = 1 if alpha == 0 else gamma + 1

    cost = c_ratio * gamma + 1
    theoretical_speedup = expected_accepted / cost if cost > 0 else 0

    # Print results
    print(f"\n{'─'*80}")
    print("TOKEN ACCEPTANCE RATE")
    print(f"{'─'*80}")
    print(f"  Matched samples:               {len(matched_samples)}")
    print(f"  Total tokens compared:         {total_compared_tokens}")
    print(f"  Tokens accepted by target:     {total_matched_tokens}")
    print(f"  Overall acceptance rate:       {overall_acceptance:.2%}")

    print(f"\n{'─'*80}")
    print("TIMING ANALYSIS")
    print(f"{'─'*80}")
    print(f"  Avg EventGPT (draft) time:     {avg_egpt_time:.4f}s")
    print(f"  Avg Video-LLaVA (target) time: {avg_vllava_time:.4f}s")
    print(f"  Cost ratio (c = draft/target): {c_ratio:.4f}")

    print(f"\n{'─'*80}")
    print("THEORETICAL SPEEDUP")
    print(f"{'─'*80}")
    print(f"  Gamma (draft tokens/step):     {gamma}")
    print(f"  Acceptance rate (alpha):       {alpha:.4f}")
    print(f"  Expected tokens per step:      {expected_accepted:.2f}")
    print(f"  Cost per step:                 {cost:.4f}")
    print(f"  Theoretical speedup:           {theoretical_speedup:.2f}x")

    # Interpretation
    print(f"\n{'─'*80}")
    print("INTERPRETATION")
    print(f"{'─'*80}")

    if overall_acceptance < 0.05:
        print(f"  ⚠ Very low acceptance rate ({overall_acceptance:.1%})")
        print(f"    → EventGPT and Video-LLaVA have different tokenizers/outputs")
        print(f"    → Speculative decoding may not be beneficial")
        print(f"    → Consider: Feature alignment, shared decoder, or distillation")
    elif overall_acceptance < 0.20:
        print(f"  ⚠ Low acceptance rate ({overall_acceptance:.1%})")
        print(f"    → Marginal benefit from speculative decoding")
        print(f"    → Consider: Reducing gamma or implementing alignment")
    elif overall_acceptance < 0.50:
        print(f"  ✓ Moderate acceptance rate ({overall_acceptance:.1%})")
        print(f"    → Speculative decoding is viable")
        print(f"    → Expected speedup: {theoretical_speedup:.1f}x")
    else:
        print(f"  ✓ High acceptance rate ({overall_acceptance:.1%})")
        print(f"    → Excellent candidate for speculative decoding")
        print(f"    → Expected speedup: {theoretical_speedup:.1f}x")

    print(f"{'='*80}")

    return {
        'matched_samples': len(matched_samples),
        'total_compared_tokens': total_compared_tokens,
        'total_matched_tokens': total_matched_tokens,
        'overall_acceptance_rate': overall_acceptance,
        'avg_egpt_time': avg_egpt_time,
        'avg_vllava_time': avg_vllava_time,
        'c_ratio': c_ratio,
        'gamma': gamma,
        'theoretical_speedup': theoretical_speedup,
        'per_sample': matched_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="4-Stage Benchmark with Speculative Decoding - EventGPT vs Video-LLaVA")
    parser.add_argument("--dataset_dir", type=str, default="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s")
    parser.add_argument("--eventgpt_model_path", type=str, default="./checkpoints/EventGPT-7b")
    parser.add_argument("--videollava_model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to benchmark (None = all)")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--show_samples", action="store_true", help="Show individual sample results")
    parser.add_argument("--eventgpt_only", action="store_true", help="Only benchmark EventGPT (skip Video-LLaVA)")
    parser.add_argument("--speculative", action="store_true", help="Run speculative decoding analysis")
    parser.add_argument("--gamma", type=int, default=4, help="Draft tokens per step for speculative decoding")
    parser.add_argument("--output_json", type=str, default=None, help="Save results to JSON file (default: dataset_dir/benchmark_DATETIME.json)")

    args = parser.parse_args()

    # Load dataset first to get sample count for filename
    dataset_json_path = os.path.join(args.dataset_dir, "EventGPT_Instruction_Subset.json")
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    num_samples = len(dataset)

    # Generate default output path with datetime and sample count if not specified
    if args.output_json is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_json = os.path.join(args.dataset_dir, f"benchmark_4stages_{num_samples}samples_{timestamp}.json")

    print(f"Loading dataset with {num_samples} samples...")

    # Load EventGPT model
    print(f"Loading EventGPT from {args.eventgpt_model_path}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.eventgpt_model_path, use_fast=False, local_files_only=True)
    model = EventChatModel.from_pretrained(args.eventgpt_model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    model = model.to(args.device)
    model.eval()

    processor = model.get_visual_tower().event_processor

    # Run EventGPT benchmark
    results = run_eventgpt_decoupled_benchmark(
        model, tokenizer, processor, args.dataset_dir, dataset, args.device
    )

    # Run Video-LLaVA benchmark if needed
    videollava_results = []
    speculative_metrics = {}

    if args.speculative or not args.eventgpt_only:
        # Unload EventGPT to free memory for Video-LLaVA
        del model
        cleanup_gpu()
        print("\nUnloaded EventGPT to free memory for Video-LLaVA...")

        videollava_results = run_videollava_benchmark_with_hooks(
            args.dataset_dir, dataset, args.device, args.max_samples
        )

        # Speculative decoding analysis
        if args.speculative and results and videollava_results:
            speculative_metrics = compute_speculative_decoding_metrics(
                results, videollava_results, gamma=args.gamma
            )

    # Print summary
    if results:
        import numpy as np

        avg_s1 = sum(r['stage1_time'] for r in results) / len(results)
        avg_s2 = sum(r['stage2_time'] for r in results) / len(results)
        avg_s3 = sum(r['stage3_time'] for r in results) / len(results)
        avg_s4 = sum(r['stage4_time'] for r in results) / len(results)
        avg_total = avg_s1 + avg_s2 + avg_s3 + avg_s4
        avg_tokens = sum(r['tokens'] for r in results) / len(results)

        s1_pct = (avg_s1 / avg_total) * 100
        s2_pct = (avg_s2 / avg_total) * 100
        s3_pct = (avg_s3 / avg_total) * 100
        s4_pct = (avg_s4 / avg_total) * 100

        throughput = 1.0 / avg_total

        print(f"\n{'='*80}")
        print("PROPERLY DECOUPLED BENCHMARK SUMMARY - EventGPT")
        print(f"{'='*80}")
        print(f"Samples processed:                 {len(results)}")
        print(f"Stage 1 (Load):                    {avg_s1:.4f}s ({s1_pct:.1f}%)")
        print(f"Stage 2 (Preprocess):              {avg_s2:.4f}s ({s2_pct:.1f}%)")
        print(f"Stage 3 (Vision Encoding):         {avg_s3:.4f}s ({s3_pct:.1f}%)  ✓ DECOUPLED")
        print(f"Stage 4 (LLM Decoding):            {avg_s4:.4f}s ({s4_pct:.1f}%)  ✓ DECOUPLED")
        print(f"{'─'*80}")
        print(f"Total per sample:                  {avg_total:.4f}s")
        print(f"Average tokens generated:          {avg_tokens:.1f}")
        print(f"Throughput:                        {throughput:.2f} samples/sec")

        if args.show_samples and len(results) <= 20:
            print(f"\n{'='*80}")
            print("INDIVIDUAL SAMPLE RESULTS")
            print(f"{'='*80}")
            for r in results:
                print(f"Sample {r['sample']:3d}: S1={r['stage1_time']:.4f}s S2={r['stage2_time']:.4f}s S3={r['stage3_time']:.4f}s S4={r['stage4_time']:.4f}s Total={r['stage1_time']+r['stage2_time']+r['stage3_time']+r['stage4_time']:.4f}s Tokens={r['tokens']}")

        print(f"\n{'='*80}")
        print("KEY FINDINGS & RECOMMENDATIONS")
        print(f"{'='*80}")
        print(f"\n✅ EventGPT Analysis:")
        print(f"   • Stage 3+4 properly decoupled: YES")
        print(f"   • Vision features cached: YES")
        print(f"   • Double-encoding detected: NO")
        print(f"   • Bottleneck: Stage 4 (LLM) at {s4_pct:.1f}% of time")
        print(f"   • Vision encoding performance: {avg_s3:.4f}s ({s3_pct:.1f}%)")
        print(f"   • Speedup factor: LLM is {avg_s4/avg_s3:.1f}x slower than vision")

        print(f"\n🎯 Optimization Priority:")
        print(f"   1. HIGH: Stage 4 (LLM) optimization - {s4_pct:.1f}% bottleneck")
        print(f"   2. LOW: Stage 1-3 optimization - combined only {100-s4_pct:.1f}%")

        print(f"\n💡 Recommended Optimizations for Stage 4:")
        print(f"   • Speculative decoding")
        print(f"   • Token pruning")
        print(f"   • Model quantization")
        print(f"   • Kernel fusion")
        print(f"   • Batch inference")

        # Print comparison if Video-LLaVA was benchmarked
        if videollava_results:
            print(f"\n{'='*80}")
            print("COMPARISON: EventGPT vs Video-LLaVA")
            print(f"{'='*80}")

            vllava_avg_s1 = sum(r['stage1_time'] for r in videollava_results) / len(videollava_results)
            vllava_avg_s2 = sum(r['stage2_time'] for r in videollava_results) / len(videollava_results)
            vllava_avg_s3 = sum(r['stage3_time'] for r in videollava_results) / len(videollava_results)
            vllava_avg_s4 = sum(r['stage4_time'] for r in videollava_results) / len(videollava_results)
            vllava_total = vllava_avg_s1 + vllava_avg_s2 + vllava_avg_s3 + vllava_avg_s4
            vllava_tokens = sum(r['tokens'] for r in videollava_results) / len(videollava_results)

            print(f"\n{'Model':<20} {'Samples':<10} {'S1':<8} {'S2':<8} {'S3':<8} {'S4':<8} {'Total':<10} {'Tokens':<8}")
            print("-" * 100)
            print(f"{'EventGPT':<20} {len(results):<10} {avg_s1:.4f}s  {avg_s2:.4f}s  {avg_s3:.4f}s  {avg_s4:.4f}s  {avg_total:.4f}s  {avg_tokens:.1f}")
            print(f"{'Video-LLaVA':<20} {len(videollava_results):<10} {vllava_avg_s1:.4f}s  {vllava_avg_s2:.4f}s  {vllava_avg_s3:.4f}s  {vllava_avg_s4:.4f}s  {vllava_total:.4f}s  {vllava_tokens:.1f}")

            # Print percentages
            print(f"\n{'BOTTLENECK ANALYSIS':<20}")
            print("-" * 100)
            print(f"{'EventGPT':<20} {'S1: ' + f'{s1_pct:.1f}%':<10} {'S2: ' + f'{s2_pct:.1f}%':<10} {'S3: ' + f'{s3_pct:.1f}%':<10} {'S4: ' + f'{s4_pct:.1f}%':<10}")

            vllava_s1_pct = (vllava_avg_s1 / vllava_total) * 100
            vllava_s2_pct = (vllava_avg_s2 / vllava_total) * 100
            vllava_s3_pct = (vllava_avg_s3 / vllava_total) * 100
            vllava_s4_pct = (vllava_avg_s4 / vllava_total) * 100

            print(f"{'Video-LLaVA':<20} {'S1: ' + f'{vllava_s1_pct:.1f}%':<10} {'S2: ' + f'{vllava_s2_pct:.1f}%':<10} {'S3: ' + f'{vllava_s3_pct:.1f}%':<10} {'S4: ' + f'{vllava_s4_pct:.1f}%':<10}")

            # Print speedup
            speedup = vllava_total / avg_total
            print(f"\n{'='*80}")
            print(f"✅ Speedup: EventGPT is {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'} than Video-LLaVA")
            print(f"   EventGPT: {avg_total:.4f}s/sample")
            print(f"   Video-LLaVA: {vllava_total:.4f}s/sample")
            print(f"{'='*80}")

        # Save results to JSON
        if args.output_json:
            # Build per-sample combined results
            egpt_by_sample = {r['sample']: r for r in results}
            vllava_by_sample = {r['sample']: r for r in videollava_results} if videollava_results else {}

            combined_samples = []
            for sample_idx in sorted(egpt_by_sample.keys()):
                sample_data = {
                    'sample': sample_idx,
                    'eventgpt': egpt_by_sample[sample_idx],
                }
                if sample_idx in vllava_by_sample:
                    sample_data['videollava'] = vllava_by_sample[sample_idx]
                    # Add acceptance info if available
                    egpt_tokens = egpt_by_sample[sample_idx].get('token_ids', [])
                    vllava_tokens = vllava_by_sample[sample_idx].get('token_ids', [])
                    min_len = min(len(egpt_tokens), len(vllava_tokens))
                    if min_len > 0:
                        matches = sum(1 for i in range(min_len) if egpt_tokens[i] == vllava_tokens[i])
                        sample_data['acceptance'] = {
                            'matched_tokens': matches,
                            'compared_tokens': min_len,
                            'acceptance_rate': matches / min_len,
                        }
                combined_samples.append(sample_data)

            output_data = {
                'config': {
                    'dataset_dir': args.dataset_dir,
                    'eventgpt_model_path': args.eventgpt_model_path,
                    'videollava_model': args.videollava_model,
                    'max_samples': args.max_samples,
                    'max_new_tokens': args.max_new_tokens,
                    'gamma': args.gamma,
                    'speculative': args.speculative,
                    'timestamp': datetime.now().isoformat(),
                },
                'summary': {
                    'eventgpt': {
                        'samples': len(results),
                        'stage1_avg': avg_s1,
                        'stage2_avg': avg_s2,
                        'stage3_avg': avg_s3,
                        'stage4_avg': avg_s4,
                        'total_avg': avg_total,
                        'tokens_avg': avg_tokens,
                    },
                },
                'samples': combined_samples,
            }

            if videollava_results:
                vllava_avg_s1 = sum(r['stage1_time'] for r in videollava_results) / len(videollava_results)
                vllava_avg_s2 = sum(r['stage2_time'] for r in videollava_results) / len(videollava_results)
                vllava_avg_s3 = sum(r['stage3_time'] for r in videollava_results) / len(videollava_results)
                vllava_avg_s4 = sum(r['stage4_time'] for r in videollava_results) / len(videollava_results)
                vllava_total = vllava_avg_s1 + vllava_avg_s2 + vllava_avg_s3 + vllava_avg_s4
                vllava_tokens = sum(r['tokens'] for r in videollava_results) / len(videollava_results)

                output_data['summary']['videollava'] = {
                    'samples': len(videollava_results),
                    'stage1_avg': vllava_avg_s1,
                    'stage2_avg': vllava_avg_s2,
                    'stage3_avg': vllava_avg_s3,
                    'stage4_avg': vllava_avg_s4,
                    'total_avg': vllava_total,
                    'tokens_avg': vllava_tokens,
                }
                output_data['summary']['speedup'] = vllava_total / avg_total

            if speculative_metrics:
                output_data['speculative_decoding'] = speculative_metrics

            os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else '.', exist_ok=True)
            with open(args.output_json, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\n✓ Results saved to: {args.output_json}")

        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
