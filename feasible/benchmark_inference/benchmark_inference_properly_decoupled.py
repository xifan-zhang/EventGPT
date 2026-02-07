#!/usr/bin/env python3
"""
Benchmark Inference Script with Properly Decoupled Stages
==========================================================

This script demonstrates the proper decoupling of Stage 3 (Vision Encoding)
and Stage 4 (LLM Decoding) for accurate benchmarking.

## 4-Stage Timing Breakdown (Properly Decoupled)

### EventGPT (Can be fully decoupled)
- Stage 1 (Load): Load event images from disk
- Stage 2 (Preprocess): CLIP preprocessing + tokenization
- Stage 3 (Vision): Call model.visval_encode(event_tensors)
- Stage 4 (Generate): Call model.generate(input_ids, event_features=<cached>)

### Video-LLaVA (Forward hooks for measurement)
- Stage 1 (Load): Load MP4 video frames
- Stage 2 (Preprocess): Video processor tokenization
- Stage 3 (Vision): Measured via forward hooks
- Stage 4 (Generate): LLM token generation

### LLaVA 1.5 (Forward hooks for measurement)
- Stage 1 (Load): Load image frames
- Stage 2 (Preprocess): CLIP processor tokenization
- Stage 3 (Vision): Measured via forward hooks
- Stage 4 (Generate): LLM token generation
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

    for sample_idx, sample in enumerate(tqdm(dataset[:10], desc="EventGPT Decoupled Test")):
        try:
            event_data = sample.get("event_data")
            if not event_data:
                continue

            # Stage 1: Load event images
            if "event_image" in sample and sample["event_image"]:
                event_image_paths = sample["event_image"]
                from benchmark_inference_4stages import load_preprocessed_event_images
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
                "tokens": len(generated_ids)
            })

            print(f"\nSample {sample_idx}: S1={stage1_time:.4f}s | S2={stage2_time:.4f}s | "
                  f"S3={stage3_time:.4f}s | S4={stage4_time:.4f}s | Total={stage1_time+stage2_time+stage3_time+stage4_time:.4f}s | "
                  f"Tokens={len(generated_ids)}")

        except Exception as e:
            print(f"Error on sample {sample_idx}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Properly Decoupled Benchmark")
    parser.add_argument("--dataset_dir", type=str, default="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s")
    parser.add_argument("--eventgpt_model_path", type=str, default="./checkpoints/EventGPT-7b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=10)

    args = parser.parse_args()

    # Load dataset
    dataset_json = os.path.join(args.dataset_dir, "EventGPT_Instruction_Subset.json")
    with open(dataset_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    print(f"Loading dataset with {len(dataset)} samples...")

    # Load EventGPT model
    print(f"Loading EventGPT from {args.eventgpt_model_path}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.eventgpt_model_path, use_fast=False)
    model = EventChatModel.from_pretrained(args.eventgpt_model_path, torch_dtype=torch.bfloat16)
    model = model.to(args.device)
    model.eval()

    processor = model.get_visual_tower().event_processor

    # Run benchmark
    results = run_eventgpt_decoupled_benchmark(
        model, tokenizer, processor, args.dataset_dir, dataset, args.device
    )

    # Print summary
    if results:
        avg_s1 = sum(r['stage1_time'] for r in results) / len(results)
        avg_s2 = sum(r['stage2_time'] for r in results) / len(results)
        avg_s3 = sum(r['stage3_time'] for r in results) / len(results)
        avg_s4 = sum(r['stage4_time'] for r in results) / len(results)
        avg_tokens = sum(r['tokens'] for r in results) / len(results)

        print(f"\n{'='*80}")
        print("PROPERLY DECOUPLED BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"Stage 1 (Load):                    {avg_s1:.4f}s")
        print(f"Stage 2 (Preprocess):              {avg_s2:.4f}s")
        print(f"Stage 3 (Vision Encoding):         {avg_s3:.4f}s  ✓ DECOUPLED")
        print(f"Stage 4 (LLM Decoding):            {avg_s4:.4f}s  ✓ DECOUPLED")
        print(f"{'─'*80}")
        print(f"Total per sample:                  {avg_s1 + avg_s2 + avg_s3 + avg_s4:.4f}s")
        print(f"Average tokens generated:          {avg_tokens:.1f}")
        print(f"\nKey insight: Stage 3 ({avg_s3:.4f}s) vs Stage 4 ({avg_s4:.4f}s)")
        print(f"Vision encoding is {avg_s4/avg_s3:.1f}x slower than LLM decoding")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
