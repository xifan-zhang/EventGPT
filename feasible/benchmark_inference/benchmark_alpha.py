#!/usr/bin/env python3
"""
Benchmark Inference Script for EventGPT and Llava7B Models
===========================================================

This script performs benchmark inference on a dataset containing both event data
and video data, running EventGPT on event sequences and Llava7B on video frames.

## Overview

The script loads a JSON dataset and processes each sample:
- **Event data** (.npy files): Processed with EventGPT model
- **Video data** (folders of images): Processed with Llava7B model
  - Images are loaded as a video sequence, sorted numerically by filename
  - Supports .png, .jpg, .jpeg formats

## Dataset Format

The input JSON file should contain entries with the following structure:

```json
{
  "id": "unique-id",
  "split": "dataset-split-name",
  "event_data": "path/to/event.npy",
  "video_data": "path/to/video/folder",
  "conversations": [
    {"from": "human", "value": "question"},
    {"from": "gpt", "value": "answer"}
  ]
}
```

## Output Format

Results are saved to a JSON file with the following structure:

```json
{
  "id": "unique-id",
  "split": "dataset-split-name",
  "query": "What are the key elements in this scene?",
  "event_data": "path/to/event.npy",
  "egpt": "EventGPT model output...",
  "video_data": "path/to/video/folder",
  "llava-1.5-7b-hf": "Llava7B model output..."
}
```

## Usage

### Basic Usage

```bash
python benchmark_inference.py \
    --dataset_json /mnt/hdd/data/my_egpt_dsec_seq_5s/EventGPT_Instruction_Subset.json \
    --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s \
    --eventgpt_model_path ./checkpoints/EventGPT-7b \
    --llava_model_path llava-hf/llava-1.5-7b-hf
```

### With Custom Query

```bash
python benchmark_inference.py \
    --dataset_json /path/to/dataset.json \
    --query "Describe the motion in this scene" \
    --max_samples 10
```

### All Options

```bash
python benchmark_inference.py \
    --dataset_json <path>              # Path to dataset JSON file
    --dataset_dir <path>               # Base directory of dataset
    --eventgpt_model_path <path>       # Path to EventGPT model
    --llava_model_path <path>          # Path to Llava7B model (HuggingFace ID or local)
    --output_json <path>               # Output JSON file (default: {dataset_json}_results.json)
    --max_samples <int>                 # Limit number of samples (for testing)
    --temperature <float>              # Generation temperature (default: 0.6)
    --top_p <float>                    # Top-p sampling (default: 1.0)
    --max_new_tokens <int>             # Max tokens to generate (default: 512)
    --device <cuda|cpu>                # Device to use (default: cuda)
    --query <string>                   # Query/prompt for inference
```

## Requirements

- PyTorch
- transformers
- EventGPT model checkpoint
- Llava7B model (from HuggingFace or local)
- Dataset with event .npy files and video image folders

## Model Loading

The script loads models similar to `feasible/tokenizer_check/tokenizer_check.py`:

- **EventGPT**: Loads model, tokenizer, and event processor with special token handling
- **Llava7B**: Uses Llava's model builder to load model, tokenizer, and image processor

## Video Processing

Video data is processed as a sequence of images:
- Images are sorted numerically by filename (e.g., 000000.png, 000001.png, ...)
- All images in the folder are loaded and processed together
- Supports multiple image formats (PNG, JPG, JPEG, case-insensitive)

## Error Handling

- Missing files are reported in the output JSON with error messages
- Failed image loads are skipped with warnings
- Model loading errors are caught and reported

## Notes

- The script automatically detects which samples need EventGPT vs Llava7B
- Models are only loaded if needed (if samples contain the corresponding data type)
- Results include both successful outputs and error messages
- The default query is "What are the key elements in this scene?"

## Example Output

```json
[
  {
    "id": "sample-001",
    "split": "my_egpt_dsec_seq_5s",
    "query": "What are the key elements in this scene?",
    "event_data": "interlaken_00_a/000000.npy",
    "egpt": "The scene shows rapid motion with multiple objects...",
    "video_data": "interlaken_00_a/000000",
    "llava-1.5-7b-hf": "I can see a dynamic scene with..."
  }
]
```

## Author

EventGPT Benchmark Inference Script
"""

import os
import sys
import json
import argparse
import torch
import re
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

# Add project root to path (two levels up from feasible/benchmark_inference)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.EventChatModel import EventChatModel
from common.common import tokenizer_event_token, process_event_data, load_image
from dataset.conversation import conv_templates, prepare_event_prompt
from dataset.constants import (
    EVENT_TOKEN_INDEX,
    DEFAULT_EVENT_TOKEN,
    DEFAULT_EV_START_TOKEN,
    DEFAULT_EV_END_TOKEN,
    EVENT_PLACEHOLDER,
    DEFAULT_EVENT_PATCH_TOKEN,
)

# Add src to path for llava imports
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

LLAVA_AVAILABLE = True


def load_eventgpt_model(model_path: str, device: str = "cuda"):
    """Load EventGPT model following the pattern from tokenizer_check.py and inference.py."""
    print(f"Loading EventGPT model from {model_path}...")
    
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = EventChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        config=config,
    )
    
    # Add special tokens if needed
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_EVENT_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN], special_tokens=True)
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Get event processor
    vision_tower = model.get_visual_tower()
    event_processor = vision_tower.event_processor
    
    # Move to device
    model.to(device)
    model.eval()
    
    print("EventGPT model loaded successfully.")
    return model, tokenizer, event_processor


def load_llava7b_model(model_path: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
    """Load Llava7B model using transformers."""
    print(f"Loading Llava7B model from {model_path}...")
    
    # Load processor with use_fast=False to avoid tokenizer errors
    try:
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    except TypeError:
        # Fallback for older transformers or if kwarg not accepted
        from transformers import AutoImageProcessor, LlavaProcessor
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )
    
    model.to(device)
    model.eval()
    
    print("Llava7B model loaded successfully.")
    return model, processor


def natural_sort_key(path):
    """Natural sort key for sorting filenames numerically (e.g., 000000.png, 000001.png)."""
    path_str = str(path)
    # Extract numbers from the filename
    numbers = re.findall(r'\d+', os.path.basename(path_str))
    if numbers:
        # Use the last number found (usually the frame number)
        return int(numbers[-1])
    return 0


def load_images_from_folder(video_folder_path):
    """Load all images from a video folder as a video sequence.
    
    Images are sorted naturally by filename (numerically) to maintain temporal order.
    Supports .png, .jpg, .jpeg formats.
    """
    video_path = Path(video_folder_path)
    
    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(video_path.glob(ext))
    
    if not image_files:
        raise ValueError(f"No images found in {video_folder_path}")
    
    # Sort images naturally by filename (numerically)
    # This ensures frames like 000000.png, 000001.png, 000002.png are in correct order
    image_files = sorted(image_files, key=natural_sort_key)
    
    print(f"Loading {len(image_files)} images from {video_folder_path} as video sequence...")
    
    images = []
    for image_file in image_files:
        try:
            image = load_image(str(image_file))
            images.append(image)
        except Exception as e:
            print(f"Warning: Failed to load {image_file}: {e}")
            continue
    
    if not images:
        raise ValueError(f"No valid images could be loaded from {video_folder_path}")
    
    print(f"Successfully loaded {len(images)} images as video sequence.")
    return images


def run_eventgpt_inference(
    model, tokenizer, event_processor, 
    event_data_path, query, 
    dataset_dir, device="cuda",
    temperature=0.6, top_p=1.0, max_new_tokens=512
):
    """Run EventGPT inference on event data."""
    full_event_path = os.path.join(dataset_dir, "event_npy", event_data_path)
    
    if not os.path.exists(full_event_path):
        return None, f"Event file not found: {full_event_path}"
    
    try:
        # Prepare prompt
        conv_mode = 'eventgpt_v1'
        prompt = prepare_event_prompt(query, conv_mode)
        
        # Process event data
        event_image_size, event_tensor = process_event_data(full_event_path, event_processor, device)
        
        # Tokenize
        input_ids = tokenizer_event_token(
            prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)
        
        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                event_tensors=event_tensor,
                event_image_sizes=event_image_size,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output, None
        
    except Exception as e:
        return None, str(e)


def run_llava_inference(
    model, processor,
    video_data_path, query,
    dataset_dir, device="cuda",
    temperature=0.2, top_p=1.0, max_new_tokens=512
):
    """Run Llava7B inference using transformers."""
    full_video_path = os.path.join(dataset_dir, "video", video_data_path)
    
    if not os.path.exists(full_video_path):
        return None, f"Video folder not found: {full_video_path}"
    
    try:
        # Load images
        images = load_images_from_folder(full_video_path)
        
        # Prepare messages
        content = []
        # Add images
        for img in images:
            content.append({"type": "image", "image": img})
        # Add text
        content.append({"type": "text", "text": query})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p
            )
        
        generated_text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        return generated_text, None
        
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference for EventGPT and Llava7B")
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="/mnt/hdd/data/my_egpt_dsec_seq_5s/EventGPT_Instruction_Subset.json",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/mnt/hdd/data/my_egpt_dsec_seq_5s",
        help="Base directory of the dataset"
    )
    parser.add_argument(
        "--eventgpt_model_path",
        type=str,
        default="./checkpoints/EventGPT-7b",
        help="Path to EventGPT model"
    )
    parser.add_argument(
        "--llava_model_path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Path to Llava7B model"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON file to save results (default: dataset_json with _results suffix)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the key elements in this scene?",
        help="Query/prompt to use for inference"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_json}...")
    with open(args.dataset_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples")
    
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for testing")
    
    # Load models
    eventgpt_model = None
    eventgpt_tokenizer = None
    event_processor = None
    
    llava_model = None
    llava_processor = None
    
    # Check if we need EventGPT (any sample has event_data)
    needs_eventgpt = any('event_data' in sample and sample.get('event_data') for sample in dataset)
    # Check if we need Llava (any sample has video_data)
    needs_llava = any('video_data' in sample and sample.get('video_data') for sample in dataset)
    
    if needs_eventgpt:
        eventgpt_model, eventgpt_tokenizer, event_processor = load_eventgpt_model(
            args.eventgpt_model_path, args.device
        )
    
    if needs_llava:
        llava_model, llava_processor = load_llava7b_model(
            args.llava_model_path, args.device
        )
    
    # Process samples
    results = []
    event_time_total = 0.0
    event_calls = 0
    video_time_total = 0.0
    video_calls = 0
    
    for sample in tqdm(dataset, desc="Processing samples"):
        result = {
            "id": sample.get("id", "unknown"),
            "split": sample.get("split", "unknown"),
            "query": args.query,
        }
        
        # Process event data if available
        if 'event_data' in sample and sample.get('event_data'):
            event_data_path = sample['event_data']
            # Measure EventGPT (draft model) inference time
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start_time = time.time()
            output, error = run_eventgpt_inference(
                eventgpt_model, eventgpt_tokenizer, event_processor,
                event_data_path, args.query,
                args.dataset_dir, args.device,
                args.temperature, args.top_p, args.max_new_tokens
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            event_time_total += elapsed
            event_calls += 1
            
            result['event_data'] = event_data_path
            result['egpt'] = output if output else None
            if error:
                result['egpt_error'] = error
        
        # Process video data if available
        if 'video_data' in sample and sample.get('video_data'):
            video_data_path = sample['video_data']
            # Measure Llava (target model) inference time
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start_time = time.time()
            output, error = run_llava_inference(
                llava_model, llava_processor,
                video_data_path, args.query,
                args.dataset_dir, args.device,
                args.temperature, args.top_p, args.max_new_tokens
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            video_time_total += elapsed
            video_calls += 1
            
            result['video_data'] = video_data_path
            result['llava-1.5-7b-hf'] = output if output else None
            if error:
                result['llava-1.5-7b-hf_error'] = error
        
        results.append(result)
    
    # Save results
    if args.output_json is None:
        base_name = os.path.splitext(args.dataset_json)[0]
        args.output_json = f"{base_name}_results.json"
    
    print(f"\nSaving results to {args.output_json}...")
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Timing summary and c value (draft / target)
    print("\nTiming summary:")
    if event_calls > 0:
        avg_event_time = event_time_total / event_calls
        print(
            f"  EventGPT (draft model): total {event_time_total:.2f}s over {event_calls} samples "
            f"(avg {avg_event_time:.3f}s/sample)"
        )
    else:
        avg_event_time = None
        print("  EventGPT (draft model): no samples processed.")
    
    if video_calls > 0:
        avg_video_time = video_time_total / video_calls
        print(
            f"  Llava (target model): total {video_time_total:.2f}s over {video_calls} samples "
            f"(avg {avg_video_time:.3f}s/sample)"
        )
    else:
        avg_video_time = None
        print("  Llava (target model): no samples processed.")
    
    if avg_event_time is not None and avg_video_time is not None and avg_video_time > 0:
        c_value = avg_event_time / avg_video_time
        print(
            f"\n  c (draft / target avg time ratio) = {c_value:.3f} "
            "(EventGPT as draft, Llava as target)"
        )
    else:
        print(
            "\n  c (draft / target avg time ratio) cannot be computed "
            "because one of the models has no valid timing data."
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

