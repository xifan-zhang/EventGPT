#!/usr/bin/env python3
"""
Benchmark Inference Script for EventGPT and Llava7B Models
===========================================================

This script performs benchmark inference on a dataset containing both event data
and video data, running EventGPT on event sequences and Llava7B on video frames.
Now using HuggingFace Transformers for Llava.

## 3-Stage Timing Breakdown (both models)
- Stage 1 (Load): Load data from disk (event .npy/images or video frames)
- Stage 2 (Preprocess): Convert to tensors, CLIP preprocessing, tokenization
- Stage 3 (Generate): Model generation (vision encoding + LLM inference)

## Key Metrics
- c = EventGPT_total / Llava_total (overall ratio)
- c_generation = EGPT_Stage3 / Llava_Stage3 (generation-only ratio, excludes load/preprocess)

## Usage
python benchmark_inference.py \\
    --dataset_json /path/to/data.json \\
    --dataset_dir /path/to/data \\
    --eventgpt_model_path ./checkpoints/EventGPT-7b \\
    --llava_model_path llava-hf/llava-1.5-7b-hf

## With Preprocessed Event Images (faster Stage 1)
# First, preprocess event images:
python feasible/my_egpt_dsec_dataset/preprocess_event_images.py --data_dir /path/to/data

# Then run benchmark with --use_event_image flag:
python benchmark_inference.py --dataset_dir /path/to/data --use_event_image
"""

import os
import sys
import json
import argparse
import torch
import re
import time
import gc
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    LlavaForConditionalGeneration,
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
)

# Video-LLaVA is always available with modern transformers
VIDEO_LLAVA_AVAILABLE = True

# Set up logger
logger = logging.getLogger(__name__)


def get_gpu_memory_mb(device="cuda"):
    """Get current GPU memory usage in MB."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_gpu_memory_peak_mb(device="cuda"):
    """Get peak GPU memory usage in MB."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def reset_gpu_memory_peak(device="cuda"):
    """Reset peak GPU memory tracker."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# EventGPT imports
from model.EventChatModel import EventChatModel
from common.common import tokenizer_event_token, process_event_data, load_image
from dataset.conversation import prepare_event_prompt
from dataset.constants import (
    EVENT_TOKEN_INDEX,
    DEFAULT_EVENT_PATCH_TOKEN,
    DEFAULT_EV_START_TOKEN,
    DEFAULT_EV_END_TOKEN,
)

def load_eventgpt_model(model_path: str, device: str = "cuda", clip_model_path: str = None):
    """Load EventGPT model."""
    print(f"Loading EventGPT model from {model_path}...")
    
    config = AutoConfig.from_pretrained(model_path)
    
    # Fix CLIP model path if needed
    if clip_model_path:
        config.mm_visual_tower = clip_model_path
    elif hasattr(config, 'mm_visual_tower') and config.mm_visual_tower:
        visual_tower = config.mm_visual_tower
        if os.path.exists(visual_tower):
            pass
        elif os.path.exists(os.path.join(ROOT, "checkpoints", visual_tower)):
            config.mm_visual_tower = os.path.join(ROOT, "checkpoints", visual_tower)
        elif visual_tower == "clip-vit-large-patch14-336":
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
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_EVENT_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN], special_tokens=True)
    
    model.resize_token_embeddings(len(tokenizer))
    
    vision_tower = model.get_visual_tower()
    event_processor = vision_tower.event_processor
    
    model.to(device)
    model.eval()
    
    print("EventGPT model loaded successfully.")
    return model, tokenizer, event_processor


def load_llava15_model(model_path: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
    """Load Llava 1.5 7B model using LlavaForConditionalGeneration with float16.

    BACKUP: This is the original LLaVA 1.5 implementation using grid images.
    """
    from transformers import CLIPImageProcessor, LlavaProcessor
    print(f"Loading Llava 1.5 model from {model_path}...")

    try:
        # Manually construct processor to avoid version compatibility issues
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        image_processor = CLIPImageProcessor.from_pretrained(model_path)
        processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        print("Llava 1.5 model loaded successfully.")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading Llava 1.5 model: {e}")
        raise


def load_video_llava_model(model_path: str = "LanguageBind/Video-LLaVA-7B-hf", device: str = "cuda"):
    """Load Video-LLaVA 7B model for native video understanding.

    Video-LLaVA is trained on video data and handles 8 frames natively.
    Requires transformers >= 4.40.
    """
    if not VIDEO_LLAVA_AVAILABLE:
        raise ImportError(
            "Video-LLaVA requires transformers >= 4.40. "
            f"Current version: {__import__('transformers').__version__}. "
            "Please upgrade with: pip install -U transformers"
        )

    print(f"Loading Video-LLaVA model from {model_path}...")

    try:
        # Load with float16 to fit in 24GB GPU
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
        processor = VideoLlavaProcessor.from_pretrained(model_path)
        model.to(device)
        print("Video-LLaVA model loaded successfully.")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading Video-LLaVA model: {e}")
        raise

def natural_sort_key(path):
    """Natural sort key for sorting filenames numerically."""
    path_str = str(path)
    numbers = re.findall(r'\d+', os.path.basename(path_str))
    if numbers:
        return int(numbers[-1])
    return 0


def create_image_grid(images, cols=4):
    """Create a grid image from multiple PIL images.

    LLaVA 1.5 doesn't handle multiple images well, so we concatenate
    video frames into a single grid image for better results.

    Args:
        images: List of PIL Images
        cols: Number of columns in the grid (default 4 for 2x4 grid with 8 frames)

    Returns:
        PIL Image: Single grid image containing all frames
    """
    if not images:
        raise ValueError("No images provided")

    if len(images) == 1:
        return images[0]

    # Calculate grid dimensions
    n = len(images)
    rows = (n + cols - 1) // cols  # Ceiling division

    # Get dimensions from first image
    w, h = images[0].size

    # Create blank canvas
    grid_img = Image.new('RGB', (cols * w, rows * h), color=(0, 0, 0))

    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        # Resize if needed to match first image dimensions
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
        grid_img.paste(img, (col * w, row * h))

    return grid_img


def load_images_from_folder(video_folder_path, max_frames=8):
    """Load images from a video folder, sampling evenly if there are too many.

    Args:
        video_folder_path: Path to folder containing images
        max_frames: Maximum number of frames to load (default 8)
    """
    video_path = Path(video_folder_path)
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(video_path.glob(ext))

    if not image_files:
        raise ValueError(f"No images found in {video_folder_path}")

    image_files = sorted(image_files, key=natural_sort_key)

    # Sample evenly if too many frames
    if len(image_files) > max_frames:
        indices = [int(i * (len(image_files) - 1) / (max_frames - 1)) for i in range(max_frames)]
        image_files = [image_files[i] for i in indices]

    images = []
    for image_file in image_files:
        try:
            image = load_image(str(image_file))
            images.append(image)
        except Exception as e:
            logger.warning(f"Failed to load {image_file}: {e}")
            continue

    if not images:
        raise ValueError(f"No valid images could be loaded from {video_folder_path}")

    return images


def load_preprocessed_event_images(event_image_paths, event_processor, dataset_dir, device):
    """Load preprocessed event images and process them with event_processor.

    Args:
        event_image_paths: List of relative paths to event images (e.g., ["interlaken_00_a/000000_0.png", ...])
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


def run_eventgpt_inference(
    model, tokenizer, event_processor,
    event_data_path, query,
    dataset_dir, device="cuda",
    temperature=0.6, top_p=1.0, max_new_tokens=512,
    event_image_paths=None
):
    """Run EventGPT inference on event data with 3-stage timing.

    Stages:
        Stage 1 (Load): Load event data from disk (NPY or preprocessed images)
        Stage 2 (Preprocess): CLIP preprocessing + tokenization
        Stage 3 (Generate): model.generate() (vision encoding + LLM generation)

    Args:
        event_image_paths: Optional list of preprocessed event image paths.
                          If provided, skips NPY processing and loads images directly.

    Returns:
        output: Generated text
        error: Error message if any
        stage1_time: Time for loading data from disk
        stage2_time: Time for CLIP preprocessing + tokenization
        stage3_time: Time for model generation
    """
    import numpy as np

    # Check if using preprocessed images or raw .npy
    use_preprocessed = event_image_paths is not None and len(event_image_paths) > 0

    if not use_preprocessed:
        full_event_path = os.path.join(dataset_dir, "event_npy", event_data_path)
        if not os.path.exists(full_event_path):
            return None, f"Event file not found: {full_event_path}", None, None, None

    try:
        conv_mode = 'eventgpt_v1'
        prompt = prepare_event_prompt(query, conv_mode)

        if use_preprocessed:
            # Using preprocessed images - load_preprocessed_event_images handles stage1 & stage2
            event_image_size, event_tensor, stage1_time, stage2_time = load_preprocessed_event_images(
                event_image_paths, event_processor, dataset_dir, device
            )
            # Add tokenization time to stage2
            if device.startswith("cuda"): torch.cuda.synchronize()
            token_start = time.time()
            input_ids = tokenizer_event_token(
                prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(device)
            if device.startswith("cuda"): torch.cuda.synchronize()
            stage2_time += time.time() - token_start
        else:
            # Original NPY processing - need to separate load from preprocess
            full_event_path = os.path.join(dataset_dir, "event_npy", event_data_path)

            # Stage 1: Load NPY from disk
            if device.startswith("cuda"): torch.cuda.synchronize()
            stage1_start = time.time()
            event_npy = np.load(full_event_path, allow_pickle=True)
            event_npy = np.array(event_npy).item()
            if device.startswith("cuda"): torch.cuda.synchronize()
            stage1_time = time.time() - stage1_start

            # Stage 2: Convert to event images + CLIP preprocessing + tokenization
            if device.startswith("cuda"): torch.cuda.synchronize()
            stage2_start = time.time()
            # Import the event image generation function
            from common.common import get_event_images_list
            event_images = get_event_images_list(event_npy)
            event_image_size = list(event_images[0].shape[:2])
            event_tensor = []
            for event_img in event_images:
                event = event_processor(event_img, return_tensors='pt')['pixel_values'][0]
                event = event.to(device, dtype=torch.bfloat16)
                event_tensor.append(event)
            input_ids = tokenizer_event_token(
                prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(device)
            if device.startswith("cuda"): torch.cuda.synchronize()
            stage2_time = time.time() - stage2_start

        # Stage 3: Generate output (vision encoding + LLM generation)
        if device.startswith("cuda"): torch.cuda.synchronize()
        stage3_start = time.time()
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
        if device.startswith("cuda"): torch.cuda.synchronize()
        stage3_time = time.time() - stage3_start

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # Return output_ids for acceptance rate calculation
        # Note: EventGPT may return only generated tokens OR input+generated
        # Check if output_ids is longer than input_ids
        if output_ids.shape[1] > input_ids.shape[1]:
            # Standard case: output includes input, extract generated portion
            generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
        else:
            # EventGPT returns only generated tokens
            generated_ids = output_ids[0].tolist()
        return output, None, stage1_time, stage2_time, stage3_time, generated_ids

    except Exception as e:
        return None, str(e), None, None, None, None


def run_llava15_inference(
    model, processor,
    video_data_path, query,
    dataset_dir, device="cuda",
    temperature=0.2, top_p=1.0, max_new_tokens=512
):
    """Run Llava 1.5 inference on video data with 3-stage timing.

    BACKUP: This is the original LLaVA 1.5 implementation using grid images.

    Stages:
        Stage 1 (Load): Load video frames from disk
        Stage 2 (Preprocess): Processor to create input tensors (includes image preprocessing)
        Stage 3 (Generate): model.generate() (vision encoding + LLM generation)

    Returns:
        output: Generated text
        error: Error message if any
        stage1_time: Time for loading frames from disk
        stage2_time: Time for processor preprocessing
        stage3_time: Time for model generation
    """
    full_video_path = os.path.join(dataset_dir, "video", video_data_path)

    if not os.path.exists(full_video_path):
        return None, f"Video folder not found: {full_video_path}", None, None, None

    # Stage 1: Load video frames from disk and create grid
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage1_start = time.time()
    images = load_images_from_folder(full_video_path, max_frames=8)
    # Create grid image from video frames (LLaVA 1.5 doesn't handle multiple images well)
    grid_image = create_image_grid(images, cols=4)
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage1_time = time.time() - stage1_start

    # Stage 2: Processor preprocessing (tokenization + image preprocessing)
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage2_start = time.time()

    # LLaVA 1.5 prompt format with single grid image
    prompt = f"USER: <image>\nThis image shows {len(images)} video frames arranged in a grid. {query}\nASSISTANT:"

    # Create inputs with float16
    inputs = processor(
        images=[grid_image],
        text=prompt,
        return_tensors="pt",
        padding=True
    ).to(device, torch.float16)
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage2_time = time.time() - stage2_start

    # Stage 3: Generate (vision encoding + LLM generation)
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage3_start = time.time()
    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            num_beams=1
        )
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage3_time = time.time() - stage3_start

    # Decode output using batch_decode (preferred method)
    decoded_outputs = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    output = decoded_outputs[0]

    # Extract only the assistant response (after ASSISTANT:)
    if "ASSISTANT:" in output:
        output = output.split("ASSISTANT:")[-1].strip()

    # Return generated token IDs for acceptance rate calculation (exclude input tokens)
    input_len = inputs['input_ids'].shape[1]
    generated_ids = generate_ids[0, input_len:].tolist()

    logger.debug(f"LLaVA 1.5 output: {output}")
    logger.debug(f"LLaVA 1.5 output length: {len(output)}")
    return output, None, stage1_time, stage2_time, stage3_time, generated_ids


def custom_video_llava_generate(model, processor, inputs, max_new_tokens=512, temperature=0.0):
    """Custom generation function that works around model.generate() bug in transformers 4.44.

    The model.generate() function in transformers 4.44.0 has a bug with Video-LLaVA
    that causes it to output garbage ("1 1 1 1..."). This custom implementation
    performs greedy/sampling decoding correctly.

    Args:
        model: VideoLlavaForConditionalGeneration model
        processor: VideoLlavaProcessor
        inputs: Dict with input_ids, attention_mask, pixel_values_videos
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)

    Returns:
        generated_ids: Tensor of generated token IDs
    """
    with torch.inference_mode():
        generated_ids = inputs['input_ids'].clone()
        past_key_values = None

        for step in range(max_new_tokens):
            outputs = model(
                input_ids=generated_ids if step == 0 else next_token.unsqueeze(0),
                pixel_values_videos=inputs.get('pixel_values_videos') if step == 0 else None,
                pixel_values_images=inputs.get('pixel_values_images') if step == 0 else None,
                attention_mask=torch.ones(1, generated_ids.shape[1] + step, device=generated_ids.device) if step > 0 else inputs.get('attention_mask'),
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == processor.tokenizer.eos_token_id:
                break

        return generated_ids


def read_video_pyav_from_mp4(mp4_path, max_frames=8):
    """
    Read video frames directly from an MP4 file using PyAV.
    This is the preferred method when pre-generated MP4 files are available.

    Args:
        mp4_path: Path to MP4 video file
        max_frames: Maximum number of frames to load

    Returns:
        np.ndarray: Video frames of shape (num_frames, height, width, 3)
    """
    import av
    import numpy as np

    container = av.open(mp4_path)
    total_frames = container.streams.video[0].frames

    # Handle case where total_frames is 0 or unknown
    if total_frames == 0:
        # Count frames manually
        frames_list = []
        for frame in container.decode(video=0):
            frames_list.append(frame)
        total_frames = len(frames_list)
        if total_frames == 0:
            raise ValueError(f"No frames found in {mp4_path}")
        # Sample uniformly
        if total_frames > max_frames:
            indices = [int(i * (total_frames - 1) / (max_frames - 1)) for i in range(max_frames)]
            frames_list = [frames_list[i] for i in indices]
        video = np.stack([x.to_ndarray(format="rgb24") for x in frames_list])
        container.close()
        return video

    # Sample uniformly to get exactly max_frames
    indices = np.arange(0, total_frames, total_frames / max_frames).astype(int)
    indices = indices[:max_frames]

    # Decode using PyAV
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    # Convert to numpy array using PyAV's rgb24 format
    video = np.stack([x.to_ndarray(format="rgb24") for x in frames])

    container.close()

    logger.debug(f"PyAV-decoded video from MP4: shape={video.shape}, dtype={video.dtype}, "
                f"min={video.min()}, max={video.max()}")

    return video


def read_video_pyav_from_images(image_folder, max_frames=8):
    """
    Create a temporary video from images and decode with PyAV.
    This matches the official Video-LLaVA example exactly.

    Args:
        image_folder: Path to folder containing video frames
        max_frames: Maximum number of frames to load

    Returns:
        np.ndarray: Video frames of shape (num_frames, height, width, 3)
    """
    import av
    import numpy as np
    import tempfile
    import shutil

    # Load images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(Path(image_folder).glob(ext))

    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    # Sort naturally
    image_files = sorted(image_files, key=natural_sort_key)

    # Sample evenly if too many frames
    if len(image_files) > max_frames:
        indices = [int(i * (len(image_files) - 1) / (max_frames - 1)) for i in range(max_frames)]
        image_files = [image_files[i] for i in indices]

    # Create temporary video file
    temp_dir = tempfile.mkdtemp()
    temp_video = os.path.join(temp_dir, "temp_video.mp4")

    try:
        # Create video from images using imageio
        import imageio
        images = []
        for img_path in image_files:
            img = load_image(str(img_path))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(np.array(img, dtype=np.uint8))

        # Write as MP4 video (10 fps, using libx264 codec)
        imageio.mimsave(temp_video, images, fps=10, codec='libx264', quality=8, pixelformat='yuv420p')

        # Now decode with PyAV (matching official Video-LLaVA example)
        container = av.open(temp_video)
        total_frames = container.streams.video[0].frames
        # Sample uniformly to get exactly max_frames
        indices = np.arange(0, total_frames, total_frames / max_frames).astype(int)
        indices = indices[:max_frames]  # Ensure we don't exceed max_frames

        # Decode using PyAV (exact same as official example)
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)

        # Convert to numpy array using PyAV's rgb24 format (exact match)
        video = np.stack([x.to_ndarray(format="rgb24") for x in frames])

        container.close()

        logger.debug(f"PyAV-decoded video: shape={video.shape}, dtype={video.dtype}, "
                    f"contiguous={video.flags['C_CONTIGUOUS']}, "
                    f"min={video.min()}, max={video.max()}")

        return video

    finally:
        # Cleanup temp files
        if os.path.exists(temp_video):
            os.remove(temp_video)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_video_llava_inference(
    model, processor,
    video_data_path, query,
    dataset_dir, device="cuda",
    temperature=0.2, top_p=1.0, max_new_tokens=512,
    mp4_data_path=None
):
    """Run Video-LLaVA inference on video data with 3-stage timing.

    Video-LLaVA is trained on video data and handles 8 frames natively via <video> token.
    This function follows the official HuggingFace Video-LLaVA example format.

    Reference: https://huggingface.co/docs/transformers/en/model_doc/video_llava

    Stages:
        Stage 1 (Load): Load video frames from disk as numpy array
        Stage 2 (Preprocess): Processor to create input tensors
        Stage 3 (Generate): model.generate() (vision encoding + LLM generation)

    Args:
        mp4_data_path: Optional path to pre-generated MP4 file (relative to dataset_dir/mp4/)
                       If provided and exists, loads from MP4 directly (faster).

    Returns:
        output: Generated text
        error: Error message if any
        stage1_time: Time for loading frames from disk
        stage2_time: Time for processor preprocessing
        stage3_time: Time for model generation
    """
    import numpy as np

    # Check for pre-generated MP4 first
    use_mp4 = False
    if mp4_data_path:
        full_mp4_path = os.path.join(dataset_dir, "mp4", mp4_data_path)
        if os.path.exists(full_mp4_path):
            use_mp4 = True
        else:
            logger.debug(f"MP4 not found, falling back to images: {full_mp4_path}")

    if not use_mp4:
        full_video_path = os.path.join(dataset_dir, "video", video_data_path)
        if not os.path.exists(full_video_path):
            return None, f"Video folder not found: {full_video_path}", None, None, None, None

    # Stage 1: Load video frames from disk as numpy array (8 frames)
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage1_start = time.time()

    # Load frames - prefer MP4 if available
    if use_mp4:
        video_frames = read_video_pyav_from_mp4(full_mp4_path, max_frames=8)
    else:
        video_frames = read_video_pyav_from_images(full_video_path, max_frames=8)

    logger.debug(f"Loaded {len(video_frames)} frames, shape: {video_frames.shape}, dtype: {video_frames.dtype}, contiguous: {video_frames.flags['C_CONTIGUOUS']}")
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage1_time = time.time() - stage1_start

    # Stage 2: Processor preprocessing (tokenization + video preprocessing)
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage2_start = time.time()

    # Video-LLaVA prompt format with <video> token (matching official HuggingFace example)
    # IMPORTANT: No newline before "ASSISTANT:" - this matches the official format
    prompt = f"USER: <video>\n{query} ASSISTANT:"

    logger.debug(f"Prompt: {prompt}")

    # Create inputs - Video-LLaVA expects videos parameter
    inputs = processor(
        text=prompt,
        videos=video_frames,
        return_tensors="pt",
        padding=True
    ).to(device, torch.float16)

    logger.debug(f"Input IDs shape: {inputs['input_ids'].shape}")
    logger.debug(f"Pixel values shape: {inputs.get('pixel_values_videos', 'N/A')}")
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage2_time = time.time() - stage2_start

    # Stage 3: Generate (vision encoding + LLM generation)
    # NOTE: Requires transformers >= 4.47.0 for model.generate() to work correctly
    # Earlier versions (e.g., 4.44.0) have a bug that produces garbage output
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage3_start = time.time()
    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
        )
    if device.startswith("cuda"): torch.cuda.synchronize()
    stage3_time = time.time() - stage3_start

    # Decode output using batch_decode (matching official HuggingFace example)
    # batch_decode is preferred over decode for Video-LLaVA
    decoded_outputs = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    output = decoded_outputs[0]

    # Extract only the assistant response (after ASSISTANT:)
    if "ASSISTANT:" in output:
        output = output.split("ASSISTANT:")[-1].strip()

    # Return generated token IDs for acceptance rate calculation (exclude input tokens)
    input_len = inputs['input_ids'].shape[1]
    generated_ids = generate_ids[0, input_len:].tolist()

    logger.debug(f"Output: {output}")
    logger.debug(f"Output length: {len(output)}")
    logger.debug(f"Generated tokens count: {len(generated_ids)}")
    return output, None, stage1_time, stage2_time, stage3_time, generated_ids


def run_eventgpt_phase(dataset, args):
    """Execute Phase 1: EventGPT Inference with 3-stage timing and memory tracking."""
    print("\n=== Phase 1: EventGPT Inference ===")
    model = None
    tokenizer = None
    processor = None
    memory_stats = {'model_load_mb': 0, 'inference_peak_mb': 0}

    try:
        # Reset peak memory before loading model
        reset_gpu_memory_peak(args.device)

        model, tokenizer, processor = load_eventgpt_model(
            args.eventgpt_model_path, args.device
        )

        # Record memory after model load
        memory_stats['model_load_mb'] = get_gpu_memory_peak_mb(args.device)
        print(f"EventGPT model GPU memory: {memory_stats['model_load_mb']:.1f} MB")

        # Reset peak for inference tracking
        reset_gpu_memory_peak(args.device)

        # Filter dataset to only include samples with event data
        event_samples = [s for s in dataset if 'event_data' in s and s['event_data']]

        # Warmup phase: run random samples
        if args.warmup_steps > 0 and len(event_samples) > 0:
            print(f"Running {args.warmup_steps} warmup samples...")
            import random
            warmup_samples = random.sample(event_samples, min(args.warmup_steps, len(event_samples)))

            warmup_pbar = tqdm(warmup_samples, desc="EventGPT Warmup")
            for sample in warmup_pbar:
                # Get event_image paths if using preprocessed images
                event_image_paths = sample.get('event_image') if args.use_event_image else None
                # Run warmup inference (results not saved)
                output, error, _, _, _, _ = run_eventgpt_inference(
                    model, tokenizer, processor,
                    sample['event_data'], args.query,
                    args.dataset_dir, args.device,
                    args.temperature, args.top_p, args.max_new_tokens,
                    event_image_paths=event_image_paths
                )
                if error:
                    raise RuntimeError(f"EventGPT warmup error for sample {sample.get('id')}: {error}")
                warmup_pbar.set_postfix({'status': 'warmup'})
            print("Warmup completed.")

        # Main inference phase
        desc = "EventGPT Inference (preprocessed)" if args.use_event_image else "EventGPT Inference"
        pbar = tqdm(event_samples, desc=desc)
        for sample in pbar:
            if 'event_data' in sample and sample['event_data']:
                try:
                    # Get event_image paths if using preprocessed images
                    event_image_paths = sample.get('event_image') if args.use_event_image else None
                    if args.device.startswith("cuda"): torch.cuda.synchronize()
                    start_time = time.time()
                    output, error, stage1_time, stage2_time, stage3_time, generated_ids = run_eventgpt_inference(
                        model, tokenizer, processor,
                        sample['event_data'], args.query,
                        args.dataset_dir, args.device,
                        args.temperature, args.top_p, args.max_new_tokens,
                        event_image_paths=event_image_paths
                    )
                    if args.device.startswith("cuda"): torch.cuda.synchronize()
                    elapsed = time.time() - start_time

                    sample['egpt'] = output
                    sample['egpt_time'] = elapsed
                    sample['egpt_token_ids'] = generated_ids
                    if stage1_time is not None:
                        sample['egpt_stage1_time'] = stage1_time
                    if stage2_time is not None:
                        sample['egpt_stage2_time'] = stage2_time
                    if stage3_time is not None:
                        sample['egpt_stage3_time'] = stage3_time
                    if error:
                        raise RuntimeError(f"EventGPT inference error for sample {sample.get('id')}: {error}")

                    # Update progress bar with stage timing
                    postfix_info = {'total': f'{elapsed:.3f}s'}

                    if stage1_time is not None and stage2_time is not None and stage3_time is not None:
                        postfix_info.update({
                            'S1': f'{stage1_time:.3f}s',
                            'S2': f'{stage2_time:.3f}s',
                            'S3': f'{stage3_time:.3f}s'
                        })

                    pbar.set_postfix(postfix_info)

                except Exception as inner_e:
                    raise RuntimeError(f"Error processing EventGPT sample {sample.get('id')}: {inner_e}") from inner_e

        # Record peak inference memory
        memory_stats['inference_peak_mb'] = get_gpu_memory_peak_mb(args.device)
        print(f"EventGPT inference peak GPU memory: {memory_stats['inference_peak_mb']:.1f} MB")

        # Store memory stats in first sample for retrieval
        if event_samples:
            event_samples[0]['egpt_memory_model_mb'] = memory_stats['model_load_mb']
            event_samples[0]['egpt_memory_inference_mb'] = memory_stats['inference_peak_mb']

    except Exception as e:
        raise RuntimeError(f"Error in EventGPT phase: {e}") from e
    finally:
        # cleanup
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if processor is not None:
            del processor

        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("EventGPT model unloaded and memory cleaned.")


def run_llava_phase(dataset, args):
    """Execute Phase 2: Llava Inference with 3-stage timing and memory tracking.

    Supports both Video-LLaVA (default) and LLaVA 1.5 (backup) via --use_llava15 flag.
    """
    # Use Video-LLaVA only if explicitly requested, otherwise default to LLaVA 1.5
    use_video_llava = getattr(args, 'use_video_llava', False)
    model_name = "Video-LLaVA" if use_video_llava else "LLaVA 1.5"
    print(f"\n=== Phase 2: {model_name} Inference ===")
    model = None
    processor = None
    memory_stats = {'model_load_mb': 0, 'inference_peak_mb': 0}

    try:
        # Reset peak memory before loading model
        reset_gpu_memory_peak(args.device)

        if use_video_llava:
            model, processor = load_video_llava_model(
                args.video_llava_model_path, args.device
            )
        else:
            model, processor = load_llava15_model(
                args.llava_model_path, args.device
            )

        # Record memory after model load
        memory_stats['model_load_mb'] = get_gpu_memory_peak_mb(args.device)
        print(f"Llava model GPU memory: {memory_stats['model_load_mb']:.1f} MB")

        # Reset peak for inference tracking
        reset_gpu_memory_peak(args.device)

        # Filter dataset to only include samples with video data
        video_samples = [s for s in dataset if 'video_data' in s and s['video_data']]

        # Select inference function based on model type
        if use_video_llava:
            run_inference_fn = run_video_llava_inference
            warmup_desc = "Video-LLaVA Warmup"
            inference_desc = "Video-LLaVA Inference"
        else:
            run_inference_fn = run_llava15_inference
            warmup_desc = "LLaVA 1.5 Warmup"
            inference_desc = "LLaVA 1.5 Inference"

        # Warmup phase: run random samples
        if args.warmup_steps > 0 and len(video_samples) > 0:
            print(f"Running {args.warmup_steps} warmup samples...")
            import random
            warmup_samples = random.sample(video_samples, min(args.warmup_steps, len(video_samples)))

            warmup_pbar = tqdm(warmup_samples, desc=warmup_desc)
            for sample in warmup_pbar:
                # Run warmup inference (results not saved)
                # Pass mp4_data for Video-LLaVA if available
                extra_kwargs = {}
                if use_video_llava and 'mp4_data' in sample:
                    extra_kwargs['mp4_data_path'] = sample['mp4_data']
                output, error, _, _, _, _ = run_inference_fn(
                    model, processor,
                    sample['video_data'], args.query,
                    args.dataset_dir, args.device,
                    args.temperature, args.top_p, args.max_new_tokens,
                    **extra_kwargs
                )
                if error:
                    raise RuntimeError(f"{model_name} warmup error for sample {sample.get('id')}: {error}")
                warmup_pbar.set_postfix({'status': 'warmup'})
            print("Warmup completed.")

        # Main inference phase
        pbar = tqdm(video_samples, desc=inference_desc)
        for sample in pbar:
            if 'video_data' in sample and sample['video_data']:
                if args.device.startswith("cuda"): torch.cuda.synchronize()
                start_time = time.time()
                # Pass mp4_data for Video-LLaVA if available
                extra_kwargs = {}
                if use_video_llava and 'mp4_data' in sample:
                    extra_kwargs['mp4_data_path'] = sample['mp4_data']
                output, error, stage1_time, stage2_time, stage3_time, generated_ids = run_inference_fn(
                    model, processor,
                    sample['video_data'], args.query,
                    args.dataset_dir, args.device,
                    args.temperature, args.top_p, args.max_new_tokens,
                    **extra_kwargs
                )
                if args.device.startswith("cuda"): torch.cuda.synchronize()
                elapsed = time.time() - start_time

                if error:
                    raise RuntimeError(f"{model_name} inference error for sample {sample.get('id')}: {error}")

                sample['llava-1.5-7b-hf'] = output
                sample['llava_time'] = elapsed
                sample['llava_token_ids'] = generated_ids
                if stage1_time is not None:
                    sample['llava_stage1_time'] = stage1_time
                if stage2_time is not None:
                    sample['llava_stage2_time'] = stage2_time
                if stage3_time is not None:
                    sample['llava_stage3_time'] = stage3_time

                # Update progress bar with stage timing
                postfix_info = {'total': f'{elapsed:.3f}s'}
                if stage1_time is not None and stage2_time is not None and stage3_time is not None:
                    postfix_info.update({
                        'S1': f'{stage1_time:.3f}s',
                        'S2': f'{stage2_time:.3f}s',
                        'S3': f'{stage3_time:.3f}s'
                    })
                pbar.set_postfix(postfix_info)

        # Record peak inference memory
        memory_stats['inference_peak_mb'] = get_gpu_memory_peak_mb(args.device)
        print(f"Llava inference peak GPU memory: {memory_stats['inference_peak_mb']:.1f} MB")

        # Store memory stats in first sample for retrieval
        if video_samples:
            video_samples[0]['llava_memory_model_mb'] = memory_stats['model_load_mb']
            video_samples[0]['llava_memory_inference_mb'] = memory_stats['inference_peak_mb']

    except Exception as e:
        import traceback
        logger.error(f"Error in Llava phase: {e}")
        traceback.print_exc()
    finally:
        # cleanup
        if model is not None:
            del model
        if processor is not None:
            del processor

        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"{model_name} model unloaded and memory cleaned.")


def calculate_acceptance_rate(draft_tokens, target_tokens):
    """Calculate token-level acceptance rate between draft and target model outputs.

    Args:
        draft_tokens: List of token IDs from draft model (EventGPT)
        target_tokens: List of token IDs from target model (LLaVA)

    Returns:
        acceptance_rate: Fraction of draft tokens that match target tokens
        matched_count: Number of matched tokens
        total_count: Total number of tokens compared
    """
    if not draft_tokens or not target_tokens:
        return 0.0, 0, 0

    # Compare tokens up to the shorter length
    min_len = min(len(draft_tokens), len(target_tokens))
    if min_len == 0:
        return 0.0, 0, 0

    matched = sum(1 for i in range(min_len) if draft_tokens[i] == target_tokens[i])
    acceptance_rate = matched / min_len

    return acceptance_rate, matched, min_len


def calculate_speculative_speedup(c, alpha, gamma):
    """Calculate theoretical speedup from speculative decoding.

    Args:
        c: Draft/Target time ratio (T_draft / T_target)
        alpha: Acceptance rate (probability draft token is accepted)
        gamma: Number of draft tokens per verification step

    Returns:
        speedup: Theoretical speedup factor
    """
    if alpha >= 1.0:
        alpha = 0.99  # Avoid division issues
    if c <= 0:
        return 0.0

    # Expected number of accepted tokens per step
    # E[accepted] = (1 - alpha^(gamma+1)) / (1 - alpha)
    numerator = 1 - (alpha ** (gamma + 1))
    denominator = (1 - alpha)

    if denominator == 0:
        return 1.0 / c  # Perfect acceptance

    expected_accepted = numerator / denominator

    # Cost per step: gamma draft evaluations + 1 target evaluation
    # In terms of target time: c * gamma + 1
    cost_per_step = c * gamma + 1

    # Speedup = expected_accepted / cost_per_step
    speedup = expected_accepted / cost_per_step

    return speedup


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference time for EventGPT and Llava7B")
    parser.add_argument("--dataset_dir", type=str, default="/mnt/hdd/data/my_egpt_dsec_seq_5s")
    parser.add_argument("--dataset_json", type=str, default=None, help="Defaults to dataset_dir/EventGPT_Instruction_Subset.json")
    parser.add_argument("--eventgpt_model_path", type=str, default="./checkpoints/EventGPT-7b")
    parser.add_argument("--llava_model_path", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="LLaVA 1.5 model path (used with --use_llava15)")
    parser.add_argument("--video_llava_model_path", type=str, default="LanguageBind/Video-LLaVA-7B-hf",
                        help="Video-LLaVA model path (default target model)")
    parser.add_argument("--use_llava15", action="store_true", default=True,
                        help="Use LLaVA 1.5 with grid images (default, recommended)")
    parser.add_argument("--use_video_llava", action="store_true",
                        help="Use Video-LLaVA instead of LLaVA 1.5 (requires transformers>=4.40, experimental)")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup samples (not included in timing), 0 no warmup")
    parser.add_argument(
        "--first_model",
        type=str,
        default="llava",
        choices=["llava", "eventgpt"],
        help="Which model to run first (for debugging): 'llava' or 'eventgpt'"
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--query", type=str, default="What are the key elements in this scene?")
    parser.add_argument("--gamma", type=int, default=5, help="Number of draft tokens per verification step for speculative decoding")
    parser.add_argument(
        "--use_event_image",
        action="store_true",
        default=True,
        help="(Default: True) Use preprocessed event images from event_image/ folder. "
             "Requires running preprocess_event_images.py first."
    )
    parser.add_argument(
        "--no_event_image",
        action="store_true",
        help="Disable preprocessed event images, process .npy files directly (slower)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for troubleshooting"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.setLevel(log_level)

    # Handle --no_event_image flag
    if args.no_event_image:
        args.use_event_image = False

    # Resolve dataset_json path
    if args.dataset_json is None:
        args.dataset_json = os.path.join(args.dataset_dir, "EventGPT_Instruction_Subset.json")
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_json}...")
    with open(args.dataset_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    
    # Determine which models are needed
    needs_llava = any('video_data' in sample and sample['video_data'] for sample in dataset)
    needs_eventgpt = any('event_data' in sample and sample['event_data'] for sample in dataset)

    # Validate event_image availability if --use_event_image is enabled
    if args.use_event_image and needs_eventgpt:
        event_samples = [s for s in dataset if 'event_data' in s and s['event_data']]
        samples_with_images = sum(1 for s in event_samples if 'event_image' in s and s['event_image'])
        if samples_with_images == 0:
            print("ERROR: --use_event_image is enabled but no samples have 'event_image' paths.")
            print("       Run preprocess_event_images.py first to generate event images.")
            sys.exit(1)
        elif samples_with_images < len(event_samples):
            print(f"WARNING: Only {samples_with_images}/{len(event_samples)} samples have 'event_image' paths.")
            print("         Samples without event_image will fall back to processing .npy files.")
        else:
            print(f"Using preprocessed event images for {samples_with_images} samples.")

    # Run models in the user-specified order
    if args.first_model == "llava":
        if needs_llava:
            run_llava_phase(dataset, args)
        if needs_eventgpt:
            run_eventgpt_phase(dataset, args)
    else:  # first_model == "eventgpt"
        if needs_eventgpt:
            run_eventgpt_phase(dataset, args)
        if needs_llava:
            run_llava_phase(dataset, args)

    # Prepare final results
    results = []
    for sample in dataset:
        res = {
            "id": sample.get("id", "unknown"),
            "split": sample.get("split", "unknown"),
            "query": args.query,
        }
        if 'event_data' in sample:
            res['event_data'] = sample.get('event_data')
            res['egpt'] = sample.get('egpt')
            if 'egpt_error' in sample: res['egpt_error'] = sample['egpt_error']
            if 'egpt_time' in sample: res['egpt_time'] = sample['egpt_time']
            if 'egpt_stage1_time' in sample: res['egpt_stage1_time'] = sample['egpt_stage1_time']
            if 'egpt_stage2_time' in sample: res['egpt_stage2_time'] = sample['egpt_stage2_time']
            if 'egpt_stage3_time' in sample: res['egpt_stage3_time'] = sample['egpt_stage3_time']
            if 'egpt_token_ids' in sample: res['egpt_token_ids'] = sample['egpt_token_ids']

        if 'video_data' in sample:
            res['video_data'] = sample.get('video_data')
            res['llava-1.5-7b-hf'] = sample.get('llava-1.5-7b-hf')
            if 'llava-1.5-7b-hf_error' in sample: res['llava-1.5-7b-hf_error'] = sample['llava-1.5-7b-hf_error']
            if 'llava_time' in sample: res['llava_time'] = sample['llava_time']
            if 'llava_stage1_time' in sample: res['llava_stage1_time'] = sample['llava_stage1_time']
            if 'llava_stage2_time' in sample: res['llava_stage2_time'] = sample['llava_stage2_time']
            if 'llava_stage3_time' in sample: res['llava_stage3_time'] = sample['llava_stage3_time']
            if 'llava_token_ids' in sample: res['llava_token_ids'] = sample['llava_token_ids']

        # Memory stats (only stored in first sample)
        if 'egpt_memory_model_mb' in sample:
            res['egpt_memory_model_mb'] = sample['egpt_memory_model_mb']
            res['egpt_memory_inference_mb'] = sample['egpt_memory_inference_mb']
        if 'llava_memory_model_mb' in sample:
            res['llava_memory_model_mb'] = sample['llava_memory_model_mb']
            res['llava_memory_inference_mb'] = sample['llava_memory_inference_mb']

        results.append(res)
    
    # Save
    if args.output_json is None:
        args.output_json = os.path.splitext(args.dataset_json)[0] + "_results.json"
        
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Done. Saved to {args.output_json}")

    # Print summary statistics
    egpt_times = [s['egpt_time'] for s in dataset if 'egpt_time' in s]
    egpt_stage1_times = [s['egpt_stage1_time'] for s in dataset if 'egpt_stage1_time' in s]
    egpt_stage2_times = [s['egpt_stage2_time'] for s in dataset if 'egpt_stage2_time' in s]
    egpt_stage3_times = [s['egpt_stage3_time'] for s in dataset if 'egpt_stage3_time' in s]
    llava_times = [s['llava_time'] for s in dataset if 'llava_time' in s]
    llava_stage1_times = [s['llava_stage1_time'] for s in dataset if 'llava_stage1_time' in s]
    llava_stage2_times = [s['llava_stage2_time'] for s in dataset if 'llava_stage2_time' in s]
    llava_stage3_times = [s['llava_stage3_time'] for s in dataset if 'llava_stage3_time' in s]

    avg_egpt = None
    avg_egpt_stage1 = None
    avg_egpt_stage2 = None
    avg_egpt_stage3 = None
    avg_llava = None
    avg_llava_stage1 = None
    avg_llava_stage2 = None
    avg_llava_stage3 = None

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY (3-Stage Timing)")
    print("="*60)

    if egpt_times:
        avg_egpt = sum(egpt_times) / len(egpt_times)
        print(f"\nEventGPT (Draft Model) Average Time: {avg_egpt:.3f}s ({len(egpt_times)} samples)")
        if egpt_stage1_times:
            avg_egpt_stage1 = sum(egpt_stage1_times) / len(egpt_stage1_times)
            print(f"  - Stage 1 (Load Data):       {avg_egpt_stage1:.3f}s")
        if egpt_stage2_times:
            avg_egpt_stage2 = sum(egpt_stage2_times) / len(egpt_stage2_times)
            print(f"  - Stage 2 (Preprocess):      {avg_egpt_stage2:.3f}s")
        if egpt_stage3_times:
            avg_egpt_stage3 = sum(egpt_stage3_times) / len(egpt_stage3_times)
            print(f"  - Stage 3 (Generate):        {avg_egpt_stage3:.3f}s")
    else:
        print("\nEventGPT (Draft Model): No samples processed.")

    if llava_times:
        avg_llava = sum(llava_times) / len(llava_times)
        print(f"\nLlava (Target Model) Average Time:   {avg_llava:.3f}s ({len(llava_times)} samples)")
        if llava_stage1_times:
            avg_llava_stage1 = sum(llava_stage1_times) / len(llava_stage1_times)
            print(f"  - Stage 1 (Load Data):       {avg_llava_stage1:.3f}s")
        if llava_stage2_times:
            avg_llava_stage2 = sum(llava_stage2_times) / len(llava_stage2_times)
            print(f"  - Stage 2 (Preprocess):      {avg_llava_stage2:.3f}s")
        if llava_stage3_times:
            avg_llava_stage3 = sum(llava_stage3_times) / len(llava_stage3_times)
            print(f"  - Stage 3 (Generate):        {avg_llava_stage3:.3f}s")
    else:
        print("\nLlava (Target Model): No samples processed.")

    print("\n" + "-"*60)
    print("PERFORMANCE METRICS")
    print("-"*60)

    # Calculate acceptance rate from token comparisons
    acceptance_rates = []
    total_matched = 0
    total_compared = 0
    for sample in dataset:
        egpt_tokens = sample.get('egpt_token_ids')
        llava_tokens = sample.get('llava_token_ids')
        if egpt_tokens and llava_tokens:
            alpha, matched, compared = calculate_acceptance_rate(egpt_tokens, llava_tokens)
            if compared > 0:
                acceptance_rates.append(alpha)
                total_matched += matched
                total_compared += compared

    avg_alpha = None
    if acceptance_rates:
        avg_alpha = sum(acceptance_rates) / len(acceptance_rates)
        overall_alpha = total_matched / total_compared if total_compared > 0 else 0
        print(f"\nAcceptance Rate (α):")
        print(f"  - Per-sample average: {avg_alpha:.3f} ({len(acceptance_rates)} samples)")
        print(f"  - Overall (weighted): {overall_alpha:.3f} ({total_matched}/{total_compared} tokens)")

    if avg_egpt is not None and avg_llava is not None and avg_llava > 0:
        c_value = avg_egpt / avg_llava
        print(f"\nc (Draft / Target Ratio) = {c_value:.3f}")

        # Speculative decoding speedup calculation
        gamma = args.gamma
        print(f"\nSpeculative Decoding (γ={gamma}):")
        if avg_alpha is not None:
            spec_speedup = calculate_speculative_speedup(c_value, avg_alpha, gamma)
            print(f"  - Theoretical Speedup: {spec_speedup:.2f}x")
            print(f"  - Formula: E[accepted]/(c*γ+1) where E[accepted]=(1-α^(γ+1))/(1-α)")
        else:
            print(f"  - Cannot calculate (no acceptance rate data)")
    else:
        print("\nCannot compute 'c' ratio (requires valid timings from both models).")

    # Print GPU memory stats
    print("\n" + "-"*60)
    print("GPU MEMORY USAGE")
    print("-"*60)

    # Get memory stats from first sample of each type
    egpt_mem_model = next((s.get('egpt_memory_model_mb') for s in dataset if 'egpt_memory_model_mb' in s), None)
    egpt_mem_infer = next((s.get('egpt_memory_inference_mb') for s in dataset if 'egpt_memory_inference_mb' in s), None)
    llava_mem_model = next((s.get('llava_memory_model_mb') for s in dataset if 'llava_memory_model_mb' in s), None)
    llava_mem_infer = next((s.get('llava_memory_inference_mb') for s in dataset if 'llava_memory_inference_mb' in s), None)

    if egpt_mem_model is not None:
        print(f"\nEventGPT GPU Memory:")
        print(f"  - Model Load:      {egpt_mem_model:.1f} MB")
        print(f"  - Inference Peak:  {egpt_mem_infer:.1f} MB")

    if llava_mem_model is not None:
        print(f"\nLlava GPU Memory:")
        print(f"  - Model Load:      {llava_mem_model:.1f} MB")
        print(f"  - Inference Peak:  {llava_mem_infer:.1f} MB")

if __name__ == "__main__":
    main()

