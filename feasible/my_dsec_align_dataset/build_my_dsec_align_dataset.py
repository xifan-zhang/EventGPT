#!/usr/bin/env python3
"""
Build DSEC Alignment Dataset for Feature and Token Alignment Training
=====================================================================

This script creates a training dataset for training:
1. Feature Alignment Adapter: EventGPT vision features → VideoLLaVA vision features
2. Token Alignment Model: EventGPT tokens → VideoLLaVA tokens

The dataset includes:
- VideoLLaVA 8-frames intermediate results (vision features, hidden states, tokens)
- All 50 top questions answered for each sample
- HuggingFace dataset compatible format

Output Structure:
    my_dsec_align_dataset/
    ├── metadata.json
    ├── features/
    │   ├── vision_features.pt       # Vision encoder outputs per frame
    │   ├── projected_features.pt    # Projected to language model space
    │   ├── prefill_hidden_states.pt # Prefill hidden states (for prefill-hidden drafting)
    │   └── all_tokens.pt             # Generated token IDs
    ├── my_dsec_align_train_1s.json  # Main dataset JSON
    └── hf_dataset/                  # HuggingFace dataset format

Usage:
    # Test with 5 samples
    python build_my_dsec_align_dataset.py --max_samples 5

    # Full dataset
    python build_my_dsec_align_dataset.py

    # Custom dataset directory
    python build_my_dsec_align_dataset.py --dataset_dir /path/to/dataset
"""

import os
import sys
import json
import torch
import time
import gc
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional
from PIL import Image

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Configuration
DEFAULT_DATASET_DIR = "/mnt/hdd/data/my_egpt_dsec_train/my_egpt_dsec_train_1s"
DEFAULT_OUTPUT_DIR = "/home/ps/Documents/code/EventGPT/feasible/my_dsec_align_dataset"
DEFAULT_QUESTIONS_FILE = "/home/ps/Documents/code/EventGPT/feasible/analysis_datasets/results_egpt_dsec_split/dsec_questions_top50.txt"
VIDEO_LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"

# Top 50 questions (hardcoded as fallback)
TOP_50_QUESTIONS = [
    "What are the key elements in this scene?",
    "Can you elaborate on the elements of the scene provided?",
    "What is this scene about?",
    "What is the layout of the road in the scene?",
    "What type of vehicles are present in the scene and how are they positioned?",
    "What do you think is going on in this scene?",
    "What elements are present in the scene?",
    "What do you see happening in this scene?",
    "Describe the following scene.",
    "What types of vehicles are present in the scene and how are they positioned?",
    "What vehicles are present in the scene and how are they positioned?",
    "Analyze the scene in a comprehensive and detailed manner.",
    "Describe the layout of the scene.",
    "What is the primary setting of this scene?",
    "What is the main feature of this scene?",
    "What can be observed in the scene?",
    "What is the main activity happening in this scene?",
    "What vehicles are present in the scene?",
    "Are there any pedestrians visible in the scene?",
    "What is the primary activity in this scene?",
    "Can you describe the layout of the scene?",
    "Are there any vehicles visible in the scene?",
    "Are there any pedestrians visible in this scene?",
    "What can be observed about the road in the scene?",
    "What is the main activity in this scene?",
    "What can be observed about the vehicles in the scene?",
    "Explain the visual content of the scene in great detail.",
    "Are there any vehicles or pedestrians visible in the scene?",
    "What is the primary activity in the scene?",
    "What is the layout of the street in the scene?",
    "What is the primary setting of the scene?",
    "What is the main activity in the scene?",
    "Describe the layout of the road in the scene.",
    "What objects are present on the side of the road in the scene?",
    "What is happening in this scene?",
    "Describe the environment surrounding the road.",
    "Can you elaborate on the elements of the event stream?",
    "Describe the environment around the road.",
    "What structures are visible in the background and how are they arranged?",
    "What type of vehicles are parked along the street, and how are they arranged?",
    "How many vehicles are visible in the scene and what are their positions?",
    "What are the vehicles doing in the scene?",
    "What is the primary mode of transportation visible in the scene?",
    "What objects are present in the scene and how are they arranged?",
    "What is the primary activity occurring in the scene?",
    "What is the primary activity happening in this scene?",
    "Can you describe the environment of the scene?",
    "Describe the environment of the scene.",
]

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    print("Warning: PyAV not found. Install with: pip install av")


def load_video_frames_from_mp4(video_path, num_frames=8):
    """Load frames from MP4 video using PyAV.

    Args:
        video_path: Path to MP4 video file
        num_frames: Number of frames to sample uniformly

    Returns:
        frames: List of PIL Images
        total_frames: Total frames in video
        sampled_indices: List of sampled frame indices
    """
    if not HAS_PYAV:
        raise ImportError("PyAV is required for MP4 video loading. Install with: pip install av")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'  # Auto for multi-threaded decoding

    # Get total frame count
    total_frames = stream.frames
    if total_frames == 0:
        # Fallback: count frames by iterating
        container.seek(0)
        total_frames = sum(1 for _ in container.decode(stream))
        container.seek(0)

    # Sample frame indices uniformly
    if total_frames >= num_frames:
        indices = sorted(set(np.linspace(0, total_frames - 1, num_frames, dtype=int)))
    else:
        indices = list(range(total_frames))

    frames = []
    sampled_indices = []

    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            pil_img = frame.to_image()
            frames.append(pil_img)
            sampled_indices.append(i)
            if len(frames) >= len(indices):
                break

    container.close()
    return frames, total_frames, sampled_indices


def load_event_images_as_frames(dataset_dir, event_data, num_frames=8):
    """Load event images as pseudo-frames (fallback when MP4 not available).

    Args:
        dataset_dir: Path to dataset directory
        event_data: Event data filename
        num_frames: Number of frames to use

    Returns:
        frames: List of PIL Images
    """
    event_image_dir = os.path.join(dataset_dir, "event_image")

    # Get all event images for this sample
    event_images = []
    for i in range(num_frames):
        img_path = os.path.join(event_image_dir, event_data.replace(".npy", ""), f"{i:06d}.png")
        if os.path.exists(img_path):
            event_images.append(img_path)
        else:
            break

    # If no individual frame images, try the base image
    if len(event_images) == 0:
        # Try to find the event image directly
        base_name = event_data.replace(".npy", "")
        for ext in ['.png', '.jpg']:
            img_path = os.path.join(event_image_dir, base_name + ext)
            if os.path.exists(img_path):
                # Duplicate to get num_frames
                return [Image.open(img_path)] * num_frames

    # Load images
    frames = []
    for img_path in event_images[:num_frames]:
        img = Image.open(img_path).convert('RGB')
        frames.append(img)

    # If not enough frames, duplicate the last one
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    return frames


class VideoLLaVAFeatureExtractor:
    """Extract intermediate features from VideoLLaVA for alignment training."""

    def __init__(
        self,
        model_path: str = VIDEO_LLAVA_MODEL,
        device: str = "cuda",
    ):
        self.device = device
        print(f"Loading VideoLLaVA from {model_path}...")
        self.load_model(model_path)
        print("VideoLLaVA loaded successfully")

    def load_model(self, model_path: str):
        """Load VideoLLaVA model and processor."""
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                local_files_only=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            print(f"Local loading failed, trying HuggingFace: {e}")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
            self.processor = AutoProcessor.from_pretrained(model_path)

        self.model.eval()
        self.tokenizer = self.processor.tokenizer

    def extract_intermediate_results(
        self,
        frames: List[Image.Image],
        question: str,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Extract all intermediate results from VideoLLaVA.

        Args:
            frames: List of PIL Images (8 frames)
            question: Question to ask
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with all intermediate data
        """
        # Prepare prompt
        image_tokens = "<image>\n" * len(frames)
        prompt = f"USER: {image_tokens}{question}\nASSISTANT:"

        # Prepare inputs
        inputs = self.processor(text=prompt, images=frames, return_tensors="pt").to(self.device)

        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        pixel_values = inputs.get('pixel_values')

        results = {}

        with torch.inference_mode():
            # ===== Stage 3: Vision Encoding =====
            if hasattr(self.model, 'vision_tower') and pixel_values is not None:
                vision_outputs = self.model.vision_tower(pixel_values, output_hidden_states=True)
                vision_features = vision_outputs.last_hidden_state  # [batch*num_frames, seq_len, hidden]

                # Get CLS token per frame
                num_frames_actual = vision_features.shape[0]
                vision_cls_features = vision_features[:, 0, :]  # [batch*num_frames, hidden]
                results['vision_features'] = vision_cls_features.cpu()
                results['num_vision_features'] = num_frames_actual
            else:
                results['vision_features'] = None
                results['num_vision_features'] = 0

            # Store prefill length
            prefill_len = input_ids.shape[1]
            results['prefill_length'] = prefill_len

            # Store input ids
            results['input_ids'] = input_ids[0].cpu()

            # ===== Use built-in generate for memory efficiency =====
            # Limit max_new_tokens for memory efficiency
            actual_max_tokens = min(max_new_tokens, 50)

            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=actual_max_tokens,
                use_cache=True,
            )

            # Extract generated tokens
            generated_ids = output_ids[0, input_ids.shape[1]:]
            results['generated_ids'] = generated_ids.cpu()
            results['output_ids'] = output_ids[0].cpu()

            # For prefill_hidden_states, do a separate forward pass without generation
            # This is more memory-efficient than capturing during generation
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states
                prefill_hidden = hidden_states[-1][0]  # [seq_len, hidden]
                results['prefill_hidden_states'] = prefill_hidden.cpu()

        # Decode output text
        output_text = self.tokenizer.decode(
            results['generated_ids'],
            skip_special_tokens=True,
        )
        results['output_text'] = output_text

        return results


def load_existing_dataset(dataset_dir: str) -> List[Dict[str, Any]]:
    """Load existing dataset JSON."""
    # Try the correct filename first
    json_path = os.path.join(dataset_dir, "EventGPT_Instruction_Subset.json")

    if not os.path.exists(json_path):
        # Try alternative paths
        json_path = os.path.join(dataset_dir, "my_egpt_dsec_train_1s.json")

    if not os.path.exists(json_path):
        # Try alternative paths
        json_path = os.path.join(dataset_dir, "my_egpt_dsec_instruction_subset.json")

    if not os.path.exists(json_path):
        # Try one more alternative
        alt_path = os.path.join(os.path.dirname(dataset_dir), "my_egpt_dsec_train_1s", "my_egpt_dsec_train_1s.json")
        if os.path.exists(alt_path):
            json_path = alt_path

    if not os.path.exists(json_path):
        # List available JSON files
        import glob
        json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
        if json_files:
            json_path = json_files[0]
            print(f"Using dataset file: {json_path}")
        else:
            raise FileNotFoundError(f"No dataset JSON found in {dataset_dir}")

    print(f"Loading dataset from {json_path}")
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples")
    return dataset


def load_questions(questions_file: str = None) -> List[str]:
    """Load questions from file or use defaults."""
    if questions_file and os.path.exists(questions_file):
        questions = []
        with open(questions_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract question (format: "count\tquestion")
                    parts = line.split('\t')
                    if len(parts) > 1:
                        questions.append(parts[1].strip())
                    elif line and not line[0].isdigit():
                        questions.append(line)

        if questions:
            print(f"Loaded {len(questions)} questions from {questions_file}")
            return questions[:50]  # Limit to top 50

    print(f"Using default top 50 questions")
    return TOP_50_QUESTIONS


def build_alignment_dataset(
    dataset_dir: str,
    output_dir: str,
    extractor: VideoLLaVAFeatureExtractor,
    questions: List[str],
    max_samples: int = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Build alignment dataset from existing dataset."""
    print("\n" + "="*80)
    print("Building Alignment Dataset")
    print("="*80)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    # Load existing dataset
    dataset = load_existing_dataset(dataset_dir)

    if max_samples:
        dataset = dataset[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Prepare results storage
    all_results = []
    all_vision_features = []
    all_projected_features = []
    all_prefill_hidden = []
    all_tokens = {'input_ids': [], 'output_ids': [], 'generated_ids': []}

    total_entries = len(dataset) * len(questions)
    print(f"Total entries to process: {total_entries}")

    # Process each sample
    for sample_idx, sample in enumerate(tqdm(dataset, desc="Samples")):
        sample_id = sample.get("id", f"sample_{sample_idx:06d}")
        event_data = sample.get("event_data", "")
        video_data = sample.get("video_data", "")

        # Try to find MP4 video first
        # video_data format: "scene_name/filename" (e.g., "interlaken_00_c/000000")
        if video_data:
            video_path = os.path.join(dataset_dir, "mp4", f"{video_data}.mp4")
        else:
            # Fallback to event_data
            video_path = os.path.join(dataset_dir, "mp4", event_data.replace(".npy", "") + ".mp4")

        # Load frames
        try:
            if os.path.exists(video_path) and HAS_PYAV:
                frames, total_frames, sampled_indices = load_video_frames_from_mp4(video_path, num_frames=8)
                source = f"mp4:{total_frames}f"
            else:
                # Fall back to event images
                frames = load_event_images_as_frames(dataset_dir, event_data, num_frames=8)
                source = f"event_images:{len(frames)}f"
        except Exception as e:
            print(f"Warning: Could not load frames for {sample_id}: {e}")
            continue

        if len(frames) == 0:
            print(f"Warning: No frames loaded for {sample_id}")
            continue

        # Process each question
        for q_idx, question in enumerate(tqdm(questions, desc="Questions", leave=False, disable=None)):
            entry_id = f"{sample_id}_q{q_idx:02d}"

            # Aggressive memory cleanup before each question
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            try:
                # Extract features from VideoLLaVA
                results = extractor.extract_intermediate_results(
                    frames=frames,
                    question=question,
                    max_new_tokens=256,
                )

                # Create entry
                entry = {
                    "id": entry_id,
                    "sample_id": sample_id,
                    "sample_idx": sample_idx,
                    "question_idx": q_idx,
                    "event_data": event_data,
                    "question": question,
                    "answer": results['output_text'],
                    "input_ids": results['input_ids'].tolist(),
                    "output_ids": results['output_ids'].tolist(),
                    "generated_ids": results['generated_ids'].tolist(),
                    "num_frames": len(frames),
                    "frame_source": source,
                    "prefill_length": results['prefill_length'],
                }

                # Add vision features if available
                if results.get('vision_features') is not None:
                    entry['has_vision_features'] = True
                    all_vision_features.append({
                        'id': entry_id,
                        'features': results['vision_features'],
                    })

                all_prefill_hidden.append({
                    'id': entry_id,
                    'hidden_states': results['prefill_hidden_states'],
                })

                all_results.append(entry)

                # Also add to token lists
                all_tokens['input_ids'].append(entry['input_ids'])
                all_tokens['output_ids'].append(entry['output_ids'])
                all_tokens['generated_ids'].append(entry['generated_ids'])

                # Clear cache after each question to avoid OOM
                del results
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {entry_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Clear cache after each sample
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Save all results
    print("\nSaving results...")
    save_results(output_dir, features_dir, all_results, all_vision_features, all_prefill_hidden, all_tokens)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "source_dataset": dataset_dir,
        "num_samples": len(dataset),
        "num_questions": len(questions),
        "total_entries": len(all_results),
        "questions": questions,
        "model_path": VIDEO_LLAVA_MODEL,
    }

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    return metadata


def save_results(output_dir, features_dir, results, vision_features, prefill_hidden, tokens):
    """Save all results to disk."""
    # Save main JSON
    json_path = os.path.join(output_dir, "my_dsec_align_train_1s.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} entries to {json_path}")

    # Save vision features
    if vision_features:
        vision_path = os.path.join(features_dir, "vision_features.pt")
        torch.save(vision_features, vision_path)
        print(f"Saved vision features to {vision_path}")

    # Save prefill hidden states
    if prefill_hidden:
        hidden_path = os.path.join(features_dir, "prefill_hidden_states.pt")
        torch.save(prefill_hidden, hidden_path)
        print(f"Saved prefill hidden states to {hidden_path}")

    # Save all tokens
    tokens_path = os.path.join(features_dir, "all_tokens.pt")
    torch.save(tokens, tokens_path)
    print(f"Saved tokens to {tokens_path}")


def create_huggingface_dataset(output_dir: str):
    """Create HuggingFace dataset from saved data."""
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        print("datasets library not found. Install with: pip install datasets")
        return None

    json_path = os.path.join(output_dir, "my_dsec_align_train_1s.json")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create dataset
    dataset = Dataset.from_list(data)

    # Save as HuggingFace dataset
    hf_dir = os.path.join(output_dir, "hf_dataset")
    dataset.save_to_disk(hf_dir)
    print(f"Saved HuggingFace dataset to {hf_dir}")
    print(f"Load with: dataset = datasets.load_from_disk('{hf_dir}')")

    return dataset


def print_summary(metadata):
    """Print summary of the built dataset."""
    print("\n" + "="*80)
    print("ALIGNMENT DATASET SUMMARY")
    print("="*80)
    print(f"Created at:       {metadata['created_at']}")
    print(f"Source dataset:   {metadata['source_dataset']}")
    print(f"Samples:          {metadata['num_samples']}")
    print(f"Questions:        {metadata['num_questions']}")
    print(f"Total entries:    {metadata['total_entries']}")
    print(f"Model:            {metadata['model_path']}")
    print(f"\nOutput directory: {os.path.dirname(metadata['created_at'])}")
    print("\nSaved files:")
    print("  - my_dsec_align_train_1s.json  (Main dataset)")
    print("  - features/vision_features.pt    (Vision encoder outputs)")
    print("  - features/prefill_hidden_states.pt  (Prefill hidden states)")
    print("  - features/all_tokens.pt         (All generated tokens)")
    print("  - metadata.json                  (Dataset metadata)")
    print("  - hf_dataset/                    (HuggingFace format)")


def main():
    parser = argparse.ArgumentParser(
        description="Build DSEC alignment dataset for feature and token alignment"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="Path to existing 1s dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for alignment dataset",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=DEFAULT_QUESTIONS_FILE,
        help="Path to questions file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of samples to process (default: 5 for testing)",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=5,
        help="Number of questions to use (top N) - default 5 for testing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--create_hf_dataset",
        action="store_true",
        help="Create HuggingFace dataset at the end",
    )

    args = parser.parse_args()

    print("="*80)
    print("DSEC Alignment Dataset Builder")
    print("="*80)
    print(f"Source dataset: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Num questions: {args.num_questions}")
    print(f"Device: {args.device}")

    # Load questions
    questions = load_questions(args.questions_file)[:args.num_questions]

    # Initialize feature extractor
    extractor = VideoLLaVAFeatureExtractor(device=args.device)

    # Build dataset
    metadata = build_alignment_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        extractor=extractor,
        questions=questions,
        max_samples=args.max_samples,
        device=args.device,
    )

    # Print summary
    print_summary(metadata)

    # Create HuggingFace dataset if requested
    if args.create_hf_dataset:
        print("\nCreating HuggingFace dataset...")
        create_huggingface_dataset(args.output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
