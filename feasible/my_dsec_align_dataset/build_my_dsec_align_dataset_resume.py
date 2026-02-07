#!/usr/bin/env python3
"""
Build DSEC Alignment Dataset - Resume Capable Version
=====================================================

This script creates a training dataset for feature and token alignment.
Supports resuming from previous runs.

Usage:
    # Process 500 samples with top 1 question
    python build_my_dsec_align_dataset_resume.py --max_samples 500 --num_questions 1

    # Resume processing (will skip already completed samples)
    python build_my_dsec_align_dataset_resume.py --max_samples 1000 --num_questions 1
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

# Top 50 questions
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


def load_video_frames_from_mp4(video_path, num_frames=8):
    """Load frames from MP4 video using PyAV."""
    if not HAS_PYAV:
        raise ImportError("PyAV is required. Install with: pip install av")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'

    total_frames = stream.frames
    if total_frames == 0:
        container.seek(0)
        total_frames = sum(1 for _ in container.decode(stream))
        container.seek(0)

    if total_frames >= num_frames:
        indices = sorted(set(np.linspace(0, total_frames - 1, num_frames, dtype=int)))
    else:
        indices = list(range(total_frames))

    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            frames.append(frame.to_image())
            if len(frames) >= len(indices):
                break

    container.close()
    return frames, total_frames


class VideoLLaVAFeatureExtractor:
    """Extract intermediate features from VideoLLaVA."""

    def __init__(self, model_path: str = VIDEO_LLAVA_MODEL, device: str = "cuda"):
        self.device = device
        print(f"Loading VideoLLaVA from {model_path}...")
        self.load_model(model_path)
        print("VideoLLaVA loaded successfully")

    def load_model(self, model_path: str):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=self.device, local_files_only=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        except Exception:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer

    def extract_intermediate_results(
        self, frames: List[Image.Image], question: str, max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        image_tokens = "<image>\n" * len(frames)
        prompt = f"USER: {image_tokens}{question}\nASSISTANT:"
        inputs = self.processor(text=prompt, images=frames, return_tensors="pt").to(self.device)

        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        pixel_values = inputs.get('pixel_values')

        results = {}

        with torch.inference_mode():
            # Vision encoding
            if hasattr(self.model, 'vision_tower') and pixel_values is not None:
                vision_outputs = self.model.vision_tower(pixel_values, output_hidden_states=True)
                vision_features = vision_outputs.last_hidden_state
                results['vision_features'] = vision_features[:, 0, :].cpu()
                results['num_vision_features'] = vision_features.shape[0]
            else:
                results['vision_features'] = None
                results['num_vision_features'] = 0

            results['prefill_length'] = input_ids.shape[1]
            results['input_ids'] = input_ids[0].cpu()

            # Generate using built-in method (memory efficient)
            output_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=min(max_new_tokens, 50), use_cache=True
            )

            generated_ids = output_ids[0, input_ids.shape[1]:]
            results['generated_ids'] = generated_ids.cpu()
            results['output_ids'] = output_ids[0].cpu()

            # Get prefill hidden states
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    pixel_values=pixel_values, output_hidden_states=True, return_dict=True
                )
                results['prefill_hidden_states'] = outputs.hidden_states[-1][0].cpu()

        results['output_text'] = self.tokenizer.decode(results['generated_ids'], skip_special_tokens=True)
        return results


def load_existing_dataset(dataset_dir: str) -> List[Dict[str, Any]]:
    """Load existing dataset JSON."""
    json_path = os.path.join(dataset_dir, "EventGPT_Instruction_Subset.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(dataset_dir, "my_egpt_dsec_train_1s.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(dataset_dir, "my_egpt_dsec_instruction_subset.json")

    with open(json_path, 'r') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples from {json_path}")
    return dataset


def load_progress(output_dir: str) -> set:
    """Load set of already processed entry IDs."""
    progress_file = os.path.join(output_dir, "progress.txt")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_progress(output_dir: str, processed_ids: set):
    """Save progress to file."""
    progress_file = os.path.join(output_dir, "progress.txt")
    with open(progress_file, 'w') as f:
        for entry_id in sorted(processed_ids):
            f.write(entry_id + '\n')


def load_output_data(output_dir: str) -> tuple:
    """Load existing output data for resuming."""
    json_path = os.path.join(output_dir, "my_dsec_align_train_1s.json")
    results = []
    vision_features = []
    prefill_hidden = []
    tokens = {'input_ids': [], 'output_ids': [], 'generated_ids': []}

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)

    features_dir = os.path.join(output_dir, "features")
    if os.path.exists(features_dir):
        vf_path = os.path.join(features_dir, "vision_features.pt")
        ph_path = os.path.join(features_dir, "prefill_hidden_states.pt")
        tk_path = os.path.join(features_dir, "all_tokens.pt")

        if os.path.exists(vf_path):
            vision_features = torch.load(vf_path)
        if os.path.exists(ph_path):
            prefill_hidden = torch.load(ph_path)
        if os.path.exists(tk_path):
            tokens = torch.load(tk_path)

    return results, vision_features, prefill_hidden, tokens


def save_checkpoint(output_dir: str, results, vision_features, prefill_hidden, tokens, processed_ids):
    """Save checkpoint data."""
    json_path = os.path.join(output_dir, "my_dsec_align_train_1s.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    if vision_features:
        torch.save(vision_features, os.path.join(features_dir, "vision_features.pt"))
    if prefill_hidden:
        torch.save(prefill_hidden, os.path.join(features_dir, "prefill_hidden_states.pt"))
    torch.save(tokens, os.path.join(features_dir, "all_tokens.pt"))

    save_progress(output_dir, processed_ids)


def build_alignment_dataset(
    dataset_dir: str, output_dir: str, extractor: VideoLLaVAFeatureExtractor,
    questions: List[str], max_samples: int = None, device: str = "cuda",
    checkpoint_every: int = 10
) -> Dict[str, Any]:
    """Build alignment dataset with resume capability."""
    print("\n" + "="*80)
    print("Building Alignment Dataset (Resume Capable)")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Load existing progress
    processed_ids = load_progress(output_dir)
    results, vision_features, prefill_hidden, tokens = load_output_data(output_dir)

    print(f"Resuming with {len(results)} completed entries")

    # Load source dataset
    dataset = load_existing_dataset(dataset_dir)
    if max_samples:
        dataset = dataset[:max_samples]

    total_entries = len(dataset) * len(questions)
    target_entries = total_entries - len(processed_ids)
    print(f"Total entries to process: {total_entries}")
    print(f"Already completed: {len(processed_ids)}")
    print(f"Remaining: {target_entries}")

    start_time = time.time()
    last_checkpoint = time.time()

    for sample_idx, sample in enumerate(tqdm(dataset, desc="Samples")):
        sample_id = sample.get("id", f"sample_{sample_idx:06d}")
        event_data = sample.get("event_data", "")
        video_data = sample.get("video_data", "")

        # Build MP4 path
        if video_data:
            video_path = os.path.join(dataset_dir, "mp4", f"{video_data}.mp4")
        else:
            video_path = os.path.join(dataset_dir, "mp4", event_data.replace(".npy", "") + ".mp4")

        # Load frames
        try:
            if os.path.exists(video_path) and HAS_PYAV:
                frames, total_frames = load_video_frames_from_mp4(video_path, num_frames=8)
                source = f"mp4:{total_frames}f"
            else:
                raise FileNotFoundError("MP4 not found")
        except Exception as e:
            print(f"Warning: Could not load frames for {sample_id}: {e}")
            continue

        if len(frames) == 0:
            continue

        # Process each question
        for q_idx, question in enumerate(questions):
            entry_id = f"{sample_id}_q{q_idx:02d}"

            # Skip if already processed
            if entry_id in processed_ids:
                continue

            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            try:
                feat = extractor.extract_intermediate_results(
                    frames=frames, question=question, max_new_tokens=256
                )

                entry = {
                    "id": entry_id,
                    "sample_id": sample_id,
                    "sample_idx": sample_idx,
                    "question_idx": q_idx,
                    "event_data": event_data,
                    "question": question,
                    "answer": feat['output_text'],
                    "input_ids": feat['input_ids'].tolist(),
                    "output_ids": feat['output_ids'].tolist(),
                    "generated_ids": feat['generated_ids'].tolist(),
                    "num_frames": len(frames),
                    "frame_source": source,
                    "prefill_length": feat['prefill_length'],
                }

                if feat.get('vision_features') is not None:
                    entry['has_vision_features'] = True
                    vision_features.append({'id': entry_id, 'features': feat['vision_features']})

                prefill_hidden.append({'id': entry_id, 'hidden_states': feat['prefill_hidden_states']})

                tokens['input_ids'].append(entry['input_ids'])
                tokens['output_ids'].append(entry['output_ids'])
                tokens['generated_ids'].append(entry['generated_ids'])

                results.append(entry)
                processed_ids.add(entry_id)

                # Periodic checkpoint
                if time.time() - last_checkpoint > 300:  # Every 5 minutes
                    save_checkpoint(output_dir, results, vision_features, prefill_hidden, tokens, processed_ids)
                    elapsed = time.time() - start_time
                    rate = len(processed_ids) / elapsed
                    eta = (target_entries - len(processed_ids) + len(results)) / rate
                    print(f"\nCheckpoint! {len(processed_ids)} done, ETA: {eta/3600:.1f}h")
                    last_checkpoint = time.time()

                del feat
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {entry_id}: {e}")
                continue

        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Final save
    save_checkpoint(output_dir, results, vision_features, prefill_hidden, tokens, processed_ids)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "source_dataset": dataset_dir,
        "num_samples": len(dataset),
        "num_questions": len(questions),
        "total_entries": len(results),
        "questions": questions,
        "model_path": VIDEO_LLAVA_MODEL,
    }

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Build DSEC alignment dataset (resume capable)")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--questions_file", type=str, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--num_questions", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("="*80)
    print("DSEC Alignment Dataset Builder (Resume Capable)")
    print("="*80)
    print(f"Max samples: {args.max_samples}")
    print(f"Num questions: {args.num_questions}")
    print(f"Device: {args.device}")

    # Load questions
    if os.path.exists(args.questions_file):
        questions = []
        with open(args.questions_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    questions.append(parts[1])
        questions = questions[:args.num_questions]
    else:
        questions = TOP_50_QUESTIONS[:args.num_questions]

    extractor = VideoLLaVAFeatureExtractor(device=args.device)

    metadata = build_alignment_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        extractor=extractor,
        questions=questions,
        max_samples=args.max_samples,
        device=args.device,
    )

    print("\n" + "="*80)
    print("ALIGNMENT DATASET SUMMARY")
    print("="*80)
    print(f"Samples: {metadata['num_samples']}")
    print(f"Questions: {metadata['num_questions']}")
    print(f"Total entries: {metadata['total_entries']}")
    print("="*80)
    print("COMPLETE!")

if __name__ == "__main__":
    main()
