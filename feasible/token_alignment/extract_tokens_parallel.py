#!/usr/bin/env python3
"""
Parallel token extraction: Load both EventGPT and Video-LLaVA simultaneously.
Faster than sequential extraction by avoiding model load/unload overhead.

Memory requirement: ~8GB VRAM (both models 4-bit quantized)
Speedup: ~1.5-2x compared to sequential extraction
"""

import os
import sys
import json
import torch
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from datetime import datetime

# Fix protobuf compatibility
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer, BitsAndBytesConfig, VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from common.common import load_image

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    raise ImportError("PyAV required. Install with: pip install av")


def load_video_frames_from_mp4(video_path: str, num_frames: int = 8) -> List:
    """Load frames from MP4 video, sampled uniformly using PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames

    if total_frames == 0:
        total_frames = sum(1 for _ in container.decode(stream))
        container.seek(0)

    if total_frames >= num_frames:
        indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int))
    else:
        indices = set(range(total_frames))

    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            pil_img = frame.to_image()
            frames.append(pil_img)

    container.close()
    return frames


class ParallelExtractor:
    """Extract tokens from both models in parallel."""

    def __init__(self, max_new_tokens: int = 50):
        self.max_new_tokens = max_new_tokens
        self.device = "cuda"

        # 4-bit config for both models
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.egpt_model = None
        self.egpt_tokenizer = None
        self.egpt_processor = None

        self.vl_model = None
        self.vl_processor = None

    def load_models(self):
        """Load both models simultaneously."""
        from model.EventChatModel import EventChatModel

        print("\n[1/2] Loading EventGPT (4-bit)...")
        eventgpt_path = "./checkpoints/EventGPT-7b"
        self.egpt_model = EventChatModel.from_pretrained(
            eventgpt_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=self.bnb_config,
        )
        self.egpt_model.eval()
        self.egpt_tokenizer = AutoTokenizer.from_pretrained(eventgpt_path, use_fast=True)
        self.egpt_processor = self.egpt_model.get_visual_tower().event_processor
        print(f"  EventGPT loaded. GPU Memory: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

        print("\n[2/2] Loading Video-LLaVA (4-bit)...")
        videollava_model_id = "LanguageBind/Video-LLaVA-7B-hf"
        self.vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
            videollava_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=self.bnb_config,
        )
        self.vl_processor = VideoLlavaProcessor.from_pretrained(videollava_model_id)
        self.vl_model.eval()
        print(f"  Video-LLaVA loaded. GPU Memory: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

        print(f"\n  Total GPU Memory: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

    def extract_eventgpt(self, sample: Dict, dataset_dir: Path, query: str) -> Optional[Dict]:
        """Extract tokens from EventGPT."""
        from model.EventChatModel import get_spatio_temporal_features
        from dataset.conversation import prepare_event_prompt
        from dataset.constants import EVENT_TOKEN_INDEX
        from common.common import tokenizer_event_token

        try:
            if "event_image" not in sample or not sample["event_image"]:
                return {'error': 'No event_image'}

            event_image_paths = sample["event_image"]
            img_path = dataset_dir / "event_image" / event_image_paths[0]
            if not img_path.exists():
                return {'error': f'Image not found: {img_path}'}

            img = load_image(str(img_path))
            img_array = np.array(img)
            event_image_size = list(img_array.shape[:2])

            event = self.egpt_processor(img_array, return_tensors='pt')['pixel_values'][0]
            event = event.to(self.device, dtype=torch.bfloat16)

            conv_mode = 'eventgpt_v1'
            prompt = prepare_event_prompt(query, conv_mode)
            input_ids = tokenizer_event_token(
                prompt, self.egpt_tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                feature = self.egpt_model.visval_encode(event.unsqueeze(0))
                feature = self.egpt_model.get_model().feature_adaptor(feature)
                feature = feature.squeeze(0)
                event_features = get_spatio_temporal_features([feature])
                event_features = event_features.unsqueeze(0)

            with torch.inference_mode():
                output_ids = self.egpt_model.generate(
                    input_ids,
                    event_features=event_features,
                    event_image_sizes=[event_image_size],
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                )

            output_text = self.egpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if "ASSISTANT:" in output_text:
                output_text = output_text.split("ASSISTANT:")[-1].strip()

            return {
                'output_tokens': output_ids[0].tolist(),
                'output_text': output_text
            }
        except Exception as e:
            return {'error': str(e)}

    def extract_videollava(self, sample: Dict, dataset_dir: Path, query: str) -> Optional[Dict]:
        """Extract tokens from Video-LLaVA."""
        try:
            video_data = sample.get("video_data")
            if not video_data:
                return {'error': 'No video_data'}

            mp4_path = dataset_dir / "mp4" / f"{video_data}.mp4"
            if not mp4_path.exists():
                return {'error': f'MP4 not found: {mp4_path}'}

            frames = load_video_frames_from_mp4(str(mp4_path), num_frames=8)
            if not frames:
                return {'error': 'No frames extracted'}

            prompt = f"USER: <video>\n{query} ASSISTANT:"
            inputs = self.vl_processor(
                text=prompt,
                videos=frames,
                return_tensors="pt",
                padding=True,
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            pixel_values_videos = inputs["pixel_values_videos"].to(self.device, dtype=torch.float16)

            with torch.inference_mode():
                outputs = self.vl_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            output_ids = outputs[0]
            generated_ids = output_ids[input_ids.shape[1]:]
            output_text = self.vl_processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

            return {
                'output_tokens': generated_ids.tolist(),
                'output_text': output_text
            }
        except Exception as e:
            return {'error': str(e)}

    def extract_pair(self, sample: Dict, dataset_dir: Path, query: str) -> Optional[Dict]:
        """Extract tokens from both models for a single sample/query pair."""
        egpt_result = self.extract_eventgpt(sample, dataset_dir, query)
        if not egpt_result or 'error' in egpt_result:
            return None

        vl_result = self.extract_videollava(sample, dataset_dir, query)
        if not vl_result or 'error' in vl_result:
            return None

        return {
            'egpt_tokens': egpt_result['output_tokens'],
            'egpt_text': egpt_result['output_text'],
            'vl_tokens': vl_result['output_tokens'],
            'vl_text': vl_result['output_text'],
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parallel token extraction (both models loaded)")
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_train/my_egpt_dsec_train_1s')
    parser.add_argument('--output_file', type=str,
                        default='./feasible/token_alignment/train_tokens_parallel.json')
    parser.add_argument('--questions_file', type=str,
                        default='./feasible/token_alignment/top50_questions.json')
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--max_questions', type=int, default=-1,
                        help='Limit number of questions (e.g., 10 for top 10)')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Load dataset
    print("Loading dataset...")
    json_path = dataset_dir / "EventGPT_Instruction_Subset.json"
    with open(json_path) as f:
        dataset = json.load(f)

    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]

    print(f"Dataset size: {len(dataset)} samples")

    # Load questions
    questions_file = Path(args.questions_file)
    if questions_file.exists():
        with open(questions_file) as f:
            questions_data = json.load(f)
        questions = [q['question'] for q in questions_data]
        if args.max_questions > 0:
            questions = questions[:args.max_questions]
        print(f"Using {len(questions)} questions")
    else:
        questions = ["What are the key elements in this scene?"]
        print(f"Questions file not found, using default question")

    total_pairs = len(dataset) * len(questions)
    print(f"Total pairs to extract: {total_pairs}")

    # Initialize extractor and load both models
    extractor = ParallelExtractor(max_new_tokens=args.max_new_tokens)
    extractor.load_models()

    # Extract tokens
    print(f"\nExtracting tokens (parallel mode)...")
    results = []
    errors = 0

    with tqdm(total=total_pairs, desc="Extracting") as pbar:
        for sample_idx, sample in enumerate(dataset):
            for q_idx, query in enumerate(questions):
                result = extractor.extract_pair(sample, dataset_dir, query)
                if result:
                    results.append({
                        'sample_idx': sample_idx,
                        'question_idx': q_idx,
                        'sample_id': sample.get('id'),
                        'query': query,
                        **result
                    })
                else:
                    errors += 1
                pbar.update(1)
                pbar.set_postfix({'success': len(results), 'errors': errors})

    print(f"\nExtraction complete: {len(results)}/{total_pairs} pairs ({errors} errors)")

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_dir': str(dataset_dir),
        'total_samples': len(dataset),
        'num_questions': len(questions),
        'questions': questions,
        'extraction_mode': 'parallel',
        'success_count': len(results),
        'error_count': errors,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Extraction Complete (Parallel Mode)")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    print(f"Questions: {len(questions)}")
    print(f"Target pairs: {total_pairs}")
    print(f"Successful pairs: {len(results)}")
    print(f"Errors: {errors}")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
