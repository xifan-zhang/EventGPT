#!/usr/bin/env python3
"""
Extract tokens from EventGPT and Video-LLaVA on full training dataset.
Uses event_image for EventGPT and mp4 (8 frames) for Video-LLaVA.
Matches the approach in benchmark_parallel_prefill_5stages.py
"""

import os
import sys
import json
import torch
import time
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


def extract_eventgpt_tokens(
    model, tokenizer, processor,
    sample: Dict, dataset_dir: Path, query: str,
    max_new_tokens: int = 50
) -> Optional[Dict]:
    """Extract tokens from EventGPT using event_image (PNG)."""
    from model.EventChatModel import get_spatio_temporal_features
    from dataset.conversation import prepare_event_prompt
    from dataset.constants import EVENT_TOKEN_INDEX
    from common.common import tokenizer_event_token

    device = "cuda"

    try:
        if "event_image" not in sample or not sample["event_image"]:
            return {'error': 'No event_image'}

        event_image_paths = sample["event_image"]

        # Stage 1: Load event image
        img_path = dataset_dir / "event_image" / event_image_paths[0]
        if not img_path.exists():
            return {'error': f'Image not found: {img_path}'}

        img = load_image(str(img_path))
        img_array = np.array(img)
        event_image_size = list(img_array.shape[:2])

        # Stage 2: Preprocess
        event = processor(img_array, return_tensors='pt')['pixel_values'][0]
        event = event.to(device, dtype=torch.bfloat16)

        conv_mode = 'eventgpt_v1'
        prompt = prepare_event_prompt(query, conv_mode)
        input_ids = tokenizer_event_token(
            prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)

        # Stage 3: Vision encoding
        with torch.inference_mode():
            feature = model.visval_encode(event.unsqueeze(0))
            feature = model.get_model().feature_adaptor(feature)
            feature = feature.squeeze(0)
            event_features = get_spatio_temporal_features([feature])
            event_features = event_features.unsqueeze(0)

        # Stage 4+5: Prefill and decode using model.generate()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                event_features=event_features,
                event_image_sizes=event_image_size,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in output_text:
            output_text = output_text.split("ASSISTANT:")[-1].strip()

        return {
            'output_tokens': output_ids[0].tolist(),
            'output_text': output_text
        }
    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()[-300:]}


def extract_videollava_tokens(
    model, processor,
    sample: Dict, dataset_dir: Path, query: str,
    max_new_tokens: int = 50
) -> Optional[Dict]:
    """Extract tokens from Video-LLaVA using mp4 (8 frames)."""
    device = "cuda"

    try:
        video_data = sample.get("video_data")

        # Stage 1: Load video frames from mp4
        if video_data:
            video_path = dataset_dir / "mp4" / f"{video_data}.mp4"
            if not video_path.exists():
                return {'error': f'MP4 not found: {video_path}'}
            rgb_images = load_video_frames_from_mp4(str(video_path), num_frames=8)
        else:
            # Fallback to event_image
            if "event_image" not in sample or not sample["event_image"]:
                return {'error': 'No video_data or event_image'}
            event_image_paths = sample["event_image"]
            rgb_images = []
            for img_path in event_image_paths[:8]:
                full_path = dataset_dir / "event_image" / img_path
                img = load_image(str(full_path))
                rgb_images.append(img)
            while len(rgb_images) < 8:
                rgb_images.append(rgb_images[-1])

        if len(rgb_images) == 0:
            return {'error': 'No frames loaded'}

        # Stage 2: Preprocess
        prompt = f"USER: <video>\n{query}\nASSISTANT:"
        inputs = processor(text=prompt, videos=rgb_images, return_tensors="pt")

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask')
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(device)

        pixel_values_videos = inputs.get('pixel_values_videos')
        if pixel_values_videos is None:
            return {'error': f'No pixel_values_videos. Keys: {list(inputs.keys())}'}

        # Stage 4+5: Generate
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        output_ids = outputs[0]
        # Only keep newly generated tokens (after the input prompt)
        generated_ids = output_ids[input_ids.shape[1]:]
        output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            'output_tokens': generated_ids.tolist(),
            'output_text': output_text
        }
    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()[-300:]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_train/my_egpt_dsec_train_1s')
    parser.add_argument('--output_file', type=str,
                        default='/home/ps/Documents/code/EventGPT/feasible/token_alignment/train_tokens_full.json')
    parser.add_argument('--questions_file', type=str,
                        default='/home/ps/Documents/code/EventGPT/feasible/token_alignment/top50_questions.json')
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--single_question', action='store_true',
                        help='Use only the top question instead of all 50')
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
        if args.single_question:
            questions = questions[:1]
        print(f"Using {len(questions)} questions")
    else:
        # Fallback to single question
        questions = ["What are the key elements in this scene?"]
        print(f"Questions file not found, using default question")

    # 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load EventGPT
    print("\n[1/2] Loading EventGPT (4-bit)...")
    from model.EventChatModel import EventChatModel

    eventgpt_path = "./checkpoints/EventGPT-7b"
    egpt_model = EventChatModel.from_pretrained(
        eventgpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    egpt_model.eval()
    egpt_tokenizer = AutoTokenizer.from_pretrained(eventgpt_path, use_fast=True)
    egpt_processor = egpt_model.get_visual_tower().event_processor
    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

    # Extract EventGPT tokens for all questions
    print(f"\nExtracting EventGPT tokens ({len(dataset)} samples x {len(questions)} questions = {len(dataset)*len(questions)} pairs)...")
    egpt_results = {}  # (sample_idx, question_idx) -> result
    egpt_errors = 0
    total_pairs = len(dataset) * len(questions)

    with tqdm(total=total_pairs, desc="EventGPT") as pbar:
        for i, sample in enumerate(dataset):
            for q_idx, query in enumerate(questions):
                result = extract_eventgpt_tokens(egpt_model, egpt_tokenizer, egpt_processor, sample, dataset_dir, query, args.max_new_tokens)
                if result and 'error' not in result:
                    egpt_results[(i, q_idx)] = {
                        'tokens': result['output_tokens'],
                        'text': result['output_text'],
                        'query': query
                    }
                else:
                    egpt_errors += 1
                    if egpt_errors <= 5:
                        print(f"\nEventGPT error sample={i} q={q_idx}: {result.get('error', 'Unknown') if result else 'None'}")
                pbar.update(1)

    print(f"EventGPT extracted: {len(egpt_results)}/{total_pairs} ({egpt_errors} errors)")

    # Unload EventGPT
    del egpt_model, egpt_tokenizer, egpt_processor
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    print(f"  GPU Memory after unload: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

    # Load Video-LLaVA
    print("\n[2/2] Loading Video-LLaVA (4-bit)...")
    videollava_model_id = "LanguageBind/Video-LLaVA-7B-hf"
    vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
        videollava_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    vl_processor = VideoLlavaProcessor.from_pretrained(videollava_model_id)
    vl_model.eval()
    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

    # Extract Video-LLaVA tokens (only for samples where EventGPT succeeded)
    print(f"\nExtracting Video-LLaVA tokens ({len(egpt_results)} pairs)...")
    final_results = []
    vl_errors = 0
    pairs_to_process = list(egpt_results.keys())

    with tqdm(total=len(pairs_to_process), desc="Video-LLaVA") as pbar:
        for (sample_idx, q_idx) in pairs_to_process:
            sample = dataset[sample_idx]
            query = egpt_results[(sample_idx, q_idx)]['query']
            result = extract_videollava_tokens(vl_model, vl_processor, sample, dataset_dir, query, args.max_new_tokens)
            if result and 'error' not in result:
                final_results.append({
                    'sample_idx': sample_idx,
                    'question_idx': q_idx,
                    'sample_id': sample.get('id'),
                    'query': query,
                    'egpt_tokens': egpt_results[(sample_idx, q_idx)]['tokens'],
                    'egpt_text': egpt_results[(sample_idx, q_idx)]['text'],
                    'vl_tokens': result['output_tokens'],
                    'vl_text': result['output_text']
                })
            else:
                vl_errors += 1
                if vl_errors <= 5:
                    print(f"\nVideo-LLaVA error sample={sample_idx} q={q_idx}: {result.get('error', 'Unknown') if result else 'None'}")
            pbar.update(1)

    print(f"Video-LLaVA extracted: {len(final_results)}/{len(pairs_to_process)} ({vl_errors} errors)")

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_dir': str(dataset_dir),
        'total_samples': len(dataset),
        'num_questions': len(questions),
        'questions': questions,
        'eventgpt_success': len(egpt_results),
        'videollava_success': len(final_results),
        'results': final_results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Extraction Complete")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    print(f"Questions: {len(questions)}")
    print(f"Target pairs: {len(dataset) * len(questions)}")
    print(f"EventGPT success: {len(egpt_results)}")
    print(f"Video-LLaVA success: {len(final_results)}")
    print(f"Final pairs: {len(final_results)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
