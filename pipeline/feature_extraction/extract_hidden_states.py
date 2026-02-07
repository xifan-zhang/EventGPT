#!/usr/bin/env python3
"""
Extract decoder hidden states from EventGPT and Video-LLaVA.

Author: Alice Zhang
Date: 2026-02-07

For each sample, extract hidden states from both models using:
- Same scene (event image for EGPT, video for VL)
- Same question
- Same partial response prefix (for autoregressive hidden states)

Output:
    hidden_states_train.pt: {
        'egpt_hidden': [N, seq, 4096],
        'vl_hidden': [N, seq, 4096],
        'metadata': [...]
    }
"""

import os
import sys
import json
import torch
import gc
import signal
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass

# Global state for signal handler
_save_state = {
    'output_path': None,
    'all_egpt_hidden': None,
    'all_vl_hidden': None,
    'metadata': None,
    'args': None,
    'dirty': False,  # True if unsaved data exists
}

def _emergency_save(signum, frame):
    """Save checkpoint on interrupt signal."""
    if _save_state['dirty'] and _save_state['all_egpt_hidden'] is not None:
        print(f"\n\nReceived signal {signum}, saving checkpoint...")
        try:
            from extract_hidden_states import save_results
            save_results(
                _save_state['output_path'],
                _save_state['all_egpt_hidden'],
                _save_state['all_vl_hidden'],
                _save_state['metadata'],
                _save_state['args']
            )
            print("Emergency save complete.")
        except Exception as e:
            print(f"Emergency save failed: {e}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, _emergency_save)
signal.signal(signal.SIGHUP, _emergency_save)
signal.signal(signal.SIGINT, _emergency_save)

# Fix protobuf compatibility
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer, BitsAndBytesConfig, VideoLlavaForConditionalGeneration, VideoLlavaProcessor

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    raise ImportError("PyAV required. Install with: pip install av")


def load_video_frames_from_mp4(video_path: str, num_frames: int = 8) -> List:
    """Load frames from MP4 video."""
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


class HiddenStateExtractor:
    """Extract hidden states from both models."""

    def __init__(self, max_new_tokens: int = 50):
        self.max_new_tokens = max_new_tokens
        self.device = "cuda"

        # 4-bit config
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
        """Load both models."""
        from model.EventChatModel import EventChatModel
        from common.common import load_image

        print("\n[1/2] Loading EventGPT (4-bit)...")
        eventgpt_path = "/home/ps/Documents/code/EventGPT/checkpoints/EventGPT-7b"
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

    def extract_egpt_hidden(
        self,
        sample: Dict,
        dataset_dir: Path,
        query: str,
    ) -> Optional[Tuple[torch.Tensor, List[int], str]]:
        """
        Extract hidden states from EventGPT.

        Returns:
            Tuple of (hidden_states [seq, hidden_dim], token_ids, output_text)
        """
        from model.EventChatModel import get_spatio_temporal_features
        from dataset.conversation import prepare_event_prompt
        from dataset.constants import EVENT_TOKEN_INDEX
        from common.common import tokenizer_event_token, load_image

        try:
            if "event_image" not in sample or not sample["event_image"]:
                return None

            event_image_paths = sample["event_image"]
            img_path = dataset_dir / "event_image" / event_image_paths[0]
            if not img_path.exists():
                return None

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

            # Generate with hidden states output
            with torch.inference_mode():
                outputs = self.egpt_model.generate(
                    input_ids,
                    event_features=event_features,
                    event_image_sizes=[event_image_size],
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Get generated tokens
            generated_ids = outputs.sequences[0]
            output_text = self.egpt_tokenizer.decode(generated_ids, skip_special_tokens=True)
            if "ASSISTANT:" in output_text:
                output_text = output_text.split("ASSISTANT:")[-1].strip()

            # Extract hidden states from generation
            # outputs.hidden_states is a tuple of (num_tokens, num_layers, batch, seq, hidden)
            # We want the last layer hidden states for each generated token
            all_hidden = []
            for step_hidden in outputs.hidden_states:
                # step_hidden is tuple of (num_layers, batch, seq, hidden)
                # Get last layer: step_hidden[-1] is [batch, seq, hidden]
                # We want the last position (newly generated token)
                last_layer = step_hidden[-1]  # [batch, seq, hidden]
                last_pos = last_layer[0, -1, :]  # [hidden]
                all_hidden.append(last_pos)

            hidden_states = torch.stack(all_hidden, dim=0)  # [num_tokens, hidden]

            return hidden_states.cpu(), generated_ids.tolist(), output_text

        except Exception as e:
            print(f"EGPT error: {e}")
            return None

    def extract_vl_hidden(
        self,
        sample: Dict,
        dataset_dir: Path,
        query: str,
    ) -> Optional[Tuple[torch.Tensor, List[int], str]]:
        """
        Extract hidden states from Video-LLaVA.

        Returns:
            Tuple of (hidden_states [seq, hidden_dim], token_ids, output_text)
        """
        try:
            video_data = sample.get("video_data")
            if not video_data:
                return None

            mp4_path = dataset_dir / "mp4" / f"{video_data}.mp4"
            if not mp4_path.exists():
                return None

            frames = load_video_frames_from_mp4(str(mp4_path), num_frames=8)
            if not frames:
                return None

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

            # Generate with hidden states output
            with torch.inference_mode():
                outputs = self.vl_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Get generated tokens
            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            output_text = self.vl_processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Extract hidden states
            all_hidden = []
            for step_hidden in outputs.hidden_states:
                last_layer = step_hidden[-1]  # [batch, seq, hidden]
                last_pos = last_layer[0, -1, :]  # [hidden]
                all_hidden.append(last_pos)

            hidden_states = torch.stack(all_hidden, dim=0)  # [num_tokens, hidden]

            return hidden_states.cpu(), generated_ids.tolist(), output_text

        except Exception as e:
            print(f"VL error: {e}")
            return None

    def extract_pair(
        self,
        sample: Dict,
        dataset_dir: Path,
        query: str,
    ) -> Optional[Dict]:
        """Extract hidden states from both models for a sample."""

        egpt_result = self.extract_egpt_hidden(sample, dataset_dir, query)
        if egpt_result is None:
            return None

        vl_result = self.extract_vl_hidden(sample, dataset_dir, query)
        if vl_result is None:
            return None

        egpt_hidden, egpt_tokens, egpt_text = egpt_result
        vl_hidden, vl_tokens, vl_text = vl_result

        # Align lengths (take minimum)
        min_len = min(len(egpt_hidden), len(vl_hidden))
        if min_len < 5:  # Too short
            return None

        egpt_hidden = egpt_hidden[:min_len]
        vl_hidden = vl_hidden[:min_len]

        return {
            'egpt_hidden': egpt_hidden,
            'vl_hidden': vl_hidden,
            'egpt_tokens': egpt_tokens,
            'vl_tokens': vl_tokens,
            'egpt_text': egpt_text,
            'vl_text': vl_text,
            'seq_len': min_len,
        }


def load_checkpoint(output_path: Path) -> Tuple[List, List, List, set]:
    """
    Load existing checkpoint for resuming.

    Returns:
        Tuple of (egpt_hidden_list, vl_hidden_list, metadata_list, processed_set)
        where processed_set contains (sample_idx, question_idx) tuples
    """
    if not output_path.exists():
        return [], [], [], set()

    print(f"Loading existing checkpoint from {output_path}...")
    data = torch.load(output_path, map_location='cpu')

    egpt_tensor = data['egpt_hidden']
    vl_tensor = data['vl_hidden']
    seq_lens = data['seq_lens']
    metadata = data['metadata']

    # Convert tensors back to list format
    all_egpt_hidden = []
    all_vl_hidden = []
    for i in range(len(egpt_tensor)):
        seq_len = seq_lens[i].item()
        all_egpt_hidden.append(egpt_tensor[i, :seq_len])
        all_vl_hidden.append(vl_tensor[i, :seq_len])

    # Build set of processed (sample_idx, question_idx)
    processed = set()
    for m in metadata:
        processed.add((m['sample_idx'], m['question_idx']))

    print(f"  Loaded {len(all_egpt_hidden)} existing samples")
    print(f"  Processed pairs: {len(processed)}")

    return all_egpt_hidden, all_vl_hidden, metadata, processed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract hidden states from EGPT and VL")
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_train/my_egpt_dsec_train_1s')
    parser.add_argument('--output_dir', type=str,
                        default=str(Path(__file__).parent / 'data'))
    parser.add_argument('--questions_file', type=str,
                        default='./feasible/token_alignment/top50_questions.json')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Limit number of samples (-1 for all)')
    parser.add_argument('--max_questions', type=int, default=10,
                        help='Number of questions to use (default: 10)')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoint')
    parser.add_argument('--duration', type=str, default='1s',
                        help='Duration tag for checkpoint filename (e.g., 1s, 500ms)')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Custom checkpoint filename (overrides auto-generated name)')
    parser.add_argument('--quant', type=str, default='4bit',
                        help='Quantization tag for checkpoint filename (e.g., 4bit, 8bit, fp16)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save checkpoint every N samples (default: 1000)')
    parser.add_argument('--chunked', action='store_true',
                        help='Use chunked incremental saving (memory efficient)')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Samples per chunk when using --chunked (default: 1000)')
    args = parser.parse_args()

    # Auto-select dataset path based on split if using default
    if args.dataset_dir == '/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_train/my_egpt_dsec_train_1s':
        if args.split == 'test':
            args.dataset_dir = '/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_test/my_egpt_dsec_seq_1s'
            print(f"Using test dataset: {args.dataset_dir}")

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading {args.split} dataset...")
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
        questions = questions[:args.max_questions]
        print(f"Using top {len(questions)} questions")
    else:
        questions = ["What are the key elements in this scene?"]
        print("Questions file not found, using default question")

    total_pairs = len(dataset) * len(questions)
    print(f"Total pairs to extract: {total_pairs}")

    # Generate checkpoint filename
    # Format: hidden_states_{split}_{duration}_{quant}_{datetime}_top{N}q.pt
    # Example: hidden_states_train_1s_4bit_20260131_top10q.pt
    if args.checkpoint_name:
        output_filename = args.checkpoint_name
        if not output_filename.endswith('.pt'):
            output_filename += '.pt'
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        output_filename = f"hidden_states_{args.split}_{args.duration}_{args.quant}_{date_str}_top{args.max_questions}q.pt"

    output_path = output_dir / output_filename
    print(f"Checkpoint file: {output_path}")

    # Also check for legacy filename for backward compatibility
    legacy_path = output_dir / f"hidden_states_{args.split}_{args.max_questions}q.pt"

    if args.resume:
        # Try loading from new path first, then legacy path
        if output_path.exists():
            all_egpt_hidden, all_vl_hidden, metadata, processed = load_checkpoint(output_path)
        elif legacy_path.exists():
            print(f"New checkpoint not found, trying legacy path: {legacy_path}")
            all_egpt_hidden, all_vl_hidden, metadata, processed = load_checkpoint(legacy_path)
        else:
            print(f"No checkpoint found at {output_path} or {legacy_path}")
            all_egpt_hidden, all_vl_hidden, metadata, processed = [], [], [], set()

        if len(all_egpt_hidden) > 0:
            print(f"Resuming from {len(all_egpt_hidden)} samples...")
            print(f"Remaining pairs: {total_pairs - len(processed)}")
    else:
        all_egpt_hidden = []
        all_vl_hidden = []
        metadata = []
        processed = set()

    # Initialize extractor
    extractor = HiddenStateExtractor(max_new_tokens=args.max_new_tokens)
    extractor.load_models()

    # Choose saving mode: chunked (incremental) or batch
    if args.chunked:
        # Chunked mode: save incrementally to avoid memory issues
        chunked_dir = output_dir / f"chunked_{args.split}_{args.duration}_{args.quant}"
        config = {
            'split': args.split,
            'duration': args.duration,
            'quant': args.quant,
            'max_questions': args.max_questions,
            'max_new_tokens': args.max_new_tokens,
            'hidden_dim': 4096,
        }
        writer = ChunkedHiddenStateWriter(chunked_dir, chunk_size=args.chunk_size, config=config)
        processed = writer.get_processed_pairs()
        start_count = writer.total_samples
        print(f"Chunked mode: saving to {chunked_dir}")
        print(f"  Chunk size: {args.chunk_size}")
        print(f"  Already processed: {len(processed)} pairs")
    else:
        # Batch mode: accumulate in memory
        writer = None
        start_count = len(all_egpt_hidden)
        last_save_count = start_count

        # Setup global state for signal handler
        _save_state['output_path'] = output_path
        _save_state['all_egpt_hidden'] = all_egpt_hidden
        _save_state['all_vl_hidden'] = all_vl_hidden
        _save_state['metadata'] = metadata
        _save_state['args'] = args

    # Extract hidden states
    print(f"\nExtracting hidden states...")
    errors = 0
    skipped = 0
    extracted_count = start_count

    with tqdm(total=total_pairs, desc="Extracting", initial=len(processed)) as pbar:
        for sample_idx, sample in enumerate(dataset):
            for q_idx, query in enumerate(questions):
                # Skip if already processed
                if (sample_idx, q_idx) in processed:
                    skipped += 1
                    continue

                result = extractor.extract_pair(sample, dataset_dir, query)
                if result:
                    meta = {
                        'sample_idx': sample_idx,
                        'question_idx': q_idx,
                        'question': query,
                        'egpt_text': result['egpt_text'],
                        'vl_text': result['vl_text'],
                        'seq_len': result['seq_len'],
                    }

                    if args.chunked:
                        # Chunked mode: add to writer (auto-saves when chunk full)
                        writer.add_sample(result['egpt_hidden'], result['vl_hidden'], meta)
                        extracted_count = writer.total_samples + len(writer.current_chunk_egpt)
                    else:
                        # Batch mode: accumulate in memory
                        all_egpt_hidden.append(result['egpt_hidden'])
                        all_vl_hidden.append(result['vl_hidden'])
                        metadata.append(meta)
                        extracted_count = len(all_egpt_hidden)
                        _save_state['dirty'] = True

                    processed.add((sample_idx, q_idx))
                else:
                    errors += 1

                pbar.update(1)
                pbar.set_postfix({
                    'extracted': extracted_count,
                    'new': extracted_count - start_count,
                    'errors': errors
                })

                # Periodic save (batch mode only)
                if not args.chunked:
                    new_count = len(all_egpt_hidden) - last_save_count
                    if new_count >= args.save_interval:
                        print(f"\nSaving checkpoint ({len(all_egpt_hidden)} samples, +{new_count} new)...")
                        try:
                            save_results(output_path, all_egpt_hidden, all_vl_hidden, metadata, args)
                            last_save_count = len(all_egpt_hidden)
                            _save_state['dirty'] = False
                        except Exception as e:
                            print(f"  WARNING: Save failed ({e}), will retry later...")

    # Final save
    print(f"\n\nExtraction complete!")
    if args.chunked:
        writer.flush()  # Save remaining samples
        final_count = writer.total_samples
        print(f"  Total samples: {final_count}")
        print(f"  New samples: {final_count - start_count}")
        print(f"  Errors: {errors}")
        print(f"  Chunks saved to: {chunked_dir}")
    else:
        final_count = len(all_egpt_hidden)
        print(f"  Total samples: {final_count}")
        print(f"  New samples: {final_count - start_count}")
        print(f"  Errors: {errors}")
        save_results(output_path, all_egpt_hidden, all_vl_hidden, metadata, args)
    print(f"  Errors: {errors}")

    save_results(output_path, all_egpt_hidden, all_vl_hidden, metadata, args)


def save_results(output_path, all_egpt_hidden, all_vl_hidden, metadata, args):
    """Save extracted hidden states with memory-efficient approach."""
    import gc

    # Stack hidden states with padding - one tensor at a time to save memory
    max_seq_len = max(h.shape[0] for h in all_egpt_hidden)
    hidden_dim = all_egpt_hidden[0].shape[1]
    n_samples = len(all_egpt_hidden)

    print(f"  Stacking {n_samples} samples (max_seq={max_seq_len}, dim={hidden_dim})...")

    # Build seq_lens first (small)
    seq_lens = [h.shape[0] for h in all_egpt_hidden]

    # Stack EGPT tensor on CPU to avoid GPU OOM
    print("  Creating EGPT tensor on CPU...")
    egpt_tensor = torch.zeros(n_samples, max_seq_len, hidden_dim, dtype=torch.float32, device='cpu')
    for i, egpt_h in enumerate(all_egpt_hidden):
        egpt_tensor[i, :seq_lens[i]] = egpt_h.cpu().float()

    # Stack VL tensor on CPU
    print("  Creating VL tensor on CPU...")
    vl_tensor = torch.zeros(n_samples, max_seq_len, hidden_dim, dtype=torch.float32, device='cpu')
    for i, vl_h in enumerate(all_vl_hidden):
        vl_tensor[i, :seq_lens[i]] = vl_h.cpu().float()

    data = {
        'egpt_hidden': egpt_tensor,
        'vl_hidden': vl_tensor,
        'seq_lens': torch.tensor(seq_lens),
        'metadata': metadata,
        'config': {
            # Basic config
            'split': args.split,
            'duration': args.duration,
            'max_questions': args.max_questions,
            'max_new_tokens': args.max_new_tokens,
            'max_samples': args.max_samples,
            # Data info
            'hidden_dim': hidden_dim,
            'num_samples': len(all_egpt_hidden),
            'max_seq_len': max_seq_len,
            # Timestamps
            'created': datetime.now().isoformat(),
            'date': datetime.now().strftime("%Y%m%d"),
            # Paths
            'dataset_dir': str(args.dataset_dir),
            'questions_file': str(args.questions_file),
        }
    }

    torch.save(data, output_path)
    print(f"Saved to {output_path}")
    print(f"  Shape: egpt={egpt_tensor.shape}, vl={vl_tensor.shape}")

    # Free memory
    del data, egpt_tensor, vl_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# INCREMENTAL (CHUNKED) SAVING - Memory efficient for large datasets
# ============================================================================

class ChunkedHiddenStateWriter:
    """
    Incrementally save hidden states in chunks to avoid memory issues.

    Directory structure:
        output_dir/
        ├── chunks/
        │   ├── chunk_000000.pt  # samples 0-999
        │   ├── chunk_001000.pt  # samples 1000-1999
        │   └── ...
        └── index.json          # metadata and chunk info
    """

    def __init__(self, output_dir: Path, chunk_size: int = 1000, config: dict = None):
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.config = config or {}

        # Create directories
        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Current chunk buffer
        self.current_chunk_egpt = []
        self.current_chunk_vl = []
        self.current_chunk_metadata = []
        self.current_chunk_start = 0

        # Global state
        self.total_samples = 0
        self.chunk_files = []

        # Load existing index if resuming
        self.index_path = self.output_dir / "index.json"
        if self.index_path.exists():
            self._load_index()

    def _load_index(self):
        """Load existing index for resume."""
        with open(self.index_path) as f:
            index = json.load(f)
        self.total_samples = index.get('total_samples', 0)
        self.chunk_files = index.get('chunks', [])
        self.current_chunk_start = self.total_samples
        print(f"  Loaded index: {self.total_samples} samples in {len(self.chunk_files)} chunks")

    def _save_index(self):
        """Save index file."""
        index = {
            'total_samples': self.total_samples,
            'chunk_size': self.chunk_size,
            'chunks': self.chunk_files,
            'config': self.config,
            'created': datetime.now().isoformat(),
        }
        with open(self.index_path, 'w') as f:
            json.dump(index, f, indent=2)

    def add_sample(self, egpt_hidden: torch.Tensor, vl_hidden: torch.Tensor, meta: dict):
        """Add a single sample to the current chunk."""
        # Move to CPU immediately to save GPU memory
        self.current_chunk_egpt.append(egpt_hidden.cpu().float())
        self.current_chunk_vl.append(vl_hidden.cpu().float())
        self.current_chunk_metadata.append(meta)

        # Check if chunk is full
        if len(self.current_chunk_egpt) >= self.chunk_size:
            self._flush_chunk()

    def _flush_chunk(self):
        """Save current chunk to disk and clear buffer."""
        if not self.current_chunk_egpt:
            return

        n = len(self.current_chunk_egpt)
        chunk_id = self.current_chunk_start
        chunk_path = self.chunks_dir / f"chunk_{chunk_id:06d}.pt"

        # Stack tensors
        max_seq = max(h.shape[0] for h in self.current_chunk_egpt)
        hidden_dim = self.current_chunk_egpt[0].shape[1]

        egpt_tensor = torch.zeros(n, max_seq, hidden_dim, dtype=torch.float32)
        vl_tensor = torch.zeros(n, max_seq, hidden_dim, dtype=torch.float32)
        seq_lens = []

        for i, (eh, vh) in enumerate(zip(self.current_chunk_egpt, self.current_chunk_vl)):
            seq_len = eh.shape[0]
            egpt_tensor[i, :seq_len] = eh
            vl_tensor[i, :seq_len] = vh
            seq_lens.append(seq_len)

        # Save chunk
        chunk_data = {
            'egpt_hidden': egpt_tensor,
            'vl_hidden': vl_tensor,
            'seq_lens': torch.tensor(seq_lens),
            'metadata': self.current_chunk_metadata,
            'chunk_id': chunk_id,
            'n_samples': n,
        }
        torch.save(chunk_data, chunk_path)

        # Update state
        self.chunk_files.append({
            'path': str(chunk_path.name),
            'start_idx': chunk_id,
            'n_samples': n,
        })
        self.total_samples += n
        self.current_chunk_start = self.total_samples

        # Clear buffer
        self.current_chunk_egpt = []
        self.current_chunk_vl = []
        self.current_chunk_metadata = []

        # Save index
        self._save_index()

        print(f"  Saved chunk {chunk_path.name}: {n} samples (total: {self.total_samples})")

        # Free memory
        del egpt_tensor, vl_tensor, chunk_data
        gc.collect()

    def flush(self):
        """Flush any remaining samples in buffer."""
        if self.current_chunk_egpt:
            self._flush_chunk()

    def get_processed_pairs(self) -> set:
        """Get set of already processed (sample_idx, question_idx) pairs."""
        processed = set()
        for chunk_info in self.chunk_files:
            chunk_path = self.chunks_dir / chunk_info['path']
            if chunk_path.exists():
                data = torch.load(chunk_path, map_location='cpu')
                for meta in data['metadata']:
                    processed.add((meta['sample_idx'], meta['question_idx']))
                del data
        return processed


def load_chunked_hidden_states(output_dir: Path) -> dict:
    """
    Load all chunks and concatenate into single tensors.

    Returns:
        {
            'egpt_hidden': tensor [N, max_seq, 4096],
            'vl_hidden': tensor [N, max_seq, 4096],
            'seq_lens': tensor [N],
            'metadata': list,
            'config': dict,
        }
    """
    output_dir = Path(output_dir)
    index_path = output_dir / "index.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    all_egpt = []
    all_vl = []
    all_seq_lens = []
    all_metadata = []

    chunks_dir = output_dir / "chunks"
    for chunk_info in index['chunks']:
        chunk_path = chunks_dir / chunk_info['path']
        data = torch.load(chunk_path, map_location='cpu')
        all_egpt.append(data['egpt_hidden'])
        all_vl.append(data['vl_hidden'])
        all_seq_lens.append(data['seq_lens'])
        all_metadata.extend(data['metadata'])
        del data

    # Find max seq len across all chunks
    max_seq = max(t.shape[1] for t in all_egpt)
    hidden_dim = all_egpt[0].shape[2]
    total_samples = sum(t.shape[0] for t in all_egpt)

    # Concatenate with padding
    egpt_tensor = torch.zeros(total_samples, max_seq, hidden_dim)
    vl_tensor = torch.zeros(total_samples, max_seq, hidden_dim)
    seq_lens_tensor = torch.cat(all_seq_lens)

    idx = 0
    for egpt, vl in zip(all_egpt, all_vl):
        n, seq, _ = egpt.shape
        egpt_tensor[idx:idx+n, :seq] = egpt
        vl_tensor[idx:idx+n, :seq] = vl
        idx += n

    return {
        'egpt_hidden': egpt_tensor,
        'vl_hidden': vl_tensor,
        'seq_lens': seq_lens_tensor,
        'metadata': all_metadata,
        'config': index.get('config', {}),
    }


if __name__ == "__main__":
    main()
