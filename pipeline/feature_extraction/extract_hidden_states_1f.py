#!/usr/bin/env python3
"""
Extract EventGPT hidden states using 1-frame event images, reuse existing VL hidden states.

Author: Alice Zhang
Date: 2026-02-07

Reads existing chunked data (extracted with 5-frame event_image) and re-extracts
only the EGPT side using event_image_1f (1 frame). VL hidden states are copied
from existing chunks unchanged.

Output format is identical to extract_hidden_states.py — works with all
downstream training/evaluation scripts.
"""

import os
import sys
import json
import torch
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Fix protobuf compatibility
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer, BitsAndBytesConfig


class EGPTExtractor:
    """Extract hidden states from EventGPT only."""

    def __init__(self, max_new_tokens: int = 50, event_image_key: str = 'event_image_1f'):
        self.max_new_tokens = max_new_tokens
        self.event_image_key = event_image_key
        self.device = "cuda"

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self):
        """Load EventGPT model only."""
        from model.EventChatModel import EventChatModel

        print("\nLoading EventGPT (4-bit)...")
        eventgpt_path = "/home/ps/Documents/code/EventGPT/checkpoints/EventGPT-7b"
        self.model = EventChatModel.from_pretrained(
            eventgpt_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=self.bnb_config,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(eventgpt_path, use_fast=True)
        self.processor = self.model.get_visual_tower().event_processor
        print(f"  EventGPT loaded. GPU Memory: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

    def extract_hidden(
        self,
        sample: Dict,
        dataset_dir: Path,
        query: str,
    ) -> Optional[Tuple[torch.Tensor, List[int], str]]:
        """
        Extract hidden states from EventGPT using event_image_1f.

        Returns:
            Tuple of (hidden_states [seq, 4096], token_ids, output_text) or None
        """
        from model.EventChatModel import get_spatio_temporal_features
        from dataset.conversation import prepare_event_prompt
        from dataset.constants import EVENT_TOKEN_INDEX
        from common.common import tokenizer_event_token, load_image

        try:
            if self.event_image_key not in sample or not sample[self.event_image_key]:
                return None

            event_image_paths = sample[self.event_image_key]
            img_path = dataset_dir / self.event_image_key / event_image_paths[0]
            if not img_path.exists():
                return None

            img = load_image(str(img_path))
            img_array = np.array(img)
            event_image_size = list(img_array.shape[:2])

            event = self.processor(img_array, return_tensors='pt')['pixel_values'][0]
            event = event.to(self.device, dtype=torch.bfloat16)

            conv_mode = 'eventgpt_v1'
            prompt = prepare_event_prompt(query, conv_mode)
            input_ids = tokenizer_event_token(
                prompt, self.tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                feature = self.model.visval_encode(event.unsqueeze(0))
                feature = self.model.get_model().feature_adaptor(feature)
                feature = feature.squeeze(0)
                event_features = get_spatio_temporal_features([feature])
                event_features = event_features.unsqueeze(0)

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids,
                    event_features=event_features,
                    event_image_sizes=[event_image_size],
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            generated_ids = outputs.sequences[0]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if "ASSISTANT:" in output_text:
                output_text = output_text.split("ASSISTANT:")[-1].strip()

            all_hidden = []
            for step_hidden in outputs.hidden_states:
                last_layer = step_hidden[-1]
                last_pos = last_layer[0, -1, :]
                all_hidden.append(last_pos)

            hidden_states = torch.stack(all_hidden, dim=0)

            return hidden_states.cpu(), generated_ids.tolist(), output_text

        except Exception as e:
            print(f"EGPT error: {e}")
            return None


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract EGPT hidden states with 1-frame input, reuse existing VL hidden states"
    )
    parser.add_argument('--existing_chunks', type=str, required=True,
                        help='Path to existing chunked data (e.g., feasible/feature_alignment/data/chunked_train_1s_4bit)')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to DSEC dataset with event images + MP4')
    parser.add_argument('--output_dir', type=str,
                        default=str(Path(__file__).parent / 'data'),
                        help='Output base directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--event_image_key', type=str, default='event_image_1f',
                        help='JSON key for event image paths (default: event_image_1f)')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--questions_file', type=str,
                        default=str(ROOT / 'feasible/token_alignment/top50_questions.json'))
    parser.add_argument('--max_questions', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output chunks')
    args = parser.parse_args()

    existing_dir = Path(args.existing_chunks)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    # Derive suffix from event_image_key (e.g., "event_image_1f" -> "1f")
    suffix = args.event_image_key.replace("event_image_", "")
    out_name = f"chunked_{args.split}_1s_4bit_{suffix}"
    chunked_out = output_dir / out_name

    print(f"=== Extract 1-Frame EGPT Hidden States ===")
    print(f"  Existing chunks: {existing_dir}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Event image key: {args.event_image_key}")
    print(f"  Output: {chunked_out}")

    # Load existing index
    index_path = existing_dir / "index.json"
    if not index_path.exists():
        print(f"ERROR: Index not found at {index_path}")
        sys.exit(1)

    with open(index_path) as f:
        existing_index = json.load(f)

    total_existing = existing_index['total_samples']
    existing_chunks = existing_index['chunks']
    print(f"  Existing data: {total_existing} samples in {len(existing_chunks)} chunks")

    # Load dataset JSON
    json_path = dataset_dir / "EventGPT_Instruction_Subset.json"
    with open(json_path) as f:
        dataset = json.load(f)
    print(f"  Dataset entries: {len(dataset)}")

    # Load questions
    questions_file = Path(args.questions_file)
    if questions_file.exists():
        with open(questions_file) as f:
            questions_data = json.load(f)
        questions = [q['question'] for q in questions_data]
        questions = questions[:args.max_questions]
        print(f"  Questions: {len(questions)}")
    else:
        questions = ["What are the key elements in this scene?"]
        print("  Questions file not found, using default question")

    # Setup output writer
    chunked_out.mkdir(parents=True, exist_ok=True)
    chunks_out_dir = chunked_out / "chunks"
    chunks_out_dir.mkdir(exist_ok=True)

    # Check resume state
    out_index_path = chunked_out / "index.json"
    completed_chunks = set()
    if args.resume and out_index_path.exists():
        with open(out_index_path) as f:
            out_index = json.load(f)
        for ci in out_index.get('chunks', []):
            completed_chunks.add(ci['source_chunk'])
        print(f"  Resuming: {len(completed_chunks)} chunks already done")

    # Load EGPT model
    extractor = EGPTExtractor(
        max_new_tokens=args.max_new_tokens,
        event_image_key=args.event_image_key,
    )
    extractor.load_model()

    # Process chunk by chunk
    out_chunks_info = []
    total_processed = 0
    total_errors = 0

    # Reload completed chunks info if resuming
    if completed_chunks:
        with open(out_index_path) as f:
            out_index = json.load(f)
        out_chunks_info = out_index.get('chunks', [])
        total_processed = sum(ci['n_samples'] for ci in out_chunks_info)

    existing_chunks_dir = existing_dir / "chunks"

    for chunk_idx, chunk_info in enumerate(existing_chunks):
        chunk_name = chunk_info['path']

        # Skip if already done
        if chunk_name in completed_chunks:
            print(f"\n[Chunk {chunk_idx+1}/{len(existing_chunks)}] {chunk_name} — already done, skipping")
            continue

        print(f"\n[Chunk {chunk_idx+1}/{len(existing_chunks)}] Processing {chunk_name}...")

        # Load existing chunk
        chunk_path = existing_chunks_dir / chunk_name
        chunk_data = torch.load(chunk_path, map_location='cpu')
        vl_hidden = chunk_data['vl_hidden']      # [N, max_seq, 4096]
        seq_lens = chunk_data['seq_lens']          # [N]
        metadata = chunk_data['metadata']          # list of dicts
        n_samples = len(metadata)

        print(f"  Samples in chunk: {n_samples}")

        # Extract new EGPT hidden states for each sample
        new_egpt_list = []
        new_vl_list = []
        new_meta_list = []
        chunk_errors = 0

        for i in tqdm(range(n_samples), desc=f"  Chunk {chunk_idx+1}", leave=False):
            meta = metadata[i]
            sample_idx = meta['sample_idx']
            question_idx = meta['question_idx']
            target_len = seq_lens[i].item()

            # Get dataset entry and question
            if sample_idx >= len(dataset):
                print(f"  WARNING: sample_idx {sample_idx} out of range, skipping")
                chunk_errors += 1
                continue

            sample = dataset[sample_idx]
            if question_idx >= len(questions):
                print(f"  WARNING: question_idx {question_idx} out of range, skipping")
                chunk_errors += 1
                continue

            query = questions[question_idx]

            # Extract EGPT hidden with 1f image
            result = extractor.extract_hidden(sample, dataset_dir, query)
            if result is None:
                chunk_errors += 1
                continue

            egpt_hidden_1f, _, egpt_text_1f = result

            # Truncate to match existing seq_len
            actual_len = min(len(egpt_hidden_1f), target_len)
            if actual_len < 5:
                chunk_errors += 1
                continue

            egpt_h = egpt_hidden_1f[:actual_len].float()
            vl_h = vl_hidden[i, :actual_len].float()

            new_egpt_list.append(egpt_h)
            new_vl_list.append(vl_h)

            # Update metadata with 1f info
            new_meta = dict(meta)
            new_meta['egpt_text'] = egpt_text_1f
            new_meta['seq_len'] = actual_len
            new_meta['event_image_key'] = args.event_image_key
            new_meta_list.append(new_meta)

        # Free existing chunk data
        del chunk_data, vl_hidden
        gc.collect()

        if not new_egpt_list:
            print(f"  No valid samples in chunk, skipping")
            total_errors += chunk_errors
            continue

        # Stack and save new chunk
        n_new = len(new_egpt_list)
        max_seq = max(h.shape[0] for h in new_egpt_list)
        hidden_dim = new_egpt_list[0].shape[1]

        egpt_tensor = torch.zeros(n_new, max_seq, hidden_dim, dtype=torch.float32)
        vl_tensor = torch.zeros(n_new, max_seq, hidden_dim, dtype=torch.float32)
        new_seq_lens = []

        for j, (eh, vh) in enumerate(zip(new_egpt_list, new_vl_list)):
            sl = eh.shape[0]
            egpt_tensor[j, :sl] = eh
            vl_tensor[j, :sl] = vh
            new_seq_lens.append(sl)

        out_chunk_path = chunks_out_dir / chunk_name
        out_chunk_data = {
            'egpt_hidden': egpt_tensor,
            'vl_hidden': vl_tensor,
            'seq_lens': torch.tensor(new_seq_lens),
            'metadata': new_meta_list,
            'chunk_id': chunk_info.get('start_idx', chunk_idx * args.chunk_size),
            'n_samples': n_new,
        }
        torch.save(out_chunk_data, out_chunk_path)

        out_chunks_info.append({
            'path': chunk_name,
            'start_idx': chunk_info.get('start_idx', chunk_idx * args.chunk_size),
            'n_samples': n_new,
            'source_chunk': chunk_name,
        })

        total_processed += n_new
        total_errors += chunk_errors

        print(f"  Saved {n_new} samples ({chunk_errors} errors)")

        # Save index after each chunk (for resume)
        out_index = {
            'total_samples': total_processed,
            'chunk_size': args.chunk_size,
            'chunks': out_chunks_info,
            'config': {
                'split': args.split,
                'duration': '1s',
                'quant': '4bit',
                'max_questions': args.max_questions,
                'max_new_tokens': args.max_new_tokens,
                'hidden_dim': 4096,
                'event_image_key': args.event_image_key,
                'source_chunks': str(existing_dir),
            },
            'created': datetime.now().isoformat(),
        }
        with open(out_index_path, 'w') as f:
            json.dump(out_index, f, indent=2)

        # Free memory
        del egpt_tensor, vl_tensor, out_chunk_data, new_egpt_list, new_vl_list
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n=== Done ===")
    print(f"  Total samples: {total_processed}")
    print(f"  Total errors: {total_errors}")
    print(f"  Output: {chunked_out}")


if __name__ == "__main__":
    main()
