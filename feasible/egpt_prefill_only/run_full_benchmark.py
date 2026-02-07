#!/usr/bin/env python3
"""
Full Benchmark: EGPT → Adapter → VL Token Comparison

Runs on full dataset with parallel execution.
Records output texts from:
1. EGPT Direct (what EventGPT generates)
2. Adapter Output (EGPT tokens transformed by TokenAdapter)
3. VL Actual (what Video-LLaVA generates)
"""

import sys
import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'feasible' / 'token_alignment'))

DATASET_DIR = './data/my_egpt_dsec_test/my_egpt_dsec_seq_1s'
ADAPTER_PATH = './feasible/token_alignment/task/starred/10q_20260129_204744/best_model.pt'


@dataclass
class BenchmarkResult:
    sample_idx: int
    question_idx: int
    question: str

    # EGPT Direct
    egpt_tokens: List[int]
    egpt_text: str

    # Adapter Output
    adapter_tokens: List[int]
    adapter_text: str

    # VL Actual
    vl_tokens: List[int]
    vl_text: str

    # Metrics
    consecutive_matches: int
    total_matches: int
    total_tokens: int
    acceptance_rate: float


def load_video_frames(video_path, num_frames=8):
    import av
    try:
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
            if len(frames) >= num_frames:
                break
        container.close()
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                return None
        return frames
    except:
        return None


class ParallelBenchmark:
    def __init__(self, adapter_path: str = ADAPTER_PATH):
        self.adapter_path = adapter_path
        self.device = 'cuda'

        # Models (loaded lazily)
        self._egpt_model = None
        self._egpt_tokenizer = None
        self._egpt_processor = None
        self._vl_model = None
        self._vl_processor = None
        self._adapter = None

        # Thread locks
        self._egpt_lock = threading.Lock()
        self._vl_lock = threading.Lock()

    def _load_models(self):
        if self._egpt_model is not None:
            return

        from transformers import AutoTokenizer, BitsAndBytesConfig
        from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
        from model.EventChatModel import EventChatModel
        from train_and_evaluate import TokenAdapter

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print("Loading EventGPT (4-bit)...")
        self._egpt_model = EventChatModel.from_pretrained(
            './checkpoints/EventGPT-7b',
            torch_dtype=torch.bfloat16,
            device_map='auto',
            quantization_config=bnb_config,
        )
        self._egpt_model.eval()
        self._egpt_tokenizer = AutoTokenizer.from_pretrained('./checkpoints/EventGPT-7b', use_fast=True)
        self._egpt_processor = self._egpt_model.get_visual_tower().event_processor
        print(f"  GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        print("Loading Video-LLaVA (4-bit)...")
        self._vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
            'LanguageBind/Video-LLaVA-7B-hf',
            torch_dtype=torch.float16,
            device_map='auto',
            quantization_config=bnb_config,
        )
        self._vl_model.eval()
        self._vl_processor = VideoLlavaProcessor.from_pretrained('LanguageBind/Video-LLaVA-7B-hf')
        print(f"  GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        print(f"Loading TokenAdapter...")
        checkpoint = torch.load(self.adapter_path, map_location=self.device, weights_only=False)
        self._adapter = TokenAdapter(vocab_size=32010, embed_dim=512, num_layers=4, num_heads=8, max_seq_len=128)
        self._adapter.load_state_dict(checkpoint['model_state_dict'])
        self._adapter = self._adapter.to(self.device)
        self._adapter.eval()

        print("All models loaded.")

    def _run_egpt(self, event_image_path: str, query: str, max_tokens: int = 50) -> Dict:
        """Generate tokens with EventGPT."""
        from model.EventChatModel import get_spatio_temporal_features
        from dataset.conversation import prepare_event_prompt
        from dataset.constants import EVENT_TOKEN_INDEX
        from common.common import tokenizer_event_token, load_image

        with self._egpt_lock:
            img = load_image(event_image_path)
            img_array = np.array(img)
            event_image_size = list(img_array.shape[:2])
            event = self._egpt_processor(img_array, return_tensors='pt')['pixel_values'][0]
            event = event.to(self.device, dtype=torch.bfloat16)

            prompt = prepare_event_prompt(query, 'eventgpt_v1')
            input_ids = tokenizer_event_token(prompt, self._egpt_tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

            with torch.inference_mode():
                feature = self._egpt_model.visval_encode(event.unsqueeze(0))
                feature = self._egpt_model.get_model().feature_adaptor(feature)
                feature = feature.squeeze(0)
                event_features = get_spatio_temporal_features([feature])
                event_features = event_features.unsqueeze(0)

                _, position_ids, attention_mask, _, inputs_embeds, _ = self._egpt_model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, torch.ones_like(input_ids, dtype=torch.bool), None, None,
                    event_tensors=None, event_image_sizes=event_image_size, event_features=event_features,
                )

                if attention_mask is None:
                    attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.bool, device=self.device)
                if position_ids is None:
                    position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)

                outputs = self._egpt_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                                 position_ids=position_ids, past_key_values=None, use_cache=True)
                logits = self._egpt_model.lm_head(outputs.last_hidden_state[:, -1:, :])

                # Decode
                output_tokens = []
                cur_pos = inputs_embeds.shape[1]
                kv_cache = outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                output_tokens.append(next_token.item())

                for _ in range(max_tokens - 1):
                    cur_embed = self._egpt_model.get_model().embed_tokens(next_token)
                    new_attn = torch.ones((1, cur_pos + 1), dtype=torch.bool, device=self.device)
                    outputs = self._egpt_model.model(inputs_embeds=cur_embed, attention_mask=new_attn,
                                                     position_ids=torch.tensor([[cur_pos]], device=self.device),
                                                     past_key_values=kv_cache, use_cache=True)
                    logits = self._egpt_model.lm_head(outputs.last_hidden_state[:, -1:, :])
                    kv_cache = outputs.past_key_values
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    output_tokens.append(next_token.item())
                    cur_pos += 1
                    if next_token.item() == self._egpt_tokenizer.eos_token_id:
                        break

            text = self._egpt_tokenizer.decode(output_tokens, skip_special_tokens=True)
            return {'tokens': output_tokens, 'text': text}

    def _run_vl(self, video_frames: List, query: str, max_tokens: int = 50) -> Dict:
        """Generate tokens with Video-LLaVA."""
        with self._vl_lock:
            prompt = f'USER: <video>\n{query}\nASSISTANT:'
            inputs = self._vl_processor(text=prompt, videos=video_frames, return_tensors='pt')
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            pixel_values_videos = inputs.get('pixel_values_videos').to(self.device)

            with torch.inference_mode():
                generated = self._vl_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )

            output_ids = generated[0, input_ids.shape[1]:].tolist()
            text = self._vl_processor.tokenizer.decode(output_ids, skip_special_tokens=True)
            return {'tokens': output_ids, 'text': text}

    def _adapt_tokens(self, egpt_tokens: List[int]) -> Dict:
        """Convert EGPT tokens to VL tokens using adapter."""
        with torch.inference_mode():
            tokens_tensor = torch.tensor([egpt_tokens], dtype=torch.long, device=self.device)

            if tokens_tensor.shape[1] < 128:
                pad_len = 128 - tokens_tensor.shape[1]
                tokens_tensor = torch.nn.functional.pad(tokens_tensor, (0, pad_len))

            logits = self._adapter(tokens_tensor[:, :-1])
            predicted = torch.argmax(logits, dim=-1).squeeze(0).tolist()[:len(egpt_tokens)]

        text = self._vl_processor.tokenizer.decode(predicted, skip_special_tokens=True)
        return {'tokens': predicted, 'text': text}

    def _compute_matches(self, adapter_tokens: List[int], vl_tokens: List[int]) -> Dict:
        """Compute consecutive and total matches."""
        min_len = min(len(adapter_tokens), len(vl_tokens))

        consecutive = 0
        for a, v in zip(adapter_tokens[:min_len], vl_tokens[:min_len]):
            if a == v:
                consecutive += 1
            else:
                break

        total = sum(1 for a, v in zip(adapter_tokens[:min_len], vl_tokens[:min_len]) if a == v)

        return {
            'consecutive': consecutive,
            'total': total,
            'length': min_len,
            'rate': total / min_len if min_len > 0 else 0,
        }

    def run_sample(self, event_image_path: str, video_frames: List, query: str,
                   sample_idx: int, question_idx: int, max_tokens: int = 50) -> BenchmarkResult:
        """Run benchmark on a single sample."""
        # Run EGPT and VL in parallel
        egpt_result = None
        vl_result = None

        def run_egpt():
            nonlocal egpt_result
            egpt_result = self._run_egpt(event_image_path, query, max_tokens)

        def run_vl():
            nonlocal vl_result
            vl_result = self._run_vl(video_frames, query, max_tokens)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(run_egpt)
            executor.submit(run_vl)
            executor.shutdown(wait=True)

        # Adapt EGPT tokens
        adapter_result = self._adapt_tokens(egpt_result['tokens'])

        # Compute matches
        matches = self._compute_matches(adapter_result['tokens'], vl_result['tokens'])

        return BenchmarkResult(
            sample_idx=sample_idx,
            question_idx=question_idx,
            question=query,
            egpt_tokens=egpt_result['tokens'],
            egpt_text=egpt_result['text'],
            adapter_tokens=adapter_result['tokens'],
            adapter_text=adapter_result['text'],
            vl_tokens=vl_result['tokens'],
            vl_text=vl_result['text'],
            consecutive_matches=matches['consecutive'],
            total_matches=matches['total'],
            total_tokens=matches['length'],
            acceptance_rate=matches['rate'],
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=-1, help='-1 for all')
    parser.add_argument('--num_questions', type=int, default=10, help='top N questions')
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./feasible/token_alignment/task/starred/10q_20260129_204744')
    args = parser.parse_args()

    print('='*70)
    print('FULL BENCHMARK: EGPT → Adapter → VL')
    print('='*70)

    # Load dataset
    with open(os.path.join(DATASET_DIR, 'EventGPT_Instruction_Subset.json')) as f:
        dataset = json.load(f)
        if args.max_samples > 0:
            dataset = dataset[:args.max_samples]

    # Load questions
    with open('./feasible/token_alignment/top50_questions.json') as f:
        questions = [q['question'] for q in json.load(f)[:args.num_questions]]

    total_evals = len(dataset) * len(questions)
    print(f'\nSamples: {len(dataset)}')
    print(f'Questions: {len(questions)}')
    print(f'Total evaluations: {total_evals}')

    # Initialize benchmark
    benchmark = ParallelBenchmark()
    benchmark._load_models()

    results = []

    pbar = tqdm(total=total_evals, desc='Benchmarking')
    for sample_idx, sample in enumerate(dataset):
        event_paths = sample.get('event_image', [])
        video_data = sample.get('video_data')

        if not event_paths or not video_data:
            pbar.update(len(questions))
            continue

        img_path = os.path.join(DATASET_DIR, 'event_image', event_paths[0])
        video_path = os.path.join(DATASET_DIR, 'mp4', video_data + '.mp4')

        frames = load_video_frames(video_path)
        if frames is None:
            pbar.update(len(questions))
            continue

        for q_idx, query in enumerate(questions):
            try:
                result = benchmark.run_sample(
                    event_image_path=img_path,
                    video_frames=frames,
                    query=query,
                    sample_idx=sample_idx,
                    question_idx=q_idx,
                    max_tokens=args.max_tokens,
                )
                results.append(result)
            except Exception as e:
                pass

            pbar.update(1)

    pbar.close()

    if not results:
        print("No results!")
        return

    # Compute summary
    avg_consecutive = np.mean([r.consecutive_matches for r in results])
    avg_total = np.mean([r.total_matches for r in results])
    avg_tokens = np.mean([r.total_tokens for r in results])
    avg_rate = np.mean([r.acceptance_rate for r in results])

    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'\nTotal samples: {len(results)}')
    print(f'\nAcceptance Metrics:')
    print(f'  Consecutive matches: {avg_consecutive:.1f} avg')
    print(f'  Total matches:       {avg_total:.1f} / {avg_tokens:.1f} avg')
    print(f'  Match rate:          {avg_rate*100:.1f}%')

    # Distribution of consecutive matches
    consec_dist = {}
    for r in results:
        c = r.consecutive_matches
        consec_dist[c] = consec_dist.get(c, 0) + 1

    print(f'\nConsecutive Match Distribution:')
    for k in sorted(consec_dist.keys())[:15]:
        pct = 100 * consec_dist[k] / len(results)
        print(f'  {k:>2} tokens: {consec_dist[k]:>5} ({pct:>5.1f}%)')

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f'full_benchmark_{timestamp}.json')

    output_data = {
        'timestamp': timestamp,
        'config': vars(args),
        'summary': {
            'total_samples': len(results),
            'avg_consecutive_matches': avg_consecutive,
            'avg_total_matches': avg_total,
            'avg_tokens': avg_tokens,
            'avg_acceptance_rate': avg_rate,
            'consecutive_distribution': consec_dist,
        },
        'results': [asdict(r) for r in results],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f'\nResults saved to: {output_file}')

    # Show examples
    print('\n' + '='*70)
    print('EXAMPLE OUTPUTS (first 5)')
    print('='*70)

    for r in results[:5]:
        print(f'\n--- Sample {r.sample_idx}, Q{r.question_idx} ---')
        print(f'Q: {r.question[:50]}...')
        print(f'\nEGPT:    "{r.egpt_text[:70]}..."')
        print(f'Adapter: "{r.adapter_text[:70]}..."')
        print(f'VL:      "{r.vl_text[:70]}..."')
        print(f'\nConsecutive: {r.consecutive_matches}, Total: {r.total_matches}/{r.total_tokens} ({r.acceptance_rate*100:.1f}%)')

    print('='*70)


if __name__ == '__main__':
    main()
