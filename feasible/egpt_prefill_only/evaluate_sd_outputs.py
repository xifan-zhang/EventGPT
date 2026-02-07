#!/usr/bin/env python3
"""
Evaluate Speculative Decoding Outputs

Records and compares:
1. Original Video-LLaVA outputs (baseline)
2. Accelerated outputs with accept/reject from EventGPT

Uses TokenAdapter checkpoint to convert EGPT tokens → VL tokens for verification.
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
from typing import List, Optional

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Configuration
DATASET_DIR = './data/my_egpt_dsec_test/my_egpt_dsec_seq_1s'
DEFAULT_ADAPTER_PATH = "./feasible/token_alignment/task/starred/10q_20260129_204744/best_model.pt"


@dataclass
class SDEvalResult:
    """Result from a single SD evaluation."""
    sample_idx: int
    question: str

    # Original VL output (baseline)
    vl_output_text: str
    vl_output_tokens: List[int]
    vl_time_ms: float

    # EventGPT draft
    egpt_draft_text: str
    egpt_draft_tokens: List[int]

    # Adapted draft (after TokenAdapter)
    adapted_draft_text: str
    adapted_draft_tokens: List[int]

    # Accelerated output (with SD verification)
    sd_output_text: str
    sd_output_tokens: List[int]
    sd_time_ms: float

    # Acceptance details
    accepted_tokens: int
    rejected_at: int
    acceptance_rate: float

    # Timing breakdown
    egpt_prefill_ms: float
    egpt_draft_ms: float
    vl_prefill_ms: float
    vl_verify_ms: float
    vl_decode_ms: float

    # Match analysis
    output_matches: bool  # Does SD output match VL output?
    speedup: float


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


class SDEvaluator:
    """Evaluator for Speculative Decoding with TokenAdapter."""

    def __init__(self, adapter_path: str = DEFAULT_ADAPTER_PATH, use_4bit: bool = True):
        self.adapter_path = adapter_path
        self.use_4bit = use_4bit
        self.device = 'cuda'

        # Models
        self._egpt_model = None
        self._egpt_tokenizer = None
        self._egpt_processor = None
        self._vl_model = None
        self._vl_processor = None
        self._token_adapter = None

    def _load_models(self):
        """Load all models."""
        if self._egpt_model is not None:
            return

        from transformers import AutoTokenizer, BitsAndBytesConfig
        from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
        from model.EventChatModel import EventChatModel

        bnb_config = None
        if self.use_4bit:
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

        # Load TokenAdapter
        print(f"Loading TokenAdapter from {self.adapter_path}...")
        self._load_adapter()
        print("All models loaded.")

    def _load_adapter(self):
        """Load TokenAdapter model."""
        sys.path.insert(0, str(ROOT / "feasible" / "token_alignment"))
        from train_and_evaluate import TokenAdapter

        checkpoint = torch.load(self.adapter_path, map_location=self.device, weights_only=False)

        self._token_adapter = TokenAdapter(
            vocab_size=32010,
            embed_dim=512,
            num_layers=4,
            num_heads=8,
            max_seq_len=128,
        )
        self._token_adapter.load_state_dict(checkpoint['model_state_dict'])
        self._token_adapter = self._token_adapter.to(self.device)
        self._token_adapter.eval()

    def _adapt_tokens(self, egpt_tokens: List[int]) -> List[int]:
        """Convert EGPT tokens to VL tokens using TokenAdapter."""
        with torch.inference_mode():
            tokens_tensor = torch.tensor([egpt_tokens], dtype=torch.long, device=self.device)

            # Pad if needed
            if tokens_tensor.shape[1] < 128:
                pad_len = 128 - tokens_tensor.shape[1]
                tokens_tensor = torch.nn.functional.pad(tokens_tensor, (0, pad_len))

            # Forward pass
            logits = self._token_adapter(tokens_tensor[:, :-1])

            # Get argmax predictions
            predicted = torch.argmax(logits, dim=-1).squeeze(0).tolist()

            return predicted[:len(egpt_tokens)]

    def _run_egpt_prefill_and_draft(self, event_image_path: str, query: str, max_draft: int = 30):
        """Run EventGPT prefill and generate draft tokens."""
        from model.EventChatModel import get_spatio_temporal_features
        from dataset.conversation import prepare_event_prompt
        from dataset.constants import EVENT_TOKEN_INDEX
        from common.common import tokenizer_event_token, load_image

        # Load image
        img = load_image(event_image_path)
        img_array = np.array(img)
        event_image_size = list(img_array.shape[:2])
        event = self._egpt_processor(img_array, return_tensors='pt')['pixel_values'][0]
        event = event.to(self.device, dtype=torch.bfloat16)

        # Prepare prompt
        prompt = prepare_event_prompt(query, 'eventgpt_v1')
        input_ids = tokenizer_event_token(prompt, self._egpt_tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        # Prefill
        torch.cuda.synchronize()
        prefill_start = time.time()

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

        torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start

        # Draft generation
        torch.cuda.synchronize()
        draft_start = time.time()
        draft_tokens = []

        with torch.inference_mode():
            cur_pos = inputs_embeds.shape[1]
            kv_cache = outputs.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens.append(next_token.item())

            for _ in range(max_draft - 1):
                cur_embed = self._egpt_model.get_model().embed_tokens(next_token)
                new_attn = torch.ones((1, cur_pos + 1), dtype=torch.bool, device=self.device)
                outputs = self._egpt_model.model(inputs_embeds=cur_embed, attention_mask=new_attn,
                                                 position_ids=torch.tensor([[cur_pos]], device=self.device),
                                                 past_key_values=kv_cache, use_cache=True)
                logits = self._egpt_model.lm_head(outputs.last_hidden_state[:, -1:, :])
                kv_cache = outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_token.item())
                cur_pos += 1
                if next_token.item() == self._egpt_tokenizer.eos_token_id:
                    break

        torch.cuda.synchronize()
        draft_time = time.time() - draft_start

        draft_text = self._egpt_tokenizer.decode(draft_tokens, skip_special_tokens=True)

        return {
            'prefill_time': prefill_time,
            'draft_time': draft_time,
            'draft_tokens': draft_tokens,
            'draft_text': draft_text,
        }

    def _run_vl_baseline(self, video_frames: List, query: str, max_tokens: int = 50):
        """Run Video-LLaVA baseline (no SD)."""
        torch.cuda.synchronize()
        start = time.time()

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

        torch.cuda.synchronize()
        total_time = time.time() - start

        output_ids = generated[0, input_ids.shape[1]:].tolist()
        output_text = self._vl_processor.tokenizer.decode(output_ids, skip_special_tokens=True)

        return {
            'time': total_time,
            'tokens': output_ids,
            'text': output_text,
        }

    def _run_vl_with_sd(self, video_frames: List, query: str, adapted_tokens: List[int], max_tokens: int = 50):
        """Run Video-LLaVA with speculative decoding verification."""
        prompt = f'USER: <video>\n{query}\nASSISTANT:'
        inputs = self._vl_processor(text=prompt, videos=video_frames, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        pixel_values_videos = inputs.get('pixel_values_videos').to(self.device)

        # === Prefill ===
        torch.cuda.synchronize()
        prefill_start = time.time()

        with torch.inference_mode():
            outputs = self._vl_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                past_key_values=None,
                use_cache=True,
            )

        torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start

        prefill_logits = outputs.logits
        past_kv = outputs.past_key_values

        # === Batch Verification ===
        torch.cuda.synchronize()
        verify_start = time.time()

        accepted_tokens = []
        rejected_at = -1

        with torch.inference_mode():
            # Send ALL adapted tokens to VL for verification
            draft_tensor = torch.tensor([adapted_tokens], dtype=torch.long, device=self.device)

            attn_dtype = attention_mask.dtype if attention_mask is not None else torch.long
            kv_len = past_kv[0][0].shape[2]
            extended_attn = torch.ones((1, kv_len + len(adapted_tokens)), dtype=attn_dtype, device=self.device)

            # Single forward pass to verify ALL draft tokens
            verify_outputs = self._vl_model(
                input_ids=draft_tensor,
                attention_mask=extended_attn,
                past_key_values=past_kv,
                use_cache=True,
            )

            verify_logits = verify_outputs.logits  # [1, num_drafts, vocab]
            verify_kv = verify_outputs.past_key_values

            # Check acceptance: compare adapted tokens with VL's predictions
            # First token: compare with prefill logits
            first_pred = torch.argmax(prefill_logits[:, -1, :], dim=-1).item()
            if adapted_tokens[0] == first_pred:
                accepted_tokens.append(adapted_tokens[0])

                # Remaining tokens: compare with batch verification logits
                for i in range(1, len(adapted_tokens)):
                    vl_pred = torch.argmax(verify_logits[:, i-1, :], dim=-1).item()
                    if adapted_tokens[i] == vl_pred:
                        accepted_tokens.append(adapted_tokens[i])
                    else:
                        rejected_at = i
                        break
            else:
                rejected_at = 0

        torch.cuda.synchronize()
        verify_time = time.time() - verify_start

        # === Continue AR Decode for remaining tokens ===
        torch.cuda.synchronize()
        decode_start = time.time()

        remaining_tokens = []
        num_accepted = len(accepted_tokens)
        remaining_count = max_tokens - num_accepted

        if remaining_count > 0:
            with torch.inference_mode():
                if num_accepted > 0:
                    # Some tokens accepted - use verify KV cache up to accepted position
                    # Use the prediction at the last accepted position for next token
                    if num_accepted < len(adapted_tokens):
                        next_token_logits = verify_logits[:, num_accepted - 1, :]
                    else:
                        next_token_logits = verify_logits[:, -1, :]
                    cur_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    cur_kv = verify_kv
                    cur_attn_len = kv_len + len(adapted_tokens) + 1
                else:
                    # NOTHING accepted - use PREFILL KV cache, NOT verify KV cache
                    # The verification corrupted the KV cache with wrong tokens
                    cur_token = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True)
                    cur_kv = past_kv  # Use original prefill KV cache!
                    cur_attn_len = kv_len + 1

                for _ in range(remaining_count):
                    cur_attn = torch.ones((1, cur_attn_len), dtype=attn_dtype, device=self.device)
                    outputs = self._vl_model(
                        input_ids=cur_token,
                        attention_mask=cur_attn,
                        past_key_values=cur_kv,
                        use_cache=True,
                    )
                    cur_kv = outputs.past_key_values
                    cur_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                    remaining_tokens.append(cur_token.item())
                    cur_attn_len += 1

                    if cur_token.item() == self._vl_processor.tokenizer.eos_token_id:
                        break

        torch.cuda.synchronize()
        decode_time = time.time() - decode_start

        # Combine output
        all_output_tokens = accepted_tokens + remaining_tokens
        output_text = self._vl_processor.tokenizer.decode(all_output_tokens, skip_special_tokens=True)

        return {
            'prefill_time': prefill_time,
            'verify_time': verify_time,
            'decode_time': decode_time,
            'total_time': prefill_time + verify_time + decode_time,
            'tokens': all_output_tokens,
            'text': output_text,
            'accepted_tokens': accepted_tokens,
            'num_accepted': num_accepted,
            'rejected_at': rejected_at,
        }

    def evaluate_sample(self, event_image_path: str, video_frames: List, query: str,
                       sample_idx: int, max_tokens: int = 50, max_draft: int = 30) -> SDEvalResult:
        """Evaluate a single sample."""
        self._load_models()

        # 1. Run EGPT prefill + draft
        egpt_result = self._run_egpt_prefill_and_draft(event_image_path, query, max_draft)

        # 2. Adapt EGPT tokens to VL tokens
        adapted_tokens = self._adapt_tokens(egpt_result['draft_tokens'])
        adapted_text = self._vl_processor.tokenizer.decode(adapted_tokens, skip_special_tokens=True)

        # 3. Run VL baseline
        vl_baseline = self._run_vl_baseline(video_frames, query, max_tokens)

        # 4. Run VL with SD verification
        sd_result = self._run_vl_with_sd(video_frames, query, adapted_tokens, max_tokens)

        # Calculate metrics
        acceptance_rate = sd_result['num_accepted'] / len(adapted_tokens) if adapted_tokens else 0
        speedup = vl_baseline['time'] / sd_result['total_time'] if sd_result['total_time'] > 0 else 1.0
        output_matches = vl_baseline['text'].strip() == sd_result['text'].strip()

        return SDEvalResult(
            sample_idx=sample_idx,
            question=query,
            vl_output_text=vl_baseline['text'],
            vl_output_tokens=vl_baseline['tokens'],
            vl_time_ms=vl_baseline['time'] * 1000,
            egpt_draft_text=egpt_result['draft_text'],
            egpt_draft_tokens=egpt_result['draft_tokens'],
            adapted_draft_text=adapted_text,
            adapted_draft_tokens=adapted_tokens,
            sd_output_text=sd_result['text'],
            sd_output_tokens=sd_result['tokens'],
            sd_time_ms=sd_result['total_time'] * 1000,
            accepted_tokens=sd_result['num_accepted'],
            rejected_at=sd_result['rejected_at'],
            acceptance_rate=acceptance_rate,
            egpt_prefill_ms=egpt_result['prefill_time'] * 1000,
            egpt_draft_ms=egpt_result['draft_time'] * 1000,
            vl_prefill_ms=sd_result['prefill_time'] * 1000,
            vl_verify_ms=sd_result['verify_time'] * 1000,
            vl_decode_ms=sd_result['decode_time'] * 1000,
            output_matches=output_matches,
            speedup=speedup,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=10, help='-1 for all samples')
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--max_draft', type=int, default=30)
    parser.add_argument('--adapter_path', type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument('--output_dir', type=str, default='./feasible/token_alignment/task/starred')
    parser.add_argument('--num_questions', type=int, default=1, choices=[1, 10], help='top1 or top10 questions')
    args = parser.parse_args()

    print('='*70)
    print('SPECULATIVE DECODING EVALUATION')
    print(f'Adapter: {args.adapter_path}')
    print(f'Questions: top{args.num_questions}')
    print('='*70)

    # Load dataset
    with open(os.path.join(DATASET_DIR, 'EventGPT_Instruction_Subset.json')) as f:
        dataset = json.load(f)
        if args.max_samples > 0:
            dataset = dataset[:args.max_samples]

    # Load questions
    with open('./feasible/token_alignment/top50_questions.json') as f:
        questions = [q['question'] for q in json.load(f)[:args.num_questions]]

    print(f'\nSamples: {len(dataset)}')
    print(f'Questions: {len(questions)}')
    print(f'Total evaluations: {len(dataset) * len(questions)}')

    # Initialize evaluator
    evaluator = SDEvaluator(adapter_path=args.adapter_path)

    results = []
    total_evals = len(dataset) * len(questions)

    pbar = tqdm(total=total_evals, desc='Evaluating')
    for idx, sample in enumerate(dataset):
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
                result = evaluator.evaluate_sample(
                    event_image_path=img_path,
                    video_frames=frames,
                    query=query,
                    sample_idx=idx,
                    max_tokens=args.max_tokens,
                    max_draft=args.max_draft,
                )
                results.append(result)

            except Exception as e:
                pass

            pbar.update(1)

    pbar.close()

    if not results:
        print("No results collected!")
        return

    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)

    avg = lambda k: np.mean([getattr(r, k) for r in results])

    print(f'\nSamples: {len(results)}')

    print(f'\n{"─"*50}')
    print('TIMING (ms):')
    print(f'  EGPT prefill:     {avg("egpt_prefill_ms"):>8.1f}')
    print(f'  EGPT draft:       {avg("egpt_draft_ms"):>8.1f}')
    print(f'  VL baseline:      {avg("vl_time_ms"):>8.1f}')
    print(f'  VL prefill:       {avg("vl_prefill_ms"):>8.1f}')
    print(f'  VL verify:        {avg("vl_verify_ms"):>8.1f}')
    print(f'  VL decode:        {avg("vl_decode_ms"):>8.1f}')
    print(f'  SD total:         {avg("sd_time_ms"):>8.1f}')

    print(f'\n{"─"*50}')
    print('ACCEPTANCE:')
    print(f'  Draft tokens:     {avg("accepted_tokens") + (30 - avg("accepted_tokens")):.1f}')
    print(f'  Accepted:         {avg("accepted_tokens"):.1f}')
    print(f'  Rate:             {avg("acceptance_rate")*100:.1f}%')

    print(f'\n{"─"*50}')
    print('SPEEDUP:')
    print(f'  Average:          {avg("speedup"):.2f}x')
    print(f'  Output matches:   {sum(1 for r in results if r.output_matches)}/{len(results)}')

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f'sd_eval_{timestamp}.json')

    with open(output_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'summary': {
                'n_samples': len(results),
                'avg_acceptance_rate': avg('acceptance_rate'),
                'avg_speedup': avg('speedup'),
                'output_match_rate': sum(1 for r in results if r.output_matches) / len(results),
            },
            'results': [asdict(r) for r in results],
        }, f, indent=2)

    print(f'\n{"─"*50}')
    print(f'Results saved to: {output_file}')

    # Print detailed comparison
    print('\n' + '='*70)
    print('DETAILED OUTPUT COMPARISON')
    print('='*70)

    for r in results[:5]:  # Show first 5
        print(f'\n--- Sample {r.sample_idx} ---')
        print(f'Q: {r.question[:60]}...')
        print(f'\nEGPT Draft ({len(r.egpt_draft_tokens)} tokens):')
        print(f'  "{r.egpt_draft_text[:100]}..."')
        print(f'\nAdapted Draft ({len(r.adapted_draft_tokens)} tokens):')
        print(f'  "{r.adapted_draft_text[:100]}..."')
        print(f'\nVL Baseline ({len(r.vl_output_tokens)} tokens, {r.vl_time_ms:.0f}ms):')
        print(f'  "{r.vl_output_text[:100]}..."')
        print(f'\nSD Output ({len(r.sd_output_tokens)} tokens, {r.sd_time_ms:.0f}ms, {r.accepted_tokens} accepted):')
        print(f'  "{r.sd_output_text[:100]}..."')
        print(f'\nMatch: {r.output_matches}, Speedup: {r.speedup:.2f}x')

    print('='*70)


if __name__ == '__main__':
    main()
