#!/usr/bin/env python3
"""
Prefill-Then-Verify: Parallel Prefill + Speculative Decoding Verification

This implements the optimal speculative decoding strategy:
1. Run EventGPT and Video-LLaVA prefill in PARALLEL
2. EventGPT finishes prefill first (~83ms) and continues generating draft tokens
3. By the time Video-LLaVA finishes prefill (~315ms), EventGPT has ~26 FREE draft tokens
4. Video-LLaVA VERIFIES these drafts in a BATCH (single forward pass)
5. Accepted tokens are FREE - they save decode time

Key insight: The 26 draft tokens cost ZERO time because they're generated
during Video-LLaVA's prefill phase!

Timing breakdown:
  - EGPT prefill: 83ms
  - VL prefill: 315ms
  - Overlap window: 232ms (EGPT generates drafts for FREE here)
  - Free draft tokens: ~26 tokens (at 9ms/token during prefill overlap)
  - VL verification: ~50ms (single forward pass for 26 tokens)
  - Accepted tokens: ~7 tokens (at 27.9% acceptance rate)
  - Time saved: 7 tokens × 14.5ms/token = ~100ms
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, asdict
import threading
import queue

# Fix protobuf compatibility
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Default trained TokenAdapter (27.9% acceptance, 51.6% top-5)
DEFAULT_ADAPTER_PATH = "./feasible/token_alignment/task/starred/1q_20260128_151847/best_model.pt"


class TokenAdapterForSD:
    """TokenAdapter wrapper for speculative decoding.

    Converts EventGPT tokens to Video-LLaVA token predictions.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        # Import TokenAdapter
        ROOT = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(ROOT / "feasible" / "token_alignment"))
        from train_and_evaluate import TokenAdapter

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self._model = TokenAdapter(
            vocab_size=32010,
            embed_dim=512,
            num_layers=4,
            num_heads=8,
            max_seq_len=128,
        )
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model = self._model.to(self.device)
        self._model.eval()

    def predict(self, egpt_tokens: List[int]) -> List[int]:
        """Convert EGPT tokens to VL token predictions."""
        self._load_model()

        with torch.inference_mode():
            # Prepare input - add padding to match training
            tokens_tensor = torch.tensor([egpt_tokens], dtype=torch.long, device=self.device)

            # Pad if needed
            if tokens_tensor.shape[1] < 128:
                pad_len = 128 - tokens_tensor.shape[1]
                tokens_tensor = torch.nn.functional.pad(tokens_tensor, (0, pad_len))

            # Forward pass - predict next tokens from input sequence
            logits = self._model(tokens_tensor[:, :-1])

            # Get argmax predictions
            predicted_tokens = torch.argmax(logits, dim=-1).squeeze(0).tolist()

            # Return only the valid positions
            return predicted_tokens[:len(egpt_tokens)]


@dataclass
class VerifyResult:
    """Result from speculative verification."""
    accepted_tokens: List[int]
    rejected_at: int  # Position where first rejection occurred (-1 if all accepted)
    acceptance_rate: float
    verification_time: float


@dataclass
class PrefillThenVerifyResult:
    """Complete result from prefill-then-verify inference."""
    # Final output
    output_text: str
    output_tokens: List[int]
    num_output_tokens: int

    # Draft info
    draft_tokens: List[int]
    num_draft_tokens: int
    free_draft_tokens: int  # Tokens generated during VL prefill (cost=0)

    # Verification
    accepted_tokens: int
    rejected_at: int
    acceptance_rate: float

    # Timing breakdown
    egpt_prefill_time: float
    egpt_draft_time: float  # Time to generate drafts (hidden in VL prefill)
    vl_prefill_time: float
    vl_verify_time: float   # Time to verify drafts in batch
    vl_decode_time: float   # Time for remaining AR decode
    wall_clock_time: float

    # Analysis
    overlap_window: float
    time_saved: float       # Time saved by accepted tokens
    effective_speedup: float


class PrefillThenVerifyInference:
    """
    Parallel Prefill + Speculative Decoding Verification.

    Strategy:
    1. Start EGPT and VL prefill in parallel
    2. EGPT finishes first, generates draft tokens while VL prefills
    3. VL finishes prefill, verifies all drafts in one batch
    4. Accept/reject tokens, continue AR decode for remaining
    """

    def __init__(
        self,
        eventgpt_path: str = "./checkpoints/EventGPT-7b",
        videollava_model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
        device: str = "cuda",
        use_4bit: bool = True,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
        use_adapter: bool = True,
    ):
        self.device = device
        self.use_4bit = use_4bit
        self.eventgpt_path = eventgpt_path
        self.videollava_model_id = videollava_model_id
        self.adapter_path = adapter_path
        self.use_adapter = use_adapter

        # Models
        self._eventgpt_model = None
        self._eventgpt_tokenizer = None
        self._eventgpt_processor = None
        self._videollava_model = None
        self._videollava_processor = None
        self._token_adapter = None

        # Thread-safety
        self._egpt_lock = threading.Lock()
        self._vl_lock = threading.Lock()

        # Communication between threads
        self._draft_queue = queue.Queue()

    def _load_models(self):
        """Load both models with 4-bit quantization."""
        if self._eventgpt_model is not None and self._videollava_model is not None:
            return

        from transformers import (
            AutoTokenizer,
            BitsAndBytesConfig,
            VideoLlavaForConditionalGeneration,
            VideoLlavaProcessor,
        )
        from model.EventChatModel import EventChatModel

        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        print("Loading EventGPT...")
        self._eventgpt_model = EventChatModel.from_pretrained(
            self.eventgpt_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )
        self._eventgpt_model.eval()
        self._eventgpt_tokenizer = AutoTokenizer.from_pretrained(
            self.eventgpt_path, use_fast=True
        )
        self._eventgpt_processor = self._eventgpt_model.get_visual_tower().event_processor

        print("Loading Video-LLaVA...")
        self._videollava_model = VideoLlavaForConditionalGeneration.from_pretrained(
            self.videollava_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
        )
        self._videollava_model.eval()
        self._videollava_processor = VideoLlavaProcessor.from_pretrained(
            self.videollava_model_id
        )

        # Load TokenAdapter for EGPT→VL token conversion
        if self.use_adapter and self.adapter_path:
            adapter_path = Path(self.adapter_path)
            if adapter_path.exists():
                print(f"Loading TokenAdapter from {self.adapter_path}...")
                self._token_adapter = TokenAdapterForSD(self.adapter_path, self.device)
                self._token_adapter._load_model()
            else:
                print(f"Warning: TokenAdapter not found at {self.adapter_path}")
                self.use_adapter = False

        print("All models loaded.")

    def _eventgpt_prefill_and_draft(
        self,
        event_image_path: str,
        query: str,
        max_draft_tokens: int,
        result_dict: Dict,
        stop_event: threading.Event,
    ):
        """
        EventGPT: Prefill then generate draft tokens.

        Runs in parallel with VL prefill. Generates draft tokens until:
        1. VL prefill completes (signaled by stop_event)
        2. max_draft_tokens reached
        3. EOS token generated
        """
        from model.EventChatModel import get_spatio_temporal_features
        from dataset.conversation import prepare_event_prompt
        from dataset.constants import EVENT_TOKEN_INDEX
        from common.common import tokenizer_event_token, load_image

        with self._egpt_lock:
            try:
                # === PREFILL ===
                torch.cuda.synchronize()
                prefill_start = time.time()

                img = load_image(event_image_path)
                img_array = np.array(img)
                event_image_size = list(img_array.shape[:2])

                event = self._eventgpt_processor(img_array, return_tensors='pt')['pixel_values'][0]
                event = event.to(self.device, dtype=torch.bfloat16)

                conv_mode = 'eventgpt_v1'
                prompt = prepare_event_prompt(query, conv_mode)
                input_ids = tokenizer_event_token(
                    prompt, self._eventgpt_tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
                ).unsqueeze(0).to(self.device)

                with torch.inference_mode():
                    feature = self._eventgpt_model.visval_encode(event.unsqueeze(0))
                    feature = self._eventgpt_model.get_model().feature_adaptor(feature)
                    feature = feature.squeeze(0)
                    event_features = get_spatio_temporal_features([feature])
                    event_features = event_features.unsqueeze(0)

                    (
                        _,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        inputs_embeds,
                        _
                    ) = self._eventgpt_model.prepare_inputs_labels_for_multimodal(
                        input_ids,
                        None,
                        torch.ones_like(input_ids, dtype=torch.bool),
                        None,
                        None,
                        event_tensors=None,
                        event_image_sizes=event_image_size,
                        event_features=event_features,
                    )

                    if attention_mask is None:
                        attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.bool, device=self.device)
                    if position_ids is None:
                        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)

                    outputs = self._eventgpt_model.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=True,
                    )

                    logits = self._eventgpt_model.lm_head(outputs.last_hidden_state[:, -1:, :])

                torch.cuda.synchronize()
                prefill_time = time.time() - prefill_start
                result_dict['egpt_prefill_time'] = prefill_time
                result_dict['egpt_prefill_complete'] = time.time()

                # === DRAFT GENERATION ===
                # Generate drafts until VL prefill completes or max reached
                torch.cuda.synchronize()
                draft_start = time.time()
                draft_tokens = []

                with torch.inference_mode():
                    cur_pos = inputs_embeds.shape[1]
                    kv_cache = outputs.past_key_values
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    draft_tokens.append(next_token.item())

                    for i in range(max_draft_tokens - 1):
                        # Check if VL prefill is done
                        if stop_event.is_set():
                            break

                        cur_embed = self._eventgpt_model.get_model().embed_tokens(next_token)
                        new_attention_mask = torch.ones((1, cur_pos + 1), dtype=torch.bool, device=self.device)

                        outputs = self._eventgpt_model.model(
                            inputs_embeds=cur_embed,
                            attention_mask=new_attention_mask,
                            position_ids=torch.tensor([[cur_pos]], device=self.device),
                            past_key_values=kv_cache,
                            use_cache=True,
                        )

                        logits = self._eventgpt_model.lm_head(outputs.last_hidden_state[:, -1:, :])
                        kv_cache = outputs.past_key_values
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                        token_id = next_token.item()
                        draft_tokens.append(token_id)
                        cur_pos += 1

                        if token_id == self._eventgpt_tokenizer.eos_token_id:
                            break

                torch.cuda.synchronize()
                draft_time = time.time() - draft_start

                result_dict['egpt_draft_time'] = draft_time
                result_dict['egpt_draft_tokens'] = draft_tokens
                result_dict['egpt_success'] = True

            except Exception as e:
                import traceback
                result_dict['egpt_error'] = str(e)
                result_dict['egpt_traceback'] = traceback.format_exc()[-500:]
                result_dict['egpt_success'] = False

    def _videollava_prefill(
        self,
        video_frames: List,
        query: str,
        result_dict: Dict,
        prefill_done_event: threading.Event,
    ):
        """
        Video-LLaVA: Prefill only. Signals when done so EGPT stops drafting.
        """
        with self._vl_lock:
            try:
                torch.cuda.synchronize()
                prefill_start = time.time()

                prompt = f"USER: <video>\n{query}\nASSISTANT:"
                inputs = self._videollava_processor(
                    text=prompt, videos=video_frames, return_tensors="pt"
                )

                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs.get('attention_mask')
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.to(self.device)

                pixel_values_videos = inputs.get('pixel_values_videos')

                with torch.inference_mode():
                    outputs = self._videollava_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos.to(self.device),
                        past_key_values=None,
                        use_cache=True,
                    )

                torch.cuda.synchronize()
                prefill_time = time.time() - prefill_start

                # Signal that prefill is done - EGPT should stop drafting
                prefill_done_event.set()

                result_dict['vl_prefill_time'] = prefill_time
                result_dict['vl_prefill_complete'] = time.time()
                result_dict['vl_past_key_values'] = outputs.past_key_values
                result_dict['vl_logits'] = outputs.logits
                result_dict['vl_attention_mask'] = attention_mask
                result_dict['vl_success'] = True

            except Exception as e:
                import traceback
                prefill_done_event.set()  # Signal anyway to unblock EGPT
                result_dict['vl_error'] = str(e)
                result_dict['vl_traceback'] = traceback.format_exc()[-500:]
                result_dict['vl_success'] = False

    def _verify_drafts_batch(
        self,
        draft_tokens: List[int],
        past_key_values,
        attention_mask,
        first_logits,
    ) -> Tuple[List[int], int, float, List[int]]:
        """
        Verify draft tokens in a SINGLE batch forward pass.

        This is the key to speculative decoding efficiency:
        - One forward pass verifies ALL draft tokens
        - Much faster than sequential verification

        If TokenAdapter is enabled:
        - Convert EGPT tokens → VL token predictions first
        - Then verify VL predictions against VL's actual outputs

        Returns:
            (accepted_tokens, rejected_at_position, verification_time, adapted_tokens)
        """
        torch.cuda.synchronize()
        verify_start = time.time()

        accepted_tokens = []
        rejected_at = -1

        # === Convert EGPT tokens → VL tokens using TokenAdapter ===
        if self._token_adapter is not None:
            # Use adapter to predict what VL would generate from EGPT drafts
            adapted_tokens = self._token_adapter.predict(draft_tokens)
        else:
            # No adapter - use raw EGPT tokens (will likely have low acceptance)
            adapted_tokens = draft_tokens

        with torch.inference_mode():
            # Prepare adapted tokens as input for verification
            draft_tensor = torch.tensor([adapted_tokens], dtype=torch.long, device=self.device)

            # Get attention mask dtype
            attn_dtype = attention_mask.dtype if isinstance(attention_mask, torch.Tensor) else torch.long

            # Extend attention mask for draft tokens
            kv_len = past_key_values[0][0].shape[2]
            extended_attention_mask = torch.ones(
                (1, kv_len + len(adapted_tokens)),
                dtype=attn_dtype,
                device=self.device
            )

            # Single forward pass to get logits for ALL draft positions
            outputs = self._videollava_model(
                input_ids=draft_tensor,
                attention_mask=extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Get VL's predictions for each position
            batch_logits = outputs.logits  # [1, num_drafts, vocab]

            # Compare VL's predictions with adapted draft tokens
            # For position i, VL predicts token i+1
            # Adapted token at position i should match VL's prediction at position i-1

            # First token: compare with first_logits (from prefill)
            first_pred = torch.argmax(first_logits[:, -1, :], dim=-1).item()
            if adapted_tokens[0] == first_pred:
                accepted_tokens.append(adapted_tokens[0])
            else:
                rejected_at = 0
                torch.cuda.synchronize()
                verify_time = time.time() - verify_start
                return accepted_tokens, rejected_at, verify_time, adapted_tokens

            # Remaining tokens: compare with batch predictions
            for i in range(1, len(adapted_tokens)):
                vl_pred = torch.argmax(batch_logits[:, i-1, :], dim=-1).item()
                if adapted_tokens[i] == vl_pred:
                    accepted_tokens.append(adapted_tokens[i])
                else:
                    rejected_at = i
                    break

        torch.cuda.synchronize()
        verify_time = time.time() - verify_start

        return accepted_tokens, rejected_at, verify_time, adapted_tokens

    def _continue_ar_decode(
        self,
        past_key_values,
        attention_mask,
        last_token: int,
        num_accepted: int,
        max_new_tokens: int,
    ) -> Tuple[List[int], float]:
        """Continue AR decode after verification."""
        torch.cuda.synchronize()
        decode_start = time.time()

        remaining_tokens = max_new_tokens - num_accepted
        if remaining_tokens <= 0:
            return [], 0.0

        output_tokens = []
        attn_dtype = attention_mask.dtype if isinstance(attention_mask, torch.Tensor) else torch.long

        with torch.inference_mode():
            cur_token = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
            kv_len = past_key_values[0][0].shape[2]
            cur_attention_mask = torch.ones((1, kv_len + 1), dtype=attn_dtype, device=self.device)

            for _ in range(remaining_tokens):
                outputs = self._videollava_model(
                    input_ids=cur_token,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits = outputs.logits
                past_key_values = outputs.past_key_values

                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                token_id = next_token.item()
                output_tokens.append(token_id)

                cur_token = next_token
                cur_attention_mask = torch.ones(
                    (1, cur_attention_mask.shape[1] + 1),
                    dtype=attn_dtype, device=self.device
                )

                if token_id == self._videollava_processor.tokenizer.eos_token_id:
                    break

        torch.cuda.synchronize()
        decode_time = time.time() - decode_start

        return output_tokens, decode_time

    def generate(
        self,
        event_image_path: str,
        video_frames: List,
        query: str,
        max_new_tokens: int = 50,
        max_draft_tokens: int = 30,
    ) -> PrefillThenVerifyResult:
        """
        Run parallel prefill + speculative verification.

        Flow:
        1. Start EGPT prefill + VL prefill in parallel
        2. EGPT generates draft tokens while VL prefills
        3. VL verifies drafts in batch
        4. Accept/reject, continue AR decode

        Args:
            event_image_path: Path to event image
            video_frames: List of PIL images for Video-LLaVA
            query: Text query
            max_new_tokens: Maximum total tokens to generate
            max_draft_tokens: Maximum draft tokens to generate

        Returns:
            PrefillThenVerifyResult with full breakdown
        """
        self._load_models()

        result_dict = {}
        prefill_done_event = threading.Event()

        wall_clock_start = time.time()

        # === PHASE 1: Parallel Prefill ===
        # EGPT prefills then drafts; VL prefills
        with ThreadPoolExecutor(max_workers=2) as executor:
            egpt_future = executor.submit(
                self._eventgpt_prefill_and_draft,
                event_image_path,
                query,
                max_draft_tokens,
                result_dict,
                prefill_done_event,  # EGPT stops when VL signals done
            )

            vl_future = executor.submit(
                self._videollava_prefill,
                video_frames,
                query,
                result_dict,
                prefill_done_event,  # VL signals when prefill done
            )

            # Wait for both
            vl_future.result()  # VL prefill done
            egpt_future.result()  # EGPT drafts done

        if not result_dict.get('egpt_success') or not result_dict.get('vl_success'):
            # Error occurred
            error_msg = result_dict.get('egpt_error', '') + result_dict.get('vl_error', '')
            raise RuntimeError(f"Inference failed: {error_msg}")

        draft_tokens = result_dict.get('egpt_draft_tokens', [])
        egpt_prefill_time = result_dict.get('egpt_prefill_time', 0)
        egpt_draft_time = result_dict.get('egpt_draft_time', 0)
        vl_prefill_time = result_dict.get('vl_prefill_time', 0)

        # Calculate overlap window and free tokens
        overlap_window = max(0, vl_prefill_time - egpt_prefill_time)
        # Tokens generated during VL prefill are "free"
        free_draft_tokens = len(draft_tokens)  # All drafts are free since generated during VL prefill

        # === PHASE 2: Batch Verification ===
        accepted_tokens, rejected_at, verify_time, adapted_tokens = self._verify_drafts_batch(
            draft_tokens,
            result_dict['vl_past_key_values'],
            result_dict['vl_attention_mask'],
            result_dict['vl_logits'],
        )

        num_accepted = len(accepted_tokens)
        acceptance_rate = num_accepted / len(adapted_tokens) if adapted_tokens else 0

        # === PHASE 3: Continue AR Decode ===
        # Start from the last accepted token (or first VL token if none accepted)
        if num_accepted > 0:
            last_token = accepted_tokens[-1]
        else:
            # Use VL's first prediction
            last_token = torch.argmax(result_dict['vl_logits'][:, -1, :], dim=-1).item()
            accepted_tokens = [last_token]
            num_accepted = 1

        remaining_tokens, decode_time = self._continue_ar_decode(
            result_dict['vl_past_key_values'],
            result_dict['vl_attention_mask'],
            last_token,
            num_accepted,
            max_new_tokens,
        )

        # === Combine Results ===
        all_output_tokens = accepted_tokens + remaining_tokens
        output_text = self._videollava_processor.tokenizer.decode(
            all_output_tokens, skip_special_tokens=True
        )

        torch.cuda.synchronize()
        wall_clock_time = time.time() - wall_clock_start

        # Calculate time saved
        # Without SD: would need to generate num_accepted tokens via AR decode
        # With SD: got them for free via verification
        vl_token_time = 14.5 / 1000  # ~14.5ms per token (from benchmark data)
        time_saved = num_accepted * vl_token_time

        # Effective speedup
        baseline_time = vl_prefill_time + max_new_tokens * vl_token_time
        effective_speedup = baseline_time / wall_clock_time if wall_clock_time > 0 else 1.0

        return PrefillThenVerifyResult(
            output_text=output_text,
            output_tokens=all_output_tokens,
            num_output_tokens=len(all_output_tokens),
            draft_tokens=draft_tokens,
            num_draft_tokens=len(draft_tokens),
            free_draft_tokens=free_draft_tokens,
            accepted_tokens=num_accepted,
            rejected_at=rejected_at,
            acceptance_rate=acceptance_rate,
            egpt_prefill_time=egpt_prefill_time,
            egpt_draft_time=egpt_draft_time,
            vl_prefill_time=vl_prefill_time,
            vl_verify_time=verify_time,
            vl_decode_time=decode_time,
            wall_clock_time=wall_clock_time,
            overlap_window=overlap_window,
            time_saved=time_saved,
            effective_speedup=effective_speedup,
        )


def main():
    """Test prefill-then-verify inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Prefill-then-verify inference test")
    parser.add_argument("--event_image", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--query", type=str, default="What are the key elements in this scene?")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--max_draft_tokens", type=int, default=30)

    args = parser.parse_args()

    # Load video frames
    try:
        import av
        container = av.open(args.video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
            if len(frames) >= 8:
                break
        container.close()
    except Exception as e:
        print(f"Error loading video: {e}")
        return

    # Run inference
    inference = PrefillThenVerifyInference()
    result = inference.generate(
        event_image_path=args.event_image,
        video_frames=frames,
        query=args.query,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
    )

    print("\n" + "="*70)
    print("PREFILL-THEN-VERIFY RESULT")
    print("="*70)

    print(f"\nQuery: {args.query}")
    print(f"\nOutput: {result.output_text}")

    print(f"\n{'─'*70}")
    print("DRAFT GENERATION (FREE during VL prefill):")
    print(f"  Draft tokens generated: {result.num_draft_tokens}")
    print(f"  Free draft tokens:      {result.free_draft_tokens}")
    print(f"  Draft generation time:  {result.egpt_draft_time*1000:.1f} ms (HIDDEN in VL prefill)")

    print(f"\n{'─'*70}")
    print("VERIFICATION (single batch forward pass):")
    print(f"  Accepted tokens:        {result.accepted_tokens} / {result.num_draft_tokens}")
    print(f"  Acceptance rate:        {result.acceptance_rate*100:.1f}%")
    print(f"  Rejected at position:   {result.rejected_at if result.rejected_at >= 0 else 'N/A'}")
    print(f"  Verification time:      {result.vl_verify_time*1000:.1f} ms")

    print(f"\n{'─'*70}")
    print("TIMING BREAKDOWN:")
    print(f"  EGPT Prefill:           {result.egpt_prefill_time*1000:.1f} ms")
    print(f"  VL Prefill:             {result.vl_prefill_time*1000:.1f} ms")
    print(f"  Overlap Window:         {result.overlap_window*1000:.1f} ms")
    print(f"  VL Verification:        {result.vl_verify_time*1000:.1f} ms")
    print(f"  VL AR Decode:           {result.vl_decode_time*1000:.1f} ms")
    print(f"  Wall-Clock Total:       {result.wall_clock_time*1000:.1f} ms")

    print(f"\n{'─'*70}")
    print("SPEEDUP ANALYSIS:")
    print(f"  Time saved:             {result.time_saved*1000:.1f} ms")
    print(f"  Effective speedup:      {result.effective_speedup:.2f}x")

    print("="*70)


if __name__ == "__main__":
    main()
