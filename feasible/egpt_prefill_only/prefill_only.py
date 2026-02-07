#!/usr/bin/env python3
"""
Prefill-Only Parallel Inference: EventGPT + Video-LLaVA

Exploits EventGPT's faster prefill (83ms vs 315ms) to get "free" draft tokens
during the parallel prefill window, then uses Video-LLaVA for AR decode.

Key benefits:
- Simpler than full speculative decoding
- No acceptance rate dependency
- Guaranteed prefill time savings
- Free EventGPT tokens as preview/priming
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import threading

# Fix protobuf compatibility
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class PrefillOnlyConfig:
    """Configuration for prefill-only inference."""
    eventgpt_path: str = "./checkpoints/EventGPT-7b"
    videollava_model_id: str = "LanguageBind/Video-LLaVA-7B-hf"
    device: str = "cuda"
    use_4bit: bool = True
    max_new_tokens: int = 50
    # Prefill-only specific settings
    generate_free_tokens: bool = True  # Generate EventGPT tokens during VL prefill
    use_draft_priming: bool = False    # Use drafts to prime VL decode (experimental)
    return_preview: bool = True        # Return EventGPT output as preview


@dataclass
class PrefillOnlyResult:
    """Result from prefill-only inference."""
    # Outputs
    output_text: str
    output_tokens: List[int]
    preview_text: Optional[str]  # EventGPT preview (available earlier)
    preview_tokens: Optional[List[int]]

    # Timing
    egpt_prefill_time: float
    egpt_decode_time: float
    vl_prefill_time: float
    vl_decode_time: float
    wall_clock_time: float

    # Analysis
    free_tokens_count: int
    overlap_window: float
    prefill_speedup: float
    total_speedup_vs_sequential: float


def compute_prefill_benefit(
    egpt_prefill_ms: float,
    vl_prefill_ms: float,
    egpt_decode_rate_ms: float = 18.0,  # ms per token
) -> Dict[str, float]:
    """
    Compute the benefit of parallel prefill.

    Args:
        egpt_prefill_ms: EventGPT prefill time in ms
        vl_prefill_ms: Video-LLaVA prefill time in ms
        egpt_decode_rate_ms: EventGPT decode rate in ms/token

    Returns:
        Dictionary with benefit analysis
    """
    overlap_window = max(0, vl_prefill_ms - egpt_prefill_ms)
    free_tokens = int(overlap_window / egpt_decode_rate_ms)

    return {
        "overlap_window_ms": overlap_window,
        "free_tokens": free_tokens,
        "prefill_speedup": vl_prefill_ms / max(egpt_prefill_ms, 1),
        "egpt_prefill_ms": egpt_prefill_ms,
        "vl_prefill_ms": vl_prefill_ms,
    }


class PrefillOnlyInference:
    """
    Parallel prefill inference with EventGPT and Video-LLaVA.

    Exploits EventGPT's faster prefill to generate "free" draft tokens
    during Video-LLaVA's prefill phase.
    """

    def __init__(
        self,
        eventgpt_path: str = "./checkpoints/EventGPT-7b",
        videollava_model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
        device: str = "cuda",
        use_4bit: bool = True,
    ):
        self.device = device
        self.use_4bit = use_4bit
        self.eventgpt_path = eventgpt_path
        self.videollava_model_id = videollava_model_id

        # Models will be loaded lazily
        self._eventgpt_model = None
        self._eventgpt_tokenizer = None
        self._eventgpt_processor = None
        self._videollava_model = None
        self._videollava_processor = None

        # Thread-safety for parallel execution
        self._egpt_lock = threading.Lock()
        self._vl_lock = threading.Lock()

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

        print("Both models loaded.")

    def _eventgpt_prefill_and_decode(
        self,
        event_image_path: str,
        query: str,
        max_new_tokens: int,
        result_dict: Dict,
    ):
        """Run EventGPT prefill and decode in a thread."""
        from model.EventChatModel import get_spatio_temporal_features
        from dataset.conversation import prepare_event_prompt
        from dataset.constants import EVENT_TOKEN_INDEX
        from common.common import tokenizer_event_token, load_image

        with self._egpt_lock:
            try:
                # Stage 1-2: Data loading and preprocessing
                torch.cuda.synchronize()
                start_time = time.time()

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

                torch.cuda.synchronize()
                preprocess_time = time.time() - start_time

                # Stage 3-4: Vision encoding and prefill
                torch.cuda.synchronize()
                prefill_start = time.time()

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

                result_dict['egpt_prefill_complete_time'] = time.time()
                result_dict['egpt_prefill_time'] = preprocess_time + prefill_time

                # Stage 5: Decode (free tokens during VL prefill)
                torch.cuda.synchronize()
                decode_start = time.time()
                output_token_ids = []

                with torch.inference_mode():
                    cur_pos = inputs_embeds.shape[1]
                    kv_cache = outputs.past_key_values
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    output_token_ids.append(next_token.item())

                    for _ in range(max_new_tokens - 1):
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
                        output_token_ids.append(token_id)
                        cur_pos += 1

                        if token_id == self._eventgpt_tokenizer.eos_token_id:
                            break

                torch.cuda.synchronize()
                decode_time = time.time() - decode_start

                result_dict['egpt_decode_time'] = decode_time
                result_dict['egpt_output_tokens'] = output_token_ids
                result_dict['egpt_output_text'] = self._eventgpt_tokenizer.decode(
                    output_token_ids, skip_special_tokens=True
                )
                result_dict['egpt_total_time'] = preprocess_time + prefill_time + decode_time
                result_dict['egpt_success'] = True

            except Exception as e:
                result_dict['egpt_error'] = str(e)
                result_dict['egpt_success'] = False

    def _videollava_prefill_and_decode(
        self,
        video_frames: List,
        query: str,
        max_new_tokens: int,
        result_dict: Dict,
    ):
        """Run Video-LLaVA prefill and decode in a thread."""
        with self._vl_lock:
            try:
                # Stage 1-2: Preprocessing
                torch.cuda.synchronize()
                start_time = time.time()

                prompt = f"USER: <video>\n{query}\nASSISTANT:"
                inputs = self._videollava_processor(
                    text=prompt, videos=video_frames, return_tensors="pt"
                )

                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs.get('attention_mask')
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.to(self.device)

                pixel_values_videos = inputs.get('pixel_values_videos')

                torch.cuda.synchronize()
                preprocess_time = time.time() - start_time

                # Stage 3-4: Vision encoding + Prefill
                torch.cuda.synchronize()
                prefill_start = time.time()

                with torch.inference_mode():
                    outputs = self._videollava_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos.to(self.device),
                        past_key_values=None,
                        use_cache=True,
                    )
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                torch.cuda.synchronize()
                prefill_time = time.time() - prefill_start

                result_dict['vl_prefill_complete_time'] = time.time()
                result_dict['vl_prefill_time'] = preprocess_time + prefill_time

                # Stage 5: Decode
                torch.cuda.synchronize()
                decode_start = time.time()
                output_token_ids = [next_token.item()]

                with torch.inference_mode():
                    cur_token = next_token
                    attn_dtype = attention_mask.dtype if isinstance(attention_mask, torch.Tensor) else torch.long
                    cur_attention_mask = torch.ones(
                        (1, past_key_values[0][0].shape[2] + 1),
                        dtype=attn_dtype,
                        device=self.device
                    )

                    for _ in range(max_new_tokens - 1):
                        outputs = self._videollava_model(
                            input_ids=cur_token,
                            attention_mask=cur_attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )

                        logits = outputs.logits
                        past_key_values = outputs.past_key_values

                        next_token_logits = logits[:, -1, :]
                        cur_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                        token_id = cur_token.item()
                        output_token_ids.append(token_id)

                        cur_attention_mask = torch.ones(
                            (1, cur_attention_mask.shape[1] + 1),
                            dtype=attn_dtype, device=self.device
                        )

                        if token_id == self._videollava_processor.tokenizer.eos_token_id:
                            break

                torch.cuda.synchronize()
                decode_time = time.time() - decode_start

                result_dict['vl_decode_time'] = decode_time
                result_dict['vl_output_tokens'] = output_token_ids
                result_dict['vl_output_text'] = self._videollava_processor.tokenizer.decode(
                    output_token_ids, skip_special_tokens=True
                )
                result_dict['vl_total_time'] = preprocess_time + prefill_time + decode_time
                result_dict['vl_success'] = True

            except Exception as e:
                result_dict['vl_error'] = str(e)
                result_dict['vl_success'] = False

    def generate(
        self,
        event_image_path: str,
        video_frames: List,
        query: str,
        max_new_tokens: int = 50,
        return_preview: bool = True,
    ) -> PrefillOnlyResult:
        """
        Run parallel prefill inference.

        Args:
            event_image_path: Path to event image for EventGPT
            video_frames: List of PIL images for Video-LLaVA
            query: Text query
            max_new_tokens: Maximum tokens to generate
            return_preview: Whether to return EventGPT preview

        Returns:
            PrefillOnlyResult with outputs and timing
        """
        self._load_models()

        result_dict = {}
        wall_clock_start = time.time()

        # Run both models in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            egpt_future = executor.submit(
                self._eventgpt_prefill_and_decode,
                event_image_path,
                query,
                max_new_tokens,
                result_dict,
            )

            vl_future = executor.submit(
                self._videollava_prefill_and_decode,
                video_frames,
                query,
                max_new_tokens,
                result_dict,
            )

            # Wait for both to complete
            egpt_future.result()
            vl_future.result()

        torch.cuda.synchronize()
        wall_clock_time = time.time() - wall_clock_start

        # Compute analysis metrics
        egpt_prefill_time = result_dict.get('egpt_prefill_time', 0)
        vl_prefill_time = result_dict.get('vl_prefill_time', 0)
        overlap_window = max(0, vl_prefill_time - egpt_prefill_time)

        egpt_decode_time = result_dict.get('egpt_decode_time', 0)
        free_tokens_count = len(result_dict.get('egpt_output_tokens', []))

        # Calculate what portion of EGPT decode happened during VL prefill
        hidden_decode_time = min(overlap_window, egpt_decode_time)
        tokens_per_sec = free_tokens_count / max(egpt_decode_time, 0.001)
        free_tokens_during_overlap = int(hidden_decode_time * tokens_per_sec)

        sequential_time = (
            result_dict.get('egpt_total_time', 0) +
            result_dict.get('vl_total_time', 0)
        )

        return PrefillOnlyResult(
            output_text=result_dict.get('vl_output_text', ''),
            output_tokens=result_dict.get('vl_output_tokens', []),
            preview_text=result_dict.get('egpt_output_text') if return_preview else None,
            preview_tokens=result_dict.get('egpt_output_tokens') if return_preview else None,
            egpt_prefill_time=egpt_prefill_time,
            egpt_decode_time=egpt_decode_time,
            vl_prefill_time=vl_prefill_time,
            vl_decode_time=result_dict.get('vl_decode_time', 0),
            wall_clock_time=wall_clock_time,
            free_tokens_count=free_tokens_during_overlap,
            overlap_window=overlap_window,
            prefill_speedup=vl_prefill_time / max(egpt_prefill_time, 0.001),
            total_speedup_vs_sequential=sequential_time / max(wall_clock_time, 0.001),
        )

    def generate_videollava_only(
        self,
        video_frames: List,
        query: str,
        max_new_tokens: int = 50,
    ) -> Dict[str, Any]:
        """
        Baseline: Run Video-LLaVA only (no parallel execution).

        Returns:
            Dictionary with output and timing
        """
        self._load_models()

        result_dict = {}
        start_time = time.time()

        self._videollava_prefill_and_decode(
            video_frames,
            query,
            max_new_tokens,
            result_dict,
        )

        torch.cuda.synchronize()
        total_time = time.time() - start_time

        return {
            'output_text': result_dict.get('vl_output_text', ''),
            'output_tokens': result_dict.get('vl_output_tokens', []),
            'prefill_time': result_dict.get('vl_prefill_time', 0),
            'decode_time': result_dict.get('vl_decode_time', 0),
            'total_time': total_time,
        }


def main():
    """Quick test of prefill-only inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Prefill-only inference test")
    parser.add_argument("--event_image", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--query", type=str, default="What are the key elements in this scene?")
    parser.add_argument("--max_new_tokens", type=int, default=50)

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
    inference = PrefillOnlyInference()
    result = inference.generate(
        event_image_path=args.event_image,
        video_frames=frames,
        query=args.query,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n" + "="*60)
    print("PREFILL-ONLY INFERENCE RESULT")
    print("="*60)
    print(f"\nQuery: {args.query}")
    print(f"\nPreview (EventGPT): {result.preview_text}")
    print(f"\nFinal Output (Video-LLaVA): {result.output_text}")
    print(f"\n{'─'*60}")
    print("TIMING ANALYSIS:")
    print(f"  EGPT Prefill:     {result.egpt_prefill_time*1000:.1f} ms")
    print(f"  EGPT Decode:      {result.egpt_decode_time*1000:.1f} ms")
    print(f"  VL Prefill:       {result.vl_prefill_time*1000:.1f} ms")
    print(f"  VL Decode:        {result.vl_decode_time*1000:.1f} ms")
    print(f"  Overlap Window:   {result.overlap_window*1000:.1f} ms")
    print(f"  Free Tokens:      {result.free_tokens_count}")
    print(f"  Wall-Clock Time:  {result.wall_clock_time*1000:.1f} ms")
    print(f"  Prefill Speedup:  {result.prefill_speedup:.2f}x")
    print(f"  Total Speedup:    {result.total_speedup_vs_sequential:.2f}x vs sequential")
    print("="*60)


if __name__ == "__main__":
    main()
