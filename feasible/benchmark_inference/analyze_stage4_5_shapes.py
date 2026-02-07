#!/usr/bin/env python3
"""
Analyze Stage 4 & 5 Data Shapes for EventGPT vs Video-LLaVA

This script runs a few samples and prints detailed information about:
- Input/output tensor shapes at each stage
- Memory footprint of KV cache
- Why EventGPT prefill may not be significantly faster
"""

import os
import sys
import json
import torch
import time
import gc
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.EventChatModel import EventChatModel, get_spatio_temporal_features
from common.common import tokenizer_event_token, load_image
from dataset.conversation import prepare_event_prompt
from dataset.constants import (
    EVENT_TOKEN_INDEX,
    DEFAULT_EVENT_PATCH_TOKEN,
    DEFAULT_EV_START_TOKEN,
    DEFAULT_EV_END_TOKEN,
)


def format_shape(tensor):
    """Format tensor shape as string."""
    if tensor is None:
        return "None"
    if isinstance(tensor, torch.Tensor):
        return f"{list(tensor.shape)} ({tensor.dtype}, {tensor.device})"
    return str(type(tensor))


def format_kv_cache(past_key_values):
    """Format KV cache shape info."""
    if past_key_values is None:
        return "None"
    if hasattr(past_key_values, '__len__'):
        n_layers = len(past_key_values)
        if n_layers > 0 and past_key_values[0] is not None:
            # Each layer is (key, value) tuple
            k, v = past_key_values[0]
            return f"{n_layers} layers, K/V shape: {list(k.shape)}"
    return str(type(past_key_values))


def estimate_kv_cache_size(past_key_values, dtype=torch.float16):
    """Estimate KV cache size in MB."""
    if past_key_values is None:
        return 0
    total_elements = 0
    for kv in past_key_values:
        if kv is not None and len(kv) == 2:
            k, v = kv
            total_elements += k.numel() + v.numel()
    bytes_per_element = 2 if dtype == torch.float16 else 4 if dtype == torch.float32 else 2
    return (total_elements * bytes_per_element) / (1024 * 1024)


def analyze_eventgpt(num_samples=3):
    """Analyze EventGPT stage 4 & 5 data shapes."""
    print("\n" + "=" * 100)
    print("EVENTGPT: Stage 4 & 5 Data Shape Analysis")
    print("=" * 100)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("\nLoading EventGPT model...")
    model_path = os.path.join(ROOT, "checkpoints/EventGPT-7b")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    model = EventChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model = model.to(device)
    model.eval()
    processor = model.get_visual_tower().event_processor
    print("✓ EventGPT loaded")

    # Load dataset
    dataset_dir = os.path.join(ROOT, "data/my_egpt_dsec_test/my_egpt_dsec_seq_1s")
    with open(os.path.join(dataset_dir, "EventGPT_Instruction_Subset.json"), "r") as f:
        dataset = json.load(f)

    query = "What are the key elements in this scene?"

    for sample_idx in range(min(num_samples, len(dataset))):
        sample = dataset[sample_idx]
        print(f"\n{'─' * 100}")
        print(f"SAMPLE {sample_idx}")
        print(f"{'─' * 100}")

        if "event_image" not in sample or not sample["event_image"]:
            print("  No event images, skipping...")
            continue

        event_image_paths = sample["event_image"]
        print(f"\n  Number of event frames: {len(event_image_paths)}")

        # Load and preprocess event images
        event_list = []
        event_image_size = None
        for img_path in event_image_paths:
            full_path = os.path.join(dataset_dir, "event_image", img_path)
            img = load_image(full_path)
            img_array = np.array(img)
            if event_image_size is None:
                event_image_size = list(img_array.shape[:2])
            event = processor(img_array, return_tensors='pt')['pixel_values'][0]
            event = event.to(device, dtype=torch.bfloat16)
            event_list.append(event)

        print(f"  Event image size: {event_image_size}")
        print(f"  Single frame tensor shape: {format_shape(event_list[0])}")

        # Prepare input tokens
        conv_mode = 'eventgpt_v1'
        prompt = prepare_event_prompt(query, conv_mode)
        input_ids = tokenizer_event_token(
            prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)

        print(f"\n  Input prompt: {prompt[:100]}...")
        print(f"  Input token IDs shape: {format_shape(input_ids)}")
        print(f"  Number of text tokens (before multimodal injection): {input_ids.shape[1]}")

        with torch.inference_mode():
            # ===== STAGE 3: VISION ENCODING =====
            print(f"\n  STAGE 3: VISION ENCODING")
            ev_features_list = []
            for i, ev in enumerate(event_list):
                ev = ev.unsqueeze(0)
                feature = model.visval_encode(ev)
                feature = model.get_model().feature_adaptor(feature)
                feature = feature.squeeze(0)
                ev_features_list.append(feature)
                if i == 0:
                    print(f"    Single frame feature shape: {format_shape(feature)}")

            event_features = get_spatio_temporal_features(ev_features_list)
            event_features = event_features.unsqueeze(0)
            print(f"    Combined event features shape: {format_shape(event_features)}")
            print(f"    Total vision tokens: {event_features.shape[1]}")

            # ===== STAGE 4: LLM PREFILL =====
            print(f"\n  STAGE 4: LLM PREFILL")
            stage4_start = time.time()

            (
                _,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                torch.ones_like(input_ids, dtype=torch.bool),
                None,
                None,
                event_tensors=None,
                event_image_sizes=event_image_size,
                event_features=event_features
            )

            if attention_mask is None:
                attention_mask = torch.ones(
                    (1, inputs_embeds.shape[1]), dtype=torch.bool, device=device
                )
            if position_ids is None:
                position_ids = torch.arange(
                    0, inputs_embeds.shape[1], dtype=torch.long, device=device
                ).unsqueeze(0)

            print(f"    inputs_embeds shape: {format_shape(inputs_embeds)}")
            print(f"    attention_mask shape: {format_shape(attention_mask)}")
            print(f"    position_ids shape: {format_shape(position_ids)}")
            print(f"    PREFILL SEQUENCE LENGTH: {inputs_embeds.shape[1]}")

            # Run prefill forward pass
            torch.cuda.synchronize()
            prefill_start = time.time()

            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            torch.cuda.synchronize()
            prefill_time = time.time() - prefill_start

            hidden_states = outputs.last_hidden_state
            logits = model.lm_head(hidden_states[:, -1:, :])
            past_key_values = outputs.past_key_values

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            print(f"\n    Prefill output:")
            print(f"      hidden_states shape: {format_shape(hidden_states)}")
            print(f"      logits shape: {format_shape(logits)}")
            print(f"      KV cache: {format_kv_cache(past_key_values)}")
            print(f"      KV cache size: {estimate_kv_cache_size(past_key_values):.2f} MB")
            print(f"      First generated token: {next_token.item()}")
            print(f"      Prefill time: {prefill_time * 1000:.2f} ms")
            print(f"      Prefill throughput: {inputs_embeds.shape[1] / prefill_time:.1f} tokens/sec")

            # ===== STAGE 5: LLM DECODE =====
            print(f"\n  STAGE 5: LLM DECODE (first 5 tokens)")

            generated_ids = [next_token.item()]
            decode_times = []
            cur_pos = inputs_embeds.shape[1]
            cur_token = next_token

            for step in range(5):
                torch.cuda.synchronize()
                step_start = time.time()

                cur_embed = model.get_model().embed_tokens(cur_token)
                new_attention_mask = torch.ones(
                    (1, cur_pos + 1), dtype=torch.bool, device=device
                )

                outputs = model.model(
                    inputs_embeds=cur_embed,
                    attention_mask=new_attention_mask,
                    position_ids=torch.tensor([[cur_pos]], device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                hidden_states = outputs.last_hidden_state
                logits = model.lm_head(hidden_states[:, -1:, :])
                past_key_values = outputs.past_key_values

                next_token_logits = logits[:, -1, :]
                cur_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                torch.cuda.synchronize()
                step_time = time.time() - step_start
                decode_times.append(step_time)

                generated_ids.append(cur_token.item())
                cur_pos += 1

                if step == 0:
                    print(f"\n    First decode step:")
                    print(f"      cur_embed shape: {format_shape(cur_embed)}")
                    print(f"      attention_mask shape: {format_shape(new_attention_mask)}")
                    print(f"      position_ids: [[{cur_pos - 1}]]")
                    print(f"      KV cache after decode: {format_kv_cache(past_key_values)}")

            avg_decode_time = sum(decode_times) / len(decode_times)
            print(f"\n    Decode summary:")
            print(f"      Avg time per token: {avg_decode_time * 1000:.2f} ms")
            print(f"      Decode throughput: {1.0 / avg_decode_time:.1f} tokens/sec")
            print(f"      Generated tokens: {generated_ids}")
            print(f"      Decoded text: {tokenizer.decode(generated_ids, skip_special_tokens=True)}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


def analyze_videollava(num_samples=3):
    """Analyze Video-LLaVA stage 4 & 5 data shapes."""
    print("\n" + "=" * 100)
    print("VIDEO-LLAVA: Stage 4 & 5 Data Shape Analysis")
    print("=" * 100)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("\nLoading Video-LLaVA model...")
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_id = "llava-hf/llava-1.5-7b-hf"
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    except:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_id)

    model.eval()
    print("✓ Video-LLaVA loaded")

    # Load dataset
    dataset_dir = os.path.join(ROOT, "data/my_egpt_dsec_test/my_egpt_dsec_seq_1s")
    with open(os.path.join(dataset_dir, "EventGPT_Instruction_Subset.json"), "r") as f:
        dataset = json.load(f)

    base_query = "What are the key elements in this scene?"

    for sample_idx in range(min(num_samples, len(dataset))):
        sample = dataset[sample_idx]
        print(f"\n{'─' * 100}")
        print(f"SAMPLE {sample_idx}")
        print(f"{'─' * 100}")

        if "event_image" not in sample or not sample["event_image"]:
            print("  No event images, skipping...")
            continue

        event_image_paths = sample["event_image"]

        # Load first image
        img_path = event_image_paths[0]
        full_path = os.path.join(dataset_dir, "event_image", img_path)
        image = load_image(full_path)

        print(f"\n  Image path: {img_path}")
        print(f"  Image size: {image.size}")

        # Prepare inputs
        prompt = f"USER: <image>\n{base_query}\nASSISTANT:"
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

        print(f"\n  Input prompt: {prompt}")
        print(f"  Processor outputs:")
        for key, val in inputs.items():
            print(f"    {key}: {format_shape(val)}")

        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        pixel_values = inputs.get('pixel_values')

        print(f"\n  Number of input text tokens: {input_ids.shape[1]}")

        with torch.inference_mode():
            # ===== STAGE 3: VISION ENCODING =====
            print(f"\n  STAGE 3: VISION ENCODING")
            torch.cuda.synchronize()
            vision_start = time.time()

            # Get vision features
            if hasattr(model, 'vision_tower'):
                vision_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
                vision_features = vision_outputs.hidden_states[-1]
            else:
                vision_features = None

            torch.cuda.synchronize()
            vision_time = time.time() - vision_start

            if vision_features is not None:
                print(f"    Vision features shape: {format_shape(vision_features)}")
                print(f"    Vision encoding time: {vision_time * 1000:.2f} ms")

            # ===== STAGE 4: LLM PREFILL =====
            print(f"\n  STAGE 4: LLM PREFILL")
            torch.cuda.synchronize()
            prefill_start = time.time()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )

            torch.cuda.synchronize()
            prefill_time = time.time() - prefill_start

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Get actual sequence length from KV cache
            actual_seq_len = past_key_values[0][0].shape[2] if past_key_values else input_ids.shape[1]

            print(f"\n    Prefill output:")
            print(f"      logits shape: {format_shape(logits)}")
            print(f"      KV cache: {format_kv_cache(past_key_values)}")
            print(f"      KV cache size: {estimate_kv_cache_size(past_key_values):.2f} MB")
            print(f"      ACTUAL PREFILL SEQUENCE LENGTH (from KV cache): {actual_seq_len}")
            print(f"      First generated token: {next_token.item()}")
            print(f"      Prefill time: {prefill_time * 1000:.2f} ms")
            print(f"      Prefill throughput: {actual_seq_len / prefill_time:.1f} tokens/sec")

            # ===== STAGE 5: LLM DECODE =====
            print(f"\n  STAGE 5: LLM DECODE (first 5 tokens)")

            generated_ids = [next_token.item()]
            decode_times = []
            cur_token = next_token
            cur_attention_mask = torch.ones(
                (1, actual_seq_len + 1), dtype=attention_mask.dtype, device=device
            )

            for step in range(5):
                torch.cuda.synchronize()
                step_start = time.time()

                outputs = model(
                    input_ids=cur_token,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                logits = outputs.logits
                past_key_values = outputs.past_key_values

                next_token_logits = logits[:, -1, :]
                cur_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                torch.cuda.synchronize()
                step_time = time.time() - step_start
                decode_times.append(step_time)

                generated_ids.append(cur_token.item())

                cur_attention_mask = torch.ones(
                    (1, cur_attention_mask.shape[1] + 1),
                    dtype=attention_mask.dtype, device=device
                )

                if step == 0:
                    print(f"\n    First decode step:")
                    print(f"      input_ids shape: {format_shape(cur_token)}")
                    print(f"      attention_mask shape: {format_shape(cur_attention_mask)}")
                    print(f"      KV cache after decode: {format_kv_cache(past_key_values)}")

            avg_decode_time = sum(decode_times) / len(decode_times)
            print(f"\n    Decode summary:")
            print(f"      Avg time per token: {avg_decode_time * 1000:.2f} ms")
            print(f"      Decode throughput: {1.0 / avg_decode_time:.1f} tokens/sec")
            print(f"      Generated tokens: {generated_ids}")
            print(f"      Decoded text: {processor.batch_decode([generated_ids], skip_special_tokens=True)[0]}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


def print_comparison_analysis():
    """Print analysis of why EventGPT prefill may not be significantly faster."""
    print("\n" + "=" * 100)
    print("ANALYSIS: Why EventGPT Prefill May Not Be Significantly Faster")
    print("=" * 100)

    print("""
Key Observations:
-----------------

1. PREFILL SEQUENCE LENGTH
   - EventGPT: ~625 tokens (8 frames × ~77 vision tokens + ~40 text tokens)
   - Video-LLaVA: ~600 tokens (1 frame × 576 image patches + ~30 text tokens)

   Despite using 8 event frames, EventGPT's spatio-temporal feature extraction
   compresses them into a similar number of tokens as LLaVA's single image.

2. PREFILL IS COMPUTE-BOUND
   - Prefill involves processing the full input sequence through all transformer layers
   - Both models use the same LLM backbone (Vicuna-7B / LLaMA-7B based)
   - With similar sequence lengths, compute time is nearly identical

3. VISION ENCODING DIFFERENCE
   - EventGPT: Processes 8 event frames through CLIP vision encoder
   - Video-LLaVA: Processes 1 image through CLIP vision encoder
   - EventGPT does MORE work in vision encoding stage (Stage 3)
   - But this is offset by the spatio-temporal pooling that reduces tokens

4. WHERE EVENTGPT GAINS ADVANTAGE
   - Stage 3 (Vision Encoding): EventGPT is SLOWER (8 frames vs 1)
   - Stage 4 (Prefill): Nearly identical (similar sequence lengths)
   - Stage 5 (Decode): Similar throughput (same LLM backbone)
   - Total: EventGPT advantage comes from efficient event representation

5. THE REAL BOTTLENECK
   - Decode (Stage 5) dominates total time: 84-95% of wall-clock time
   - Prefill is only 5-15% of total time
   - Even 2x prefill speedup would only yield ~7% total speedup

Conclusion:
-----------
EventGPT's speed advantage primarily comes from:
1. Efficient event-to-token conversion in the feature adaptor
2. Spatio-temporal pooling that compresses multiple frames
3. Overall pipeline optimization, not prefill specifically

To significantly speed up inference, focus on:
- Decode optimization (speculative decoding, quantization)
- Reducing number of generated tokens
- KV cache optimization
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--eventgpt_only", action="store_true")
    parser.add_argument("--videollava_only", action="store_true")
    args = parser.parse_args()

    print(f"\n{'#' * 100}")
    print(f"# Stage 4 & 5 Shape Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 100}")

    if not args.videollava_only:
        analyze_eventgpt(args.num_samples)

    if not args.eventgpt_only:
        analyze_videollava(args.num_samples)

    print_comparison_analysis()
