#!/usr/bin/env python3
"""
================================================================================
                    5-STAGE BENCHMARK: LLaSA (IMU MLLM)
================================================================================

README
------
This script provides comprehensive benchmarking of LLaSA (Large Language and
Sensor Assistant) with detailed 5-stage timing separation for inference analysis.

5-STAGE PIPELINE:
  Stage 1: Load IMU data from disk
  Stage 2: Preprocess IMU data (normalization)
  Stage 3: IMU encoding (LIMU-BERT forward pass)
  Stage 4: LLM Prefill (process input + IMU tokens, build KV cache)
  Stage 5: LLM Decode (autoregressive token generation)

KEY METRICS:
  - Prefill latency (input-dependent, parallelizable)
  - Decode throughput (memory-bound, sequential)
  - Time-to-first-token (TTFT) = Stages 1-4
  - Tokens per second in decode phase
  - KV cache memory usage

USAGE EXAMPLES:
  # Run benchmark with sample IMU data
  python benchmark_inference_5stages.py

  # Limit samples for quick testing
  python benchmark_inference_5stages.py --max_samples 10

  # Custom output path
  python benchmark_inference_5stages.py --output_dir ./results

REQUIREMENTS:
  - PyTorch with CUDA support
  - transformers >= 4.36.0
  - tqdm

MODEL: LLaSA-7B (LIMU-BERT encoder + Vicuna-7B decoder)
  - LIMU-BERT encoder: 62K parameters
  - Hidden size: 72 -> 4096 (via 2-layer MLP)
  - LLaMA-compatible tokenizer

CHANGELOG:
----------
[2026-01-29] v1.0.0 - Initial IMU 5-stage benchmark implementation
  - Adapted from EventGPT benchmark_inference_5stages.py
  - Uses LLaSA-7B for IMU-based multimodal inference
  - Measures LIMU-BERT encoding separately from LLM prefill

AUTHOR: EventGPT Team
================================================================================
"""

import os
import sys
import json
import argparse
import torch
import time
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

# Add project root and LLaSA paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LLASA_ROOT = os.path.join(ROOT, "research/imu_mllm/LLaSA/LLaSA")
if LLASA_ROOT not in sys.path:
    sys.path.insert(0, LLASA_ROOT)


def cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def estimate_kv_cache_mb(past_key_values):
    """Estimate KV cache size in MB."""
    if past_key_values is None:
        return 0
    total = 0
    for kv in past_key_values:
        if kv is not None and len(kv) == 2:
            k, v = kv
            total += k.numel() + v.numel()
    return (total * 2) / (1024 * 1024)  # 2 bytes for float16


def generate_synthetic_imu_data(num_samples: int, seq_len: int = 120, feature_num: int = 6) -> List[np.ndarray]:
    """Generate synthetic IMU data for benchmarking.

    Args:
        num_samples: Number of samples to generate
        seq_len: Sequence length (default 120 = 2 seconds at 60Hz)
        feature_num: Number of IMU features (default 6: 3 accel + 3 gyro)

    Returns:
        List of IMU data arrays, each with shape [seq_len, feature_num]
    """
    samples = []
    for i in range(num_samples):
        # Generate realistic IMU patterns
        t = np.linspace(0, 2, seq_len)

        # Accelerometer data (simulate walking motion)
        accel_x = 0.1 * np.sin(2 * np.pi * 2 * t) + np.random.randn(seq_len) * 0.05
        accel_y = 0.05 * np.sin(2 * np.pi * 4 * t + np.pi/4) + np.random.randn(seq_len) * 0.05
        accel_z = 9.8 + 0.3 * np.sin(2 * np.pi * 2 * t + np.pi/2) + np.random.randn(seq_len) * 0.1

        # Gyroscope data (simulate rotation)
        gyro_x = 0.1 * np.sin(2 * np.pi * 1 * t) + np.random.randn(seq_len) * 0.02
        gyro_y = 0.05 * np.cos(2 * np.pi * 1 * t) + np.random.randn(seq_len) * 0.02
        gyro_z = 0.02 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(seq_len) * 0.01

        imu_data = np.stack([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z], axis=1)
        samples.append(imu_data.astype(np.float32))

    return samples


def normalize_imu_data(imu_data: np.ndarray) -> np.ndarray:
    """Normalize IMU data for LIMU-BERT input.

    Args:
        imu_data: Raw IMU data with shape [seq_len, feature_num]

    Returns:
        Normalized IMU data
    """
    # Z-score normalization per feature
    mean = imu_data.mean(axis=0, keepdims=True)
    std = imu_data.std(axis=0, keepdims=True) + 1e-8
    normalized = (imu_data - mean) / std
    return normalized


def run_llasa_5stage_benchmark(
    model,
    tokenizer,
    imu_samples: List[np.ndarray],
    device: str = "cuda",
    max_new_tokens: int = 64
) -> List[Dict[str, Any]]:
    """
    Run LLaSA benchmark with 5-stage timing separation.

    Stages:
        1. Load IMU data from disk (simulated with memory access)
        2. Preprocess IMU data (normalization)
        3. IMU encoding (LIMU-BERT forward pass)
        4. LLM Prefill (process embeddings, build KV cache, generate first token)
        5. LLM Decode (autoregressive generation of remaining tokens)
    """
    print("\n" + "=" * 80)
    print("LLaSA: 5-Stage Benchmark (Prefill/Decode Separated)")
    print("=" * 80)

    results = []
    query = "What activity is the person doing based on this motion sensor data?"

    for sample_idx, imu_data in enumerate(tqdm(imu_samples, desc="LLaSA 5-Stage")):
        try:
            # ===== STAGE 1: LOAD IMU DATA =====
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage1_start = time.time()

            # Simulate loading from disk (actually just copy the data)
            imu_raw = imu_data.copy()

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage1_time = time.time() - stage1_start

            # ===== STAGE 2: PREPROCESS =====
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage2_start = time.time()

            # Normalize IMU data
            imu_normalized = normalize_imu_data(imu_raw)
            # Convert to tensor
            imu_tensor = torch.from_numpy(imu_normalized).float().unsqueeze(0).to(device)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage2_time = time.time() - stage2_start

            # Prepare input tokens
            # LLaSA uses <image> token as placeholder for sensor data
            prompt = f"USER: <image>\n{query}\nASSISTANT:"
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

            # ===== STAGE 3: IMU ENCODING (LIMU-BERT) =====
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage3_start = time.time()

            with torch.inference_mode():
                # Encode IMU data through LIMU-BERT
                imu_features = model.encode_images(imu_tensor.to(dtype=torch.float16))
                # imu_features shape: [1, seq_len, hidden_size=72] -> projected to [1, seq_len, 4096]

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage3_time = time.time() - stage3_start

            # ===== STAGE 4: LLM PREFILL =====
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage4_start = time.time()

            with torch.inference_mode():
                # Prepare multimodal inputs (IMU features already computed)
                (
                    _,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    _
                ) = model.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    None,  # position_ids
                    torch.ones_like(input_ids, dtype=torch.bool),  # attention_mask
                    None,  # past_key_values
                    None,  # labels
                    imu_tensor.to(dtype=torch.float16),  # images (IMU data)
                )

                # Create default attention mask if None
                if attention_mask is None:
                    attention_mask = torch.ones(
                        (1, inputs_embeds.shape[1]), dtype=torch.bool, device=device
                    )
                if position_ids is None:
                    position_ids = torch.arange(
                        0, inputs_embeds.shape[1], dtype=torch.long, device=device
                    ).unsqueeze(0)

                # Run prefill: single forward pass to build KV cache
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

                # Get logits for next token prediction
                hidden_states = outputs.last_hidden_state
                logits = model.lm_head(hidden_states[:, -1:, :])
                past_key_values = outputs.past_key_values

                # Generate first token
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage4_time = time.time() - stage4_start

            prefill_tokens = inputs_embeds.shape[1]
            kv_cache_mb = estimate_kv_cache_mb(past_key_values)

            # ===== STAGE 5: LLM DECODE =====
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage5_start = time.time()

            generated_ids = [next_token.item()]
            eos_token_id = tokenizer.eos_token_id

            with torch.inference_mode():
                cur_pos = inputs_embeds.shape[1]
                cur_token = next_token

                for step in range(max_new_tokens - 1):
                    # Get embedding for current token
                    cur_embed = model.get_model().embed_tokens(cur_token)

                    # Update attention mask
                    new_attention_mask = torch.ones(
                        (1, cur_pos + 1), dtype=torch.bool, device=device
                    )

                    # Forward pass with KV cache
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

                    # Sample next token (greedy for benchmarking)
                    next_token_logits = logits[:, -1, :]
                    cur_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    generated_ids.append(cur_token.item())
                    cur_pos += 1

                    # Check for EOS
                    if cur_token.item() == eos_token_id:
                        break

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            stage5_time = time.time() - stage5_start

            # Decode output text
            output_ids = torch.tensor([generated_ids], device=device)
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            num_generated = len(generated_ids)
            decode_tokens_per_sec = num_generated / stage5_time if stage5_time > 0 else 0
            ttft = stage1_time + stage2_time + stage3_time + stage4_time

            results.append({
                "sample": sample_idx,
                "stage1_time": stage1_time,
                "stage2_time": stage2_time,
                "stage3_time": stage3_time,
                "stage4_time": stage4_time,
                "stage5_time": stage5_time,
                "total_time": stage1_time + stage2_time + stage3_time + stage4_time + stage5_time,
                "ttft": ttft,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": num_generated,
                "decode_tokens_per_sec": decode_tokens_per_sec,
                "kv_cache_mb": kv_cache_mb,
                "output": output,
                "imu_seq_len": imu_data.shape[0],
            })

            if sample_idx % 10 == 0:
                print(f"\nSample {sample_idx}: "
                      f"S1={stage1_time*1000:.2f}ms | S2={stage2_time*1000:.2f}ms | "
                      f"S3(LIMU-BERT)={stage3_time*1000:.2f}ms | "
                      f"S4(prefill)={stage4_time*1000:.2f}ms | S5(decode)={stage5_time*1000:.2f}ms | "
                      f"TTFT={ttft*1000:.2f}ms | Decode={decode_tokens_per_sec:.1f} tok/s | "
                      f"Tokens={num_generated}")

        except Exception as e:
            print(f"\nError on sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def print_5stage_summary(results: List[Dict], model_name: str = "LLaSA") -> Dict:
    """Print summary statistics for 5-stage benchmark."""
    if not results:
        print(f"No results for {model_name}")
        return {}

    n = len(results)

    avg_s1 = sum(r['stage1_time'] for r in results) / n
    avg_s2 = sum(r['stage2_time'] for r in results) / n
    avg_s3 = sum(r['stage3_time'] for r in results) / n
    avg_s4 = sum(r['stage4_time'] for r in results) / n
    avg_s5 = sum(r['stage5_time'] for r in results) / n
    avg_total = avg_s1 + avg_s2 + avg_s3 + avg_s4 + avg_s5
    avg_ttft = sum(r['ttft'] for r in results) / n
    avg_prefill_tokens = sum(r['prefill_tokens'] for r in results) / n
    avg_decode_tokens = sum(r['decode_tokens'] for r in results) / n
    avg_decode_tps = sum(r['decode_tokens_per_sec'] for r in results) / n
    avg_kv_cache_mb = sum(r.get('kv_cache_mb', 0) for r in results) / n

    # Percentages
    s1_pct = (avg_s1 / avg_total) * 100 if avg_total > 0 else 0
    s2_pct = (avg_s2 / avg_total) * 100 if avg_total > 0 else 0
    s3_pct = (avg_s3 / avg_total) * 100 if avg_total > 0 else 0
    s4_pct = (avg_s4 / avg_total) * 100 if avg_total > 0 else 0
    s5_pct = (avg_s5 / avg_total) * 100 if avg_total > 0 else 0

    prefill_total = avg_s1 + avg_s2 + avg_s3 + avg_s4
    prefill_pct = (prefill_total / avg_total) * 100 if avg_total > 0 else 0
    decode_pct = s5_pct

    print(f"\n{'=' * 80}")
    print(f"5-STAGE BENCHMARK SUMMARY - {model_name}")
    print(f"{'=' * 80}")
    print(f"Samples processed:                 {n}")
    print(f"\n{'─' * 80}")
    print("STAGE BREAKDOWN:")
    print(f"{'─' * 80}")
    print(f"  Stage 1 (Load IMU):              {avg_s1*1000:.3f}ms ({s1_pct:.1f}%)")
    print(f"  Stage 2 (Preprocess):            {avg_s2*1000:.3f}ms ({s2_pct:.1f}%)")
    print(f"  Stage 3 (LIMU-BERT Encoding):    {avg_s3*1000:.3f}ms ({s3_pct:.1f}%)")
    print(f"  Stage 4 (LLM Prefill):           {avg_s4*1000:.3f}ms ({s4_pct:.1f}%)")
    print(f"  Stage 5 (LLM Decode):            {avg_s5*1000:.3f}ms ({s5_pct:.1f}%)")
    print(f"{'─' * 80}")
    print(f"  Total per sample:                {avg_total*1000:.3f}ms")

    print(f"\n{'─' * 80}")
    print("PREFILL vs DECODE:")
    print(f"{'─' * 80}")
    print(f"  Prefill (S1-S4):                 {prefill_total*1000:.3f}ms ({prefill_pct:.1f}%)")
    print(f"  Decode (S5):                     {avg_s5*1000:.3f}ms ({decode_pct:.1f}%)")
    print(f"  Time to First Token (TTFT):      {avg_ttft*1000:.3f}ms")

    print(f"\n{'─' * 80}")
    print("THROUGHPUT:")
    print(f"{'─' * 80}")
    print(f"  Avg prefill tokens:              {avg_prefill_tokens:.1f}")
    print(f"  Avg decode tokens:               {avg_decode_tokens:.1f}")
    print(f"  Decode throughput:               {avg_decode_tps:.1f} tokens/sec")
    print(f"  Prefill throughput:              {avg_prefill_tokens / avg_s4:.1f} tokens/sec" if avg_s4 > 0 else "")
    print(f"  End-to-end throughput:           {1.0 / avg_total:.2f} samples/sec" if avg_total > 0 else "")

    print(f"\n{'─' * 80}")
    print("MEMORY:")
    print(f"{'─' * 80}")
    print(f"  Avg KV Cache:                    {avg_kv_cache_mb:.2f} MB")

    return {
        'samples': n,
        'stage1_avg': avg_s1,
        'stage2_avg': avg_s2,
        'stage3_avg': avg_s3,
        'stage4_avg': avg_s4,
        'stage5_avg': avg_s5,
        'total_avg': avg_total,
        'ttft_avg': avg_ttft,
        'prefill_tokens_avg': avg_prefill_tokens,
        'decode_tokens_avg': avg_decode_tokens,
        'decode_tps_avg': avg_decode_tps,
        'kv_cache_mb_avg': avg_kv_cache_mb,
        'prefill_pct': prefill_pct,
        'decode_pct': decode_pct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="5-Stage Benchmark for LLaSA (IMU MLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic IMU data
  python benchmark_inference_5stages.py

  # Limit samples for quick testing
  python benchmark_inference_5stages.py --max_samples 10
        """
    )
    parser.add_argument("--llasa_model_path", type=str,
                        default=None,
                        help="Path to LLaSA model (default: research/imu_mllm/LLaSA/checkpoints/LLaSA-7B)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Number of samples to benchmark")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max tokens to generate per sample")
    parser.add_argument("--imu_seq_len", type=int, default=120,
                        help="IMU sequence length (default 120 = 2 seconds at 60Hz)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")

    args = parser.parse_args()

    # Set default paths
    if args.llasa_model_path is None:
        args.llasa_model_path = os.path.join(ROOT, "research/imu_mllm/LLaSA/checkpoints/LLaSA-7B")

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(__file__))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'#' * 80}")
    print(f"# LLaSA 5-Stage Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}")
    print(f"Model path: {args.llasa_model_path}")
    print(f"Samples: {args.max_samples}")
    print(f"IMU sequence length: {args.imu_seq_len}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Output directory: {args.output_dir}")

    # Generate synthetic IMU data
    print(f"\nGenerating {args.max_samples} synthetic IMU samples...")
    imu_samples = generate_synthetic_imu_data(
        num_samples=args.max_samples,
        seq_len=args.imu_seq_len,
        feature_num=6
    )
    print(f"Generated {len(imu_samples)} samples with shape {imu_samples[0].shape}")

    # Load LLaSA model
    print(f"\nLoading LLaSA from {args.llasa_model_path}...")

    try:
        from llava.model.builder import load_pretrained_model

        tokenizer, model, _, context_len = load_pretrained_model(
            model_path=args.llasa_model_path,
            model_base=None,
            model_name="llava",
            device=args.device
        )
        model.eval()
        print("✓ LLaSA loaded successfully")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params / 1e9:.2f}B")
        print(f"  Context length: {context_len}")

    except Exception as e:
        print(f"Error loading LLaSA: {e}")
        print("\nNote: LLaSA model requires the full checkpoint from HuggingFace.")
        print("The current checkpoint may be incomplete.")
        print("\nRunning benchmark with synthetic timing instead...")

        # Generate synthetic results for demonstration
        results = []
        for i in tqdm(range(args.max_samples), desc="Synthetic benchmark"):
            results.append({
                "sample": i,
                "stage1_time": 0.0001,  # ~0.1ms for memory copy
                "stage2_time": 0.0005,  # ~0.5ms for normalization
                "stage3_time": 0.002,   # ~2ms for LIMU-BERT (62K params)
                "stage4_time": 0.050,   # ~50ms for prefill
                "stage5_time": 0.800,   # ~800ms for 64 token decode
                "total_time": 0.853,
                "ttft": 0.053,
                "prefill_tokens": 150,
                "decode_tokens": 64,
                "decode_tokens_per_sec": 80.0,
                "kv_cache_mb": 64.0,
                "output": "[Synthetic output]",
                "imu_seq_len": args.imu_seq_len,
            })

        summary = print_5stage_summary(results, "LLaSA-7B (Synthetic)")

        # Save results
        if args.output_json is None:
            args.output_json = os.path.join(
                args.output_dir,
                f"benchmark_llasa_5stages_synthetic_{timestamp}.json"
            )

        output_data = {
            'config': {
                'model': 'LLaSA-7B (Synthetic)',
                'max_samples': args.max_samples,
                'imu_seq_len': args.imu_seq_len,
                'max_new_tokens': args.max_new_tokens,
                'timestamp': datetime.now().isoformat(),
                'note': 'Synthetic results - model not loaded'
            },
            'summary': summary,
            'samples': results
        }

        os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n✓ Synthetic results saved to: {args.output_json}")

        return

    # Run benchmark
    results = run_llasa_5stage_benchmark(
        model=model,
        tokenizer=tokenizer,
        imu_samples=imu_samples,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )

    # Print summary
    summary = print_5stage_summary(results, "LLaSA-7B")

    # Save results
    if args.output_json is None:
        args.output_json = os.path.join(
            args.output_dir,
            f"benchmark_llasa_5stages_{args.max_samples}samples_{timestamp}.json"
        )

    output_data = {
        'config': {
            'llasa_model_path': args.llasa_model_path,
            'max_samples': args.max_samples,
            'imu_seq_len': args.imu_seq_len,
            'max_new_tokens': args.max_new_tokens,
            'timestamp': datetime.now().isoformat(),
        },
        'summary': summary,
        'samples': results
    }

    os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {args.output_json}")

    # Print recommendations
    print(f"\n{'=' * 80}")
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print(f"{'=' * 80}")

    if summary:
        print(f"\n📊 LLaSA Analysis:")
        print(f"   • LIMU-BERT encoding (S3): {summary.get('stage3_avg', 0)*1000:.3f}ms")
        print(f"   • Prefill: {summary.get('prefill_pct', 0):.1f}% of total time")
        print(f"   • Decode: {summary.get('decode_pct', 0):.1f}% of total time")
        print(f"   • TTFT (Time to First Token): {summary.get('ttft_avg', 0)*1000:.3f}ms")
        print(f"   • Decode throughput: {summary.get('decode_tps_avg', 0):.1f} tokens/sec")

        print(f"\n🔑 Key Observations:")
        print(f"   • LIMU-BERT encoder is extremely lightweight (62K params)")
        print(f"   • Stage 3 (IMU encoding) should be negligible compared to LLM stages")
        print(f"   • IMU tokens: ~120 (vs ~577 for CLIP image tokens)")
        print(f"   • Potential for fast draft model in speculative decoding")

    print(f"\n{'=' * 80}")
    print("5-STAGE BENCHMARK COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
