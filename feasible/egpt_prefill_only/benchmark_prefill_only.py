#!/usr/bin/env python3
"""
================================================================================
        PREFILL-ONLY BENCHMARK: Parallel Prefill Benefit Analysis
================================================================================

Benchmarks the prefill-only approach where:
1. EventGPT and Video-LLaVA prefill run in parallel
2. EventGPT generates "free" tokens during Video-LLaVA's slower prefill
3. Video-LLaVA uses standard AR decode (no speculative verification)

This measures the benefit of parallel prefill without the complexity of
speculative decoding.

USAGE:
    python benchmark_prefill_only.py --dataset_dir DATASET_DIR --max_samples N

OUTPUT:
    JSON file: prefill_only_benchmark_{timestamp}.json
    Markdown report: prefill_only_benchmark_{timestamp}.md
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
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from dataclasses import asdict

# Fix protobuf compatibility
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    print("Warning: PyAV not found. Install with: pip install av")


def load_video_frames_from_mp4(video_path: str, num_frames: int = 8) -> List:
    """Load frames from MP4 video using PyAV."""
    if not HAS_PYAV:
        raise ImportError("PyAV required for MP4 loading. Install with: pip install av")

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
            frames.append(frame.to_image())

    container.close()
    return frames


def get_gpu_memory_mb() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def benchmark_sample(
    inference,
    sample: Dict,
    dataset_dir: str,
    query: str,
    max_new_tokens: int,
) -> Optional[Dict]:
    """Benchmark a single sample with prefill-only approach."""
    from common.common import load_image

    try:
        # Get event image path
        event_image_paths = sample.get("event_image", [])
        if not event_image_paths:
            return None

        event_image_path = os.path.join(dataset_dir, "event_image", event_image_paths[0])

        # Get video frames
        video_data = sample.get("video_data")
        if video_data:
            video_path = os.path.join(dataset_dir, "mp4", video_data + ".mp4")
            if not os.path.exists(video_path):
                return None
            video_frames = load_video_frames_from_mp4(video_path, num_frames=8)
        else:
            # Fall back to event images as video frames
            video_frames = []
            for img_path in event_image_paths[:8]:
                full_path = os.path.join(dataset_dir, "event_image", img_path)
                img = load_image(full_path)
                video_frames.append(img)
            while len(video_frames) < 8:
                video_frames.append(video_frames[-1])

        if len(video_frames) == 0:
            return None

        # Run parallel prefill inference
        result = inference.generate(
            event_image_path=event_image_path,
            video_frames=video_frames,
            query=query,
            max_new_tokens=max_new_tokens,
        )

        # Run Video-LLaVA only baseline
        baseline = inference.generate_videollava_only(
            video_frames=video_frames,
            query=query,
            max_new_tokens=max_new_tokens,
        )

        return {
            "parallel": asdict(result),
            "baseline": baseline,
            "speedup": baseline['total_time'] / max(result.wall_clock_time, 0.001),
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()[-500:]}


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute aggregate statistics from benchmark results."""
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        return {}

    def safe_avg(values):
        return float(np.mean(values)) if values else 0.0

    def safe_std(values):
        return float(np.std(values)) if values else 0.0

    # Parallel prefill statistics
    parallel_stats = {
        "egpt_prefill_time_avg": safe_avg([r["parallel"]["egpt_prefill_time"] for r in valid_results]),
        "egpt_prefill_time_std": safe_std([r["parallel"]["egpt_prefill_time"] for r in valid_results]),
        "egpt_decode_time_avg": safe_avg([r["parallel"]["egpt_decode_time"] for r in valid_results]),
        "vl_prefill_time_avg": safe_avg([r["parallel"]["vl_prefill_time"] for r in valid_results]),
        "vl_prefill_time_std": safe_std([r["parallel"]["vl_prefill_time"] for r in valid_results]),
        "vl_decode_time_avg": safe_avg([r["parallel"]["vl_decode_time"] for r in valid_results]),
        "wall_clock_time_avg": safe_avg([r["parallel"]["wall_clock_time"] for r in valid_results]),
        "wall_clock_time_std": safe_std([r["parallel"]["wall_clock_time"] for r in valid_results]),
        "overlap_window_avg": safe_avg([r["parallel"]["overlap_window"] for r in valid_results]),
        "free_tokens_avg": safe_avg([r["parallel"]["free_tokens_count"] for r in valid_results]),
        "prefill_speedup_avg": safe_avg([r["parallel"]["prefill_speedup"] for r in valid_results]),
    }

    # Baseline statistics
    baseline_stats = {
        "prefill_time_avg": safe_avg([r["baseline"]["prefill_time"] for r in valid_results]),
        "decode_time_avg": safe_avg([r["baseline"]["decode_time"] for r in valid_results]),
        "total_time_avg": safe_avg([r["baseline"]["total_time"] for r in valid_results]),
        "total_time_std": safe_std([r["baseline"]["total_time"] for r in valid_results]),
    }

    # Overall speedup
    speedup_values = [r["speedup"] for r in valid_results]

    return {
        "n_samples": len(valid_results),
        "n_errors": len(results) - len(valid_results),
        "parallel": parallel_stats,
        "baseline": baseline_stats,
        "speedup_avg": safe_avg(speedup_values),
        "speedup_std": safe_std(speedup_values),
        "speedup_min": min(speedup_values) if speedup_values else 0,
        "speedup_max": max(speedup_values) if speedup_values else 0,
    }


def generate_markdown_report(
    stats: Dict,
    results: List[Dict],
    dataset_name: str,
    output_path: str,
    config: Dict,
    memory_stats: Dict = None,
):
    """Generate markdown report."""
    par = stats.get("parallel", {})
    base = stats.get("baseline", {})
    mem = memory_stats or {}

    # Sample outputs
    sample_outputs = []
    for r in results[:3]:
        if "error" not in r:
            sample_outputs.append({
                "preview": r["parallel"].get("preview_text", "")[:200],
                "output": r["parallel"].get("output_text", "")[:200],
            })

    report = f"""# Prefill-Only Benchmark Report: {dataset_name}

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Samples:** {stats.get('n_samples', 0)}
**Errors:** {stats.get('n_errors', 0)}
**Max New Tokens:** {config.get('max_new_tokens', 50)}

## Executive Summary

This benchmark measures the benefit of parallel prefill without speculative decoding.
EventGPT's faster prefill creates an overlap window for "free" draft token generation.

| Metric | Value |
|--------|-------|
| **Overlap Window** | {par.get('overlap_window_avg', 0)*1000:.1f} ms |
| **Free Tokens** | {par.get('free_tokens_avg', 0):.1f} tokens |
| **Prefill Speedup** | {par.get('prefill_speedup_avg', 0):.2f}x |
| **Overall Speedup** | {stats.get('speedup_avg', 0):.3f}x |

## Timing Comparison

### Parallel Prefill Approach

| Stage | Time (ms) |
|-------|-----------|
| EventGPT Prefill | {par.get('egpt_prefill_time_avg', 0)*1000:.1f} ± {par.get('egpt_prefill_time_std', 0)*1000:.1f} |
| EventGPT Decode | {par.get('egpt_decode_time_avg', 0)*1000:.1f} |
| Video-LLaVA Prefill | {par.get('vl_prefill_time_avg', 0)*1000:.1f} ± {par.get('vl_prefill_time_std', 0)*1000:.1f} |
| Video-LLaVA Decode | {par.get('vl_decode_time_avg', 0)*1000:.1f} |
| **Wall-Clock Total** | **{par.get('wall_clock_time_avg', 0)*1000:.1f} ± {par.get('wall_clock_time_std', 0)*1000:.1f}** |

### Baseline (Video-LLaVA Only)

| Stage | Time (ms) |
|-------|-----------|
| Prefill | {base.get('prefill_time_avg', 0)*1000:.1f} |
| Decode | {base.get('decode_time_avg', 0)*1000:.1f} |
| **Total** | **{base.get('total_time_avg', 0)*1000:.1f} ± {base.get('total_time_std', 0)*1000:.1f}** |

## Parallel Execution Analysis

```
Timeline (Parallel Prefill):

  EventGPT:     |--Prefill--|---------Decode---------|
                ({par.get('egpt_prefill_time_avg', 0)*1000:.0f}ms)     ({par.get('egpt_decode_time_avg', 0)*1000:.0f}ms)

  Video-LLaVA:  |----------Prefill----------|---------Decode---------|
                    ({par.get('vl_prefill_time_avg', 0)*1000:.0f}ms)              ({par.get('vl_decode_time_avg', 0)*1000:.0f}ms)

                |<---- Overlap Window ----->|
                     ({par.get('overlap_window_avg', 0)*1000:.0f}ms)
                     ({par.get('free_tokens_avg', 0):.0f} free tokens)
```

## Speedup Analysis

| Metric | Value |
|--------|-------|
| Average Speedup | {stats.get('speedup_avg', 0):.3f}x |
| Std Dev | ± {stats.get('speedup_std', 0):.3f}x |
| Min | {stats.get('speedup_min', 0):.3f}x |
| Max | {stats.get('speedup_max', 0):.3f}x |

## Interpretation

The speedup from parallel prefill is limited because:
1. Wall-clock time = max(EGPT, VL) for prefill + VL decode
2. Since VL prefill dominates, parallel prefill ≈ VL total time
3. Main benefit is the "free" EventGPT tokens during overlap

### Time Saved
- Overlap window: {par.get('overlap_window_avg', 0)*1000:.0f}ms
- Free tokens: {par.get('free_tokens_avg', 0):.0f} tokens
- Effective savings: {par.get('free_tokens_avg', 0) * 18:.0f}ms worth of EventGPT generation

### Use Cases
1. **Early Preview**: Return EventGPT output immediately as preview
2. **Draft Priming**: Use drafts to guide Video-LLaVA's first tokens
3. **Hybrid Output**: Combine fast EGPT response with high-quality VL refinement

## GPU Memory (4-bit Quantization)

| Model | Memory |
|-------|--------|
| EventGPT | {mem.get('eventgpt_model_mb', 0):.0f} MB |
| Video-LLaVA | {mem.get('videollava_model_mb', 0):.0f} MB |
| **Total** | **{mem.get('total_models_mb', 0):.0f} MB** |

## Sample Outputs

"""

    for i, s in enumerate(sample_outputs):
        report += f"""### Sample {i+1}

**EventGPT Preview:**
> {s['preview']}...

**Video-LLaVA Output:**
> {s['output']}...

---

"""

    report += f"""
## Conclusions

1. **Prefill Speedup**: EventGPT prefill is {par.get('prefill_speedup_avg', 0):.1f}x faster than Video-LLaVA
2. **Free Tokens**: {par.get('free_tokens_avg', 0):.0f} EventGPT tokens generated during VL prefill
3. **Overall Speedup**: {stats.get('speedup_avg', 0):.3f}x (limited by VL decode time)

### Comparison with Full Speculative Decoding

| Approach | Complexity | Speedup | Requirements |
|----------|------------|---------|--------------|
| Prefill Only | Low | {stats.get('speedup_avg', 0):.2f}x | Just parallel execution |
| Full SD (α=30%) | High | 1.3x | Token alignment model |
| Full SD (α=50%) | High | 1.5x | Well-trained adapter |

**Recommendation**: Use Prefill Only for simplicity. Only invest in Full SD if acceptance rate can reach >50%.

---

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to {output_path}")


# Default trained TokenAdapter (27.9% acceptance, 51.6% top-5)
DEFAULT_ADAPTER_PATH = "./feasible/token_alignment/task/starred/1q_20260128_151847/best_model.pt"


def main():
    parser = argparse.ArgumentParser(description="Prefill-only benchmark")
    parser.add_argument("--dataset_dir", type=str,
                        default="./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_alignment", action="store_true",
                        help="Use TokenAdapter for draft alignment")
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH,
                        help="Path to trained TokenAdapter checkpoint")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    json_path = os.path.join(args.dataset_dir, "EventGPT_Instruction_Subset.json")
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))

    print("\n" + "="*80)
    print("PREFILL-ONLY BENCHMARK")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(dataset) if args.max_samples == -1 else args.max_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")

    # Initialize inference (loads models)
    from prefill_only import PrefillOnlyInference

    print("\nLoading models...")
    inference = PrefillOnlyInference(
        eventgpt_path="./checkpoints/EventGPT-7b",
        videollava_model_id="LanguageBind/Video-LLaVA-7B-hf",
        device=args.device,
        use_4bit=True,
    )
    inference._load_models()

    memory_stats = {
        "eventgpt_model_mb": get_gpu_memory_mb()["allocated_mb"],
    }
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats["total_models_mb"] = get_gpu_memory_mb()["allocated_mb"]
    memory_stats["videollava_model_mb"] = memory_stats["total_models_mb"] - memory_stats["eventgpt_model_mb"]

    print(f"GPU Memory: {memory_stats['total_models_mb']:.0f} MB")

    # Run benchmark
    query = "What are the key elements in this scene?"
    results = []

    samples_to_process = dataset if args.max_samples == -1 else dataset[:args.max_samples]

    print(f"\nRunning benchmark on {len(samples_to_process)} samples...")

    for sample_idx, sample in enumerate(tqdm(samples_to_process, desc="Benchmark")):
        result = benchmark_sample(
            inference,
            sample,
            args.dataset_dir,
            query,
            args.max_new_tokens,
        )

        if result:
            result["sample_idx"] = sample_idx
            results.append(result)

    # Compute statistics
    stats = compute_statistics(results)

    # Save results
    output_data = {
        "timestamp": timestamp,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name,
        "config": {
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
        },
        "gpu_memory": memory_stats,
        "statistics": stats,
        "results": results,
    }

    json_path = os.path.join(args.output_dir, f"prefill_only_benchmark_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {json_path}")

    # Generate report
    md_path = os.path.join(args.output_dir, f"prefill_only_benchmark_{timestamp}.md")
    generate_markdown_report(stats, results, dataset_name, md_path,
                            {"max_new_tokens": args.max_new_tokens}, memory_stats)

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    if stats:
        par = stats.get("parallel", {})
        base = stats.get("baseline", {})

        print(f"\nSamples processed: {stats['n_samples']}")
        print(f"Errors: {stats['n_errors']}")

        print(f"\n{'─'*40}")
        print("Parallel Prefill:")
        print(f"{'─'*40}")
        print(f"  EGPT Prefill:     {par.get('egpt_prefill_time_avg', 0)*1000:.1f} ms")
        print(f"  VL Prefill:       {par.get('vl_prefill_time_avg', 0)*1000:.1f} ms")
        print(f"  Overlap Window:   {par.get('overlap_window_avg', 0)*1000:.1f} ms")
        print(f"  Free Tokens:      {par.get('free_tokens_avg', 0):.1f}")
        print(f"  Wall-Clock Time:  {par.get('wall_clock_time_avg', 0)*1000:.1f} ms")

        print(f"\n{'─'*40}")
        print("Baseline (VL Only):")
        print(f"{'─'*40}")
        print(f"  Total Time:       {base.get('total_time_avg', 0)*1000:.1f} ms")

        print(f"\n{'─'*40}")
        print("Speedup:")
        print(f"{'─'*40}")
        print(f"  Prefill Speedup:  {par.get('prefill_speedup_avg', 0):.2f}x (EGPT vs VL)")
        print(f"  Overall Speedup:  {stats.get('speedup_avg', 0):.3f}x")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
