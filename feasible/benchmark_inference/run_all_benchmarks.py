#!/usr/bin/env python3
"""
Run comprehensive benchmarks for all DSEC test datasets.

Usage:
    conda activate egpt
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python run_all_benchmarks.py
"""

import os
import subprocess
import json
from pathlib import Path

# Datasets to benchmark
DURATIONS = ["500ms", "1s", "2s", "4s", "5s", "8s", "10s", "16s", "20s"]
BASE_DIR = "/mnt/hdd/data/my_egpt_dsec_test"
MAX_SAMPLES = 100

def run_benchmark(duration: str) -> bool:
    """Run benchmark for a specific duration."""
    dataset_dir = f"{BASE_DIR}/my_egpt_dsec_seq_{duration}"
    output_file = f"{dataset_dir}/benchmark_results.json"

    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"⚠️  Skipping {duration}: dataset not found")
        return False

    # Check if already has 100 samples
    if os.path.exists(output_file):
        with open(output_file) as f:
            data = json.load(f)
            if len(data) >= 100:
                print(f"✓ {duration}: already has {len(data)} samples")
                return True

    print(f"▶️  Running {duration} benchmark...")

    cmd = [
        "python", "feasible/benchmark_inference/benchmark_inference.py",
        "--dataset_dir", dataset_dir,
        "--output_json", output_file,
        "--use_video_llava",
        "--max_samples", str(MAX_SAMPLES),
        "--device", "cuda"
    ]

    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    result = subprocess.run(
        ["conda", "run", "-n", "egpt", "--no-capture-output"] + cmd,
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ {duration} failed: {result.stderr[:200]}")
        return False

    # Verify output
    if os.path.exists(output_file):
        with open(output_file) as f:
            data = json.load(f)
        print(f"✓ {duration}: {len(data)} samples")
        return True

    return False

def calculate_metrics():
    """Calculate metrics across all datasets."""
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)

    results = []

    for duration in DURATIONS:
        dataset_dir = f"{BASE_DIR}/my_egpt_dsec_seq_{duration}"
        output_file = f"{dataset_dir}/benchmark_results.json"

        if not os.path.exists(output_file):
            continue

        with open(output_file) as f:
            data = json.load(f)

        if not data:
            continue

        # Calculate averages
        n = len(data)
        egpt_s3 = sum(s.get('egpt_stage3_time', 0) for s in data) / n
        llava_s3 = sum(s.get('llava_stage3_time', 0) for s in data) / n
        egpt_total = sum(s.get('egpt_time', 0) for s in data) / n
        llava_total = sum(s.get('llava_time', 0) for s in data) / n

        speedup_gen = llava_s3 / egpt_s3 if egpt_s3 > 0 else 0
        speedup_total = llava_total / egpt_total if egpt_total > 0 else 0

        # Calculate acceptance rate
        total_draft = 0
        total_accepted = 0
        for s in data:
            egpt_toks = s.get('egpt_token_ids', [])
            llava_toks = s.get('llava_token_ids', [])
            if egpt_toks and llava_toks:
                min_len = min(len(egpt_toks), len(llava_toks))
                accepted = sum(1 for j in range(min_len) if egpt_toks[j] == llava_toks[j])
                total_draft += len(egpt_toks)
                total_accepted += accepted

        alpha = total_accepted / total_draft if total_draft > 0 else 0

        results.append({
            'duration': duration,
            'samples': n,
            'egpt_s3': egpt_s3,
            'llava_s3': llava_s3,
            'speedup_gen': speedup_gen,
            'egpt_total': egpt_total,
            'llava_total': llava_total,
            'speedup_total': speedup_total,
            'alpha': alpha
        })

    # Print results table
    print()
    print("| Duration | Samples | EGPT S3 | VLLaVA S3 | Gen Speedup | EGPT Total | VLLaVA Total | Total Speedup | α |")
    print("|----------|---------|---------|-----------|-------------|------------|--------------|---------------|----|")

    for r in results:
        print(f"| {r['duration']} | {r['samples']} "
              f"| {r['egpt_s3']:.3f}s | {r['llava_s3']:.3f}s | {r['speedup_gen']:.2f}x "
              f"| {r['egpt_total']:.3f}s | {r['llava_total']:.3f}s | {r['speedup_total']:.2f}x "
              f"| {r['alpha']:.3f} |")

    # Summary
    if results:
        avg_gen_speedup = sum(r['speedup_gen'] for r in results) / len(results)
        avg_alpha = sum(r['alpha'] for r in results) / len(results)

        print()
        print(f"**Average Generation Speedup:** {avg_gen_speedup:.2f}x")
        print(f"**Average Acceptance Rate (α):** {avg_alpha:.3f} ({avg_alpha*100:.1f}%)")

    return results

def main():
    print("="*70)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("="*70)
    print(f"Datasets: {', '.join(DURATIONS)}")
    print(f"Max samples: {MAX_SAMPLES}")
    print(f"Model: Video-LLaVA-7B")
    print("="*70)
    print()

    # Run benchmarks
    completed = 0
    for duration in DURATIONS:
        if run_benchmark(duration):
            completed += 1
        print()

    print("="*70)
    print(f"COMPLETED: {completed}/{len(DURATIONS)} datasets")
    print("="*70)

    # Calculate metrics
    calculate_metrics()

if __name__ == "__main__":
    main()
