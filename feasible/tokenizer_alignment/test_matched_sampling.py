"""
Test if matching sampling parameters improves acceptance rate (α).

Hypothesis: If both models use greedy decoding (temperature=0), they should
generate more similar text, improving α.
"""

import json
import os
import subprocess
import sys

def run_benchmark_with_params(temperature, top_p, max_samples=5):
    """Run benchmark with specific sampling parameters."""
    output_path = f"/mnt/hdd/data/my_egpt_dsec_seq_20s/benchmark_results_t{temperature}.json"

    cmd = [
        "python", "feasible/benchmark_inference/benchmark_inference.py",
        "--dataset_dir", "/mnt/hdd/data/my_egpt_dsec_seq_20s",
        "--output_json", output_path,
        "--use_video_llava",
        "--max_samples", str(max_samples),
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--device", "cuda"
    ]

    print(f"Running benchmark with temperature={temperature}, top_p={top_p}")
    print(f"Command: {' '.join(cmd)}")

    # Use conda run to activate environment
    cmd_with_conda = ["conda", "run", "-n", "egpt", "--no-capture-output"] + cmd
    # Set env var for protobuf
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    result = subprocess.run(
        cmd_with_conda,
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    return output_path

def analyze_acceptance_rate(benchmark_path):
    """Calculate acceptance rate from benchmark results."""
    with open(benchmark_path) as f:
        data = json.load(f)

    total_draft = 0
    total_accepted = 0

    for sample in data:
        egpt_tokens = sample.get('egpt_token_ids', [])
        llava_tokens = sample.get('llava_token_ids', [])

        if not egpt_tokens or not llava_tokens:
            continue

        min_len = min(len(egpt_tokens), len(llava_tokens))
        accepted = sum(1 for j in range(min_len) if egpt_tokens[j] == llava_tokens[j])

        total_draft += len(egpt_tokens)
        total_accepted += accepted

    alpha = total_accepted / total_draft if total_draft > 0 else 0
    return alpha, len(data)

def main():
    print("="*70)
    print("Testing: Does matching sampling parameters improve α?")
    print("="*70)
    print()

    # Test configurations
    configs = [
        {"temperature": 0.0, "top_p": 1.0, "name": "Greedy (temp=0)"},
        {"temperature": 0.2, "top_p": 1.0, "name": "Low temp (0.2)"},
        {"temperature": 0.6, "top_p": 1.0, "name": "Default EventGPT (0.6)"},
    ]

    results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")

        output_path = run_benchmark_with_params(
            config['temperature'],
            config['top_p'],
            max_samples=5
        )

        if output_path and os.path.exists(output_path):
            alpha, samples = analyze_acceptance_rate(output_path)
            results.append({
                "config": config['name'],
                "temperature": config['temperature'],
                "alpha": alpha,
                "samples": samples
            })
            print(f"α = {alpha:.4f} ({alpha*100:.1f}%)")
        else:
            print(f"Benchmark failed for {config['name']}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Acceptance Rate by Sampling Configuration")
    print("="*70)
    print()
    print("| Configuration | Temperature | α (Acceptance) | Samples |")
    print("|---------------|--------------|-----------------|---------|")

    for r in results:
        print(f"| {r['config']} | {r['temperature']} | {r['alpha']:.4f} ({r['alpha']*100:.1f}%) | {r['samples']} |")

    print()
    print("Baseline comparison:")
    print("| Target Model | α |")
    print("|--------------|-----|")
    print("| LLaVA 1.5 (default) | 0.068 (6.8%) |")
    print("| Video-LLaVA (default) | 0.039 (3.9%) |")

    # Best result
    if results:
        best = max(results, key=lambda x: x['alpha'])
        print()
        print(f"Best configuration: {best['config']} with α = {best['alpha']:.4f} ({best['alpha']*100:.1f}%)")

        improvement = (best['alpha'] - 0.039) / 0.039 * 100
        print(f"Improvement over baseline: {improvement:+.1f}%")

if __name__ == "__main__":
    main()
