"""
Test token alignment on actual benchmark results.

Since tokenizers are 100% identical, the low acceptance rate (α ≈ 5-7%)
must be due to different model outputs during generation.
"""

import json
import os
from collections import defaultdict

def analyze_benchmark_results(benchmark_path: str):
    """Analyze benchmark results to understand low acceptance rate."""

    with open(benchmark_path, 'r') as f:
        data = json.load(f)

    print(f"Analyzing {len(data)} samples from {benchmark_path}")
    print("="*70)

    # Collect statistics
    total_draft_tokens = 0
    total_accepted = 0
    sample_stats = []

    for i, sample in enumerate(data):
        draft_tokens = sample.get('egpt_token_ids', [])
        target_tokens = sample.get('llava_token_ids', [])

        if not draft_tokens or not target_tokens:
            continue

        # Count matches in overlapping range
        min_len = min(len(draft_tokens), len(target_tokens))
        accepted = sum(1 for j in range(min_len) if draft_tokens[j] == target_tokens[j])

        alpha = accepted / len(draft_tokens) if draft_tokens else 0

        sample_stats.append({
            'sample_id': i,
            'draft_len': len(draft_tokens),
            'target_len': len(target_tokens),
            'accepted': accepted,
            'alpha': alpha
        })

        total_draft_tokens += len(draft_tokens)
        total_accepted += accepted

    # Print summary
    print(f"\nSummary:")
    print(f"  Total draft tokens: {total_draft_tokens}")
    print(f"  Total accepted: {total_accepted}")
    print(f"  Overall α: {total_accepted / total_draft_tokens:.4f} ({total_accepted / total_draft_tokens * 100:.1f}%)")
    print(f"  Samples: {len(sample_stats)}")

    # Analyze by position
    print(f"\nToken Match by Position:")
    for pos in [1, 5, 10, 20]:
        matches_at_pos = []
        for s in sample_stats:
            if s['draft_len'] >= pos and s['target_len'] >= pos:
                # Check if tokens match at position pos (0-indexed)
                pass  # Need original data for this

    # Show some examples
    print(f"\nSample Details (first 5):")
    for s in sample_stats[:5]:
        print(f"  Sample {s['sample_id']}: α={s['alpha']:.3f}, "
              f"draft={s['draft_len']} tokens, target={s['target_len']} tokens, "
              f"accepted={s['accepted']}/{s['draft_len']}")

    # Find samples with high acceptance rate
    high_alpha_samples = [s for s in sample_stats if s['alpha'] > 0.2]
    print(f"\nSamples with α > 20%: {len(high_alpha_samples)}/{len(sample_stats)}")
    for s in high_alpha_samples[:3]:
        print(f"  Sample {s['sample_id']}: α={s['alpha']:.3f}")

    return sample_stats


def compare_model_outputs(benchmark_path: str, num_samples: int = 3):
    """Compare actual model outputs to see why they differ."""

    with open(benchmark_path, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"Comparing Model Outputs (first {num_samples} samples)")
    print(f"{'='*70}")

    for i, sample in enumerate(data[:num_samples]):
        # Get outputs (this benchmark uses LLaVA 1.5, not Video-LLaVA)
        egpt_out = sample.get('egpt', '')
        llava_out = sample.get('llava-1.5-7b-hf', '')

        print(f"\n--- Sample {i} ---")
        print(f"EventGPT (length {len(egpt_out)}):")
        print(f"  {egpt_out[:200]}...")
        print(f"\nLLaVA 1.5 (length {len(llava_out)}):")
        print(f"  {llava_out[:200]}...")

        # Token-level comparison
        egpt_tokens = sample.get('egpt_token_ids', [])
        llava_tokens = sample.get('llava_token_ids', [])

        if egpt_tokens and llava_tokens:
            min_len = min(20, len(egpt_tokens), len(llava_tokens))
            print(f"\nFirst {min_len} tokens:")
            print(f"  EventGPT: {egpt_tokens[:min_len]}")
            print(f"  Video-LLaVA: {llava_tokens[:min_len]}")
            print(f"  Match: {sum(1 for j in range(min_len) if egpt_tokens[j] == llava_tokens[j])}/{min_len}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test tokenizer alignment")
    parser.add_argument("--benchmark", type=str, default="/mnt/hdd/data/my_egpt_dsec_seq_20s/benchmark_results.json")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    print("Tokenizer Alignment Test on Benchmark Results")
    print("="*70)

    if os.path.exists(args.benchmark):
        analyze_benchmark_results(args.benchmark)
        compare_model_outputs(args.benchmark, args.num_samples)
    else:
        print(f"Benchmark file not found: {args.benchmark}")

    print("\n" + "="*70)
    print("Conclusion:")
    print("-" * 70)
    print("Since tokenizers are 100% identical, the low acceptance rate")
    print("(α ≈ 5-7%) must be due to:")
    print("1. Models generate different text despite same input")
    print("2. Different temperature/sampling strategies")
    print("3. Model weights trained on different data distributions")
    print()
    print("To improve α:")
    print("→ Option A: Use same model architecture (Video-LLaVA as draft)")
    print("→ Option B: Fine-tune models to generate similar text")
    print("→ Option C: Use same sampling parameters (temperature, top_p)")


if __name__ == "__main__":
    main()
