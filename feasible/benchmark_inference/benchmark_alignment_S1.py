#!/usr/bin/env python3
"""
Benchmark Alignment Adapter on 1s Dataset
==========================================

Tests the trained lightweight adapter on the 1s test set and measures
acceptance rate metrics for speculative decoding feasibility.

Usage:
    python benchmark_alignment_S1.py --output_json benchmark_results_S1.json
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# Add paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'feasible', 'feature_alignment'))
sys.path.insert(0, os.path.join(ROOT, 'feasible'))

from feature_alignment import LightweightAlignmentModule


def load_adapter(checkpoint_path: str, device: str = 'cpu'):
    """Load trained lightweight adapter."""
    print(f"Loading adapter from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    model = LightweightAlignmentModule(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Adapter loaded: {config.event_dim} -> {config.target_dim}")
    return model, config


def compute_acceptance_metrics(cos_sim: torch.Tensor, thresholds: list = None):
    """Compute acceptance rate at various thresholds."""
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

    metrics = {}
    for t in thresholds:
        acceptance = (cos_sim > t).float().mean().item() * 100
        metrics[f'acceptance_at_{t}'] = acceptance

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Benchmark alignment adapter on 1s dataset')
    parser.add_argument('--checkpoint', type=str,
                        default='./feasible/feature_alignment/checkpoints/alignment_1s/lightweight_alignment.pt',
                        help='Path to adapter checkpoint')
    parser.add_argument('--features_dir', type=str,
                        default='./feasible/feature_alignment/checkpoints/alignment_1s/features',
                        help='Directory containing pre-extracted features')
    parser.add_argument('--output_json', type=str,
                        default='./feasible/benchmark_inference/benchmark_results_S1.json',
                        help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')

    args = parser.parse_args()

    # Load adapter
    model, config = load_adapter(args.checkpoint, args.device)

    # Load features
    print(f"Loading features from {args.features_dir}...")
    event_features = torch.load(os.path.join(args.features_dir, 'event_features.pt'))
    target_features = torch.load(os.path.join(args.features_dir, 'target_features.pt'))

    total_samples = len(event_features)
    print(f"Total samples: {total_samples}")
    print(f"Feature dim: {event_features.shape[1]}")

    # Recreate train/val split (same seed as training)
    torch.manual_seed(42)
    train_size = int(0.8 * total_samples)
    indices = torch.randperm(total_samples)
    test_indices = indices[train_size:]

    test_event = event_features[test_indices]
    test_target = target_features[test_indices]
    test_size = len(test_event)

    print(f"Test set size: {test_size}")

    # Compute metrics BEFORE alignment
    print("\nComputing metrics before alignment...")
    event_norm = F.normalize(test_event.float(), dim=1)
    target_norm = F.normalize(test_target.float(), dim=1)
    cos_sim_before = (event_norm * target_norm).sum(dim=1)

    before_metrics = {
        'mean_cosine_similarity': cos_sim_before.mean().item(),
        'std_cosine_similarity': cos_sim_before.std().item(),
        'min_cosine_similarity': cos_sim_before.min().item(),
        'max_cosine_similarity': cos_sim_before.max().item(),
    }
    before_metrics.update(compute_acceptance_metrics(cos_sim_before))

    # Compute metrics AFTER alignment
    print("Computing metrics after alignment...")
    aligned_features = []

    with torch.no_grad():
        for i in tqdm(range(0, test_size, args.batch_size), desc="Aligning"):
            batch = test_event[i:i+args.batch_size].float()
            if args.device == 'cuda':
                batch = batch.cuda()
            aligned = model(batch)
            aligned_features.append(aligned.cpu())

    aligned_feat = torch.cat(aligned_features, dim=0)
    aligned_norm = F.normalize(aligned_feat, dim=1)
    cos_sim_after = (aligned_norm * target_norm).sum(dim=1)

    after_metrics = {
        'mean_cosine_similarity': cos_sim_after.mean().item(),
        'std_cosine_similarity': cos_sim_after.std().item(),
        'min_cosine_similarity': cos_sim_after.min().item(),
        'max_cosine_similarity': cos_sim_after.max().item(),
    }
    after_metrics.update(compute_acceptance_metrics(cos_sim_after))

    # Compute percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = {}
    for p in percentiles:
        percentile_values[f'P{p}'] = torch.quantile(cos_sim_after, p/100).item()

    # Build results
    results = {
        'dataset': '1s',
        'checkpoint': args.checkpoint,
        'test_samples': test_size,
        'feature_dim': event_features.shape[1],
        'before_alignment': before_metrics,
        'after_alignment': after_metrics,
        'percentiles_after': percentile_values,
        'improvement': {
            'cosine_similarity_gain': after_metrics['mean_cosine_similarity'] - before_metrics['mean_cosine_similarity'],
            'relative_improvement': (after_metrics['mean_cosine_similarity'] - before_metrics['mean_cosine_similarity']) / before_metrics['mean_cosine_similarity'] * 100,
        },
        'config': {
            'event_dim': config.event_dim,
            'target_dim': config.target_dim,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'adapter_type': config.adapter_type,
        }
    }

    # Save results
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_json}")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS (1s Dataset - Test Set)")
    print("="*60)

    print(f"\nTest samples: {test_size}")

    print(f"\nBEFORE Alignment:")
    print(f"  Mean Cosine Similarity: {before_metrics['mean_cosine_similarity']:.4f}")
    print(f"  Acceptance @ 0.8: {before_metrics['acceptance_at_0.8']:.1f}%")

    print(f"\nAFTER Alignment:")
    print(f"  Mean Cosine Similarity: {after_metrics['mean_cosine_similarity']:.4f}")
    print(f"  Acceptance @ 0.8: {after_metrics['acceptance_at_0.8']:.1f}%")

    print(f"\nImprovement:")
    print(f"  Cosine Sim Gain: +{results['improvement']['cosine_similarity_gain']:.4f}")
    print(f"  Relative: +{results['improvement']['relative_improvement']:.1f}%")

    print(f"\nAcceptance Rate by Threshold:")
    print(f"  {'Threshold':<12} {'Before':<10} {'After':<10} {'Gain':<10}")
    print(f"  {'-'*42}")
    for t in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]:
        before = before_metrics[f'acceptance_at_{t}']
        after = after_metrics[f'acceptance_at_{t}']
        print(f"  {t:<12} {before:<10.1f} {after:<10.1f} {after-before:+.1f}")

    print(f"\nPercentiles (After Alignment):")
    for p, val in percentile_values.items():
        print(f"  {p}: {val:.4f}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
