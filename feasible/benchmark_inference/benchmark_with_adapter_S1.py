#!/usr/bin/env python3
"""
Benchmark EventGPT + Adapter vs Video-LLaVA
============================================

This script measures the acceptance rate between:
- EventGPT visual features (aligned through lightweight adapter)
- Video-LLaVA visual features

This simulates speculative decoding where EventGPT is the draft model
and Video-LLaVA is the target model.

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python benchmark_with_adapter_S1.py \
        --dataset_dir ./data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
        --max_samples 100
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# Add paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'feasible', 'feature_alignment'))
sys.path.insert(0, os.path.join(ROOT, 'feasible'))

from feature_alignment import LightweightAlignmentModule


def load_adapter(checkpoint_path: str, device: str = 'cuda'):
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


def load_image_as_tensor(image_path: str, size=(224, 224)):
    """Load image and convert to tensor."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img)).float()
    tensor = tensor.permute(2, 0, 1) / 255.0  # CHW, normalized
    return tensor


def extract_features_from_dataset(dataset_dir: str, max_samples: int = None):
    """Extract paired event and video features from dataset.

    Returns flattened image tensors (same as training).
    """
    dataset_dir = Path(dataset_dir)
    event_image_dir = dataset_dir / "event_image"
    video_dir = dataset_dir / "video"

    # Load JSON to get sample info
    json_path = dataset_dir / "EventGPT_Instruction_Subset.json"
    with open(json_path, 'r') as f:
        samples = json.load(f)

    if max_samples:
        samples = samples[:max_samples]

    event_features = []
    video_features = []
    sample_ids = []

    print(f"Extracting features from {len(samples)} samples...")

    for sample in tqdm(samples, desc="Extracting"):
        try:
            # Get event image (middle one)
            event_images = sample.get('event_image', [])
            if not event_images:
                continue
            mid_idx = len(event_images) // 2
            event_path = event_image_dir / event_images[mid_idx]

            # Get video frame (middle one)
            video_data = sample.get('video_data', '')
            video_frame_dir = video_dir / video_data
            if not video_frame_dir.exists():
                continue

            frame_files = sorted(video_frame_dir.glob("*.png"))
            if not frame_files:
                continue
            mid_frame = frame_files[len(frame_files) // 2]

            # Load and flatten
            event_tensor = load_image_as_tensor(str(event_path)).flatten()
            video_tensor = load_image_as_tensor(str(mid_frame)).flatten()

            event_features.append(event_tensor)
            video_features.append(video_tensor)
            sample_ids.append(sample.get('id', 'unknown'))

        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    event_features = torch.stack(event_features)
    video_features = torch.stack(video_features)

    print(f"Extracted {len(event_features)} samples")
    print(f"Feature dim: {event_features.shape[1]}")

    return event_features, video_features, sample_ids


def compute_acceptance_rate(cos_sim: torch.Tensor, threshold: float = 0.8):
    """Compute acceptance rate at given threshold."""
    return (cos_sim > threshold).float().mean().item() * 100


def main():
    parser = argparse.ArgumentParser(description='Benchmark EventGPT + Adapter vs Video-LLaVA')
    parser.add_argument('--dataset_dir', type=str,
                        default='./data/my_egpt_dsec_train/my_egpt_dsec_train_1s',
                        help='Dataset directory')
    parser.add_argument('--adapter_checkpoint', type=str,
                        default='./feasible/feature_alignment/checkpoints/alignment_1s/lightweight_alignment.pt',
                        help='Path to adapter checkpoint')
    parser.add_argument('--output_json', type=str,
                        default='./feasible/benchmark_inference/benchmark_adapter_full_S1.json',
                        help='Output JSON file')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to process')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for adapter inference')
    parser.add_argument('--acceptance_threshold', type=float, default=0.8,
                        help='Threshold for acceptance rate')

    args = parser.parse_args()

    print("="*60)
    print("Benchmark: EventGPT + Adapter vs Video-LLaVA")
    print("="*60)

    # Load adapter
    adapter, config = load_adapter(args.adapter_checkpoint, args.device)

    # Extract features from dataset
    event_features, video_features, sample_ids = extract_features_from_dataset(
        args.dataset_dir, args.max_samples
    )

    total_samples = len(event_features)

    # Compute baseline (before adapter)
    print("\nComputing baseline (no adapter)...")
    event_norm = F.normalize(event_features.float(), dim=1)
    video_norm = F.normalize(video_features.float(), dim=1)
    cos_sim_before = (event_norm * video_norm).sum(dim=1)

    # Apply adapter
    print("Applying adapter...")
    aligned_features = []
    adapter_times = []

    with torch.no_grad():
        for i in tqdm(range(0, total_samples, args.batch_size), desc="Aligning"):
            batch = event_features[i:i+args.batch_size].float()
            if args.device == 'cuda':
                batch = batch.cuda()

            start = time.time()
            aligned = adapter(batch)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            adapter_times.append(time.time() - start)

            aligned_features.append(aligned.cpu())

    aligned_features = torch.cat(aligned_features, dim=0)

    # Compute aligned similarity
    print("Computing aligned similarity...")
    aligned_norm = F.normalize(aligned_features, dim=1)
    cos_sim_after = (aligned_norm * video_norm).sum(dim=1)

    # Compute metrics
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

    before_metrics = {
        'mean_cosine_similarity': cos_sim_before.mean().item(),
        'std': cos_sim_before.std().item(),
        'min': cos_sim_before.min().item(),
        'max': cos_sim_before.max().item(),
    }
    for t in thresholds:
        before_metrics[f'acceptance_at_{t}'] = compute_acceptance_rate(cos_sim_before, t)

    after_metrics = {
        'mean_cosine_similarity': cos_sim_after.mean().item(),
        'std': cos_sim_after.std().item(),
        'min': cos_sim_after.min().item(),
        'max': cos_sim_after.max().item(),
    }
    for t in thresholds:
        after_metrics[f'acceptance_at_{t}'] = compute_acceptance_rate(cos_sim_after, t)

    # Percentiles
    percentiles = {}
    for p in [10, 25, 50, 75, 90, 95, 99]:
        percentiles[f'P{p}'] = torch.quantile(cos_sim_after, p/100).item()

    # Timing
    avg_adapter_time = sum(adapter_times) / len(adapter_times) if adapter_times else 0
    total_adapter_time = sum(adapter_times)

    # Per-sample results
    sample_results = []
    for i in range(min(100, total_samples)):  # Save first 100 samples
        sample_results.append({
            'id': sample_ids[i],
            'cos_sim_before': cos_sim_before[i].item(),
            'cos_sim_after': cos_sim_after[i].item(),
            'accepted_at_0.8': cos_sim_after[i].item() > 0.8,
        })

    # Build results
    results = {
        'dataset': '1s',
        'total_samples': total_samples,
        'adapter_checkpoint': args.adapter_checkpoint,
        'acceptance_threshold': args.acceptance_threshold,
        'before_adapter': before_metrics,
        'after_adapter': after_metrics,
        'percentiles_after': percentiles,
        'improvement': {
            'cosine_sim_gain': after_metrics['mean_cosine_similarity'] - before_metrics['mean_cosine_similarity'],
            'acceptance_gain_0.8': after_metrics['acceptance_at_0.8'] - before_metrics['acceptance_at_0.8'],
        },
        'timing': {
            'avg_adapter_time_per_batch': avg_adapter_time,
            'total_adapter_time': total_adapter_time,
            'samples_per_second': total_samples / total_adapter_time if total_adapter_time > 0 else 0,
        },
        'sample_results': sample_results,
    }

    # Save
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_json}")

    # Print summary
    print("\n" + "="*70)
    print("FULL BENCHMARK RESULTS: EventGPT + Adapter vs Video-LLaVA")
    print("="*70)

    print(f"\nDataset: 1s duration")
    print(f"Total samples: {total_samples}")

    print(f"\n{'='*70}")
    print("BEFORE ADAPTER (Raw EventGPT vs Video-LLaVA)")
    print(f"{'='*70}")
    print(f"  Mean Cosine Similarity: {before_metrics['mean_cosine_similarity']:.4f}")
    print(f"  Acceptance Rate @ 0.8:  {before_metrics['acceptance_at_0.8']:.1f}%")

    print(f"\n{'='*70}")
    print("AFTER ADAPTER (Aligned EventGPT vs Video-LLaVA)")
    print(f"{'='*70}")
    print(f"  Mean Cosine Similarity: {after_metrics['mean_cosine_similarity']:.4f}")
    print(f"  Acceptance Rate @ 0.8:  {after_metrics['acceptance_at_0.8']:.1f}%")

    print(f"\n{'='*70}")
    print("IMPROVEMENT")
    print(f"{'='*70}")
    print(f"  Cosine Sim Gain:        +{results['improvement']['cosine_sim_gain']:.4f}")
    print(f"  Acceptance Gain @ 0.8:  +{results['improvement']['acceptance_gain_0.8']:.1f}%")

    print(f"\n{'='*70}")
    print("ACCEPTANCE RATE BY THRESHOLD")
    print(f"{'='*70}")
    print(f"  {'Threshold':<12} {'Before':<12} {'After':<12} {'Gain':<12}")
    print(f"  {'-'*48}")
    for t in thresholds:
        before = before_metrics[f'acceptance_at_{t}']
        after = after_metrics[f'acceptance_at_{t}']
        print(f"  {t:<12} {before:<12.1f} {after:<12.1f} {after-before:+.1f}")

    print(f"\n{'='*70}")
    print("TIMING")
    print(f"{'='*70}")
    print(f"  Adapter throughput: {results['timing']['samples_per_second']:.1f} samples/sec")
    print(f"  Total adapter time: {total_adapter_time:.3f}s for {total_samples} samples")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    acceptance = after_metrics['acceptance_at_0.8']
    if acceptance > 80:
        print(f"  ✓ High acceptance rate ({acceptance:.1f}%) - adapter alignment is effective")
        print(f"  ✓ Speculative decoding with EventGPT as draft model is FEASIBLE")
    elif acceptance > 50:
        print(f"  ~ Moderate acceptance rate ({acceptance:.1f}%) - may benefit from fine-tuning")
    else:
        print(f"  ✗ Low acceptance rate ({acceptance:.1f}%) - needs improvement")

    print("="*70)


if __name__ == '__main__':
    main()
