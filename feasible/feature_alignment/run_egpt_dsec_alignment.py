"""
Run Feature Alignment on EGPT DSEC Train Dataset
在 EGPT DSEC 训练数据集上运行特征对齐

支持所有 duration: 200ms, 500ms, 1s, 2s, 4s, 5s, 10s, 20s
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import os
import sys
from datetime import datetime

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_alignment import (
    ContrastiveAlignmentConfig,
    ContrastiveAlignmentModule,
    ContrastiveAlignmentTrainer,
    LightweightAlignmentConfig,
    LightweightAlignmentModule,
    LightweightAlignmentTrainer,
    auto_select_alignment_strategy,
    print_strategy_recommendation,
)


class EGPTDSECPairedDataset(Dataset):
    """
    EGPT DSEC Paired Dataset for Feature Alignment

    Loads paired event images and video frames from DSEC dataset
    """

    def __init__(
        self,
        data_dir: str,
        event_transform=None,
        video_transform=None,
        max_samples: int = None,
        cache_in_memory: bool = False,
        image_size: tuple = (224, 224)
    ):
        """
        Args:
            data_dir: Path to dataset (e.g., my_egpt_dsec_seq_200ms)
            event_transform: Transform for event data
            video_transform: Transform for video frames
            max_samples: Maximum number of samples to load
            cache_in_memory: Whether to cache all data in memory
            image_size: Target image size for resizing
        """
        self.data_dir = Path(data_dir)
        self.event_image_dir = self.data_dir / "event_image"
        self.video_dir = self.data_dir / "video"
        self.event_transform = event_transform
        self.video_transform = video_transform
        self.cache_in_memory = cache_in_memory
        self.image_size = image_size

        # Collect all sample paths
        self.samples = []
        sequences = sorted([d for d in self.event_image_dir.iterdir() if d.is_dir()])

        for seq in sequences:
            seq_name = seq.name
            # Get unique frame IDs from event images (format: 000000_0.png, 000000_1.png, ...)
            event_files = sorted(seq.glob("*.png"))
            frame_ids = sorted(set(f.stem.split('_')[0] for f in event_files))

            for frame_id in frame_ids:
                video_frame_dir = self.video_dir / seq_name / frame_id
                event_images = sorted(seq.glob(f"{frame_id}_*.png"))

                if video_frame_dir.exists() and event_images:
                    self.samples.append({
                        'event_images': [str(p) for p in event_images],
                        'video_dir': str(video_frame_dir),
                        'seq_name': seq_name,
                        'frame_id': frame_id
                    })

        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

        # Cache if requested
        self.cached_data = None
        if cache_in_memory:
            self._cache_all_data()

    def _cache_all_data(self):
        """Cache all data in memory"""
        print("Caching data in memory...")
        self.cached_data = []
        for i in tqdm(range(len(self.samples)), desc="Caching"):
            self.cached_data.append(self._load_sample(i))

    def _load_sample(self, idx: int):
        """Load a single sample"""
        sample = self.samples[idx]

        # Load event images and stack them (use middle one for simplicity)
        event_images = sample['event_images']
        mid_idx = len(event_images) // 2
        event_img = Image.open(event_images[mid_idx]).convert('RGB')
        event_img = event_img.resize(self.image_size, Image.BILINEAR)
        event_tensor = torch.from_numpy(np.array(event_img)).float()
        # Normalize to [0, 1] and change to CHW format
        event_tensor = event_tensor.permute(2, 0, 1) / 255.0

        if self.event_transform:
            event_tensor = self.event_transform(event_tensor)

        # Load video frame (use middle frame from sequence)
        video_dir = Path(sample['video_dir'])
        frame_files = sorted(video_dir.glob("*.png"))

        if frame_files:
            # Use middle frame
            mid_idx = len(frame_files) // 2
            frame_path = frame_files[mid_idx]
            frame = Image.open(frame_path).convert('RGB')
            frame = frame.resize(self.image_size, Image.BILINEAR)
            frame_tensor = torch.from_numpy(np.array(frame)).float()
            # Normalize to [0, 1] and change to CHW format
            frame_tensor = frame_tensor.permute(2, 0, 1) / 255.0
        else:
            # Fallback if no frames found
            frame_tensor = torch.zeros(3, *self.image_size)

        if self.video_transform:
            frame_tensor = self.video_transform(frame_tensor)

        return {
            'event_features': event_tensor,
            'target_features': frame_tensor,
            'metadata': {
                'seq_name': sample['seq_name'],
                'frame_id': sample['frame_id']
            }
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cached_data is not None:
            return self.cached_data[idx]
        return self._load_sample(idx)


class PreExtractedFeatureDataset(Dataset):
    """
    Dataset for pre-extracted features
    """

    def __init__(
        self,
        event_features: torch.Tensor,
        target_features: torch.Tensor
    ):
        assert len(event_features) == len(target_features)
        self.event_features = event_features
        self.target_features = target_features

    def __len__(self):
        return len(self.event_features)

    def __getitem__(self, idx):
        return {
            'event_features': self.event_features[idx],
            'target_features': self.target_features[idx]
        }

    @classmethod
    def from_directory(cls, feature_dir: str):
        """Load features from directory"""
        feature_dir = Path(feature_dir)
        event_features = torch.load(feature_dir / 'event_features.pt')
        target_features = torch.load(feature_dir / 'target_features.pt')
        return cls(event_features, target_features)


def extract_features(
    dataset: EGPTDSECPairedDataset,
    event_encoder: nn.Module,
    target_encoder: nn.Module,
    device: str = 'cuda',
    batch_size: int = 32,
    save_dir: str = None
):
    """
    Extract features from event and video data using encoders

    Args:
        dataset: The raw dataset
        event_encoder: Event encoder (e.g., EventGPT visual encoder)
        target_encoder: Target encoder (e.g., CLIP visual encoder)
        device: Device to use
        batch_size: Batch size for extraction
        save_dir: Directory to save extracted features

    Returns:
        event_features, target_features
    """
    event_encoder = event_encoder.to(device).eval()
    target_encoder = target_encoder.to(device).eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_event_features = []
    all_target_features = []

    print("Extracting features...")
    with torch.no_grad():
        for batch in tqdm(loader):
            event_data = batch['event_features'].to(device)
            video_data = batch['target_features'].to(device)

            # Extract features
            event_feat = event_encoder(event_data)
            target_feat = target_encoder(video_data)

            # Handle different output formats
            if isinstance(event_feat, tuple):
                event_feat = event_feat[0]
            if isinstance(target_feat, tuple):
                target_feat = target_feat[0]

            # If features have sequence dimension, take mean or CLS token
            if event_feat.dim() == 3:
                event_feat = event_feat[:, 0]  # CLS token
            if target_feat.dim() == 3:
                target_feat = target_feat[:, 0]

            all_event_features.append(event_feat.cpu())
            all_target_features.append(target_feat.cpu())

    event_features = torch.cat(all_event_features, dim=0)
    target_features = torch.cat(all_target_features, dim=0)

    print(f"Extracted features: event {event_features.shape}, target {target_features.shape}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(event_features, save_dir / 'event_features.pt')
        torch.save(target_features, save_dir / 'target_features.pt')
        print(f"Saved features to {save_dir}")

    return event_features, target_features


def run_alignment_on_duration(
    duration: str,
    data_root: str,
    output_dir: str,
    strategy: str = 'contrastive',
    event_dim: int = 768,
    target_dim: int = 1024,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    use_preextracted: bool = True
):
    """
    Run alignment training on a specific duration

    Args:
        duration: Duration string (e.g., '200ms', '1s', '10s')
        data_root: Root data directory
        output_dir: Output directory for checkpoints
        strategy: 'contrastive' or 'lightweight'
        event_dim: Event feature dimension
        target_dim: Target feature dimension
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        use_preextracted: If True, look for pre-extracted features
    """
    print("="*70)
    print(f"Running Feature Alignment for Duration: {duration}")
    print("="*70)

    data_dir = Path(data_root) / f"my_egpt_dsec_seq_{duration}"
    output_path = Path(output_dir) / f"alignment_{duration}"
    output_path.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return None

    # Check for pre-extracted features
    feature_dir = output_path / 'features'
    if use_preextracted and (feature_dir / 'event_features.pt').exists():
        print("Loading pre-extracted features...")
        train_dataset = PreExtractedFeatureDataset.from_directory(str(feature_dir))
    else:
        # Load raw dataset
        print("Loading raw dataset...")
        raw_dataset = EGPTDSECPairedDataset(str(data_dir))

        # For now, we'll use the raw data directly without encoder extraction
        # This assumes the event_npy files already contain extracted features
        # If not, you'll need to provide the encoders

        # Collect all features
        event_features = []
        target_features = []

        print("Processing samples...")
        for i in tqdm(range(len(raw_dataset))):
            sample = raw_dataset[i]
            event_feat = sample['event_features']
            target_feat = sample['target_features']

            # Flatten if needed
            if event_feat.dim() > 1:
                event_feat = event_feat.flatten()
            if target_feat.dim() > 1:
                target_feat = target_feat.flatten()

            event_features.append(event_feat)
            target_features.append(target_feat)

        event_features = torch.stack(event_features)
        target_features = torch.stack(target_features)

        # Save features for reuse
        feature_dir.mkdir(parents=True, exist_ok=True)
        torch.save(event_features, feature_dir / 'event_features.pt')
        torch.save(target_features, feature_dir / 'target_features.pt')

        train_dataset = PreExtractedFeatureDataset(event_features, target_features)

    print(f"Dataset size: {len(train_dataset)}")

    # Get actual feature dimensions
    sample = train_dataset[0]
    actual_event_dim = sample['event_features'].numel()
    actual_target_dim = sample['target_features'].numel()

    print(f"Event feature dim: {actual_event_dim}")
    print(f"Target feature dim: {actual_target_dim}")

    # Split into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

    # Create alignment module
    if strategy == 'contrastive':
        config = ContrastiveAlignmentConfig(
            event_dim=actual_event_dim,
            target_dim=actual_target_dim,
            hidden_dim=min(512, actual_event_dim),
            num_layers=2,
            temperature=0.07
        )
        alignment = ContrastiveAlignmentModule(config)

        trainer = ContrastiveAlignmentTrainer(
            alignment,
            train_subset,
            val_subset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )
    else:
        config = LightweightAlignmentConfig(
            event_dim=actual_event_dim,
            target_dim=actual_target_dim,
            hidden_dim=min(512, actual_event_dim),
            num_layers=2,
            adapter_type='mlp'
        )
        alignment = LightweightAlignmentModule(config)

        trainer = LightweightAlignmentTrainer(
            alignment,
            train_subset,
            val_subset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )

    # Train
    save_path = str(output_path / f'{strategy}_alignment.pt')
    trainer.train(
        num_epochs=num_epochs,
        save_path=save_path,
        early_stopping_patience=10
    )

    # Save training info
    info = {
        'duration': duration,
        'strategy': strategy,
        'event_dim': actual_event_dim,
        'target_dim': actual_target_dim,
        'train_size': len(train_subset),
        'val_size': len(val_subset),
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'timestamp': datetime.now().isoformat()
    }

    import json
    with open(output_path / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nAlignment saved to: {save_path}")

    return alignment


def run_all_durations(
    data_root: str,
    output_dir: str,
    durations: list = None,
    strategy: str = 'contrastive',
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    """
    Run alignment on all durations
    """
    if durations is None:
        durations = ['200ms', '500ms', '1s', '2s', '4s', '5s', '10s', '20s']

    results = {}

    for duration in durations:
        print(f"\n{'#'*70}")
        print(f"# Processing Duration: {duration}")
        print(f"{'#'*70}\n")

        try:
            alignment = run_alignment_on_duration(
                duration=duration,
                data_root=data_root,
                output_dir=output_dir,
                strategy=strategy,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device
            )
            results[duration] = 'success'
        except Exception as e:
            print(f"Error processing {duration}: {e}")
            import traceback
            traceback.print_exc()
            results[duration] = f'error: {str(e)}'

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for duration, status in results.items():
        print(f"  {duration}: {status}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Feature Alignment on EGPT DSEC Dataset')
    parser.add_argument('--data_root', type=str, default='/home/ps/Documents/code/EventGPT/data',
                        help='Root directory containing the datasets')
    parser.add_argument('--output_dir', type=str, default='/home/ps/Documents/code/EventGPT/feasible/feature_alignment/checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--durations', type=str, nargs='+',
                        default=['200ms', '500ms', '1s', '2s', '4s', '5s', '10s', '20s'],
                        help='Durations to process')
    parser.add_argument('--strategy', type=str, default='contrastive',
                        choices=['contrastive', 'lightweight'],
                        help='Alignment strategy')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    print("="*70)
    print("EGPT DSEC Feature Alignment")
    print("="*70)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Durations: {args.durations}")
    print(f"Strategy: {args.strategy}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print("="*70)

    results = run_all_durations(
        data_root=args.data_root,
        output_dir=args.output_dir,
        durations=args.durations,
        strategy=args.strategy,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
