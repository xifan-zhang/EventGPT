#!/usr/bin/env python3
"""
Train Hidden State Adapter for Cross-Modal Speculative Decoding.

Trains a lightweight bottleneck adapter to align EventGPT decoder hidden states
to Video-LLaVA hidden state space.

Usage:
    python train_hidden_adapter.py \
        --train_data ./hidden_states/hidden_states_train_10q.pt \
        --val_split 0.1 \
        --num_epochs 50 \
        --batch_size 64
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from feasible.feature_alignment.hidden_adapter import (
    HiddenStateAdapter,
    HiddenAdapterConfig,
    create_hidden_adapter,
    create_adapter,
    MultiLayerBottleneckAdapter,
    WideBottleneckAdapter,
    AttentionAdapter,
)


class HiddenStateDataset(Dataset):
    """Dataset for hidden state pairs."""

    def __init__(
        self,
        egpt_hidden: torch.Tensor,
        vl_hidden: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """
        Args:
            egpt_hidden: [N, max_seq, hidden_dim]
            vl_hidden: [N, max_seq, hidden_dim]
            seq_lens: [N] actual sequence lengths
        """
        self.egpt_hidden = egpt_hidden
        self.vl_hidden = vl_hidden
        self.seq_lens = seq_lens

    def __len__(self):
        return len(self.egpt_hidden)

    def __getitem__(self, idx):
        seq_len = self.seq_lens[idx].item()
        return {
            'egpt_hidden': self.egpt_hidden[idx, :seq_len],
            'vl_hidden': self.vl_hidden[idx, :seq_len],
            'seq_len': seq_len,
        }


def collate_fn(batch):
    """Collate function with padding."""
    max_len = max(item['seq_len'] for item in batch)
    hidden_dim = batch[0]['egpt_hidden'].shape[-1]
    batch_size = len(batch)

    egpt_padded = torch.zeros(batch_size, max_len, hidden_dim)
    vl_padded = torch.zeros(batch_size, max_len, hidden_dim)
    mask = torch.zeros(batch_size, max_len)

    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        egpt_padded[i, :seq_len] = item['egpt_hidden']
        vl_padded[i, :seq_len] = item['vl_hidden']
        mask[i, :seq_len] = 1

    return {
        'egpt_hidden': egpt_padded,
        'vl_hidden': vl_padded,
        'mask': mask,
    }


class HiddenAdapterTrainer:
    """Trainer for hidden state adapter (supports all adapter levels)."""

    def __init__(
        self,
        model: nn.Module,  # Any adapter type (L1-L4)
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6,
        )

        # History
        self.history = {
            'train_loss': [],
            'train_cos_sim': [],
            'val_loss': [],
            'val_cos_sim': [],
        }

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_cos_sim = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            egpt_hidden = batch['egpt_hidden'].to(self.device)
            vl_hidden = batch['vl_hidden'].to(self.device)
            mask = batch['mask'].to(self.device)

            self.optimizer.zero_grad()

            losses = self.model.compute_loss(egpt_hidden, vl_hidden, mask)
            loss = losses['total_loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_cos_sim += losses['cos_sim'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cos_sim': f"{losses['cos_sim'].item():.4f}",
            })

        self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'cos_sim': total_cos_sim / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_cos_sim = 0.0
        num_batches = 0

        # For acceptance rate calculation
        all_cos_sims = []

        for batch in self.val_loader:
            egpt_hidden = batch['egpt_hidden'].to(self.device)
            vl_hidden = batch['vl_hidden'].to(self.device)
            mask = batch['mask'].to(self.device)

            losses = self.model.compute_loss(egpt_hidden, vl_hidden, mask)

            total_loss += losses['total_loss'].item()
            total_cos_sim += losses['cos_sim'].item()
            num_batches += 1

            # Per-position cosine similarity
            aligned = self.model(egpt_hidden)
            aligned_norm = F.normalize(aligned, dim=-1)
            vl_norm = F.normalize(vl_hidden, dim=-1)
            cos_sim = (aligned_norm * vl_norm).sum(dim=-1)  # [batch, seq]

            # Flatten and filter by mask
            for i in range(cos_sim.shape[0]):
                valid_len = int(mask[i].sum().item())
                all_cos_sims.extend(cos_sim[i, :valid_len].cpu().tolist())

        # Compute acceptance rates at different thresholds
        cos_tensor = torch.tensor(all_cos_sims)
        accept_80 = (cos_tensor > 0.80).float().mean().item()
        accept_85 = (cos_tensor > 0.85).float().mean().item()
        accept_90 = (cos_tensor > 0.90).float().mean().item()
        accept_95 = (cos_tensor > 0.95).float().mean().item()

        return {
            'loss': total_loss / num_batches,
            'cos_sim': total_cos_sim / num_batches,
            'accept_80': accept_80,
            'accept_85': accept_85,
            'accept_90': accept_90,
            'accept_95': accept_95,
        }

    def train(
        self,
        num_epochs: int,
        save_dir: str,
        early_stopping: int = 10,
    ):
        """Full training loop."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("Training Hidden State Adapter")
        print(f"  Parameters: {self.model.get_num_parameters():,}")
        print(f"  Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"  Val batches: {len(self.val_loader)}")
        print("=" * 70)

        patience = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_cos_sim'].append(train_metrics['cos_sim'])

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, CosSim: {train_metrics['cos_sim']:.4f}")

            # Validate
            if self.val_loader:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_cos_sim'].append(val_metrics['cos_sim'])

                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, CosSim: {val_metrics['cos_sim']:.4f}")
                print(f"  Accept@0.80: {val_metrics['accept_80']:.2%}")
                print(f"  Accept@0.85: {val_metrics['accept_85']:.2%}")
                print(f"  Accept@0.90: {val_metrics['accept_90']:.2%}")
                print(f"  Accept@0.95: {val_metrics['accept_95']:.2%}")

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    patience = 0
                    self.model.save_checkpoint(
                        str(save_path / 'best_model.pt'),
                        self.optimizer,
                        epoch,
                        val_metrics,
                    )
                    print("  ** Saved best model **")
                else:
                    patience += 1

                # Early stopping
                if patience >= early_stopping:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        # Save final model
        self.model.save_checkpoint(
            str(save_path / 'final_model.pt'),
            self.optimizer,
            epoch,
        )

        # Save training history
        self.save_history(save_path)

        # Plot curves
        self.plot_curves(save_path)

        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        print(f"  Saved to: {save_path}")
        print("=" * 70)

    def save_history(self, save_path: Path):
        """Save training history."""
        with open(save_path / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_curves(self, save_path: Path):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curve
        axes[0].plot(self.history['train_loss'], label='Train')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Cosine similarity curve
        axes[1].plot(self.history['train_cos_sim'], label='Train')
        if self.history['val_cos_sim']:
            axes[1].plot(self.history['val_cos_sim'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].set_title('Feature Alignment')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path / 'training_curves.png', dpi=150)
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Hidden State Adapter")
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to hidden_states_train.pt')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to hidden_states_val.pt (optional, will split train if not provided)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio if val_data not provided')
    parser.add_argument('--output_dir', type=str,
                        default='./feasible/feature_alignment/checkpoints/hidden_adapter')

    # Adapter level selection
    parser.add_argument('--adapter_level', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Adapter complexity level: 1=Bottleneck(2M), 2=MultiLayer(8M), 3=Wide(16M), 4=Attention(50M)')

    # L1/L2/L3 parameters
    parser.add_argument('--hidden_dim', type=int, default=4096,
                        help='Hidden dimension (4096 for Vicuna 7B)')
    parser.add_argument('--bottleneck_dim', type=int, default=None,
                        help='Bottleneck dimension (default: 256 for L1/L2, 1024 for L3)')
    parser.add_argument('--num_blocks', type=int, default=3,
                        help='Number of blocks for L2/L3')

    # L4 parameters
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads for L4')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of transformer layers for L4')
    parser.add_argument('--ffn_dim', type=int, default=2048,
                        help='FFN dimension for L4')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--early_stopping', type=int, default=10)
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    data = torch.load(args.train_data)
    egpt_hidden = data['egpt_hidden']
    vl_hidden = data['vl_hidden']
    seq_lens = data['seq_lens']

    print(f"  Loaded {len(egpt_hidden)} samples")
    print(f"  Hidden dim: {egpt_hidden.shape[-1]}")
    print(f"  Max seq len: {egpt_hidden.shape[1]}")

    # Create dataset
    full_dataset = HiddenStateDataset(egpt_hidden, vl_hidden, seq_lens)

    # Split into train/val
    if args.val_data:
        val_data = torch.load(args.val_data)
        train_dataset = full_dataset
        val_dataset = HiddenStateDataset(
            val_data['egpt_hidden'],
            val_data['vl_hidden'],
            val_data['seq_lens'],
        )
    else:
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Create model based on adapter level
    hidden_dim = egpt_hidden.shape[-1]

    # Set default bottleneck dims based on level
    if args.bottleneck_dim is None:
        bottleneck_dim = 1024 if args.adapter_level == 3 else 256
    else:
        bottleneck_dim = args.bottleneck_dim

    print(f"\nCreating L{args.adapter_level} adapter...")
    model = create_adapter(
        level=args.adapter_level,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
    )

    # Create trainer
    trainer = HiddenAdapterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
    )

    # Create output directory with timestamp and level
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"L{args.adapter_level}_{timestamp}"

    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=str(output_dir),
        early_stopping=args.early_stopping,
    )

    # Save config
    config = {
        'adapter_level': args.adapter_level,
        'hidden_dim': hidden_dim,
        'bottleneck_dim': bottleneck_dim,
        'num_blocks': args.num_blocks,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'ffn_dim': args.ffn_dim,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'timestamp': timestamp,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
