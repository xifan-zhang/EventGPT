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
    """Dataset for hidden state pairs (in-memory)."""

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


class ChunkedTrainLoader:
    """
    Memory-efficient training data loader for chunked hidden states.

    Loads ONE chunk at a time (~1.6GB), shuffles within it, yields batches,
    then moves to next chunk. Chunk order is also shuffled each epoch.

    Uses a background prefetch thread to load the next chunk while GPU
    processes the current one, hiding HDD I/O latency.

    Memory: ~4.8GB (current chunk + prefetched chunk + batch on GPU) instead of 80GB.
    """

    def __init__(self, data_dir: str, batch_size: int = 64, collate_fn=None):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        with open(self.data_dir / 'index.json') as f:
            self.index = json.load(f)

        self.chunks_dir = self.data_dir / 'chunks'
        self.chunk_infos = self.index['chunks']
        self.total_samples = self.index['total_samples']

        # Compute total batches for progress bar
        self.num_batches = 0
        for ci in self.chunk_infos:
            self.num_batches += (ci['n_samples'] + batch_size - 1) // batch_size

        print(f"  ChunkedTrainLoader: {self.total_samples} samples in {len(self.chunk_infos)} chunks")
        print(f"  Batch size: {batch_size}, ~{self.num_batches} batches/epoch")
        print(f"  Memory: ~3.2GB (current + prefetch) instead of {self.total_samples * 4096 * 4 * 2 / 1e9:.0f}GB all at once")

    def __len__(self):
        return self.num_batches

    def _load_chunk(self, chunk_idx):
        """Load a single chunk from disk."""
        chunk_path = self.chunks_dir / self.chunk_infos[chunk_idx]['path']
        return torch.load(chunk_path, map_location='cpu')

    def __iter__(self):
        """Iterate: shuffle chunk order, within each chunk shuffle samples.
        Prefetches the next chunk in a background thread."""
        import random
        from concurrent.futures import ThreadPoolExecutor

        chunk_order = list(range(len(self.chunk_infos)))
        random.shuffle(chunk_order)

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Start prefetching the first chunk
            future = executor.submit(self._load_chunk, chunk_order[0])

            for i, ci in enumerate(chunk_order):
                # Wait for the current chunk
                chunk = future.result()

                # Start prefetching the next chunk
                if i + 1 < len(chunk_order):
                    future = executor.submit(self._load_chunk, chunk_order[i + 1])

                n = chunk['seq_lens'].shape[0]
                egpt = chunk['egpt_hidden']
                vl = chunk['vl_hidden']
                seq_lens = chunk['seq_lens']

                # Shuffle within chunk
                perm = torch.randperm(n)

                # Yield batches
                for start in range(0, n, self.batch_size):
                    end = min(start + self.batch_size, n)
                    indices = perm[start:end]

                    batch_items = []
                    for idx in indices:
                        sl = seq_lens[idx].item()
                        batch_items.append({
                            'egpt_hidden': egpt[idx, :sl],
                            'vl_hidden': vl[idx, :sl],
                            'seq_len': sl,
                        })

                    if self.collate_fn:
                        yield self.collate_fn(batch_items)
                    else:
                        yield batch_items

                # Free chunk memory
                del chunk, egpt, vl, seq_lens


class ChunkedValLoader:
    """Stream val chunks one at a time for memory-efficient validation.

    Like ChunkedTrainLoader but sequential (no shuffling).
    Uses background prefetch thread to hide HDD I/O latency.
    """

    def __init__(self, data_dir, batch_size, collate_fn=None):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        with open(self.data_dir / 'index.json') as f:
            self.index = json.load(f)

        self.chunks_dir = self.data_dir / 'chunks'
        self.chunk_infos = self.index['chunks']
        self.total_samples = self.index['total_samples']

        self.num_batches = 0
        for ci in self.chunk_infos:
            self.num_batches += (ci['n_samples'] + batch_size - 1) // batch_size

        print(f"  ChunkedValLoader: {self.total_samples} samples in {len(self.chunk_infos)} chunks")
        print(f"  Batch size: {batch_size}, ~{self.num_batches} batches/epoch")

    def __len__(self):
        return self.num_batches

    def _load_chunk(self, chunk_idx):
        """Load a single chunk from disk."""
        chunk_path = self.chunks_dir / self.chunk_infos[chunk_idx]['path']
        return torch.load(chunk_path, map_location='cpu')

    def __iter__(self):
        """Iterate sequentially through all chunks with prefetching."""
        from concurrent.futures import ThreadPoolExecutor

        n_chunks = len(self.chunk_infos)
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Start prefetching the first chunk
            future = executor.submit(self._load_chunk, 0)

            for ci_idx in range(n_chunks):
                # Wait for current chunk
                chunk = future.result()

                # Prefetch next chunk
                if ci_idx + 1 < n_chunks:
                    future = executor.submit(self._load_chunk, ci_idx + 1)

                n = chunk['seq_lens'].shape[0]
                egpt = chunk['egpt_hidden']
                vl = chunk['vl_hidden']
                seq_lens = chunk['seq_lens']

                for start in range(0, n, self.batch_size):
                    end = min(start + self.batch_size, n)

                batch_items = []
                for idx in range(start, end):
                    sl = seq_lens[idx].item()
                    batch_items.append({
                        'egpt_hidden': egpt[idx, :sl],
                        'vl_hidden': vl[idx, :sl],
                        'seq_len': sl,
                    })

                if self.collate_fn:
                    yield self.collate_fn(batch_items)
                else:
                    yield batch_items

            del chunk, egpt, vl, seq_lens


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
        model: nn.Module,  # Any adapter type (L1-L5, L5F, B1)
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        device: str = 'cuda',
        vlm_only: bool = False,
        fused: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vlm_only = vlm_only
        self.fused = fused

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
            'val_accept_80': [],
            'val_accept_85': [],
            'val_accept_90': [],
            'val_accept_95': [],
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
            if self.vlm_only:
                source_hidden = batch['vl_hidden'].to(self.device)
            else:
                source_hidden = batch['egpt_hidden'].to(self.device)
            target_hidden = batch['vl_hidden'].to(self.device)
            mask = batch['mask'].to(self.device)

            self.optimizer.zero_grad()

            losses = self.model.compute_loss(source_hidden, target_hidden, mask)
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
            if self.vlm_only:
                source_hidden = batch['vl_hidden'].to(self.device)
            else:
                source_hidden = batch['egpt_hidden'].to(self.device)
            target_hidden = batch['vl_hidden'].to(self.device)
            mask = batch['mask'].to(self.device)

            losses = self.model.compute_loss(source_hidden, target_hidden, mask)

            total_loss += losses['total_loss'].item()
            total_cos_sim += losses['cos_sim'].item()
            num_batches += 1

            # Per-position cosine similarity
            if self.fused:
                aligned = self.model(source_hidden, target_hidden)
            else:
                aligned = self.model(source_hidden)
            aligned_norm = F.normalize(aligned, dim=-1)
            vl_norm = F.normalize(target_hidden, dim=-1)
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
                self.history['val_accept_80'].append(val_metrics.get('accept_80', 0))
                self.history['val_accept_85'].append(val_metrics.get('accept_85', 0))
                self.history['val_accept_90'].append(val_metrics.get('accept_90', 0))
                self.history['val_accept_95'].append(val_metrics.get('accept_95', 0))

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
        """Plot training curves: loss, cosine sim, and acceptance rates."""
        has_accept = 'val_accept_90' in self.history and len(self.history['val_accept_90']) > 0
        ncols = 3 if has_accept else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

        # Loss curve
        axes[0].plot(self.history['train_loss'], label='Train')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE + 0.5*CosLoss)')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Cosine similarity curve
        axes[1].plot(self.history['train_cos_sim'], label='Train')
        if self.history['val_cos_sim']:
            axes[1].plot(self.history['val_cos_sim'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].set_title('Feature Alignment (EGPT → VLLaVA)')
        axes[1].legend()
        axes[1].grid(True)

        # Acceptance rate curves
        if has_accept:
            for thresh in ['80', '85', '90', '95']:
                key = f'val_accept_{thresh}'
                if key in self.history:
                    axes[2].plot(self.history[key], label=f'@0.{thresh}')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Acceptance Rate')
            axes[2].set_title('Token Acceptance Rate (SD)')
            axes[2].legend()
            axes[2].grid(True)
            axes[2].set_ylim(0, 1)

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
    parser.add_argument('--adapter_level', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help='Adapter complexity level: 1=Bottleneck(2M), 2=MultiLayer(8M), 3=Wide(16M), 4=Attention(50M), 5=EAGLE(50M), 6=L5F FusedEAGLE(67M)')

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

    # Baseline mode
    parser.add_argument('--vlm_only', action='store_true',
                        help='B1 VLM-only EAGLE baseline: use VL hidden states as both input and target')
    args = parser.parse_args()

    # Load data (supports both single .pt file and chunked directory)
    print("Loading data...")

    def load_small_data(data_path):
        """Load data fully into memory (for small files or val data)."""
        if os.path.isdir(data_path):
            index_path = os.path.join(data_path, 'index.json')
            with open(index_path) as f:
                index = json.load(f)

            all_egpt, all_vl, all_lens = [], [], []
            chunks_dir = os.path.join(data_path, 'chunks')

            for chunk_info in index['chunks']:
                chunk_path = os.path.join(chunks_dir, chunk_info['path'])
                print(f"  Loading chunk: {chunk_info['path']}")
                chunk = torch.load(chunk_path, map_location='cpu')
                all_egpt.append(chunk['egpt_hidden'])
                all_vl.append(chunk['vl_hidden'])
                all_lens.append(chunk['seq_lens'])

            return {
                'egpt_hidden': torch.cat(all_egpt, dim=0),
                'vl_hidden': torch.cat(all_vl, dim=0),
                'seq_lens': torch.cat(all_lens, dim=0),
            }
        else:
            return torch.load(data_path, map_location='cpu')

    # Determine if train data is chunked directory
    is_chunked_train = os.path.isdir(args.train_data)

    if is_chunked_train:
        # Memory-efficient: stream 1 chunk at a time (~1.6GB instead of 80GB)
        print(f"  Using ChunkedTrainLoader (memory-efficient)")
        train_loader = ChunkedTrainLoader(
            args.train_data, batch_size=args.batch_size, collate_fn=collate_fn,
        )
        train_samples = train_loader.total_samples

        # Get hidden_dim from first chunk
        first_chunk = torch.load(
            os.path.join(args.train_data, 'chunks', 'chunk_000000.pt'),
            map_location='cpu',
        )
        hidden_dim = first_chunk['egpt_hidden'].shape[-1]
        del first_chunk
    else:
        print(f"  Loading train data into memory...")
        data = load_small_data(args.train_data)
        hidden_dim = data['egpt_hidden'].shape[-1]
        train_dataset = HiddenStateDataset(
            data['egpt_hidden'], data['vl_hidden'], data['seq_lens'],
        )
        train_samples = len(train_dataset)
        print(f"  Loaded {train_samples} samples, hidden_dim={hidden_dim}")
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )

    # Load val data - stream all chunks (memory-efficient, like train)
    val_loader = None
    if args.val_data:
        if os.path.isdir(args.val_data):
            # Use chunked streaming for val (loads 1 chunk at a time, ~1.6GB)
            val_loader = ChunkedValLoader(
                args.val_data, batch_size=args.batch_size, collate_fn=collate_fn,
            )
            val_samples = val_loader.total_samples
        else:
            val_data = load_small_data(args.val_data)
            val_dataset = HiddenStateDataset(
                val_data['egpt_hidden'], val_data['vl_hidden'], val_data['seq_lens'],
            )
            del val_data
            val_samples = len(val_dataset)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0, pin_memory=True,
            )
    else:
        if not is_chunked_train:
            val_size = int(train_samples * args.val_split)
            train_size = train_samples - val_size
            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=4, pin_memory=True,
            )
            val_samples = val_size
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0, pin_memory=True,
            )
        else:
            # Use last train chunk as val
            print(f"  No val_data, using last train chunk as validation")
            last_path = os.path.join(
                args.train_data, 'chunks', train_loader.chunk_infos[-1]['path'],
            )
            last_chunk = torch.load(last_path, map_location='cpu')
            val_dataset = HiddenStateDataset(
                last_chunk['egpt_hidden'], last_chunk['vl_hidden'], last_chunk['seq_lens'],
            )
            val_samples = len(val_dataset)
            del last_chunk
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0, pin_memory=True,
            )

    print(f"  Train: {train_samples}, Val: {val_samples}")

    # Set default bottleneck dims based on level
    if args.bottleneck_dim is None:
        bottleneck_dim = 1024 if args.adapter_level == 3 else 256
    else:
        bottleneck_dim = args.bottleneck_dim

    if args.vlm_only:
        print(f"\nCreating B1 VLM-only EAGLE baseline (adapter_level={args.adapter_level})...")
    else:
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
        vlm_only=args.vlm_only,
        fused=(args.adapter_level == 6),
    )

    # Create output directory: tasks/<adapter>_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.vlm_only:
        prefix = 'B1'
    elif args.adapter_level == 6:
        prefix = 'L5F'
    else:
        prefix = f'L{args.adapter_level}'
    task_dir = Path(args.output_dir) / f"{prefix}_{timestamp}"

    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=str(task_dir),
        early_stopping=args.early_stopping,
    )

    # Save comprehensive config
    level_names = {
        1: 'Bottleneck', 2: 'MultiLayerBottleneck',
        3: 'WideBottleneck', 4: 'Attention', 5: 'EAGLE',
        6: 'FusedEAGLE',
    }
    if args.vlm_only:
        task_name = f'train_B1_VLMOnly_{level_names.get(args.adapter_level, "Unknown")}'
        alignment_desc = 'Video-LLaVA → Video-LLaVA (VLM-only baseline)'
    elif args.adapter_level == 6:
        task_name = 'train_L5F_FusedEAGLE'
        alignment_desc = 'Fused(EventGPT + Video-LLaVA) → Video-LLaVA (gated fusion)'
    else:
        task_name = f'train_L{args.adapter_level}_{level_names.get(args.adapter_level, "Unknown")}'
        alignment_desc = 'EventGPT → Video-LLaVA (hidden states)'
    config = {
        'task': task_name,
        'vlm_only': args.vlm_only,
        'adapter': {
            'level': args.adapter_level,
            'name': level_names.get(args.adapter_level, 'Unknown'),
            'hidden_dim': hidden_dim,
            'bottleneck_dim': bottleneck_dim,
            'num_blocks': args.num_blocks,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'ffn_dim': args.ffn_dim,
            'params': model.get_num_parameters(),
        },
        'data': {
            'train_path': args.train_data,
            'val_path': args.val_data or 'split from train',
            'train_samples': train_samples,
            'val_samples': val_samples,
            'questions_per_sample': 10,
            'duration': '1s',
            'quant': '4bit',
            'alignment': alignment_desc,
        },
        'training': {
            'epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'early_stopping': args.early_stopping,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'loss': 'MSE + 0.5 * CosLoss',
        },
        'timestamp': timestamp,
        'best_val_loss': trainer.best_val_loss,
    }
    with open(task_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nTask saved to: {task_dir}")


if __name__ == "__main__":
    main()
