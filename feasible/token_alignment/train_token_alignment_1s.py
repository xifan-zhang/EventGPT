#!/usr/bin/env python3
"""
Train Token Alignment on 1s Dataset (Standalone)
================================================

Simple script to train token alignment using existing benchmark results.
No complex imports - everything in one file.

Usage:
    python train_token_alignment_1s.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TokenAlignmentConfig:
    """Configuration for token alignment."""
    draft_vocab_size: int = 32000
    target_vocab_size: int = 32000
    hidden_dim: int = 1024
    num_layers: int = 2
    dropout: float = 0.1
    temperature: float = 1.0


class TokenMappingModule(nn.Module):
    """Simple token-to-token mapping with MLP."""

    def __init__(self, config: TokenAlignmentConfig):
        super().__init__()
        self.config = config

        # Token embedding for draft tokens
        self.draft_embed = nn.Embedding(config.draft_vocab_size, config.hidden_dim)

        # MLP to transform
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Output projection to target vocab
        self.output_proj = nn.Linear(config.hidden_dim, config.target_vocab_size)

    def forward(self, draft_tokens: torch.Tensor) -> torch.Tensor:
        """
        Map draft tokens to target logits.

        Args:
            draft_tokens: [batch, seq_len] draft token IDs

        Returns:
            logits: [batch, seq_len, target_vocab_size]
        """
        x = self.draft_embed(draft_tokens)
        x = self.mlp(x)
        logits = self.output_proj(x)
        return logits

    def compute_loss(
        self,
        draft_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> dict:
        """Compute cross-entropy loss."""
        logits = self.forward(draft_tokens)

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_tokens.view(-1)

        # Cross-entropy loss (ignore padding token 0)
        ce_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)

        # Accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = (target_tokens != 0)
            correct = ((preds == target_tokens) & mask).sum()
            total = mask.sum()
            accuracy = correct.float() / total.clamp(min=1)

        return {
            'loss': ce_loss,
            'accuracy': accuracy,
        }


class TokenDataset(Dataset):
    """Simple dataset for token pairs."""

    def __init__(self, draft_tokens, target_tokens):
        self.draft_tokens = draft_tokens
        self.target_tokens = target_tokens

    def __len__(self):
        return len(self.draft_tokens)

    def __getitem__(self, idx):
        return {
            'draft_tokens': self.draft_tokens[idx],
            'target_tokens': self.target_tokens[idx],
        }


def load_from_benchmark(json_path: str, max_seq_len: int = 128):
    """Load token data from benchmark results."""
    print(f"Loading from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    draft_list = []
    target_list = []

    for item in data:
        draft_ids = item.get('egpt_token_ids', [])
        target_ids = item.get('llava_token_ids', [])

        if not draft_ids or not target_ids:
            continue

        # Truncate and pad
        draft_ids = draft_ids[:max_seq_len]
        target_ids = target_ids[:max_seq_len]

        draft_ids = draft_ids + [0] * (max_seq_len - len(draft_ids))
        target_ids = target_ids + [0] * (max_seq_len - len(target_ids))

        draft_list.append(draft_ids)
        target_list.append(target_ids)

    draft_tokens = torch.tensor(draft_list, dtype=torch.long)
    target_tokens = torch.tensor(target_list, dtype=torch.long)

    print(f"Loaded {len(draft_tokens)} samples")
    return draft_tokens, target_tokens


def train(
    benchmark_json: str,
    output_dir: str,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    train_ratio: float = 0.8,
):
    """Train token alignment model."""
    print("="*60)
    print("Token Alignment Training")
    print("="*60)

    # Load data
    draft_tokens, target_tokens = load_from_benchmark(benchmark_json)

    if len(draft_tokens) == 0:
        print("ERROR: No valid samples found!")
        return

    # Split train/val
    total = len(draft_tokens)
    train_size = int(train_ratio * total)

    torch.manual_seed(42)
    indices = torch.randperm(total)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_dataset = TokenDataset(draft_tokens[train_idx], target_tokens[train_idx])
    val_dataset = TokenDataset(draft_tokens[val_idx], target_tokens[val_idx])

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    config = TokenAlignmentConfig()
    model = TokenMappingModule(config).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_acc = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_acc = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            draft = batch['draft_tokens'].to(device)
            target = batch['target_tokens'].to(device)

            optimizer.zero_grad()
            losses = model.compute_loss(draft, target)
            losses['loss'].backward()
            optimizer.step()

            train_loss += losses['loss'].item()
            train_acc += losses['accuracy'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.4f}",
                'acc': f"{losses['accuracy'].item()*100:.1f}%"
            })

        train_loss /= num_batches
        train_acc /= num_batches

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                draft = batch['draft_tokens'].to(device)
                target = batch['target_tokens'].to(device)

                logits = model(draft)
                preds = logits.argmax(dim=-1)

                mask = (target != 0)
                val_correct += ((preds == target) & mask).sum().item()
                val_total += mask.sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
              f"Train Acc={train_acc*100:.1f}%, Val Acc={val_acc*100:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_acc': val_acc,
            }
            torch.save(checkpoint, output_dir / 'token_alignment.pt')
            print(f"  Saved best model (acc={val_acc*100:.1f}%)")

    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc*100:.1f}%")
    print(f"Checkpoint saved to: {output_dir / 'token_alignment.pt'}")

    # Save training info
    info = {
        'benchmark_json': benchmark_json,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'best_val_acc': best_val_acc,
        'num_epochs': num_epochs,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    return model, best_val_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_json', type=str,
                        default='./feasible/benchmark_inference/benchmark_results_S1.json')
    parser.add_argument('--output_dir', type=str,
                        default='./feasible/token_alignment/checkpoints_1s')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train(
        benchmark_json=args.benchmark_json,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
