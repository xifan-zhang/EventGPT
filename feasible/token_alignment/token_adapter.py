"""
Token Adapter - Lightweight Token-Level Alignment (No Hidden States Required)
=============================================================================

A practical approach that learns to predict Video-LLaVA tokens from EventGPT
token sequences WITHOUT requiring hidden states.

Key Insight: The main bottleneck is the SEMANTIC GAP - EventGPT and Video-LLaVA
describe scenes differently. This adapter learns:
1. Token-to-token vocabulary mapping
2. Position-aware context (n-gram patterns)
3. Autoregressive refinement

Memory: ~100-500MB (tiny)
Training: Works with existing cached tokens
Expected Improvement: 3% -> 15-25% (modest but meaningful)

For higher accuracy (50%+), you need:
- Hidden states extraction (EAGLE fusion)
- Or fine-tuning EventGPT's LM head directly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import math

try:
    from .base import (
        TokenAlignmentConfig,
        BaseTokenAlignmentModule,
        BaseTokenAlignmentTrainer,
        TokenAlignmentDataset,
    )
except ImportError:
    from base import (
        TokenAlignmentConfig,
        BaseTokenAlignmentModule,
        BaseTokenAlignmentTrainer,
        TokenAlignmentDataset,
    )


@dataclass
class TokenAdapterConfig(TokenAlignmentConfig):
    """Configuration for token adapter."""

    # Architecture
    embed_dim: int = 512  # Token embedding dimension
    num_layers: int = 4   # Transformer layers
    num_heads: int = 8    # Attention heads
    ffn_dim: int = 2048   # FFN dimension

    # Training
    label_smoothing: float = 0.1
    use_auxiliary_loss: bool = True  # Position-wise auxiliary loss

    # Context
    max_context: int = 64  # Max context window


class TokenAdapter(BaseTokenAlignmentModule):
    """
    Lightweight token adapter that predicts target tokens from draft tokens.

    Architecture:
    1. Shared embedding for draft tokens (with position)
    2. Small transformer encoder
    3. Output projection to target vocabulary

    This learns patterns like:
    - "parked" (EventGPT) often maps to "driving"/"moving" (Video-LLaVA)
    - "car" stays as "car" (identity mapping for common tokens)
    - Position-dependent patterns (sentence structure differences)
    """

    def __init__(self, config: TokenAdapterConfig):
        super().__init__(config)
        self.config = config

        # Token embedding (shared for both models since same tokenizer)
        self.embedding = nn.Embedding(config.draft_vocab_size, config.embed_dim)

        # Position embedding
        self.position_embedding = nn.Embedding(config.max_context, config.embed_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output projection to target vocabulary
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.target_vocab_size),
        )

        # Direct token mapping (learns vocab alignment)
        # This captures simple 1-to-1 token mappings
        self.token_bias = nn.Parameter(torch.zeros(config.target_vocab_size))

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        draft_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            draft_tokens: [batch, seq_len] draft token IDs
            attention_mask: [batch, seq_len] mask for padding

        Returns:
            logits: [batch, seq_len, target_vocab_size]
        """
        batch_size, seq_len = draft_tokens.shape
        device = draft_tokens.device

        # Truncate to max context
        if seq_len > self.config.max_context:
            draft_tokens = draft_tokens[:, :self.config.max_context]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.config.max_context]
            seq_len = self.config.max_context

        # Get embeddings
        token_emb = self.embedding(draft_tokens)  # [batch, seq, embed_dim]

        # Add position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb

        # Create causal mask for autoregressive prediction
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        # Padding mask
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        # Transform
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Output logits
        logits = self.output_proj(x)

        # Add token bias (encourages learning direct mappings)
        logits = logits + self.token_bias

        return logits

    def compute_loss(
        self,
        draft_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            draft_tokens: [batch, seq] EventGPT tokens
            target_tokens: [batch, seq] Video-LLaVA tokens
            attention_mask: [batch, seq] mask

        Returns:
            Dict with losses
        """
        # Shift for autoregressive: predict next token
        draft_input = draft_tokens[:, :-1]
        target_labels = target_tokens[:, 1:]

        if attention_mask is not None:
            attention_mask = attention_mask[:, :-1]

        # Forward
        logits = self.forward(draft_input, attention_mask)

        # Truncate to match
        min_len = min(logits.size(1), target_labels.size(1))
        logits = logits[:, :min_len]
        target_labels = target_labels[:, :min_len]

        # Cross-entropy loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = target_labels.reshape(-1)

        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=0,
            label_smoothing=self.config.label_smoothing,
        )

        # Accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            mask = (target_labels != 0)
            correct = ((predictions == target_labels) & mask).sum()
            total = mask.sum()
            accuracy = correct.float() / total.clamp(min=1)

            # Top-5 accuracy
            top5_preds = logits.topk(5, dim=-1).indices
            top5_match = (top5_preds == target_labels.unsqueeze(-1)).any(dim=-1)
            top5_acc = (top5_match & mask).sum().float() / total.clamp(min=1)

        # Auxiliary loss: position-wise consistency
        aux_loss = torch.tensor(0.0, device=logits.device)
        if self.config.use_auxiliary_loss:
            # Encourage similar predictions for similar draft tokens
            # (regularization to prevent overfitting)
            pred_probs = F.softmax(logits, dim=-1)
            pred_entropy = -(pred_probs * pred_probs.log().clamp(min=-100)).sum(dim=-1)
            aux_loss = -pred_entropy.mean() * 0.01  # Encourage confident predictions

        total_loss = ce_loss + aux_loss

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'aux_loss': aux_loss,
            'accuracy': accuracy,
            'top5_accuracy': top5_acc,
        }


class TokenAdapterTrainer(BaseTokenAlignmentTrainer):
    """Trainer for token adapter."""

    def __init__(
        self,
        model: TokenAdapter,
        train_dataset: TokenAlignmentDataset,
        val_dataset: Optional[TokenAlignmentDataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
    ):
        super().__init__(
            model, train_dataset, val_dataset,
            batch_size, learning_rate, device
        )

        # Cosine scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_loader) * 50,
            eta_min=1e-6,
        )

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training TokenAdapter")
        for batch in pbar:
            draft_tokens = batch['draft_tokens'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)

            attention_mask = (draft_tokens != 0).float()

            # Forward and loss
            self.optimizer.zero_grad()
            losses = self.model.compute_loss(draft_tokens, target_tokens, attention_mask)
            loss = losses['total_loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += losses['accuracy'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{losses['accuracy'].item()*100:.1f}%",
                'top5': f"{losses['top5_accuracy'].item()*100:.1f}%",
            })

        return total_loss / num_batches

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_top5 = 0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                draft_tokens = batch['draft_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)

                attention_mask = (draft_tokens != 0).float()

                losses = self.model.compute_loss(draft_tokens, target_tokens, attention_mask)
                total_loss += losses['total_loss'].item()

                # Get predictions
                draft_input = draft_tokens[:, :-1]
                target_labels = target_tokens[:, 1:]

                logits = self.model.forward(draft_input, attention_mask[:, :-1])

                min_len = min(logits.size(1), target_labels.size(1))
                logits = logits[:, :min_len]
                target_labels = target_labels[:, :min_len]

                predictions = logits.argmax(dim=-1)
                mask = (target_labels != 0)

                correct = ((predictions == target_labels) & mask).sum().item()
                total_correct += correct

                top5_preds = logits.topk(5, dim=-1).indices
                top5_match = (top5_preds == target_labels.unsqueeze(-1)).any(dim=-1)
                total_top5 += (top5_match & mask).sum().item()

                total_tokens += mask.sum().item()
                num_batches += 1

        acceptance_rate = total_correct / total_tokens if total_tokens > 0 else 0
        top5_rate = total_top5 / total_tokens if total_tokens > 0 else 0

        return {
            'loss': total_loss / num_batches,
            'acceptance_rate': acceptance_rate,
            'top5_rate': top5_rate,
        }


def train_token_adapter(
    cached_dir: str,
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
) -> Dict:
    """
    Train token adapter on cached token data.

    Args:
        cached_dir: Path to cached tokens directory
        output_dir: Path to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use

    Returns:
        Training results dict
    """
    from pathlib import Path
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {cached_dir}...")
    dataset = TokenAlignmentDataset.from_directory(cached_dir)
    print(f"Loaded {len(dataset)} samples")

    # Split
    num_train = int(len(dataset) * 0.9)
    indices = torch.randperm(len(dataset))

    train_dataset = TokenAlignmentDataset(
        draft_tokens=dataset.draft_tokens[indices[:num_train]],
        target_tokens=dataset.target_tokens[indices[:num_train]],
        sample_ids=[dataset.sample_ids[i] for i in indices[:num_train].tolist()],
    )

    val_dataset = TokenAlignmentDataset(
        draft_tokens=dataset.draft_tokens[indices[num_train:]],
        target_tokens=dataset.target_tokens[indices[num_train:]],
        sample_ids=[dataset.sample_ids[i] for i in indices[num_train:].tolist()],
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Compute baseline
    baseline_correct = 0
    baseline_total = 0
    for i in range(len(val_dataset)):
        d, t = val_dataset.draft_tokens[i], val_dataset.target_tokens[i]
        min_len = min(len(d), len(t))
        mask = (t[:min_len] != 0) & (d[:min_len] != 0)
        baseline_correct += ((d[:min_len] == t[:min_len]) & mask).sum().item()
        baseline_total += mask.sum().item()
    baseline_acc = baseline_correct / baseline_total if baseline_total > 0 else 0
    print(f"Baseline acceptance: {baseline_acc*100:.2f}%")

    # Create model
    config = TokenAdapterConfig(
        draft_vocab_size=32000,
        target_vocab_size=32000,
        embed_dim=512,
        num_layers=4,
        num_heads=8,
        ffn_dim=2048,
        dropout=0.1,
    )

    model = TokenAdapter(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,} (~{params*4/1024/1024:.1f} MB)")

    # Train
    trainer = TokenAdapterTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    checkpoint_path = output_path / 'best_model.pt'
    trainer.train(num_epochs=num_epochs, save_path=str(checkpoint_path), early_stopping_patience=10)

    # Final evaluation
    final_metrics = trainer.validate()
    print(f"\nFinal Results:")
    print(f"  Baseline: {baseline_acc*100:.2f}%")
    print(f"  TokenAdapter: {final_metrics['acceptance_rate']*100:.2f}%")
    print(f"  Top-5: {final_metrics['top5_rate']*100:.2f}%")
    print(f"  Improvement: +{(final_metrics['acceptance_rate']-baseline_acc)*100:.2f}%")

    # Save results
    results = {
        'baseline_acceptance': baseline_acc,
        'final_acceptance': final_metrics['acceptance_rate'],
        'top5_rate': final_metrics['top5_rate'],
        'improvement': final_metrics['acceptance_rate'] - baseline_acc,
        'num_samples': len(dataset),
        'model_params': params,
    }

    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cached_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train_token_adapter(
        cached_dir=args.cached_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
