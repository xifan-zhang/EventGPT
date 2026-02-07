"""
Sequence Distillation Module
============================

Trains a small model to predict Video-LLaVA tokens given EventGPT hidden states.
Uses pre-extracted hidden states (encoders fixed).

This is lightweight enough for a 4090 with fixed encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict
from tqdm import tqdm

from .base import (
    TokenAlignmentConfig,
    BaseTokenAlignmentModule,
    BaseTokenAlignmentTrainer,
    TokenAlignmentDataset,
    MLP,
)


@dataclass
class SequenceDistillationConfig(TokenAlignmentConfig):
    """Configuration for sequence distillation."""

    # Distillation specific
    num_transformer_layers: int = 2
    num_attention_heads: int = 8
    use_cross_attention: bool = False  # Whether to attend to draft hidden states
    label_smoothing: float = 0.1


class SequenceDistillationModule(BaseTokenAlignmentModule):
    """
    Distills Video-LLaVA sequences using EventGPT hidden states.

    Input: EventGPT hidden states [batch, seq_len, hidden_size]
    Output: Predicted token logits [batch, seq_len, vocab_size]

    Trains to predict Video-LLaVA tokens from EventGPT representations.
    """

    def __init__(self, config: SequenceDistillationConfig):
        super().__init__(config)
        self.config = config

        # Project draft hidden to intermediate dim
        self.input_projection = nn.Linear(config.draft_hidden_size, config.hidden_dim)

        # Small transformer for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers,
        )

        # Output projection to target vocab
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.target_vocab_size),
        )

    def forward(
        self,
        draft_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict target tokens from draft hidden states.

        Args:
            draft_hidden: [batch, seq_len, draft_hidden_size] EventGPT hidden states
            attention_mask: [batch, seq_len] mask for padding

        Returns:
            logits: [batch, seq_len, target_vocab_size]
        """
        # Project to intermediate dimension
        x = self.input_projection(draft_hidden)

        # Create causal mask for autoregressive prediction
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        # Create padding mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len], 1 = valid, 0 = pad
            # Transformer expects True = ignore
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Transform
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

        # Output logits
        logits = self.output_projection(x)

        return logits

    def compute_loss(
        self,
        draft_hidden: torch.Tensor,
        target_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.

        Args:
            draft_hidden: [batch, seq_len, hidden_size] EventGPT hidden states
            target_tokens: [batch, seq_len] Video-LLaVA token IDs
            attention_mask: [batch, seq_len] mask for valid positions

        Returns:
            Dict with losses
        """
        # Forward
        logits = self.forward(draft_hidden, attention_mask)

        # Cross-entropy loss with label smoothing
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_tokens.view(-1)

        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=0,  # Ignore padding
            label_smoothing=self.config.label_smoothing,
        )

        # Accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            mask = (target_tokens != 0)
            correct = ((predictions == target_tokens) & mask).sum()
            total = mask.sum()
            accuracy = correct.float() / total.clamp(min=1)

        return {
            'total_loss': ce_loss,
            'ce_loss': ce_loss,
            'accuracy': accuracy,
        }


class SequenceDistillationTrainer(BaseTokenAlignmentTrainer):
    """Trainer for sequence distillation."""

    def __init__(
        self,
        model: SequenceDistillationModule,
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

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            draft_hidden = batch['draft_hidden'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)

            # Create attention mask
            draft_tokens = batch['draft_tokens'].to(self.device)
            attention_mask = (draft_tokens != 0).float()

            # Forward and loss
            self.optimizer.zero_grad()
            losses = self.model.compute_loss(draft_hidden, target_tokens, attention_mask)
            loss = losses['total_loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += losses['accuracy'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{losses['accuracy'].item()*100:.1f}%",
            })

        return total_loss / num_batches

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                draft_hidden = batch['draft_hidden'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                draft_tokens = batch['draft_tokens'].to(self.device)

                attention_mask = (draft_tokens != 0).float()

                # Compute loss
                losses = self.model.compute_loss(draft_hidden, target_tokens, attention_mask)
                total_loss += losses['total_loss'].item()

                # Compute acceptance rate
                logits = self.model(draft_hidden, attention_mask)
                predictions = logits.argmax(dim=-1)

                mask = (target_tokens != 0)
                correct = ((predictions == target_tokens) & mask).sum().item()
                total = mask.sum().item()

                total_correct += correct
                total_tokens += total
                num_batches += 1

        acceptance_rate = total_correct / total_tokens if total_tokens > 0 else 0

        return {
            'loss': total_loss / num_batches,
            'acceptance_rate': acceptance_rate,
        }
