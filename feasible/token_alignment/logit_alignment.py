"""
Logit Alignment Module
======================

Aligns EventGPT output logits to Video-LLaVA output logits using KL divergence.
Uses pre-extracted logits (encoders fixed, no gradients needed).

Memory: ~1-2GB VRAM for training adapter on cached logits.
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
class LogitAlignmentConfig(TokenAlignmentConfig):
    """Configuration for logit alignment."""

    # Logit alignment specific
    use_top_k_logits: bool = True  # Only use top-k logits to save memory
    top_k: int = 100  # Number of top logits to keep
    kl_temperature: float = 2.0  # Temperature for KL loss
    label_smoothing: float = 0.0


class LogitAlignmentModule(BaseTokenAlignmentModule):
    """
    Aligns draft model logits to target model logits.

    Input: Draft logits [batch, seq_len, vocab_size] or [batch, seq_len, top_k]
    Output: Aligned logits [batch, seq_len, vocab_size]

    The module learns a transformation to make draft logits closer to target logits.
    """

    def __init__(self, config: LogitAlignmentConfig):
        super().__init__(config)
        self.config = config

        # If using top-k, we need to project from top_k to full vocab
        input_dim = config.top_k if config.use_top_k_logits else config.draft_vocab_size
        output_dim = config.top_k if config.use_top_k_logits else config.target_vocab_size

        # Lightweight logit transformer
        self.logit_transform = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, output_dim),
        )

        # Optional residual scaling
        if config.use_residual:
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        else:
            self.residual_scale = None

        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * config.temperature)

    def forward(
        self,
        draft_logits: torch.Tensor,
        draft_top_k_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Transform draft logits to align with target distribution.

        Args:
            draft_logits: [batch, seq_len, input_dim] draft model logits
            draft_top_k_indices: [batch, seq_len, top_k] indices for top-k (if using sparse)

        Returns:
            aligned_logits: [batch, seq_len, output_dim]
        """
        # Transform logits
        aligned = self.logit_transform(draft_logits)

        # Optional residual connection
        if self.residual_scale is not None and draft_logits.shape == aligned.shape:
            aligned = aligned + self.residual_scale * draft_logits

        # Apply learned temperature
        aligned = aligned / self.temperature.clamp(min=0.1)

        return aligned

    def compute_loss(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute alignment loss.

        Args:
            draft_logits: [batch, seq_len, dim] from EventGPT
            target_logits: [batch, seq_len, dim] from Video-LLaVA
            mask: [batch, seq_len] mask for valid positions

        Returns:
            Dict with 'total_loss', 'kl_loss', etc.
        """
        # Transform draft logits
        aligned_logits = self.forward(draft_logits)

        # KL divergence loss
        kl_temp = self.config.kl_temperature
        draft_log_probs = F.log_softmax(aligned_logits / kl_temp, dim=-1)
        target_probs = F.softmax(target_logits / kl_temp, dim=-1)

        # KL per position
        kl_per_pos = F.kl_div(draft_log_probs, target_probs, reduction='none').sum(dim=-1)

        # Apply mask
        if mask is not None:
            kl_per_pos = kl_per_pos * mask
            kl_loss = kl_per_pos.sum() / mask.sum().clamp(min=1)
        else:
            kl_loss = kl_per_pos.mean()

        kl_loss = kl_loss * (kl_temp ** 2)

        # MSE loss on logits (auxiliary)
        mse_loss = F.mse_loss(aligned_logits, target_logits)

        # Total loss
        total_loss = kl_loss + 0.1 * mse_loss

        return {
            'total_loss': total_loss,
            'kl_loss': kl_loss,
            'mse_loss': mse_loss,
        }


class LogitAlignmentTrainer(BaseTokenAlignmentTrainer):
    """Trainer for logit alignment."""

    def __init__(
        self,
        model: LogitAlignmentModule,
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
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            draft_logits = batch['draft_logits'].to(self.device)
            target_logits = batch['target_logits'].to(self.device)

            # Create mask for non-padding positions
            draft_tokens = batch['draft_tokens'].to(self.device)
            mask = (draft_tokens != 0).float()

            # Forward and loss
            self.optimizer.zero_grad()
            losses = self.model.compute_loss(draft_logits, target_logits, mask)
            loss = losses['total_loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'kl': f"{losses['kl_loss'].item():.4f}",
            })

        return total_loss / num_batches

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_kl = 0.0
        num_batches = 0
        all_acceptance = []

        with torch.no_grad():
            for batch in self.val_loader:
                draft_logits = batch['draft_logits'].to(self.device)
                target_logits = batch['target_logits'].to(self.device)
                draft_tokens = batch['draft_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)

                mask = (draft_tokens != 0).float()

                # Compute loss
                losses = self.model.compute_loss(draft_logits, target_logits, mask)
                total_loss += losses['total_loss'].item()
                total_kl += losses['kl_loss'].item()

                # Compute acceptance rate (argmax of aligned logits vs target tokens)
                aligned_logits = self.model(draft_logits)
                predicted_tokens = aligned_logits.argmax(dim=-1)

                acc_rate, _, _ = self.model.compute_acceptance_rate(
                    predicted_tokens, target_tokens, pad_token_id=0
                )
                all_acceptance.append(acc_rate)

                num_batches += 1

        avg_acceptance = sum(all_acceptance) / len(all_acceptance) if all_acceptance else 0

        return {
            'loss': total_loss / num_batches,
            'kl_loss': total_kl / num_batches,
            'acceptance_rate': avg_acceptance,
        }
