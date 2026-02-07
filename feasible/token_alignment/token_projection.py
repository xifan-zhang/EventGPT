"""
Token Projection Module
=======================

Projects EventGPT output token embeddings to Video-LLaVA token embedding space.
Simple and lightweight approach using pre-extracted token embeddings.

Memory: <1GB VRAM for training on cached embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from tqdm import tqdm

from .base import (
    TokenAlignmentConfig,
    BaseTokenAlignmentModule,
    BaseTokenAlignmentTrainer,
    TokenAlignmentDataset,
    MLP,
)


@dataclass
class TokenProjectionConfig(TokenAlignmentConfig):
    """Configuration for token projection."""

    # Projection specific
    projection_type: str = 'mlp'  # 'linear', 'mlp', 'bottleneck'
    bottleneck_dim: int = 512  # For bottleneck projection
    contrastive_weight: float = 0.1  # Weight for contrastive loss
    use_token_type_embeddings: bool = False  # Add position-dependent projection


class TokenProjectionModule(BaseTokenAlignmentModule):
    """
    Projects draft token embeddings to target token embedding space.

    Input: Draft token embeddings [batch, seq_len, embed_dim]
    Output: Projected embeddings [batch, seq_len, target_embed_dim]

    Can also map draft token IDs directly to target token space using
    a learned token mapping.
    """

    def __init__(self, config: TokenProjectionConfig):
        super().__init__(config)
        self.config = config

        # Choose projection type
        if config.projection_type == 'linear':
            self.projection = nn.Linear(config.draft_hidden_size, config.target_hidden_size)
        elif config.projection_type == 'mlp':
            self.projection = MLP(
                input_dim=config.draft_hidden_size,
                hidden_dim=config.hidden_dim,
                output_dim=config.target_hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_layer_norm=config.use_layer_norm,
            )
        elif config.projection_type == 'bottleneck':
            self.projection = nn.Sequential(
                nn.Linear(config.draft_hidden_size, config.bottleneck_dim),
                nn.LayerNorm(config.bottleneck_dim),
                nn.GELU(),
                nn.Linear(config.bottleneck_dim, config.target_hidden_size),
            )
        else:
            raise ValueError(f"Unknown projection type: {config.projection_type}")

        # Optional token mapping (learns which draft tokens map to which target tokens)
        self.token_mapping = nn.Linear(config.draft_vocab_size, config.target_vocab_size, bias=False)

        # Initialize token mapping as identity-like (diagonal)
        with torch.no_grad():
            nn.init.eye_(self.token_mapping.weight[:min(config.draft_vocab_size, config.target_vocab_size),
                                                   :min(config.draft_vocab_size, config.target_vocab_size)])

    def forward(
        self,
        draft_embeddings: Optional[torch.Tensor] = None,
        draft_one_hot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project draft representations to target space.

        Args:
            draft_embeddings: [batch, seq_len, draft_hidden_size] token embeddings
            draft_one_hot: [batch, seq_len, vocab_size] one-hot token representations

        Returns:
            projected: [batch, seq_len, target_hidden_size] or [batch, seq_len, target_vocab_size]
        """
        if draft_embeddings is not None:
            return self.projection(draft_embeddings)
        elif draft_one_hot is not None:
            return self.token_mapping(draft_one_hot)
        else:
            raise ValueError("Must provide either draft_embeddings or draft_one_hot")

    def project_tokens(
        self,
        draft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map draft token IDs to target token probabilities.

        Args:
            draft_tokens: [batch, seq_len] draft token IDs

        Returns:
            target_probs: [batch, seq_len, target_vocab_size] soft token predictions
        """
        # Convert to one-hot
        one_hot = F.one_hot(draft_tokens, num_classes=self.config.draft_vocab_size).float()

        # Map through learned projection
        target_logits = self.token_mapping(one_hot)

        return F.softmax(target_logits / self.config.temperature, dim=-1)

    def compute_loss(
        self,
        draft_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        draft_tokens: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute projection loss.

        Args:
            draft_hidden: [batch, seq_len, hidden_size] EventGPT hidden states
            target_hidden: [batch, seq_len, hidden_size] Video-LLaVA hidden states
            draft_tokens: [batch, seq_len] draft token IDs (optional)
            target_tokens: [batch, seq_len] target token IDs (optional)
            mask: [batch, seq_len] mask for valid positions

        Returns:
            Dict with losses
        """
        # Project draft hidden states
        projected = self.projection(draft_hidden)

        # MSE loss on hidden states
        mse_loss = F.mse_loss(projected, target_hidden, reduction='none')
        if mask is not None:
            mse_loss = (mse_loss.mean(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
        else:
            mse_loss = mse_loss.mean()

        # Cosine similarity loss
        projected_norm = F.normalize(projected, dim=-1)
        target_norm = F.normalize(target_hidden, dim=-1)
        cos_sim = (projected_norm * target_norm).sum(dim=-1)
        if mask is not None:
            cos_loss = (1 - (cos_sim * mask).sum() / mask.sum().clamp(min=1))
        else:
            cos_loss = 1 - cos_sim.mean()

        # Token mapping loss (if tokens provided)
        token_loss = torch.tensor(0.0, device=draft_hidden.device)
        if draft_tokens is not None and target_tokens is not None:
            target_probs = self.project_tokens(draft_tokens)
            # Cross-entropy with target tokens
            target_probs_flat = target_probs.view(-1, target_probs.size(-1))
            target_tokens_flat = target_tokens.view(-1)
            token_loss = F.cross_entropy(
                target_probs_flat.log().clamp(min=-100),
                target_tokens_flat,
                ignore_index=0,
            )

        # Total loss
        total_loss = mse_loss + 0.5 * cos_loss + self.config.contrastive_weight * token_loss

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'cos_loss': cos_loss,
            'token_loss': token_loss,
            'cos_sim': cos_sim.mean() if mask is None else (cos_sim * mask).sum() / mask.sum().clamp(min=1),
        }


class TokenProjectionTrainer(BaseTokenAlignmentTrainer):
    """Trainer for token projection."""

    def __init__(
        self,
        model: TokenProjectionModule,
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
        total_cos_sim = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            draft_hidden = batch['draft_hidden'].to(self.device)
            target_hidden = batch['target_hidden'].to(self.device)
            draft_tokens = batch['draft_tokens'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)

            mask = (draft_tokens != 0).float()

            # Forward and loss
            self.optimizer.zero_grad()
            losses = self.model.compute_loss(
                draft_hidden, target_hidden,
                draft_tokens, target_tokens, mask
            )
            loss = losses['total_loss']

            # Backward
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

        return total_loss / num_batches

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_cos_sim = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                draft_hidden = batch['draft_hidden'].to(self.device)
                target_hidden = batch['target_hidden'].to(self.device)
                draft_tokens = batch['draft_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)

                mask = (draft_tokens != 0).float()

                # Compute loss
                losses = self.model.compute_loss(
                    draft_hidden, target_hidden,
                    draft_tokens, target_tokens, mask
                )
                total_loss += losses['total_loss'].item()
                total_cos_sim += losses['cos_sim'].item()

                # Compute acceptance rate using token mapping
                target_probs = self.model.project_tokens(draft_tokens)
                predictions = target_probs.argmax(dim=-1)

                mask_bool = (target_tokens != 0)
                correct = ((predictions == target_tokens) & mask_bool).sum().item()
                total = mask_bool.sum().item()

                total_correct += correct
                total_tokens += total
                num_batches += 1

        acceptance_rate = total_correct / total_tokens if total_tokens > 0 else 0

        return {
            'loss': total_loss / num_batches,
            'cos_sim': total_cos_sim / num_batches,
            'acceptance_rate': acceptance_rate,
        }
