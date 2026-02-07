"""
EAGLE-style Feature Fusion for Token-Level Alignment
=====================================================

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) achieves
high acceptance rates by using a lightweight fusion layer that combines:
1. Draft model's hidden states
2. Previous token embeddings

This enables the small model to predict the target model's next token distribution
more accurately than direct token mapping.

Reference: Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", 2024

Key Insight:
- Token-level mapping fails because models describe scenes differently
- Hidden state fusion captures semantic intent better than token sequences
- A lightweight fusion layer can learn to bridge the semantic gap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import math

from .base import (
    TokenAlignmentConfig,
    BaseTokenAlignmentModule,
    BaseTokenAlignmentTrainer,
    TokenAlignmentDataset,
    MLP,
)


@dataclass
class EAGLEFusionConfig(TokenAlignmentConfig):
    """Configuration for EAGLE-style feature fusion."""

    # Fusion architecture
    fusion_type: str = 'eagle'  # 'eagle', 'medusa', 'simple'
    num_fusion_layers: int = 2
    num_attention_heads: int = 8
    use_rotary_embeddings: bool = True

    # Multi-head speculation (Medusa-style)
    num_speculation_heads: int = 1  # Number of parallel speculation heads
    max_speculation_depth: int = 5  # Max tokens to speculate ahead

    # Training
    use_kl_loss: bool = True
    kl_weight: float = 0.5
    ce_weight: float = 1.0
    feature_matching_weight: float = 0.1
    temperature: float = 1.0

    # Embedding
    use_position_embeddings: bool = True
    max_position: int = 512


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding."""

    def __init__(self, dim: int, max_position: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding to query and key."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class EAGLEFusionLayer(nn.Module):
    """
    Single EAGLE fusion layer that combines:
    1. Draft hidden states (from EventGPT)
    2. Token embeddings (from previous predictions)

    Uses attention to fuse information and predict next token distribution.
    """

    def __init__(self, config: EAGLEFusionConfig):
        super().__init__()
        self.config = config

        # Input projection: combine draft hidden + token embedding
        self.input_proj = nn.Linear(
            config.draft_hidden_size + config.target_hidden_size,
            config.hidden_dim
        )

        # Self-attention for sequence modeling
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Rotary embedding
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                config.hidden_dim // config.num_attention_heads,
                config.max_position
            )
        else:
            self.rotary_emb = None

    def forward(
        self,
        draft_hidden: torch.Tensor,  # [batch, seq, draft_hidden]
        token_emb: torch.Tensor,     # [batch, seq, target_hidden]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            draft_hidden: EventGPT hidden states [batch, seq, draft_hidden_size]
            token_emb: Target token embeddings [batch, seq, target_hidden_size]
            attention_mask: [batch, seq] mask for padding

        Returns:
            fused: [batch, seq, hidden_dim] fused representation
        """
        # Concatenate and project
        combined = torch.cat([draft_hidden, token_emb], dim=-1)
        x = self.input_proj(combined)

        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=(attention_mask == 0) if attention_mask is not None else None,
        )
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class EAGLEFusionModule(BaseTokenAlignmentModule):
    """
    EAGLE-style Feature Fusion Module.

    Architecture:
    1. Takes EventGPT hidden states + previous token embeddings
    2. Fuses them through attention layers
    3. Projects to target vocabulary distribution

    This approach achieves high acceptance rates because it:
    - Captures semantic intent from draft hidden states
    - Uses autoregressive token context from embeddings
    - Learns the mapping from draft "personality" to target "personality"
    """

    def __init__(self, config: EAGLEFusionConfig):
        super().__init__(config)
        self.config = config

        # Token embedding layer (shared with target model or learned)
        self.token_embedding = nn.Embedding(
            config.target_vocab_size,
            config.target_hidden_size,
        )

        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            EAGLEFusionLayer(config)
            for _ in range(config.num_fusion_layers)
        ])

        # Output projection to target vocabulary
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.target_vocab_size),
        )

        # Optional: Multiple speculation heads (Medusa-style)
        if config.num_speculation_heads > 1:
            self.speculation_heads = nn.ModuleList([
                nn.Linear(config.hidden_dim, config.target_vocab_size)
                for _ in range(config.num_speculation_heads - 1)
            ])
        else:
            self.speculation_heads = None

        # Feature matching projection (to align with target hidden states)
        self.feature_proj = nn.Linear(config.hidden_dim, config.target_hidden_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        input_tokens: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for EAGLE fusion.

        Args:
            draft_hidden: [batch, seq, draft_hidden_size] EventGPT hidden states
            input_tokens: [batch, seq] previous token IDs (for embedding lookup)
            input_embeds: [batch, seq, target_hidden_size] pre-computed embeddings
            attention_mask: [batch, seq] mask for valid positions

        Returns:
            Dict with 'logits', 'features', and optionally 'speculation_logits'
        """
        # Get token embeddings
        if input_embeds is not None:
            token_emb = input_embeds
        elif input_tokens is not None:
            token_emb = self.token_embedding(input_tokens)
        else:
            # Use zeros if no token input (first position)
            token_emb = torch.zeros(
                draft_hidden.size(0),
                draft_hidden.size(1),
                self.config.target_hidden_size,
                device=draft_hidden.device,
            )

        # Pad or truncate to match draft_hidden length
        if token_emb.size(1) < draft_hidden.size(1):
            pad_len = draft_hidden.size(1) - token_emb.size(1)
            token_emb = F.pad(token_emb, (0, 0, 0, pad_len))
        elif token_emb.size(1) > draft_hidden.size(1):
            token_emb = token_emb[:, :draft_hidden.size(1)]

        # Pass through fusion layers
        x = draft_hidden  # Will be replaced by fused representation
        for i, layer in enumerate(self.fusion_layers):
            if i == 0:
                x = layer(draft_hidden, token_emb, attention_mask)
            else:
                x = layer(x, token_emb, attention_mask)

        # Main output logits
        logits = self.output_proj(x)

        # Feature projection for matching
        features = self.feature_proj(x)

        result = {
            'logits': logits,
            'features': features,
        }

        # Additional speculation heads
        if self.speculation_heads is not None:
            spec_logits = [head(x) for head in self.speculation_heads]
            result['speculation_logits'] = torch.stack(spec_logits, dim=1)

        return result

    def compute_loss(
        self,
        draft_hidden: torch.Tensor,
        target_tokens: torch.Tensor,
        target_hidden: Optional[torch.Tensor] = None,
        target_logits: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            draft_hidden: [batch, seq, hidden] EventGPT hidden states
            target_tokens: [batch, seq] Video-LLaVA token IDs
            target_hidden: [batch, seq, hidden] Video-LLaVA hidden states (optional)
            target_logits: [batch, seq, vocab] Video-LLaVA logits (optional)
            attention_mask: [batch, seq] mask for valid positions

        Returns:
            Dict with losses
        """
        # Shift target tokens for autoregressive prediction
        # Input: tokens 0..N-1, Target: tokens 1..N
        input_tokens = target_tokens[:, :-1]
        target_labels = target_tokens[:, 1:]

        # Truncate hidden states to match
        draft_hidden_input = draft_hidden[:, :-1]

        if attention_mask is not None:
            attention_mask = attention_mask[:, :-1]

        # Forward pass
        outputs = self.forward(
            draft_hidden_input,
            input_tokens=input_tokens,
            attention_mask=attention_mask,
        )

        logits = outputs['logits']
        features = outputs['features']

        # Cross-entropy loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = target_labels.reshape(-1)

        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=0,  # Ignore padding
            label_smoothing=0.1,
        )

        # KL divergence loss (if target logits available)
        kl_loss = torch.tensor(0.0, device=logits.device)
        if target_logits is not None and self.config.use_kl_loss:
            target_logits_shifted = target_logits[:, 1:]

            # Soft targets from Video-LLaVA
            soft_targets = F.softmax(target_logits_shifted / self.config.temperature, dim=-1)
            log_probs = F.log_softmax(logits / self.config.temperature, dim=-1)

            kl_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
            kl_loss = kl_loss * (self.config.temperature ** 2)

        # Feature matching loss (if target hidden states available)
        feature_loss = torch.tensor(0.0, device=logits.device)
        if target_hidden is not None:
            target_hidden_shifted = target_hidden[:, 1:]
            feature_loss = F.mse_loss(features, target_hidden_shifted)

        # Total loss
        total_loss = (
            self.config.ce_weight * ce_loss +
            self.config.kl_weight * kl_loss +
            self.config.feature_matching_weight * feature_loss
        )

        # Compute accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            mask = (target_labels != 0)
            correct = ((predictions == target_labels) & mask).sum()
            total = mask.sum()
            accuracy = correct.float() / total.clamp(min=1)

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'feature_loss': feature_loss,
            'accuracy': accuracy,
        }

    def generate_speculation(
        self,
        draft_hidden: torch.Tensor,
        initial_token: torch.Tensor,
        num_tokens: int = 5,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate speculative tokens autoregressively.

        Args:
            draft_hidden: [batch, 1, hidden] single position hidden state
            initial_token: [batch] starting token
            num_tokens: number of tokens to speculate
            temperature: sampling temperature
            top_k: top-k filtering

        Returns:
            (tokens, probs) - speculated tokens and their probabilities
        """
        batch_size = draft_hidden.size(0)
        device = draft_hidden.device

        tokens = [initial_token]
        probs = []

        current_token = initial_token

        for i in range(num_tokens):
            # Get embedding of current token
            token_emb = self.token_embedding(current_token.unsqueeze(1))  # [batch, 1, hidden]

            # Forward through fusion (using same draft hidden for all positions)
            outputs = self.forward(draft_hidden, input_embeds=token_emb)
            logits = outputs['logits'][:, -1, :]  # [batch, vocab]

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs_dist = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs_dist, 1).squeeze(-1)

            # Get probability of selected token
            token_prob = probs_dist.gather(1, next_token.unsqueeze(1)).squeeze(1)

            tokens.append(next_token)
            probs.append(token_prob)
            current_token = next_token

        return torch.stack(tokens[1:], dim=1), torch.stack(probs, dim=1)


class EAGLEFusionTrainer(BaseTokenAlignmentTrainer):
    """Trainer for EAGLE fusion module."""

    def __init__(
        self,
        model: EAGLEFusionModule,
        train_dataset: TokenAlignmentDataset,
        val_dataset: Optional[TokenAlignmentDataset] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
        warmup_steps: int = 100,
    ):
        super().__init__(
            model, train_dataset, val_dataset,
            batch_size, learning_rate, device
        )
        self.warmup_steps = warmup_steps
        self.global_step = 0

        # Use cosine scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_loader) * 50,  # Assume 50 epochs max
            eta_min=1e-6,
        )

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training EAGLE")
        for batch in pbar:
            draft_hidden = batch['draft_hidden'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)

            # Optional: target hidden states and logits
            target_hidden = batch.get('target_hidden')
            if target_hidden is not None:
                target_hidden = target_hidden.to(self.device)

            target_logits = batch.get('target_logits')
            if target_logits is not None:
                target_logits = target_logits.to(self.device)

            # Create attention mask from draft tokens
            draft_tokens = batch['draft_tokens'].to(self.device)
            attention_mask = (draft_tokens != 0).float()

            # Forward and loss
            self.optimizer.zero_grad()
            losses = self.model.compute_loss(
                draft_hidden,
                target_tokens,
                target_hidden=target_hidden,
                target_logits=target_logits,
                attention_mask=attention_mask,
            )
            loss = losses['total_loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += losses['accuracy'].item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{losses['accuracy'].item()*100:.1f}%",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return total_loss / num_batches

    def validate(self) -> Dict[str, float]:
        """Validate model and compute acceptance rate."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_top5_correct = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                draft_hidden = batch['draft_hidden'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                draft_tokens = batch['draft_tokens'].to(self.device)

                target_hidden = batch.get('target_hidden')
                if target_hidden is not None:
                    target_hidden = target_hidden.to(self.device)

                attention_mask = (draft_tokens != 0).float()

                # Compute loss
                losses = self.model.compute_loss(
                    draft_hidden,
                    target_tokens,
                    target_hidden=target_hidden,
                    attention_mask=attention_mask,
                )
                total_loss += losses['total_loss'].item()

                # Get predictions
                input_tokens = target_tokens[:, :-1]
                target_labels = target_tokens[:, 1:]
                draft_hidden_input = draft_hidden[:, :-1]

                outputs = self.model.forward(
                    draft_hidden_input,
                    input_tokens=input_tokens,
                    attention_mask=attention_mask[:, :-1] if attention_mask is not None else None,
                )
                logits = outputs['logits']

                # Top-1 accuracy (acceptance rate)
                predictions = logits.argmax(dim=-1)
                mask = (target_labels != 0)
                correct = ((predictions == target_labels) & mask).sum().item()
                total = mask.sum().item()

                # Top-5 accuracy
                top5_preds = logits.topk(5, dim=-1).indices
                top5_match = (top5_preds == target_labels.unsqueeze(-1)).any(dim=-1)
                top5_correct = (top5_match & mask).sum().item()

                total_correct += correct
                total_top5_correct += top5_correct
                total_tokens += total
                num_batches += 1

        acceptance_rate = total_correct / total_tokens if total_tokens > 0 else 0
        top5_rate = total_top5_correct / total_tokens if total_tokens > 0 else 0

        return {
            'loss': total_loss / num_batches,
            'acceptance_rate': acceptance_rate,
            'top5_rate': top5_rate,
        }
