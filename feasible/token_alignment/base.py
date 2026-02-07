"""
Base Classes for Token Alignment
================================

Provides base classes and utilities for token-level alignment between
EventGPT and Video-LLaVA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np


@dataclass
class TokenAlignmentConfig:
    """Base configuration for token alignment modules."""

    # Model dimensions
    draft_vocab_size: int = 32000  # EventGPT (LLaMA) vocab size
    target_vocab_size: int = 32000  # Video-LLaVA (LLaMA) vocab size
    draft_hidden_size: int = 4096  # EventGPT hidden size
    target_hidden_size: int = 4096  # Video-LLaVA hidden size

    # Alignment settings
    hidden_dim: int = 2048
    num_layers: int = 2
    dropout: float = 0.1
    temperature: float = 1.0

    # Training settings
    use_layer_norm: bool = True
    use_residual: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TokenAlignmentConfig':
        return cls(**d)


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class BaseTokenAlignmentModule(nn.Module):
    """Base class for token alignment modules."""

    def __init__(self, config: TokenAlignmentConfig):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_kl_loss(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Compute KL divergence loss between logit distributions.

        Args:
            draft_logits: [batch, seq_len, vocab_size] from draft model
            target_logits: [batch, seq_len, vocab_size] from target model
            temperature: Temperature for softmax

        Returns:
            KL divergence loss
        """
        draft_probs = F.log_softmax(draft_logits / temperature, dim=-1)
        target_probs = F.softmax(target_logits / temperature, dim=-1)

        # KL(target || draft) - we want draft to match target
        kl_loss = F.kl_div(draft_probs, target_probs, reduction='batchmean')

        return kl_loss * (temperature ** 2)

    def compute_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len] token IDs
            ignore_index: Index to ignore in loss

        Returns:
            Cross-entropy loss
        """
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)

    def compute_mse_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss between hidden states."""
        return F.mse_loss(predicted, target)

    def compute_acceptance_rate(
        self,
        draft_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        pad_token_id: int = 0
    ) -> Tuple[float, int, int]:
        """Compute token-level acceptance rate.

        Args:
            draft_tokens: [batch, seq_len] draft token IDs
            target_tokens: [batch, seq_len] target token IDs
            pad_token_id: Padding token ID to ignore

        Returns:
            (acceptance_rate, matched_count, total_count)
        """
        # Create mask for non-padding tokens
        mask = (target_tokens != pad_token_id) & (draft_tokens != pad_token_id)

        # Count matches
        matches = (draft_tokens == target_tokens) & mask
        matched_count = matches.sum().item()
        total_count = mask.sum().item()

        if total_count == 0:
            return 0.0, 0, 0

        acceptance_rate = matched_count / total_count
        return acceptance_rate, matched_count, total_count

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class TokenAlignmentDataset(Dataset):
    """Dataset for token alignment training.

    Stores paired outputs from EventGPT and Video-LLaVA:
    - draft_tokens: Token IDs from EventGPT
    - target_tokens: Token IDs from Video-LLaVA
    - draft_logits: (optional) Logit distributions from EventGPT
    - target_logits: (optional) Logit distributions from Video-LLaVA
    """

    def __init__(
        self,
        draft_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        draft_logits: Optional[torch.Tensor] = None,
        target_logits: Optional[torch.Tensor] = None,
        draft_hidden: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        sample_ids: Optional[List[str]] = None,
    ):
        """
        Args:
            draft_tokens: [N, seq_len] EventGPT token IDs
            target_tokens: [N, seq_len] Video-LLaVA token IDs
            draft_logits: [N, seq_len, vocab_size] EventGPT logits (optional)
            target_logits: [N, seq_len, vocab_size] Video-LLaVA logits (optional)
            draft_hidden: [N, seq_len, hidden_size] EventGPT hidden states (optional)
            target_hidden: [N, seq_len, hidden_size] Video-LLaVA hidden states (optional)
            sample_ids: List of sample IDs (optional)
        """
        assert len(draft_tokens) == len(target_tokens)

        self.draft_tokens = draft_tokens
        self.target_tokens = target_tokens
        self.draft_logits = draft_logits
        self.target_logits = target_logits
        self.draft_hidden = draft_hidden
        self.target_hidden = target_hidden
        self.sample_ids = sample_ids or [f"sample_{i}" for i in range(len(draft_tokens))]

    def __len__(self):
        return len(self.draft_tokens)

    def __getitem__(self, idx):
        item = {
            'draft_tokens': self.draft_tokens[idx],
            'target_tokens': self.target_tokens[idx],
            'sample_id': self.sample_ids[idx],
        }

        if self.draft_logits is not None:
            item['draft_logits'] = self.draft_logits[idx]
        if self.target_logits is not None:
            item['target_logits'] = self.target_logits[idx]
        if self.draft_hidden is not None:
            item['draft_hidden'] = self.draft_hidden[idx]
        if self.target_hidden is not None:
            item['target_hidden'] = self.target_hidden[idx]

        return item

    @classmethod
    def from_json(cls, json_path: str, max_seq_len: int = 128, pad_token_id: int = 0):
        """Load dataset from benchmark results JSON.

        Expects JSON with 'egpt_token_ids' and 'llava_token_ids' fields.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        draft_tokens_list = []
        target_tokens_list = []
        sample_ids = []

        for item in data:
            draft_ids = item.get('egpt_token_ids', [])
            target_ids = item.get('llava_token_ids', [])

            if not draft_ids or not target_ids:
                continue

            # Pad/truncate to max_seq_len
            draft_ids = draft_ids[:max_seq_len]
            target_ids = target_ids[:max_seq_len]

            # Pad
            draft_ids = draft_ids + [pad_token_id] * (max_seq_len - len(draft_ids))
            target_ids = target_ids + [pad_token_id] * (max_seq_len - len(target_ids))

            draft_tokens_list.append(draft_ids)
            target_tokens_list.append(target_ids)
            sample_ids.append(item.get('id', f'sample_{len(sample_ids)}'))

        draft_tokens = torch.tensor(draft_tokens_list, dtype=torch.long)
        target_tokens = torch.tensor(target_tokens_list, dtype=torch.long)

        return cls(
            draft_tokens=draft_tokens,
            target_tokens=target_tokens,
            sample_ids=sample_ids,
        )

    @classmethod
    def from_directory(cls, dir_path: str):
        """Load pre-saved tensors from directory."""
        dir_path = Path(dir_path)

        draft_tokens = torch.load(dir_path / 'draft_tokens.pt')
        target_tokens = torch.load(dir_path / 'target_tokens.pt')

        draft_logits = None
        target_logits = None
        draft_hidden = None
        target_hidden = None
        sample_ids = None

        if (dir_path / 'draft_logits.pt').exists():
            draft_logits = torch.load(dir_path / 'draft_logits.pt')
        if (dir_path / 'target_logits.pt').exists():
            target_logits = torch.load(dir_path / 'target_logits.pt')
        if (dir_path / 'draft_hidden.pt').exists():
            draft_hidden = torch.load(dir_path / 'draft_hidden.pt')
        if (dir_path / 'target_hidden.pt').exists():
            target_hidden = torch.load(dir_path / 'target_hidden.pt')
        if (dir_path / 'sample_ids.json').exists():
            with open(dir_path / 'sample_ids.json', 'r') as f:
                sample_ids = json.load(f)

        return cls(
            draft_tokens=draft_tokens,
            target_tokens=target_tokens,
            draft_logits=draft_logits,
            target_logits=target_logits,
            draft_hidden=draft_hidden,
            target_hidden=target_hidden,
            sample_ids=sample_ids,
        )

    def save(self, dir_path: str):
        """Save dataset tensors to directory."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.draft_tokens, dir_path / 'draft_tokens.pt')
        torch.save(self.target_tokens, dir_path / 'target_tokens.pt')

        if self.draft_logits is not None:
            torch.save(self.draft_logits, dir_path / 'draft_logits.pt')
        if self.target_logits is not None:
            torch.save(self.target_logits, dir_path / 'target_logits.pt')
        if self.draft_hidden is not None:
            torch.save(self.draft_hidden, dir_path / 'draft_hidden.pt')
        if self.target_hidden is not None:
            torch.save(self.target_hidden, dir_path / 'target_hidden.pt')

        with open(dir_path / 'sample_ids.json', 'w') as f:
            json.dump(self.sample_ids, f)


class BaseTokenAlignmentTrainer:
    """Base trainer for token alignment modules."""

    def __init__(
        self,
        model: BaseTokenAlignmentModule,
        train_dataset: TokenAlignmentDataset,
        val_dataset: Optional[TokenAlignmentDataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            self.val_loader = None

        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'acceptance_rates': [],
        }

    def train_epoch(self) -> float:
        """Train for one epoch. Override in subclass."""
        raise NotImplementedError

    def validate(self) -> Dict[str, float]:
        """Validate model. Override in subclass."""
        raise NotImplementedError

    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 5,
    ):
        """Train the model."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.training_stats['train_losses'].append(train_loss)

            # Validate
            if self.val_loader:
                val_metrics = self.validate()
                val_loss = val_metrics.get('loss', float('inf'))
                acceptance_rate = val_metrics.get('acceptance_rate', 0.0)

                self.training_stats['val_losses'].append(val_loss)
                self.training_stats['acceptance_rates'].append(acceptance_rate)

                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Acceptance: {acceptance_rate*100:.1f}%")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        self.save_checkpoint(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
                if save_path:
                    self.save_checkpoint(save_path)

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'training_stats': self.training_stats,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
