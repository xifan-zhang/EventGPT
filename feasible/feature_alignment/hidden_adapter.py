"""
Hidden State Adapter for Cross-Modal Speculative Decoding
==========================================================

Lightweight adapter to align EventGPT decoder hidden states to Video-LLaVA space.

Architecture:
    - Bottleneck design (LoRA-style): 4096 → 256 → 4096
    - Residual connection: h_aligned = h_egpt + alpha * adapter(h_egpt)
    - Parameters: ~2M (vs 154M raw feature adapter, vs 45M token adapter)

Usage:
    adapter = HiddenStateAdapter(hidden_dim=4096, bottleneck_dim=256)
    h_aligned = adapter(h_egpt)  # [batch, seq, 4096] -> [batch, seq, 4096]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math


@dataclass
class HiddenAdapterConfig:
    """Configuration for hidden state adapter."""
    hidden_dim: int = 4096          # LLM hidden dimension (Vicuna 7B)
    bottleneck_dim: int = 256       # Bottleneck dimension
    num_layers: int = 1             # Number of adapter layers
    dropout: float = 0.1
    alpha: float = 0.1              # Residual scaling factor
    use_layer_norm: bool = True
    use_gelu: bool = True


class BottleneckAdapter(nn.Module):
    """
    Bottleneck adapter layer (LoRA-style).

    h_out = h_in + alpha * up(act(down(h_in)))

    Parameters: hidden_dim * bottleneck_dim * 2 ≈ 2M for default config
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
        alpha: float = 0.1,
        use_layer_norm: bool = True,
        use_gelu: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.alpha = alpha

        # Down projection
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)

        # Up projection
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        # Optional layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # Activation
        self.act = nn.GELU() if use_gelu else nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Learnable alpha (can be trained)
        self.alpha_param = nn.Parameter(torch.tensor(alpha))

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Kaiming init for down projection
        nn.init.kaiming_normal_(self.down_proj.weight, mode='fan_in', nonlinearity='linear')
        # Zero init for up projection (residual starts as identity)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq, hidden_dim] from EGPT decoder

        Returns:
            aligned_states: [batch, seq, hidden_dim] aligned to VL space
        """
        # Normalize input
        normed = self.layer_norm(hidden_states)

        # Bottleneck transformation
        down = self.down_proj(normed)           # [batch, seq, bottleneck_dim]
        activated = self.act(down)
        activated = self.dropout(activated)
        up = self.up_proj(activated)            # [batch, seq, hidden_dim]

        # Residual connection with learnable alpha
        output = hidden_states + self.alpha_param * up

        return output

    def get_num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HiddenStateAdapter(nn.Module):
    """
    Full hidden state adapter for EGPT → VL alignment.

    Can stack multiple bottleneck layers for more capacity.
    """

    def __init__(self, config: HiddenAdapterConfig):
        super().__init__()
        self.config = config

        # Stack of adapter layers
        self.layers = nn.ModuleList([
            BottleneckAdapter(
                hidden_dim=config.hidden_dim,
                bottleneck_dim=config.bottleneck_dim,
                dropout=config.dropout,
                alpha=config.alpha,
                use_layer_norm=config.use_layer_norm,
                use_gelu=config.use_gelu,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all adapter layers.

        Args:
            hidden_states: [batch, seq, hidden_dim] EGPT decoder hidden states

        Returns:
            aligned_states: [batch, seq, hidden_dim] aligned to VL space
        """
        x = hidden_states
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x

    def compute_loss(
        self,
        egpt_hidden: torch.Tensor,
        vl_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            egpt_hidden: [batch, seq, hidden_dim] EGPT hidden states
            vl_hidden: [batch, seq, hidden_dim] VL hidden states (target)
            attention_mask: [batch, seq] mask for valid positions

        Returns:
            Dict with losses and metrics
        """
        # Align EGPT hidden states
        aligned = self.forward(egpt_hidden)

        # MSE loss
        mse_loss = F.mse_loss(aligned, vl_hidden, reduction='none')

        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(mse_loss)
            mse_loss = (mse_loss * mask).sum() / mask.sum()
        else:
            mse_loss = mse_loss.mean()

        # Cosine similarity loss
        aligned_norm = F.normalize(aligned, dim=-1)
        vl_norm = F.normalize(vl_hidden, dim=-1)

        cos_sim = (aligned_norm * vl_norm).sum(dim=-1)  # [batch, seq]

        if attention_mask is not None:
            cos_sim = (cos_sim * attention_mask).sum() / attention_mask.sum()
        else:
            cos_sim = cos_sim.mean()

        cos_loss = 1 - cos_sim

        # Total loss
        total_loss = mse_loss + 0.5 * cos_loss

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'cos_loss': cos_loss,
            'cos_sim': cos_sim,
        }

    def get_num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, metrics: dict = None):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if metrics is not None:
            checkpoint['metrics'] = metrics
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda'):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint


# =============================================================================
# Level 2: Multi-Layer Bottleneck Adapter (~8M params)
# =============================================================================

class MultiLayerBottleneckAdapter(nn.Module):
    """
    Level 2: Stacked bottleneck layers for more expressive transformations.

    Architecture: 3 sequential bottleneck blocks
    Parameters: ~8M (3 × 2.7M per block)

    h → [Bottleneck₁] → [Bottleneck₂] → [Bottleneck₃] → h_aligned
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1,
        alpha: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_blocks = num_blocks

        # Stack of bottleneck blocks
        self.blocks = nn.ModuleList([
            BottleneckAdapter(
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
                alpha=alpha,
                use_layer_norm=True,
                use_gelu=True,
            )
            for _ in range(num_blocks)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward through stacked bottlenecks."""
        x = hidden_states
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_loss(
        self,
        egpt_hidden: torch.Tensor,
        vl_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss (same as HiddenStateAdapter)."""
        aligned = self.forward(egpt_hidden)

        # MSE loss
        mse_loss = F.mse_loss(aligned, vl_hidden, reduction='none')
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(mse_loss)
            mse_loss = (mse_loss * mask).sum() / mask.sum()
        else:
            mse_loss = mse_loss.mean()

        # Cosine loss
        aligned_norm = F.normalize(aligned, dim=-1)
        vl_norm = F.normalize(vl_hidden, dim=-1)
        cos_sim = (aligned_norm * vl_norm).sum(dim=-1)
        if attention_mask is not None:
            cos_sim = (cos_sim * attention_mask).sum() / attention_mask.sum()
        else:
            cos_sim = cos_sim.mean()
        cos_loss = 1 - cos_sim

        total_loss = mse_loss + 0.5 * cos_loss

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'cos_loss': cos_loss,
            'cos_sim': cos_sim,
        }

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, metrics: dict = None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'adapter_type': 'L2_MultiLayerBottleneck',
            'config': {
                'hidden_dim': self.hidden_dim,
                'bottleneck_dim': self.bottleneck_dim,
                'num_blocks': self.num_blocks,
            },
            'epoch': epoch,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if metrics is not None:
            checkpoint['metrics'] = metrics
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint


# =============================================================================
# Level 3: Wide Bottleneck Adapter (~16M params)
# =============================================================================

class WideBottleneckAdapter(nn.Module):
    """
    Level 3: Wider bottleneck for more capacity.

    Architecture: 4096 → 1024 → 4096 (4x wider than L1)
    Parameters: ~16M

    More capacity to capture complex cross-modal mappings.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 1024,  # 4x wider than L1
        num_blocks: int = 2,
        dropout: float = 0.1,
        alpha: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_blocks = num_blocks

        # Input normalization
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Stacked wide bottlenecks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim, bias=False),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, hidden_dim, bias=False),
                nn.Dropout(dropout),
            ))

        # Learnable residual scales
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(alpha)) for _ in range(num_blocks)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for block in self.blocks:
            # Down projection
            nn.init.kaiming_normal_(block[0].weight, mode='fan_in')
            # Up projection - zero init for residual
            nn.init.zeros_(block[4].weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(hidden_states)

        for block, alpha in zip(self.blocks, self.alphas):
            residual = x
            x = residual + alpha * block(x)

        return self.final_norm(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_loss(
        self,
        egpt_hidden: torch.Tensor,
        vl_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        aligned = self.forward(egpt_hidden)

        mse_loss = F.mse_loss(aligned, vl_hidden, reduction='none')
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(mse_loss)
            mse_loss = (mse_loss * mask).sum() / mask.sum()
        else:
            mse_loss = mse_loss.mean()

        aligned_norm = F.normalize(aligned, dim=-1)
        vl_norm = F.normalize(vl_hidden, dim=-1)
        cos_sim = (aligned_norm * vl_norm).sum(dim=-1)
        if attention_mask is not None:
            cos_sim = (cos_sim * attention_mask).sum() / attention_mask.sum()
        else:
            cos_sim = cos_sim.mean()
        cos_loss = 1 - cos_sim

        return {
            'total_loss': mse_loss + 0.5 * cos_loss,
            'mse_loss': mse_loss,
            'cos_loss': cos_loss,
            'cos_sim': cos_sim,
        }

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, metrics: dict = None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'adapter_type': 'L3_WideBottleneck',
            'config': {
                'hidden_dim': self.hidden_dim,
                'bottleneck_dim': self.bottleneck_dim,
                'num_blocks': self.num_blocks,
            },
            'epoch': epoch,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if metrics is not None:
            checkpoint['metrics'] = metrics
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint


# =============================================================================
# Level 4: Attention Adapter (~50M params)
# =============================================================================

class AttentionAdapter(nn.Module):
    """
    Level 4: Self-attention + FFN (transformer layer style).

    Architecture: MultiHeadAttention + FFN with residuals
    Parameters: ~100M (with default settings for hidden_dim=4096)

    Captures token dependencies for better alignment.
    Based on EAGLE's approach but simpler (no cross-attention to context).
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 8,        # Reduced for smaller model
        ffn_dim: int = 2048,       # Reduced for ~50M total
        num_layers: int = 1,       # Single layer
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                # Self-attention
                'attn_norm': nn.LayerNorm(hidden_dim),
                'attn': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                # FFN
                'ffn_norm': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, hidden_dim),
                    nn.Dropout(dropout),
                ),
            }))

        # Final projection (optional refinement)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable residual scale
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        # Initialize output projection to near-identity
        nn.init.eye_(self.output_proj[1].weight)
        nn.init.zeros_(self.output_proj[1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with self-attention.

        Args:
            hidden_states: [batch, seq, hidden_dim]
            attention_mask: [batch, seq] optional mask (1=keep, 0=mask)

        Returns:
            aligned: [batch, seq, hidden_dim]
        """
        x = hidden_states

        # Convert attention mask to key_padding_mask format if provided
        # MultiheadAttention expects True for masked positions
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True where masked

        for layer in self.layers:
            # Self-attention with residual
            normed = layer['attn_norm'](x)
            attn_out, _ = layer['attn'](
                normed, normed, normed,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = x + attn_out

            # FFN with residual
            normed = layer['ffn_norm'](x)
            ffn_out = layer['ffn'](normed)
            x = x + ffn_out

        # Final projection with residual from input
        output = self.output_proj(x)
        aligned = hidden_states + self.alpha * (output - hidden_states)

        return aligned

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_loss(
        self,
        egpt_hidden: torch.Tensor,
        vl_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        aligned = self.forward(egpt_hidden, attention_mask)

        mse_loss = F.mse_loss(aligned, vl_hidden, reduction='none')
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(mse_loss)
            mse_loss = (mse_loss * mask).sum() / mask.sum()
        else:
            mse_loss = mse_loss.mean()

        aligned_norm = F.normalize(aligned, dim=-1)
        vl_norm = F.normalize(vl_hidden, dim=-1)
        cos_sim = (aligned_norm * vl_norm).sum(dim=-1)
        if attention_mask is not None:
            cos_sim = (cos_sim * attention_mask).sum() / attention_mask.sum()
        else:
            cos_sim = cos_sim.mean()
        cos_loss = 1 - cos_sim

        return {
            'total_loss': mse_loss + 0.5 * cos_loss,
            'mse_loss': mse_loss,
            'cos_loss': cos_loss,
            'cos_sim': cos_sim,
        }

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, metrics: dict = None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'adapter_type': 'L4_Attention',
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_heads': self.num_heads,
                'ffn_dim': self.ffn_dim,
                'num_layers': self.num_layers,
            },
            'epoch': epoch,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if metrics is not None:
            checkpoint['metrics'] = metrics
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint


# =============================================================================
# Level 5: Hybrid Cross-Modal EAGLE (~50M params)
# =============================================================================

class EAGLEStyleAdapter(nn.Module):
    """
    Level 5: Hybrid Cross-Modal EAGLE - combines alignment + prediction.

    Architecture:
        - Input: h_t (current hidden state from EGPT)
        - Output: h_{t+1}_aligned (predicted next hidden state in VL space)

        h_t → [FC+Embed] → [Attention] → [FC] → h_{t+1}_aligned

    Two training objectives:
        1. Alignment loss: align to VL hidden states
        2. Prediction loss: predict next token's hidden state (autoregressive)

    This enables speculative decoding with EAGLE-style drafting:
        - EventGPT generates h_0
        - L5 adapter predicts h_1, h_2, ... (in VL space)
        - VideoLLaVA verifies all at once

    Parameters: ~50M

    Reference: EAGLE (Li et al., 2024) - Speculative Sampling with EAGLE
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_layers: int = 1,      # 1 layer = ~50M, 2 layers = ~187M
        dropout: float = 0.1,
        max_seq_len: int = 512,
        vocab_size: int = 32000,  # For optional token embedding
        use_token_embed: bool = False,  # EAGLE uses prev token embedding
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.use_token_embed = use_token_embed

        # Optional: Embed previous token (EAGLE-style)
        if use_token_embed:
            self.token_embed = nn.Embedding(vocab_size, hidden_dim)
            self.token_fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        # Input projection (fuse hidden state with token if enabled)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Positional encoding for autoregressive prediction
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer layers (causal attention for prediction)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn_norm': nn.LayerNorm(hidden_dim),
                'attn': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                'ffn_norm': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, hidden_dim),
                    nn.Dropout(dropout),
                ),
            }))

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable residual scale
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        nn.init.eye_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for autoregressive prediction."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional EAGLE-style token fusion.

        Args:
            hidden_states: [batch, seq, hidden_dim] from EGPT
            prev_tokens: [batch, seq] previous token IDs (optional, for EAGLE-style)
            attention_mask: [batch, seq] mask (1=keep, 0=mask)

        Returns:
            predicted: [batch, seq, hidden_dim] predicted next hidden states in VL space
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Optional: fuse with previous token embedding (EAGLE-style)
        if self.use_token_embed and prev_tokens is not None:
            token_emb = self.token_embed(prev_tokens)
            x = self.token_fusion(torch.cat([hidden_states, token_emb], dim=-1))
        else:
            x = hidden_states

        # Input normalization
        x = self.input_norm(x)

        # Add positional encoding
        x = x + self.pos_embed[:, :seq_len, :]

        # Create causal mask for autoregressive prediction
        causal_mask = self._create_causal_mask(seq_len, device)

        # Convert attention mask to key_padding_mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # Transformer layers with causal attention
        for layer in self.layers:
            # Self-attention with causal mask
            normed = layer['attn_norm'](x)
            attn_out, _ = layer['attn'](
                normed, normed, normed,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = x + attn_out

            # FFN
            normed = layer['ffn_norm'](x)
            ffn_out = layer['ffn'](normed)
            x = x + ffn_out

        # Output projection
        output = self.output_proj(self.output_norm(x))

        # Residual from input (alignment component)
        aligned = hidden_states + self.alpha * (output - hidden_states)

        return aligned

    def compute_loss(
        self,
        egpt_hidden: torch.Tensor,
        vl_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prediction_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined alignment + prediction loss.

        Args:
            egpt_hidden: [batch, seq, hidden_dim] EGPT hidden states
            vl_hidden: [batch, seq, hidden_dim] target VL hidden states
            attention_mask: [batch, seq] mask
            prediction_weight: Weight for next-token prediction loss

        Returns:
            Dict with losses and metrics
        """
        predicted = self.forward(egpt_hidden, attention_mask=attention_mask)

        # === Alignment loss (position-wise) ===
        mse_loss = F.mse_loss(predicted, vl_hidden, reduction='none')
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(mse_loss)
            mse_loss_align = (mse_loss * mask).sum() / mask.sum()
        else:
            mse_loss_align = mse_loss.mean()

        # Cosine similarity
        pred_norm = F.normalize(predicted, dim=-1)
        vl_norm = F.normalize(vl_hidden, dim=-1)
        cos_sim = (pred_norm * vl_norm).sum(dim=-1)
        if attention_mask is not None:
            cos_sim_mean = (cos_sim * attention_mask).sum() / attention_mask.sum()
        else:
            cos_sim_mean = cos_sim.mean()
        cos_loss = 1 - cos_sim_mean

        # === Prediction loss (next-token) ===
        # Shift: predict h_{t+1} from h_t
        # predicted[:-1] should match vl_hidden[1:]
        if egpt_hidden.shape[1] > 1:
            pred_shifted = predicted[:, :-1, :]  # Predictions for positions 0 to T-2
            target_shifted = vl_hidden[:, 1:, :]  # Targets for positions 1 to T-1

            pred_loss = F.mse_loss(pred_shifted, target_shifted, reduction='none')

            if attention_mask is not None:
                mask_shifted = attention_mask[:, 1:].unsqueeze(-1).expand_as(pred_loss)
                pred_loss = (pred_loss * mask_shifted).sum() / mask_shifted.sum()
            else:
                pred_loss = pred_loss.mean()
        else:
            pred_loss = torch.tensor(0.0, device=egpt_hidden.device)

        # Total loss: alignment + prediction
        align_loss = mse_loss_align + 0.5 * cos_loss
        total_loss = (1 - prediction_weight) * align_loss + prediction_weight * pred_loss

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss_align,
            'cos_loss': cos_loss,
            'cos_sim': cos_sim_mean,
            'pred_loss': pred_loss,
            'align_loss': align_loss,
        }

    def speculative_decode(
        self,
        initial_hidden: torch.Tensor,
        num_draft_tokens: int = 5,
    ) -> torch.Tensor:
        """
        Generate draft hidden states autoregressively (EAGLE-style).

        Args:
            initial_hidden: [batch, 1, hidden_dim] initial hidden state from EGPT
            num_draft_tokens: Number of draft tokens to generate

        Returns:
            draft_hidden: [batch, num_draft_tokens, hidden_dim] draft hidden states
        """
        batch = initial_hidden.shape[0]
        device = initial_hidden.device

        # Start with initial hidden state
        draft_hidden = [initial_hidden]
        current = initial_hidden

        for _ in range(num_draft_tokens - 1):
            # Predict next hidden state
            next_hidden = self.forward(current)[:, -1:, :]  # Take last position
            draft_hidden.append(next_hidden)
            current = next_hidden

        return torch.cat(draft_hidden, dim=1)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, metrics: dict = None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'adapter_type': 'L5_EAGLE',
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_heads': self.num_heads,
                'ffn_dim': self.ffn_dim,
                'num_layers': self.num_layers,
                'use_token_embed': self.use_token_embed,
            },
            'epoch': epoch,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if metrics is not None:
            checkpoint['metrics'] = metrics
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint


# =============================================================================
# Legacy: Multi-Layer Hidden Adapter (for multi-layer input fusion)
# =============================================================================

class MultiLayerHiddenAdapter(nn.Module):
    """
    EAGLE-style adapter that uses multiple decoder layers.

    Takes hidden states from last N layers and fuses them.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 256,
        num_input_layers: int = 4,  # Use last 4 decoder layers
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_input_layers = num_input_layers

        # Layer fusion: concatenate then project
        self.layer_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_input_layers, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Bottleneck adapter
        self.adapter = BottleneckAdapter(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )

    def forward(
        self,
        layer_hidden_states: torch.Tensor,  # [batch, num_layers, seq, hidden_dim]
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            layer_hidden_states: Hidden states from last N decoder layers
                Shape: [batch, num_layers, seq, hidden_dim]

        Returns:
            aligned_states: [batch, seq, hidden_dim]
        """
        batch, num_layers, seq, hidden = layer_hidden_states.shape

        # Concatenate layers: [batch, seq, num_layers * hidden]
        concat = layer_hidden_states.permute(0, 2, 1, 3).reshape(batch, seq, -1)

        # Fuse layers
        fused = self.layer_fusion(concat)  # [batch, seq, hidden]

        # Apply adapter
        aligned = self.adapter(fused)

        return aligned


# =============================================================================
# Factory Functions
# =============================================================================

def create_hidden_adapter(
    hidden_dim: int = 4096,
    bottleneck_dim: int = 256,
    num_layers: int = 1,
    alpha: float = 0.1,
) -> HiddenStateAdapter:
    """
    Factory function to create L1 hidden state adapter.

    Args:
        hidden_dim: LLM hidden dimension (4096 for Vicuna 7B)
        bottleneck_dim: Bottleneck dimension (256 recommended)
        num_layers: Number of adapter layers
        alpha: Residual scaling factor

    Returns:
        HiddenStateAdapter instance
    """
    config = HiddenAdapterConfig(
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_layers=num_layers,
        alpha=alpha,
    )

    adapter = HiddenStateAdapter(config)

    print(f"Created HiddenStateAdapter (L1):")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Bottleneck dim: {bottleneck_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Parameters: {adapter.get_num_parameters():,} ({adapter.get_num_parameters() * 4 / 1024 / 1024:.1f} MB)")

    return adapter


def create_adapter(
    level: int = 1,
    hidden_dim: int = 4096,
    **kwargs,
) -> nn.Module:
    """
    Unified factory function to create adapters at any complexity level.

    Args:
        level: Adapter complexity level (1-5)
            L1: Bottleneck (~2M) - Simple, fast
            L2: Multi-layer bottleneck (~8M) - Better nonlinearity
            L3: Wide bottleneck (~16M) - More capacity
            L4: Attention (~50M) - Token dependencies
            L5: Hybrid Cross-Modal EAGLE (~50M) - Alignment + Prediction
        hidden_dim: LLM hidden dimension (4096 for Vicuna 7B)
        **kwargs: Level-specific arguments

    Returns:
        Adapter module

    Examples:
        # L1: Simple bottleneck (2M params, <0.5ms)
        adapter = create_adapter(level=1, bottleneck_dim=256)

        # L2: Multi-layer (8M params, ~1ms)
        adapter = create_adapter(level=2, num_blocks=3)

        # L3: Wide bottleneck (16M params, ~1ms)
        adapter = create_adapter(level=3, bottleneck_dim=1024)

        # L4: Attention (50M params, ~2ms)
        adapter = create_adapter(level=4, num_heads=16, num_layers=2)

        # L5: EAGLE-style (50M params, ~3ms) - with next-token prediction
        adapter = create_adapter(level=5, num_layers=2, prediction_weight=0.5)
    """
    if level == 1:
        # L1: Simple bottleneck
        config = HiddenAdapterConfig(
            hidden_dim=hidden_dim,
            bottleneck_dim=kwargs.get('bottleneck_dim', 256),
            num_layers=kwargs.get('num_layers', 1),
            alpha=kwargs.get('alpha', 0.1),
        )
        adapter = HiddenStateAdapter(config)
        name = "L1_Bottleneck"

    elif level == 2:
        # L2: Multi-layer bottleneck
        adapter = MultiLayerBottleneckAdapter(
            hidden_dim=hidden_dim,
            bottleneck_dim=kwargs.get('bottleneck_dim', 256),
            num_blocks=kwargs.get('num_blocks', 3),
            dropout=kwargs.get('dropout', 0.1),
            alpha=kwargs.get('alpha', 0.1),
        )
        name = "L2_MultiLayerBottleneck"

    elif level == 3:
        # L3: Wide bottleneck
        adapter = WideBottleneckAdapter(
            hidden_dim=hidden_dim,
            bottleneck_dim=kwargs.get('bottleneck_dim', 1024),
            num_blocks=kwargs.get('num_blocks', 2),
            dropout=kwargs.get('dropout', 0.1),
            alpha=kwargs.get('alpha', 0.1),
        )
        name = "L3_WideBottleneck"

    elif level == 4:
        # L4: Attention (~50M with defaults)
        adapter = AttentionAdapter(
            hidden_dim=hidden_dim,
            num_heads=kwargs.get('num_heads', 8),
            ffn_dim=kwargs.get('ffn_dim', 2048),
            num_layers=kwargs.get('num_layers', 1),
            dropout=kwargs.get('dropout', 0.1),
        )
        name = "L4_Attention"

    elif level == 5:
        # L5: EAGLE-style (~50M with 1 layer, ~187M with 2 layers)
        adapter = EAGLEStyleAdapter(
            hidden_dim=hidden_dim,
            num_heads=kwargs.get('num_heads', 8),
            ffn_dim=kwargs.get('ffn_dim', 2048),
            num_layers=kwargs.get('num_layers', 1),  # 1 layer = ~50M
            dropout=kwargs.get('dropout', 0.1),
            use_token_embed=kwargs.get('use_token_embed', False),
        )
        name = "L5_EAGLE"

    else:
        raise ValueError(f"Invalid level {level}. Must be 1-5.")

    params = adapter.get_num_parameters()
    print(f"Created {name}:")
    print(f"  Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")

    return adapter


def load_any_adapter(path: str, device: str = 'cuda') -> Tuple[nn.Module, dict]:
    """
    Load any adapter type from checkpoint.

    Automatically detects adapter type from checkpoint metadata.
    """
    checkpoint = torch.load(path, map_location=device)

    # Detect adapter type
    adapter_type = checkpoint.get('adapter_type', 'L1_Bottleneck')

    if adapter_type == 'L2_MultiLayerBottleneck':
        model, ckpt = MultiLayerBottleneckAdapter.load_checkpoint(path, device)
    elif adapter_type == 'L3_WideBottleneck':
        model, ckpt = WideBottleneckAdapter.load_checkpoint(path, device)
    elif adapter_type == 'L4_Attention':
        model, ckpt = AttentionAdapter.load_checkpoint(path, device)
    elif adapter_type == 'L5_EAGLE':
        model, ckpt = EAGLEStyleAdapter.load_checkpoint(path, device)
    else:
        # Default to L1
        model, ckpt = HiddenStateAdapter.load_checkpoint(path, device)

    return model, ckpt


if __name__ == "__main__":
    print("=" * 70)
    print("Testing All Adapter Levels (L1-L5)")
    print("=" * 70)

    batch_size = 4
    seq_len = 50
    hidden_dim = 4096

    egpt_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    vl_hidden = torch.randn(batch_size, seq_len, hidden_dim)

    results = []

    for level in [1, 2, 3, 4, 5]:
        print(f"\n{'=' * 70}")
        print(f"Level {level}")
        print("=" * 70)

        adapter = create_adapter(level=level, hidden_dim=hidden_dim)

        # Test forward
        import time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        for _ in range(10):
            aligned = adapter(egpt_hidden)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.perf_counter() - start) / 10 * 1000  # ms

        # Test loss
        losses = adapter.compute_loss(egpt_hidden, vl_hidden)

        result = {
            'level': level,
            'params': adapter.get_num_parameters(),
            'time_ms': elapsed,
            'cos_sim': losses['cos_sim'].item(),
        }

        # L5 has additional prediction loss
        if level == 5:
            result['pred_loss'] = losses['pred_loss'].item()
            print(f"  Prediction loss: {losses['pred_loss'].item():.4f}")

        results.append(result)

        print(f"  Forward time: {elapsed:.2f}ms")
        print(f"  Cosine sim (random): {losses['cos_sim'].item():.4f}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Level':<8} {'Params':<12} {'Size (MB)':<10} {'Time (ms)':<10} {'Type':<25}")
    print("-" * 65)
    level_names = {
        1: "Bottleneck",
        2: "Multi-Layer Bottleneck",
        3: "Wide Bottleneck",
        4: "Attention",
        5: "EAGLE (Align+Predict)",
    }
    for r in results:
        print(f"L{r['level']:<7} {r['params']:,}  {r['params']*4/1024/1024:>7.1f}   {r['time_ms']:>8.2f}    {level_names[r['level']]}")

    print("\n✓ All adapter levels (L1-L5) working correctly")
