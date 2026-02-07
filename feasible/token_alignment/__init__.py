"""
Token Alignment Module for EventGPT → Video-LLaVA
=================================================

Aligns EventGPT output tokens to Video-LLaVA output tokens for speculative decoding.

Problem: Baseline token-level acceptance is only ~3.4% because EventGPT and
Video-LLaVA generate semantically different descriptions of the same scene.

Target: Improve acceptance to 50%+ for meaningful speculative decoding speedup.

Strategies (ordered by effectiveness):
1. EAGLEFusion - EAGLE-style feature fusion (RECOMMENDED, 50-80% target)
2. SequenceDistillation - Transformer on hidden states (30-50% expected)
3. TokenProjection - Direct token mapping (10-30% expected)
4. LogitAlignment - KL divergence alignment (40-60% expected, requires logits)

Memory Requirements (4090 24GB):
- Feature Extraction: ~14GB (one model at a time)
- Training: ~1-2GB (uses pre-cached features)

Usage:
    # Step 1: Extract features (runs models separately)
    python feasible/token_alignment/extract_features.py ...

    # Step 2: Train alignment (very lightweight)
    python feasible/token_alignment/train_eagle_fusion.py ...

    # Step 3: Evaluate
    python feasible/token_alignment/evaluate_speculative.py ...
"""

from .base import (
    TokenAlignmentConfig,
    BaseTokenAlignmentModule,
    TokenAlignmentDataset,
)

from .logit_alignment import (
    LogitAlignmentConfig,
    LogitAlignmentModule,
    LogitAlignmentTrainer,
)

from .sequence_distillation import (
    SequenceDistillationConfig,
    SequenceDistillationModule,
    SequenceDistillationTrainer,
)

from .token_projection import (
    TokenProjectionConfig,
    TokenProjectionModule,
    TokenProjectionTrainer,
)

from .eagle_fusion import (
    EAGLEFusionConfig,
    EAGLEFusionModule,
    EAGLEFusionTrainer,
)

from .token_adapter import (
    TokenAdapterConfig,
    TokenAdapter,
    TokenAdapterTrainer,
    train_token_adapter,
)

__all__ = [
    # Base
    'TokenAlignmentConfig',
    'BaseTokenAlignmentModule',
    'TokenAlignmentDataset',
    # EAGLE Fusion (RECOMMENDED)
    'EAGLEFusionConfig',
    'EAGLEFusionModule',
    'EAGLEFusionTrainer',
    # Logit Alignment
    'LogitAlignmentConfig',
    'LogitAlignmentModule',
    'LogitAlignmentTrainer',
    # Sequence Distillation
    'SequenceDistillationConfig',
    'SequenceDistillationModule',
    'SequenceDistillationTrainer',
    # Token Projection
    'TokenProjectionConfig',
    'TokenProjectionModule',
    'TokenProjectionTrainer',
    # Token Adapter (lightweight, no hidden states needed)
    'TokenAdapterConfig',
    'TokenAdapter',
    'TokenAdapterTrainer',
    'train_token_adapter',
]
