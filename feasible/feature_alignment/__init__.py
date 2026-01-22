"""
Feature Alignment Module for EventGPT
特征对齐模块 - 用于将 Event 特征对齐到 RGB/CLIP 特征空间

策略选择指南:
    1. 有 >10K 配对数据 + 追求最佳性能 → Triple-Modal Alignment
    2. 有 5-10K 配对数据 → Contrastive Alignment (CEIA)
    3. 有预训练 E2VID，无配对数据 → Reconstruction Bridging
    4. 数据/资源有限，快速验证 → Lightweight Alignment
    5. 有 <5K 配对数据 → Lightweight + Fine-tune

使用示例:

    # 1. 快速可行性测试
    from feature_alignment import quick_feasibility_test
    result = quick_feasibility_test(event_encoder, clip_encoder, test_loader)

    # 2. 自动选择策略
    from feature_alignment import auto_select_alignment_strategy
    recommendation = auto_select_alignment_strategy(
        has_paired_data=True,
        data_size=8000,
        has_e2vid=False,
        target_performance='good'
    )

    # 3. 训练对齐模块
    from feature_alignment import train_contrastive_alignment
    alignment = train_contrastive_alignment(
        event_encoder, clip_encoder,
        train_dataset, val_dataset,
        num_epochs=10
    )

    # 4. 诊断对齐效果
    from feature_alignment import diagnose_alignment_quality
    result = diagnose_alignment_quality(aligned_model, target_model, test_cases)
"""

# Base classes
from .base import (
    AlignmentConfig,
    AlignmentMetrics,
    BaseAlignmentModule,
    MLP,
    ProjectionHead,
    FeatureAdapter
)

# Contrastive Alignment (CEIA-style)
from .contrastive import (
    ContrastiveAlignmentConfig,
    ContrastiveAlignmentModule,
    ContrastiveAlignmentTrainer,
    PairedEventRGBDataset,
    train_contrastive_alignment
)

# Reconstruction Bridging (E2VID-based)
from .reconstruction import (
    ReconstructionBridgingConfig,
    ReconstructionBridgingModule,
    ReconstructionBridgingWithRefinement,
    E2VIDWrapper,
    SimpleE2VIDNet,
    create_reconstruction_bridging,
    finetune_e2vid_on_events
)

# Triple-Modal Alignment (E-CLIP style)
from .triple_modal import (
    TripleModalAlignmentConfig,
    TripleModalAlignmentModule,
    TripleModalAlignmentTrainer,
    TripleModalEncoder,
    CrossModalAttention,
    TripleModalDataset,
    train_triple_modal_alignment
)

# Lightweight Alignment
from .lightweight import (
    LightweightAlignmentConfig,
    LightweightAlignmentModule,
    LightweightAlignmentTrainer,
    LinearAdapter,
    BottleneckAdapter,
    WhiteningLayer,
    train_lightweight_alignment,
    finetune_on_downstream
)

# Utilities
from .utils import (
    quick_feasibility_test,
    diagnose_alignment_quality,
    DiagnosisResult,
    auto_select_alignment_strategy,
    print_strategy_recommendation,
    StrategyRecommendation,
    AlignmentDataset,
    extract_and_save_features,
    evaluate_alignment_comprehensive,
    compare_alignment_strategies
)

# Diagnosis (original)
from .diagnosis import (
    FeatureAlignmentDiagnostics,
    DiagnosticMetrics
)


__all__ = [
    # Base
    'AlignmentConfig',
    'AlignmentMetrics',
    'BaseAlignmentModule',
    'MLP',
    'ProjectionHead',
    'FeatureAdapter',

    # Contrastive
    'ContrastiveAlignmentConfig',
    'ContrastiveAlignmentModule',
    'ContrastiveAlignmentTrainer',
    'PairedEventRGBDataset',
    'train_contrastive_alignment',

    # Reconstruction
    'ReconstructionBridgingConfig',
    'ReconstructionBridgingModule',
    'ReconstructionBridgingWithRefinement',
    'E2VIDWrapper',
    'SimpleE2VIDNet',
    'create_reconstruction_bridging',
    'finetune_e2vid_on_events',

    # Triple-Modal
    'TripleModalAlignmentConfig',
    'TripleModalAlignmentModule',
    'TripleModalAlignmentTrainer',
    'TripleModalEncoder',
    'CrossModalAttention',
    'TripleModalDataset',
    'train_triple_modal_alignment',

    # Lightweight
    'LightweightAlignmentConfig',
    'LightweightAlignmentModule',
    'LightweightAlignmentTrainer',
    'LinearAdapter',
    'BottleneckAdapter',
    'WhiteningLayer',
    'train_lightweight_alignment',
    'finetune_on_downstream',

    # Utils
    'quick_feasibility_test',
    'diagnose_alignment_quality',
    'DiagnosisResult',
    'auto_select_alignment_strategy',
    'print_strategy_recommendation',
    'StrategyRecommendation',
    'AlignmentDataset',
    'extract_and_save_features',
    'evaluate_alignment_comprehensive',
    'compare_alignment_strategies',

    # Diagnosis
    'FeatureAlignmentDiagnostics',
    'DiagnosticMetrics'
]


# Version
__version__ = '1.0.0'
