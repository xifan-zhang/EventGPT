"""
Feature Alignment Utilities
特征对齐工具集

包含:
    - 快速可行性测试
    - 对齐效果诊断
    - 自动策略选择
    - 数据集工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path


# ============================================================================
# 快速可行性测试
# ============================================================================

def quick_feasibility_test(
    event_encoder: nn.Module,
    clip_encoder: nn.Module,
    test_data: DataLoader,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[str, float]:
    """
    快速测试：Event 和 RGB features 的初始相似度

    判断标准：
        - 如果初始相似度太低 (<0.3)，需要对齐
        - 如果中等 (0.3-0.6)，轻量对齐可能足够
        - 如果较高 (>0.6)，可能不需要复杂对齐

    Args:
        event_encoder: Event 编码器
        clip_encoder: CLIP 编码器
        test_data: 测试数据 DataLoader
        device: 设备
        verbose: 是否打印详细信息

    Returns:
        结果字典包含：avg_similarity, recommendation, needs_alignment
    """
    event_encoder.eval()
    clip_encoder.eval()

    similarities = []
    l2_distances = []

    with torch.no_grad():
        for batch in test_data:
            event_input = batch.get('event_stream', batch.get('event_features'))
            rgb_input = batch.get('rgb_image', batch.get('rgb_features'))

            if event_input is None or rgb_input is None:
                continue

            event_input = event_input.to(device)
            rgb_input = rgb_input.to(device)

            # 编码
            event_feat = event_encoder(event_input)
            rgb_feat = clip_encoder(rgb_input)

            # 处理不同的输出格式
            if isinstance(event_feat, tuple):
                event_feat = event_feat[0]
            if isinstance(rgb_feat, tuple):
                rgb_feat = rgb_feat[0]

            # 如果是序列，取 CLS token 或平均池化
            if event_feat.dim() == 3:
                event_feat = event_feat[:, 0]  # CLS token
            if rgb_feat.dim() == 3:
                rgb_feat = rgb_feat[:, 0]

            # 归一化
            event_feat = F.normalize(event_feat, dim=-1)
            rgb_feat = F.normalize(rgb_feat, dim=-1)

            # 计算相似度
            sim = F.cosine_similarity(event_feat, rgb_feat, dim=-1)
            similarities.extend(sim.cpu().numpy().tolist())

            # 计算 L2 距离
            l2_dist = torch.norm(event_feat - rgb_feat, dim=-1)
            l2_distances.extend(l2_dist.cpu().numpy().tolist())

    avg_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    avg_l2 = np.mean(l2_distances)

    # 判断
    if avg_sim < 0.3:
        recommendation = "需要强对齐 (Contrastive 或 Triple-modal)"
        needs_alignment = "strong"
    elif avg_sim < 0.6:
        recommendation = "需要中度对齐 (Lightweight fine-tune)"
        needs_alignment = "medium"
    else:
        recommendation = "可能只需要轻微调整"
        needs_alignment = "light"

    result = {
        'avg_similarity': avg_sim,
        'std_similarity': std_sim,
        'avg_l2_distance': avg_l2,
        'num_samples': len(similarities),
        'recommendation': recommendation,
        'needs_alignment': needs_alignment
    }

    if verbose:
        print("="*70)
        print("Feature Alignment Feasibility Test")
        print("="*70)
        print(f"\nInitial Feature Similarity: {avg_sim:.4f} (std: {std_sim:.4f})")
        print(f"L2 Distance: {avg_l2:.4f}")
        print(f"Samples tested: {len(similarities)}")
        print(f"\nRecommendation: {recommendation}")
        print("="*70)

    return result


# ============================================================================
# 对齐效果诊断
# ============================================================================

@dataclass
class DiagnosisResult:
    """诊断结果"""
    feature_similarity: float
    acceptance_rate: float
    position_decay: str
    overall_quality: str
    recommendations: List[str]


def diagnose_alignment_quality(
    aligned_draft_model: nn.Module,
    target_model: nn.Module,
    test_cases: List[Dict],
    device: str = 'cuda',
    verbose: bool = True
) -> DiagnosisResult:
    """
    诊断对齐效果

    通过多个指标判断对齐是否充分:
        1. Feature similarity
        2. Acceptance rate
        3. Position decay

    Args:
        aligned_draft_model: 对齐后的 draft model
        target_model: Target model
        test_cases: 测试用例
        device: 设备
        verbose: 是否打印详细信息

    Returns:
        DiagnosisResult 对象
    """
    aligned_draft_model.eval()
    target_model.eval()

    similarities = []
    acceptance_rates = []
    position_acceptances = []

    if verbose:
        print("="*70)
        print("Alignment Quality Diagnosis")
        print("="*70)

    with torch.no_grad():
        for i, case in enumerate(test_cases):
            # 获取特征
            event_input = case.get('event_stream', case.get('event_features'))
            rgb_input = case.get('rgb_image', case.get('rgb_features'))

            if event_input is not None and rgb_input is not None:
                event_input = event_input.to(device)
                rgb_input = rgb_input.to(device)

                # 获取 draft 和 target 的视觉特征
                if hasattr(aligned_draft_model, 'visual_encoder'):
                    draft_feat = aligned_draft_model.visual_encoder(event_input)
                else:
                    draft_feat = aligned_draft_model(event_input)

                if hasattr(target_model, 'visual_encoder'):
                    target_feat = target_model.visual_encoder(rgb_input)
                else:
                    target_feat = target_model(rgb_input)

                # 处理输出格式
                if isinstance(draft_feat, tuple):
                    draft_feat = draft_feat[0]
                if isinstance(target_feat, tuple):
                    target_feat = target_feat[0]

                if draft_feat.dim() == 3:
                    draft_feat = draft_feat[:, 0]
                if target_feat.dim() == 3:
                    target_feat = target_feat[:, 0]

                # 归一化并计算相似度
                draft_feat = F.normalize(draft_feat, dim=-1)
                target_feat = F.normalize(target_feat, dim=-1)

                sim = F.cosine_similarity(draft_feat, target_feat, dim=-1)
                similarities.extend(sim.cpu().numpy().tolist())

    # 计算指标
    avg_sim = np.mean(similarities) if similarities else 0.0

    # 模拟接受率（基于相似度估算）
    # 实际应用中应该运行 speculative decoding 来获取真实接受率
    estimated_acc_rate = min(0.8, avg_sim * 1.1)

    # 判断位置衰减
    if avg_sim > 0.7:
        position_decay = "slow"
        position_decay_desc = "缓慢衰减（好）"
    elif avg_sim > 0.5:
        position_decay = "medium"
        position_decay_desc = "中等衰减"
    else:
        position_decay = "fast"
        position_decay_desc = "快速衰减（对齐问题）"

    # 综合评估
    recommendations = []

    if avg_sim > 0.7 and estimated_acc_rate > 0.65:
        overall_quality = "excellent"
        overall_desc = "对齐质量优秀！"
    elif avg_sim > 0.5 and estimated_acc_rate > 0.50:
        overall_quality = "medium"
        overall_desc = "对齐质量中等"
        recommendations.extend([
            "增加训练数据",
            "尝试更强的对齐方法",
            "调整 gamma (可能太大)"
        ])
    else:
        overall_quality = "poor"
        overall_desc = "对齐质量不足，必须改进"
        recommendations.extend([
            "使用 Contrastive Alignment",
            "收集更多配对数据",
            "检查 Event encoder 质量"
        ])

    result = DiagnosisResult(
        feature_similarity=avg_sim,
        acceptance_rate=estimated_acc_rate,
        position_decay=position_decay,
        overall_quality=overall_quality,
        recommendations=recommendations
    )

    if verbose:
        print(f"\n1. Feature Similarity: {avg_sim:.4f}")
        if avg_sim > 0.7:
            print("   [OK] 很好")
        elif avg_sim > 0.5:
            print("   [WARN] 中等，可以改进")
        else:
            print("   [FAIL] 较差，需要更好的对齐")

        print(f"\n2. Estimated Acceptance Rate: {estimated_acc_rate:.2%}")
        if estimated_acc_rate > 0.65:
            print("   [OK] 很好")
        elif estimated_acc_rate > 0.50:
            print("   [WARN] 可接受，但有改进空间")
        else:
            print("   [FAIL] 太低，对齐不充分")

        print(f"\n3. Position Decay: {position_decay_desc}")

        print("\n" + "="*70)
        print(f"Diagnosis Summary: {overall_desc}")

        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")

        print("="*70)

    return result


# ============================================================================
# 自动策略选择
# ============================================================================

@dataclass
class StrategyRecommendation:
    """策略推荐"""
    strategy: str
    confidence: str
    reasons: List[str]
    estimated_improvement: str
    caveats: Optional[List[str]] = None


def auto_select_alignment_strategy(
    has_paired_data: bool,
    data_size: int,
    has_e2vid: bool,
    compute_budget: str = 'medium',  # 'low', 'medium', 'high'
    target_performance: str = 'good'  # 'quick', 'good', 'best'
) -> StrategyRecommendation:
    """
    自动选择最适合的对齐策略

    Args:
        has_paired_data: 是否有配对 Event-RGB 数据
        data_size: 数据集大小
        has_e2vid: 是否有预训练 E2VID
        compute_budget: 计算预算
        target_performance: 目标性能

    Returns:
        StrategyRecommendation 对象
    """
    # 决策逻辑
    if target_performance == 'best' and has_paired_data and data_size > 10000:
        return StrategyRecommendation(
            strategy='Triple-Modal Alignment',
            confidence='high',
            reasons=[
                '数据充足 (>10K)',
                '追求最佳性能',
                '可以利用大规模 image-text 数据'
            ],
            estimated_improvement='+30-40%',
            caveats=['训练时间较长 (2-3天)', '需要较多 GPU 内存']
        )

    elif has_paired_data and data_size > 5000:
        return StrategyRecommendation(
            strategy='Contrastive Alignment (CEIA)',
            confidence='high',
            reasons=[
                f'有足够配对数据 ({data_size})',
                '性能优秀',
                '训练稳定'
            ],
            estimated_improvement='+20-30%',
            caveats=['训练时间 1-2 天']
        )

    elif has_e2vid:
        return StrategyRecommendation(
            strategy='Reconstruction Bridging',
            confidence='medium',
            reasons=[
                '有预训练 E2VID',
                '无需配对数据',
                '快速部署'
            ],
            estimated_improvement='+10-20%',
            caveats=['增加 5-10ms 延迟']
        )

    elif has_paired_data and data_size > 1000:
        return StrategyRecommendation(
            strategy='Lightweight Alignment + Fine-tune',
            confidence='medium',
            reasons=[
                f'有少量配对数据 ({data_size})',
                '快速原型',
                '简单有效'
            ],
            estimated_improvement='+10-15%',
            caveats=['效果可能受限']
        )

    else:
        return StrategyRecommendation(
            strategy='Lightweight Alignment',
            confidence='low',
            reasons=[
                '数据受限',
                '资源受限',
                '快速原型验证'
            ],
            estimated_improvement='+5-15%',
            caveats=['先验证可行性，再考虑更强方法']
        )


def print_strategy_recommendation(recommendation: StrategyRecommendation):
    """打印策略推荐"""
    print("="*70)
    print("Alignment Strategy Recommendation")
    print("="*70)
    print(f"\nRecommended Strategy: {recommendation.strategy}")
    print(f"Confidence: {recommendation.confidence}")
    print(f"Expected Improvement: {recommendation.estimated_improvement}")

    print("\nReasons:")
    for reason in recommendation.reasons:
        print(f"  - {reason}")

    if recommendation.caveats:
        print("\nCaveats:")
        for caveat in recommendation.caveats:
            print(f"  ! {caveat}")

    print("="*70)


# ============================================================================
# 数据集工具
# ============================================================================

class AlignmentDataset(Dataset):
    """
    通用对齐数据集

    支持从文件或张量加载配对的 Event-RGB 特征
    """

    def __init__(
        self,
        event_features: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
        event_features_path: Optional[str] = None,
        target_features_path: Optional[str] = None,
        text_features: Optional[torch.Tensor] = None,
        text_features_path: Optional[str] = None,
        normalize: bool = False
    ):
        """
        初始化数据集

        可以通过张量或文件路径初始化

        Args:
            event_features: Event 特征张量
            target_features: Target 特征张量
            event_features_path: Event 特征文件路径
            target_features_path: Target 特征文件路径
            text_features: Text 特征张量 (可选)
            text_features_path: Text 特征文件路径 (可选)
            normalize: 是否归一化特征
        """
        # 加载 event features
        if event_features is not None:
            self.event_features = event_features
        elif event_features_path:
            self.event_features = torch.load(event_features_path)
        else:
            raise ValueError("Must provide event_features or event_features_path")

        # 加载 target features
        if target_features is not None:
            self.target_features = target_features
        elif target_features_path:
            self.target_features = torch.load(target_features_path)
        else:
            raise ValueError("Must provide target_features or target_features_path")

        # 加载 text features (可选)
        if text_features is not None:
            self.text_features = text_features
        elif text_features_path:
            self.text_features = torch.load(text_features_path)
        else:
            self.text_features = None

        # 验证
        assert len(self.event_features) == len(self.target_features), \
            f"Mismatch: {len(self.event_features)} event vs {len(self.target_features)} target"

        if self.text_features is not None:
            assert len(self.event_features) == len(self.text_features), \
                f"Mismatch: {len(self.event_features)} event vs {len(self.text_features)} text"

        # 归一化
        if normalize:
            self.event_features = F.normalize(self.event_features.float(), dim=-1)
            self.target_features = F.normalize(self.target_features.float(), dim=-1)
            if self.text_features is not None:
                self.text_features = F.normalize(self.text_features.float(), dim=-1)

    def __len__(self) -> int:
        return len(self.event_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'event_features': self.event_features[idx],
            'target_features': self.target_features[idx]
        }

        if self.text_features is not None:
            item['text_features'] = self.text_features[idx]

        return item

    @classmethod
    def from_directory(
        cls,
        directory: str,
        event_filename: str = 'event_features.pt',
        target_filename: str = 'target_features.pt',
        text_filename: str = 'text_features.pt',
        normalize: bool = False
    ) -> 'AlignmentDataset':
        """
        从目录加载数据集

        Args:
            directory: 数据目录
            event_filename: Event 特征文件名
            target_filename: Target 特征文件名
            text_filename: Text 特征文件名
            normalize: 是否归一化

        Returns:
            AlignmentDataset 实例
        """
        dir_path = Path(directory)

        event_path = dir_path / event_filename
        target_path = dir_path / target_filename
        text_path = dir_path / text_filename

        return cls(
            event_features_path=str(event_path) if event_path.exists() else None,
            target_features_path=str(target_path) if target_path.exists() else None,
            text_features_path=str(text_path) if text_path.exists() else None,
            normalize=normalize
        )

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple['AlignmentDataset', 'AlignmentDataset']:
        """
        划分训练集和验证集

        Args:
            train_ratio: 训练集比例
            seed: 随机种子

        Returns:
            (train_dataset, val_dataset)
        """
        np.random.seed(seed)
        indices = np.random.permutation(len(self))
        split_idx = int(len(self) * train_ratio)

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_dataset = AlignmentDataset(
            event_features=self.event_features[train_indices],
            target_features=self.target_features[train_indices],
            text_features=self.text_features[train_indices] if self.text_features is not None else None
        )

        val_dataset = AlignmentDataset(
            event_features=self.event_features[val_indices],
            target_features=self.target_features[val_indices],
            text_features=self.text_features[val_indices] if self.text_features is not None else None
        )

        return train_dataset, val_dataset


def extract_and_save_features(
    event_encoder: nn.Module,
    target_encoder: nn.Module,
    dataloader: DataLoader,
    save_dir: str,
    device: str = 'cuda',
    event_key: str = 'event_stream',
    target_key: str = 'rgb_image'
):
    """
    提取并保存特征

    Args:
        event_encoder: Event 编码器
        target_encoder: Target 编码器 (如 CLIP)
        dataloader: 数据加载器
        save_dir: 保存目录
        device: 设备
        event_key: batch 中 event 数据的 key
        target_key: batch 中 target 数据的 key
    """
    event_encoder.eval()
    target_encoder.eval()

    all_event_features = []
    all_target_features = []

    print("Extracting features...")

    with torch.no_grad():
        for batch in dataloader:
            event_input = batch[event_key].to(device)
            target_input = batch[target_key].to(device)

            # 编码
            event_feat = event_encoder(event_input)
            target_feat = target_encoder(target_input)

            # 处理输出格式
            if isinstance(event_feat, tuple):
                event_feat = event_feat[0]
            if isinstance(target_feat, tuple):
                target_feat = target_feat[0]

            if event_feat.dim() == 3:
                event_feat = event_feat[:, 0]
            if target_feat.dim() == 3:
                target_feat = target_feat[:, 0]

            all_event_features.append(event_feat.cpu())
            all_target_features.append(target_feat.cpu())

    # 合并
    event_features = torch.cat(all_event_features, dim=0)
    target_features = torch.cat(all_target_features, dim=0)

    # 保存
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(event_features, save_path / 'event_features.pt')
    torch.save(target_features, save_path / 'target_features.pt')

    print(f"Saved {len(event_features)} features to {save_dir}")
    print(f"  Event features shape: {event_features.shape}")
    print(f"  Target features shape: {target_features.shape}")


# ============================================================================
# 评估工具
# ============================================================================

def evaluate_alignment_comprehensive(
    alignment_module: nn.Module,
    test_dataset: Dataset,
    device: str = 'cuda',
    batch_size: int = 32
) -> Dict[str, float]:
    """
    全面评估对齐模块

    Args:
        alignment_module: 对齐模块
        test_dataset: 测试数据集
        device: 设备
        batch_size: 批次大小

    Returns:
        评估指标字典
    """
    alignment_module.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_aligned = []
    all_target = []
    total_mse = 0
    total_cos_sim = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            event_features = batch['event_features'].to(device)
            target_features = batch['target_features'].to(device)

            aligned = alignment_module(event_features)

            # 归一化
            aligned_norm = F.normalize(aligned, dim=-1)
            target_norm = F.normalize(target_features, dim=-1)

            # MSE
            mse = F.mse_loss(aligned_norm, target_norm)
            total_mse += mse.item()

            # 余弦相似度
            cos_sim = F.cosine_similarity(aligned_norm, target_norm, dim=-1).mean()
            total_cos_sim += cos_sim.item()

            all_aligned.append(aligned_norm.cpu())
            all_target.append(target_norm.cpu())

            num_batches += 1

    # 检索评估
    all_aligned = torch.cat(all_aligned, dim=0)
    all_target = torch.cat(all_target, dim=0)

    similarity = all_aligned @ all_target.T
    num_samples = len(all_aligned)

    # R@K
    recall_at = {}
    for k in [1, 5, 10]:
        _, topk_indices = similarity.topk(k, dim=1)
        correct = torch.arange(num_samples).unsqueeze(1).expand_as(topk_indices)
        hits = (topk_indices == correct).any(dim=1).float()
        recall_at[f'R@{k}'] = hits.mean().item()

    return {
        'mse': total_mse / num_batches,
        'cosine_similarity': total_cos_sim / num_batches,
        **recall_at
    }


def compare_alignment_strategies(
    strategies: Dict[str, nn.Module],
    test_dataset: Dataset,
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    比较多个对齐策略

    Args:
        strategies: 策略名称到模块的字典
        test_dataset: 测试数据集
        device: 设备

    Returns:
        各策略的评估结果
    """
    results = {}

    print("="*70)
    print("Comparing Alignment Strategies")
    print("="*70)

    for name, module in strategies.items():
        print(f"\nEvaluating: {name}")
        metrics = evaluate_alignment_comprehensive(module, test_dataset, device)
        results[name] = metrics

        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  Cos Sim: {metrics['cosine_similarity']:.4f}")
        print(f"  R@1: {metrics['R@1']:.2%}")
        print(f"  R@5: {metrics['R@5']:.2%}")
        print(f"  R@10: {metrics['R@10']:.2%}")

    # 找最佳
    best_strategy = max(results.keys(), key=lambda k: results[k]['R@1'])
    print(f"\nBest strategy by R@1: {best_strategy}")

    return results
