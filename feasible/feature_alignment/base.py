"""
Feature Alignment Base Module
特征对齐基础模块 - 定义通用接口和基类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class AlignmentConfig:
    """对齐模块配置"""
    event_dim: int = 768           # Event encoder 输出维度
    target_dim: int = 1024         # Target (CLIP) encoder 输出维度
    hidden_dim: int = 512          # 隐藏层维度
    num_layers: int = 2            # MLP 层数
    dropout: float = 0.1           # Dropout 率
    temperature: float = 0.07     # 对比学习温度
    use_layer_norm: bool = True    # 是否使用 LayerNorm
    freeze_event_encoder: bool = False  # 是否冻结 event encoder
    freeze_target_encoder: bool = True  # 是否冻结 target encoder


@dataclass
class AlignmentMetrics:
    """对齐效果指标"""
    feature_similarity: float      # 特征余弦相似度
    acceptance_rate: float         # Speculative decoding 接受率
    retrieval_recall_at_1: float   # 检索 R@1
    retrieval_recall_at_5: float   # 检索 R@5
    retrieval_recall_at_10: float  # 检索 R@10
    kl_divergence: Optional[float] = None
    training_loss: Optional[float] = None


class BaseAlignmentModule(nn.Module, ABC):
    """
    特征对齐模块基类

    所有对齐策略都需要继承这个基类并实现 forward 和 align_features 方法
    """

    def __init__(self, config: AlignmentConfig):
        super().__init__()
        self.config = config
        self.training_stats = {
            'losses': [],
            'similarities': [],
            'epochs': 0
        }

    @abstractmethod
    def forward(self, event_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将 event features 映射到对齐空间

        Args:
            event_features: Event encoder 输出 [batch_size, event_dim]

        Returns:
            aligned_features: 对齐后的特征 [batch_size, target_dim]
        """
        pass

    @abstractmethod
    def align_features(
        self,
        event_features: torch.Tensor,
        target_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐特征

        Args:
            event_features: Event 特征
            target_features: 可选的 target 特征（用于训练）

        Returns:
            aligned_event: 对齐后的 event 特征
            aligned_target: 对齐后的 target 特征（如果提供）
        """
        pass

    def compute_similarity(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        计算特征相似度矩阵

        Args:
            features_a: 第一组特征 [batch_size, dim]
            features_b: 第二组特征 [batch_size, dim]
            normalize: 是否归一化

        Returns:
            similarity_matrix: 相似度矩阵 [batch_size, batch_size]
        """
        if normalize:
            features_a = F.normalize(features_a, dim=-1)
            features_b = F.normalize(features_b, dim=-1)

        return torch.matmul(features_a, features_b.T)

    def compute_contrastive_loss(
        self,
        event_features: torch.Tensor,
        target_features: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算对比学习损失 (InfoNCE)

        Args:
            event_features: Event 特征 [batch_size, dim]
            target_features: Target 特征 [batch_size, dim]
            temperature: 温度参数

        Returns:
            loss: 对比损失
        """
        if temperature is None:
            temperature = self.config.temperature

        # 归一化
        event_features = F.normalize(event_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)

        # 计算相似度
        logits = torch.matmul(event_features, target_features.T) / temperature

        # 标签：对角线为正样本
        batch_size = event_features.shape[0]
        labels = torch.arange(batch_size, device=event_features.device)

        # 双向对比损失
        loss_e2t = F.cross_entropy(logits, labels)
        loss_t2e = F.cross_entropy(logits.T, labels)

        return (loss_e2t + loss_t2e) / 2

    def compute_mse_loss(
        self,
        event_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 MSE 损失

        Args:
            event_features: Event 特征
            target_features: Target 特征

        Returns:
            loss: MSE 损失
        """
        return F.mse_loss(event_features, target_features)

    def evaluate_retrieval(
        self,
        event_features: torch.Tensor,
        target_features: torch.Tensor,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        评估检索性能

        Args:
            event_features: Event 特征 [num_samples, dim]
            target_features: Target 特征 [num_samples, dim]
            k_values: 要计算的 R@k 值

        Returns:
            metrics: 包含各 R@k 的字典
        """
        # 归一化
        event_features = F.normalize(event_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)

        # 计算相似度矩阵
        similarity = torch.matmul(event_features, target_features.T)

        num_samples = event_features.shape[0]
        metrics = {}

        for k in k_values:
            # 获取 top-k 索引
            _, topk_indices = similarity.topk(k, dim=1)

            # 检查正确匹配是否在 top-k 中
            correct = torch.arange(num_samples, device=event_features.device)
            correct = correct.unsqueeze(1).expand_as(topk_indices)

            hits = (topk_indices == correct).any(dim=1).float()
            recall = hits.mean().item()

            metrics[f'R@{k}'] = recall

        return metrics

    def save_checkpoint(self, filepath: str, optimizer: Optional[Any] = None):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, optimizer: Optional[Any] = None):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {filepath}")


class MLP(nn.Module):
    """多层感知机模块"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 激活函数
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            act_fn = nn.GELU()

        # 构建层
        layers = []

        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_fn)
        layers.append(nn.Dropout(dropout))

        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))

        # 输出层
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # 单层情况
            layers = [nn.Linear(input_dim, output_dim)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ProjectionHead(nn.Module):
    """投影头模块 - 用于对比学习"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class FeatureAdapter(nn.Module):
    """
    特征适配器 - 轻量级特征转换

    用于快速原型验证
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool = True,
        use_scale: bool = True
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.use_scale = use_scale

        if use_scale:
            self.scale = nn.Parameter(torch.ones(1))

        # 初始化
        nn.init.xavier_uniform_(self.linear.weight)
        if use_bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.use_scale:
            out = out * self.scale
        return out
