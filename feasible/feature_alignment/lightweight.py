"""
Lightweight Alignment Module
轻量级对齐模块 - 用于快速原型验证

优点:
    - 极快实现（数小时）
    - 最少数据需求
    - 易于调试
    - 最低推理延迟

缺点:
    - 性能可能不如其他方法
    - 适合验证可行性

使用场景:
    - 快速原型验证
    - 数据/资源有限
    - 验证想法可行性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from .base import (
    BaseAlignmentModule,
    AlignmentConfig,
    AlignmentMetrics,
    MLP,
    FeatureAdapter
)


@dataclass
class LightweightAlignmentConfig(AlignmentConfig):
    """轻量级对齐配置"""
    num_layers: int = 2            # 投影层数
    use_residual: bool = True      # 是否使用残差连接
    use_whitening: bool = False    # 是否使用特征白化
    finetune_on_downstream: bool = True  # 是否在下游任务上微调
    adapter_type: str = 'mlp'      # 适配器类型: 'linear', 'mlp', 'adapter'


class LinearAdapter(nn.Module):
    """线性适配器 - 最简单的特征映射"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)

        # 初始化为近似恒等映射
        nn.init.eye_(self.linear.weight[:min(input_dim, output_dim), :min(input_dim, output_dim)])
        if use_bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BottleneckAdapter(nn.Module):
    """
    瓶颈适配器 - 低秩近似

    使用低秩分解减少参数量
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_normal_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)  # 初始化为零，使残差连接有效

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down_proj(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        return h


class WhiteningLayer(nn.Module):
    """
    特征白化层

    将特征变换为均值为0、协方差为单位矩阵的分布
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # 学习的白化参数
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_cov', torch.eye(dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 更新统计量
            batch_mean = x.mean(dim=0)
            batch_cov = torch.mm((x - batch_mean).T, (x - batch_mean)) / x.shape[0]

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_cov = (1 - self.momentum) * self.running_cov + self.momentum * batch_cov
                self.num_batches_tracked += 1

            mean = batch_mean
            cov = batch_cov
        else:
            mean = self.running_mean
            cov = self.running_cov

        # 白化
        x_centered = x - mean

        # 使用 SVD 进行稳定的白化
        U, S, V = torch.svd(cov + self.eps * torch.eye(self.dim, device=x.device))
        whitening_matrix = U @ torch.diag(1.0 / torch.sqrt(S + self.eps)) @ V.T

        x_whitened = x_centered @ whitening_matrix

        return x_whitened


class LightweightAlignmentModule(BaseAlignmentModule):
    """
    轻量级对齐模块

    用于快速原型验证的简单特征映射

    Example:
        >>> config = LightweightAlignmentConfig(event_dim=768, target_dim=1024)
        >>> alignment = LightweightAlignmentModule(config)
        >>>
        >>> # 训练
        >>> aligned_event = alignment(event_features)
        >>> loss = alignment.compute_loss(event_features, target_features)
        >>>
        >>> # 在下游任务上微调
        >>> alignment.finetune_on_downstream(draft_model, target_model, dataset)
    """

    def __init__(
        self,
        config: LightweightAlignmentConfig,
        event_encoder: Optional[nn.Module] = None
    ):
        super().__init__(config)
        self.config = config
        self.event_encoder = event_encoder

        # 选择适配器类型
        if config.adapter_type == 'linear':
            self.adapter = LinearAdapter(
                config.event_dim,
                config.target_dim
            )
        elif config.adapter_type == 'mlp':
            self.adapter = MLP(
                input_dim=config.event_dim,
                output_dim=config.target_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_layer_norm=config.use_layer_norm
            )
        elif config.adapter_type == 'adapter':
            self.adapter = BottleneckAdapter(
                config.event_dim,
                config.target_dim,
                bottleneck_dim=config.hidden_dim // 4,
                dropout=config.dropout
            )
        else:
            raise ValueError(f"Unknown adapter type: {config.adapter_type}")

        # 可选的白化层
        if config.use_whitening:
            self.whitening = WhiteningLayer(config.event_dim)
        else:
            self.whitening = None

        # 残差投影（如果维度不同）
        if config.use_residual and config.event_dim != config.target_dim:
            self.residual_proj = nn.Linear(config.event_dim, config.target_dim, bias=False)
        elif config.use_residual:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = None

        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1))

        # 冻结 event encoder
        if event_encoder is not None and config.freeze_event_encoder:
            for param in event_encoder.parameters():
                param.requires_grad = False

    def forward(self, event_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            event_features: Event 特征 [batch_size, event_dim]

        Returns:
            aligned_features: 对齐后的特征 [batch_size, target_dim]
        """
        # 白化
        if self.whitening is not None:
            event_features = self.whitening(event_features)

        # 适配器
        adapted = self.adapter(event_features)

        # 残差连接
        if self.residual_proj is not None:
            residual = self.residual_proj(event_features)
            adapted = adapted + 0.1 * residual

        # 缩放
        adapted = adapted * self.scale

        return adapted

    def align_features(
        self,
        event_features: torch.Tensor,
        target_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """对齐特征"""
        aligned_event = self.forward(event_features)
        return aligned_event, target_features

    def compute_loss(
        self,
        event_features: torch.Tensor,
        target_features: torch.Tensor,
        return_metrics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失

        Args:
            event_features: Event 特征
            target_features: Target 特征
            return_metrics: 是否返回额外指标

        Returns:
            loss_dict: 损失字典
        """
        aligned_event = self.forward(event_features)

        # 归一化
        aligned_norm = F.normalize(aligned_event, dim=-1)
        target_norm = F.normalize(target_features, dim=-1)

        # MSE 损失
        mse_loss = F.mse_loss(aligned_norm, target_norm)

        # 余弦相似度损失
        cos_loss = 1 - F.cosine_similarity(aligned_norm, target_norm, dim=-1).mean()

        # 总损失
        total_loss = mse_loss + 0.5 * cos_loss

        loss_dict = {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'cosine_loss': cos_loss
        }

        if return_metrics:
            with torch.no_grad():
                cos_sim = F.cosine_similarity(aligned_norm, target_norm, dim=-1).mean()
                loss_dict['cosine_similarity'] = cos_sim

        return loss_dict

    def get_num_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightAlignmentTrainer:
    """轻量级对齐训练器"""

    def __init__(
        self,
        alignment_module: LightweightAlignmentModule,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        num_workers: int = 4,
        device: str = 'cuda'
    ):
        self.module = alignment_module.to(device)
        self.device = device

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.module.train()

        total_loss = 0
        total_cos_sim = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            event_features = batch['event_features'].to(self.device)
            target_features = batch['target_features'].to(self.device)

            self.optimizer.zero_grad()

            loss_dict = self.module.compute_loss(
                event_features,
                target_features,
                return_metrics=True
            )

            loss = loss_dict['total_loss']
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_cos_sim += loss_dict['cosine_similarity'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cos_sim': f'{loss_dict["cosine_similarity"].item():.4f}'
            })

        return {
            'loss': total_loss / num_batches,
            'cosine_similarity': total_cos_sim / num_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}

        self.module.eval()

        total_loss = 0
        total_cos_sim = 0
        all_event_features = []
        all_target_features = []
        num_batches = 0

        for batch in self.val_loader:
            event_features = batch['event_features'].to(self.device)
            target_features = batch['target_features'].to(self.device)

            loss_dict = self.module.compute_loss(
                event_features,
                target_features,
                return_metrics=True
            )

            total_loss += loss_dict['total_loss'].item()
            total_cos_sim += loss_dict['cosine_similarity'].item()

            # 收集特征用于检索评估
            aligned_event, _ = self.module.align_features(event_features)
            all_event_features.append(aligned_event.cpu())
            all_target_features.append(target_features.cpu())

            num_batches += 1

        # 检索评估
        all_event_features = torch.cat(all_event_features, dim=0)
        all_target_features = torch.cat(all_target_features, dim=0)

        recall_metrics = self.module.evaluate_retrieval(
            all_event_features,
            all_target_features,
            k_values=[1, 5, 10]
        )

        return {
            'val_loss': total_loss / num_batches,
            'val_cos_sim': total_cos_sim / num_batches,
            **recall_metrics
        }

    def train(
        self,
        num_epochs: int = 20,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 5
    ):
        """训练循环"""
        patience_counter = 0

        print("="*70)
        print("Starting Lightweight Alignment Training")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Trainable parameters: {self.module.get_num_parameters():,}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("="*70)

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Cos Sim: {train_metrics['cosine_similarity']:.4f}")

            # 验证
            if self.val_loader:
                val_metrics = self.validate()

                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Val Cos Sim: {val_metrics['val_cos_sim']:.4f}")
                print(f"  R@1: {val_metrics['R@1']:.2%}")
                print(f"  R@5: {val_metrics['R@5']:.2%}")
                print(f"  R@10: {val_metrics['R@10']:.2%}")

                # 学习率调度
                self.scheduler.step(val_metrics['val_loss'])

                # 保存最佳模型
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    if save_path:
                        self.module.save_checkpoint(save_path, self.optimizer)
                        print(f"  Saved best model!")
                else:
                    patience_counter += 1

                # 早停
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print("\n" + "="*70)
        print("Training Complete!")
        if self.val_loader:
            print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("="*70)


def finetune_on_downstream(
    alignment_module: LightweightAlignmentModule,
    draft_model: nn.Module,
    target_model: nn.Module,
    dataset: Dataset,
    num_steps: int = 500,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
) -> LightweightAlignmentModule:
    """
    在下游任务 (speculative decoding) 上微调对齐模块

    Args:
        alignment_module: 对齐模块
        draft_model: Draft model (EventGPT)
        target_model: Target model (LLaVA)
        dataset: 数据集
        num_steps: 微调步数
        learning_rate: 学习率
        device: 设备

    Returns:
        alignment_module: 微调后的对齐模块
    """
    alignment_module = alignment_module.to(device)
    draft_model = draft_model.to(device)
    target_model = target_model.to(device)

    alignment_module.train()
    draft_model.eval()
    target_model.eval()

    # 冻结模型
    for param in draft_model.parameters():
        param.requires_grad = False
    for param in target_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        alignment_module.parameters(),
        lr=learning_rate
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    step = 0
    total_loss = 0

    print("Finetuning on downstream task...")

    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break

            event_features = batch['event_features'].to(device)
            input_ids = batch['input_ids'].to(device)

            # 对齐 event 特征
            aligned_event = alignment_module(event_features)

            # 获取 draft 和 target 的 logits
            with torch.no_grad():
                # Draft model 使用对齐后的特征
                draft_logits = draft_model.get_logits(
                    input_ids, visual_features=aligned_event
                )

                # Target model 使用原始特征（假设有 RGB）
                if 'rgb_features' in batch:
                    target_logits = target_model.get_logits(
                        input_ids, visual_features=batch['rgb_features'].to(device)
                    )
                else:
                    target_logits = draft_logits  # fallback

            # KL 散度损失
            draft_probs = F.softmax(draft_logits / 0.8, dim=-1)
            target_probs = F.softmax(target_logits / 0.8, dim=-1)

            kl_loss = F.kl_div(
                draft_probs.log(),
                target_probs,
                reduction='batchmean'
            )

            optimizer.zero_grad()
            kl_loss.backward()
            optimizer.step()

            total_loss += kl_loss.item()
            step += 1

            if step % 100 == 0:
                avg_loss = total_loss / 100
                print(f"Step {step}/{num_steps}, KL Loss: {avg_loss:.4f}")
                total_loss = 0

    alignment_module.eval()
    print("Finetuning complete!")

    return alignment_module


def train_lightweight_alignment(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    event_dim: int = 768,
    target_dim: int = 1024,
    adapter_type: str = 'mlp',
    num_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    save_path: str = 'lightweight_alignment.pt',
    device: str = 'cuda'
) -> LightweightAlignmentModule:
    """
    便捷函数：训练轻量级对齐模块

    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        event_dim: Event 特征维度
        target_dim: Target 特征维度
        adapter_type: 适配器类型
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_path: 保存路径
        device: 设备

    Returns:
        alignment: 训练好的对齐模块
    """
    # 创建配置
    config = LightweightAlignmentConfig(
        event_dim=event_dim,
        target_dim=target_dim,
        adapter_type=adapter_type
    )

    # 创建模块
    alignment = LightweightAlignmentModule(config)

    # 创建训练器
    trainer = LightweightAlignmentTrainer(
        alignment,
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )

    # 训练
    trainer.train(
        num_epochs=num_epochs,
        save_path=save_path
    )

    return alignment
