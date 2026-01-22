"""
Contrastive Alignment Module (CEIA-style)
对比学习对齐模块 - 基于 CEIA (Contrastive Event-Image Alignment) 方法

性能预期:
    - R@1 提升: +30-40%
    - 接受率提升: +20-30%
    - 需要配对的 Event-RGB 数据

参考论文:
    - CEIA: Contrastive Event-Image Alignment for Scene Understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from .base import (
    BaseAlignmentModule,
    AlignmentConfig,
    AlignmentMetrics,
    MLP,
    ProjectionHead
)


@dataclass
class ContrastiveAlignmentConfig(AlignmentConfig):
    """对比对齐配置"""
    projection_dim: int = 256      # 投影空间维度
    queue_size: int = 65536        # MoCo-style queue 大小
    momentum: float = 0.999        # 动量编码器更新率
    use_hard_negatives: bool = True  # 是否使用困难负样本
    hard_negative_weight: float = 0.5  # 困难负样本权重
    warmup_epochs: int = 5         # 学习率预热轮数
    use_queue: bool = False        # 是否使用 MoCo-style queue


class ContrastiveAlignmentModule(BaseAlignmentModule):
    """
    对比学习对齐模块

    使用 InfoNCE 损失将 Event 特征对齐到 CLIP 特征空间

    Example:
        >>> config = ContrastiveAlignmentConfig(event_dim=768, target_dim=1024)
        >>> alignment = ContrastiveAlignmentModule(config)
        >>>
        >>> # 训练
        >>> aligned_event = alignment(event_features)
        >>> loss = alignment.compute_loss(event_features, rgb_features)
        >>>
        >>> # 推理
        >>> aligned_features = alignment.align_features(event_features)
    """

    def __init__(
        self,
        config: ContrastiveAlignmentConfig,
        event_encoder: Optional[nn.Module] = None,
        target_encoder: Optional[nn.Module] = None
    ):
        super().__init__(config)
        self.config = config

        # 可选的编码器
        self.event_encoder = event_encoder
        self.target_encoder = target_encoder

        # Event 特征投影
        self.event_projector = MLP(
            input_dim=config.event_dim,
            output_dim=config.target_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm
        )

        # 对比学习投影头
        self.event_projection_head = ProjectionHead(
            input_dim=config.target_dim,
            output_dim=config.projection_dim,
            hidden_dim=config.hidden_dim
        )

        self.target_projection_head = ProjectionHead(
            input_dim=config.target_dim,
            output_dim=config.projection_dim,
            hidden_dim=config.hidden_dim
        )

        # MoCo-style queue (可选)
        if config.use_queue:
            self.register_buffer(
                'queue',
                torch.randn(config.projection_dim, config.queue_size)
            )
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # 温度参数 (可学习)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / config.temperature)
        )

        # 冻结编码器
        if event_encoder is not None and config.freeze_event_encoder:
            for param in event_encoder.parameters():
                param.requires_grad = False

        if target_encoder is not None and config.freeze_target_encoder:
            for param in target_encoder.parameters():
                param.requires_grad = False

    def forward(self, event_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将 event features 映射到对齐空间

        Args:
            event_features: Event encoder 输出 [batch_size, event_dim]

        Returns:
            aligned_features: 对齐后的特征 [batch_size, target_dim]
        """
        return self.event_projector(event_features)

    def align_features(
        self,
        event_features: torch.Tensor,
        target_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        对齐特征

        Args:
            event_features: Event 特征 [batch_size, event_dim]
            target_features: 可选的 target 特征

        Returns:
            aligned_event: 对齐后的 event 特征 [batch_size, target_dim]
            target_features: 原始 target 特征（如果提供）
        """
        aligned_event = self.forward(event_features)
        return aligned_event, target_features

    def encode_event(
        self,
        event_stream: torch.Tensor,
        use_encoder: bool = True
    ) -> torch.Tensor:
        """
        编码 Event 数据

        Args:
            event_stream: Event 数据
            use_encoder: 是否使用 event_encoder

        Returns:
            features: Event 特征
        """
        if use_encoder and self.event_encoder is not None:
            with torch.no_grad() if self.config.freeze_event_encoder else torch.enable_grad():
                event_features = self.event_encoder(event_stream)
        else:
            event_features = event_stream

        return self.forward(event_features)

    def encode_target(
        self,
        rgb_image: torch.Tensor,
        use_encoder: bool = True
    ) -> torch.Tensor:
        """
        编码 RGB 图像

        Args:
            rgb_image: RGB 图像
            use_encoder: 是否使用 target_encoder

        Returns:
            features: Target 特征
        """
        if use_encoder and self.target_encoder is not None:
            with torch.no_grad() if self.config.freeze_target_encoder else torch.enable_grad():
                return self.target_encoder(rgb_image)
        return rgb_image

    def compute_loss(
        self,
        event_features: torch.Tensor,
        target_features: torch.Tensor,
        return_metrics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失

        Args:
            event_features: Event 特征 [batch_size, event_dim]
            target_features: Target 特征 [batch_size, target_dim]
            return_metrics: 是否返回额外指标

        Returns:
            loss_dict: 包含各种损失的字典
        """
        # 投影 event 特征到 target 空间
        aligned_event = self.forward(event_features)

        # 对比学习投影
        event_proj = self.event_projection_head(aligned_event)
        target_proj = self.target_projection_head(target_features)

        # 归一化
        event_proj = F.normalize(event_proj, dim=-1)
        target_proj = F.normalize(target_proj, dim=-1)

        # 温度缩放
        logit_scale = self.logit_scale.exp()

        # 计算相似度
        logits_e2t = logit_scale * event_proj @ target_proj.T
        logits_t2e = logits_e2t.T

        # 标签
        batch_size = event_features.shape[0]
        labels = torch.arange(batch_size, device=event_features.device)

        # InfoNCE 损失
        loss_e2t = F.cross_entropy(logits_e2t, labels)
        loss_t2e = F.cross_entropy(logits_t2e, labels)
        contrastive_loss = (loss_e2t + loss_t2e) / 2

        # 特征级 MSE 损失 (辅助)
        mse_loss = F.mse_loss(
            F.normalize(aligned_event, dim=-1),
            F.normalize(target_features, dim=-1)
        )

        # 总损失
        total_loss = contrastive_loss + 0.1 * mse_loss

        loss_dict = {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'mse_loss': mse_loss,
            'logit_scale': logit_scale.detach()
        }

        if return_metrics:
            with torch.no_grad():
                # 计算准确率
                acc_e2t = (logits_e2t.argmax(dim=1) == labels).float().mean()
                acc_t2e = (logits_t2e.argmax(dim=1) == labels).float().mean()
                loss_dict['acc_e2t'] = acc_e2t
                loss_dict['acc_t2e'] = acc_t2e

                # 计算余弦相似度
                cos_sim = F.cosine_similarity(
                    aligned_event, target_features, dim=-1
                ).mean()
                loss_dict['cosine_similarity'] = cos_sim

        return loss_dict

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """更新 MoCo-style queue"""
        if not self.config.use_queue:
            return

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 替换 queue 中的 keys
        if ptr + batch_size > self.config.queue_size:
            # 处理边界情况
            remaining = self.config.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.config.queue_size

        self.queue_ptr[0] = ptr


class ContrastiveAlignmentTrainer:
    """
    对比对齐训练器

    Example:
        >>> trainer = ContrastiveAlignmentTrainer(
        ...     alignment_module,
        ...     train_dataset,
        ...     val_dataset,
        ...     device='cuda'
        ... )
        >>> trainer.train(num_epochs=10)
    """

    def __init__(
        self,
        alignment_module: ContrastiveAlignmentModule,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100  # 将在 train() 中更新
        )

        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_recall': []
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.module.train()

        total_loss = 0
        total_contrastive = 0
        total_mse = 0
        total_acc = 0
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
            total_contrastive += loss_dict['contrastive_loss'].item()
            total_mse += loss_dict['mse_loss'].item()
            total_acc += loss_dict['acc_e2t'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{loss_dict["acc_e2t"].item():.2%}'
            })

        self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'contrastive_loss': total_contrastive / num_batches,
            'mse_loss': total_mse / num_batches,
            'accuracy': total_acc / num_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}

        self.module.eval()

        total_loss = 0
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
            **recall_metrics
        }

    def train(
        self,
        num_epochs: int = 10,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 5
    ):
        """
        训练循环

        Args:
            num_epochs: 训练轮数
            save_path: 检查点保存路径
            early_stopping_patience: 早停轮数
        """
        # 更新调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )

        patience_counter = 0

        print("="*70)
        print("Starting Contrastive Alignment Training")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("="*70)

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            self.training_history['train_loss'].append(train_metrics['loss'])

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Acc: {train_metrics['accuracy']:.2%}")

            # 验证
            if self.val_loader:
                val_metrics = self.validate()
                self.training_history['val_loss'].append(val_metrics['val_loss'])
                self.training_history['val_recall'].append(val_metrics['R@1'])

                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  R@1: {val_metrics['R@1']:.2%}")
                print(f"  R@5: {val_metrics['R@5']:.2%}")
                print(f"  R@10: {val_metrics['R@10']:.2%}")

                # 保存最佳模型
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    if save_path:
                        self.module.save_checkpoint(
                            save_path,
                            self.optimizer
                        )
                        print(f"  Saved best model!")
                else:
                    patience_counter += 1

                # 早停
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("="*70)


class PairedEventRGBDataset(Dataset):
    """
    配对的 Event-RGB 数据集

    Example:
        >>> dataset = PairedEventRGBDataset(
        ...     event_features_path='event_features.pt',
        ...     rgb_features_path='rgb_features.pt'
        ... )
    """

    def __init__(
        self,
        event_features: Optional[torch.Tensor] = None,
        rgb_features: Optional[torch.Tensor] = None,
        event_features_path: Optional[str] = None,
        rgb_features_path: Optional[str] = None,
        transform: Optional[Callable] = None
    ):
        """
        初始化数据集

        Args:
            event_features: Event 特征张量 [num_samples, event_dim]
            rgb_features: RGB 特征张量 [num_samples, target_dim]
            event_features_path: Event 特征文件路径
            rgb_features_path: RGB 特征文件路径
            transform: 可选的数据变换
        """
        if event_features is not None and rgb_features is not None:
            self.event_features = event_features
            self.rgb_features = rgb_features
        elif event_features_path and rgb_features_path:
            self.event_features = torch.load(event_features_path)
            self.rgb_features = torch.load(rgb_features_path)
        else:
            raise ValueError("Must provide either tensors or file paths")

        assert len(self.event_features) == len(self.rgb_features), \
            "Event and RGB features must have same length"

        self.transform = transform

    def __len__(self) -> int:
        return len(self.event_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        event_feat = self.event_features[idx]
        rgb_feat = self.rgb_features[idx]

        if self.transform:
            event_feat = self.transform(event_feat)
            rgb_feat = self.transform(rgb_feat)

        return {
            'event_features': event_feat,
            'target_features': rgb_feat
        }


def train_contrastive_alignment(
    event_encoder: nn.Module,
    clip_encoder: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    event_dim: int = 768,
    target_dim: int = 1024,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    save_path: str = 'contrastive_alignment.pt',
    device: str = 'cuda'
) -> ContrastiveAlignmentModule:
    """
    便捷函数：训练对比对齐模块

    Args:
        event_encoder: Event 编码器
        clip_encoder: CLIP 编码器
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        event_dim: Event 特征维度
        target_dim: Target 特征维度
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_path: 保存路径
        device: 设备

    Returns:
        alignment: 训练好的对齐模块
    """
    # 创建配置
    config = ContrastiveAlignmentConfig(
        event_dim=event_dim,
        target_dim=target_dim,
        freeze_event_encoder=True,
        freeze_target_encoder=True
    )

    # 创建模块
    alignment = ContrastiveAlignmentModule(
        config,
        event_encoder=event_encoder,
        target_encoder=clip_encoder
    )

    # 创建训练器
    trainer = ContrastiveAlignmentTrainer(
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
