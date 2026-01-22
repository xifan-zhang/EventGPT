"""
Triple-Modal Alignment Module (E-CLIP style)
三模态对齐模块 - 同时对齐 Event-Image-Text

性能预期:
    - 最佳性能（R@1 +40%+）
    - 支持零样本分类
    - 可扩展到大规模数据

特点:
    - 利用大规模 image-text 数据
    - Event-Image-Text 联合对齐
    - 支持多种下游任务

参考:
    - E-CLIP: Event-CLIP for Scene Understanding
    - EventCLIP: Adapting CLIP for Event-based Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any, Union
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
class TripleModalAlignmentConfig(AlignmentConfig):
    """三模态对齐配置"""
    text_dim: int = 512            # Text encoder 输出维度
    projection_dim: int = 256      # 共享投影空间维度
    use_text_alignment: bool = True  # 是否使用文本对齐
    text_loss_weight: float = 0.5  # 文本损失权重
    image_loss_weight: float = 1.0  # 图像损失权重
    use_cross_modal_attention: bool = True  # 是否使用跨模态注意力
    num_attention_heads: int = 8   # 注意力头数
    use_prompt_tuning: bool = False  # 是否使用提示调优


class CrossModalAttention(nn.Module):
    """跨模态注意力模块"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        跨模态注意力

        Args:
            query: [batch_size, seq_len_q, dim]
            key: [batch_size, seq_len_k, dim]
            value: [batch_size, seq_len_v, dim]
            mask: 可选的注意力掩码

        Returns:
            output: [batch_size, seq_len_q, dim]
        """
        batch_size = query.shape[0]

        # 投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 重塑为多头
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 输出
        out = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        out = self.out_proj(out)

        return out


class TripleModalEncoder(nn.Module):
    """
    三模态编码器

    将 Event、Image、Text 编码到共享空间
    """

    def __init__(self, config: TripleModalAlignmentConfig):
        super().__init__()
        self.config = config

        # Event 编码投影
        self.event_projector = nn.Sequential(
            nn.Linear(config.event_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.projection_dim)
        )

        # Image 编码投影
        self.image_projector = nn.Sequential(
            nn.Linear(config.target_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.projection_dim)
        )

        # Text 编码投影
        if config.use_text_alignment:
            self.text_projector = nn.Sequential(
                nn.Linear(config.text_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.projection_dim)
            )
        else:
            self.text_projector = None

        # 跨模态注意力
        if config.use_cross_modal_attention:
            self.event_image_attention = CrossModalAttention(
                config.projection_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout
            )
            self.event_text_attention = CrossModalAttention(
                config.projection_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout
            ) if config.use_text_alignment else None
        else:
            self.event_image_attention = None
            self.event_text_attention = None

        # 可学习的温度参数
        self.logit_scale_ei = nn.Parameter(torch.ones([]) * np.log(1 / config.temperature))
        self.logit_scale_et = nn.Parameter(torch.ones([]) * np.log(1 / config.temperature))
        self.logit_scale_it = nn.Parameter(torch.ones([]) * np.log(1 / config.temperature))

    def encode_event(self, event_features: torch.Tensor) -> torch.Tensor:
        """编码 Event 特征"""
        return F.normalize(self.event_projector(event_features), dim=-1)

    def encode_image(self, image_features: torch.Tensor) -> torch.Tensor:
        """编码 Image 特征"""
        return F.normalize(self.image_projector(image_features), dim=-1)

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """编码 Text 特征"""
        if self.text_projector is None:
            raise ValueError("Text projector not initialized")
        return F.normalize(self.text_projector(text_features), dim=-1)

    def forward(
        self,
        event_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            event_features: Event 特征 [batch_size, event_dim]
            image_features: Image 特征 [batch_size, target_dim]
            text_features: Text 特征 [batch_size, text_dim]

        Returns:
            encodings: 包含各模态编码的字典
        """
        encodings = {}

        # 编码 Event
        event_enc = self.encode_event(event_features)
        encodings['event'] = event_enc

        # 编码 Image
        if image_features is not None:
            image_enc = self.encode_image(image_features)
            encodings['image'] = image_enc

            # 跨模态注意力
            if self.event_image_attention is not None:
                # [batch, 1, dim] -> 用于注意力
                event_enc_seq = event_enc.unsqueeze(1)
                image_enc_seq = image_enc.unsqueeze(1)

                attended_event = self.event_image_attention(
                    event_enc_seq, image_enc_seq, image_enc_seq
                ).squeeze(1)

                # 残差连接
                encodings['event_attended'] = F.normalize(
                    event_enc + 0.1 * attended_event, dim=-1
                )

        # 编码 Text
        if text_features is not None and self.text_projector is not None:
            text_enc = self.encode_text(text_features)
            encodings['text'] = text_enc

            # Event-Text 跨模态注意力
            if self.event_text_attention is not None:
                event_enc_seq = event_enc.unsqueeze(1)
                text_enc_seq = text_enc.unsqueeze(1)

                attended_event_text = self.event_text_attention(
                    event_enc_seq, text_enc_seq, text_enc_seq
                ).squeeze(1)

                encodings['event_text_attended'] = F.normalize(
                    event_enc + 0.1 * attended_event_text, dim=-1
                )

        return encodings


class TripleModalAlignmentModule(BaseAlignmentModule):
    """
    三模态对齐模块

    同时对齐 Event-Image-Text 三种模态

    Example:
        >>> config = TripleModalAlignmentConfig()
        >>> alignment = TripleModalAlignmentModule(config)
        >>>
        >>> # 训练
        >>> loss = alignment.compute_loss(event_feat, image_feat, text_feat)
        >>>
        >>> # 推理
        >>> aligned_event = alignment(event_features)
    """

    def __init__(
        self,
        config: TripleModalAlignmentConfig,
        event_encoder: Optional[nn.Module] = None,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None
    ):
        super().__init__(config)
        self.config = config

        # 外部编码器
        self.event_encoder = event_encoder
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # 三模态编码器
        self.triple_encoder = TripleModalEncoder(config)

        # 用于推理的直接投影 (event_dim -> target_dim)
        self.inference_projector = MLP(
            input_dim=config.event_dim,
            output_dim=config.target_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

        # 冻结编码器
        self._freeze_encoders()

    def _freeze_encoders(self):
        """冻结外部编码器"""
        if self.event_encoder is not None and self.config.freeze_event_encoder:
            for param in self.event_encoder.parameters():
                param.requires_grad = False

        if self.image_encoder is not None and self.config.freeze_target_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, event_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将 event features 映射到 target 空间

        Args:
            event_features: Event 特征 [batch_size, event_dim]

        Returns:
            aligned_features: 对齐后的特征 [batch_size, target_dim]
        """
        return self.inference_projector(event_features)

    def align_features(
        self,
        event_features: torch.Tensor,
        target_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """对齐特征"""
        aligned_event = self.forward(event_features)
        return aligned_event, target_features

    def encode_all_modalities(
        self,
        event_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        编码所有模态到共享空间

        用于训练和评估
        """
        return self.triple_encoder(event_features, image_features, text_features)

    def compute_loss(
        self,
        event_features: torch.Tensor,
        image_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        计算三模态对齐损失

        Args:
            event_features: Event 特征 [batch_size, event_dim]
            image_features: Image 特征 [batch_size, target_dim]
            text_features: Text 特征 [batch_size, text_dim]
            return_metrics: 是否返回额外指标

        Returns:
            loss_dict: 包含各种损失的字典
        """
        batch_size = event_features.shape[0]
        labels = torch.arange(batch_size, device=event_features.device)

        # 编码所有模态
        encodings = self.encode_all_modalities(
            event_features, image_features, text_features
        )

        event_enc = encodings['event']
        image_enc = encodings['image']

        loss_dict = {}

        # 1. Event-Image 对比损失
        logit_scale_ei = self.triple_encoder.logit_scale_ei.exp()
        logits_ei = logit_scale_ei * event_enc @ image_enc.T

        loss_e2i = F.cross_entropy(logits_ei, labels)
        loss_i2e = F.cross_entropy(logits_ei.T, labels)
        loss_ei = (loss_e2i + loss_i2e) / 2

        loss_dict['loss_event_image'] = loss_ei

        total_loss = self.config.image_loss_weight * loss_ei

        # 2. Event-Text 对比损失 (可选)
        if text_features is not None and 'text' in encodings:
            text_enc = encodings['text']
            logit_scale_et = self.triple_encoder.logit_scale_et.exp()
            logits_et = logit_scale_et * event_enc @ text_enc.T

            loss_e2t = F.cross_entropy(logits_et, labels)
            loss_t2e = F.cross_entropy(logits_et.T, labels)
            loss_et = (loss_e2t + loss_t2e) / 2

            loss_dict['loss_event_text'] = loss_et
            total_loss = total_loss + self.config.text_loss_weight * loss_et

            # 3. Image-Text 对比损失 (辅助)
            logit_scale_it = self.triple_encoder.logit_scale_it.exp()
            logits_it = logit_scale_it * image_enc @ text_enc.T

            loss_i2t = F.cross_entropy(logits_it, labels)
            loss_t2i = F.cross_entropy(logits_it.T, labels)
            loss_it = (loss_i2t + loss_t2i) / 2

            loss_dict['loss_image_text'] = loss_it
            total_loss = total_loss + 0.5 * loss_it

        # 4. 推理投影器的 MSE 损失
        projected_event = self.inference_projector(event_features)
        mse_loss = F.mse_loss(
            F.normalize(projected_event, dim=-1),
            F.normalize(image_features, dim=-1)
        )
        loss_dict['loss_mse'] = mse_loss
        total_loss = total_loss + 0.1 * mse_loss

        loss_dict['total_loss'] = total_loss

        # 额外指标
        if return_metrics:
            with torch.no_grad():
                # Event-Image 准确率
                acc_ei = (logits_ei.argmax(dim=1) == labels).float().mean()
                loss_dict['acc_event_image'] = acc_ei

                # 余弦相似度
                cos_sim = F.cosine_similarity(event_enc, image_enc, dim=-1).mean()
                loss_dict['cosine_similarity'] = cos_sim

                if text_features is not None and 'text' in encodings:
                    acc_et = (logits_et.argmax(dim=1) == labels).float().mean()
                    loss_dict['acc_event_text'] = acc_et

        return loss_dict

    def zero_shot_classify(
        self,
        event_features: torch.Tensor,
        class_text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        零样本分类

        Args:
            event_features: Event 特征 [batch_size, event_dim]
            class_text_features: 类别文本特征 [num_classes, text_dim]

        Returns:
            predictions: 预测的类别索引 [batch_size]
        """
        encodings = self.encode_all_modalities(event_features)
        event_enc = encodings['event']

        # 编码类别文本
        class_enc = self.triple_encoder.encode_text(class_text_features)

        # 计算相似度
        similarity = event_enc @ class_enc.T

        return similarity.argmax(dim=-1)


class TripleModalAlignmentTrainer:
    """三模态对齐训练器"""

    def __init__(
        self,
        alignment_module: TripleModalAlignmentModule,
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
            pin_memory=True,
            drop_last=True
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
            weight_decay=weight_decay,
            betas=(0.9, 0.98)
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )

        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc_ei': [],
            'val_acc_et': []
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.module.train()

        total_loss = 0
        total_acc_ei = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            event_features = batch['event_features'].to(self.device)
            image_features = batch['image_features'].to(self.device)
            text_features = batch.get('text_features')
            if text_features is not None:
                text_features = text_features.to(self.device)

            self.optimizer.zero_grad()

            loss_dict = self.module.compute_loss(
                event_features,
                image_features,
                text_features,
                return_metrics=True
            )

            loss = loss_dict['total_loss']
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_acc_ei += loss_dict['acc_event_image'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc_ei': f'{loss_dict["acc_event_image"].item():.2%}'
            })

        self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'acc_event_image': total_acc_ei / num_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}

        self.module.eval()

        total_loss = 0
        total_acc_ei = 0
        total_acc_et = 0
        all_event_enc = []
        all_image_enc = []
        num_batches = 0

        for batch in self.val_loader:
            event_features = batch['event_features'].to(self.device)
            image_features = batch['image_features'].to(self.device)
            text_features = batch.get('text_features')
            if text_features is not None:
                text_features = text_features.to(self.device)

            loss_dict = self.module.compute_loss(
                event_features,
                image_features,
                text_features,
                return_metrics=True
            )

            total_loss += loss_dict['total_loss'].item()
            total_acc_ei += loss_dict['acc_event_image'].item()
            if 'acc_event_text' in loss_dict:
                total_acc_et += loss_dict['acc_event_text'].item()

            # 收集编码用于检索评估
            encodings = self.module.encode_all_modalities(event_features, image_features)
            all_event_enc.append(encodings['event'].cpu())
            all_image_enc.append(encodings['image'].cpu())

            num_batches += 1

        # 检索评估
        all_event_enc = torch.cat(all_event_enc, dim=0)
        all_image_enc = torch.cat(all_image_enc, dim=0)

        recall_metrics = self.module.evaluate_retrieval(
            all_event_enc,
            all_image_enc,
            k_values=[1, 5, 10]
        )

        return {
            'val_loss': total_loss / num_batches,
            'val_acc_ei': total_acc_ei / num_batches,
            'val_acc_et': total_acc_et / num_batches if total_acc_et > 0 else 0,
            **recall_metrics
        }

    def train(
        self,
        num_epochs: int = 15,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 5
    ):
        """训练循环"""
        patience_counter = 0

        print("="*70)
        print("Starting Triple-Modal Alignment Training")
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
            print(f"  Train Acc (E-I): {train_metrics['acc_event_image']:.2%}")

            # 验证
            if self.val_loader:
                val_metrics = self.validate()
                self.training_history['val_loss'].append(val_metrics['val_loss'])
                self.training_history['val_acc_ei'].append(val_metrics['val_acc_ei'])

                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Val Acc (E-I): {val_metrics['val_acc_ei']:.2%}")
                print(f"  R@1: {val_metrics['R@1']:.2%}")
                print(f"  R@5: {val_metrics['R@5']:.2%}")
                print(f"  R@10: {val_metrics['R@10']:.2%}")

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
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("="*70)


class TripleModalDataset(Dataset):
    """
    三模态数据集

    支持 Event-Image-Text 配对数据
    """

    def __init__(
        self,
        event_features: torch.Tensor,
        image_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        transform: Optional[Any] = None
    ):
        """
        初始化数据集

        Args:
            event_features: Event 特征 [num_samples, event_dim]
            image_features: Image 特征 [num_samples, image_dim]
            text_features: Text 特征 [num_samples, text_dim] (可选)
        """
        self.event_features = event_features
        self.image_features = image_features
        self.text_features = text_features
        self.transform = transform

        assert len(event_features) == len(image_features), \
            "Event and Image features must have same length"

        if text_features is not None:
            assert len(event_features) == len(text_features), \
                "All modalities must have same length"

    def __len__(self) -> int:
        return len(self.event_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'event_features': self.event_features[idx],
            'image_features': self.image_features[idx]
        }

        if self.text_features is not None:
            item['text_features'] = self.text_features[idx]

        return item


def train_triple_modal_alignment(
    event_encoder: nn.Module,
    image_encoder: nn.Module,
    text_encoder: Optional[nn.Module],
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    event_dim: int = 768,
    image_dim: int = 1024,
    text_dim: int = 512,
    num_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    save_path: str = 'triple_modal_alignment.pt',
    device: str = 'cuda'
) -> TripleModalAlignmentModule:
    """
    便捷函数：训练三模态对齐模块

    Args:
        event_encoder: Event 编码器
        image_encoder: Image 编码器
        text_encoder: Text 编码器 (可选)
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        event_dim: Event 特征维度
        image_dim: Image 特征维度
        text_dim: Text 特征维度
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_path: 保存路径
        device: 设备

    Returns:
        alignment: 训练好的对齐模块
    """
    # 创建配置
    config = TripleModalAlignmentConfig(
        event_dim=event_dim,
        target_dim=image_dim,
        text_dim=text_dim,
        use_text_alignment=text_encoder is not None,
        freeze_event_encoder=True,
        freeze_target_encoder=True
    )

    # 创建模块
    alignment = TripleModalAlignmentModule(
        config,
        event_encoder=event_encoder,
        image_encoder=image_encoder,
        text_encoder=text_encoder
    )

    # 创建训练器
    trainer = TripleModalAlignmentTrainer(
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
