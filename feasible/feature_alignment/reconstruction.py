"""
Reconstruction Bridging Module
重建桥接模块 - 通过 E2VID 将 Event 重建为 RGB 图像，然后使用 CLIP 编码

优点:
    - 不需要配对的 Event-RGB 数据
    - 可以使用预训练的 E2VID
    - 直接利用现有 CLIP encoder

缺点:
    - 增加 5-10ms 推理延迟
    - 重建质量影响最终效果

参考:
    - E2VID: https://github.com/uzh-rpg/rpg_e2vid
    - HyperE2VID: https://github.com/ercanburak/HyperE2VID
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import time

from .base import (
    BaseAlignmentModule,
    AlignmentConfig,
    AlignmentMetrics,
    MLP
)


@dataclass
class ReconstructionBridgingConfig(AlignmentConfig):
    """重建桥接配置"""
    e2vid_type: str = 'e2vid'      # 'e2vid', 'hypere2vid', 'custom'
    reconstruction_size: Tuple[int, int] = (224, 224)  # 重建图像尺寸
    use_temporal_aggregation: bool = True  # 是否使用时序聚合
    num_temporal_bins: int = 5     # 时序 bins 数量
    finetune_e2vid: bool = False   # 是否微调 E2VID
    cache_reconstructions: bool = False  # 是否缓存重建结果
    normalize_output: bool = True  # 是否归一化输出


class E2VIDWrapper(nn.Module):
    """
    E2VID 模型包装器

    支持多种 E2VID 变体的统一接口
    """

    def __init__(
        self,
        e2vid_type: str = 'e2vid',
        pretrained_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.e2vid_type = e2vid_type
        self.device = device

        # 根据类型加载模型
        if e2vid_type == 'e2vid':
            self.model = self._load_e2vid(pretrained_path)
        elif e2vid_type == 'hypere2vid':
            self.model = self._load_hypere2vid(pretrained_path)
        elif e2vid_type == 'custom':
            self.model = None  # 由用户设置
        else:
            raise ValueError(f"Unknown E2VID type: {e2vid_type}")

        # 状态
        self.hidden_state = None

    def _load_e2vid(self, pretrained_path: Optional[str]) -> nn.Module:
        """加载标准 E2VID 模型"""
        # 这里需要实际的 E2VID 模型实现
        # 作为占位符，创建一个简单的 UNet 结构
        print("Loading E2VID model...")

        if pretrained_path:
            model = SimpleE2VIDNet()
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = SimpleE2VIDNet()
            print("Warning: Using untrained E2VID model")

        return model

    def _load_hypere2vid(self, pretrained_path: Optional[str]) -> nn.Module:
        """加载 HyperE2VID 模型"""
        print("Loading HyperE2VID model...")

        if pretrained_path:
            model = SimpleE2VIDNet()  # 替换为实际的 HyperE2VID
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = SimpleE2VIDNet()
            print("Warning: Using untrained HyperE2VID model")

        return model

    def forward(
        self,
        event_voxel: torch.Tensor,
        reset_state: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            event_voxel: Event voxel grid [batch_size, num_bins, H, W]
            reset_state: 是否重置隐藏状态

        Returns:
            reconstructed_image: 重建的灰度图像 [batch_size, 1, H, W]
        """
        if reset_state:
            self.hidden_state = None

        if self.model is not None:
            output = self.model(event_voxel, self.hidden_state)

            if isinstance(output, tuple):
                reconstructed, self.hidden_state = output
            else:
                reconstructed = output

            return reconstructed
        else:
            raise ValueError("E2VID model not initialized")

    def reset_states(self):
        """重置隐藏状态"""
        self.hidden_state = None


class SimpleE2VIDNet(nn.Module):
    """
    简化版 E2VID 网络

    实际使用时应替换为完整的 E2VID 实现
    """

    def __init__(
        self,
        num_input_bins: int = 5,
        base_channels: int = 32,
        num_residual_blocks: int = 2
    ):
        super().__init__()

        self.num_input_bins = num_input_bins

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_bins, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # ConvLSTM (简化为 ConvGRU)
        self.recurrent = ConvGRUCell(
            input_dim=base_channels * 4,
            hidden_dim=base_channels * 4,
            kernel_size=3
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        event_voxel: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            event_voxel: [batch_size, num_bins, H, W]
            hidden_state: 可选的隐藏状态

        Returns:
            reconstructed: [batch_size, 1, H, W]
            new_hidden_state: 新的隐藏状态
        """
        # Encode
        encoded = self.encoder(event_voxel)

        # Recurrent
        new_hidden = self.recurrent(encoded, hidden_state)

        # Decode
        reconstructed = self.decoder(new_hidden)

        return reconstructed, new_hidden


class ConvGRUCell(nn.Module):
    """卷积 GRU 单元"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )
        self.update_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )
        self.candidate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, _, h, w = x.shape

        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.hidden_dim, h, w,
                device=x.device, dtype=x.dtype
            )

        combined = torch.cat([x, hidden], dim=1)

        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))

        combined_reset = torch.cat([x, reset * hidden], dim=1)
        candidate = torch.tanh(self.candidate(combined_reset))

        new_hidden = (1 - update) * hidden + update * candidate

        return new_hidden


class ReconstructionBridgingModule(BaseAlignmentModule):
    """
    重建桥接对齐模块

    通过 E2VID 重建 RGB 图像，然后使用 CLIP 编码

    Example:
        >>> config = ReconstructionBridgingConfig()
        >>> bridging = ReconstructionBridgingModule(config, e2vid_model, clip_encoder)
        >>>
        >>> # 推理
        >>> aligned_features = bridging(event_voxel)
    """

    def __init__(
        self,
        config: ReconstructionBridgingConfig,
        e2vid_model: Optional[nn.Module] = None,
        clip_encoder: Optional[nn.Module] = None,
        e2vid_path: Optional[str] = None
    ):
        super().__init__(config)
        self.config = config

        # E2VID 模型
        if e2vid_model is not None:
            self.e2vid = e2vid_model
        elif e2vid_path is not None:
            self.e2vid = E2VIDWrapper(
                e2vid_type=config.e2vid_type,
                pretrained_path=e2vid_path
            )
        else:
            self.e2vid = E2VIDWrapper(e2vid_type=config.e2vid_type)

        # CLIP encoder
        self.clip_encoder = clip_encoder

        # 灰度转 RGB
        self.gray_to_rgb = nn.Conv2d(1, 3, 1, bias=False)
        nn.init.constant_(self.gray_to_rgb.weight, 1.0)

        # 可选的特征投影
        if config.event_dim != config.target_dim:
            self.feature_projector = MLP(
                input_dim=config.event_dim,
                output_dim=config.target_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers
            )
        else:
            self.feature_projector = None

        # 图像预处理
        self.register_buffer(
            'mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

        # 缓存
        self.reconstruction_cache = {} if config.cache_reconstructions else None

        # 冻结 E2VID
        if not config.finetune_e2vid:
            for param in self.e2vid.parameters():
                param.requires_grad = False

        # 冻结 CLIP
        if clip_encoder is not None and config.freeze_target_encoder:
            for param in clip_encoder.parameters():
                param.requires_grad = False

        # 延迟统计
        self.latency_stats = {
            'reconstruction_times': [],
            'encoding_times': []
        }

    def forward(self, event_voxel: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            event_voxel: Event voxel grid [batch_size, num_bins, H, W]

        Returns:
            features: 对齐后的特征 [batch_size, target_dim]
        """
        # Step 1: 重建图像
        start_time = time.time()
        reconstructed = self.reconstruct_image(event_voxel)
        recon_time = time.time() - start_time
        self.latency_stats['reconstruction_times'].append(recon_time * 1000)

        # Step 2: 预处理
        rgb_image = self.preprocess_image(reconstructed)

        # Step 3: CLIP 编码
        start_time = time.time()
        if self.clip_encoder is not None:
            with torch.no_grad() if self.config.freeze_target_encoder else torch.enable_grad():
                features = self.clip_encoder(rgb_image)
        else:
            features = rgb_image.flatten(1)
        encode_time = time.time() - start_time
        self.latency_stats['encoding_times'].append(encode_time * 1000)

        # Step 4: 可选的特征投影
        if self.feature_projector is not None:
            features = self.feature_projector(features)

        return features

    def reconstruct_image(
        self,
        event_voxel: torch.Tensor,
        reset_state: bool = False
    ) -> torch.Tensor:
        """
        重建图像

        Args:
            event_voxel: Event voxel grid [batch_size, num_bins, H, W]
            reset_state: 是否重置 E2VID 状态

        Returns:
            reconstructed: 重建的图像 [batch_size, 1, H, W]
        """
        with torch.no_grad() if not self.config.finetune_e2vid else torch.enable_grad():
            reconstructed = self.e2vid(event_voxel, reset_state=reset_state)

        # 调整尺寸
        target_size = self.config.reconstruction_size
        if reconstructed.shape[-2:] != target_size:
            reconstructed = F.interpolate(
                reconstructed,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

        return reconstructed

    def preprocess_image(self, gray_image: torch.Tensor) -> torch.Tensor:
        """
        预处理图像用于 CLIP

        Args:
            gray_image: 灰度图像 [batch_size, 1, H, W]

        Returns:
            rgb_image: 预处理后的 RGB 图像 [batch_size, 3, H, W]
        """
        # 灰度转 RGB
        rgb = self.gray_to_rgb(gray_image)

        # CLIP 标准化
        if self.config.normalize_output:
            rgb = (rgb - self.mean) / self.std

        return rgb

    def align_features(
        self,
        event_features: torch.Tensor,
        target_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        对齐特征

        注意: 这个方法期望输入是 event voxel，而不是预提取的特征
        """
        aligned_event = self.forward(event_features)
        return aligned_event, target_features

    def get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        stats = {}

        if self.latency_stats['reconstruction_times']:
            stats['avg_reconstruction_ms'] = np.mean(
                self.latency_stats['reconstruction_times']
            )
            stats['max_reconstruction_ms'] = np.max(
                self.latency_stats['reconstruction_times']
            )

        if self.latency_stats['encoding_times']:
            stats['avg_encoding_ms'] = np.mean(
                self.latency_stats['encoding_times']
            )
            stats['max_encoding_ms'] = np.max(
                self.latency_stats['encoding_times']
            )

        stats['total_avg_ms'] = stats.get('avg_reconstruction_ms', 0) + \
                                stats.get('avg_encoding_ms', 0)

        return stats

    def reset_latency_stats(self):
        """重置延迟统计"""
        self.latency_stats = {
            'reconstruction_times': [],
            'encoding_times': []
        }

    def reset_e2vid_state(self):
        """重置 E2VID 隐藏状态"""
        if hasattr(self.e2vid, 'reset_states'):
            self.e2vid.reset_states()


class ReconstructionBridgingWithRefinement(ReconstructionBridgingModule):
    """
    带细化网络的重建桥接模块

    在 E2VID 重建后添加一个细化网络来提高图像质量
    """

    def __init__(
        self,
        config: ReconstructionBridgingConfig,
        e2vid_model: Optional[nn.Module] = None,
        clip_encoder: Optional[nn.Module] = None,
        e2vid_path: Optional[str] = None
    ):
        super().__init__(config, e2vid_model, clip_encoder, e2vid_path)

        # 细化网络
        self.refinement_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def reconstruct_image(
        self,
        event_voxel: torch.Tensor,
        reset_state: bool = False
    ) -> torch.Tensor:
        """带细化的重建"""
        # 基础重建
        reconstructed = super().reconstruct_image(event_voxel, reset_state)

        # 细化
        refined = self.refinement_net(reconstructed)

        # 残差连接
        output = reconstructed + 0.1 * (refined - reconstructed)

        return output


def create_reconstruction_bridging(
    e2vid_path: Optional[str] = None,
    clip_encoder: Optional[nn.Module] = None,
    e2vid_type: str = 'e2vid',
    target_dim: int = 1024,
    finetune_e2vid: bool = False,
    device: str = 'cuda'
) -> ReconstructionBridgingModule:
    """
    便捷函数：创建重建桥接模块

    Args:
        e2vid_path: E2VID 预训练权重路径
        clip_encoder: CLIP 编码器
        e2vid_type: E2VID 类型
        target_dim: 目标特征维度
        finetune_e2vid: 是否微调 E2VID
        device: 设备

    Returns:
        bridging: 重建桥接模块
    """
    config = ReconstructionBridgingConfig(
        e2vid_type=e2vid_type,
        target_dim=target_dim,
        finetune_e2vid=finetune_e2vid
    )

    bridging = ReconstructionBridgingModule(
        config,
        clip_encoder=clip_encoder,
        e2vid_path=e2vid_path
    ).to(device)

    return bridging


def finetune_e2vid_on_events(
    e2vid_model: nn.Module,
    event_dataset,
    num_steps: int = 1000,
    learning_rate: float = 1e-5,
    device: str = 'cuda'
) -> nn.Module:
    """
    在目标域 Event 数据上微调 E2VID

    Args:
        e2vid_model: E2VID 模型
        event_dataset: Event 数据集
        num_steps: 训练步数
        learning_rate: 学习率
        device: 设备

    Returns:
        e2vid_model: 微调后的模型
    """
    from torch.utils.data import DataLoader

    e2vid_model = e2vid_model.to(device)
    e2vid_model.train()

    optimizer = torch.optim.Adam(e2vid_model.parameters(), lr=learning_rate)

    loader = DataLoader(event_dataset, batch_size=4, shuffle=True)

    step = 0
    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break

            event_voxel = batch['event_voxel'].to(device)

            # 自监督损失：时序一致性
            if event_voxel.shape[0] >= 2:
                recon1, _ = e2vid_model(event_voxel[:-1])
                recon2, _ = e2vid_model(event_voxel[1:])

                # 时序平滑损失
                temporal_loss = F.mse_loss(recon1, recon2)

                # 总变分损失 (鼓励平滑)
                tv_loss = (
                    (recon1[:, :, :, :-1] - recon1[:, :, :, 1:]).abs().mean() +
                    (recon1[:, :, :-1, :] - recon1[:, :, 1:, :]).abs().mean()
                )

                loss = temporal_loss + 0.1 * tv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

            step += 1

    e2vid_model.eval()
    return e2vid_model
