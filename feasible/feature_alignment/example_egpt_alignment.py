"""
Example: Feature Alignment with EGPT Dataset
示例：使用 EGPT 数据集进行特征对齐训练

使用 EGPT 训练集作为配对数据进行对齐训练，
使用测试集进行评估。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_alignment import (
    # 策略选择
    auto_select_alignment_strategy,
    print_strategy_recommendation,
    quick_feasibility_test,

    # 对齐模块
    ContrastiveAlignmentConfig,
    ContrastiveAlignmentModule,
    ContrastiveAlignmentTrainer,
    LightweightAlignmentModule,
    LightweightAlignmentConfig,

    # 工具
    AlignmentDataset,
    diagnose_alignment_quality,
    evaluate_alignment_comprehensive,
    extract_and_save_features
)


# ============================================================================
# Step 1: 数据集适配器 - 连接 EGPT 数据集
# ============================================================================

class EGPTAlignmentDataset(Dataset):
    """
    EGPT 数据集适配器

    从 EGPT 训练/测试数据中提取 Event 和 RGB 特征对
    """

    def __init__(
        self,
        egpt_dataset,
        event_encoder: nn.Module,
        rgb_encoder: nn.Module,
        device: str = 'cuda',
        cache_features: bool = True
    ):
        """
        Args:
            egpt_dataset: EGPT 原始数据集
            event_encoder: Event 编码器 (EventGPT 的 visual encoder)
            rgb_encoder: RGB 编码器 (CLIP visual encoder)
            device: 设备
            cache_features: 是否缓存特征
        """
        self.egpt_dataset = egpt_dataset
        self.event_encoder = event_encoder
        self.rgb_encoder = rgb_encoder
        self.device = device

        self.event_encoder.eval()
        self.rgb_encoder.eval()

        # 缓存
        self.cached_event_features = None
        self.cached_rgb_features = None

        if cache_features:
            self._cache_all_features()

    def _cache_all_features(self):
        """预先提取并缓存所有特征"""
        print("Caching features...")

        event_features = []
        rgb_features = []

        with torch.no_grad():
            for i in range(len(self.egpt_dataset)):
                sample = self.egpt_dataset[i]

                # 获取 event 数据
                event_data = sample.get('event_image', sample.get('image'))
                if event_data is not None:
                    if not isinstance(event_data, torch.Tensor):
                        event_data = torch.tensor(event_data)
                    event_data = event_data.unsqueeze(0).to(self.device)

                    event_feat = self.event_encoder(event_data)
                    if isinstance(event_feat, tuple):
                        event_feat = event_feat[0]
                    if event_feat.dim() == 3:
                        event_feat = event_feat[:, 0]  # CLS token

                    event_features.append(event_feat.cpu())

                # 获取 RGB 数据 (如果有)
                rgb_data = sample.get('rgb_image', sample.get('image'))
                if rgb_data is not None:
                    if not isinstance(rgb_data, torch.Tensor):
                        rgb_data = torch.tensor(rgb_data)
                    rgb_data = rgb_data.unsqueeze(0).to(self.device)

                    rgb_feat = self.rgb_encoder(rgb_data)
                    if isinstance(rgb_feat, tuple):
                        rgb_feat = rgb_feat[0]
                    if rgb_feat.dim() == 3:
                        rgb_feat = rgb_feat[:, 0]

                    rgb_features.append(rgb_feat.cpu())

                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(self.egpt_dataset)}")

        self.cached_event_features = torch.cat(event_features, dim=0)
        self.cached_rgb_features = torch.cat(rgb_features, dim=0)

        print(f"Cached {len(self.cached_event_features)} feature pairs")
        print(f"  Event feature dim: {self.cached_event_features.shape[-1]}")
        print(f"  RGB feature dim: {self.cached_rgb_features.shape[-1]}")

    def __len__(self):
        if self.cached_event_features is not None:
            return len(self.cached_event_features)
        return len(self.egpt_dataset)

    def __getitem__(self, idx):
        if self.cached_event_features is not None:
            return {
                'event_features': self.cached_event_features[idx],
                'target_features': self.cached_rgb_features[idx]
            }

        # 实时提取（较慢）
        sample = self.egpt_dataset[idx]
        event_data = sample.get('event_image', sample.get('image'))
        rgb_data = sample.get('rgb_image', sample.get('image'))

        with torch.no_grad():
            event_data = torch.tensor(event_data).unsqueeze(0).to(self.device)
            rgb_data = torch.tensor(rgb_data).unsqueeze(0).to(self.device)

            event_feat = self.event_encoder(event_data).squeeze(0)
            rgb_feat = self.rgb_encoder(rgb_data).squeeze(0)

        return {
            'event_features': event_feat.cpu(),
            'target_features': rgb_feat.cpu()
        }

    def save_features(self, save_dir: str):
        """保存缓存的特征到文件"""
        if self.cached_event_features is None:
            print("No cached features to save. Call _cache_all_features first.")
            return

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.cached_event_features, save_path / 'event_features.pt')
        torch.save(self.cached_rgb_features, save_path / 'target_features.pt')

        print(f"Features saved to {save_dir}")


# ============================================================================
# Step 2: 完整训练流程
# ============================================================================

def run_alignment_training(
    egpt_train_dataset,
    egpt_test_dataset,
    event_encoder: nn.Module,
    rgb_encoder: nn.Module,
    strategy: str = 'contrastive',  # 'contrastive', 'lightweight', 'auto'
    event_dim: int = 768,
    target_dim: int = 1024,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    save_dir: str = './alignment_checkpoints',
    device: str = 'cuda'
):
    """
    运行完整的对齐训练流程

    Args:
        egpt_train_dataset: EGPT 训练数据集
        egpt_test_dataset: EGPT 测试数据集
        event_encoder: Event 编码器
        rgb_encoder: RGB 编码器
        strategy: 对齐策略
        event_dim: Event 特征维度
        target_dim: Target 特征维度
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_dir: 保存目录
        device: 设备

    Returns:
        alignment_module: 训练好的对齐模块
    """
    print("="*70)
    print("EGPT Feature Alignment Training")
    print("="*70)

    # Step 1: 准备数据
    print("\n[1/4] Preparing datasets...")

    train_dataset = EGPTAlignmentDataset(
        egpt_train_dataset,
        event_encoder,
        rgb_encoder,
        device=device,
        cache_features=True
    )

    test_dataset = EGPTAlignmentDataset(
        egpt_test_dataset,
        event_encoder,
        rgb_encoder,
        device=device,
        cache_features=True
    )

    # 保存特征
    feature_dir = Path(save_dir) / 'features'
    train_dataset.save_features(str(feature_dir / 'train'))
    test_dataset.save_features(str(feature_dir / 'test'))

    # Step 2: 自动策略选择 (如果需要)
    if strategy == 'auto':
        print("\n[2/4] Auto-selecting alignment strategy...")
        recommendation = auto_select_alignment_strategy(
            has_paired_data=True,
            data_size=len(train_dataset),
            has_e2vid=False,
            target_performance='good'
        )
        print_strategy_recommendation(recommendation)
        strategy = 'contrastive' if 'Contrastive' in recommendation.strategy else 'lightweight'
    else:
        print(f"\n[2/4] Using specified strategy: {strategy}")

    # Step 3: 训练对齐模块
    print(f"\n[3/4] Training {strategy} alignment...")

    if strategy == 'contrastive':
        config = ContrastiveAlignmentConfig(
            event_dim=event_dim,
            target_dim=target_dim,
            hidden_dim=512,
            num_layers=2,
            temperature=0.07
        )

        alignment = ContrastiveAlignmentModule(config)

        trainer = ContrastiveAlignmentTrainer(
            alignment,
            train_dataset,
            test_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )

        trainer.train(
            num_epochs=num_epochs,
            save_path=str(Path(save_dir) / 'contrastive_alignment.pt')
        )

    else:  # lightweight
        config = LightweightAlignmentConfig(
            event_dim=event_dim,
            target_dim=target_dim,
            hidden_dim=512,
            num_layers=2,
            adapter_type='mlp'
        )

        alignment = LightweightAlignmentModule(config)

        from feature_alignment import LightweightAlignmentTrainer
        trainer = LightweightAlignmentTrainer(
            alignment,
            train_dataset,
            test_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )

        trainer.train(
            num_epochs=num_epochs,
            save_path=str(Path(save_dir) / 'lightweight_alignment.pt')
        )

    # Step 4: 评估
    print("\n[4/4] Evaluating alignment...")

    metrics = evaluate_alignment_comprehensive(
        alignment,
        test_dataset,
        device=device
    )

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nFinal Metrics:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
    print(f"  R@1: {metrics['R@1']:.2%}")
    print(f"  R@5: {metrics['R@5']:.2%}")
    print(f"  R@10: {metrics['R@10']:.2%}")

    return alignment


# ============================================================================
# Step 3: 使用预提取特征的简化版本
# ============================================================================

def run_alignment_from_features(
    train_features_dir: str,
    test_features_dir: str,
    strategy: str = 'contrastive',
    event_dim: int = 768,
    target_dim: int = 1024,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    save_path: str = './alignment.pt',
    device: str = 'cuda'
):
    """
    从预提取的特征文件运行对齐训练

    使用方法:
        1. 先用 extract_and_save_features() 提取特征
        2. 然后用这个函数训练对齐

    Args:
        train_features_dir: 训练特征目录
        test_features_dir: 测试特征目录
        strategy: 对齐策略
        event_dim: Event 特征维度
        target_dim: Target 特征维度
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_path: 模型保存路径
        device: 设备
    """
    # 加载数据集
    train_dataset = AlignmentDataset.from_directory(train_features_dir)
    test_dataset = AlignmentDataset.from_directory(test_features_dir)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 训练
    if strategy == 'contrastive':
        from feature_alignment import train_contrastive_alignment

        # 不需要编码器（已经有特征了）
        config = ContrastiveAlignmentConfig(
            event_dim=event_dim,
            target_dim=target_dim
        )
        alignment = ContrastiveAlignmentModule(config)

        trainer = ContrastiveAlignmentTrainer(
            alignment,
            train_dataset,
            test_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )

        trainer.train(num_epochs=num_epochs, save_path=save_path)

    else:
        from feature_alignment import train_lightweight_alignment

        alignment = train_lightweight_alignment(
            train_dataset,
            test_dataset,
            event_dim=event_dim,
            target_dim=target_dim,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_path=save_path,
            device=device
        )

    return alignment


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EGPT Feature Alignment Example")
    print("="*70)

    print("""
Usage Examples:

1. From EGPT datasets (requires loading models):
   -----------------------------------------------
   from model import EventChatModel
   from llava.model import LlavaLlamaForCausalLM

   # Load models
   event_encoder = EventChatModel.from_pretrained(...).visual_encoder
   rgb_encoder = LlavaLlamaForCausalLM.from_pretrained(...).get_vision_tower()

   # Load EGPT datasets
   from dataset.conversation import ConversationDataset
   train_dataset = ConversationDataset(split='train')
   test_dataset = ConversationDataset(split='test')

   # Run training
   alignment = run_alignment_training(
       train_dataset, test_dataset,
       event_encoder, rgb_encoder,
       strategy='contrastive',
       num_epochs=10
   )

2. From pre-extracted features:
   -----------------------------
   alignment = run_alignment_from_features(
       train_features_dir='./features/train',
       test_features_dir='./features/test',
       strategy='contrastive',
       num_epochs=10
   )

3. Quick feasibility test:
   ------------------------
   from feature_alignment import quick_feasibility_test

   result = quick_feasibility_test(
       event_encoder, rgb_encoder,
       test_loader, device='cuda'
   )
   # Output: Initial similarity and recommendation

4. Auto strategy selection:
   -------------------------
   from feature_alignment import auto_select_alignment_strategy

   recommendation = auto_select_alignment_strategy(
       has_paired_data=True,
       data_size=len(train_dataset),
       has_e2vid=False,
       target_performance='good'
   )
    """)
