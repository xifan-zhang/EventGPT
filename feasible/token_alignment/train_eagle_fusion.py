"""
Train EAGLE Fusion for Token-Level Alignment
=============================================

Trains the EAGLE-style fusion module to improve token-level acceptance rate
from ~3% to target 50%+.

Usage:
    # Step 1: Extract features (if not done)
    python feasible/token_alignment/extract_features.py \
        --dataset_dir ./data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
        --output_dir ./feasible/token_alignment/cached_outputs_1s \
        --max_samples 1000 \
        --extract_hidden_states

    # Step 2: Train EAGLE fusion
    python feasible/token_alignment/train_eagle_fusion.py \
        --cached_dir ./feasible/token_alignment/cached_outputs_1s \
        --output_dir ./feasible/token_alignment/checkpoints_1s \
        --num_epochs 50 \
        --batch_size 16 \
        --learning_rate 1e-4
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from datetime import datetime

# Fix protobuf issue
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from feasible.token_alignment.base import TokenAlignmentDataset
from feasible.token_alignment.eagle_fusion import (
    EAGLEFusionConfig,
    EAGLEFusionModule,
    EAGLEFusionTrainer,
)


def load_dataset(cached_dir: str, split_ratio: float = 0.9):
    """Load and split dataset."""
    print(f"Loading dataset from {cached_dir}...")

    dataset = TokenAlignmentDataset.from_directory(cached_dir)

    print(f"Loaded {len(dataset)} samples")
    print(f"  Draft tokens shape: {dataset.draft_tokens.shape}")
    print(f"  Target tokens shape: {dataset.target_tokens.shape}")

    if dataset.draft_hidden is not None:
        print(f"  Draft hidden shape: {dataset.draft_hidden.shape}")
    if dataset.target_hidden is not None:
        print(f"  Target hidden shape: {dataset.target_hidden.shape}")

    # Split into train/val
    num_samples = len(dataset)
    num_train = int(num_samples * split_ratio)

    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Create split datasets
    train_dataset = TokenAlignmentDataset(
        draft_tokens=dataset.draft_tokens[train_indices],
        target_tokens=dataset.target_tokens[train_indices],
        draft_hidden=dataset.draft_hidden[train_indices] if dataset.draft_hidden is not None else None,
        target_hidden=dataset.target_hidden[train_indices] if dataset.target_hidden is not None else None,
        draft_logits=dataset.draft_logits[train_indices] if dataset.draft_logits is not None else None,
        target_logits=dataset.target_logits[train_indices] if dataset.target_logits is not None else None,
        sample_ids=[dataset.sample_ids[i] for i in train_indices.tolist()],
    )

    val_dataset = TokenAlignmentDataset(
        draft_tokens=dataset.draft_tokens[val_indices],
        target_tokens=dataset.target_tokens[val_indices],
        draft_hidden=dataset.draft_hidden[val_indices] if dataset.draft_hidden is not None else None,
        target_hidden=dataset.target_hidden[val_indices] if dataset.target_hidden is not None else None,
        draft_logits=dataset.draft_logits[val_indices] if dataset.draft_logits is not None else None,
        target_logits=dataset.target_logits[val_indices] if dataset.target_logits is not None else None,
        sample_ids=[dataset.sample_ids[i] for i in val_indices.tolist()],
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    return train_dataset, val_dataset


def compute_baseline_acceptance(dataset: TokenAlignmentDataset) -> float:
    """Compute baseline token-level acceptance rate."""
    total_correct = 0
    total_tokens = 0

    for i in range(len(dataset)):
        draft = dataset.draft_tokens[i]
        target = dataset.target_tokens[i]

        mask = (target != 0) & (draft != 0)
        min_len = min(len(draft), len(target))

        correct = ((draft[:min_len] == target[:min_len]) & mask[:min_len]).sum().item()
        total = mask[:min_len].sum().item()

        total_correct += correct
        total_tokens += total

    return total_correct / total_tokens if total_tokens > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Train EAGLE Fusion for token alignment")
    parser.add_argument('--cached_dir', type=str, required=True,
                        help="Path to cached features directory")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Path to save checkpoints")
    parser.add_argument('--num_epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help="Hidden dimension for fusion layers")
    parser.add_argument('--num_fusion_layers', type=int, default=2,
                        help="Number of fusion layers")
    parser.add_argument('--num_attention_heads', type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use")
    parser.add_argument('--use_kl_loss', action='store_true',
                        help="Use KL divergence loss (requires target logits)")
    parser.add_argument('--kl_weight', type=float, default=0.5,
                        help="Weight for KL loss")
    parser.add_argument('--early_stopping', type=int, default=10,
                        help="Early stopping patience")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_dataset, val_dataset = load_dataset(args.cached_dir)

    # Compute baseline
    print("\nComputing baseline acceptance rate...")
    baseline_acc = compute_baseline_acceptance(val_dataset)
    print(f"Baseline token-level acceptance: {baseline_acc*100:.2f}%")

    # Get dimensions from data
    if train_dataset.draft_hidden is not None:
        draft_hidden_size = train_dataset.draft_hidden.size(-1)
    else:
        draft_hidden_size = 4096  # Default for LLaMA-7B

    if train_dataset.target_hidden is not None:
        target_hidden_size = train_dataset.target_hidden.size(-1)
    else:
        target_hidden_size = 4096

    # Create config
    config = EAGLEFusionConfig(
        draft_vocab_size=32000,
        target_vocab_size=32000,
        draft_hidden_size=draft_hidden_size,
        target_hidden_size=target_hidden_size,
        hidden_dim=args.hidden_dim,
        num_fusion_layers=args.num_fusion_layers,
        num_attention_heads=args.num_attention_heads,
        use_kl_loss=args.use_kl_loss,
        kl_weight=args.kl_weight,
        dropout=0.1,
    )

    print(f"\nModel config:")
    print(f"  Draft hidden: {draft_hidden_size}")
    print(f"  Target hidden: {target_hidden_size}")
    print(f"  Fusion hidden: {args.hidden_dim}")
    print(f"  Fusion layers: {args.num_fusion_layers}")

    # Create model
    model = EAGLEFusionModule(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: ~{trainable_params * 4 / 1024 / 1024:.1f} MB")

    # Create trainer
    trainer = EAGLEFusionTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    # Save config
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'model_config': config.to_dict(),
            'training_args': vars(args),
            'baseline_acceptance': baseline_acc,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    # Train
    print(f"\nTraining for {args.num_epochs} epochs...")
    print(f"Target: Improve from {baseline_acc*100:.1f}% to 50%+ acceptance rate")
    print("-" * 60)

    checkpoint_path = output_path / 'best_model.pt'
    trainer.train(
        num_epochs=args.num_epochs,
        save_path=str(checkpoint_path),
        early_stopping_patience=args.early_stopping,
    )

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = trainer.validate()
    print(f"Final acceptance rate: {final_metrics['acceptance_rate']*100:.2f}%")
    print(f"Top-5 rate: {final_metrics['top5_rate']*100:.2f}%")
    print(f"Improvement: {baseline_acc*100:.2f}% -> {final_metrics['acceptance_rate']*100:.2f}%")

    # Save final metrics
    metrics_path = output_path / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'baseline_acceptance': baseline_acc,
            'final_acceptance': final_metrics['acceptance_rate'],
            'top5_rate': final_metrics['top5_rate'],
            'improvement': final_metrics['acceptance_rate'] - baseline_acc,
            'training_stats': trainer.training_stats,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
