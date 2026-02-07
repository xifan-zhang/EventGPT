#!/usr/bin/env python3
"""
Run Token Alignment Training
============================

Two-stage process:
1. Extract: Run EventGPT & Video-LLaVA to collect tokens (one-time, ~4GB VRAM each)
2. Train: Train alignment module on cached data (~1-2GB VRAM)

Usage:
    # Stage 1: Extract outputs (requires both models)
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python run_token_alignment.py \
        --mode extract \
        --dataset_dir ./data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
        --output_dir ./feasible/token_alignment/cached_outputs_1s

    # Stage 2: Train alignment (only needs cached data)
    python run_token_alignment.py \
        --mode train \
        --cached_dir ./feasible/token_alignment/cached_outputs_1s \
        --strategy sequence_distillation
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules - deferred to avoid import issues in extract mode
token_alignment_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, token_alignment_dir)

from dataclasses import dataclass
import torch.nn as nn

# Lazy loading for training modules
_modules_loaded = False
TokenAlignmentDataset = None
TokenProjectionConfig = None
TokenProjectionModule = None
TokenProjectionTrainer = None
SequenceDistillationConfig = None
SequenceDistillationModule = None
SequenceDistillationTrainer = None

def _load_training_modules():
    """Load training modules only when needed."""
    global _modules_loaded, TokenAlignmentDataset
    global TokenProjectionConfig, TokenProjectionModule, TokenProjectionTrainer
    global SequenceDistillationConfig, SequenceDistillationModule, SequenceDistillationTrainer

    if _modules_loaded:
        return

    import importlib.util
    import types

    # Create a fake package for relative imports
    package_name = 'token_alignment'
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [token_alignment_dir]
        pkg.__package__ = package_name
        sys.modules[package_name] = pkg

    def load_module(name, path, submodule_name=None):
        full_name = f"{package_name}.{submodule_name}" if submodule_name else name
        spec = importlib.util.spec_from_file_location(
            full_name, path,
            submodule_search_locations=[token_alignment_dir]
        )
        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
        sys.modules[full_name] = module
        if submodule_name:
            sys.modules[name] = module  # Also register short name
        spec.loader.exec_module(module)
        return module

    # Load base module first (no relative imports)
    base_module = load_module('base', os.path.join(token_alignment_dir, 'base.py'), 'base')
    TokenAlignmentDataset = base_module.TokenAlignmentDataset

    # Load token projection module
    tp_module = load_module('token_projection', os.path.join(token_alignment_dir, 'token_projection.py'), 'token_projection')
    TokenProjectionConfig = tp_module.TokenProjectionConfig
    TokenProjectionModule = tp_module.TokenProjectionModule
    TokenProjectionTrainer = tp_module.TokenProjectionTrainer

    # Load sequence distillation module
    sd_module = load_module('sequence_distillation', os.path.join(token_alignment_dir, 'sequence_distillation.py'), 'sequence_distillation')
    SequenceDistillationConfig = sd_module.SequenceDistillationConfig
    SequenceDistillationModule = sd_module.SequenceDistillationModule
    SequenceDistillationTrainer = sd_module.SequenceDistillationTrainer

    _modules_loaded = True


def extract_from_benchmark_results(json_path: str, output_dir: str, max_seq_len: int = 128):
    """
    Extract token alignment data from existing benchmark results.

    Uses pre-computed token IDs from benchmark_inference.py output.
    """
    print(f"Loading benchmark results from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    draft_tokens_list = []
    target_tokens_list = []
    sample_ids = []

    for item in data:
        draft_ids = item.get('egpt_token_ids', [])
        target_ids = item.get('llava_token_ids', [])

        if not draft_ids or not target_ids:
            continue

        # Pad/truncate to max_seq_len
        draft_ids = draft_ids[:max_seq_len]
        target_ids = target_ids[:max_seq_len]

        # Pad with 0 (assuming pad token is 0)
        draft_ids = draft_ids + [0] * (max_seq_len - len(draft_ids))
        target_ids = target_ids + [0] * (max_seq_len - len(target_ids))

        draft_tokens_list.append(draft_ids)
        target_tokens_list.append(target_ids)
        sample_ids.append(item.get('id', f'sample_{len(sample_ids)}'))

    if not draft_tokens_list:
        print("ERROR: No valid samples found in benchmark results.")
        print("Make sure the benchmark was run with token IDs saved.")
        return None

    draft_tokens = torch.tensor(draft_tokens_list, dtype=torch.long)
    target_tokens = torch.tensor(target_tokens_list, dtype=torch.long)

    print(f"Extracted {len(draft_tokens)} samples")
    print(f"Draft tokens shape: {draft_tokens.shape}")
    print(f"Target tokens shape: {target_tokens.shape}")

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(draft_tokens, output_dir / 'draft_tokens.pt')
    torch.save(target_tokens, output_dir / 'target_tokens.pt')
    with open(output_dir / 'sample_ids.json', 'w') as f:
        json.dump(sample_ids, f)

    # Save metadata
    metadata = {
        'source': json_path,
        'num_samples': len(draft_tokens),
        'max_seq_len': max_seq_len,
        'extracted_at': datetime.now().isoformat(),
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved to {output_dir}")
    return output_dir


def train_alignment(
    cached_dir: str,
    strategy: str = 'token_mapping',
    output_dir: str = None,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    train_ratio: float = 0.8,
):
    """
    Train token alignment module on cached data.

    Strategies:
    - token_mapping: Simple learned token mapping (fastest, ~50% accuracy expected)
    - sequence_distillation: Transformer-based distillation (slower, higher accuracy)
    - logit_alignment: KL divergence on logits (requires cached logits)
    """
    # Load training modules
    _load_training_modules()
    global TokenAlignmentDataset, TokenProjectionConfig, TokenProjectionModule, TokenProjectionTrainer
    global SequenceDistillationConfig, SequenceDistillationModule, SequenceDistillationTrainer

    print(f"\n{'='*60}")
    print(f"Training Token Alignment: {strategy}")
    print(f"{'='*60}")

    cached_dir = Path(cached_dir)

    # Load cached data
    print(f"Loading cached data from {cached_dir}...")
    draft_tokens = torch.load(cached_dir / 'draft_tokens.pt')
    target_tokens = torch.load(cached_dir / 'target_tokens.pt')

    with open(cached_dir / 'sample_ids.json', 'r') as f:
        sample_ids = json.load(f)

    print(f"Loaded {len(draft_tokens)} samples")

    # Create dataset
    # For simple strategies without hidden states, we create dummy hidden states
    # based on one-hot token embeddings
    vocab_size = 32000  # LLaMA vocab size
    hidden_size = 4096

    # Create pseudo hidden states from tokens (one-hot projected)
    # This is a placeholder - real hidden states would come from model extraction
    print("Creating pseudo hidden states from tokens...")
    draft_hidden = F.one_hot(draft_tokens, num_classes=vocab_size).float()
    draft_hidden = draft_hidden @ torch.randn(vocab_size, hidden_size) * 0.1  # Random projection

    target_hidden = F.one_hot(target_tokens, num_classes=vocab_size).float()
    target_hidden = target_hidden @ torch.randn(vocab_size, hidden_size) * 0.1

    # Split train/val
    total = len(draft_tokens)
    train_size = int(train_ratio * total)

    torch.manual_seed(42)
    indices = torch.randperm(total)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = TokenAlignmentDataset(
        draft_tokens=draft_tokens[train_indices],
        target_tokens=target_tokens[train_indices],
        draft_hidden=draft_hidden[train_indices],
        target_hidden=target_hidden[train_indices],
        sample_ids=[sample_ids[i] for i in train_indices.tolist()],
    )

    val_dataset = TokenAlignmentDataset(
        draft_tokens=draft_tokens[val_indices],
        target_tokens=target_tokens[val_indices],
        draft_hidden=draft_hidden[val_indices],
        target_hidden=target_hidden[val_indices],
        sample_ids=[sample_ids[i] for i in val_indices.tolist()],
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model based on strategy
    if strategy == 'token_mapping' or strategy == 'token_projection':
        config = TokenProjectionConfig(
            draft_vocab_size=vocab_size,
            target_vocab_size=vocab_size,
            draft_hidden_size=hidden_size,
            target_hidden_size=hidden_size,
            hidden_dim=2048,
            num_layers=2,
            projection_type='mlp',
        )
        model = TokenProjectionModule(config)
        trainer = TokenProjectionTrainer(
            model, train_dataset, val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

    elif strategy == 'sequence_distillation':
        config = SequenceDistillationConfig(
            draft_vocab_size=vocab_size,
            target_vocab_size=vocab_size,
            draft_hidden_size=hidden_size,
            target_hidden_size=hidden_size,
            hidden_dim=1024,
            num_transformer_layers=2,
            num_attention_heads=8,
        )
        model = SequenceDistillationModule(config)
        trainer = SequenceDistillationTrainer(
            model, train_dataset, val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Output directory
    if output_dir is None:
        output_dir = cached_dir / f'checkpoints_{strategy}'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    save_path = str(output_dir / f'{strategy}_alignment.pt')
    trainer.train(
        num_epochs=num_epochs,
        save_path=save_path,
        early_stopping_patience=5,
    )

    # Save training info
    info = {
        'strategy': strategy,
        'num_epochs': num_epochs,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'final_stats': trainer.training_stats,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nTraining complete. Saved to {output_dir}")

    return model, trainer


def evaluate_alignment(
    checkpoint_path: str,
    cached_dir: str,
    strategy: str = 'token_mapping',
    device: str = 'cuda',
):
    """Evaluate trained alignment model."""
    # Load training modules
    _load_training_modules()
    global TokenProjectionModule, SequenceDistillationModule

    print(f"\n{'='*60}")
    print(f"Evaluating Token Alignment: {strategy}")
    print(f"{'='*60}")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    if strategy == 'token_mapping' or strategy == 'token_projection':
        model = TokenProjectionModule(config)
    elif strategy == 'sequence_distillation':
        model = SequenceDistillationModule(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test data
    cached_dir = Path(cached_dir)
    draft_tokens = torch.load(cached_dir / 'draft_tokens.pt')
    target_tokens = torch.load(cached_dir / 'target_tokens.pt')

    # Use last 20% as test
    total = len(draft_tokens)
    test_start = int(0.8 * total)
    test_draft = draft_tokens[test_start:].to(device)
    test_target = target_tokens[test_start:].to(device)

    print(f"Test samples: {len(test_draft)}")

    # Compute metrics
    with torch.no_grad():
        if strategy == 'token_mapping' or strategy == 'token_projection':
            # Project tokens
            target_probs = model.project_tokens(test_draft)
            predictions = target_probs.argmax(dim=-1)
        else:
            # For sequence distillation, we need hidden states
            vocab_size = 32000
            hidden_size = 4096
            draft_hidden = F.one_hot(test_draft, num_classes=vocab_size).float()
            draft_hidden = draft_hidden @ torch.randn(vocab_size, hidden_size, device=device) * 0.1

            logits = model(draft_hidden)
            predictions = logits.argmax(dim=-1)

        # Acceptance rate
        mask = (test_target != 0)
        correct = ((predictions == test_target) & mask).sum().item()
        total_tokens = mask.sum().item()
        acceptance_rate = correct / total_tokens if total_tokens > 0 else 0

    print(f"\nTest Results:")
    print(f"  Acceptance Rate: {acceptance_rate*100:.1f}%")
    print(f"  Matched Tokens: {correct}/{total_tokens}")

    return acceptance_rate


def main():
    parser = argparse.ArgumentParser(description='Token Alignment Training')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['extract', 'train', 'evaluate'],
                        help='Mode: extract outputs, train alignment, or evaluate')

    # Extract mode options
    parser.add_argument('--benchmark_json', type=str,
                        default='./feasible/benchmark_inference/benchmark_results_S1.json',
                        help='Path to benchmark results JSON (for extract mode)')
    parser.add_argument('--dataset_dir', type=str,
                        default='./data/my_egpt_dsec_train/my_egpt_dsec_train_1s',
                        help='Dataset directory (for extract mode)')

    # Train/Evaluate mode options
    parser.add_argument('--cached_dir', type=str,
                        default='./feasible/token_alignment/cached_outputs_1s',
                        help='Directory with cached outputs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints')
    parser.add_argument('--strategy', type=str, default='token_projection',
                        choices=['token_projection', 'sequence_distillation', 'logit_alignment'],
                        help='Alignment strategy')

    # Training options
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_seq_len', type=int, default=128)

    # Evaluate options
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for evaluation')

    args = parser.parse_args()

    if args.mode == 'extract':
        extract_from_benchmark_results(
            args.benchmark_json,
            args.cached_dir,
            args.max_seq_len,
        )

    elif args.mode == 'train':
        train_alignment(
            cached_dir=args.cached_dir,
            strategy=args.strategy,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )

    elif args.mode == 'evaluate':
        if not args.checkpoint:
            # Default checkpoint path
            args.checkpoint = f'{args.cached_dir}/checkpoints_{args.strategy}/{args.strategy}_alignment.pt'

        evaluate_alignment(
            checkpoint_path=args.checkpoint,
            cached_dir=args.cached_dir,
            strategy=args.strategy,
            device=args.device,
        )


if __name__ == '__main__':
    main()
