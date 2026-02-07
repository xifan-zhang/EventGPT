"""
Token Alignment: Train and Evaluate with 5-Stage Timing
========================================================

Train on DSEC train set, evaluate on DSEC test set.
Records 5-stage timing and runs evaluation in parallel.

Usage:
    python feasible/token_alignment/train_and_evaluate.py \
        --train_json ./data/my_egpt_dsec_train/my_egpt_dsec_train_1s/EventGPT_Instruction_Subset.json \
        --test_benchmark ./feasible/benchmark_parallel_prefill/results/parallel_prefill_5stages_20260127_160820.json \
        --output_dir ./feasible/token_alignment/results_1s \
        --num_epochs 50 \
        --device cuda
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Fix protobuf
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


@dataclass
class TrainConfig:
    """Training configuration."""
    train_json: str
    test_benchmark: str
    output_dir: str
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_seq_len: int = 128
    embed_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    device: str = 'cuda'
    max_train_samples: int = -1  # -1 for all
    early_stopping: int = 10


class TokenAlignmentDataset(Dataset):
    """Dataset for token alignment from benchmark results."""

    def __init__(
        self,
        draft_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        sample_ids: List[str],
        stage_times: Optional[Dict] = None,
    ):
        self.draft_tokens = draft_tokens
        self.target_tokens = target_tokens
        self.sample_ids = sample_ids
        self.stage_times = stage_times or {}

    def __len__(self):
        return len(self.draft_tokens)

    def __getitem__(self, idx):
        return {
            'draft_tokens': self.draft_tokens[idx],
            'target_tokens': self.target_tokens[idx],
            'sample_id': self.sample_ids[idx],
        }

    @classmethod
    def from_benchmark_json(cls, json_path: str, max_seq_len: int = 128, max_samples: int = -1):
        """Load from benchmark results JSON (parallel_prefill format)."""
        with open(json_path) as f:
            data = json.load(f)

        results = data.get('results', [])
        if max_samples > 0:
            results = results[:max_samples]

        draft_tokens_list = []
        target_tokens_list = []
        sample_ids = []
        stage_times = {'egpt': [], 'vl': []}

        for r in results:
            egpt = r.get('eventgpt', {})
            vl = r.get('videollava', {})

            draft_ids = egpt.get('output_tokens', [])
            target_ids = vl.get('output_tokens', [])

            if not draft_ids or not target_ids:
                continue

            # Pad/truncate
            draft_ids = draft_ids[:max_seq_len]
            target_ids = target_ids[:max_seq_len]
            draft_ids = draft_ids + [0] * (max_seq_len - len(draft_ids))
            target_ids = target_ids + [0] * (max_seq_len - len(target_ids))

            draft_tokens_list.append(draft_ids)
            target_tokens_list.append(target_ids)
            sample_ids.append(str(r.get('sample_idx', len(sample_ids))))

            # Store stage times
            stage_times['egpt'].append({
                'stage1': egpt.get('stage1_time', 0),
                'stage2': egpt.get('stage2_time', 0),
                'stage3': egpt.get('stage3_time', 0),
                'stage4': egpt.get('stage4_time', 0),
                'stage5': egpt.get('stage5_time', 0),
                'total': egpt.get('total_time', 0),
            })
            stage_times['vl'].append({
                'stage1': vl.get('stage1_time', 0),
                'stage2': vl.get('stage2_time', 0),
                'stage3': vl.get('stage3_time', 0),
                'stage4': vl.get('stage4_time', 0),
                'stage5': vl.get('stage5_time', 0),
                'total': vl.get('total_time', 0),
            })

        return cls(
            draft_tokens=torch.tensor(draft_tokens_list, dtype=torch.long),
            target_tokens=torch.tensor(target_tokens_list, dtype=torch.long),
            sample_ids=sample_ids,
            stage_times=stage_times,
        )

    @classmethod
    def from_inference_json(cls, json_path: str, max_seq_len: int = 128, max_samples: int = -1):
        """Load from benchmark_inference format."""
        with open(json_path) as f:
            data = json.load(f)

        if max_samples > 0:
            data = data[:max_samples]

        draft_tokens_list = []
        target_tokens_list = []
        sample_ids = []
        stage_times = {'egpt': [], 'vl': []}

        for item in data:
            draft_ids = item.get('egpt_token_ids', [])
            target_ids = item.get('llava_token_ids', [])

            if not draft_ids or not target_ids:
                continue

            # Pad/truncate
            draft_ids = draft_ids[:max_seq_len]
            target_ids = target_ids[:max_seq_len]
            draft_ids = draft_ids + [0] * (max_seq_len - len(draft_ids))
            target_ids = target_ids + [0] * (max_seq_len - len(target_ids))

            draft_tokens_list.append(draft_ids)
            target_tokens_list.append(target_ids)
            sample_ids.append(item.get('id', str(len(sample_ids))))

            stage_times['egpt'].append({
                'stage1': item.get('egpt_stage1_time', 0),
                'stage2': item.get('egpt_stage2_time', 0),
                'stage3': item.get('egpt_stage3_time', 0),
                'total': item.get('egpt_time', 0),
            })
            stage_times['vl'].append({
                'stage1': item.get('llava_stage1_time', 0),
                'stage2': item.get('llava_stage2_time', 0),
                'stage3': item.get('llava_stage3_time', 0),
                'total': item.get('llava_time', 0),
            })

        return cls(
            draft_tokens=torch.tensor(draft_tokens_list, dtype=torch.long),
            target_tokens=torch.tensor(target_tokens_list, dtype=torch.long),
            sample_ids=sample_ids,
            stage_times=stage_times,
        )

    @classmethod
    def from_extraction_json(cls, json_path: str, max_seq_len: int = 128, max_samples: int = -1):
        """Load from extract_tokens_train.py output format."""
        with open(json_path) as f:
            data = json.load(f)

        results = data.get('results', [])
        if max_samples > 0:
            results = results[:max_samples]

        draft_tokens_list = []
        target_tokens_list = []
        sample_ids = []
        stage_times = {'egpt': [], 'vl': []}

        for r in results:
            # extract_tokens_train.py format
            draft_ids = r.get('egpt_tokens', [])
            target_ids = r.get('vl_tokens', [])

            if not draft_ids or not target_ids:
                continue

            # Pad/truncate
            draft_ids = draft_ids[:max_seq_len]
            target_ids = target_ids[:max_seq_len]
            draft_ids = draft_ids + [0] * (max_seq_len - len(draft_ids))
            target_ids = target_ids + [0] * (max_seq_len - len(target_ids))

            draft_tokens_list.append(draft_ids)
            target_tokens_list.append(target_ids)
            sample_ids.append(str(r.get('sample_idx', len(sample_ids))))

            # No timing info in extraction format
            stage_times['egpt'].append({})
            stage_times['vl'].append({})

        return cls(
            draft_tokens=torch.tensor(draft_tokens_list, dtype=torch.long),
            target_tokens=torch.tensor(target_tokens_list, dtype=torch.long),
            sample_ids=sample_ids,
            stage_times=stage_times,
        )


class TokenAdapter(torch.nn.Module):
    """Lightweight token adapter for alignment."""

    def __init__(self, vocab_size: int = 32010, embed_dim: int = 512,
                 num_layers: int = 4, num_heads: int = 8, max_seq_len: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = torch.nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, draft_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = draft_tokens.shape
        device = draft_tokens.device

        seq_len = min(seq_len, self.max_seq_len)
        draft_tokens = draft_tokens[:, :seq_len]

        token_emb = self.embedding(draft_tokens)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )

        x = self.transformer(x, mask=causal_mask)
        logits = self.output_proj(x)

        return logits


def compute_acceptance_metrics(
    model: TokenAdapter,
    dataset: TokenAlignmentDataset,
    device: str = 'cuda',
    batch_size: int = 32,
) -> Dict:
    """Compute acceptance metrics in parallel batches."""
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_correct = 0
    total_top5 = 0
    total_tokens = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            draft_tokens = batch['draft_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            # Shift for autoregressive
            draft_input = draft_tokens[:, :-1]
            target_labels = target_tokens[:, 1:]

            logits = model(draft_input)

            min_len = min(logits.size(1), target_labels.size(1))
            logits = logits[:, :min_len]
            target_labels = target_labels[:, :min_len]

            predictions = logits.argmax(dim=-1)
            mask = (target_labels != 0)

            correct = ((predictions == target_labels) & mask).sum().item()
            total_correct += correct

            top5_preds = logits.topk(5, dim=-1).indices
            top5_match = (top5_preds == target_labels.unsqueeze(-1)).any(dim=-1)
            total_top5 += (top5_match & mask).sum().item()

            total_tokens += mask.sum().item()

            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(target_labels.cpu().tolist())

    acceptance_rate = total_correct / total_tokens if total_tokens > 0 else 0
    top5_rate = total_top5 / total_tokens if total_tokens > 0 else 0

    return {
        'acceptance_rate': acceptance_rate,
        'top5_rate': top5_rate,
        'total_correct': total_correct,
        'total_tokens': total_tokens,
        'predictions': all_predictions,
        'targets': all_targets,
    }


def compute_baseline_acceptance(dataset: TokenAlignmentDataset) -> Dict:
    """Compute baseline (direct token matching) acceptance."""
    total_correct = 0
    total_tokens = 0

    for i in range(len(dataset)):
        draft = dataset.draft_tokens[i]
        target = dataset.target_tokens[i]

        min_len = min(len(draft), len(target))
        mask = (target[:min_len] != 0) & (draft[:min_len] != 0)

        correct = ((draft[:min_len] == target[:min_len]) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()

    return {
        'acceptance_rate': total_correct / total_tokens if total_tokens > 0 else 0,
        'total_correct': total_correct,
        'total_tokens': total_tokens,
    }


def train_model(
    model: TokenAdapter,
    train_dataset: TokenAlignmentDataset,
    val_dataset: Optional[TokenAlignmentDataset],
    config: TrainConfig,
) -> Dict:
    """Train the model."""
    device = torch.device(config.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs * len(train_dataset) // config.batch_size
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    best_val_acc = -1  # Start at -1 so first epoch always saves
    patience_counter = 0
    training_history = []

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            draft_tokens = batch['draft_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            draft_input = draft_tokens[:, :-1]
            target_labels = target_tokens[:, 1:]

            optimizer.zero_grad()
            logits = model(draft_input)

            min_len = min(logits.size(1), target_labels.size(1))
            logits = logits[:, :min_len]
            target_labels = target_labels[:, :min_len]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_labels.reshape(-1),
                ignore_index=0,
                label_smoothing=0.1,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = (target_labels != 0)
                acc = ((preds == target_labels) & mask).sum().float() / mask.sum().clamp(min=1)

            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc.item()*100:.1f}%"})

        avg_train_loss = total_loss / num_batches
        avg_train_acc = total_acc / num_batches

        # Validation
        val_metrics = {}
        if val_dataset is not None:
            val_metrics = compute_acceptance_metrics(model, val_dataset, config.device, config.batch_size)
            val_acc = val_metrics['acceptance_rate']

            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc*100:.1f}%, "
                  f"Val Acc={val_acc*100:.2f}%, Val Top5={val_metrics['top5_rate']*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                }, Path(config.output_dir) / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc*100:.1f}%")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'val_acc': val_metrics.get('acceptance_rate', 0),
            'val_top5': val_metrics.get('top5_rate', 0),
        })

    return {
        'history': training_history,
        'best_val_acc': best_val_acc,
    }


def evaluate_with_timing(
    model: TokenAdapter,
    dataset: TokenAlignmentDataset,
    config: TrainConfig,
    dataset_name: str,
) -> Dict:
    """Evaluate with detailed timing."""
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    # Compute acceptance
    start_time = time.time()
    metrics = compute_acceptance_metrics(model, dataset, config.device, config.batch_size)
    inference_time = time.time() - start_time

    # Get stage times from dataset
    stage_times = dataset.stage_times

    # Compute stage time statistics
    def compute_stats(times_list: List[Dict], key: str) -> Dict:
        values = [t.get(key, 0) for t in times_list if t.get(key, 0) > 0]
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }

    egpt_times = stage_times.get('egpt', [])
    vl_times = stage_times.get('vl', [])

    timing_stats = {
        'eventgpt': {
            'stage1': compute_stats(egpt_times, 'stage1'),
            'stage2': compute_stats(egpt_times, 'stage2'),
            'stage3': compute_stats(egpt_times, 'stage3'),
            'stage4': compute_stats(egpt_times, 'stage4'),
            'stage5': compute_stats(egpt_times, 'stage5'),
            'total': compute_stats(egpt_times, 'total'),
        },
        'videollava': {
            'stage1': compute_stats(vl_times, 'stage1'),
            'stage2': compute_stats(vl_times, 'stage2'),
            'stage3': compute_stats(vl_times, 'stage3'),
            'stage4': compute_stats(vl_times, 'stage4'),
            'stage5': compute_stats(vl_times, 'stage5'),
            'total': compute_stats(vl_times, 'total'),
        },
    }

    return {
        'dataset': dataset_name,
        'num_samples': len(dataset),
        'acceptance_rate': metrics['acceptance_rate'],
        'top5_rate': metrics['top5_rate'],
        'total_correct': metrics['total_correct'],
        'total_tokens': metrics['total_tokens'],
        'inference_time_sec': inference_time,
        'samples_per_sec': len(dataset) / inference_time,
        'timing_stats': timing_stats,
    }


def save_training_curves(history: List[Dict], output_dir: Path, baseline_train: float, baseline_test: float):
    """Save training curves as images."""
    if not history:
        return

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_acc = [h['train_acc'] * 100 for h in history]
    val_acc = [h['val_acc'] * 100 for h in history]
    val_top5 = [h['val_top5'] * 100 for h in history]

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, train_acc, 'b-', linewidth=2, label='Train Acc')
    ax2.plot(epochs, val_acc, 'g-', linewidth=2, label='Val Acc (Top-1)')
    ax2.plot(epochs, val_top5, 'g--', linewidth=2, label='Val Acc (Top-5)')
    ax2.axhline(y=baseline_train * 100, color='b', linestyle=':', alpha=0.5, label=f'Train Baseline ({baseline_train*100:.1f}%)')
    ax2.axhline(y=baseline_test * 100, color='g', linestyle=':', alpha=0.5, label=f'Test Baseline ({baseline_test*100:.1f}%)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Acceptance Rate', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(max(val_top5) + 10, 60))

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Also save individual high-res plots
    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, train_loss, 'b-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, train_acc, 'b-', linewidth=2, marker='o', markersize=3, label='Train')
    ax.plot(epochs, val_acc, 'g-', linewidth=2, marker='s', markersize=3, label='Validation')
    ax.plot(epochs, val_top5, 'r-', linewidth=2, marker='^', markersize=3, label='Val Top-5')
    ax.axhline(y=baseline_test * 100, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_test*100:.1f}%)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax.set_title('Token Acceptance Rate During Training', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_benchmark', type=str,
                        default='./feasible/benchmark_inference/benchmark_results_S1_train.json')
    parser.add_argument('--test_benchmark', type=str,
                        default='./feasible/benchmark_parallel_prefill/results/parallel_prefill_5stages_20260127_160820.json')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory. If not specified, creates task folder with timestamp.')
    parser.add_argument('--task_name', type=str, default=None,
                        help='Task name prefix (e.g., "1s", "500ms"). Auto-detected if not specified.')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_train_samples', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--early_stopping', type=int, default=10)

    args = parser.parse_args()

    # Auto-detect task name from train benchmark path
    task_name = args.task_name
    if task_name is None:
        # Try to extract from path (e.g., "1s", "500ms")
        train_path = args.train_benchmark.lower()
        for duration in ['500ms', '200ms', '1s', '2s', '4s', '5s', '8s', '10s', '16s', '20s']:
            if duration in train_path:
                task_name = duration
                break
        if task_name is None:
            task_name = 'default'

    # Create task folder with timestamp if output_dir not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        task_folder = Path('./feasible/token_alignment/task') / f'{task_name}_{timestamp}'
    else:
        task_folder = Path(args.output_dir)

    config = TrainConfig(
        train_json=args.train_benchmark,
        test_benchmark=args.test_benchmark,
        output_dir=str(task_folder),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        device=args.device,
        early_stopping=args.early_stopping,
    )

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Task folder: {output_path}")

    print("="*60)
    print("Token Alignment: Train and Evaluate")
    print("="*60)

    # Load datasets
    print("\nLoading datasets...")

    # Load train data - detect format automatically
    if 'benchmark_inference' in config.train_json or 'S1_train' in config.train_json:
        train_dataset = TokenAlignmentDataset.from_inference_json(
            config.train_json, max_samples=config.max_train_samples
        )
    elif 'train_tokens' in config.train_json or 'extraction' in config.train_json:
        # Format from extract_tokens_train.py
        train_dataset = TokenAlignmentDataset.from_extraction_json(
            config.train_json, max_samples=config.max_train_samples
        )
    else:
        # Try to auto-detect format by reading the file
        with open(config.train_json) as f:
            probe = json.load(f)
        if isinstance(probe, dict) and 'results' in probe:
            first_result = probe['results'][0] if probe['results'] else {}
            if 'egpt_tokens' in first_result:
                # extraction format
                train_dataset = TokenAlignmentDataset.from_extraction_json(
                    config.train_json, max_samples=config.max_train_samples
                )
            elif 'eventgpt' in first_result:
                # benchmark format
                train_dataset = TokenAlignmentDataset.from_benchmark_json(
                    config.train_json, max_samples=config.max_train_samples
                )
            else:
                train_dataset = TokenAlignmentDataset.from_benchmark_json(
                    config.train_json, max_samples=config.max_train_samples
                )
        else:
            train_dataset = TokenAlignmentDataset.from_benchmark_json(
                config.train_json, max_samples=config.max_train_samples
            )
    print(f"Train samples: {len(train_dataset)}")

    # Load test data (auto-detect format)
    with open(config.test_benchmark) as f:
        test_probe = json.load(f)
    if 'results' in test_probe and test_probe['results']:
        first_test_result = test_probe['results'][0]
        if 'egpt_tokens' in first_test_result:
            # extraction format
            test_dataset = TokenAlignmentDataset.from_extraction_json(config.test_benchmark)
        else:
            # benchmark format
            test_dataset = TokenAlignmentDataset.from_benchmark_json(config.test_benchmark)
    else:
        test_dataset = TokenAlignmentDataset.from_benchmark_json(config.test_benchmark)
    print(f"Test samples: {len(test_dataset)}")

    # Compute baselines
    print("\nComputing baselines...")
    train_baseline = compute_baseline_acceptance(train_dataset)
    test_baseline = compute_baseline_acceptance(test_dataset)
    print(f"Train baseline: {train_baseline['acceptance_rate']*100:.2f}%")
    print(f"Test baseline: {test_baseline['acceptance_rate']*100:.2f}%")

    # Create model (vocab_size=32010 to accommodate Video-LLaVA special tokens like 32001)
    model = TokenAdapter(
        vocab_size=32010,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {params:,} (~{params*4/1024/1024:.1f} MB)")

    # Train
    print("\n" + "="*60)
    print("Training")
    print("="*60)

    train_result = train_model(model, train_dataset, test_dataset, config)

    # Load best model
    best_checkpoint = torch.load(output_path / 'best_model.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation (Parallel)")
    print("="*60)

    train_eval = evaluate_with_timing(model, train_dataset, config, "train")
    test_eval = evaluate_with_timing(model, test_dataset, config, "test")

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Dataset':<10} {'Samples':<10} {'Baseline':<12} {'Model':<12} {'Top-5':<10} {'Improve':<10}")
    print("-"*60)
    print(f"{'Train':<10} {len(train_dataset):<10} {train_baseline['acceptance_rate']*100:>10.2f}% "
          f"{train_eval['acceptance_rate']*100:>10.2f}% {train_eval['top5_rate']*100:>8.2f}% "
          f"+{(train_eval['acceptance_rate']-train_baseline['acceptance_rate'])*100:>7.2f}%")
    print(f"{'Test':<10} {len(test_dataset):<10} {test_baseline['acceptance_rate']*100:>10.2f}% "
          f"{test_eval['acceptance_rate']*100:>10.2f}% {test_eval['top5_rate']*100:>8.2f}% "
          f"+{(test_eval['acceptance_rate']-test_baseline['acceptance_rate'])*100:>7.2f}%")

    # Print timing
    print(f"\n{'='*60}")
    print("5-STAGE TIMING (Test Set)")
    print(f"{'='*60}")
    print(f"\n{'Stage':<12} {'EventGPT (ms)':<20} {'Video-LLaVA (ms)':<20}")
    print("-"*60)
    for stage in ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'total']:
        egpt_stat = test_eval['timing_stats']['eventgpt'].get(stage, {})
        vl_stat = test_eval['timing_stats']['videollava'].get(stage, {})
        egpt_str = f"{egpt_stat.get('mean', 0)*1000:.1f} ± {egpt_stat.get('std', 0)*1000:.1f}"
        vl_str = f"{vl_stat.get('mean', 0)*1000:.1f} ± {vl_stat.get('std', 0)*1000:.1f}"
        print(f"{stage:<12} {egpt_str:<20} {vl_str:<20}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': asdict(config),
        'model_params': params,
        'baseline': {
            'train': train_baseline,
            'test': test_baseline,
        },
        'final': {
            'train': train_eval,
            'test': test_eval,
        },
        'training_history': train_result['history'],
        'best_val_acc': train_result['best_val_acc'],
    }

    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save training curves
    save_training_curves(
        train_result['history'],
        output_path,
        train_baseline['acceptance_rate'],
        test_baseline['acceptance_rate']
    )

    # Save markdown report
    report = f"""# Token Alignment Results

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Dataset | Samples | Baseline | Model | Top-5 | Improvement |
|---------|---------|----------|-------|-------|-------------|
| Train | {len(train_dataset)} | {train_baseline['acceptance_rate']*100:.2f}% | {train_eval['acceptance_rate']*100:.2f}% | {train_eval['top5_rate']*100:.2f}% | +{(train_eval['acceptance_rate']-train_baseline['acceptance_rate'])*100:.2f}% |
| Test | {len(test_dataset)} | {test_baseline['acceptance_rate']*100:.2f}% | {test_eval['acceptance_rate']*100:.2f}% | {test_eval['top5_rate']*100:.2f}% | +{(test_eval['acceptance_rate']-test_baseline['acceptance_rate'])*100:.2f}% |

## 5-Stage Timing (Test Set, ms)

| Stage | EventGPT | Video-LLaVA |
|-------|----------|-------------|
"""
    for stage in ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'total']:
        egpt_stat = test_eval['timing_stats']['eventgpt'].get(stage, {})
        vl_stat = test_eval['timing_stats']['videollava'].get(stage, {})
        report += f"| {stage} | {egpt_stat.get('mean', 0)*1000:.1f} ± {egpt_stat.get('std', 0)*1000:.1f} | {vl_stat.get('mean', 0)*1000:.1f} ± {vl_stat.get('std', 0)*1000:.1f} |\n"

    report += f"""
## Training Curves

![Training Curves](training_curves.png)

- Loss curve: [loss_curve.png](loss_curve.png)
- Accuracy curve: [accuracy_curve.png](accuracy_curve.png)

## Configuration

- Epochs: {config.num_epochs} (early stopping: {config.early_stopping})
- Batch size: {config.batch_size}
- Learning rate: {config.learning_rate}
- Model parameters: {params:,}

## Speedup Analysis

With acceptance rate α = {test_eval['acceptance_rate']*100:.1f}% and γ = 5 draft tokens:
- Theoretical speedup: {(1 - test_eval['acceptance_rate']**6) / (1 - test_eval['acceptance_rate']):.2f}x
"""

    with open(output_path / 'RESULTS.md', 'w') as f:
        f.write(report)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
