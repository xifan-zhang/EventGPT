"""
Evaluate Speculative Decoding Performance
==========================================

Evaluates the trained EAGLE fusion module for actual speculative decoding speedup.

Key Metrics:
1. Token-level acceptance rate (α)
2. Expected speedup: γ / (1 - α^(γ+1)) / (1-α) where γ = speculation length
3. Wall-clock speedup in simulation

Usage:
    python feasible/token_alignment/evaluate_speculative.py \
        --model_path ./feasible/token_alignment/checkpoints_1s/best_model.pt \
        --cached_dir ./feasible/token_alignment/cached_outputs_1s \
        --device cuda
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# Fix protobuf
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from feasible.token_alignment.base import TokenAlignmentDataset
from feasible.token_alignment.eagle_fusion import EAGLEFusionModule, EAGLEFusionConfig


def compute_theoretical_speedup(acceptance_rate: float, gamma: int) -> float:
    """
    Compute theoretical speedup for speculative decoding.

    Formula: S = γ / (1 - α^(γ+1)) / (1-α)

    Where:
        γ = number of draft tokens
        α = acceptance rate
        S = expected speedup

    For high α:
        α = 0.5, γ = 5: S ≈ 2.0x
        α = 0.7, γ = 5: S ≈ 2.8x
        α = 0.9, γ = 5: S ≈ 4.0x
    """
    if acceptance_rate >= 1.0:
        return gamma + 1  # All tokens accepted
    if acceptance_rate <= 0.0:
        return 1.0  # No speedup

    alpha = acceptance_rate
    numerator = 1 - alpha ** (gamma + 1)
    denominator = (1 - alpha)

    expected_accepted = numerator / denominator
    speedup = expected_accepted

    return speedup


def simulate_speculative_decoding(
    model: EAGLEFusionModule,
    draft_hidden: torch.Tensor,
    target_tokens: torch.Tensor,
    gamma: int = 5,
    device: str = 'cuda',
) -> Dict:
    """
    Simulate speculative decoding process.

    Args:
        model: Trained EAGLE fusion module
        draft_hidden: [batch, seq, hidden] draft hidden states
        target_tokens: [batch, seq] target token sequence
        gamma: Number of tokens to speculate
        device: Device to use

    Returns:
        Dict with simulation results
    """
    model.eval()
    batch_size = draft_hidden.size(0)

    all_accepted = []
    all_speculated = []
    all_steps = []

    with torch.no_grad():
        for b in range(batch_size):
            sample_hidden = draft_hidden[b:b+1].to(device)
            sample_target = target_tokens[b].to(device)

            # Find sequence length (non-padding)
            seq_len = (sample_target != 0).sum().item()
            if seq_len < gamma + 1:
                continue

            # Simulate position by position
            position = 0
            total_accepted = 0
            total_speculated = 0
            num_steps = 0

            while position < seq_len - 1:
                # Get current hidden state
                current_hidden = sample_hidden[:, position:position+1]

                # Speculate gamma tokens
                spec_len = min(gamma, seq_len - position - 1)

                # Get draft predictions
                input_tokens = sample_target[position:position+spec_len].unsqueeze(0)
                hidden_for_spec = sample_hidden[:, position:position+spec_len]

                outputs = model.forward(
                    hidden_for_spec,
                    input_tokens=input_tokens,
                )
                logits = outputs['logits']  # [1, spec_len, vocab]

                # Get predicted tokens
                predicted = logits.argmax(dim=-1)[0]  # [spec_len]

                # Get target tokens
                target_slice = sample_target[position+1:position+spec_len+1]

                # Count consecutive matches (speculative decoding accepts longest prefix)
                accepted_count = 0
                for i in range(spec_len):
                    if predicted[i].item() == target_slice[i].item():
                        accepted_count += 1
                    else:
                        break

                total_accepted += accepted_count
                total_speculated += spec_len
                num_steps += 1

                # Move position: accepted tokens + 1 target verification token
                position += accepted_count + 1

            if total_speculated > 0:
                all_accepted.append(total_accepted)
                all_speculated.append(total_speculated)
                all_steps.append(num_steps)

    if not all_accepted:
        return {'error': 'No valid samples'}

    total_accepted = sum(all_accepted)
    total_speculated = sum(all_speculated)
    total_steps = sum(all_steps)

    acceptance_rate = total_accepted / total_speculated if total_speculated > 0 else 0
    avg_accepted_per_step = total_accepted / total_steps if total_steps > 0 else 0

    # Compute effective speedup
    # Without speculation: seq_len steps
    # With speculation: num_steps (each step processes accepted+1 tokens)
    avg_seq_len = sum([(t != 0).sum().item() for t in target_tokens]) / len(target_tokens)
    baseline_steps = avg_seq_len
    spec_steps = sum(all_steps) / len(all_steps)

    simulated_speedup = baseline_steps / spec_steps if spec_steps > 0 else 1.0

    return {
        'acceptance_rate': acceptance_rate,
        'total_accepted': total_accepted,
        'total_speculated': total_speculated,
        'avg_accepted_per_step': avg_accepted_per_step,
        'theoretical_speedup': compute_theoretical_speedup(acceptance_rate, gamma),
        'simulated_speedup': simulated_speedup,
        'baseline_steps': baseline_steps,
        'speculative_steps': spec_steps,
    }


def evaluate_at_different_gammas(
    model: EAGLEFusionModule,
    draft_hidden: torch.Tensor,
    target_tokens: torch.Tensor,
    device: str = 'cuda',
) -> Dict:
    """Evaluate at different speculation lengths."""
    results = {}

    for gamma in [1, 2, 3, 4, 5, 7, 10]:
        print(f"  Evaluating γ={gamma}...")
        result = simulate_speculative_decoding(
            model, draft_hidden, target_tokens, gamma=gamma, device=device
        )
        results[f'gamma_{gamma}'] = result

    return results


def compare_with_baseline(
    model: EAGLEFusionModule,
    dataset: TokenAlignmentDataset,
    device: str = 'cuda',
) -> Dict:
    """
    Compare EAGLE fusion with baseline direct token matching.
    """
    # Baseline: direct token matching
    baseline_correct = 0
    baseline_total = 0

    for i in range(len(dataset)):
        draft = dataset.draft_tokens[i]
        target = dataset.target_tokens[i]

        min_len = min(len(draft), len(target))
        mask = (target[:min_len] != 0) & (draft[:min_len] != 0)

        correct = ((draft[:min_len] == target[:min_len]) & mask).sum().item()
        total = mask.sum().item()

        baseline_correct += correct
        baseline_total += total

    baseline_acceptance = baseline_correct / baseline_total if baseline_total > 0 else 0

    # EAGLE fusion
    model.eval()
    eagle_correct = 0
    eagle_total = 0

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating EAGLE"):
            draft_hidden = dataset.draft_hidden[i:i+1].to(device)
            target_tokens = dataset.target_tokens[i:i+1].to(device)

            # Shift for autoregressive prediction
            input_tokens = target_tokens[:, :-1]
            target_labels = target_tokens[:, 1:]
            draft_hidden_input = draft_hidden[:, :-1]

            outputs = model.forward(
                draft_hidden_input,
                input_tokens=input_tokens,
            )
            predictions = outputs['logits'].argmax(dim=-1)

            mask = (target_labels != 0)
            correct = ((predictions == target_labels) & mask).sum().item()
            total = mask.sum().item()

            eagle_correct += correct
            eagle_total += total

    eagle_acceptance = eagle_correct / eagle_total if eagle_total > 0 else 0

    # Compute improvement
    improvement = eagle_acceptance - baseline_acceptance
    improvement_pct = (improvement / baseline_acceptance * 100) if baseline_acceptance > 0 else 0

    return {
        'baseline_acceptance': baseline_acceptance,
        'eagle_acceptance': eagle_acceptance,
        'improvement_absolute': improvement,
        'improvement_relative_pct': improvement_pct,
        'baseline_speedup_gamma5': compute_theoretical_speedup(baseline_acceptance, 5),
        'eagle_speedup_gamma5': compute_theoretical_speedup(eagle_acceptance, 5),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate speculative decoding performance")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument('--cached_dir', type=str, required=True,
                        help="Path to cached features directory")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use")
    parser.add_argument('--output_path', type=str, default=None,
                        help="Path to save evaluation results")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    config = checkpoint['config']

    if isinstance(config, dict):
        config = EAGLEFusionConfig(**config)

    model = EAGLEFusionModule(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset
    print(f"\nLoading dataset from {args.cached_dir}...")
    dataset = TokenAlignmentDataset.from_directory(args.cached_dir)
    print(f"Loaded {len(dataset)} samples")

    # Compare with baseline
    print("\n" + "="*60)
    print("COMPARISON: Baseline vs EAGLE Fusion")
    print("="*60)

    comparison = compare_with_baseline(model, dataset, args.device)

    print(f"\nBaseline (direct token matching):")
    print(f"  Acceptance rate: {comparison['baseline_acceptance']*100:.2f}%")
    print(f"  Theoretical speedup (γ=5): {comparison['baseline_speedup_gamma5']:.2f}x")

    print(f"\nEAGLE Fusion:")
    print(f"  Acceptance rate: {comparison['eagle_acceptance']*100:.2f}%")
    print(f"  Theoretical speedup (γ=5): {comparison['eagle_speedup_gamma5']:.2f}x")

    print(f"\nImprovement:")
    print(f"  Absolute: +{comparison['improvement_absolute']*100:.2f}%")
    print(f"  Relative: +{comparison['improvement_relative_pct']:.1f}%")

    # Evaluate at different gamma values
    print("\n" + "="*60)
    print("SPECULATIVE DECODING SIMULATION")
    print("="*60)

    gamma_results = evaluate_at_different_gammas(
        model,
        dataset.draft_hidden,
        dataset.target_tokens,
        device=args.device,
    )

    print("\nResults by speculation length (γ):")
    print("-" * 60)
    print(f"{'γ':>3} | {'Accept%':>8} | {'Accepted/Step':>13} | {'Theoretical':>11} | {'Simulated':>9}")
    print("-" * 60)

    for key, result in gamma_results.items():
        if 'error' in result:
            continue
        gamma = int(key.split('_')[1])
        print(f"{gamma:>3} | {result['acceptance_rate']*100:>7.2f}% | "
              f"{result['avg_accepted_per_step']:>13.2f} | "
              f"{result['theoretical_speedup']:>10.2f}x | "
              f"{result['simulated_speedup']:>8.2f}x")

    # Save results
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(args.cached_dir).parent / 'evaluation_results.json'

    results = {
        'model_path': args.model_path,
        'cached_dir': args.cached_dir,
        'num_samples': len(dataset),
        'comparison': comparison,
        'gamma_results': gamma_results,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    baseline_speedup = comparison['baseline_speedup_gamma5']
    eagle_speedup = comparison['eagle_speedup_gamma5']
    speedup_improvement = eagle_speedup / baseline_speedup

    print(f"\nWith EAGLE Fusion trained on {len(dataset)} samples:")
    print(f"  Token acceptance: {comparison['baseline_acceptance']*100:.1f}% -> {comparison['eagle_acceptance']*100:.1f}%")
    print(f"  Speedup (γ=5): {baseline_speedup:.2f}x -> {eagle_speedup:.2f}x")
    print(f"  Speedup improvement: {speedup_improvement:.2f}x")

    if comparison['eagle_acceptance'] >= 0.5:
        print(f"\n  TARGET ACHIEVED: Acceptance rate ≥ 50%")
    else:
        gap = 0.5 - comparison['eagle_acceptance']
        print(f"\n  Gap to 50% target: {gap*100:.1f}%")
        print(f"  Recommendations:")
        print(f"    1. Train on more data")
        print(f"    2. Use target logits for KL loss")
        print(f"    3. Increase model capacity")
        print(f"    4. Fine-tune EventGPT LM head directly")


if __name__ == "__main__":
    main()
