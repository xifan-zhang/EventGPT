#!/usr/bin/env python3
"""
Measure Feature-Level Acceptance Rate for Speculative Decoding.

Author: Alice Zhang
Date: 2026-02-07

Computes comprehensive metrics for evaluating hidden state alignment quality:
- Cosine similarity statistics
- Acceptance rates at multiple thresholds
- Consecutive accepts (key metric for SD)
- Prefill speedup estimation
- Decoding speedup estimation (gamma=5)
- End-to-end speedup estimation

Usage:
    python measure_feature_acceptance.py \
        --checkpoint ./checkpoints/hidden_adapter/best_model.pt \
        --test_data ./hidden_states/hidden_states_test_10q.pt

Updated: 2026-02-06
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from feasible.feature_alignment.hidden_adapter import load_any_adapter


@dataclass
class TimingConfig:
    """Timing configuration for speedup estimation (in ms)."""
    egpt_prefill: float = 130.0      # EventGPT vision + prefill
    egpt_per_token: float = 25.0     # EventGPT per-token generation
    vl_prefill: float = 310.0        # Video-LLaVA prefill
    vl_per_token: float = 25.0       # Video-LLaVA per-token generation
    adapter_latency: float = 1.5     # Hidden adapter forward pass


@dataclass
class SDConfig:
    """Speculative decoding configuration."""
    gamma_decode: int = 5            # Draft tokens in decode phase
    thresholds: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)


def compute_all_metrics_parallel(
    aligned_hidden: torch.Tensor,
    vl_hidden: torch.Tensor,
    mask: torch.Tensor,
    sd_config: SDConfig = None,
    timing_config: TimingConfig = None,
) -> Dict:
    """
    Compute all metrics in parallel using vectorized operations.

    Args:
        aligned_hidden: [batch, seq, hidden] aligned EGPT hidden states
        vl_hidden: [batch, seq, hidden] VL hidden states
        mask: [batch, seq] valid positions (1 = valid, 0 = padding)
        sd_config: Speculative decoding configuration
        timing_config: Timing configuration for speedup estimation

    Returns:
        Dict with all metrics
    """
    if sd_config is None:
        sd_config = SDConfig()
    if timing_config is None:
        timing_config = TimingConfig()

    batch_size, max_seq, hidden_dim = aligned_hidden.shape
    device = aligned_hidden.device

    # =========================================================================
    # 1. COSINE SIMILARITY (Parallel)
    # =========================================================================
    aligned_norm = F.normalize(aligned_hidden, dim=-1)
    vl_norm = F.normalize(vl_hidden, dim=-1)
    cos_sim = (aligned_norm * vl_norm).sum(dim=-1)  # [batch, seq]

    # Mask out padding positions
    cos_sim_masked = cos_sim * mask  # [batch, seq]
    valid_count = mask.sum()

    # Statistics over valid positions only
    valid_sims = cos_sim[mask.bool()]

    metrics = {
        'timestamp': datetime.now().isoformat(),
        'batch_size': batch_size,
        'max_seq_len': max_seq,
        'hidden_dim': hidden_dim,
        'total_valid_positions': int(valid_count.item()),

        # Cosine similarity statistics
        'cos_sim_mean': valid_sims.mean().item(),
        'cos_sim_std': valid_sims.std().item(),
        'cos_sim_min': valid_sims.min().item(),
        'cos_sim_max': valid_sims.max().item(),
        'cos_sim_median': valid_sims.median().item(),
    }

    # =========================================================================
    # 2. ACCEPTANCE RATES AT THRESHOLDS (Parallel)
    # =========================================================================
    for thresh in sd_config.thresholds:
        accept_mask = (cos_sim > thresh) & mask.bool()  # [batch, seq]
        accept_rate = accept_mask.float().sum() / valid_count
        metrics[f'accept_{int(thresh*100)}'] = accept_rate.item()

    # =========================================================================
    # 3. CONSECUTIVE ACCEPTS (Parallel using cumsum trick)
    # =========================================================================
    # For each sample, find first position where cos_sim < threshold
    # consecutive = argmax(cos_sim < threshold) or seq_len if all accepted

    consecutive_metrics = {}
    for thresh in sd_config.thresholds:
        thresh_key = int(thresh * 100)

        # [batch, seq] - True where accepted
        accept_per_pos = cos_sim > thresh

        # Find first rejection: use cumsum + min trick
        # If accepted at all positions: consecutive = seq_len
        # Otherwise: consecutive = first False position

        # Convert to int: 1 = accept, 0 = reject
        accept_int = accept_per_pos.int()  # [batch, seq]

        # Cumulative product: becomes 0 after first rejection
        cumprod = accept_int.cumprod(dim=1)  # [batch, seq]

        # Sum to get consecutive count (sum of 1s before first 0)
        consecutive_per_sample = cumprod.sum(dim=1).float()  # [batch]

        # Clamp to valid sequence lengths
        seq_lens = mask.sum(dim=1)  # [batch]
        consecutive_per_sample = torch.minimum(consecutive_per_sample, seq_lens)

        consecutive_metrics[f'consecutive_mean_{thresh_key}'] = consecutive_per_sample.mean().item()
        consecutive_metrics[f'consecutive_std_{thresh_key}'] = consecutive_per_sample.std().item()
        consecutive_metrics[f'consecutive_max_{thresh_key}'] = consecutive_per_sample.max().item()
        consecutive_metrics[f'consecutive_min_{thresh_key}'] = consecutive_per_sample.min().item()
        consecutive_metrics[f'consecutive_median_{thresh_key}'] = consecutive_per_sample.median().item()

    metrics.update(consecutive_metrics)

    # =========================================================================
    # 4. NUM HIDDEN TOKENS (gamma_prefill)
    # =========================================================================
    # gamma_prefill = average sequence length (valid tokens per sample)
    seq_lens = mask.sum(dim=1)  # [batch]
    gamma_prefill = seq_lens.float().mean().item()

    metrics['num_hidden_tokens_mean'] = gamma_prefill
    metrics['num_hidden_tokens_std'] = seq_lens.float().std().item()
    metrics['num_hidden_tokens_min'] = seq_lens.min().item()
    metrics['num_hidden_tokens_max'] = seq_lens.max().item()

    # =========================================================================
    # 5. SPECULATIVE DECODING ACCEPTANCE RATES
    # =========================================================================
    gamma_decode = sd_config.gamma_decode

    # SD with gamma = gamma_prefill (all prefill tokens as draft)
    # Use consecutive accepts as the number of accepted tokens
    consec_90 = metrics['consecutive_mean_90']
    accept_rate_prefill = consec_90 / gamma_prefill if gamma_prefill > 0 else 0
    metrics['sd_accept_rate_prefill_gamma'] = accept_rate_prefill
    metrics['sd_gamma_prefill'] = gamma_prefill

    # SD with gamma = 5 (standard decode phase)
    # For gamma=5, we look at first 5 positions
    if max_seq >= gamma_decode:
        first_5_accept = (cos_sim[:, :gamma_decode] > 0.90) & mask[:, :gamma_decode].bool()
        # Consecutive in first 5
        first_5_cumprod = first_5_accept.int().cumprod(dim=1)
        consec_5 = first_5_cumprod.sum(dim=1).float().mean().item()
        accept_rate_decode = consec_5 / gamma_decode
    else:
        consec_5 = consec_90
        accept_rate_decode = accept_rate_prefill

    metrics['sd_accept_rate_decode_gamma5'] = accept_rate_decode
    metrics['sd_consecutive_gamma5'] = consec_5
    metrics['sd_gamma_decode'] = gamma_decode

    # =========================================================================
    # 6. SPEEDUP ESTIMATIONS
    # =========================================================================
    t = timing_config

    # --- Prefill Speedup ---
    # Baseline: VL prefill time
    # With adapter: EGPT prefill + adapter (can run in parallel with VL prefill)
    # Overlap = min(VL_prefill, EGPT_prefill + adapter)

    egpt_total_prefill = t.egpt_prefill + t.adapter_latency
    overlap_window = min(t.vl_prefill, egpt_total_prefill)

    # Free draft tokens during overlap
    # During VL prefill, EGPT can generate tokens
    free_draft_time = max(0, t.vl_prefill - t.egpt_prefill)
    free_draft_tokens = free_draft_time / t.egpt_per_token if t.egpt_per_token > 0 else 0

    # Effective accepted tokens in prefill phase
    effective_prefill_tokens = min(consec_90, free_draft_tokens)

    # Prefill speedup = tokens saved / baseline
    # Baseline decode for these tokens = effective_prefill_tokens * vl_per_token
    prefill_time_saved = effective_prefill_tokens * t.vl_per_token
    prefill_speedup = 1 + (prefill_time_saved / t.vl_prefill) if t.vl_prefill > 0 else 1.0

    metrics['speedup_prefill'] = prefill_speedup
    metrics['prefill_overlap_window_ms'] = overlap_window
    metrics['prefill_free_draft_tokens'] = free_draft_tokens
    metrics['prefill_effective_tokens'] = effective_prefill_tokens

    # --- Decode Speedup (gamma=5) ---
    # Standard SD formula: speedup = (accepted + 1) / (1 + overhead_ratio)
    # accepted = consecutive accepts with gamma=5
    overhead_ratio = t.adapter_latency / t.vl_per_token if t.vl_per_token > 0 else 0.1

    decode_speedup = (consec_5 + 1) / (1 + overhead_ratio)
    metrics['speedup_decode_gamma5'] = decode_speedup
    metrics['decode_overhead_ratio'] = overhead_ratio

    # --- End-to-End Speedup ---
    # Baseline: VL_prefill + N * VL_per_token
    # With SD: max(VL_prefill, EGPT_prefill) + adapter + verify_time
    # where verify_time depends on acceptance

    # Assume 50 output tokens for estimation
    num_output_tokens = 50

    baseline_time = t.vl_prefill + num_output_tokens * t.vl_per_token

    # With prefill hiding + decode SD
    # Prefill phase: parallel, so max(VL, EGPT+adapter)
    parallel_prefill = max(t.vl_prefill, t.egpt_prefill + t.adapter_latency)

    # Decode phase with SD:
    # Each iteration: draft gamma tokens, verify, accept some
    # Time per accepted token ≈ vl_per_token * (gamma + 1) / (accepted + 1)
    avg_accepted = consec_5
    if avg_accepted > 0:
        iterations_needed = num_output_tokens / (avg_accepted + 1)
        decode_time = iterations_needed * (t.vl_per_token * (gamma_decode + 1) + t.adapter_latency)
    else:
        decode_time = num_output_tokens * t.vl_per_token

    total_sd_time = parallel_prefill + decode_time
    e2e_speedup = baseline_time / total_sd_time if total_sd_time > 0 else 1.0

    metrics['speedup_e2e'] = e2e_speedup
    metrics['e2e_baseline_time_ms'] = baseline_time
    metrics['e2e_sd_time_ms'] = total_sd_time
    metrics['e2e_num_output_tokens'] = num_output_tokens

    # =========================================================================
    # 7. PER-POSITION ACCEPTANCE (Parallel)
    # =========================================================================
    # Mean acceptance rate at each position
    position_accept_90 = []
    position_mean_sim = []

    for pos in range(min(max_seq, 20)):  # First 20 positions
        valid_at_pos = mask[:, pos].bool()
        if valid_at_pos.sum() > 0:
            sims_at_pos = cos_sim[valid_at_pos, pos]
            position_accept_90.append((sims_at_pos > 0.90).float().mean().item())
            position_mean_sim.append(sims_at_pos.mean().item())
        else:
            break

    metrics['position_accept_90'] = position_accept_90
    metrics['position_mean_sim'] = position_mean_sim

    return metrics, cos_sim, valid_sims


def compute_per_position_stats(
    cos_sim: torch.Tensor,
    mask: torch.Tensor,
    thresholds: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
) -> List[Dict]:
    """Compute detailed per-position statistics."""
    max_seq = cos_sim.shape[1]
    position_stats = []

    for pos in range(max_seq):
        valid_mask = mask[:, pos].bool()
        if valid_mask.sum() == 0:
            break

        pos_sims = cos_sim[valid_mask, pos]
        stats = {
            'position': pos,
            'mean': pos_sims.mean().item(),
            'std': pos_sims.std().item(),
            'num_samples': int(valid_mask.sum().item()),
        }

        for thresh in thresholds:
            stats[f'accept_{int(thresh*100)}'] = (pos_sims > thresh).float().mean().item()

        position_stats.append(stats)

    return position_stats


def print_metrics_report(metrics: Dict, position_stats: List[Dict]):
    """Print formatted metrics report."""
    print("\n" + "=" * 80)
    print("FEATURE-LEVEL SPECULATIVE DECODING METRICS REPORT")
    print(f"Generated: {metrics['timestamp']}")
    print("=" * 80)

    print("\n[1] DATASET STATISTICS")
    print("-" * 40)
    print(f"  Samples:           {metrics['batch_size']}")
    print(f"  Max sequence len:  {metrics['max_seq_len']}")
    print(f"  Hidden dim:        {metrics['hidden_dim']}")
    print(f"  Total positions:   {metrics['total_valid_positions']}")

    print("\n[2] COSINE SIMILARITY")
    print("-" * 40)
    print(f"  Mean:    {metrics['cos_sim_mean']:.4f}")
    print(f"  Std:     {metrics['cos_sim_std']:.4f}")
    print(f"  Median:  {metrics['cos_sim_median']:.4f}")
    print(f"  Min:     {metrics['cos_sim_min']:.4f}")
    print(f"  Max:     {metrics['cos_sim_max']:.4f}")

    print("\n[3] ACCEPTANCE RATES (Overall)")
    print("-" * 40)
    for thresh in [80, 85, 90, 95]:
        rate = metrics.get(f'accept_{thresh}', 0)
        bar = "█" * int(rate * 20)
        print(f"  @0.{thresh}: {rate:6.2%} {bar}")

    print("\n[4] CONSECUTIVE ACCEPTS (Key SD Metric)")
    print("-" * 40)
    for thresh in [80, 85, 90, 95]:
        mean = metrics.get(f'consecutive_mean_{thresh}', 0)
        std = metrics.get(f'consecutive_std_{thresh}', 0)
        max_val = metrics.get(f'consecutive_max_{thresh}', 0)
        print(f"  @0.{thresh}: {mean:5.2f} ± {std:4.2f} tokens (max: {max_val:.0f})")

    print("\n[5] NUM HIDDEN TOKENS (γ_prefill)")
    print("-" * 40)
    print(f"  Mean:    {metrics['num_hidden_tokens_mean']:.2f} tokens")
    print(f"  Std:     {metrics['num_hidden_tokens_std']:.2f}")
    print(f"  Range:   [{metrics['num_hidden_tokens_min']:.0f}, {metrics['num_hidden_tokens_max']:.0f}]")

    print("\n[6] SPECULATIVE DECODING ACCEPTANCE")
    print("-" * 40)
    print(f"  Prefill SD (γ={metrics['sd_gamma_prefill']:.1f}):")
    print(f"    Consecutive accepts: {metrics['consecutive_mean_90']:.2f} tokens")
    print(f"    Accept rate:         {metrics['sd_accept_rate_prefill_gamma']:.2%}")
    print(f"  Decode SD (γ={metrics['sd_gamma_decode']}):")
    print(f"    Consecutive accepts: {metrics['sd_consecutive_gamma5']:.2f} tokens")
    print(f"    Accept rate:         {metrics['sd_accept_rate_decode_gamma5']:.2%}")

    print("\n[7] SPEEDUP ESTIMATIONS")
    print("-" * 40)
    print(f"  Prefill Speedup:      {metrics['speedup_prefill']:.2f}x")
    print(f"    - Overlap window:   {metrics['prefill_overlap_window_ms']:.1f} ms")
    print(f"    - Free draft tokens:{metrics['prefill_free_draft_tokens']:.1f}")
    print(f"    - Effective tokens: {metrics['prefill_effective_tokens']:.1f}")
    print(f"  Decode Speedup (γ=5): {metrics['speedup_decode_gamma5']:.2f}x")
    print(f"    - Overhead ratio:   {metrics['decode_overhead_ratio']:.3f}")
    print(f"  End-to-End Speedup:   {metrics['speedup_e2e']:.2f}x")
    print(f"    - Baseline time:    {metrics['e2e_baseline_time_ms']:.1f} ms")
    print(f"    - SD time:          {metrics['e2e_sd_time_ms']:.1f} ms")

    print("\n[8] PER-POSITION ACCEPTANCE @0.90 (First 10 positions)")
    print("-" * 40)
    for i, rate in enumerate(metrics['position_accept_90'][:10]):
        bar = "█" * int(rate * 20)
        print(f"  Pos {i:2d}: {rate:6.2%} {bar}")

    print("\n" + "=" * 80)


def plot_stage_timeline(
    metrics: Dict,
    timing_config: TimingConfig,
    output_dir: Path,
):
    """
    Generate per-stage timeline visualization showing time consumption ratio.

    Shows:
    - Baseline: VL prefill → VL decode (sequential)
    - With SD: Parallel prefill + speculative decode iterations
    """
    t = timing_config
    num_tokens = metrics['e2e_num_output_tokens']
    consec_5 = metrics['sd_consecutive_gamma5']
    gamma = metrics['sd_gamma_decode']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # =========================================================================
    # Plot 1: Stage-by-stage time breakdown (horizontal bar)
    # =========================================================================
    ax1 = axes[0]

    # Baseline stages
    baseline_stages = [
        ('VL Prefill', t.vl_prefill, '#3498db'),
        ('VL Decode', num_tokens * t.vl_per_token, '#2ecc71'),
    ]

    # SD stages (showing one iteration cycle)
    # Prefill phase (parallel)
    parallel_prefill_time = max(t.vl_prefill, t.egpt_prefill + t.adapter_latency)

    # Decode phase per iteration
    if consec_5 > 0:
        iterations = num_tokens / (consec_5 + 1)
        time_per_iter = t.vl_per_token * (gamma + 1) + t.adapter_latency
        total_decode_time = iterations * time_per_iter
    else:
        total_decode_time = num_tokens * t.vl_per_token

    sd_stages = [
        ('EGPT Prefill', t.egpt_prefill, '#e74c3c'),
        ('Adapter', t.adapter_latency, '#9b59b6'),
        ('VL Prefill\n(parallel)', t.vl_prefill, '#3498db'),
        ('SD Decode', total_decode_time, '#f39c12'),
    ]

    # Plot baseline
    y_baseline = 1.5
    x_pos = 0
    for name, duration, color in baseline_stages:
        ax1.barh(y_baseline, duration, left=x_pos, height=0.4,
                color=color, edgecolor='black', alpha=0.8, label=name if x_pos == 0 else '')
        ax1.text(x_pos + duration/2, y_baseline, f'{name}\n{duration:.0f}ms',
                ha='center', va='center', fontsize=9, fontweight='bold')
        x_pos += duration
    baseline_total = x_pos

    # Plot SD (with parallel visualization)
    y_sd = 0.5

    # EGPT + Adapter (top row of parallel section)
    ax1.barh(y_sd + 0.25, t.egpt_prefill, left=0, height=0.2,
            color='#e74c3c', edgecolor='black', alpha=0.8)
    ax1.text(t.egpt_prefill/2, y_sd + 0.25, f'EGPT\n{t.egpt_prefill:.0f}ms',
            ha='center', va='center', fontsize=8)

    ax1.barh(y_sd + 0.25, t.adapter_latency, left=t.egpt_prefill, height=0.2,
            color='#9b59b6', edgecolor='black', alpha=0.8)

    # VL Prefill (bottom row, parallel)
    ax1.barh(y_sd - 0.05, t.vl_prefill, left=0, height=0.2,
            color='#3498db', edgecolor='black', alpha=0.8)
    ax1.text(t.vl_prefill/2, y_sd - 0.05, f'VL Prefill\n{t.vl_prefill:.0f}ms',
            ha='center', va='center', fontsize=8)

    # SD Decode (after parallel prefill)
    sd_decode_start = parallel_prefill_time
    ax1.barh(y_sd, total_decode_time, left=sd_decode_start, height=0.4,
            color='#f39c12', edgecolor='black', alpha=0.8)
    ax1.text(sd_decode_start + total_decode_time/2, y_sd,
            f'SD Decode\n{total_decode_time:.0f}ms\n({iterations:.1f} iters)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    sd_total = sd_decode_start + total_decode_time

    # Add parallel bracket
    ax1.annotate('', xy=(0, y_sd + 0.5), xytext=(parallel_prefill_time, y_sd + 0.5),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
    ax1.text(parallel_prefill_time/2, y_sd + 0.6, 'Parallel', ha='center', fontsize=9, color='gray')

    # Labels and formatting
    ax1.set_yticks([y_sd, y_baseline])
    ax1.set_yticklabels(['With SD', 'Baseline'], fontsize=11)
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_title(f'Per-Stage Time Breakdown (Baseline: {baseline_total:.0f}ms vs SD: {sd_total:.0f}ms)', fontsize=12)
    ax1.set_xlim(0, max(baseline_total, sd_total) * 1.1)
    ax1.grid(True, alpha=0.3, axis='x')

    # Add speedup annotation
    speedup = baseline_total / sd_total
    ax1.text(max(baseline_total, sd_total) * 0.95, 1.0,
            f'Speedup: {speedup:.2f}x', ha='right', va='center',
            fontsize=14, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='green'))

    # =========================================================================
    # Plot 2: Time ratio pie charts (Baseline vs SD)
    # =========================================================================
    ax2 = axes[1]
    ax2.axis('off')

    # Create two pie charts side by side
    ax2_left = fig.add_axes([0.1, 0.05, 0.35, 0.35])
    ax2_right = fig.add_axes([0.55, 0.05, 0.35, 0.35])

    # Baseline pie
    baseline_sizes = [t.vl_prefill, num_tokens * t.vl_per_token]
    baseline_labels = [f'VL Prefill\n{t.vl_prefill:.0f}ms\n({t.vl_prefill/baseline_total*100:.1f}%)',
                      f'VL Decode\n{num_tokens * t.vl_per_token:.0f}ms\n({num_tokens * t.vl_per_token/baseline_total*100:.1f}%)']
    baseline_colors = ['#3498db', '#2ecc71']
    ax2_left.pie(baseline_sizes, labels=baseline_labels, colors=baseline_colors,
                autopct='', startangle=90, explode=(0.02, 0.02))
    ax2_left.set_title(f'Baseline\nTotal: {baseline_total:.0f}ms', fontsize=11, fontweight='bold')

    # SD pie
    sd_sizes = [parallel_prefill_time, total_decode_time]
    sd_labels = [f'Prefill (parallel)\n{parallel_prefill_time:.0f}ms\n({parallel_prefill_time/sd_total*100:.1f}%)',
                f'SD Decode\n{total_decode_time:.0f}ms\n({total_decode_time/sd_total*100:.1f}%)']
    sd_colors = ['#3498db', '#f39c12']
    ax2_right.pie(sd_sizes, labels=sd_labels, colors=sd_colors,
                 autopct='', startangle=90, explode=(0.02, 0.02))
    ax2_right.set_title(f'With SD\nTotal: {sd_total:.0f}ms', fontsize=11, fontweight='bold')

    plt.savefig(output_dir / 'stage_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'stage_timeline.png'}")

    # Return stage breakdown for metrics
    return {
        'baseline_vl_prefill_ms': t.vl_prefill,
        'baseline_vl_decode_ms': num_tokens * t.vl_per_token,
        'baseline_total_ms': baseline_total,
        'sd_parallel_prefill_ms': parallel_prefill_time,
        'sd_decode_ms': total_decode_time,
        'sd_total_ms': sd_total,
        'sd_iterations': iterations if consec_5 > 0 else num_tokens,
        'time_ratio_prefill': parallel_prefill_time / sd_total,
        'time_ratio_decode': total_decode_time / sd_total,
    }


def plot_metrics(
    metrics: Dict,
    cos_sim_flat: torch.Tensor,
    position_stats: List[Dict],
    output_dir: Path,
):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cosine similarity histogram
    ax1 = axes[0, 0]
    ax1.hist(cos_sim_flat.cpu().numpy(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    for thresh, color in [(0.80, 'green'), (0.85, 'orange'), (0.90, 'red'), (0.95, 'purple')]:
        ax1.axvline(x=thresh, linestyle='--', color=color, label=f'τ={thresh}')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Hidden State Cosine Similarity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Per-position acceptance rates
    ax2 = axes[0, 1]
    positions = [ps['position'] for ps in position_stats[:20]]
    for thresh, color in [(80, 'green'), (85, 'orange'), (90, 'red'), (95, 'purple')]:
        rates = [ps.get(f'accept_{thresh}', 0) for ps in position_stats[:20]]
        ax2.plot(positions, rates, marker='o', label=f'@0.{thresh}', color=color, alpha=0.7)
    ax2.set_xlabel('Position in Sequence')
    ax2.set_ylabel('Acceptance Rate')
    ax2.set_title('Acceptance Rate by Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # 3. Consecutive accepts comparison
    ax3 = axes[1, 0]
    thresholds = [80, 85, 90, 95]
    means = [metrics.get(f'consecutive_mean_{t}', 0) for t in thresholds]
    stds = [metrics.get(f'consecutive_std_{t}', 0) for t in thresholds]
    x = range(len(thresholds))
    bars = ax3.bar(x, means, yerr=stds, capsize=5, color=['green', 'orange', 'red', 'purple'], alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'τ=0.{t}' for t in thresholds])
    ax3.set_ylabel('Consecutive Accepts (tokens)')
    ax3.set_title('Consecutive Accepts by Threshold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    # 4. Speedup summary
    ax4 = axes[1, 1]
    speedups = [
        ('Prefill', metrics['speedup_prefill']),
        ('Decode\n(γ=5)', metrics['speedup_decode_gamma5']),
        ('E2E', metrics['speedup_e2e']),
    ]
    names, values = zip(*speedups)
    colors = ['steelblue', 'coral', 'seagreen']
    bars = ax4.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=1.0, color='gray', linestyle='--', label='Baseline (1x)')
    ax4.set_ylabel('Speedup (x)')
    ax4.set_title('Estimated Speedup')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=150)
    plt.close()

    print(f"  Saved: {output_dir / 'metrics_summary.png'}")


def load_chunked_data(data_path: Path, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load data from chunked or single file format."""
    data_path = Path(data_path)

    if data_path.is_dir():
        # Chunked format — stream chunks one at a time to limit peak memory
        index_path = data_path / 'index.json'
        if not index_path.exists():
            raise ValueError(f"No index.json found in {data_path}")

        with open(index_path) as f:
            index = json.load(f)

        chunks_dir = data_path / 'chunks'
        all_egpt = []
        all_vl = []
        all_lens = []

        print(f"Loading {len(index['chunks'])} chunks (streaming)...")
        for chunk_info in tqdm(index['chunks']):
            chunk_path = chunks_dir / chunk_info['path']
            chunk = torch.load(chunk_path, map_location='cpu')
            all_egpt.append(chunk['egpt_hidden'].half())
            all_vl.append(chunk['vl_hidden'].half())
            all_lens.append(chunk['seq_lens'])
            del chunk

        egpt_hidden = torch.cat(all_egpt, dim=0)
        vl_hidden = torch.cat(all_vl, dim=0)
        seq_lens = torch.cat(all_lens, dim=0)
        del all_egpt, all_vl, all_lens

    else:
        # Single file format
        data = torch.load(data_path, map_location='cpu')
        egpt_hidden = data['egpt_hidden'].to(device).float()
        vl_hidden = data['vl_hidden'].to(device).float()
        seq_lens = data['seq_lens']

    # Create mask
    batch_size, max_seq = egpt_hidden.shape[:2]
    mask = torch.zeros(batch_size, max_seq, device=device)
    for i, seq_len in enumerate(seq_lens):
        mask[i, :seq_len] = 1

    return egpt_hidden, vl_hidden, mask


# =========================================================================
# TOKEN-LEVEL METRICS (requires LM head weights)
# =========================================================================

def load_lm_head(lm_head_path: str) -> torch.Tensor:
    """Load pre-extracted VL LM head weights."""
    data = torch.load(lm_head_path, map_location='cpu')
    weight = data['lm_head_weight']  # [vocab_size, hidden_dim]
    print(f"  LM head: [{weight.shape[0]}, {weight.shape[1]}] {weight.dtype}")
    return weight


def project_to_tokens(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    mask: torch.Tensor,
    top_k: int = 10,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project hidden states to token IDs via LM head (memory-efficient).

    Does chunked matmul + immediate argmax to avoid materialising [B, S, 32000].

    Args:
        hidden: [N, seq, 4096]
        lm_head_weight: [vocab_size, 4096]
        mask: [N, seq]
        top_k: number of top-k predictions to keep
        batch_size: chunk size for processing

    Returns:
        token_ids: [N, seq] greedy token IDs
        topk_ids:  [N, seq, k] top-k token IDs
    """
    N, S, D = hidden.shape
    # Use GPU for matmul even if data is on CPU (much faster)
    gpu = 'cuda' if torch.cuda.is_available() else hidden.device
    W = lm_head_weight.to(gpu).float()  # [V, D]

    token_ids = torch.zeros(N, S, dtype=torch.long)
    topk_ids = torch.zeros(N, S, top_k, dtype=torch.long)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        h_batch = hidden[start:end].to(gpu).float()  # [B, S, D]
        # logits = h @ W^T  →  [B, S, V]
        logits = torch.matmul(h_batch, W.T)
        token_ids[start:end] = logits.argmax(dim=-1).cpu()
        topk_ids[start:end] = logits.topk(top_k, dim=-1).indices.cpu()
        del logits, h_batch

    return token_ids, topk_ids


def compute_token_level_metrics(
    aligned_hidden: torch.Tensor,
    vl_hidden: torch.Tensor,
    mask: torch.Tensor,
    lm_head_weight: torch.Tensor,
    top_k: int = 10,
    batch_size: int = 64,
) -> Dict:
    """
    Compute token-level acceptance metrics via LM head projection.

    Returns dict with top-1/5/10 match rates, consecutive token accepts,
    and per-position token acceptance.
    """
    print("  Projecting aligned hidden states to tokens...")
    aligned_tokens, aligned_topk = project_to_tokens(
        aligned_hidden, lm_head_weight, mask, top_k=top_k, batch_size=batch_size,
    )
    print("  Projecting VL hidden states to tokens...")
    vl_tokens, _ = project_to_tokens(
        vl_hidden, lm_head_weight, mask, top_k=1, batch_size=batch_size,
    )

    mask_bool = mask.bool()
    valid_count = mask.sum().item()

    metrics = {}

    # --- Top-1 match ---
    top1_match = (aligned_tokens == vl_tokens) & mask_bool
    metrics['token_top1_match'] = top1_match.float().sum().item() / valid_count

    # --- Top-5 match ---
    vl_expanded = vl_tokens.unsqueeze(-1)  # [N, S, 1]
    top5_hit = (aligned_topk[:, :, :5] == vl_expanded).any(dim=-1) & mask_bool
    metrics['token_top5_match'] = top5_hit.float().sum().item() / valid_count

    # --- Top-10 match ---
    top10_hit = (aligned_topk == vl_expanded).any(dim=-1) & mask_bool
    metrics['token_top10_match'] = top10_hit.float().sum().item() / valid_count

    # --- Consecutive token accepts (cumprod trick) ---
    accept_int = top1_match.int()
    cumprod = accept_int.cumprod(dim=1)
    consecutive = cumprod.sum(dim=1).float()  # [N]
    seq_lens = mask.sum(dim=1)
    consecutive = torch.minimum(consecutive, seq_lens)

    metrics['token_consecutive_mean'] = consecutive.mean().item()
    metrics['token_consecutive_std'] = consecutive.std().item()
    metrics['token_consecutive_median'] = consecutive.median().item()
    metrics['token_consecutive_max'] = consecutive.max().item()
    metrics['token_consecutive_min'] = consecutive.min().item()

    # --- Per-position token acceptance (first 20 positions) ---
    max_seq = aligned_tokens.shape[1]
    position_token_accept = []
    for pos in range(min(max_seq, 20)):
        valid_at_pos = mask_bool[:, pos]
        if valid_at_pos.sum() == 0:
            break
        match_at_pos = top1_match[valid_at_pos, pos].float().mean().item()
        position_token_accept.append(match_at_pos)

    metrics['position_token_accept'] = position_token_accept

    return metrics, aligned_tokens, vl_tokens


def compute_two_phase_sd_metrics(
    aligned_tokens: torch.Tensor,
    vl_tokens: torch.Tensor,
    mask: torch.Tensor,
    timing_config: TimingConfig,
    gamma_decode: int = 5,
) -> Dict:
    """
    Compute two-phase speculative decoding metrics using token-level matches.

    Phase 1 (Prefill Hiding): gamma_prefill = (VL_prefill - EGPT_prefill) / EGPT_per_token
      Count consecutive token matches in first gamma_prefill positions.
    Phase 2 (Decode): gamma_decode=5
      Per-iteration: draft gamma tokens, accept consecutive prefix.
    """
    t = timing_config
    mask_bool = mask.bool()

    # gamma_prefill: how many free draft tokens during VL prefill overlap
    free_draft_time = max(0, t.vl_prefill - t.egpt_prefill)
    gamma_prefill = int(free_draft_time / t.egpt_per_token) if t.egpt_per_token > 0 else 0

    metrics = {'token_sd_gamma_prefill': gamma_prefill}

    # --- Phase 1: Prefill hiding ---
    if gamma_prefill > 0:
        first_gp = min(gamma_prefill, aligned_tokens.shape[1])
        match_prefill = (aligned_tokens[:, :first_gp] == vl_tokens[:, :first_gp])
        match_prefill = match_prefill & mask_bool[:, :first_gp]
        cumprod_prefill = match_prefill.int().cumprod(dim=1)
        consec_prefill = cumprod_prefill.sum(dim=1).float()
        metrics['token_sd_prefill_consecutive_mean'] = consec_prefill.mean().item()
        metrics['token_sd_prefill_accept_rate'] = consec_prefill.mean().item() / first_gp
    else:
        metrics['token_sd_prefill_consecutive_mean'] = 0.0
        metrics['token_sd_prefill_accept_rate'] = 0.0

    # --- Phase 2: Decode (gamma=gamma_decode) ---
    # Look at first gamma_decode positions for expected accept per iteration
    max_seq = aligned_tokens.shape[1]
    if max_seq >= gamma_decode:
        first_g = gamma_decode
        match_decode = (aligned_tokens[:, :first_g] == vl_tokens[:, :first_g])
        match_decode = match_decode & mask_bool[:, :first_g]
        cumprod_decode = match_decode.int().cumprod(dim=1)
        consec_decode = cumprod_decode.sum(dim=1).float()
        avg_accepted = consec_decode.mean().item()
    else:
        avg_accepted = 0.0

    metrics['token_sd_decode_consecutive_mean'] = avg_accepted
    metrics['token_sd_decode_accept_rate'] = avg_accepted / gamma_decode if gamma_decode > 0 else 0.0
    metrics['token_sd_gamma_decode'] = gamma_decode

    # --- E2E speedup estimate from token-level rates ---
    num_output_tokens = 50
    baseline_time = t.vl_prefill + num_output_tokens * t.vl_per_token

    # Prefill phase (parallel)
    parallel_prefill = max(t.vl_prefill, t.egpt_prefill + t.adapter_latency)
    prefill_saved = metrics.get('token_sd_prefill_consecutive_mean', 0) * t.vl_per_token

    # Decode phase with SD
    if avg_accepted > 0:
        iterations = num_output_tokens / (avg_accepted + 1)
        decode_time = iterations * (t.vl_per_token * (gamma_decode + 1) + t.adapter_latency)
    else:
        decode_time = num_output_tokens * t.vl_per_token

    total_sd_time = parallel_prefill + decode_time
    metrics['token_sd_e2e_speedup'] = baseline_time / total_sd_time if total_sd_time > 0 else 1.0
    metrics['token_sd_e2e_baseline_ms'] = baseline_time
    metrics['token_sd_e2e_sd_ms'] = total_sd_time

    return metrics


def compute_cosim_vs_token_correlation(
    cos_sim: torch.Tensor,
    token_match: torch.Tensor,
    mask: torch.Tensor,
) -> Dict:
    """
    Sweep cos_sim thresholds and measure what fraction also match at token level.

    Answers: "is cos_sim > 0.90 a good proxy for token acceptance?"
    """
    mask_bool = mask.bool()
    results = {}

    for thresh in [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]:
        above = (cos_sim > thresh) & mask_bool
        n_above = above.sum().item()
        if n_above > 0:
            token_match_given_above = (token_match & above).sum().item() / n_above
        else:
            token_match_given_above = 0.0
        results[f'cosim_{int(thresh*100)}_token_match'] = token_match_given_above
        results[f'cosim_{int(thresh*100)}_count'] = int(n_above)

    # Overall correlation (Pearson on valid positions)
    valid = mask_bool
    cs_flat = cos_sim[valid].float().cpu()
    tm_flat = token_match[valid].float().cpu()
    if len(cs_flat) > 1:
        corr = torch.corrcoef(torch.stack([cs_flat, tm_flat]))[0, 1].item()
    else:
        corr = 0.0
    results['cosim_token_pearson_r'] = corr

    return results


def print_token_metrics_report(metrics: Dict):
    """Print formatted token-level metrics report section."""
    print("\n" + "=" * 80)
    print("TOKEN-LEVEL METRICS (via LM head projection)")
    print("=" * 80)

    print("\n[T1] TOKEN MATCH RATES")
    print("-" * 40)
    for k, label in [('token_top1_match', 'Top-1'), ('token_top5_match', 'Top-5'),
                     ('token_top10_match', 'Top-10')]:
        rate = metrics.get(k, 0)
        bar = "\u2588" * int(rate * 20)
        print(f"  {label:>6}: {rate:6.2%} {bar}")

    print("\n[T2] CONSECUTIVE TOKEN ACCEPTS")
    print("-" * 40)
    print(f"  Mean:    {metrics.get('token_consecutive_mean', 0):.2f} tokens")
    print(f"  Std:     {metrics.get('token_consecutive_std', 0):.2f}")
    print(f"  Median:  {metrics.get('token_consecutive_median', 0):.2f}")
    print(f"  Max:     {metrics.get('token_consecutive_max', 0):.0f}")

    print("\n[T3] TWO-PHASE SD (Token-Level)")
    print("-" * 40)
    gp = metrics.get('token_sd_gamma_prefill', 0)
    gd = metrics.get('token_sd_gamma_decode', 5)
    print(f"  Phase 1 - Prefill Hiding (\u03b3_prefill={gp}):")
    print(f"    Consecutive accepts: {metrics.get('token_sd_prefill_consecutive_mean', 0):.2f}")
    print(f"    Accept rate:         {metrics.get('token_sd_prefill_accept_rate', 0):.2%}")
    print(f"  Phase 2 - Decode (\u03b3_decode={gd}):")
    print(f"    Consecutive accepts: {metrics.get('token_sd_decode_consecutive_mean', 0):.2f}")
    print(f"    Accept rate:         {metrics.get('token_sd_decode_accept_rate', 0):.2%}")
    print(f"  E2E Speedup Estimate:  {metrics.get('token_sd_e2e_speedup', 1.0):.2f}x")

    print("\n[T4] COSINE SIM vs TOKEN MATCH CORRELATION")
    print("-" * 40)
    for thresh in [80, 85, 90, 92, 95, 98]:
        k = f'cosim_{thresh}_token_match'
        rate = metrics.get(k, 0)
        count = metrics.get(f'cosim_{thresh}_count', 0)
        t_str = f"0.{thresh}" if thresh < 100 else f"1.{thresh-100:02d}"
        print(f"  cos>{t_str}: {rate:6.2%} also token-match  (n={count})")
    print(f"  Pearson r: {metrics.get('cosim_token_pearson_r', 0):.4f}")

    print("\n[T5] PER-POSITION TOKEN ACCEPT (First 10)")
    print("-" * 40)
    for i, rate in enumerate(metrics.get('position_token_accept', [])[:10]):
        bar = "\u2588" * int(rate * 20)
        print(f"  Pos {i:2d}: {rate:6.2%} {bar}")


def generate_metrics_comparison_md(metrics: Dict, output_dir: Path):
    """Auto-generate METRICS_COMPARISON.md comparing hidden-state vs token-level."""
    md_path = output_dir / 'METRICS_COMPARISON.md'

    lines = [
        "# Metrics Comparison: Hidden-State vs Token-Level",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Metric | Hidden-State (@0.90) | Token-Level (Top-1) |",
        "|--------|---------------------|---------------------|",
    ]

    hs_accept = metrics.get('accept_90', 0)
    tk_accept = metrics.get('token_top1_match', 0)
    lines.append(f"| Overall Accept Rate | {hs_accept:.2%} | {tk_accept:.2%} |")

    hs_consec = metrics.get('consecutive_mean_90', 0)
    tk_consec = metrics.get('token_consecutive_mean', 0)
    lines.append(f"| Consecutive Accepts | {hs_consec:.2f} | {tk_consec:.2f} |")

    hs_e2e = metrics.get('speedup_e2e', 1.0)
    tk_e2e = metrics.get('token_sd_e2e_speedup', 1.0)
    lines.append(f"| E2E Speedup | {hs_e2e:.2f}x | {tk_e2e:.2f}x |")

    lines.extend([
        "",
        "## Cosine Similarity as Token Proxy",
        "",
        "| cos_sim threshold | P(token match | cos > thresh) | Count |",
        "|-------------------|-------------------------------|-------|",
    ])
    for thresh in [80, 85, 90, 92, 95, 98]:
        rate = metrics.get(f'cosim_{thresh}_token_match', 0)
        count = metrics.get(f'cosim_{thresh}_count', 0)
        t_str = f"0.{thresh}" if thresh < 100 else f"1.{thresh-100:02d}"
        lines.append(f"| > {t_str} | {rate:.2%} | {count} |")

    r = metrics.get('cosim_token_pearson_r', 0)
    lines.extend([
        "",
        f"Pearson r (cos_sim, token_match): **{r:.4f}**",
        "",
        "## Per-Position Comparison (First 10)",
        "",
        "| Position | Hidden Accept @0.90 | Token Top-1 |",
        "|----------|--------------------:|------------:|",
    ])
    hs_pos = metrics.get('position_accept_90', [])
    tk_pos = metrics.get('position_token_accept', [])
    for i in range(min(10, len(hs_pos), len(tk_pos))):
        lines.append(f"| {i} | {hs_pos[i]:.2%} | {tk_pos[i]:.2%} |")

    lines.extend([
        "",
        "## Two-Phase SD (Token-Level)",
        "",
        f"- gamma_prefill = {metrics.get('token_sd_gamma_prefill', 0)}",
        f"- Prefill consecutive accepts: {metrics.get('token_sd_prefill_consecutive_mean', 0):.2f}",
        f"- gamma_decode = {metrics.get('token_sd_gamma_decode', 5)}",
        f"- Decode consecutive accepts: {metrics.get('token_sd_decode_consecutive_mean', 0):.2f}",
        f"- E2E speedup estimate: {metrics.get('token_sd_e2e_speedup', 1.0):.2f}x",
        "",
    ])

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {md_path}")


def plot_token_metrics(
    metrics: Dict,
    output_dir: Path,
):
    """Generate token-level metrics plot panel."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Token match rates bar chart
    ax1 = axes[0]
    labels = ['Top-1', 'Top-5', 'Top-10']
    values = [metrics.get('token_top1_match', 0),
              metrics.get('token_top5_match', 0),
              metrics.get('token_top10_match', 0)]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax1.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Match Rate')
    ax1.set_title('Token Match Rates (via LM Head)')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Per-position token accept
    ax2 = axes[1]
    tk_pos = metrics.get('position_token_accept', [])
    hs_pos = metrics.get('position_accept_90', [])
    n_pos = min(len(tk_pos), len(hs_pos), 20)
    if n_pos > 0:
        positions = list(range(n_pos))
        ax2.plot(positions, tk_pos[:n_pos], 'o-', color='#e74c3c', label='Token Top-1', alpha=0.8)
        ax2.plot(positions, hs_pos[:n_pos], 's--', color='#3498db', label='Hidden @0.90', alpha=0.8)
        ax2.set_xlabel('Position in Sequence')
        ax2.set_ylabel('Acceptance Rate')
        ax2.set_title('Per-Position: Hidden vs Token Accept')
        ax2.legend()
        ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # 3. Cosim threshold → token match
    ax3 = axes[2]
    thresholds = [80, 85, 90, 92, 95, 98]
    cos_token_rates = [metrics.get(f'cosim_{t}_token_match', 0) for t in thresholds]
    thresh_labels = [f'.{t}' for t in thresholds]
    ax3.plot(thresh_labels, cos_token_rates, 'D-', color='#9b59b6', markersize=8, alpha=0.8)
    for i, (lbl, val) in enumerate(zip(thresh_labels, cos_token_rates)):
        ax3.annotate(f'{val:.0%}', (lbl, val), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)
    ax3.set_xlabel('Cosine Similarity Threshold')
    ax3.set_ylabel('P(token match | cos > thresh)')
    ax3.set_title('Cosine Sim as Token Match Proxy')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'token_metrics.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'token_metrics.png'}")


def _serialize_metrics(metrics: Dict) -> Dict:
    """Convert metrics dict to JSON-serializable form."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (list, tuple)):
            out[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]
        else:
            out[k] = v
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Measure feature-level acceptance metrics")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained hidden adapter checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to hidden states test data (file or chunked dir)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing')

    # Token-level metrics (optional)
    parser.add_argument('--lm_head', type=str, default=None,
                        help='Path to vl_lm_head.pt (enables token-level metrics)')

    # Timing configuration
    parser.add_argument('--egpt_prefill_ms', type=float, default=130.0)
    parser.add_argument('--egpt_per_token_ms', type=float, default=25.0)
    parser.add_argument('--vl_prefill_ms', type=float, default=310.0)
    parser.add_argument('--vl_per_token_ms', type=float, default=25.0)
    parser.add_argument('--adapter_latency_ms', type=float, default=1.5)

    # SD configuration
    parser.add_argument('--gamma_decode', type=int, default=5)

    # Adapter mode overrides
    parser.add_argument('--vlm_only', action='store_true',
                        help='VLM-only mode (B1): use vl_hidden as adapter input')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, checkpoint = load_any_adapter(args.checkpoint, device)
    model = model.to(device)
    model.eval()
    adapter_class = type(model).__name__
    print(f"  Adapter type: {adapter_class}")
    print(f"  Parameters: {model.get_num_parameters():,}")

    # Detect adapter mode from checkpoint config or CLI flag
    is_vlm_only = args.vlm_only
    is_fused = adapter_class == 'FusedEAGLEAdapter'

    if is_vlm_only:
        print(f"  Mode: VLM-only (B1) — using vl_hidden as input")
    elif is_fused:
        print(f"  Mode: Fused (L5F) — using both egpt_hidden + vl_hidden as input")
    else:
        print(f"  Mode: Standard — using egpt_hidden as input")

    # Load test data to CPU (avoid GPU OOM with large test sets)
    print(f"\nLoading test data: {args.test_data}")
    egpt_hidden, vl_hidden, mask = load_chunked_data(args.test_data, 'cpu')
    print(f"  Samples: {len(egpt_hidden)}")
    print(f"  Max seq len: {egpt_hidden.shape[1]}")
    print(f"  Hidden dim: {egpt_hidden.shape[-1]}")

    # Optionally load LM head for token-level metrics
    lm_head_weight = None
    if args.lm_head:
        print(f"\nLoading LM head: {args.lm_head}")
        lm_head_weight = load_lm_head(args.lm_head)

    # Create configs
    timing_config = TimingConfig(
        egpt_prefill=args.egpt_prefill_ms,
        egpt_per_token=args.egpt_per_token_ms,
        vl_prefill=args.vl_prefill_ms,
        vl_per_token=args.vl_per_token_ms,
        adapter_latency=args.adapter_latency_ms,
    )

    sd_config = SDConfig(
        gamma_decode=args.gamma_decode,
    )

    # Compute aligned hidden states batch-by-batch (data on CPU, compute on GPU)
    # B1 (vlm_only): source = vl_hidden
    # L5F (fused): forward(egpt_hidden, vl_hidden)
    # L1-L5: source = egpt_hidden
    print("\nComputing aligned hidden states (batch-by-batch on GPU)...")
    aligned_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(egpt_hidden), args.batch_size)):
            if is_fused:
                batch_egpt = egpt_hidden[i:i+args.batch_size].to(device).float()
                batch_vl = vl_hidden[i:i+args.batch_size].to(device).float()
                out = model(batch_egpt, batch_vl).cpu().half()
            elif is_vlm_only:
                batch_vl = vl_hidden[i:i+args.batch_size].to(device).float()
                out = model(batch_vl).cpu().half()
            else:
                batch = egpt_hidden[i:i+args.batch_size].to(device).float()
                out = model(batch).cpu().half()
            aligned_list.append(out)
    aligned_hidden = torch.cat(aligned_list, dim=0).float()
    del aligned_list

    # Free egpt_hidden — no longer needed after alignment
    del egpt_hidden
    import gc; gc.collect()

    # Cast vl_hidden to float32 for metrics
    vl_hidden = vl_hidden.float()

    # =====================================================================
    # HIDDEN-STATE METRICS
    # =====================================================================

    print("\n[HIDDEN-STATE METRICS]")
    print("Computing metrics (parallel)...")
    metrics, cos_sim, valid_sims = compute_all_metrics_parallel(
        aligned_hidden, vl_hidden, mask,
        sd_config=sd_config,
        timing_config=timing_config,
    )

    # Compute per-position stats
    position_stats = compute_per_position_stats(cos_sim, mask)

    # Print hidden-state report
    print_metrics_report(metrics, position_stats)

    # =====================================================================
    # TOKEN-LEVEL METRICS (if LM head provided)
    # =====================================================================
    if lm_head_weight is not None:
        print("\n[TOKEN-LEVEL METRICS]")
        print("Computing token-level metrics via LM head projection...")

        with torch.no_grad():
            token_metrics, aligned_tokens, vl_tokens = compute_token_level_metrics(
                aligned_hidden, vl_hidden, mask, lm_head_weight,
                top_k=10, batch_size=args.batch_size,
            )
        metrics.update(token_metrics)

        # Two-phase SD metrics
        sd_token_metrics = compute_two_phase_sd_metrics(
            aligned_tokens, vl_tokens, mask, timing_config,
            gamma_decode=args.gamma_decode,
        )
        metrics.update(sd_token_metrics)

        # Cosine sim vs token match correlation
        top1_match = (aligned_tokens == vl_tokens) & mask.bool()
        corr_metrics = compute_cosim_vs_token_correlation(cos_sim, top1_match, mask)
        metrics.update(corr_metrics)

        # Print token-level report
        print_token_metrics_report(metrics)

        # Sanity check: VL hidden → tokens should be self-consistent
        print("\n[SANITY CHECK] VL self-consistency:")
        vl_self_tokens, _ = project_to_tokens(
            vl_hidden, lm_head_weight, mask, top_k=1, batch_size=args.batch_size,
        )
        self_match = ((vl_self_tokens == vl_tokens) & mask.bool()).float().sum() / mask.sum()
        print(f"  VL→tokens self-match: {self_match.item():.4f} (should be 1.0000)")

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = Path(__file__).parent / 'tasks'

    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_file = output_path / 'acceptance_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(_serialize_metrics(metrics), f, indent=2)
    print(f"\nSaved metrics: {metrics_file}")

    # Save position stats
    pos_file = output_path / 'position_stats.json'
    with open(pos_file, 'w') as f:
        json.dump(position_stats, f, indent=2)
    print(f"Saved position stats: {pos_file}")

    # Generate plots
    print("\nGenerating plots...")
    plot_metrics(metrics, valid_sims, position_stats, output_path)

    # Generate stage timeline visualization
    stage_metrics = plot_stage_timeline(metrics, timing_config, output_path)
    metrics.update(stage_metrics)

    # Token-level plots and comparison MD
    if lm_head_weight is not None:
        plot_token_metrics(metrics, output_path)
        generate_metrics_comparison_md(metrics, output_path)

    # Save final metrics with all sections
    with open(metrics_file, 'w') as f:
        json.dump(_serialize_metrics(metrics), f, indent=2)

    print(f"\nAll results saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    main()
