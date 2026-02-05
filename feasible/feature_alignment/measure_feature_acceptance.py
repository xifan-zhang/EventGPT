#!/usr/bin/env python3
"""
Measure Feature-Level Acceptance Rate for Speculative Decoding.

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
        # Chunked format
        index_path = data_path / 'index.json'
        if not index_path.exists():
            raise ValueError(f"No index.json found in {data_path}")

        with open(index_path) as f:
            index = json.load(f)

        chunks_dir = data_path / 'chunks'
        all_egpt = []
        all_vl = []
        all_lens = []

        print(f"Loading {len(index['chunks'])} chunks...")
        for chunk_info in tqdm(index['chunks']):
            chunk_path = chunks_dir / chunk_info['path']
            chunk = torch.load(chunk_path, map_location='cpu')
            all_egpt.append(chunk['egpt_hidden'])
            all_vl.append(chunk['vl_hidden'])
            all_lens.append(chunk['seq_lens'])

        egpt_hidden = torch.cat(all_egpt, dim=0).to(device).float()
        vl_hidden = torch.cat(all_vl, dim=0).to(device).float()
        seq_lens = torch.cat(all_lens, dim=0)

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

    # Timing configuration
    parser.add_argument('--egpt_prefill_ms', type=float, default=130.0)
    parser.add_argument('--egpt_per_token_ms', type=float, default=25.0)
    parser.add_argument('--vl_prefill_ms', type=float, default=310.0)
    parser.add_argument('--vl_per_token_ms', type=float, default=25.0)
    parser.add_argument('--adapter_latency_ms', type=float, default=1.5)

    # SD configuration
    parser.add_argument('--gamma_decode', type=int, default=5)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, checkpoint = load_any_adapter(args.checkpoint, device)
    model = model.to(device)
    model.eval()
    print(f"  Adapter type: {type(model).__name__}")
    print(f"  Parameters: {model.get_num_parameters():,}")

    # Load test data
    print(f"\nLoading test data: {args.test_data}")
    egpt_hidden, vl_hidden, mask = load_chunked_data(args.test_data, device)
    print(f"  Samples: {len(egpt_hidden)}")
    print(f"  Max seq len: {egpt_hidden.shape[1]}")
    print(f"  Hidden dim: {egpt_hidden.shape[-1]}")

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

    # Compute aligned hidden states
    print("\nComputing aligned hidden states...")
    with torch.no_grad():
        # Process in batches if needed
        if len(egpt_hidden) > args.batch_size:
            aligned_list = []
            for i in tqdm(range(0, len(egpt_hidden), args.batch_size)):
                batch = egpt_hidden[i:i+args.batch_size]
                aligned_list.append(model(batch))
            aligned_hidden = torch.cat(aligned_list, dim=0)
        else:
            aligned_hidden = model(egpt_hidden)

    # Compute all metrics in parallel
    print("Computing metrics (parallel)...")
    metrics, cos_sim, valid_sims = compute_all_metrics_parallel(
        aligned_hidden, vl_hidden, mask,
        sd_config=sd_config,
        timing_config=timing_config,
    )

    # Compute per-position stats
    position_stats = compute_per_position_stats(cos_sim, mask)

    # Print report
    print_metrics_report(metrics, position_stats)

    # Save results
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = Path(args.checkpoint).parent

    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_file = output_path / 'acceptance_metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert any remaining tensors to Python types
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, (list, tuple)):
                metrics_serializable[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]
            else:
                metrics_serializable[k] = v
        json.dump(metrics_serializable, f, indent=2)
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

    # Save updated metrics with stage breakdown
    with open(metrics_file, 'w') as f:
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, (list, tuple)):
                metrics_serializable[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]
            else:
                metrics_serializable[k] = v
        json.dump(metrics_serializable, f, indent=2)

    print(f"\nAll results saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    main()
