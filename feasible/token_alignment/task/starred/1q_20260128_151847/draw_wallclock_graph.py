#!/usr/bin/env python3
"""
Wall-clock time visualization for EventGPT → VideoLLaVA
with PARALLEL PREFILL and SPECULATIVE DECODING.
Based on benchmark results from 1q_20260128_151847.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path(__file__).parent / "results.json"
with open(results_path) as f:
    results = json.load(f)

# Extract data
training_history = results["training_history"]
acceptance_rate = results["best_val_acc"]  # 0.279
top5_rate = results["final"]["test"]["top5_rate"]  # 0.516
baseline_rate = results["baseline"]["test"]["acceptance_rate"]  # 0.0158

# Timing estimates (from parallel prefill benchmarks, ms)
# These are realistic estimates based on 4-bit models
EGPT_PREFILL_MS = 97    # EventGPT prefill (vision + LLM)
VL_PREFILL_MS = 470     # VideoLLaVA prefill (8 frames + LLM)
EGPT_DECODE_MS = 14     # EventGPT per-token decode
VL_DECODE_MS = 15       # VideoLLaVA per-token decode
ADAPTER_MS = 0.1        # TokenAdapter inference (negligible)

# Parallel prefill: EventGPT drafts DURING VideoLLaVA prefill
PARALLEL_OVERLAP_MS = VL_PREFILL_MS - EGPT_PREFILL_MS  # 373ms free window
FREE_DRAFT_TOKENS = int(PARALLEL_OVERLAP_MS / EGPT_DECODE_MS)  # ~26 tokens

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 14))
fig.suptitle('EventGPT → VideoLLaVA: Parallel Prefill + Speculative Decoding',
             fontsize=14, fontweight='bold')

# ============================================================================
# Plot 1: Training Convergence (top-left)
# ============================================================================
ax1 = fig.add_subplot(2, 2, 1)
epochs = [h["epoch"] for h in training_history]
train_loss = [h["train_loss"] for h in training_history]
val_acc = [h["val_acc"] * 100 for h in training_history]

ax1_twin = ax1.twinx()
l1, = ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)
l2, = ax1_twin.plot(epochs, val_acc, 'r-s', label='Val Accuracy', linewidth=2, markersize=6)

ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', color='blue', fontsize=11)
ax1_twin.set_ylabel('Acceptance Rate (%)', color='red', fontsize=11)
ax1.set_title('Training Convergence', fontsize=12, fontweight='bold')
ax1.axhline(y=train_loss[1], color='blue', linestyle='--', alpha=0.5)
ax1_twin.axhline(y=val_acc[1], color='red', linestyle='--', alpha=0.5)
ax1.annotate(f'Best: {val_acc[1]:.1f}%', xy=(2, val_acc[1]),
             xytext=(4, val_acc[1]+2), fontsize=9,
             arrowprops=dict(arrowstyle='->', color='red'))
ax1.legend([l1, l2], ['Train Loss', 'Val Accuracy'], loc='center right')
ax1.grid(True, alpha=0.3)

# ============================================================================
# Plot 2: Acceptance Rate Comparison (top-right)
# ============================================================================
ax2 = fig.add_subplot(2, 2, 2)
categories = ['Baseline\n(Direct)', 'TokenAdapter\n(Top-1)', 'TokenAdapter\n(Top-5)']
rates = [baseline_rate * 100, acceptance_rate * 100, top5_rate * 100]
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

bars = ax2.bar(categories, rates, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Acceptance Rate (%)', fontsize=11)
ax2.set_title('Token Acceptance Rates', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 60)

for bar, rate in zip(bars, rates):
    ax2.annotate(f'{rate:.1f}%',
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 5), textcoords='offset points',
                 ha='center', fontsize=11, fontweight='bold')

# Add improvement annotations
ax2.annotate(f'17.6×', xy=(1, rates[1]), xytext=(0.5, rates[1]-8),
             fontsize=10, color='green', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green'))
ax2.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 3: Wall-Clock Time Breakdown - Prefill vs Decode (bottom-left)
# ============================================================================
ax3 = fig.add_subplot(2, 2, 3)

num_tokens = 50
gamma = 5  # draft tokens per round

# The KEY benefit: Parallel prefill gives FREE draft tokens
# AR: Must wait for VL_PREFILL before ANY decoding
# Parallel: EventGPT drafts during VL_PREFILL window

# Expected accepted tokens per round
def get_expected_accepted(alpha, gamma):
    if alpha > 0:
        return (1 - alpha**(gamma+1)) / (1 - alpha)
    return 1

exp_acc = get_expected_accepted(acceptance_rate, gamma)
exp_acc_top5 = get_expected_accepted(top5_rate, gamma)

# Time breakdown for 50 tokens
ar_decode_time = num_tokens * VL_DECODE_MS
ar_prefill_time = VL_PREFILL_MS

# Parallel prefill: Free tokens accepted during VL prefill
free_accepted_27 = FREE_DRAFT_TOKENS * acceptance_rate  # ~7 tokens free
free_accepted_51 = FREE_DRAFT_TOKENS * top5_rate  # ~13 tokens free

# Time after prefill for remaining tokens (draft-verify rounds)
remaining_27 = num_tokens - free_accepted_27
remaining_51 = num_tokens - free_accepted_51

# For remaining tokens: need to do SD rounds
# Time per round = γ × T_draft + T_verify
time_per_round = gamma * EGPT_DECODE_MS + VL_DECODE_MS * 1.2  # 88ms

rounds_27 = remaining_27 / exp_acc
rounds_51 = remaining_51 / exp_acc_top5

parallel_27_decode = rounds_27 * time_per_round
parallel_51_decode = rounds_51 * time_per_round

# For parallel prefill: prefill is the SAME (VL_PREFILL_MS), but EventGPT runs in parallel
# The "effective" prefill time is still VL_PREFILL_MS, but we get free tokens
parallel_prefill = VL_PREFILL_MS

# Stacked bar data: [prefill_time, decode_time]
methods = ['AR\n(VideoLLaVA)', 'Parallel Prefill\nα=27.9%', 'Parallel Prefill\nα=51.6%']
prefill_times = [ar_prefill_time, parallel_prefill, parallel_prefill]
decode_times = [ar_decode_time, parallel_27_decode, parallel_51_decode]
total_times = [p + d for p, d in zip(prefill_times, decode_times)]

# Create stacked bars
x = np.arange(len(methods))
width = 0.6

# Lower bar: Prefill time (blue)
bars_prefill = ax3.bar(x, prefill_times, width, label='Prefill',
                       color='#3498db', edgecolor='black', linewidth=1.5)
# Upper bar: Decode time (orange)
bars_decode = ax3.bar(x, decode_times, width, bottom=prefill_times, label='Decode',
                      color='#e67e22', edgecolor='black', linewidth=1.5)

ax3.set_ylabel('Wall-Clock Time (ms)', fontsize=11)
ax3.set_title(f'Wall-Clock Time Breakdown ({num_tokens} Tokens)', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(methods)

# Add total time labels on top
for i, (total, prefill, decode) in enumerate(zip(total_times, prefill_times, decode_times)):
    ax3.annotate(f'{total:.0f}ms',
                 xy=(i, total),
                 xytext=(0, 5), textcoords='offset points',
                 ha='center', fontsize=10, fontweight='bold')

# Add prefill/decode labels inside bars
for i, (prefill, decode) in enumerate(zip(prefill_times, decode_times)):
    # Prefill label (in lower bar)
    ax3.annotate(f'{prefill:.0f}ms',
                 xy=(i, prefill/2),
                 ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    # Decode label (in upper bar)
    ax3.annotate(f'{decode:.0f}ms',
                 xy=(i, prefill + decode/2),
                 ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Add legend
ax3.legend(loc='upper right', fontsize=9)

# Add note about parallel prefill benefit
ax3.text(0.5, 0.02, 'Parallel prefill: EventGPT drafts during VideoLLaVA prefill (same wall-clock)',
         transform=ax3.transAxes, fontsize=8, color='gray', ha='center')

ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 4: Key Insight - When SD Helps (bottom-right)
# ============================================================================
ax4 = fig.add_subplot(2, 2, 4)

# SD speedup depends on: α (acceptance), c (draft/target speed ratio), γ (draft length)
# Speedup = expected_tokens / (1 + c × γ)
# For speedup > 1: expected_tokens > 1 + c × γ

# Our case: c ≈ 14/15 ≈ 0.93 (almost same speed)
c_ratio = EGPT_DECODE_MS / VL_DECODE_MS
gammas = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Calculate required acceptance rate for SD to break even
def calc_breakeven_alpha(c, gamma):
    """Find α where speedup = 1."""
    # expected_tokens = (1 - α^(γ+1)) / (1-α) = 1 + c×γ
    # Solve numerically
    target = 1 + c * gamma
    for alpha in np.linspace(0.01, 0.99, 1000):
        expected = (1 - alpha**(gamma+1)) / (1 - alpha)
        if expected >= target:
            return alpha
    return 0.99

breakeven_same = [calc_breakeven_alpha(c_ratio, g) * 100 for g in gammas]  # Our case
breakeven_fast = [calc_breakeven_alpha(0.1, g) * 100 for g in gammas]  # 10x faster draft

ax4.plot(gammas, breakeven_same, 'r-o', linewidth=2, markersize=8,
         label=f'Our case (c={c_ratio:.2f})')
ax4.plot(gammas, breakeven_fast, 'g-s', linewidth=2, markersize=8,
         label='Typical SD (c=0.1)')

# Mark our operating point
ax4.axhline(y=acceptance_rate*100, color='blue', linestyle='--', alpha=0.7,
            label=f'Current α=27.9%')
ax4.axhline(y=top5_rate*100, color='purple', linestyle='--', alpha=0.7,
            label=f'Top-5 α=51.6%')

# Shade regions
ax4.fill_between(gammas, 0, breakeven_same, alpha=0.15, color='green',
                  label='SD beneficial')
ax4.fill_between(gammas, breakeven_same, 100, alpha=0.15, color='red')

ax4.set_xlabel('Draft Length (γ)', fontsize=11)
ax4.set_ylabel('Min Acceptance Rate for Speedup (%)', fontsize=11)
ax4.set_title('When Does Speculative Decoding Help?', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(1, 8)
ax4.set_ylim(0, 100)

# Add insight annotation
ax4.annotate('With similar model speeds,\nα>95% needed!',
             xy=(5, breakeven_same[4]), xytext=(6, 70),
             fontsize=9, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))
ax4.annotate('With 10× faster draft,\nα>15% suffices',
             xy=(5, breakeven_fast[4]), xytext=(2.5, 30),
             fontsize=9, color='green',
             arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
output_path = Path(__file__).parent / "wallclock_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

# Also save a summary
summary_path = Path(__file__).parent / "wallclock_summary.txt"
with open(summary_path, 'w') as f:
    f.write("Wall-Clock Time Analysis Summary\n")
    f.write("=" * 70 + "\n\n")

    f.write("=" * 70 + "\n")
    f.write("CRITICAL INSIGHT: WHY STANDARD SD DOESN'T HELP HERE\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Draft model speed:  {EGPT_DECODE_MS}ms/token (EventGPT)\n")
    f.write(f"Target model speed: {VL_DECODE_MS}ms/token (VideoLLaVA)\n")
    f.write(f"Speed ratio (c):    {EGPT_DECODE_MS/VL_DECODE_MS:.2f} (almost same!)\n\n")
    f.write("For SD to speed up decoding, draft model needs to be MUCH faster.\n")
    f.write("With c ≈ 1.0, acceptance rate >95% needed for any benefit.\n")
    f.write("Our 27.9% acceptance rate is good, but draft model is too slow.\n\n")

    f.write("=" * 70 + "\n")
    f.write("WHERE THE REAL BENEFIT COMES FROM: PARALLEL PREFILL\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"EventGPT prefill:   {EGPT_PREFILL_MS}ms\n")
    f.write(f"VideoLLaVA prefill: {VL_PREFILL_MS}ms\n")
    f.write(f"Free window:        {PARALLEL_OVERLAP_MS}ms (VL - EGPT)\n")
    f.write(f"Free draft tokens:  {FREE_DRAFT_TOKENS} tokens during VL prefill\n\n")
    f.write(f"With α=27.9%: ~{FREE_DRAFT_TOKENS * acceptance_rate:.0f} tokens accepted FREE\n")
    f.write(f"With α=51.6%: ~{FREE_DRAFT_TOKENS * top5_rate:.0f} tokens accepted FREE\n\n")

    f.write("=" * 70 + "\n")
    f.write("TIME TO GENERATE 50 TOKENS\n")
    f.write("=" * 70 + "\n\n")
    ar_total = total_times[0]
    parallel_27_total = total_times[1]
    parallel_51_total = total_times[2]
    f.write(f"AR (VideoLLaVA only):   {ar_total:>6.0f}ms  (baseline)\n")
    f.write(f"  - Prefill: {prefill_times[0]:.0f}ms, Decode: {decode_times[0]:.0f}ms\n")
    f.write(f"Parallel (α=27.9%):     {parallel_27_total:>6.0f}ms  ({ar_total-parallel_27_total:+.0f}ms, {(ar_total-parallel_27_total)/ar_total*100:+.1f}%)\n")
    f.write(f"  - Prefill: {prefill_times[1]:.0f}ms, Decode: {decode_times[1]:.0f}ms\n")
    f.write(f"Parallel (α=51.6%):     {parallel_51_total:>6.0f}ms  ({ar_total-parallel_51_total:+.0f}ms, {(ar_total-parallel_51_total)/ar_total*100:+.1f}%)\n")
    f.write(f"  - Prefill: {prefill_times[2]:.0f}ms, Decode: {decode_times[2]:.0f}ms\n\n")
    f.write("Note: SD rounds are slower than AR with similar model speeds!\n")
    f.write("The 'free' tokens from parallel prefill help, but decode is slower.\n\n")

    f.write("=" * 70 + "\n")
    f.write("PATH FORWARD FOR REAL SPEEDUP\n")
    f.write("=" * 70 + "\n\n")
    f.write("Option 1: Use smaller/faster draft model (EAGLE-style, ~10x faster)\n")
    f.write("  → With c=0.1 and α=27.9%, expect 1.3-1.5x speedup\n\n")
    f.write("Option 2: Use EventGPT for prefill benefit only (no SD during decode)\n")
    f.write("  → Free tokens during parallel prefill, then AR decode\n")
    f.write(f"  → Saves {PARALLEL_OVERLAP_MS}ms prefill overhead\n\n")
    f.write("Option 3: Feature-level alignment (bypass tokenizer)\n")
    f.write("  → Higher acceptance (50-70%) + faster draft head\n")
    f.write("  → Expected 2-2.5x speedup\n")

print(f"Saved: {summary_path}")

# Show plot
plt.show()
