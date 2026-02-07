#!/usr/bin/env python3
"""
Two-Phase Pipeline Evaluation: L1-L4 (prefill) + L5F (decode)

Phase 1 (Prefill Hiding): L1-L4 adapter aligns h_egpt → VL space
  - Same-position comparison: adapted[t] vs vl[t]
  - Consecutive accepts during free window (7 draft slots)

Phase 2 (Decode): L5F fused adapter predicts next VL token
  - Shifted comparison: adapted[t] vs vl[t+1]
  - Consecutive accepts per SD iteration (gamma=5)

Combined E2E speedup estimation.

Use --no_prefill for decode-only baselines (B1-only or L5F-only).
"""
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from feasible.feature_alignment.hidden_adapter import load_any_adapter
from feasible.feature_alignment.measure_feature_acceptance import (
    load_chunked_data, load_lm_head, project_to_tokens, TimingConfig
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefill_checkpoint', default=None, help='L1-L4 checkpoint for prefill (skip with --no_prefill)')
    parser.add_argument('--decode_checkpoint', required=True, help='L5F/B1 checkpoint for decode')
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--lm_head', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--decode_vlm_only', action='store_true', help='Decode adapter is vlm_only (B1)')
    parser.add_argument('--gamma_decode', type=int, default=5)
    parser.add_argument('--no_prefill', action='store_true', help='Skip Phase 1 (decode-only baseline)')
    parser.add_argument('--label', default=None, help='Label for this run (used in output filename)')
    args = parser.parse_args()

    if not args.no_prefill and args.prefill_checkpoint is None:
        parser.error("--prefill_checkpoint is required unless --no_prefill is set")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Timing config
    t = TimingConfig()
    gamma_prefill = int(max(0, t.vl_prefill - t.egpt_prefill) / t.egpt_per_token)
    print(f"Timing: EGPT prefill={t.egpt_prefill}ms, VL prefill={t.vl_prefill}ms")
    print(f"Free window: {t.vl_prefill - t.egpt_prefill:.0f}ms → gamma_prefill={gamma_prefill}")
    print(f"Decode: gamma_decode={args.gamma_decode}")
    if args.no_prefill:
        print(f"Mode: DECODE-ONLY BASELINE (no prefill hiding)")
    print()

    # Load data to CPU
    print("Loading test data...")
    egpt_hidden, vl_hidden, mask = load_chunked_data(args.test_data, 'cpu')
    N, S, D = egpt_hidden.shape
    print(f"  {N} samples, seq_len={S}, dim={D}")

    # Load LM head
    print("Loading LM head...")
    lm_head_weight = load_lm_head(args.lm_head)

    # Get VL ground truth tokens
    print("Computing VL ground truth tokens...")
    vl_tokens, _ = project_to_tokens(vl_hidden, lm_head_weight, mask, top_k=1, batch_size=args.batch_size)

    mask_bool = mask.bool()

    # ================================================================
    # PHASE 1: Prefill Hiding (L1-L4 adapter, same-position)
    # ================================================================
    if not args.no_prefill:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Prefill Hiding")
        print(f"{'='*60}")
        prefill_model, _ = load_any_adapter(args.prefill_checkpoint, device)
        prefill_model = prefill_model.to(device)
        prefill_model.eval()
        print(f"  Adapter: {type(prefill_model).__name__}")

        print("  Computing aligned hidden states...")
        prefill_aligned = []
        with torch.no_grad():
            for i in tqdm(range(0, N, args.batch_size)):
                batch = egpt_hidden[i:i+args.batch_size].to(device)
                out = prefill_model(batch).cpu()
                prefill_aligned.append(out)
        prefill_aligned = torch.cat(prefill_aligned, dim=0)

        # Free GPU
        del prefill_model
        torch.cuda.empty_cache()

        # Project to tokens
        print("  Projecting to tokens...")
        prefill_tokens, _ = project_to_tokens(prefill_aligned, lm_head_weight, mask, top_k=1, batch_size=args.batch_size)

        # Same-position match
        prefill_match = (prefill_tokens == vl_tokens) & mask_bool
        prefill_top1 = prefill_match.float().sum() / mask.sum()
        print(f"  Token top-1 match (same-position): {prefill_top1:.2%}")

        # Consecutive accepts in first gamma_prefill positions
        gp = min(gamma_prefill, S)
        if gp > 0:
            match_prefill = prefill_match[:, :gp]
            cumprod_prefill = match_prefill.int().cumprod(dim=1)
            consec_prefill = cumprod_prefill.sum(dim=1).float()
            prefill_consec_mean = consec_prefill.mean().item()
        else:
            prefill_consec_mean = 0.0

        print(f"  Free tokens accepted: {prefill_consec_mean:.2f} / {gp}")
        prefill_time_saved = prefill_consec_mean * t.vl_per_token
        print(f"  Time saved: {prefill_time_saved:.1f}ms")

        del prefill_aligned, prefill_tokens
    else:
        # No prefill hiding — decode-only baseline
        print(f"\n{'='*60}")
        print(f"PHASE 1: SKIPPED (decode-only baseline)")
        print(f"{'='*60}")
        prefill_top1 = torch.tensor(0.0)
        prefill_consec_mean = 0.0
        prefill_time_saved = 0.0
        gp = gamma_prefill

    # ================================================================
    # PHASE 2: Decode (L5F/B1 adapter, SHIFTED comparison)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: Decode (shifted next-token prediction)")
    print(f"{'='*60}")
    decode_model, _ = load_any_adapter(args.decode_checkpoint, device)
    decode_model = decode_model.to(device)
    decode_model.eval()
    adapter_class = type(decode_model).__name__
    is_fused = adapter_class == 'FusedEAGLEAdapter'
    is_vlm_only = args.decode_vlm_only
    print(f"  Adapter: {adapter_class} (fused={is_fused}, vlm_only={is_vlm_only})")

    print("  Computing decode hidden states...")
    decode_aligned = []
    with torch.no_grad():
        for i in tqdm(range(0, N, args.batch_size)):
            if is_fused:
                be = egpt_hidden[i:i+args.batch_size].to(device)
                bv = vl_hidden[i:i+args.batch_size].to(device)
                out = decode_model(be, bv).cpu()
            elif is_vlm_only:
                bv = vl_hidden[i:i+args.batch_size].to(device)
                out = decode_model(bv).cpu()
            else:
                be = egpt_hidden[i:i+args.batch_size].to(device)
                out = decode_model(be).cpu()
            decode_aligned.append(out)
    decode_aligned = torch.cat(decode_aligned, dim=0)

    del decode_model
    torch.cuda.empty_cache()

    # Project to tokens
    print("  Projecting to tokens...")
    decode_tokens, decode_topk = project_to_tokens(decode_aligned, lm_head_weight, mask, top_k=10, batch_size=args.batch_size)

    # ---- Same-position metrics (for reference) ----
    decode_match_same = (decode_tokens == vl_tokens) & mask_bool
    decode_top1_same = decode_match_same.float().sum() / mask.sum()
    print(f"  Token top-1 (same-position): {decode_top1_same:.2%}")

    # ---- SHIFTED metrics (decode prediction quality) ----
    # decode_tokens[:, t] should predict vl_tokens[:, t+1]
    if S > 1:
        shifted_pred = decode_tokens[:, :-1]     # [N, S-1]
        shifted_target = vl_tokens[:, 1:]         # [N, S-1]
        shifted_mask = mask_bool[:, 1:]            # [N, S-1]

        shifted_match = (shifted_pred == shifted_target) & shifted_mask
        shifted_top1 = shifted_match.float().sum() / shifted_mask.float().sum()
        print(f"  Token top-1 (SHIFTED, next-token): {shifted_top1:.2%}")

        # Top-5 shifted
        shifted_topk_pred = decode_topk[:, :-1, :5]  # [N, S-1, 5]
        shifted_target_exp = shifted_target.unsqueeze(-1)  # [N, S-1, 1]
        shifted_top5_hit = (shifted_topk_pred == shifted_target_exp).any(dim=-1) & shifted_mask
        shifted_top5 = shifted_top5_hit.float().sum() / shifted_mask.float().sum()
        print(f"  Token top-5 (SHIFTED): {shifted_top5:.2%}")

        # Consecutive shifted accepts
        shifted_cumprod = shifted_match.int().cumprod(dim=1)
        shifted_consec = shifted_cumprod.sum(dim=1).float()
        shifted_consec_mean = shifted_consec.mean().item()
        print(f"  Consecutive shifted accepts: {shifted_consec_mean:.2f}")

        # Decode acceptance per gamma iteration
        gd = args.gamma_decode
        if S - 1 >= gd:
            first_gd_match = shifted_match[:, :gd]
            first_gd_cumprod = first_gd_match.int().cumprod(dim=1)
            decode_consec_gd = first_gd_cumprod.sum(dim=1).float()
            decode_consec_mean = decode_consec_gd.mean().item()
        else:
            decode_consec_mean = shifted_consec_mean
        print(f"  Decode consecutive (gamma={gd}): {decode_consec_mean:.2f} / {gd}")

        # Per-position shifted acceptance (first 10)
        print(f"\n  Per-position SHIFTED top-1 (first 10):")
        for pos in range(min(10, S-1)):
            valid = shifted_mask[:, pos]
            if valid.sum() == 0:
                break
            rate = shifted_match[valid, pos].float().mean().item()
            bar = "#" * int(rate * 20)
            print(f"    Pos {pos}→{pos+1}: {rate:6.2%} {bar}")
    else:
        shifted_top1 = 0.0
        shifted_top5 = 0.0
        decode_consec_mean = 0.0

    # ================================================================
    # COMBINED E2E SPEEDUP
    # ================================================================
    print(f"\n{'='*60}")
    print(f"COMBINED E2E SPEEDUP")
    print(f"{'='*60}")

    num_output_tokens = 50

    # Baseline: VL prefill + N decode steps
    baseline_time = t.vl_prefill + num_output_tokens * t.vl_per_token
    print(f"  Baseline: {t.vl_prefill:.0f}ms prefill + {num_output_tokens}×{t.vl_per_token:.0f}ms decode = {baseline_time:.0f}ms")

    # Phase 1: Parallel prefill + free draft tokens
    parallel_prefill = max(t.vl_prefill, t.egpt_prefill + t.adapter_latency)
    remaining_tokens = max(0, num_output_tokens - prefill_consec_mean)
    print(f"  Phase 1: parallel_prefill={parallel_prefill:.0f}ms, free_tokens={prefill_consec_mean:.2f}")

    # Phase 2: SD decode for remaining tokens
    # In real SD, verification is ONE forward pass for γ+1 tokens (not γ+1 separate passes)
    # Cost per SD iteration ≈ 1 × vl_forward + adapter_latency
    # With KV cache, verifying γ+1 tokens costs ~1 forward pass for small γ
    if decode_consec_mean > 0:
        tokens_per_iter = decode_consec_mean + 1
        iterations = remaining_tokens / tokens_per_iter
        # Realistic: single VL forward pass for verification + adapter
        cost_per_iter = t.vl_per_token + t.adapter_latency
        decode_time = iterations * cost_per_iter
    else:
        iterations = remaining_tokens
        decode_time = remaining_tokens * t.vl_per_token

    total_sd_time = parallel_prefill + decode_time
    e2e_speedup = baseline_time / total_sd_time if total_sd_time > 0 else 1.0

    print(f"  Phase 2: {remaining_tokens:.1f} tokens → {iterations:.1f} SD iters")
    print(f"    cost/iter: {t.vl_per_token:.0f}ms verify + {t.adapter_latency:.0f}ms adapter = {t.vl_per_token + t.adapter_latency:.0f}ms")
    print(f"    tokens/iter: {decode_consec_mean + 1:.2f} (α={decode_consec_mean:.2f})")
    print(f"    decode_time: {decode_time:.0f}ms")
    print(f"  Total SD time: {total_sd_time:.0f}ms")
    print(f"  E2E Speedup: {e2e_speedup:.2f}x")

    # Save results
    results = {
        'label': args.label or ('decode_only' if args.no_prefill else 'two_phase'),
        'no_prefill': args.no_prefill,
        'prefill_adapter': args.prefill_checkpoint,
        'decode_adapter': args.decode_checkpoint,
        'decode_vlm_only': args.decode_vlm_only,
        'gamma_prefill': gamma_prefill,
        'gamma_decode': args.gamma_decode,
        'num_output_tokens': num_output_tokens,
        'phase1_prefill_top1': prefill_top1.item() if isinstance(prefill_top1, torch.Tensor) else prefill_top1,
        'phase1_free_tokens': prefill_consec_mean,
        'phase1_time_saved_ms': prefill_time_saved,
        'phase2_same_pos_top1': decode_top1_same.item(),
        'phase2_shifted_top1': shifted_top1.item() if isinstance(shifted_top1, torch.Tensor) else shifted_top1,
        'phase2_shifted_top5': shifted_top5.item() if isinstance(shifted_top5, torch.Tensor) else 0,
        'phase2_decode_consecutive': decode_consec_mean,
        'baseline_time_ms': baseline_time,
        'sd_time_ms': total_sd_time,
        'e2e_speedup': e2e_speedup,
    }

    # Determine output path
    output_dir = Path(args.decode_checkpoint).parent
    if args.label:
        output_file = output_dir / f'two_phase_results_{args.label}.json'
    elif args.no_prefill:
        output_file = output_dir / 'decode_only_results.json'
    else:
        output_file = output_dir / 'two_phase_results.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_file}")


if __name__ == '__main__':
    main()
