#!/usr/bin/env python3
"""
E2E Benchmark Runner
====================

Phase 0: Debug — load all adapters, run 1 sample through every method
Phase 1: benchmark_e2e_vl.py — Video-LLaVA standalone (3-stage timing)
Phase 2: benchmark_e2e_wallclock.py — True E2E SD (both models, all adapters, graphs)

Shares the same output directory and timestamp so results are grouped together.

Usage:
  conda run -n egpt python benchmark_e2e.py --max_samples 50 --max_new_tokens 50
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_BASE = SCRIPT_DIR / 'tasks'

# Exact checkpoints from TRAINING_RESULTS_20260207.md
CHECKPOINT_MAP = {
    'L1': 'L1/L1_20260206_095906/best_model.pt',
    'L2': 'L2/L2_20260206_181048/best_model.pt',
    'L3': 'L3/L3_20260206_183919/best_model.pt',
    'L4': 'L4/L4_20260206_192256/best_model.pt',
    'L5': 'L5/L5_20260206_202939/best_model.pt',
    'B1': 'B1/B1_20260206_213805/best_model.pt',
    'L5F': 'L5F/L5F_20260206_224537/best_model.pt',
}

# 6 configs from TRAINING_RESULTS_20260207.md
SD_CONFIGS = [
    {'name': 'B1-only',  'decode': 'B1',  'prefill': None},
    {'name': 'L5F-only', 'decode': 'L5F', 'prefill': None},
    {'name': 'L1+L5F',   'decode': 'L5F', 'prefill': 'L1'},
    {'name': 'L2+L5F',   'decode': 'L5F', 'prefill': 'L2'},
    {'name': 'L3+L5F',   'decode': 'L5F', 'prefill': 'L3'},
    {'name': 'L4+L5F',   'decode': 'L5F', 'prefill': 'L4'},
]


def debug_all_methods(adapter_dir):
    """Phase 0: Load every adapter checkpoint and verify it loads correctly."""
    print("\n" + "=" * 70)
    print("[0/3] DEBUG — Verifying all adapter checkpoints")
    print("=" * 70)

    # Check which adapters each config needs
    needed = set()
    for cfg in SD_CONFIGS:
        needed.add(cfg['decode'])
        if cfg['prefill']:
            needed.add(cfg['prefill'])

    # Check all checkpoint files exist
    missing = []
    found = {}
    for level in sorted(needed):
        rel = CHECKPOINT_MAP.get(level)
        if rel is None:
            missing.append(f"  {level}: no entry in CHECKPOINT_MAP")
            continue
        full = adapter_dir / rel
        if not full.exists():
            missing.append(f"  {level}: {full} NOT FOUND")
        else:
            found[level] = str(full)
            print(f"  {level}: {full} OK")

    if missing:
        print("\n  MISSING CHECKPOINTS:")
        for m in missing:
            print(m)
        print("\n  Cannot continue. Fix missing checkpoints first.")
        return False

    # Try loading each adapter
    print("\n  Loading adapters...")

    # Add project root so we can import
    ROOT = SCRIPT_DIR.parents[1]
    sys.path.insert(0, str(ROOT))

    import torch
    from feasible.feature_alignment.hidden_adapter import load_any_adapter

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_errors = []

    for level, ckpt_path in sorted(found.items()):
        try:
            model, checkpoint = load_any_adapter(ckpt_path, device)
            model = model.to(device)
            model.eval()
            adapter_type = checkpoint.get('adapter_type', '?')
            n_params = sum(p.numel() for p in model.parameters())
            is_fused = type(model).__name__ == 'FusedEAGLEAdapter'
            is_vlm_only = 'B1' in level

            # Quick forward test with dummy data
            hidden_dim = 4096
            dummy = torch.randn(1, 10, hidden_dim, device=device)
            with torch.inference_mode():
                if is_fused:
                    out = model(dummy, dummy)
                else:
                    out = model(dummy)

            print(f"  {level}: {adapter_type} | {n_params:,} params | "
                  f"fwd OK [{list(out.shape)}]")
            del model, out
        except Exception as e:
            load_errors.append(f"  {level}: {e}")
            print(f"  {level}: LOAD FAILED — {e}")

    torch.cuda.empty_cache()

    if load_errors:
        print("\n  ADAPTER LOAD ERRORS:")
        for e in load_errors:
            print(e)
        return False

    # Verify all 6 configs have their adapters
    print("\n  Verifying 6 SD configs:")
    for cfg in SD_CONFIGS:
        parts = []
        if cfg['prefill']:
            parts.append(f"prefill={cfg['prefill']}")
        parts.append(f"decode={cfg['decode']}")
        status = "OK" if cfg['decode'] in found and (
            cfg['prefill'] is None or cfg['prefill'] in found) else "MISSING"
        print(f"  {cfg['name']:<10} [{', '.join(parts)}] — {status}")

    print("\n  All adapters loaded and verified successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run all E2E benchmarks (debug + VL standalone + SD wallclock)")
    parser.add_argument('--dataset_dir', type=str,
                        default='./data/my_egpt_dsec_test/my_egpt_dsec_seq_1s')
    parser.add_argument('--max_samples', type=int, default=50)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--gamma', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--questions_file', type=str, default=None)
    parser.add_argument('--adapter_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_hidden', action='store_true', default=False)
    parser.add_argument('--skip_vl', action='store_true',
                        help='Skip VL standalone benchmark')
    parser.add_argument('--skip_sd', action='store_true',
                        help='Skip SD wallclock benchmark')
    parser.add_argument('--skip_debug', action='store_true',
                        help='Skip debug phase')
    args = parser.parse_args()

    script_dir = SCRIPT_DIR
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.output_dir is None:
        args.output_dir = str(script_dir / 'tasks' / f'e2e_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else ADAPTER_BASE

    python = sys.executable

    print("=" * 70)
    print("E2E BENCHMARK RUNNER")
    print(f"Date: {timestamp}")
    print(f"Output: {args.output_dir}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Samples: {args.max_samples}, Tokens: {args.max_new_tokens}")
    print(f"Adapters: {adapter_dir}")
    print("=" * 70)

    # ---- Phase 0: Debug all methods ----
    if not args.skip_debug:
        # 0a: Verify adapter checkpoints load
        ok = debug_all_methods(adapter_dir)
        if not ok:
            print("\nDebug phase FAILED. Aborting benchmark.")
            sys.exit(1)

        # 0b: Quick E2E run with 1 sample to verify full pipeline
        print("\n" + "=" * 70)
        print("[0b/3] Running 1-sample E2E debug (all methods, raw data)")
        print("=" * 70)

        debug_cmd = [
            python, str(script_dir / 'benchmark_e2e_wallclock.py'),
            '--dataset_dir', args.dataset_dir,
            '--max_samples', '1',
            '--max_new_tokens', '10',
            '--gamma', str(args.gamma),
            '--warmup', '0',
            '--output_dir', args.output_dir,
        ]
        if args.adapter_dir:
            debug_cmd += ['--adapter_dir', args.adapter_dir]

        ret = subprocess.run(debug_cmd)
        if ret.returncode != 0:
            print(f"\n  1-sample debug FAILED (exit code {ret.returncode})")
            print("  Fix errors above before running full benchmark.")
            sys.exit(1)
        else:
            print("\n  1-sample debug PASSED — all methods work E2E")
    else:
        print("\n[0/3] Skipped (--skip_debug)")

    # ---- Phase 1: Video-LLaVA standalone ----
    if not args.skip_vl:
        print("\n" + "=" * 70)
        print("[1/3] Running benchmark_e2e_vl.py")
        print("=" * 70)

        vl_cmd = [
            python, str(script_dir / 'benchmark_e2e_vl.py'),
            '--dataset_dir', args.dataset_dir,
            '--max_samples', str(args.max_samples),
            '--max_new_tokens', str(args.max_new_tokens),
            '--warmup', str(args.warmup),
            '--output_dir', args.output_dir,
        ]
        if args.questions_file:
            vl_cmd += ['--questions_file', args.questions_file]
        if args.save_hidden:
            vl_cmd += ['--save_hidden']

        ret = subprocess.run(vl_cmd)
        if ret.returncode != 0:
            print(f"\n  benchmark_e2e_vl.py exited with code {ret.returncode}")
        else:
            print("\n  benchmark_e2e_vl.py completed successfully")
    else:
        print("\n[1/3] Skipped (--skip_vl)")

    # ---- Phase 2: True E2E SD wallclock ----
    if not args.skip_sd:
        print("\n" + "=" * 70)
        print("[2/3] Running benchmark_e2e_wallclock.py")
        print("=" * 70)

        sd_cmd = [
            python, str(script_dir / 'benchmark_e2e_wallclock.py'),
            '--dataset_dir', args.dataset_dir,
            '--max_samples', str(args.max_samples),
            '--max_new_tokens', str(args.max_new_tokens),
            '--gamma', str(args.gamma),
            '--warmup', str(args.warmup),
            '--output_dir', args.output_dir,
        ]
        if args.questions_file:
            sd_cmd += ['--questions_file', args.questions_file]
        if args.adapter_dir:
            sd_cmd += ['--adapter_dir', args.adapter_dir]

        ret = subprocess.run(sd_cmd)
        if ret.returncode != 0:
            print(f"\n  benchmark_e2e_wallclock.py exited with code "
                  f"{ret.returncode}")
        else:
            print("\n  benchmark_e2e_wallclock.py completed successfully")
    else:
        print("\n[2/3] Skipped (--skip_sd)")

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print(f"Results: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
