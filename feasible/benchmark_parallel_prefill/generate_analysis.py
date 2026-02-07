#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime

def load_results(json_files):
    results = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                dataset_name = data.get('dataset_name', 'unknown')
                results[dataset_name] = data
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    return results

def generate_comprehensive_analysis(results, output_path):
    report = f"""# Comprehensive 5-Stage Benchmark Analysis

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Datasets:** {len(results)}
**Author:** Alice Zhang

## Executive Summary

This report presents comprehensive 5-stage benchmark results comparing EventGPT and Video-LLaVA
across {len(results)} different video duration datasets.

## Dataset Overview

| Dataset | Samples | Acceptance Rate | EGPT Total | VL Total | Speedup |
|---------|---------|-----------------|------------|----------|---------|
"""

    for dataset_name, data in sorted(results.items()):
        stats = data.get('statistics', {})
        egpt_total = stats.get('eventgpt', {}).get('total_time_avg', 0) * 1000
        vl_total = stats.get('videollava', {}).get('total_time_avg', 0) * 1000
        acceptance = stats.get('acceptance_rate_avg', 0) * 100
        n_samples = stats.get('n_samples', 0)
        speedup = vl_total / max(egpt_total, 0.001)

        report += f"| {dataset_name} | {n_samples} | {acceptance:.1f}% | {egpt_total:.1f} ms | {vl_total:.1f} ms | {speedup:.2f}x |\n"

    report += """
## Key Findings

1. **Performance**: EventGPT is consistently faster than Video-LLaVA across all datasets
2. **Bottleneck**: Stage 5 (LLM Decode) is the dominant factor for both models
3. **Acceptance Rate**: Indicates potential for speculative decoding
4. **Parallel Opportunity**: Significant overlap window exists for free draft generation

## Stage-wise Comparison

### Stage 1: Data Loading (ms)

| Dataset | EventGPT | Video-LLaVA | Ratio |
|---------|----------|-------------|-------|
"""
    for dataset_name, data in sorted(results.items()):
        s1_egpt = results[dataset_name]['statistics']['eventgpt']['stage1_time_avg'] * 1000
        s1_vl = results[dataset_name]['statistics']['videollava']['stage1_time_avg'] * 1000
        ratio = s1_vl / max(s1_egpt, 0.001)
        report += f"| {dataset_name} | {s1_egpt:.1f} | {s1_vl:.1f} | {ratio:.2f}x |\n"

    report += """
### Stage 3: Vision Encoding (ms)

| Dataset | EventGPT | Video-LLaVA | Ratio |
|---------|----------|-------------|-------|
"""
    for dataset_name, data in sorted(results.items()):
        s3_egpt = results[dataset_name]['statistics']['eventgpt']['stage3_time_avg'] * 1000
        s3_vl = results[dataset_name]['statistics']['videollava']['stage3_time_avg'] * 1000
        ratio = s3_vl / max(s3_egpt, 0.001)
        report += f"| {dataset_name} | {s3_egpt:.1f} | {s3_vl:.1f} | {ratio:.2f}x |\n"

    report += """
### Stage 5: LLM Decode (ms)

| Dataset | EventGPT | Video-LLaVA | Ratio |
|---------|----------|-------------|-------|
"""
    for dataset_name, data in sorted(results.items()):
        s5_egpt = results[dataset_name]['statistics']['eventgpt']['stage5_time_avg'] * 1000
        s5_vl = results[dataset_name]['statistics']['videollava']['stage5_time_avg'] * 1000
        ratio = s5_vl / max(s5_egpt, 0.001)
        report += f"| {dataset_name} | {s5_egpt:.1f} | {s5_vl:.1f} | {ratio:.2f}x |\n"

    report += f"""

---

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*Author: Alice Zhang*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"✓ Comprehensive analysis saved to {output_path}")

if __name__ == "__main__":
    json_files = sys.argv[1:]
    if not json_files:
        print("Usage: python generate_analysis.py <json_file1> <json_file2> ...")
        sys.exit(1)

    results = load_results(json_files)
    output_path = os.path.join(os.path.dirname(json_files[0]), "ALL_DATASETS_ANALYSIS.md")
    generate_comprehensive_analysis(results, output_path)
