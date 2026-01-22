#!/usr/bin/env python3
"""
Analysis script for LLaVA benchmark results
Calculates percentage breakdown of execution time by step (similar to EventGPT analysis)
"""

import json
import statistics
import glob
import os

def analyze_llava_results(json_file):
    """Analyze LLaVA benchmark results and create percentage breakdown table."""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract data
    step_stats = data['step_statistics']
    total_times = data['total_times']
    avg_total_time = statistics.mean(total_times)
    
    # Filter to main steps only (step1, step2, step3)
    main_steps = {}
    for step_name, stats in step_stats.items():
        if any(step_name.startswith(f'step{i}:') for i in [1, 2, 3]):
            # Only include the main steps, not the frame count variations
            if not any(x in step_name for x in ['1 frame', '2 frame', '3 frame']):
                main_steps[step_name] = stats
    
    # Calculate step percentages (only for main steps 1-3, relative to each other)
    results = []
    total_step_time = 0
    
    # First pass: calculate total time for main steps only
    for step_name, stats in main_steps.items():
        total_step_time += stats['average']
    
    # Second pass: calculate percentages relative to step time only
    step_order = ['step1', 'step2', 'step3']
    
    for i, step_prefix in enumerate(step_order, 1):
        # Find the step that starts with this prefix
        for step_name, stats in main_steps.items():
            if step_name.startswith(step_prefix):
                avg_time = stats['average']
                min_time = stats['minimum']
                max_time = stats['maximum']
                std_dev = stats['std_dev']
                
                # Calculate percentage of total STEP time (not including overhead)
                percentage = (avg_time / total_step_time) * 100
                
                results.append({
                    'step_num': i,
                    'step': step_name,
                    'avg_time_ms': avg_time * 1000,
                    'min_time_ms': min_time * 1000,
                    'max_time_ms': max_time * 1000,
                    'std_dev_ms': std_dev * 1000,
                    'percentage': percentage
                })
                break
    
    # Print comprehensive analysis
    print("🔍 LLAVA-7B BENCHMARK ANALYSIS - STEPS 1-3 ONLY")
    print("=" * 80)
    print(f"📊 Total Runs: {data['benchmark_info']['num_runs']}")
    print(f"⏱️  Total Step Time: {total_step_time:.3f}s ({total_step_time*1000:.1f}ms)")
    print(f"📅 Benchmark Date: {data['benchmark_info']['timestamp']}")
    print()
    
    print("📈 STEP-BY-STEP TIME BREAKDOWN (Steps 1-3 Only)")
    print("=" * 80)
    
    # Table header
    print(f"{'Step':<45} {'Time (ms)':<12} {'Min (ms)':<10} {'Max (ms)':<10} {'Std Dev':<10} {'% of Steps':<12}")
    print("-" * 45, "-" * 12, "-" * 10, "-" * 10, "-" * 10, "-" * 12)
    
    # Display in order (results are already in order 1-3)
    for result in results:
        step_short = result['step'][:44]  # Truncate if too long
        print(f"{step_short:<45} "
              f"{result['avg_time_ms']:<12.1f} "
              f"{result['min_time_ms']:<10.1f} "
              f"{result['max_time_ms']:<10.1f} "
              f"{result['std_dev_ms']:<10.1f} "
              f"{result['percentage']:<12.2f}%")
    
    print("-" * 45, "-" * 12, "-" * 10, "-" * 10, "-" * 10, "-" * 12)
    print(f"{'TOTAL (Steps 1-3)':<45} "
          f"{total_step_time*1000:<12.1f} "
          f"{'-':<10} "
          f"{'-':<10} "
          f"{'-':<10} "
          f"{'100.00':<12}%")
    
    print()
    print("🎯 KEY INSIGHTS (Steps 1-3 Only)")
    print("=" * 40)
    
    # Find most/least time consuming steps
    most_time_consuming = max(results, key=lambda x: x['percentage'])
    least_time_consuming = min(results, key=lambda x: x['percentage'])
    most_variable = max(results, key=lambda x: x['std_dev_ms'])
    most_consistent = min(results, key=lambda x: x['std_dev_ms'])
    
    print(f"🔥 Most Time-Consuming: {most_time_consuming['step']}")
    print(f"   └─ {most_time_consuming['percentage']:.1f}% of step execution time")
    print()
    print(f"⚡ Least Time-Consuming: {least_time_consuming['step']}")
    print(f"   └─ {least_time_consuming['percentage']:.1f}% of step execution time")
    print()
    print(f"📊 Most Variable: {most_variable['step']}")
    print(f"   └─ Standard deviation: {most_variable['std_dev_ms']:.1f}ms")
    print()
    print(f"🎯 Most Consistent: {most_consistent['step']}")
    print(f"   └─ Standard deviation: {most_consistent['std_dev_ms']:.1f}ms")
    print()
    
    # Performance categories
    preprocessing_steps = [r for r in results if any(x in r['step'].lower() for x in ['convert', 'frames'])]
    processing_steps = [r for r in results if 'encode' in r['step'].lower()]
    generation_steps = [r for r in results if 'generate' in r['step'].lower()]
    
    preprocessing_total = sum(r['percentage'] for r in preprocessing_steps)
    processing_total = sum(r['percentage'] for r in processing_steps)
    generation_total = sum(r['percentage'] for r in generation_steps)
    
    print("📋 PERFORMANCE CATEGORIES (Steps 1-3)")
    print("-" * 40)
    print(f"🔄 Video Processing (Step 1): {preprocessing_total:.1f}%")
    print(f"🧠 Feature Encoding (Step 2): {processing_total:.1f}%")
    print(f"✨ Text Generation (Step 3): {generation_total:.1f}%")
    
    # Frame count analysis if available
    frame_steps = {k: v for k, v in step_stats.items() if 'frame(s)' in k}
    if frame_steps:
        print()
        print("📊 FRAME COUNT ANALYSIS")
        print("-" * 40)
        for step_name, stats in sorted(frame_steps.items()):
            frame_count = step_name.split()[2]  # Extract frame count
            avg_time = stats['average']
            print(f"Encoding {frame_count}: {avg_time*1000:.1f}ms (avg)")

def find_latest_benchmark():
    """Find the latest LLaVA benchmark file"""
    pattern = "llava_benchmark_*.json"
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LLaVA benchmark results")
    parser.add_argument("--file", type=str, help="JSON file to analyze (default: latest)")
    
    args = parser.parse_args()
    
    if args.file:
        json_file = args.file
    else:
        json_file = find_latest_benchmark()
    
    if not json_file:
        print("❌ No LLaVA benchmark files found!")
        print("💡 Run 'python benchmark_llava.py' first to generate benchmark data")
        return
    
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return
    
    print(f"📊 Analyzing: {json_file}")
    print()
    
    analyze_llava_results(json_file)

if __name__ == "__main__":
    main()
