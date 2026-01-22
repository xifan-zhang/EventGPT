#!/usr/bin/env python3
"""
Benchmark script to run LLaVA-7B inference with different video lengths.
Supports video lengths from 2s to 1min at 15fps.
"""

import subprocess
import re
import statistics
import json
import time
import argparse
from typing import Dict, List, Tuple
import sys
import os

class LLaVAVideoBenchmark:
    def __init__(self, video_length_seconds: float, fps: int = 15, num_runs: int = 5):
        """
        Initialize benchmark for specific video length.
        
        Args:
            video_length_seconds (float): Video duration in seconds
            fps (int): Frames per second (default: 15)
            num_runs (int): Number of benchmark runs (default: 5)
        """
        self.video_length_seconds = video_length_seconds
        self.fps = fps
        self.num_frames = int(video_length_seconds * fps)
        self.num_runs = num_runs
        self.step_timings: Dict[str, List[float]] = {}
        self.total_times: List[float] = []
        
        print(f"🎬 Video Configuration:")
        print(f"   Duration: {video_length_seconds}s")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {self.num_frames}")
        print(f"   Benchmark Runs: {num_runs}")
        
    def run_single_inference(self, run_number: int) -> Tuple[Dict[str, float], float, bool]:
        """Run a single inference with specified frame count."""
        print(f"\n{'='*60}")
        print(f"🚀 LLaVA-7B Inference - Run {run_number}/{self.num_runs}")
        print(f"📹 Video: {self.video_length_seconds}s ({self.num_frames} frames @ {self.fps}fps)")
        print(f"{'='*60}")
        
        try:
            # Run the profiling script with custom frame count
            start_time = time.time()
            
            # Create a Python script call that imports and runs with custom frames
            python_code = f"""
import sys
sys.path.append('.')
from llava_profiling_demo import run_profiled_inference
run_profiled_inference({self.num_frames})
"""
            
            result = subprocess.run(
                ["python", "-c", python_code],
                capture_output=True,
                text=True,
                cwd="/aiot-nvme-15T-x2-hk01/zhangxifan/code/EventGPT/mllm-profiling",
                timeout=600  # 10 minute timeout
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if result.returncode != 0:
                print(f"❌ Run {run_number} failed with return code {result.returncode}")
                print("STDERR:", result.stderr)
                return {}, total_time, False
            
            # Parse timing data from output
            step_times = self.parse_timing_output(result.stdout)
            
            if not step_times:
                print(f"⚠️  Run {run_number}: No timing data extracted")
                return {}, total_time, False
            
            print(f"✅ Run {run_number} completed successfully")
            for step, duration in step_times.items():
                print(f"   {step}: {duration:.4f}s ({duration*1000:.1f}ms)")
            print(f"   Total execution time: {total_time:.4f}s")
            
            return step_times, total_time, True
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Run {run_number} timed out after 10 minutes")
            return {}, 600.0, False
        except Exception as e:
            print(f"💥 Run {run_number} failed with exception: {e}")
            return {}, 0.0, False
    
    def parse_timing_output(self, output: str) -> Dict[str, float]:
        """Parse step timing data from profiling output."""
        step_times = {}
        
        # Strip ANSI escape codes
        import re as regex_module
        ansi_escape = regex_module.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_output = ansi_escape.sub('', output)
        
        lines = clean_output.split('\n')
        current_step = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for step statistics headers
            step_header_match = re.search(r'===\s*([^=]+?)\s*Statistics', line, re.IGNORECASE)
            if step_header_match:
                current_step = step_header_match.group(1).strip()
                continue
            
            # Look for average timing lines
            if current_step:
                avg_match = re.search(r'Average:\s*([\d.]+)\s*seconds', line, re.IGNORECASE)
                if avg_match:
                    duration = float(avg_match.group(1))
                    step_times[current_step] = duration
                    current_step = None
        
        return step_times
    
    def run_benchmark(self) -> Dict:
        """Run the complete benchmark suite."""
        print(f"\n🎯 Starting LLaVA-7B Video Length Benchmark")
        print(f"📊 Configuration: {self.video_length_seconds}s video, {self.fps}fps, {self.num_frames} frames")
        
        successful_runs = 0
        failed_runs = 0
        
        for run_num in range(1, self.num_runs + 1):
            step_times, total_time, success = self.run_single_inference(run_num)
            
            if success:
                successful_runs += 1
                self.total_times.append(total_time)
                
                # Collect step timings
                for step_name, duration in step_times.items():
                    if step_name not in self.step_timings:
                        self.step_timings[step_name] = []
                    self.step_timings[step_name].append(duration)
            else:
                failed_runs += 1
            
            # Progress update
            if run_num % 2 == 0 or run_num == self.num_runs:
                print(f"\n📈 Progress: {run_num}/{self.num_runs} runs completed")
                print(f"   ✅ Successful: {successful_runs}")
                print(f"   ❌ Failed: {failed_runs}")
        
        if successful_runs == 0:
            print("\n💥 All runs failed! Cannot generate statistics.")
            return {}
        
        # Generate summary statistics
        results = self.generate_summary()
        
        # Save results to JSON
        timestamp = int(time.time())
        filename = f"llava_video_benchmark_{self.video_length_seconds}s_{timestamp}.json"
        self.save_results(results, filename)
        
        return results
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics."""
        print(f"\n{'='*80}")
        print(f"📊 LLAVA-7B VIDEO BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"🎬 Video Configuration:")
        print(f"   Duration: {self.video_length_seconds}s")
        print(f"   FPS: {self.fps}")
        print(f"   Total Frames: {self.num_frames}")
        print(f"   Successful Runs: {len(self.total_times)}/{self.num_runs}")
        
        results = {
            'video_config': {
                'duration_seconds': self.video_length_seconds,
                'fps': self.fps,
                'total_frames': self.num_frames
            },
            'benchmark_info': {
                'num_runs': self.num_runs,
                'successful_runs': len(self.total_times),
                'failed_runs': self.num_runs - len(self.total_times)
            },
            'step_statistics': {},
            'total_execution_stats': {}
        }
        
        # Step-by-step statistics
        if self.step_timings:
            print(f"\n🔍 Step-by-Step Performance:")
            print(f"{'Step':<40} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'StdDev (ms)':<12}")
            print("-" * 88)
            
            for step_name, timings in self.step_timings.items():
                avg_time = statistics.mean(timings)
                min_time = min(timings)
                max_time = max(timings)
                std_dev = statistics.stdev(timings) if len(timings) > 1 else 0.0
                
                print(f"{step_name:<40} {avg_time*1000:<12.1f} {min_time*1000:<12.1f} {max_time*1000:<12.1f} {std_dev*1000:<12.1f}")
                
                results['step_statistics'][step_name] = {
                    'average': avg_time,
                    'minimum': min_time,
                    'maximum': max_time,
                    'std_dev': std_dev,
                    'count': len(timings)
                }
        
        # Total execution statistics
        if self.total_times:
            avg_total = statistics.mean(self.total_times)
            min_total = min(self.total_times)
            max_total = max(self.total_times)
            std_total = statistics.stdev(self.total_times) if len(self.total_times) > 1 else 0.0
            
            print(f"\n⏱️  Total Execution Time:")
            print(f"   Average: {avg_total:.3f}s ({avg_total*1000:.1f}ms)")
            print(f"   Minimum: {min_total:.3f}s ({min_total*1000:.1f}ms)")
            print(f"   Maximum: {max_total:.3f}s ({max_total*1000:.1f}ms)")
            print(f"   Std Dev: {std_total:.3f}s ({std_total*1000:.1f}ms)")
            
            results['total_execution_stats'] = {
                'average': avg_total,
                'minimum': min_total,
                'maximum': max_total,
                'std_dev': std_total,
                'count': len(self.total_times)
            }
        
        # Performance insights
        self.print_performance_insights()
        
        return results
    
    def print_performance_insights(self):
        """Print performance insights and scaling analysis."""
        print(f"\n🔬 Performance Insights:")
        
        if 'step2: encode image tensors into features' in self.step_timings:
            encode_times = self.step_timings['step2: encode image tensors into features']
            avg_encode_time = statistics.mean(encode_times)
            
            # Calculate per-frame encoding time
            per_frame_time = avg_encode_time / self.num_frames if self.num_frames > 0 else 0
            print(f"   📹 Video Processing:")
            print(f"      Encoding time: {avg_encode_time*1000:.1f}ms for {self.num_frames} frames")
            print(f"      Per-frame time: {per_frame_time*1000:.2f}ms/frame")
            
            # Real-time performance analysis
            video_duration = self.video_length_seconds
            if avg_encode_time > 0:
                realtime_ratio = video_duration / avg_encode_time
                print(f"      Real-time ratio: {realtime_ratio:.2f}x")
                if realtime_ratio >= 1.0:
                    print(f"      ✅ Can process faster than real-time")
                else:
                    print(f"      ⚠️  Processing slower than real-time")
        
        # Memory estimation
        estimated_memory = self.num_frames * 768 * 4 / (1024 * 1024)  # Assuming 768-dim features, 4 bytes per float
        print(f"   💾 Estimated Memory:")
        print(f"      Feature memory: ~{estimated_memory:.1f} MB for {self.num_frames} frames")
    
    def save_results(self, results: Dict, filename: str):
        """Save benchmark results to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")

def parse_video_duration(duration_str: str) -> float:
    """Parse video duration string (e.g., '2s', '30s', '1m', '1.5m') to seconds."""
    duration_str = duration_str.lower().strip()
    
    if duration_str.endswith('s'):
        return float(duration_str[:-1])
    elif duration_str.endswith('m'):
        return float(duration_str[:-1]) * 60
    elif duration_str.endswith('min'):
        return float(duration_str[:-3]) * 60
    else:
        # Assume seconds if no unit
        return float(duration_str)

def main():
    parser = argparse.ArgumentParser(description='Benchmark LLaVA-7B with different video lengths')
    parser.add_argument('video_length', type=str, 
                       help='Video duration (e.g., 2s, 30s, 1m, 1.5m). Range: 2s to 1min')
    parser.add_argument('--fps', type=int, default=15,
                       help='Frames per second (default: 15)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of benchmark runs (default: 5)')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch mode with multiple video lengths')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode: run multiple video lengths
        video_lengths = ['2s', '5s', '10s', '15s', '30s', '45s', '1m']
        print(f"🚀 Running batch benchmark for video lengths: {video_lengths}")
        
        all_results = {}
        for duration_str in video_lengths:
            try:
                duration_seconds = parse_video_duration(duration_str)
                if duration_seconds < 2 or duration_seconds > 60:
                    print(f"⚠️  Skipping {duration_str} - outside 2s-1min range")
                    continue
                
                print(f"\n{'='*100}")
                print(f"🎬 BENCHMARKING VIDEO LENGTH: {duration_str} ({duration_seconds}s)")
                print(f"{'='*100}")
                
                benchmark = LLaVAVideoBenchmark(duration_seconds, args.fps, args.runs)
                results = benchmark.run_benchmark()
                all_results[duration_str] = results
                
            except Exception as e:
                print(f"❌ Failed to benchmark {duration_str}: {e}")
        
        # Save combined results
        timestamp = int(time.time())
        combined_filename = f"llava_video_batch_benchmark_{timestamp}.json"
        try:
            with open(combined_filename, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n💾 Combined batch results saved to: {combined_filename}")
        except Exception as e:
            print(f"\n❌ Failed to save combined results: {e}")
            
    else:
        # Single video length mode
        try:
            duration_seconds = parse_video_duration(args.video_length)
            
            if duration_seconds < 2 or duration_seconds > 60:
                print(f"❌ Video length {duration_seconds}s is outside the supported range (2s-1min)")
                return
            
            print(f"🚀 Starting LLaVA-7B Video Benchmark")
            print(f"📹 Video Length: {args.video_length} ({duration_seconds}s)")
            print(f"🎞️  FPS: {args.fps}")
            print(f"🔄 Runs: {args.runs}")
            
            benchmark = LLaVAVideoBenchmark(duration_seconds, args.fps, args.runs)
            results = benchmark.run_benchmark()
            
            if results:
                print(f"\n🎉 Benchmark completed successfully!")
            else:
                print(f"\n💥 Benchmark failed!")
                
        except ValueError as e:
            print(f"❌ Invalid video length format: {args.video_length}")
            print(f"   Use formats like: 2s, 30s, 1m, 1.5m")
            return
        except Exception as e:
            print(f"💥 Benchmark failed with error: {e}")

if __name__ == "__main__":
    main()