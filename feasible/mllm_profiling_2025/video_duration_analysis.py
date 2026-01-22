#!/usr/bin/env python3
"""
Video Duration Analysis for LLaVA-7B
Analyzes frame count scaling from 2 seconds to 1 minute at 15 fps
"""

import torch
import time
import sys
import os
import numpy as np
from PIL import Image

# Add profiling utilities
sys.path.append(os.path.dirname(__file__))
from profiler import AveragingProfiler

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class MockLLaVAModel:
    """Mock LLaVA model optimized for large frame count analysis"""
    
    def __init__(self):
        self.model_name = "LLaVA-7B (Mock - Video Duration Analysis)"
        print(f"✅ {self.model_name} loaded successfully!")
    
    def process_images(self, images, prompt):
        """Simulate image processing with realistic scaling for large frame counts"""
        batch_size = len(images)
        
        # More realistic timing model for large batches
        base_time = 0.05  # 50ms base time
        frame_time = batch_size * 0.008  # 8ms per frame (more optimized for batches)
        batch_overhead = max(0, (batch_size - 10) * 0.002)  # Additional overhead for very large batches
        random_variation = np.random.uniform(0, 0.02)  # 0-20ms variation
        
        total_time = base_time + frame_time + batch_overhead + random_variation
        time.sleep(total_time)
        
        # Create mock tensor
        mock_features = torch.randn(batch_size, 768, device=device)
        return {
            'image_features': mock_features,
            'input_ids': torch.randint(0, 1000, (1, 50), device=device),
            'attention_mask': torch.ones(1, 50, device=device)
        }
    
    def generate_response(self, inputs, max_tokens=100):
        """Simulate text generation - time doesn't scale much with frame count"""
        # Generation time is relatively independent of frame count
        base_time = 0.8  # 800ms base
        token_time = max_tokens * 0.012  # 12ms per token
        random_variation = np.random.uniform(0, 0.1)  # 0-100ms variation
        
        total_time = base_time + token_time + random_variation
        time.sleep(total_time)
        
        responses = [
            "This video shows a sequence of events over time with various activities and movements.",
            "I can observe multiple frames showing the progression of actions in this video sequence.",
            "The video contains a temporal sequence with different visual elements across the frames."
        ]
        
        return np.random.choice(responses)

# Initialize model
model = MockLLaVAModel()

def create_video_frames(num_frames):
    """Create frames for video of specified length"""
    frames = []
    
    print(f"  Creating {num_frames} frames for video analysis...")
    for i in range(num_frames):
        # Create frames with temporal variation
        time_progress = i / max(1, num_frames - 1)  # 0 to 1
        
        # Simulate temporal changes in video
        color = (
            int(64 + time_progress * 128),      # Red increases over time
            int(128 + np.sin(time_progress * 4 * np.pi) * 64),  # Green oscillates
            int(192 - time_progress * 64)       # Blue decreases over time
        )
        
        img = Image.new('RGB', (224, 224), color=color)
        frames.append(img)
        
        if i % 50 == 0 or i == num_frames - 1:  # Progress indicator
            print(f"    Created frame {i+1}/{num_frames}")
    
    return frames

def encode_images_to_features(video_frames, prompt):
    """Step 2: Encode image tensors into features"""
    print(f"  Processing {len(video_frames)} frames...")
    inputs = model.process_images(video_frames, prompt)
    print(f"  Generated features shape: {inputs['image_features'].shape}")
    return inputs

def generate_output(inputs, max_tokens=100):
    """Step 3: Generate the output"""
    print(f"  Generating response...")
    response = model.generate_response(inputs, max_tokens)
    return response

def analyze_video_durations():
    """Analyze different video durations at 15 fps"""
    
    print(f"\n{'='*80}")
    print("🎬 VIDEO DURATION ANALYSIS AT 15 FPS")
    print(f"{'='*80}")
    
    # Define video durations and corresponding frame counts at 15 fps
    video_specs = [
        ("2 seconds", 2, 30),      # 2s * 15fps = 30 frames
        ("5 seconds", 5, 75),      # 5s * 15fps = 75 frames  
        ("10 seconds", 10, 150),   # 10s * 15fps = 150 frames
        ("20 seconds", 20, 300),   # 20s * 15fps = 300 frames
        ("30 seconds", 30, 450),   # 30s * 15fps = 450 frames
        ("1 minute", 60, 900),     # 60s * 15fps = 900 frames
    ]
    
    results = []
    
    for duration_name, duration_sec, frame_count in video_specs:
        print(f"\n📊 Testing {duration_name} video ({frame_count} frames)")
        print("-" * 60)
        
        try:
            # Step 1: Create frames (not profiled as it's just setup)
            print("Step 1: Creating video frames...")
            start_create = time.time()
            frames = create_video_frames(frame_count)
            create_time = time.time() - start_create
            print(f"  Frame creation: {create_time:.3f}s")
            
            # Step 2: Profile encoding
            def encode_frames():
                return encode_images_to_features(frames, "Analyze this video sequence.")
            
            profiler2 = AveragingProfiler(f"step2: encode {frame_count} frames ({duration_name})", num_runs=1)
            inputs = profiler2.profile_function(encode_frames)
            encode_time = profiler2.get_average_duration()
            
            # Step 3: Profile generation  
            def generate_text():
                return generate_output(inputs, 80)
            
            profiler3 = AveragingProfiler(f"step3: generate response for {duration_name}", num_runs=1)
            response = profiler3.profile_function(generate_text)
            generate_time = profiler3.get_average_duration()
            
            # Calculate metrics
            total_processing = encode_time + generate_time
            frames_per_sec = frame_count / encode_time if encode_time > 0 else 0
            memory_mb = (frame_count * 768 * 4) / (1024 * 1024)  # Approximate memory usage
            
            result = {
                'duration': duration_name,
                'duration_sec': duration_sec,
                'frames': frame_count,
                'encode_time': encode_time,
                'generate_time': generate_time,
                'total_time': total_processing,
                'frames_per_sec': frames_per_sec,
                'memory_mb': memory_mb,
                'response': response
            }
            results.append(result)
            
            print(f"  ✅ Encoding: {encode_time:.3f}s ({frames_per_sec:.1f} fps)")
            print(f"  ✅ Generation: {generate_time:.3f}s")
            print(f"  ✅ Total: {total_processing:.3f}s")
            print(f"  📊 Memory: {memory_mb:.1f} MB")
            print(f"  💬 Response: {response[:100]}...")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    # Display comprehensive results table
    print(f"\n{'='*80}")
    print("📈 COMPREHENSIVE VIDEO DURATION ANALYSIS")
    print(f"{'='*80}")
    
    print(f"{'Duration':<12} {'Frames':<8} {'Encode (s)':<12} {'Gen (s)':<10} {'Total (s)':<12} {'FPS':<8} {'Memory (MB)':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['duration']:<12} "
              f"{result['frames']:<8} "
              f"{result['encode_time']:<12.3f} "
              f"{result['generate_time']:<10.3f} "
              f"{result['total_time']:<12.3f} "
              f"{result['frames_per_sec']:<8.1f} "
              f"{result['memory_mb']:<12.1f}")
    
    # Analysis insights
    print(f"\n🎯 KEY INSIGHTS")
    print("=" * 40)
    
    if results:
        # Scaling analysis
        min_frames = min(r['frames'] for r in results)
        max_frames = max(r['frames'] for r in results)
        min_encode = min(r['encode_time'] for r in results)
        max_encode = max(r['encode_time'] for r in results)
        
        scaling_factor = (max_encode / min_encode) / (max_frames / min_frames)
        
        print(f"📊 Encoding Time Scaling:")
        print(f"   {min_frames} → {max_frames} frames: {min_encode:.3f}s → {max_encode:.3f}s")
        print(f"   Scaling efficiency: {scaling_factor:.2f} (1.0 = perfect linear)")
        
        # Memory analysis
        max_memory = max(r['memory_mb'] for r in results)
        print(f"\n💾 Memory Usage:")
        print(f"   Maximum: {max_memory:.1f} MB (for 1 minute video)")
        print(f"   Per frame: {max_memory/max_frames:.3f} MB")
        
        # Performance recommendations
        print(f"\n💡 Performance Recommendations:")
        fast_results = [r for r in results if r['total_time'] < 5.0]
        if fast_results:
            best = max(fast_results, key=lambda x: x['frames'])
            print(f"   Optimal duration: {best['duration']} ({best['frames']} frames)")
            print(f"   Processing time: {best['total_time']:.3f}s")
        
        slow_results = [r for r in results if r['total_time'] > 10.0]
        if slow_results:
            print(f"   Heavy processing: {len(slow_results)} durations >10s")

def analyze_frame_rate_impact():
    """Analyze impact of different frame rates for same duration"""
    
    print(f"\n{'='*80}")
    print("🎞️  FRAME RATE IMPACT ANALYSIS (10 second video)")
    print(f"{'='*80}")
    
    # Different frame rates for 10-second video
    frame_rates = [5, 10, 15, 24, 30]  # fps
    duration = 10  # seconds
    
    for fps in frame_rates:
        frame_count = duration * fps
        print(f"\n📊 Testing {fps} FPS ({frame_count} frames for {duration}s video)")
        print("-" * 50)
        
        try:
            frames = create_video_frames(frame_count)
            
            def encode_frames():
                return encode_images_to_features(frames, f"Analyze this {fps}fps video.")
            
            profiler = AveragingProfiler(f"{fps}fps encoding", num_runs=1)
            inputs = profiler.profile_function(encode_frames)
            encode_time = profiler.get_average_duration()
            
            frames_per_sec = frame_count / encode_time if encode_time > 0 else 0
            memory_mb = (frame_count * 768 * 4) / (1024 * 1024)
            
            print(f"  ✅ Encoding: {encode_time:.3f}s ({frames_per_sec:.1f} processing fps)")
            print(f"  📊 Memory: {memory_mb:.1f} MB")
            print(f"  ⚖️  Efficiency: {frames_per_sec/fps:.2f}x realtime")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    try:
        print(f"🎯 Starting Video Duration Analysis")
        print(f"📍 Using device: {device}")
        print(f"🤖 Model: {model.model_name}")
        
        # Analyze different video durations
        analyze_video_durations()
        
        # Analyze frame rate impact
        analyze_frame_rate_impact()
        
        print(f"\n🎉 Video duration analysis completed!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
