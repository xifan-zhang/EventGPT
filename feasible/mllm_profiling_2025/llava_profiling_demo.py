#!/usr/bin/env python3
"""
LLaVA-7B Profiling Demo Script
Demonstrates step-by-step profiling similar to EventGPT approach
Uses mock functions to simulate the actual LLaVA pipeline
"""

import torch
import time
import sys
import os
import numpy as np
from PIL import Image

# Add profiling utilities
sys.path.append(os.path.dirname(__file__))
from profiler import AveragingProfiler, time_block

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class MockLLaVAModel:
    """Mock LLaVA model for demonstration purposes"""
    
    def __init__(self):
        self.model_name = "LLaVA-7B (Mock)"
        print(f"✅ Mock {self.model_name} loaded successfully!")
    
    def process_images(self, images, prompt):
        """Simulate image processing"""
        # Simulate processing time that scales with number of frames
        batch_size = len(images)
        base_time = 0.08  # 80ms base time
        frame_time = batch_size * 0.015  # 15ms per frame
        random_variation = np.random.uniform(0, 0.03)  # 0-30ms variation
        
        total_time = base_time + frame_time + random_variation
        time.sleep(total_time)
        
        # Create mock tensor
        mock_features = torch.randn(batch_size, 768, device=device)
        return {
            'image_features': mock_features,
            'input_ids': torch.randint(0, 1000, (1, 50), device=device),
            'attention_mask': torch.ones(1, 50, device=device)
        }
    
    def generate_response(self, inputs, max_tokens=100):
        """Simulate text generation"""
        # Simulate generation time based on max_tokens
        base_time = 0.5  # 500ms base
        token_time = max_tokens * 0.01  # 10ms per token
        time.sleep(base_time + token_time + np.random.uniform(0, 0.1))
        
        # Mock generated text
        responses = [
            "I can see a beaver in this image. The beaver appears to be in its natural habitat.",
            "This image shows a detailed view of a beaver. The animal is clearly visible and appears to be engaged in typical beaver behavior.",
            "The image contains a beaver, which is a large rodent known for building dams. The beaver is shown in what appears to be its natural environment."
        ]
        
        return np.random.choice(responses)

# Initialize mock model
model = MockLLaVAModel()

def create_demo_frames(num_frames=5):
    """创建一些演示用的图像帧来模拟视频"""
    frames = []
    
    print(f"  Creating {num_frames} demo frames...")
    for i in range(num_frames):
        # Create simple colored images as demo frames with more variety
        color = (
            64 + (i * 40) % 192,   # Red component
            128 + (i * 30) % 128,  # Green component  
            192 - (i * 25) % 128   # Blue component
        )
        img = Image.new('RGB', (224, 224), color=color)
        frames.append(img)
        print(f"  Created frame {i+1}/{num_frames} with color {color}")
    
    return frames

# step1: convert video frames to images
def convert_video_to_frames(num_frames=5):
    """Step 1: Convert video to frames"""
    video_frames = create_demo_frames(num_frames)
    print(f"Number of frames: {len(video_frames)}")
    return video_frames

# step2: encode image tensors into features
def encode_images_to_features(video_frames, prompt):
    """Step 2: Encode image tensors into features"""
    print(f"  Processing {len(video_frames)} frames with prompt: '{prompt[:50]}...'")
    inputs = model.process_images(video_frames, prompt)
    print(f"  Generated features shape: {inputs['image_features'].shape}")
    return inputs

# step3: generate the output
def generate_output(inputs, max_tokens=1000):
    """Step 3: Generate the output"""
    print(f"  Generating response with max_tokens={max_tokens}")
    response = model.generate_response(inputs, max_tokens)
    print(f"  Generated {len(response.split())} words")
    return response

def run_profiled_inference(num_frames=5):
    """Run the complete profiled inference pipeline
    
    Args:
        num_frames (int): Number of frames to process (default: 5)
    """
    
    # 3. 定义不同的提示词
    prompts = [
        "What is happening in this video?",
        "Describe the main action in this video in detail:",
        "Write a comprehensive analysis of this video, including the setting, actions, and potential outcomes:"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"PROFILING RUN {i}: '{prompt}'")
        print(f"{'='*60}")
        
        # Profile Step 1: Convert video to frames
        profiler1 = AveragingProfiler("step1: convert video frames to images", num_runs=1)
        video_frames = profiler1.profile_function(convert_video_to_frames, num_frames)
        
        # Profile Step 2: Encode images to features
        profiler2 = AveragingProfiler("step2: encode image tensors into features", num_runs=1)
        inputs = profiler2.profile_function(encode_images_to_features, video_frames, prompt)
        
        # Clear cache before step 3
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Profile Step 3: Generate output
        profiler3 = AveragingProfiler("step3: generate the output", num_runs=1)
        generated_text = profiler3.profile_function(generate_output, inputs, 100)
        
        # Display results
        print(f"\n🎯 Generated Response:")
        print(f"{'='*50}")
        print(f"{generated_text}")
        print(f"{'='*50}")
        
        # Additional metrics
        print(f"\n📊 Processing Information:")
        print(f"Frames processed: {len(video_frames)}")
        print(f"Feature dimensions: {inputs['image_features'].shape}")
        print(f"Response length: {len(generated_text)} characters")

def run_frame_count_analysis():
    """Analyze the impact of frame count on encoding time"""
    print(f"\n{'='*60}")
    print("🔍 ANALYZING FRAME COUNT IMPACT ON ENCODING TIME")
    print(f"{'='*60}")

    frame_counts = [1, 2, 3, 5, 8]  # Added 5 and 8 frames
    
    for num_frames in frame_counts:
        print(f"\n📊 Testing with {num_frames} frame(s)")
        print("-" * 40)
        
        # Create frames for this test
        def create_n_frames():
            frames = []
            for i in range(num_frames):
                color = (
                    128 + (i * 25) % 128,
                    128 + (i * 20) % 128, 
                    128 + (i * 30) % 128
                )
                img = Image.new('RGB', (224, 224), color=color)
                frames.append(img)
            return frames
        
        frames = create_n_frames()
        
        # Profile step2 with different frame counts
        def encode_with_n_frames():
            return encode_images_to_features(frames, "What is in the video?")
        
        profiler = AveragingProfiler(f"step2: encode {num_frames} frame(s)", num_runs=3)
        inputs = profiler.profile_function(encode_with_n_frames)
        
        # Show processing info
        print(f"   Feature shape: {inputs['image_features'].shape}")
        print(f"   Memory usage (approx): {inputs['image_features'].numel() * 4 / 1024:.2f} KB")
        
        # Calculate expected vs actual time
        expected_time = 0.08 + num_frames * 0.015  # Base + frame time
        actual_time = profiler.get_average_duration()
        if actual_time:
            print(f"   Expected time: {expected_time*1000:.1f}ms, Actual avg: {actual_time*1000:.1f}ms")

def run_comprehensive_benchmark():
    """Run comprehensive benchmark similar to EventGPT"""
    print(f"\n{'='*60}")
    print("🚀 COMPREHENSIVE LLAVA-7B BENCHMARK")
    print(f"{'='*60}")
    
    # Single comprehensive run
    prompt = "Describe what you see in this video in detail."
    
    print(f"Prompt: {prompt}")
    print(f"Device: {device}")
    print(f"Model: {model.model_name}")
    
    # Step 1: Convert video to frames
    profiler1 = AveragingProfiler("step1: convert video frames to images", num_runs=1)
    video_frames = profiler1.profile_function(convert_video_to_frames)
    
    # Step 2: Encode images to features  
    profiler2 = AveragingProfiler("step2: encode image tensors into features", num_runs=1)
    inputs = profiler2.profile_function(encode_images_to_features, video_frames, prompt)
    
    # Step 3: Generate output
    profiler3 = AveragingProfiler("step3: generate the output", num_runs=1)
    generated_text = profiler3.profile_function(generate_output, inputs, 100)
    
    # Final output
    print(f"\n✨ Final Generated Text:")
    print(f"{'='*50}")
    print(f"{generated_text}")
    print(f"{'='*50}")

def run_multiple_iterations(num_runs=3):
    """Run multiple iterations for statistical analysis"""
    print(f"\n{'='*60}")
    print(f"🔄 RUNNING {num_runs} ITERATIONS FOR STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    prompt = "What do you see in this image?"
    
    # Step 1: Multiple runs
    profiler1 = AveragingProfiler("step1: convert video frames to images", num_runs=num_runs)
    video_frames = profiler1.profile_function(convert_video_to_frames)
    
    # Step 2: Multiple runs
    profiler2 = AveragingProfiler("step2: encode image tensors into features", num_runs=num_runs)
    inputs = profiler2.profile_function(encode_images_to_features, video_frames, prompt)
    
    # Step 3: Multiple runs
    profiler3 = AveragingProfiler("step3: generate the output", num_runs=num_runs)
    generated_text = profiler3.profile_function(generate_output, inputs, 50)
    
    print(f"\n🎯 Statistical Analysis Complete!")
    print(f"Final response: {generated_text}")

if __name__ == "__main__":
    try:
        print(f"🎯 Starting LLaVA-7B Profiling Demo")
        print(f"📍 Using device: {device}")
        print(f"🤖 Model: {model.model_name}")
        
        # Run main profiled inference
        run_profiled_inference()
        
        # Run frame count analysis
        # run_frame_count_analysis()
        
        # Run comprehensive benchmark
        # run_comprehensive_benchmark()
        
        # Run multiple iterations
        # run_multiple_iterations(num_runs=5)
        
        print(f"\n🎉 Profiling demo completed successfully!")
        print(f"\n💡 This demo shows the profiling structure used in EventGPT")
        print(f"   You can replace the mock functions with real LLaVA model calls")
        
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
