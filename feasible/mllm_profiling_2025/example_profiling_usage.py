#!/usr/bin/env python3
"""
Example usage of the profiling utilities.
This demonstrates different ways to use the profiling functions.
"""

import time
from profiler import (
    Profiler, 
    profile_function, 
    profile_function_avg,
    time_block, 
    profile_step, 
    profile_step_avg,
    MultiStepProfiler,
    AveragingProfiler
)


def simulate_work(duration=0.1):
    """Simulate some work by sleeping."""
    time.sleep(duration)


def example_context_manager():
    """Example using context manager."""
    print("\n=== Context Manager Example ===")
    
    with time_block("Data Processing"):
        simulate_work(0.2)
        
    with time_block("Model Inference"):
        simulate_work(0.5)


def example_manual_profiler():
    """Example using manual profiler control."""
    print("\n=== Manual Profiler Example ===")
    
    profiler = Profiler("Custom Operation")
    profiler.start()
    simulate_work(0.3)
    profiler.stop()
    profiler.print_result()
    
    print(f"Duration was: {profiler.get_duration():.4f} seconds")
    print(f"Duration was: {profiler.get_duration_ms():.2f} ms")


@profile_function("Decorated Function")
def decorated_function():
    """Example using function decorator."""
    simulate_work(0.15)
    return "Function result"


@profile_function()  # Uses function name automatically
def another_function():
    """Another decorated function example."""
    simulate_work(0.1)
    return 42


@profile_function_avg("Averaged Function", num_runs=5)
def averaged_function():
    """Example using averaging decorator."""
    simulate_work(0.1)
    return "Averaged result"


@profile_function_avg(num_runs=3)  # Uses function name automatically
def another_averaged_function():
    """Another averaged function example."""
    simulate_work(0.05)
    return 123


def example_function_decorator():
    """Example using function decorators."""
    print("\n=== Function Decorator Example ===")
    
    result1 = decorated_function()
    print(f"Result: {result1}")
    
    result2 = another_function()
    print(f"Result: {result2}")


def example_function_decorator_avg():
    """Example using averaging function decorators."""
    print("\n=== Averaging Function Decorator Example ===")
    
    result1 = averaged_function()
    print(f"Result: {result1}")
    
    result2 = another_averaged_function()
    print(f"Result: {result2}")


def example_averaging_profiler():
    """Example using AveragingProfiler class directly."""
    print("\n=== AveragingProfiler Class Example ===")
    
    # Profile a function with arguments
    profiler = AveragingProfiler("Simulate Work", num_runs=5)
    profiler.profile_function(simulate_work, 0.1)
    
    # Profile a lambda function
    profiler2 = AveragingProfiler("Lambda Function", num_runs=3)
    result = profiler2.profile_function(lambda: simulate_work(0.05) or "lambda result")
    print(f"Lambda result: {result}")


def example_profile_step():
    """Example using profile_step utility."""
    print("\n=== Profile Step Example ===")
    
    result = profile_step("Step A", simulate_work, 0.2)
    result = profile_step("Step B", lambda: simulate_work(0.1))


def example_profile_step_avg():
    """Example using profile_step_avg utility."""
    print("\n=== Profile Step Averaging Example ===")
    
    result = profile_step_avg("Averaged Step A", simulate_work, 5, 0.1)
    result = profile_step_avg("Averaged Step B", lambda: simulate_work(0.05), 3)


def example_multi_step_profiler():
    """Example using MultiStepProfiler for complex workflows."""
    print("\n=== Multi-Step Profiler Example ===")
    
    profiler = MultiStepProfiler("EventGPT Inference Pipeline")
    profiler.start_process()
    
    with profiler.step("step1: convert event data to images"):
        simulate_work(0.1)
    
    with profiler.step("step2: convert images to tensors"):
        simulate_work(0.2)
    
    with profiler.step("step3: model.generate"):
        simulate_work(0.8)
    
    profiler.end_process()


def example_nested_profiling():
    """Example with nested profiling."""
    print("\n=== Nested Profiling Example ===")
    
    with time_block("Overall Process"):
        with time_block("Preprocessing"):
            simulate_work(0.1)
            
        with time_block("Main Processing"):
            with time_block("Sub-step 1"):
                simulate_work(0.05)
            with time_block("Sub-step 2"):
                simulate_work(0.15)
        
        with time_block("Postprocessing"):
            simulate_work(0.08)


if __name__ == "__main__":
    print("Profiling Examples")
    print("==================")
    
    # Run all examples
    example_context_manager()
    example_manual_profiler()
    example_function_decorator()
    example_function_decorator_avg()
    example_averaging_profiler()
    example_profile_step()
    example_profile_step_avg()
    example_multi_step_profiler()
    example_nested_profiling()
    
    print("\n=== All Examples Complete ===")
