import time
import functools
import statistics
from typing import Optional, Callable, Any, List

# Global configuration for color output
ENABLE_COLORS = True

# ANSI color codes for colored output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

def colorize_profiling_tag(tag: str = "[PROFILING]") -> str:
    """Colorize the profiling tag with bright cyan and bold formatting."""
    if not ENABLE_COLORS:
        return tag
    return f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{tag}{Colors.RESET}"

def colorize_text(text: str, color: str = Colors.WHITE) -> str:
    """Colorize text with the specified color."""
    if not ENABLE_COLORS:
        return text
    return f"{color}{text}{Colors.RESET}"

def colorize_duration(duration: float) -> str:
    """Colorize duration text based on the time taken."""
    duration_text = f"{duration:.4f} seconds ({duration*1000:.2f} ms)"
    
    if not ENABLE_COLORS:
        return duration_text
    
    if duration < 0.001:  # Less than 1ms - green (very fast)
        return colorize_text(duration_text, Colors.BRIGHT_GREEN)
    elif duration < 0.1:  # Less than 100ms - green (fast)
        return colorize_text(duration_text, Colors.GREEN)
    elif duration < 1.0:  # Less than 1s - yellow (moderate)
        return colorize_text(duration_text, Colors.YELLOW)
    elif duration < 5.0:  # Less than 5s - orange-ish (slow)
        return colorize_text(duration_text, Colors.BRIGHT_YELLOW)
    else:  # 5s or more - red (very slow)
        return colorize_text(duration_text, Colors.RED)

def set_color_enabled(enabled: bool) -> None:
    """Enable or disable colored output for profiling.
    
    Args:
        enabled: True to enable colors, False to disable
    """
    global ENABLE_COLORS
    ENABLE_COLORS = enabled

def is_color_enabled() -> bool:
    """Check if colored output is enabled.
    
    Returns:
        True if colors are enabled, False otherwise
    """
    return ENABLE_COLORS

class Profiler:
    """A simple profiler class for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        """Context manager entry - start timing."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop timing and print results."""
        self.stop()
        self.print_result()
    
    def start(self):
        """Start timing."""
        print(f"{colorize_profiling_tag()} Starting {colorize_text(self.name, Colors.BRIGHT_WHITE)}")
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing."""
        if self.start_time is None:
            raise RuntimeError("Profiler not started. Call start() first.")
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def print_result(self):
        """Print the timing result."""
        if self.duration is None:
            raise RuntimeError("Profiler not stopped. Call stop() first.")
        print(f"{colorize_profiling_tag()} {colorize_text(self.name, Colors.BRIGHT_WHITE)} completed in {colorize_duration(self.duration)}")
    
    def get_duration(self) -> Optional[float]:
        """Get the duration in seconds."""
        return self.duration
    
    def get_duration_ms(self) -> Optional[float]:
        """Get the duration in milliseconds."""
        return self.duration * 1000 if self.duration is not None else None


class AveragingProfiler:
    """A profiler class that runs measurements multiple times and computes statistics."""
    
    def __init__(self, name: str = "Operation", num_runs: int = 10):
        self.name = name
        self.num_runs = num_runs
        self.durations: List[float] = []
        self.current_run = 0
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """Profile a function multiple times and compute average.
        
        Args:
            func: Function to profile
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The result of the last function call
        """
        print(f"{colorize_profiling_tag()} Running {colorize_text(self.name, Colors.BRIGHT_WHITE)} {colorize_text(str(self.num_runs), Colors.BRIGHT_YELLOW)} times...")
        
        result = None
        self.durations = []
        
        for i in range(self.num_runs):
            run_info = f"Run {i+1}/{self.num_runs}"
            print(f"{colorize_profiling_tag()} {colorize_text(run_info, Colors.BRIGHT_BLUE)}")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            self.durations.append(duration)
            print(f"{colorize_profiling_tag()} {colorize_text(f'Run {i+1}', Colors.BRIGHT_BLUE)} completed in {colorize_duration(duration)}")
        
        self.print_statistics()
        return result
    
    def profile_callable(self, callable_obj: Callable) -> Callable:
        """Profile a callable (function with no arguments) multiple times.
        
        Args:
            callable_obj: A callable that takes no arguments
            
        Returns:
            The result of the last call
        """
        return self.profile_function(callable_obj)
    
    def print_statistics(self):
        """Print comprehensive timing statistics."""
        if not self.durations:
            print(f"{colorize_profiling_tag()} {colorize_text('No timing data available', Colors.RED)}")
            return
            
        avg_time = statistics.mean(self.durations)
        min_time = min(self.durations)
        max_time = max(self.durations)
        
        # Header with bright colors
        header = f"=== {self.name} Statistics ({self.num_runs} runs) ==="
        print(f"\n{colorize_profiling_tag()} {colorize_text(header, Colors.BRIGHT_MAGENTA)}")
        
        # Statistics with appropriate colors
        print(f"{colorize_profiling_tag()} {colorize_text('Average:', Colors.BRIGHT_WHITE)} {colorize_duration(avg_time)}")
        print(f"{colorize_profiling_tag()} {colorize_text('Minimum:', Colors.BRIGHT_GREEN)} {colorize_duration(min_time)}")
        print(f"{colorize_profiling_tag()} {colorize_text('Maximum:', Colors.BRIGHT_RED)} {colorize_duration(max_time)}")
        
        if len(self.durations) > 1:
            std_dev = statistics.stdev(self.durations)
            std_dev_text = f"{std_dev:.4f} seconds ({std_dev*1000:.2f} ms)"
            print(f"{colorize_profiling_tag()} {colorize_text('Std Dev: ', Colors.BRIGHT_WHITE)} {colorize_text(std_dev_text, Colors.YELLOW)}")
            
        # All runs with dim color for less emphasis
        runs_text = [f'{d:.4f}s' for d in self.durations]
        print(f"{colorize_profiling_tag()} {colorize_text('All runs:', Colors.BRIGHT_WHITE)} {colorize_text(str(runs_text), Colors.DIM)}")
        print(f"{colorize_profiling_tag()} {colorize_text('======================================', Colors.BRIGHT_MAGENTA)}")
    
    def get_average_duration(self) -> Optional[float]:
        """Get the average duration in seconds."""
        return statistics.mean(self.durations) if self.durations else None
    
    def get_average_duration_ms(self) -> Optional[float]:
        """Get the average duration in milliseconds."""
        avg = self.get_average_duration()
        return avg * 1000 if avg is not None else None
    
    def get_all_durations(self) -> List[float]:
        """Get all recorded durations."""
        return self.durations.copy()


def profile_function(name: Optional[str] = None):
    """Decorator to profile function execution time.
    
    Args:
        name: Optional name for the profiling output. If not provided, uses function name.
    
    Example:
        @profile_function("My Function")
        def my_function():
            time.sleep(1)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler_name = name or f"function '{func.__name__}'"
            with Profiler(profiler_name):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


def profile_function_avg(name: Optional[str] = None, num_runs: int = 10):
    """Decorator to profile function execution time multiple times and compute average.
    
    Args:
        name: Optional name for the profiling output. If not provided, uses function name.
        num_runs: Number of times to run the function (default: 10).
    
    Example:
        @profile_function_avg("My Function", num_runs=5)
        def my_function():
            time.sleep(1)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler_name = name or f"function '{func.__name__}'"
            profiler = AveragingProfiler(profiler_name, num_runs)
            return profiler.profile_function(func, *args, **kwargs)
        return wrapper
    return decorator


def time_block(name: str):
    """Create a context manager for timing a block of code.
    
    Args:
        name: Name for the profiling output.
    
    Example:
        with time_block("Data Processing"):
            # Your code here
            process_data()
    """
    return Profiler(name)


# Convenience functions for quick profiling
def profile_step(step_name: str, func: Callable, *args, **kwargs) -> Any:
    """Profile a single function call.
    
    Args:
        step_name: Name for the profiling output.
        func: Function to call.
        *args, **kwargs: Arguments to pass to the function.
    
    Returns:
        The result of the function call.
    
    Example:
        result = profile_step("Step 1", my_function, arg1, arg2, kwarg1=value1)
    """
    with Profiler(step_name):
        return func(*args, **kwargs)


def profile_step_avg(step_name: str, func: Callable, num_runs: int = 10, *args, **kwargs) -> Any:
    """Profile a function call multiple times and compute average.
    
    Args:
        step_name: Name for the profiling output.
        func: Function to call.
        num_runs: Number of times to run the function (default: 10).
        *args, **kwargs: Arguments to pass to the function.
    
    Returns:
        The result of the last function call.
    
    Example:
        result = profile_step_avg("Step 1", my_function, 5, arg1, arg2, kwarg1=value1)
    """
    profiler = AveragingProfiler(step_name, num_runs)
    return profiler.profile_function(func, *args, **kwargs)


class MultiStepProfiler:
    """A profiler for tracking multiple steps in a process."""
    
    def __init__(self, process_name: str = "Process"):
        self.process_name = process_name
        self.steps = []
        self.total_start_time = None
        self.total_end_time = None
    
    def start_process(self):
        """Start timing the overall process."""
        print(f"{colorize_profiling_tag()} Starting {colorize_text(self.process_name, Colors.BRIGHT_MAGENTA)}")
        self.total_start_time = time.time()
    
    def step(self, step_name: str):
        """Start timing a new step."""
        return ProfileStep(self, step_name)
    
    def end_process(self):
        """End timing the overall process and print summary."""
        if self.total_start_time is None:
            raise RuntimeError("Process not started. Call start_process() first.")
        
        self.total_end_time = time.time()
        total_duration = self.total_end_time - self.total_start_time
        
        print(f"{colorize_profiling_tag()} {colorize_text(self.process_name, Colors.BRIGHT_MAGENTA)} completed in {colorize_duration(total_duration)}")
        
        if self.steps:
            step_durations = [step['duration'] for step in self.steps]
            step_names = [step['name'] for step in self.steps]
            step_times = [f"{colorize_text(step['name'], Colors.BRIGHT_WHITE)}: {colorize_duration(step['duration'])}" for step in self.steps]
            breakdown_text = ', '.join(step_times)
            print(f"{colorize_profiling_tag()} {colorize_text('Breakdown', Colors.BRIGHT_CYAN)} - {breakdown_text}")
    
    def add_step_result(self, step_name: str, duration: float):
        """Add a completed step result."""
        self.steps.append({'name': step_name, 'duration': duration})


class ProfileStep:
    """A single step in a multi-step profiler."""
    
    def __init__(self, parent: MultiStepProfiler, step_name: str):
        self.parent = parent
        self.step_name = step_name
        self.start_time = None
    
    def __enter__(self):
        print(f"{colorize_profiling_tag()} Starting {colorize_text(self.step_name, Colors.BRIGHT_YELLOW)}")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is None:
            return
        
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"{colorize_profiling_tag()} {colorize_text(self.step_name, Colors.BRIGHT_YELLOW)} completed in {colorize_duration(duration)}")
        self.parent.add_step_result(self.step_name, duration)
