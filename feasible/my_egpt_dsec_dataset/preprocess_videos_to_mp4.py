#!/usr/bin/env python3
"""
Preprocess Video Frames to MP4 Files for Video-LLaVA
=========================================================

This script converts image sequences to MP4 video files for use with Video-LLaVA.
Video-LLaVA requires actual video files decoded with PyAV, not pre-extracted image sequences.

## Dataset Structure

Input:
    dataset_dir/
    └── video/              # Video data (folder of images)
        └── interlaken_00_a/
            └── 000000/     # Folder containing image frames
                ├── 000000.png
                ├── 000001.png
                └── ...

Output:
    dataset_dir/
    └── mp4/                # MP4 video files (created by this script)
        └── interlaken_00_a/
            └── 000000.mp4   # MP4 video file

## Usage

# Preprocess all video frames to MP4
python preprocess_videos_to_mp4.py --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

# Process specific sequence
python preprocess_videos_to_mp4.py --sequence interlaken_00_a --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

# Limit samples for testing
python preprocess_videos_to_mp4.py --max_samples 10 --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

# Specify FPS (default 20, interval=50ms)
python preprocess_videos_to_mp4.py --fps 20 --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

## Requirements

- imageio
- imageio-ffmpeg
- PIL/Pillow
"""

import os
import re
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def natural_sort_key(path):
    """Natural sort key for sorting filenames numerically."""
    path_str = str(path)
    numbers = [int(n) for n in __import__('re').findall(r'\d+', os.path.basename(path_str))]
    if numbers:
        return numbers[-1]
    return 0


def create_video_from_images(image_folder, output_path, fps=10):
    """
    Create an MP4 video from a folder of images using ffmpeg.

    Args:
        image_folder: Path to folder containing video frames
        output_path: Path where MP4 will be saved
        fps: Frames per second for output video

    Returns:
        bool: True if successful, False otherwise
    """
    import subprocess

    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(Path(image_folder).glob(ext))

    if not image_files:
        return False

    # Sort naturally
    image_files = sorted(image_files, key=natural_sort_key)

    # Use ffmpeg directly (much faster than imageio)
    try:
        # Get first image to determine pattern
        first_img = image_files[0]
        first_num = int(re.search(r'(\d+)', first_img.stem).group(1))

        # Determine padding (e.g., %06d for 000000.png)
        padding = len(first_img.stem)

        # Build ffmpeg command with start_number (optimized for speed)
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-framerate', str(fps),
            '-start_number', str(first_num),  # Start from this number
            '-i', os.path.join(image_folder, f'%0{padding}d.png'),
            '-c:v', 'libx264',
            '-preset', 'fast',  # Faster encoding
            '-pix_fmt', 'yuv420p',
            '-crf', '28',  # Lower quality but faster (28 is still good)
            '-loglevel', 'error',  # Suppress ffmpeg output
            output_path
        ]

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode()
        # Extract just the error message, not the full ffmpeg header
        error_line = [line for line in stderr.split('\n') if 'Error' in line or 'error' in line]
        if error_line:
            print(f"  ffmpeg error: {error_line[0][:100]}")
        else:
            print(f"  ffmpeg failed")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def process_single_clip(args):
    """Process a single video clip (for parallel execution)."""
    sequence, clip, video_dir, mp4_dir, fps = args

    # Create output directory structure
    output_dir = os.path.join(mp4_dir, sequence)
    os.makedirs(output_dir, exist_ok=True)

    # Input folder (images)
    input_folder = os.path.join(video_dir, sequence, clip)

    # Output MP4 file
    output_file = os.path.join(output_dir, f"{clip}.mp4")

    # Skip if already exists
    if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
        return (sequence, clip, True, "skipped")

    # Create video
    if create_video_from_images(input_folder, output_file, fps):
        return (sequence, clip, True, "success")
    else:
        return (sequence, clip, False, "failed")


def process_dataset(dataset_dir, max_samples=None, fps=10, sequence=None, num_workers=8):
    """
    Process all video folders and create MP4 files.

    Args:
        dataset_dir: Base directory containing video/ folder
        max_samples: Maximum number of samples to process (None = all)
        fps: Frames per second for output videos
        sequence: Process only this specific sequence
        num_workers: Number of parallel workers (default: 8)
    """
    video_dir = os.path.join(dataset_dir, "video")
    mp4_dir = os.path.join(dataset_dir, "mp4")

    if not os.path.exists(video_dir):
        print(f"Error: video directory not found: {video_dir}")
        return

    # Create mp4 directory
    os.makedirs(mp4_dir, exist_ok=True)

    # Find all video folders
    video_folders = []
    for sequence_dir in sorted(os.listdir(video_dir)):
        sequence_path = os.path.join(video_dir, sequence_dir)
        if os.path.isdir(sequence_path):
            for clip_dir in sorted(os.listdir(sequence_path)):
                clip_path = os.path.join(sequence_path, clip_dir)
                if os.path.isdir(clip_path):
                    video_folders.append((sequence_dir, clip_dir))

    print(f"Found {len(video_folders)} video folders")

    # Filter by sequence if specified
    if sequence:
        video_folders = [(s, c) for s, c in video_folders if s == sequence]

    # Limit samples if specified
    if max_samples:
        video_folders = video_folders[:max_samples]

    print(f"Processing {len(video_folders)} video folders to MP4 with {num_workers} workers...")

    # Prepare arguments for parallel processing
    args_list = [(s, c, video_dir, mp4_dir, fps) for s, c in video_folders]

    # Process in parallel
    success_count = 0
    fail_count = 0
    skip_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_clip, args): args for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting to MP4"):
            sequence, clip, success, status = future.result()
            if status == "skipped":
                skip_count += 1
            elif success:
                success_count += 1
            else:
                fail_count += 1

    print(f"\nConversion complete:")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  MP4 files saved to: {mp4_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess video frames to MP4 files for Video-LLaVA"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/mnt/hdd/data/my_egpt_dsec_seq_5s",
        help="Base directory containing video/ folder"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Process only this specific sequence (e.g., 'interlaken_00_a')"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for output MP4 videos (default: 20, interval=50ms)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )

    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("Video Frames to MP4 Preprocessing for Video-LLaVA")
    print("=" * 60)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"FPS: {args.fps}")
    print(f"Workers: {args.num_workers}")
    if args.sequence:
        print(f"Sequence: {args.sequence}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print()

    process_dataset(
        dataset_dir=args.dataset_dir,
        max_samples=args.max_samples,
        fps=args.fps,
        sequence=args.sequence,
        num_workers=args.num_workers
    )

    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"\nTotal time: {elapsed_str}")


if __name__ == "__main__":
    main()
