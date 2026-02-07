#!/usr/bin/env python3
"""
Preprocess Event Images Script
==============================

This script preprocesses event .npy files into event images in advance,
significantly reducing EventGPT inference time by eliminating stage1 preprocessing.

The script:
1. Reads event .npy files from event_npy/ folder
2. Generates N event images per .npy file (default 5, same as EventGPT preprocessing)
3. Saves images to event_image/ (N=5) or event_image_1f/ (N=1) folder as PNG files
4. Updates JSON with event_image or event_image_1f paths

## Usage
python preprocess_event_images.py --data_dir /mnt/hdd/data/my_egpt_dsec_seq_5s
python preprocess_event_images.py --data_dir /mnt/hdd/data/my_egpt_dsec_seq_5s --num_frames 1
python preprocess_event_images.py --data_dir /mnt/hdd/data/my_egpt_dsec_seq_5s --hz 2

## Apply to all datasets
python preprocess_event_images.py --data_dir /mnt/hdd/data --all_datasets
"""

import os
import sys
import json
import re
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Default number of event images to generate per .npy file (same as EventGPT)
DEFAULT_NUM_FRAMES = 5


def parse_duration_from_dir(data_dir: Path):
    """Parse duration in seconds from dataset directory name.

    Supports patterns like: my_egpt_dsec_seq_5s, my_egpt_dsec_train_500ms, etc.
    Returns duration in seconds as a float, or None if not parseable.
    """
    name = data_dir.name
    # Match trailing duration: e.g., _500ms, _5s, _10s
    match = re.search(r'_(\d+)(ms|s)$', name)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit == 'ms':
        return value / 1000.0
    return float(value)


def generate_event_image_vectorized(x, y, p, height=None, width=None):
    """Generate an event image from event coordinates and polarities (vectorized).

    Red (255, 0, 0) = positive polarity
    Blue (0, 0, 255) = negative polarity
    White (255, 255, 255) = background

    This vectorized version is ~100x faster than the loop-based version.
    """
    if height is None:
        height = int(y.max()) + 1
    if width is None:
        width = int(x.max()) + 1

    # Initialize with white background
    event_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Convert to int for indexing
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    p = p.astype(np.int32)

    # Mask for positive and negative polarities
    pos_mask = p == 1
    neg_mask = p == 0

    # Vectorized assignment for positive polarity (Red)
    event_image[y[pos_mask], x[pos_mask]] = [255, 0, 0]

    # Vectorized assignment for negative polarity (Blue)
    event_image[y[neg_mask], x[neg_mask]] = [0, 0, 255]

    return event_image


def generate_event_image(x, y, p):
    """Wrapper for backward compatibility."""
    return generate_event_image_vectorized(x, y, p)


def get_event_images_list(event_npy, n=DEFAULT_NUM_FRAMES):
    """Split event data into n parts and generate event images for each.

    This matches the logic in common/common.py:get_event_images_list()
    """
    x, y, p, t = event_npy['x'], event_npy['y'], event_npy['p'], event_npy['t']

    total_events = len(t)
    events_per_image = total_events // n

    event_image_list = []

    for i in range(n):
        start_idx = i * events_per_image
        end_idx = (i + 1) * events_per_image if i < n - 1 else total_events

        x_part = x[start_idx:end_idx]
        y_part = y[start_idx:end_idx]
        p_part = p[start_idx:end_idx]

        event_img = generate_event_image(x_part, y_part, p_part)
        event_image_list.append(event_img)

    return event_image_list


def process_single_npy(args):
    """Process a single .npy file and save event images.

    Args:
        args: tuple of (npy_path, output_dir, relative_path, num_frames)

    Returns:
        tuple of (relative_path, list of saved image paths, skipped) or (relative_path, None, False) on error
    """
    npy_path, output_dir, relative_path, num_frames = args

    try:
        # Create output directory path
        npy_stem = Path(npy_path).stem  # e.g., "000000"
        npy_parent = Path(relative_path).parent  # e.g., "interlaken_00_a"
        output_subdir = output_dir / npy_parent

        # Check if all output images already exist
        expected_paths = []
        for i in range(num_frames):
            img_filename = f"{npy_stem}_{i}.png"
            img_path = output_subdir / img_filename
            expected_paths.append((img_path, str(npy_parent / img_filename)))

        if all(p[0].exists() for p in expected_paths):
            # All images exist, skip processing
            return (relative_path, [p[1] for p in expected_paths], True)

        # Load event data
        event_npy = np.load(npy_path, allow_pickle=True)
        event_npy = np.array(event_npy).item()

        # Generate event images
        event_images = get_event_images_list(event_npy, num_frames)

        # Create output directory
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Save images
        saved_paths = []
        for i, img_array in enumerate(event_images):
            img_filename = f"{npy_stem}_{i}.png"
            img_path = output_subdir / img_filename

            img = Image.fromarray(img_array)
            img.save(img_path)

            # Store relative path for JSON
            saved_paths.append(str(npy_parent / img_filename))

        return (relative_path, saved_paths, False)

    except Exception as e:
        print(f"Error processing {npy_path}: {e}")
        return (relative_path, None, False)


def find_json_file(data_dir: Path):
    """Find the instruction JSON file in the dataset directory.

    Looks for common naming patterns:
    - EventGPT_Instruction_Subset.json
    - *_instruction*.json
    - *.json (fallback to first JSON file)
    """
    # Try common patterns in order of priority
    patterns = [
        "EventGPT_Instruction_Subset.json",
        "*_instruction*.json",
        "*Instruction*.json",
        "*.json"
    ]

    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            # Filter out any output/generated JSON files
            for match in matches:
                if "output" not in match.name.lower():
                    return match
    return None


def preprocess_dataset(data_dir: Path, num_workers: int = None, num_frames: int = DEFAULT_NUM_FRAMES, hz: int = None):
    """Preprocess all event .npy files in a dataset directory.

    Args:
        data_dir: Path to dataset directory containing event_npy/ folder
        num_workers: Number of parallel workers (default: CPU count)
        num_frames: Number of event images per clip (1 for single-frame, 5 for default)
        hz: If set, override num_frames by computing duration * hz from dir name
    """
    # If --hz is set, compute num_frames from directory name duration
    if hz is not None:
        duration = parse_duration_from_dir(data_dir)
        if duration is None:
            print(f"Skipping {data_dir}: cannot parse duration from directory name")
            return False
        num_frames = max(1, int(duration * hz))
        print(f"  Hz mode: {duration}s * {hz}Hz = {num_frames} frames")

    event_npy_dir = data_dir / "event_npy"
    # Output dir naming
    if hz is not None:
        output_dir_name = f"event_image_{hz}Hz"
        json_image_key = f"event_image_{hz}Hz"
    elif num_frames == 1:
        output_dir_name = "event_image_1f"
        json_image_key = "event_image_1f"
    else:
        output_dir_name = "event_image"
        json_image_key = "event_image"
    event_image_dir = data_dir / output_dir_name
    json_path = find_json_file(data_dir)

    if not event_npy_dir.exists():
        print(f"Skipping {data_dir}: no event_npy folder found")
        return False

    print(f"\nProcessing dataset: {data_dir}")
    print(f"  Event NPY dir: {event_npy_dir}")
    print(f"  Event Image dir: {event_image_dir}")

    # Find all .npy files
    npy_files = list(event_npy_dir.rglob("*.npy"))

    if not npy_files:
        print(f"  No .npy files found, skipping")
        return False

    print(f"  Found {len(npy_files)} .npy files")

    # Create output directory
    event_image_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for parallel processing
    process_args = []
    for npy_path in npy_files:
        relative_path = npy_path.relative_to(event_npy_dir)
        process_args.append((npy_path, event_image_dir, relative_path, num_frames))

    # Process files in parallel
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    results = {}
    skipped_count = 0
    processed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_npy, args): args[2] for args in process_args}

        with tqdm(total=len(npy_files), desc="  Generating event images") as pbar:
            for future in as_completed(futures):
                relative_path, saved_paths, was_skipped = future.result()
                if saved_paths:
                    # Convert relative_path to match JSON format (e.g., "interlaken_00_a/000000.npy")
                    json_key = str(relative_path)
                    results[json_key] = saved_paths
                    if was_skipped:
                        skipped_count += 1
                    else:
                        processed_count += 1
                pbar.update(1)

    print(f"  Processed {processed_count} files, skipped {skipped_count} existing")

    # Update JSON if it exists
    if json_path and json_path.exists():
        print(f"  Updating JSON: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        updated_count = 0
        for sample in dataset:
            if 'event_data' in sample and sample['event_data']:
                event_data_key = sample['event_data']
                if event_data_key in results:
                    sample[json_image_key] = results[event_data_key]
                    updated_count += 1

        # Save updated JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"  Updated {updated_count} samples in JSON (key: {json_image_key})")
    else:
        # Create a new JSON with the mappings
        output_json_path = data_dir / "EventGPT_Instruction_Subset.json"
        print(f"  No JSON file found, creating: {output_json_path}")
        output_json = []
        for event_data_key, image_paths in results.items():
            output_json.append({
                "event_data": event_data_key,
                json_image_key: image_paths
            })

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)

    return True


def find_all_datasets(base_dir: Path):
    """Find all my_egpt_dsec* dataset directories."""
    datasets = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("my_egpt_dsec"):
            datasets.append(item)
    return sorted(datasets)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess event .npy files into event images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory or base directory (with --all_datasets)"
    )
    parser.add_argument(
        "--all_datasets",
        action="store_true",
        help="Process all my_egpt_dsec* datasets in data_dir"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of event images per clip (default: {DEFAULT_NUM_FRAMES}, use 1 for single-frame)"
    )
    parser.add_argument(
        "--hz",
        type=int,
        default=None,
        help="Generate frames at N Hz (e.g., --hz 2 for 2Hz). Overrides --num_frames by computing duration * hz from dir name."
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    num_frames = args.num_frames
    hz = args.hz

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)

    if args.all_datasets:
        # Process all datasets
        datasets = find_all_datasets(data_dir)
        if not datasets:
            print(f"No my_egpt_dsec* datasets found in {data_dir}")
            sys.exit(1)

        print(f"Found {len(datasets)} datasets to process:")
        for d in datasets:
            print(f"  - {d.name}")

        success_count = 0
        for dataset_dir in datasets:
            if preprocess_dataset(dataset_dir, args.num_workers, num_frames, hz):
                success_count += 1

        print(f"\nCompleted: {success_count}/{len(datasets)} datasets processed")
    else:
        # Process single dataset
        if not preprocess_dataset(data_dir, args.num_workers, num_frames, hz):
            sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
