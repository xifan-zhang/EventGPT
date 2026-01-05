#!/usr/bin/env python3
"""
Script to build EventGPT DSEC dataset from DSEC sequences.

This script:
1. Processes DSEC sequences (zurich_city, interlaken, thun)
2. Splits event streams into continuous 50ms clips
3. Saves event clips as .npy files
4. Creates instruction dataset with questions (answers generated separately)

Next step: Run generate_answers_qwen.py to analyze event visualizations with Qwen model.

NOTE: Requires hdf5plugin for DSEC HDF5 access: pip install hdf5plugin

Usage: python build_my_egpt_dsec_dataset.py
"""

# Set HDF5 plugin path BEFORE importing h5py
import os
# Try to find the correct plugin path
plugin_paths = [
    '/home/ps/anaconda3/envs/egpt/lib/python3.10/site-packages/hdf5plugin/plugins',
    '/home/ps/anaconda3/lib/python3.11/site-packages/hdf5plugin/plugins',
    '/usr/local/hdf5/lib/plugin'
]

for path in plugin_paths:
    if os.path.exists(path):
        os.environ['HDF5_PLUGIN_PATH'] = path
        print(f"Set HDF5_PLUGIN_PATH to: {path}")
        break

# Also set library path to conda HDF5
conda_lib = '/home/ps/anaconda3/envs/egpt/lib'
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = conda_lib + ':' + os.environ['LD_LIBRARY_PATH']
else:
    os.environ['LD_LIBRARY_PATH'] = conda_lib

import json
import numpy as np
import h5py
import gc
import argparse
import shutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def get_memory_usage():
    """Get current memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    return 0
# Note: Qwen model imports and loading moved to answer generation script
# This script only creates the dataset structure

# Configuration
DSEC_ROOTS = ['/mnt/hdd/data/DSEC/test']  # Only test data for testing
OUTPUT_DIR = "/mnt/hdd/data/my_egpt_dsec_dataset"
TOP50_QUESTIONS = '/home/ps/Documents/code/EventGPT/feasible/analysis_datasets/results_egpt_dsec_split/dsec_questions_top50.txt'

# Qwen model loading removed - handled in generate_answers_qwen.py

# Event rendering removed - handled in generate_answers_qwen.py

# Answer generation removed - handled in generate_answers_qwen.py

def load_image_timestamps(seq_path):
    """Load image timestamps for alignment."""
    timestamps_file = os.path.join(seq_path, 'images', 'timestamps.txt')
    if os.path.exists(timestamps_file):
        timestamps = []
        with open(timestamps_file, 'r') as f:
            for line in f:
                timestamps.append(int(line.strip()))
        return timestamps
    return None

def estimate_clip_count(sequence_duration_us, clip_duration_us, image_timestamps=None):
    """Estimate the number of clips that will be created for a given sequence and clip duration.

    This estimation matches the actual sequential clip creation algorithm:
    - Creates sequential, non-overlapping clips from sequence start
    - Only complete clips that fit entirely within sequence duration
    - Number of clips = floor(sequence_duration / clip_duration)
    """
    if sequence_duration_us < clip_duration_us:
        return 0

    # For sequential clips, the count is simply how many complete clips fit
    # This doesn't depend on image timestamps, just sequence and clip durations
    sequence_duration_s = sequence_duration_us / 1000000.0
    clip_duration_s = clip_duration_us / 1000000.0

    # Calculate how many complete sequential clips fit in the sequence
    estimated_clips = int(sequence_duration_s / clip_duration_s)

    # For very short sequences, ensure at least 1 clip if possible
    if estimated_clips == 0 and sequence_duration_us >= clip_duration_us:
        estimated_clips = 1

    return estimated_clips

def split_event_by_time(event_npy, time_interval=50000, image_timestamps=None, sequence_duration_us=None):
    """Split event data into 50ms clips, one for each image frame."""
    p = event_npy['p']
    t = event_npy['t']
    x = event_npy['x']
    y = event_npy['y']

    # If we have image timestamps, create sequential clips of specified duration
    if image_timestamps and len(image_timestamps) > 0:
        clip_duration_s = time_interval / 1000000.0
        print(f"  Creating sequential {clip_duration_s:.1f}s event clips")

        split_data = []
        total_events_used = 0

        # Calculate sequence start and end times
        sequence_start_us = min(image_timestamps)
        sequence_end_us = max(image_timestamps) if sequence_duration_us is None else min(max(image_timestamps), sequence_duration_us)

        # Create sequential, non-overlapping clips from sequence start
        from tqdm import tqdm
        clip_start = sequence_start_us
        clip_idx = 0

        with tqdm(desc="    Sequential clips", unit="clip", leave=False) as pbar:
            while clip_start + time_interval <= sequence_end_us:
                clip_end = clip_start + time_interval

                # Find events within this sequential time window
                mask = (t >= clip_start) & (t < clip_end)
                indices = np.where(mask)[0]
                num_events = len(indices)

                if num_events > 0:  # Only create clip if there are events
                    # Find which images fall within this clip's time window
                    image_indices_in_clip = []
                    for img_idx, img_ts in enumerate(image_timestamps):
                        if clip_start <= img_ts < clip_end:
                            image_indices_in_clip.append(img_idx)

                    # Store indices and metadata
                    split_data.append({
                        'indices': indices,
                        'time_start': clip_start,
                        'time_end': clip_end,
                        'duration_ms': clip_duration_s * 1000.0,
                        'image_timestamp': clip_end,
                        'image_idx': image_indices_in_clip[0] if image_indices_in_clip else clip_idx,
                        'image_indices_in_clip': image_indices_in_clip,
                        'num_events': num_events
                    })
                    total_events_used += num_events

                clip_start = clip_end
                clip_idx += 1
                pbar.update(1)

        if split_data:
            avg_events_per_clip = total_events_used / len(split_data)
            print(f"  ✓ Created {len(split_data)} image-aligned 50ms event clips")
            print(f"  ✓ Average events per clip: {avg_events_per_clip:.0f}")
            return split_data
        else:
            print("  ✗ No events found in any image time windows")
            print("  Falling back to fixed time bins")

    # Fallback to fixed time bins
    print("  Using fixed 50ms intervals")
    time_bins = (t // time_interval) * time_interval
    unique_bins = np.unique(time_bins)

    split_data = []
    for bin in unique_bins:
        mask = time_bins == bin
        if np.sum(mask) > 0:
            split_data.append({
                'p': p[mask],
                't': t[mask],
                'x': x[mask],
                'y': y[mask],
                'time_start': bin,
                'time_end': bin + time_interval,
                'duration_ms': time_interval / 1000.0
            })
    return split_data

def process_sequence(seq_path, clip_counter, output_npy_dir=None):
    """Process a single DSEC sequence, generating only event_npy clips (no JSON)."""
    seq_name = os.path.basename(seq_path)
    print(f"Processing {seq_name}...")

    if output_npy_dir is None:
        output_npy_dir = os.path.join(os.path.dirname(os.path.dirname(seq_path)), 'my_egpt_dsec_dataset_real', 'event_npy')

    events_path = os.path.join(seq_path, 'events', 'left', 'events.h5')
    if not os.path.exists(events_path):
        print(f"  No events.h5: {events_path}")
        return clip_counter

    try:
        print(f"  Opening HDF5 file: {events_path}")
        with h5py.File(events_path, 'r') as f:
            print("  HDF5 file opened successfully")
            if 'events' not in f:
                print("  No 'events' group")
                return clip_counter

            ev = f['events']
            required_fields = ['p', 't', 'x', 'y']
            if not all(field in ev for field in required_fields):
                print("  Missing event fields")
                return clip_counter

            # Load image timestamps for alignment
            image_timestamps = load_image_timestamps(seq_path)
            if image_timestamps:
                print(f"  Loaded {len(image_timestamps)} image timestamps for alignment")
                print(f"  Image range: {image_timestamps[0]:,} to {image_timestamps[-1]:,} μs")
            else:
                print("  No image timestamps found, using fixed 50ms intervals")

            print("  Loading event data...")
            # Load events efficiently based on image timestamp range
            total_events = ev['t'].shape[0]

            if image_timestamps:
                # Calculate the temporal range needed for image clips
                min_image_time = min(image_timestamps)
                max_image_time = max(image_timestamps)

                if 't_offset' in f:
                    t_offset = f['t_offset'][()]
                    # Convert image timestamps to raw event timestamps for efficient loading
                    raw_min_time = min_image_time - t_offset - 25000  # -25ms buffer
                    raw_max_time = max_image_time - t_offset + 25000  # +25ms buffer

                    # Find indices of events within this range
                    event_times = ev['t'][:]
                    mask = (event_times >= raw_min_time) & (event_times <= raw_max_time)
                    indices = np.where(mask)[0]

                    if len(indices) > 0:
                        start_idx, end_idx = indices[0], indices[-1]
                        event_npy = {field: ev[field][start_idx:end_idx+1] for field in required_fields}
                        print(f"  Loaded {len(event_npy['t']):,} events in image range (from {total_events:,} total)")

                        # Clean up temporary arrays
                        del event_times, mask, indices
                        gc.collect()
                    else:
                        raise ValueError(f"No events found in image timestamp range for sequence {seq_path}")
                else:
                    raise KeyError(f"No timestamp offset (t_offset) found in HDF5 file for sequence {seq_path}")
            else:
                raise FileNotFoundError(f"No image timestamps found for sequence {seq_path}. Expected timestamps.txt file.")

            # Apply timestamp offset to align with image timestamps
            if 't_offset' in f:
                t_offset = f['t_offset'][()]
                event_npy['t'] = event_npy['t'] + t_offset
                print(f"  Applied timestamp offset: {t_offset:,} μs")
            else:
                print("  Warning: No timestamp offset found in HDF5 file")

            print(f"  Loaded {len(event_npy['t']):,} events (all {total_events:,} total)")
            print(f"  Event time range: {event_npy['t'][0]:,} to {event_npy['t'][-1]:,} μs")

            # Estimate and verify clip count
            estimated_clips = estimate_clip_count(sequence_duration_us, 50000, image_timestamps)  # 50000 = 50ms
            print(f"  Estimated clips: {estimated_clips}")

            # Split into clips aligned with image frames or fixed 50ms intervals
            # Only create complete clips that fit within sequence duration
            clips = split_event_by_time(event_npy, time_interval=50000, image_timestamps=image_timestamps, sequence_duration_us=sequence_duration_us)
            actual_clips = len(clips)
            print(f"  Split into {actual_clips} clips")

            # Verify estimation accuracy
            if abs(estimated_clips - actual_clips) > max(1, actual_clips * 0.2):  # Allow 20% error margin
                print(f"  ⚠️  Estimation off by {abs(estimated_clips - actual_clips)} clips ({estimated_clips} estimated vs {actual_clips} actual)")
            else:
                print(f"  ✅ Estimation accurate: {estimated_clips} estimated, {actual_clips} actual")

            # Log clip duration statistics
            if clips:
                durations = [clip['duration_ms'] for clip in clips]
                print(f"  Clip durations: min={min(durations):.1f}ms, max={max(durations):.1f}ms, avg={sum(durations)/len(durations):.1f}ms")

            # Process each clip in batches to reduce memory pressure
            from tqdm import tqdm
            valid_clips = [clip for clip in clips if clip['num_events'] > 0]
            BATCH_SIZE = 50  # Process clips in batches

            print(f"  Processing {len(valid_clips)} clips in batches of {BATCH_SIZE}")
            for batch_start in range(0, len(valid_clips), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(valid_clips))
                batch_clips = valid_clips[batch_start:batch_end]

                for i, clip in enumerate(batch_clips):
                    global_clip_idx = batch_start + i

                    # Extract actual event data from indices and save clip as .npy
                    clip_filename = f"{global_clip_idx:06d}.npy"
                    seq_dir = os.path.join(output_npy_dir, seq_name)
                    os.makedirs(seq_dir, exist_ok=True)
                    npy_path = os.path.join(seq_dir, clip_filename)

                    # Extract event data using indices
                    indices = clip['indices']
                    clip_data = {
                        'p': event_npy['p'][indices],
                        't': event_npy['t'][indices],
                        'x': event_npy['x'][indices],
                        'y': event_npy['y'][indices],
                        'time_start': clip['time_start'],
                        'time_end': clip['time_end'],
                        'duration_ms': clip['duration_ms'],
                        'image_timestamp': clip['image_timestamp'],
                        'image_idx': clip['image_idx'],
                        'num_events': clip['num_events']
                    }

                    np.save(npy_path, clip_data)

                    clip_counter += 1

                # Memory cleanup after each batch
                del batch_clips
                gc.collect()

            # Final cleanup for this sequence
            del valid_clips, clips, event_npy
            gc.collect()

    except Exception as e:
        print(f"  Error processing sequence: {e}")
        import traceback
        traceback.print_exc()
        return clip_counter

    return clip_counter

def main():
    """Main function to build the dataset."""
    parser = argparse.ArgumentParser(description="Build EventGPT DSEC dataset")
    parser.add_argument("--sequence", type=str, default=None,
                       help="Process only this sequence (e.g., 'thun_01_a')")
    args = parser.parse_args()

    print("Starting dataset creation...")
    if args.sequence:
        print(f"Processing only sequence: {args.sequence}")

    # Clean output directory to ensure a fresh run
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global OUTPUT_NPY_DIR
    OUTPUT_NPY_DIR = os.path.join(OUTPUT_DIR, 'event_npy')
    os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Find DSEC sequences
    dsec_locs = ['zurich_city', 'interlaken', 'thun']
    sequences = []
    for root in DSEC_ROOTS:
        if os.path.exists(root):
            for name in os.listdir(root):
                seq_path = os.path.join(root, name)
                if os.path.isdir(seq_path) and any(name.startswith(loc + '_') for loc in dsec_locs):
                    sequences.append(seq_path)

    print(f"Found {len(sequences)} DSEC sequences")

    # Filter sequences if specified
    if args.sequence:
        sequences = [seq for seq in sequences if os.path.basename(seq) == args.sequence]
        print(f"Filtered to {len(sequences)} sequences matching '{args.sequence}'")
        if len(sequences) == 0:
            print(f"No sequence found matching '{args.sequence}'")
            return

    print(f"Output directory: {OUTPUT_DIR}")

    # Process sequences (event_npy only)
    clip_counter = 0

    from tqdm import tqdm
    print(f"Processing {len(sequences)} DSEC sequences...")

    for i, seq_path in enumerate(tqdm(sequences, desc="Sequences", unit="seq")):
        seq_name = os.path.basename(seq_path)
        mem_before = get_memory_usage()
        tqdm.write(f"Processing sequence {i+1}/{len(sequences)}: {seq_name} (Memory: {mem_before:.1f}MB)")
        clip_counter = process_sequence(seq_path, clip_counter, OUTPUT_NPY_DIR)
        mem_after = get_memory_usage()
        tqdm.write(f"Sequence complete. Total clips: {clip_counter}, Memory: {mem_after:.1f}MB (+{mem_after-mem_before:.1f}MB)")

    print(f"Created NPY clips: {clip_counter}")
    print(f"NPY files in {OUTPUT_NPY_DIR}")
    print(f"To generate answers, run: python generate_answers_qwen.py --dataset_dir {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
