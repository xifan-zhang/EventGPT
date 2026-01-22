#!/usr/bin/env python3
"""
Script to build EventGPT DSEC TRAINING dataset from DSEC train sequences.

This script processes DSEC training sequences with the same time durations as test set:
200ms, 500ms, 1s, 2s, 4s, 5s, 10s, 20s

Input: /mnt/hdd/data/DSEC/train
Output: /mnt/hdd/data/my_egpt_dsec_train_<duration>/

Usage: python build_my_egpt_dsec_train.py [--durations "200ms,500ms,1s,2s,4s,5s,10s,20s"]
"""

# Set HDF5 plugin path BEFORE importing h5py
import os
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
import time
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def get_memory_usage():
    """Get current memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return 0

# Configuration for TRAINING data
DSEC_ROOTS = ['/mnt/hdd/data/DSEC/train']
OUTPUT_DIR_BASE = '/mnt/hdd/data/my_egpt_dsec_train'

# All time durations matching test set
DEFAULT_DURATIONS = "200ms,500ms,1s,2s,4s,5s,10s,20s"

def format_duration(duration_us):
    """Format duration in us to a human-readable string with units."""
    if duration_us % 1000000 == 0:
        return f"{duration_us // 1000000}s"
    elif duration_us % 1000 == 0:
        return f"{duration_us // 1000}ms"
    else:
        return f"{duration_us}us"

def get_output_dir(clip_duration_us):
    """Get output directory for specific clip duration (training set)."""
    duration_str = format_duration(clip_duration_us)
    return f'{OUTPUT_DIR_BASE}/my_egpt_dsec_train_{duration_str}'

def calculate_potential_clips(sequence_duration_us, clip_durations_us):
    """Calculate how many clips of each duration can be created from a sequence."""
    potential_clips = {}
    for duration_us in clip_durations_us:
        if sequence_duration_us < duration_us:
            potential_clips[duration_us] = 0
            continue
        step_size = duration_us // 2
        sequence_start_us = 0
        sequence_end_us = sequence_duration_us
        clip_count = 0
        clip_start = sequence_start_us
        while clip_start + duration_us <= sequence_end_us:
            clip_count += 1
            clip_start += step_size
        potential_clips[duration_us] = max(1, clip_count)
    return potential_clips

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

def split_event_by_time(event_npy, time_interval=50000, image_timestamps=None, image_paths=None):
    """Split event data into clips of specified duration."""
    p = event_npy['p']
    t = event_npy['t']
    x = event_npy['x']
    y = event_npy['y']

    if image_timestamps and len(image_timestamps) > 0:
        clip_duration_s = time_interval / 1000000.0
        print(f"  Creating sequential {clip_duration_s:.1f}s event clips with 50% overlap")

        split_data = []
        total_events_used = 0

        sequence_start_us = min(image_timestamps)
        sequence_end_us = max(image_timestamps)
        step_size = time_interval // 2

        from tqdm import tqdm
        clip_start = sequence_start_us
        clip_idx = 0

        with tqdm(desc="    Sequential clips", unit="clip", leave=False) as pbar:
            while clip_start + time_interval <= sequence_end_us:
                clip_end = clip_start + time_interval

                mask = (t >= clip_start) & (t < clip_end)
                indices = np.where(mask)[0]
                num_events = len(indices)

                if num_events > 0:
                    image_indices_in_clip = []
                    for img_idx, img_ts in enumerate(image_timestamps):
                        if clip_start <= img_ts < clip_end:
                            image_indices_in_clip.append(img_idx)

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

                clip_start += step_size
                clip_idx += 1
                pbar.update(1)

        if split_data:
            avg_events_per_clip = total_events_used / len(split_data)
            print(f"  Created {len(split_data)} sequential {clip_duration_s:.1f}s event clips with 50% overlap")
            print(f"  Average events per clip: {avg_events_per_clip:.0f}")
            return split_data
        else:
            print("  No events found in any image time windows")
            print("  Falling back to fixed time bins")

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

def find_event_indices_for_time_range(h5file, t_offset, time_start_us, time_end_us):
    """
    Find event indices for a given time range using ms_to_idx lookup table.
    This avoids loading all events into memory.

    Returns (start_idx, end_idx) or (None, None) if no events found.
    """
    # Convert to raw time (without offset)
    raw_start = time_start_us - t_offset
    raw_end = time_end_us - t_offset

    # Use ms_to_idx lookup table if available
    if 'ms_to_idx' in h5file:
        ms_to_idx = h5file['ms_to_idx']
        # Convert us to ms for lookup
        start_ms = max(0, int(raw_start // 1000))
        end_ms = int(raw_end // 1000) + 1

        # Clamp to valid range
        max_ms = len(ms_to_idx) - 1
        start_ms = min(start_ms, max_ms)
        end_ms = min(end_ms, max_ms)

        if start_ms <= end_ms:
            start_idx = int(ms_to_idx[start_ms])
            end_idx = int(ms_to_idx[end_ms])
            return start_idx, end_idx

    # Fallback: binary search on timestamps (still memory efficient)
    ev_t = h5file['events']['t']
    total_events = ev_t.shape[0]

    # Binary search for start
    lo, hi = 0, total_events
    while lo < hi:
        mid = (lo + hi) // 2
        if ev_t[mid] < raw_start:
            lo = mid + 1
        else:
            hi = mid
    start_idx = lo

    # Binary search for end
    lo, hi = start_idx, total_events
    while lo < hi:
        mid = (lo + hi) // 2
        if ev_t[mid] < raw_end:
            lo = mid + 1
        else:
            hi = mid
    end_idx = lo

    return start_idx, end_idx


def process_sequence(seq_path, clip_counter, output_npy_dir=None, output_video_dir=None, clip_duration_us=50000):
    """
    Process a single DSEC sequence using STREAMING approach.
    Events are loaded per-clip, not all at once, to avoid memory issues.
    """
    seq_name = os.path.basename(seq_path)
    print(f"Processing {seq_name}...")

    if output_npy_dir is None:
        output_npy_dir = os.path.join(os.path.dirname(os.path.dirname(seq_path)), 'my_egpt_dsec_train', 'event_npy')

    if output_video_dir is None:
        output_video_dir = os.path.join(os.path.dirname(output_npy_dir), 'video')

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

            image_timestamps = load_image_timestamps(seq_path)
            if not image_timestamps:
                if clip_duration_us > 50000:
                    print(f"  No image timestamps found and clip duration ({clip_duration_us/1000000.0:.1f}s) > 50ms")
                    print("  Cannot determine sequence duration, skipping this sequence")
                    return clip_counter
                print("  No image timestamps found, skipping")
                return clip_counter

            print(f"  Loaded {len(image_timestamps)} image timestamps for alignment")
            print(f"  Image range: {image_timestamps[0]:,} to {image_timestamps[-1]:,} us")

            sequence_duration_us = image_timestamps[-1] - image_timestamps[0]
            sequence_duration_s = sequence_duration_us / 1000000.0
            required_duration_s = clip_duration_us / 1000000.0

            if sequence_duration_us < clip_duration_us:
                print(f"  Sequence duration ({sequence_duration_s:.1f}s) is shorter than required clip duration ({required_duration_s:.1f}s)")
                print("  Skipping this sequence")
                return clip_counter

            print(f"  Sequence duration: {sequence_duration_s:.1f}s, required: {required_duration_s:.1f}s")

            if 't_offset' not in f:
                print("  No t_offset found in HDF5 file, skipping")
                return clip_counter

            t_offset = f['t_offset'][()]
            total_events = ev['t'].shape[0]
            print(f"  Total events in file: {total_events:,}")
            print(f"  Using STREAMING mode (memory-efficient)")

            # Calculate clip time windows
            sequence_start_us = min(image_timestamps)
            sequence_end_us = max(image_timestamps)
            step_size = clip_duration_us // 2
            clip_duration_s = clip_duration_us / 1000000.0

            # Generate clip metadata (time windows only - no event loading yet)
            clip_windows = []
            clip_start = sequence_start_us
            while clip_start + clip_duration_us <= sequence_end_us:
                clip_end = clip_start + clip_duration_us

                # Find which images fall in this clip
                image_indices_in_clip = []
                for img_idx, img_ts in enumerate(image_timestamps):
                    if clip_start <= img_ts < clip_end:
                        image_indices_in_clip.append(img_idx)

                clip_windows.append({
                    'time_start': clip_start,
                    'time_end': clip_end,
                    'duration_ms': clip_duration_s * 1000.0,
                    'image_timestamp': clip_end,
                    'image_idx': image_indices_in_clip[0] if image_indices_in_clip else len(clip_windows),
                    'image_indices_in_clip': image_indices_in_clip,
                })

                clip_start += step_size

            print(f"  Planned {len(clip_windows)} clips with 50% overlap")

            # Get image directory info
            image_dir = os.path.join(seq_path, 'images', 'left', 'rectified')
            img_files = []
            if os.path.exists(image_dir):
                img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

            # Process clips one at a time (STREAMING)
            from tqdm import tqdm
            seq_dir = os.path.join(output_npy_dir, seq_name)
            os.makedirs(seq_dir, exist_ok=True)
            seq_video_dir = os.path.join(output_video_dir, seq_name)
            os.makedirs(seq_video_dir, exist_ok=True)

            clips_saved = 0
            for clip_idx, clip_window in enumerate(tqdm(clip_windows, desc="    Clips", unit="clip", leave=False)):
                # Find event indices for this clip's time window
                start_idx, end_idx = find_event_indices_for_time_range(
                    f, t_offset, clip_window['time_start'], clip_window['time_end']
                )

                if start_idx is None or end_idx is None or start_idx >= end_idx:
                    continue  # No events in this window

                num_events = end_idx - start_idx
                if num_events == 0:
                    continue

                # Load ONLY this clip's events (memory efficient!)
                clip_data = {
                    'p': ev['p'][start_idx:end_idx],
                    't': ev['t'][start_idx:end_idx] + t_offset,  # Apply offset
                    'x': ev['x'][start_idx:end_idx],
                    'y': ev['y'][start_idx:end_idx],
                    'time_start': clip_window['time_start'],
                    'time_end': clip_window['time_end'],
                    'duration_ms': clip_window['duration_ms'],
                    'image_timestamp': clip_window['image_timestamp'],
                    'image_idx': clip_window['image_idx'],
                    'image_indices_in_clip': clip_window['image_indices_in_clip'],
                    'num_events': num_events
                }

                # Save clip
                clip_filename = f"{clip_idx:06d}.npy"
                npy_path = os.path.join(seq_dir, clip_filename)
                np.save(npy_path, clip_data)

                # Copy associated images
                video_clip_dir = os.path.join(seq_video_dir, f"{clip_idx:06d}")
                os.makedirs(video_clip_dir, exist_ok=True)

                for img_idx in clip_window['image_indices_in_clip']:
                    if img_idx < len(image_timestamps) and img_idx < len(img_files):
                        original_filename = img_files[img_idx]
                        src_img_path = os.path.join(image_dir, original_filename)
                        dst_img_path = os.path.join(video_clip_dir, original_filename)
                        if os.path.exists(src_img_path):
                            shutil.copy2(src_img_path, dst_img_path)

                clips_saved += 1
                clip_counter += 1

                # Clean up per-clip memory
                del clip_data
                if clip_idx % 100 == 0:
                    gc.collect()

            print(f"  Saved {clips_saved} clips for {seq_name}")
            gc.collect()

    except Exception as e:
        print(f"  Error processing sequence: {e}")
        import traceback
        traceback.print_exc()
        return clip_counter

    return clip_counter

def main():
    """Main function to build the training dataset."""
    parser = argparse.ArgumentParser(description="Build EventGPT DSEC TRAINING dataset")
    parser.add_argument("--sequence", type=str, default=None,
                       help="Process only this sequence (e.g., 'zurich_city_00_a')")
    parser.add_argument("--durations", type=str, default=DEFAULT_DURATIONS,
                       help=f"Comma-separated list of clip durations. Default: {DEFAULT_DURATIONS}")
    args = parser.parse_args()

    # Parse durations
    try:
        clip_durations_us = []
        for d in args.durations.split(','):
            d = d.strip()
            if d.lower().endswith('us'):
                value = float(d[:-2])
            elif d.lower().endswith('ms'):
                value = float(d[:-2]) * 1000
            elif d.lower().endswith('s'):
                value = float(d[:-1]) * 1000000
            else:
                value = float(d) * 1000
            clip_durations_us.append(int(value))
    except ValueError as e:
        print(f"Error: Invalid durations format: {args.durations}")
        print(f"Expected format: --durations '1s,5000ms,1000000us' (with units)")
        print(f"Error details: {e}")
        return

    start_time = time.time()
    print("Starting TRAINING dataset creation...")
    print(f"Clip durations: {args.durations}")
    print(f"Input: {DSEC_ROOTS}")
    print(f"Output base: {OUTPUT_DIR_BASE}/my_egpt_dsec_train_<duration>/")
    if args.sequence:
        print(f"Processing only sequence: {args.sequence}")

    # Find DSEC sequences
    dsec_locs = ['zurich_city', 'interlaken', 'thun']
    sequences = []
    for root in DSEC_ROOTS:
        if os.path.exists(root):
            for name in os.listdir(root):
                seq_path = os.path.join(root, name)
                if os.path.isdir(seq_path) and any(name.startswith(loc + '_') for loc in dsec_locs):
                    sequences.append(seq_path)

    print(f"Found {len(sequences)} DSEC training sequences")

    if args.sequence:
        sequences = [seq for seq in sequences if os.path.basename(seq) == args.sequence]
        print(f"Filtered to {len(sequences)} sequences matching '{args.sequence}'")
        if len(sequences) == 0:
            print(f"No sequence found matching '{args.sequence}'")
            return

    # Process each clip duration
    for duration_idx, duration_us in enumerate(clip_durations_us):
        duration_start_time = time.time()
        duration_s = duration_us / 1000000.0
        output_dir = get_output_dir(duration_us)
        print(f"\n{'='*50}")
        print(f"Processing clips of {format_duration(duration_us)} duration ({duration_idx+1}/{len(clip_durations_us)})")
        print(f"{'='*50}")

        if os.path.exists(output_dir):
            print(f"Cleaning output directory: {output_dir}")
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        event_npy_dir = os.path.join(output_dir, 'event_npy')
        video_dir = os.path.join(output_dir, 'video')
        os.makedirs(event_npy_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        clip_counter = 0

        from tqdm import tqdm
        print(f"Processing {len(sequences)} DSEC sequences for {format_duration(duration_us)} clips...")

        for i, seq_path in enumerate(tqdm(sequences, desc="Sequences", unit="seq")):
            seq_name = os.path.basename(seq_path)
            mem_before = get_memory_usage()
            tqdm.write(f"Processing sequence {i+1}/{len(sequences)}: {seq_name} (Memory: {mem_before:.1f}MB)")
            clip_counter = process_sequence(seq_path, clip_counter, event_npy_dir, video_dir, duration_us)
            mem_after = get_memory_usage()
            tqdm.write(f"Sequence complete. Total clips: {clip_counter}, Memory: {mem_after:.1f}MB (+{mem_after-mem_before:.1f}MB)")

        duration_end_time = time.time()
        duration_elapsed = duration_end_time - duration_start_time
        duration_time_str = time.strftime("%H:%M:%S", time.gmtime(duration_elapsed))

        print(f"Created {clip_counter} clips for {format_duration(duration_us)} duration")
        print(f"Duration processing time: {duration_time_str}")
        print(f"Event NPY files in {event_npy_dir}")
        print(f"Video files in {video_dir}")

    end_time = time.time()
    total_duration = end_time - start_time
    completion_time_str = time.strftime("%H:%M:%S", time.gmtime(total_duration))

    print(f"\n{'='*50}")
    print("All durations processed!")
    print(f"Total processing time: {completion_time_str}")
    print(f"Output directories: {[get_output_dir(d) for d in clip_durations_us]}")
    print("\nNext steps:")
    print("1. Run generate_json.py for each output directory")
    print("2. Run preprocess_event_images.py for each output directory")

if __name__ == "__main__":
    main()
