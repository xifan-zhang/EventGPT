#!/usr/bin/env python3
"""
Script to build EventGPT DSEC dataset from DSEC sequences with multiple time intervals.

This script:
1. Processes DSEC sequences (zurich_city, interlaken, thun)
2. Splits event streams into continuous clips of 1s, 5s, and 10s durations
3. Saves event clips as .npy files (events from that specific time period)
4. Saves corresponding video frames for each clip (images from the same time period)
5. Creates instruction dataset with questions (answers generated separately)

Directory Structure:
- event_npy/sequence_name/clip_id.npy: Event data from continuous time duration clips
- video/sequence_name/clip_id/: Video frames from the same continuous time duration

Next step: Run generate_answers_qwen.py to analyze event visualizations with Qwen model.

NOTE: Requires hdf5plugin for DSEC HDF5 access: pip install hdf5plugin

Usage: python build_my_egpt_dsec_seq_dataset.py [--durations "1,5,10"]
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
        return process.memory_info().rss / 1024 / 1024  # MB
    return 0
# Note: Qwen model imports and loading moved to answer generation script
# This script only creates the dataset structure

# Configuration
DSEC_ROOTS = ['/mnt/hdd/data/DSEC/test']  # Only test data for testing
OUTPUT_DIR_BASE = '/mnt/hdd/data'  # Base directory for datasets
TOP50_QUESTIONS = '/home/ps/Documents/code/EventGPT/feasible/analysis_datasets/results_egpt_dsec_split/dsec_questions_top50.txt'

# Time intervals for clips (in microseconds: 1s, 5s, 10s)
CLIP_DURATIONS_US = [1000000, 5000000, 10000000]  # 1s, 5s, 10s in microseconds

def format_duration(duration_us):
    """Format duration in us to a human-readable string with units."""
    if duration_us % 1000000 == 0:
        return f"{duration_us // 1000000}s"
    elif duration_us % 1000 == 0:
        return f"{duration_us // 1000}ms"
    else:
        return f"{duration_us}us"

def get_output_dir(clip_duration_us):
    """Get output directory for specific clip duration."""
    duration_str = format_duration(clip_duration_us)
    return f'{OUTPUT_DIR_BASE}/my_egpt_dsec_seq_{duration_str}'

def calculate_potential_clips(sequence_duration_us, clip_durations_us):
    """Calculate how many clips of each duration can be created from a sequence.

    This estimation matches the actual sequential overlapping clip creation algorithm:
    - Creates sequential clips with 50% overlap from sequence start
    - Gap between consecutive clips is half the clip duration
    - Only complete clips that fit entirely within sequence duration
    """
    potential_clips = {}

    for duration_us in clip_durations_us:
        if sequence_duration_us < duration_us:
            potential_clips[duration_us] = 0
            continue

        # Use the same logic as split_event_by_time for sequential overlapping clips
        step_size = duration_us // 2  # 50% overlap
        sequence_start_us = 0  # Normalized
        sequence_end_us = sequence_duration_us

        # Count clips using the same algorithm
        clip_count = 0
        clip_start = sequence_start_us
        while clip_start + duration_us <= sequence_end_us:
            clip_count += 1
            clip_start += step_size

        potential_clips[duration_us] = max(1, clip_count)

    return potential_clips

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

def split_event_by_time(event_npy, time_interval=50000, image_timestamps=None, image_paths=None):
    """Split event data into clips of specified duration, one for each image frame."""
    p = event_npy['p']
    t = event_npy['t']
    x = event_npy['x']
    y = event_npy['y']

    # If we have image timestamps, create sequential overlapping clips
    if image_timestamps and len(image_timestamps) > 0:
        clip_duration_s = time_interval / 1000000.0
        print(f"  Creating sequential {clip_duration_s:.1f}s event clips with 50% overlap")

        split_data = []
        total_events_used = 0

        # Calculate sequence start and end times
        sequence_start_us = min(image_timestamps)
        sequence_end_us = max(image_timestamps)

        # Create sequential overlapping clips with 50% overlap
        # Gap between consecutive clips is half the clip duration for optimal temporal coverage
        step_size = time_interval // 2  # 50% of clip duration = gap between consecutive clips

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

                # Move to next clip with 50% overlap (gap = half duration)
                clip_start += step_size
                clip_idx += 1
                pbar.update(1)

        if split_data:
            avg_events_per_clip = total_events_used / len(split_data)
            print(f"  ✓ Created {len(split_data)} sequential {clip_duration_s:.1f}s event clips with 50% overlap")
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

def process_sequence(seq_path, clip_counter, output_npy_dir=None, output_video_dir=None, clip_duration_us=50000):
    """Process a single DSEC sequence, generating event_npy clips and video frames.

    Organization:
    - event_npy/sequence_name/clip_id.npy: Events from a specific continuous time period
    - video/sequence_name/clip_id/: Images from the same continuous time period as the events
    """
    seq_name = os.path.basename(seq_path)
    print(f"🎞️  Processing {seq_name}...")

    if output_npy_dir is None:
        output_npy_dir = os.path.join(os.path.dirname(os.path.dirname(seq_path)), 'my_egpt_dsec_dataset_real', 'event_npy')

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

            # Load image timestamps for alignment
            image_timestamps = load_image_timestamps(seq_path)
            if image_timestamps:
                print(f"  Loaded {len(image_timestamps)} image timestamps for alignment")
                print(f"  Image range: {image_timestamps[0]:,} to {image_timestamps[-1]:,} μs")

                # Check if sequence duration is sufficient for the required clip duration
                sequence_duration_us = image_timestamps[-1] - image_timestamps[0]
                sequence_duration_s = sequence_duration_us / 1000000.0
                required_duration_s = clip_duration_us / 1000000.0

                # Show potential clip counts for all durations
                all_clip_durations = [1000000, 5000000, 10000000]  # 1s, 5s, 10s in microseconds
                potential_clips = calculate_potential_clips(sequence_duration_us, all_clip_durations)
                clip_info = []
                for dur_us, count in potential_clips.items():
                    dur_str = format_duration(dur_us)
                    clip_info.append(f"{dur_str}: {count}")
                print(f"  Potential clips: {', '.join(clip_info)}")

                if sequence_duration_us < clip_duration_us:
                    print(f"  ✗ Sequence duration ({sequence_duration_s:.1f}s) is shorter than required clip duration ({required_duration_s:.1f}s)")
                    print("  Skipping this sequence")
                    return clip_counter

                print(f"  ✓ Sequence duration: {sequence_duration_s:.1f}s, required: {required_duration_s:.1f}s")
            else:
                # For sequences without image timestamps, we cannot reliably determine duration
                # Skip if clip duration is long (> 50ms)
                if clip_duration_us > 50000:  # > 50ms
                    print(f"  ✗ No image timestamps found and clip duration ({clip_duration_us/1000000.0:.1f}s) > 50ms")
                    print("  Cannot determine sequence duration, skipping this sequence")
                    return clip_counter
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

            # Split into clips aligned with image frames or fixed time intervals
            image_dir = os.path.join(seq_path, 'images', 'left', 'rectified')
            image_paths = None
            if os.path.exists(image_dir):
                image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
            clips = split_event_by_time(event_npy, time_interval=clip_duration_us, image_timestamps=image_timestamps, image_paths=image_paths)
            print(f"  Split into {len(clips)} clips")

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

                    # Each event npy file corresponds to exactly one video clip
                    # Both contain data from the same time period: clip['time_start'] to clip['time_end']
                    clip_filename = f"{global_clip_idx:06d}.npy"
                    seq_dir = os.path.join(output_npy_dir, seq_name)
                    os.makedirs(seq_dir, exist_ok=True)
                    npy_path = os.path.join(seq_dir, clip_filename)

                    # Extract event data from this specific time period
                    # Events are filtered to: clip['time_start'] <= event_timestamp < clip['time_end']
                    indices = clip['indices']
                    
                    # Verify events are within the time window (safety check)
                    event_times = event_npy['t'][indices]
                    if len(event_times) > 0:
                        assert np.all(event_times >= clip['time_start']), f"Events before time_start in clip {global_clip_idx}"
                        assert np.all(event_times < clip['time_end']), f"Events after time_end in clip {global_clip_idx}"
                    
                    clip_data = {
                        'p': event_npy['p'][indices],  # Event polarities from this time period
                        't': event_npy['t'][indices],  # Event timestamps from this time period (within time window)
                        'x': event_npy['x'][indices],  # Event x-coordinates from this time period
                        'y': event_npy['y'][indices],  # Event y-coordinates from this time period
                        'time_start': clip['time_start'],  # Start time of this clip (same for events and video)
                        'time_end': clip['time_end'],     # End time of this clip (same for events and video)
                        'duration_ms': clip['duration_ms'], # Duration of this clip
                        'image_timestamp': clip['image_timestamp'],
                        'image_idx': clip['image_idx'],
                        'image_indices_in_clip': clip.get('image_indices_in_clip', []),  # Image indices aligned with this clip
                        'num_events': clip['num_events']   # Number of events in this time period
                    }

                    np.save(npy_path, clip_data)

                    # Save corresponding video frames for the SAME time period as events
                    # Video clip directory name matches the event npy filename (without .npy extension)
                    # This ensures: event_npy/seq_name/000000.npy <-> video/seq_name/000000/
                    # Both contain data from the same time window: clip['time_start'] to clip['time_end']
                    seq_video_dir = os.path.join(output_video_dir, seq_name)
                    os.makedirs(seq_video_dir, exist_ok=True)
                    video_clip_dir = os.path.join(seq_video_dir, clip_filename.replace('.npy', ''))
                    os.makedirs(video_clip_dir, exist_ok=True)

                    # Copy images that fall within the same time window as events
                    # Video frames are aligned with events using the same time boundaries:
                    # - Same time_start and time_end as the event clip
                    # - Images within clip['time_start'] <= img_ts < clip['time_end'] are included
                    # - This ensures perfect 1-to-1 alignment: one event npy <-> one video clip directory
                    image_dir = os.path.join(seq_path, 'images', 'left', 'rectified')
                    if os.path.exists(image_dir) and 'image_indices_in_clip' in clip:
                        # Get sorted list of PNG files to maintain consistent ordering
                        img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
                        # Copy all images that fall within this clip's time window (same as events)
                        for img_idx in clip['image_indices_in_clip']:
                            if img_idx < len(image_timestamps) and img_idx < len(img_files):
                                # Verify image timestamp is within clip time window (same as events)
                                img_ts = image_timestamps[img_idx]
                                assert clip['time_start'] <= img_ts < clip['time_end'], \
                                    f"Image timestamp {img_ts} outside clip time window [{clip['time_start']}, {clip['time_end']})"
                                # Copy with original filename - NO MODIFICATION
                                original_filename = img_files[img_idx]  # e.g., "000027.png"
                                src_img_path = os.path.join(image_dir, original_filename)
                                dst_img_path = os.path.join(video_clip_dir, original_filename)  # Same name
                                shutil.copy2(src_img_path, dst_img_path)

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
    parser.add_argument("--durations", type=str, default="1s,5s,10s",
                       help="Comma-separated list of clip durations with units (s=seconds, ms=milliseconds, us=microseconds). Default: 1s,5s,10s")
    args = parser.parse_args()

    # Parse durations with automatic unit detection (s, ms, us)
    try:
        clip_durations_us = []
        for d in args.durations.split(','):
            d = d.strip()
            if d.lower().endswith('us'):
                # Microseconds
                value = float(d[:-2])
            elif d.lower().endswith('ms'):
                # Milliseconds
                value = float(d[:-2]) * 1000  # Convert to microseconds
            elif d.lower().endswith('s'):
                # Seconds
                value = float(d[:-1]) * 1000000  # Convert to microseconds
            else:
                # Default to milliseconds for backward compatibility
                value = float(d) * 1000  # Assume milliseconds

            clip_durations_us.append(int(value))
    except ValueError as e:
        print(f"Error: Invalid durations format: {args.durations}")
        print("Expected format: --durations '1s,5000ms,1000000us' (with units)")
        print(f"Error details: {e}")
        return

    start_time = time.time()
    print("🚀 Starting dataset creation...")
    # Format durations for display
    duration_display = []
    for i, us in enumerate(clip_durations_us):
        original = args.durations.split(',')[i].strip()
        duration_display.append(original)
    print(f"⏱️  Clip durations: {duration_display}")
    if args.sequence:
        print(f"🎯 Processing only sequence: {args.sequence}")

    # Find DSEC sequences
    dsec_locs = ['zurich_city', 'interlaken', 'thun']
    sequences = []
    for root in DSEC_ROOTS:
        if os.path.exists(root):
            for name in os.listdir(root):
                seq_path = os.path.join(root, name)
                if os.path.isdir(seq_path) and any(name.startswith(loc + '_') for loc in dsec_locs):
                    sequences.append(seq_path)

    print(f"🔍 Found {len(sequences)} DSEC sequences")

    # Filter sequences if specified
    if args.sequence:
        sequences = [seq for seq in sequences if os.path.basename(seq) == args.sequence]
        print(f"Filtered to {len(sequences)} sequences matching '{args.sequence}'")
        if len(sequences) == 0:
            print(f"No sequence found matching '{args.sequence}'")
            return

    # Estimate total processing time
    total_sequences = len(sequences) * len(clip_durations_us)
    # Rough estimate: ~3 minutes per sequence (based on observed processing time)
    estimated_seconds = total_sequences * 180  # 3 minutes per sequence
    estimated_time_str = time.strftime("%H:%M:%S", time.gmtime(estimated_seconds))
    print(f"⏳ Estimated total processing time: {estimated_time_str} ({total_sequences} sequence-duration combinations)")

    # Process each clip duration
    for duration_idx, duration_us in enumerate(clip_durations_us):
        duration_start_time = time.time()
        duration_s = duration_us / 1000000.0  # Use float division
        output_dir = get_output_dir(duration_us)
        print(f"\n{'='*50}")
        print(f"🎬 Processing clips of {duration_s:.1f}s duration ({duration_idx+1}/{len(clip_durations_us)})")
        print(f"{'='*50}")

        # Clean output directory to ensure a fresh run
        if os.path.exists(output_dir):
            print(f"🧹 Cleaning output directory: {output_dir}")
            shutil.rmtree(output_dir)

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        event_npy_dir = os.path.join(output_dir, 'event_npy')
        video_dir = os.path.join(output_dir, 'video')
        os.makedirs(event_npy_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")

        # Process sequences for this duration
        clip_counter = 0

        from tqdm import tqdm
        print(f"🎥 Processing {len(sequences)} DSEC sequences for {duration_s}s clips...")

        for i, seq_path in enumerate(tqdm(sequences, desc="Sequences", unit="seq")):
            seq_name = os.path.basename(seq_path)
            mem_before = get_memory_usage()
            tqdm.write(f"Processing sequence {i+1}/{len(sequences)}: {seq_name} (Memory: {mem_before:.1f}MB)")
            clip_counter = process_sequence(seq_path, clip_counter, event_npy_dir, video_dir, duration_us)
            mem_after = get_memory_usage()
            tqdm.write(f"Sequence complete. Total clips: {clip_counter}, Memory: {mem_after:.1f}MB (+{mem_after-mem_before:.1f}MB)")

        # Show duration processing completion time
        duration_end_time = time.time()
        duration_elapsed = duration_end_time - duration_start_time
        duration_time_str = time.strftime("%H:%M:%S", time.gmtime(duration_elapsed))

        print(f"✅ Created {clip_counter} clips for {duration_s}s duration")
        print(f"⏱️  Duration processing time: {duration_time_str}")
        print(f"💾 Event NPY files in {event_npy_dir}")
        print(f"🎥 Video files in {video_dir}")
        print(f"To generate answers, run: python generate_answers_qwen.py --dataset_dir {output_dir}")

    # Calculate and display completion time
    end_time = time.time()
    total_duration = end_time - start_time
    completion_time_str = time.strftime("%H:%M:%S", time.gmtime(total_duration))

    print(f"\n{'='*50}")
    print("🎉 All durations processed!")
    print(f"⏱️  Total processing time: {completion_time_str}")
    print(f"📁 Output directories: {[get_output_dir(d) for d in clip_durations_us]}")

if __name__ == "__main__":
    main()
