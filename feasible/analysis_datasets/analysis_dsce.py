"""
Analysis Script for DSEC Dataset

Author: Adapted from EventGPT analysis script
Date: 2025-12-03

Description:
    This script analyzes the DSEC (Dynamic Stereo Event Camera) dataset, providing comprehensive statistics
    and visualizations based on available metadata and image timestamps:

    1. Dataset structure and sequence information
    2. Duration calculation from image timestamps.txt files
    3. Metadata analysis from HDF5 files:
       - File sizes for left/right cameras per sequence
       - Event counts estimated from HDF5 dataset shapes
       - Event density calculations using image-based duration
       - Sequence-by-sequence comparisons

    4. Comprehensive visualizations:
       - Figure: Event Counts Comparison (left vs right camera per sequence)
       - Figure: File Sizes Comparison (left vs right camera per sequence)
       - Figure: Size vs Count Correlation (file size vs event count relationship)
       - Figure: Duration Distribution (from image timestamps)
       - Figure: Event Density Distribution (events/ms using image duration)
       - Figure: Event Count Distribution per camera

    Note: While HDF5 files use compression requiring special plugins, duration is calculated from
    image timestamps, enabling full temporal analysis and density calculations.

    The results are saved as figures and text files in the configured output directory.
"""

import os
import sys
import json
from typing import Optional, Sequence, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm


# Ensure project root is on sys.path so we can import any shared utilities
# ROOT should be the project root (EventGPT/)
# __file__ -> feasible/analysis_datasets/analysis_dsce.py
# dirname -> feasible/analysis_datasets
# dirname -> feasible
# dirname -> EventGPT
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# DSEC dataset path
# DATA_DIR = "/mnt/hdd/data/DSEC/test"
DATA_DIR = "/mnt/hdd/data/DSEC/train"
# FIG_DIR = "/home/ps/Documents/code/EventGPT/feasible/egpt_dataset/results_dsce_test"
FIG_DIR = "/home/ps/Documents/code/EventGPT/feasible/analysis_datasets/results_dsce_train"
os.makedirs(FIG_DIR, exist_ok=True)


def _print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")
    print("🔍 Starting analysis... ✨")


def _compute_duration_stats(durations: Sequence[float]) -> None:
    arr = np.asarray(durations, dtype=float)
    _print_header("Duration statistics")
    print(f"Count: {arr.size}")
    print(f"Min:   {arr.min():.4f}")
    print(f"Max:   {arr.max():.4f}")
    print(f"Mean:  {arr.mean():.4f}")
    print(f"Median:{np.median(arr):.4f}")
    for q in (0.25, 0.5, 0.75, 0.9, 0.95):
        print(f"Quantile {q:.2f}: {np.quantile(arr, q):.4f}")


def _analyze_sequence_events(sequence_name: str, sequence_path: str) -> Dict[str, Any]:
    """Analyze events for a single DSEC sequence."""
    print(f"\n⏱️ Analyzing events for sequence '{sequence_name}'...")

    # Check for left and right event files
    left_events_path = os.path.join(sequence_path, "events", "left", "events.h5")
    right_events_path = os.path.join(sequence_path, "events", "right", "events.h5")

    # Check for image timestamps (for duration calculation)
    images_timestamps_path = os.path.join(sequence_path, "images", "timestamps.txt")

    sequence_metrics = {
        "sequence_name": sequence_name,
        "left_events": {},
        "right_events": {},
        "combined_events": {},
        "file_info": {},
        "image_info": {}
    }

    # Calculate duration from image timestamps
    duration_ms = 0
    if os.path.exists(images_timestamps_path):
        try:
            timestamps = []
            with open(images_timestamps_path, 'r') as f:
                for line in f:
                    timestamps.append(int(line.strip()))

            if len(timestamps) > 1:
                # Timestamps are in microseconds, convert to milliseconds
                duration_us = timestamps[-1] - timestamps[0]
                duration_ms = duration_us / 1000.0
                sequence_metrics["image_info"]["duration_ms"] = duration_ms
                sequence_metrics["image_info"]["frame_count"] = len(timestamps)
                sequence_metrics["image_info"]["first_timestamp_us"] = timestamps[0]
                sequence_metrics["image_info"]["last_timestamp_us"] = timestamps[-1]
                print(f"  📷 Image sequence: {len(timestamps)} frames, {duration_ms:.2f} ms duration")
        except Exception as e:
            print(f"❌ Error reading timestamps: {e}")

    # Get basic file information
    if os.path.exists(left_events_path):
        left_size = os.path.getsize(left_events_path)
        sequence_metrics["file_info"]["left_events_file_size_mb"] = left_size / (1024 * 1024)
        print(f"  📄 Left events file: {left_size / (1024*1024):.1f} MB")
    else:
        print(f"❌ Left events file not found: {left_events_path}")

    if os.path.exists(right_events_path):
        right_size = os.path.getsize(right_events_path)
        sequence_metrics["file_info"]["right_events_file_size_mb"] = right_size / (1024 * 1024)
        print(f"  📄 Right events file: {right_size / (1024*1024):.1f} MB")
    else:
        print(f"❌ Right events file not found: {right_events_path}")

    # Try to get HDF5 metadata (what we can read without plugins)
    left_metadata = _get_hdf5_metadata(left_events_path, "left")
    right_metadata = _get_hdf5_metadata(right_events_path, "right")

    sequence_metrics["left_events"].update(left_metadata)
    sequence_metrics["right_events"].update(right_metadata)

    # Calculate density using image-based duration
    left_count = sequence_metrics["left_events"].get("event_count", 0)
    right_count = sequence_metrics["right_events"].get("event_count", 0)

    # Use image duration for density calculations if available
    if duration_ms > 0:
        left_density = left_count / duration_ms if left_count > 0 else 0.0
        right_density = right_count / duration_ms if right_count > 0 else 0.0

        sequence_metrics["left_events"]["duration_ms"] = duration_ms
        sequence_metrics["left_events"]["density_ev_per_ms"] = left_density

        sequence_metrics["right_events"]["duration_ms"] = duration_ms
        sequence_metrics["right_events"]["density_ev_per_ms"] = right_density

    combined_count = left_count + right_count
    combined_density = combined_count / duration_ms if duration_ms > 0 and combined_count > 0 else 0.0

    sequence_metrics["combined_events"] = {
        "duration_ms": duration_ms,
        "event_count": combined_count,
        "density_ev_per_ms": combined_density
    }

    return sequence_metrics


def _get_hdf5_metadata(events_path: str, camera_label: str) -> Dict[str, Any]:
    """Get HDF5 metadata that doesn't require reading compressed data."""
    if not os.path.exists(events_path):
        return {"error": "File not found"}

    try:
        with h5py.File(events_path, 'r') as f:
            metadata = {}

            # Get basic file structure
            metadata["hdf5_keys"] = list(f.keys())

            if 'events' in f:
                events_group = f['events']
                metadata["events_keys"] = list(events_group.keys())

                # Get dataset shapes and dtypes (metadata only)
                for key in events_group.keys():
                    dataset = events_group[key]
                    metadata[f"{key}_shape"] = dataset.shape
                    metadata[f"{key}_dtype"] = str(dataset.dtype)

                # Try to estimate event count from shape
                if 't' in events_group:
                    metadata["event_count"] = int(events_group['t'].shape[0])
                    print(f"  📊 {camera_label.capitalize()} camera: ~{metadata['event_count']:,} events (estimated from metadata)")
                else:
                    metadata["event_count"] = 0
            else:
                metadata["error"] = "No 'events' group found"

            return metadata

    except Exception as e:
        print(f"❌ Error reading {camera_label} metadata: {e}")
        return {"error": str(e)}


def _analyze_all_sequences() -> Dict[str, Any]:
    """Analyze all sequences in the DSEC dataset."""
    print(f"🔍 Scanning DSEC dataset at: {DATA_DIR}")

    # Get all sequence directories (exclude zip files and other files)
    sequences = []
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path) and not item.endswith('.zip'):
            sequences.append(item)

    print(f"📂 Found {len(sequences)} sequences: {sequences[:5]}...")

    all_metrics = {}
    # Separate collections for left, right, and combined
    left_durations = []
    left_event_counts = []
    left_event_densities = []
    right_durations = []
    right_event_counts = []
    right_event_densities = []
    combined_durations = []
    combined_event_counts = []
    combined_event_densities = []
    detailed_metrics = []

    with tqdm(total=len(sequences), desc="🚀 Processing sequences", unit="sequence") as pbar:
        for sequence_name in sequences:
            sequence_path = os.path.join(DATA_DIR, sequence_name)
            try:
                metrics = _analyze_sequence_events(sequence_name, sequence_path)
                all_metrics[sequence_name] = metrics

                # Collect metrics for left camera
                left = metrics.get("left_events", {})
                if "duration_ms" in left and left["duration_ms"] > 0:
                    left_durations.append(left["duration_ms"])
                    left_event_counts.append(left.get("event_count", 0))
                    left_event_densities.append(left.get("density_ev_per_ms", 0.0))

                # Collect metrics for right camera
                right = metrics.get("right_events", {})
                if "duration_ms" in right and right["duration_ms"] > 0:
                    right_durations.append(right["duration_ms"])
                    right_event_counts.append(right.get("event_count", 0))
                    right_event_densities.append(right.get("density_ev_per_ms", 0.0))

                # Collect metrics for combined
                combined = metrics.get("combined_events", {})
                if "duration_ms" in combined and combined["duration_ms"] > 0:
                    combined_durations.append(combined["duration_ms"])
                    combined_event_counts.append(combined.get("event_count", 0))
                    combined_event_densities.append(combined.get("density_ev_per_ms", 0.0))

                detailed_metrics.append(metrics)

            except Exception as e:
                print(f"❌ Error analyzing sequence {sequence_name}: {e}")
                all_metrics[sequence_name] = {"error": str(e)}

            pbar.update(1)

    # Save detailed metrics to JSON
    detailed_json_file = os.path.join(FIG_DIR, "detailed_sequence_metrics.json")
    with open(detailed_json_file, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"💾 Saved detailed metrics JSON to: {detailed_json_file}")

    return (all_metrics,
            left_durations, left_event_counts, left_event_densities,
            right_durations, right_event_counts, right_event_densities,
            combined_durations, combined_event_counts, combined_event_densities)


def _create_visualizations(all_metrics: Dict[str, Any],
                          left_durations: Sequence[float], left_event_counts: Sequence[int], left_event_densities: Sequence[float],
                          right_durations: Sequence[float], right_event_counts: Sequence[int], right_event_densities: Sequence[float],
                          combined_durations: Sequence[float], combined_event_counts: Sequence[int], combined_event_densities: Sequence[float]) -> None:
    """Create the three main visualization figures for left, right, and combined cameras."""

    # Create metadata-based visualizations (available even without compressed data)
    _create_metadata_visualizations(all_metrics)

    # Create visualizations for left camera (if duration data available)
    if left_durations:
        _create_camera_visualizations(left_durations, left_event_counts, left_event_densities, "Left Camera")
    else:
        print("⚠️ No valid left camera duration data found for temporal visualizations.")

    # Create visualizations for right camera (if duration data available)
    if right_durations:
        _create_camera_visualizations(right_durations, right_event_counts, right_event_densities, "Right Camera")
    else:
        print("⚠️ No valid right camera duration data found for temporal visualizations.")

    # Create visualizations for combined (if duration data available)
    if combined_durations:
        _create_camera_visualizations(combined_durations, combined_event_counts, combined_event_densities, "Combined Cameras")
    else:
        print("⚠️ No valid combined duration data found for temporal visualizations.")


def _create_metadata_visualizations(all_metrics: Dict[str, Any]) -> None:
    """Create visualizations based on metadata that doesn't require reading compressed data."""
    print("\n📊 Creating metadata-based visualizations...")

    # Extract data for all sequences
    sequence_names = []
    left_event_counts = []
    right_event_counts = []
    combined_event_counts = []
    left_file_sizes = []
    right_file_sizes = []
    combined_file_sizes = []

    for seq_name, metrics in all_metrics.items():
        sequence_names.append(seq_name)

        left_events = metrics.get("left_events", {})
        right_events = metrics.get("right_events", {})
        file_info = metrics.get("file_info", {})

        # Event counts
        left_count = left_events.get("event_count", 0)
        right_count = right_events.get("event_count", 0)

        left_event_counts.append(left_count)
        right_event_counts.append(right_count)
        combined_event_counts.append(left_count + right_count)

        # File sizes
        left_size = file_info.get("left_events_file_size_mb", 0)
        right_size = file_info.get("right_events_file_size_mb", 0)

        left_file_sizes.append(left_size)
        right_file_sizes.append(right_size)
        combined_file_sizes.append(left_size + right_size)

    # Figure 1: Event counts per sequence (left vs right)
    plt.figure(figsize=(15, 8))
    x = np.arange(len(sequence_names))
    width = 0.35

    plt.bar(x - width/2, left_event_counts, width, label='Left Camera', alpha=0.7, color='blue')
    plt.bar(x + width/2, right_event_counts, width, label='Right Camera', alpha=0.7, color='red')

    plt.xlabel('Sequence')
    plt.ylabel('Event Count')
    plt.title('Event Counts per Sequence (Left vs Right Camera)')
    plt.xticks(x, sequence_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization of large range
    plt.tight_layout()

    fname = os.path.join(FIG_DIR, "figure_event_counts_comparison.png")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"📊 Saved Event Counts Comparison to: {fname}")

    # Figure 2: File sizes per sequence
    plt.figure(figsize=(15, 8))
    plt.bar(x - width/2, left_file_sizes, width, label='Left Camera', alpha=0.7, color='lightblue')
    plt.bar(x + width/2, right_file_sizes, width, label='Right Camera', alpha=0.7, color='lightcoral')

    plt.xlabel('Sequence')
    plt.ylabel('File Size (MB)')
    plt.title('File Sizes per Sequence (Left vs Right Camera)')
    plt.xticks(x, sequence_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(FIG_DIR, "figure_file_sizes_comparison.png")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"📊 Saved File Sizes Comparison to: {fname}")

    # Figure 3: Event count vs file size correlation
    plt.figure(figsize=(12, 8))

    # Left camera
    plt.scatter(left_file_sizes, left_event_counts, alpha=0.7, color='blue', label='Left Camera', s=100)

    # Right camera
    plt.scatter(right_file_sizes, right_event_counts, alpha=0.7, color='red', label='Right Camera', s=100)

    # Add sequence labels
    for i, seq_name in enumerate(sequence_names):
        if left_file_sizes[i] > 0:
            plt.annotate(seq_name, (left_file_sizes[i], left_event_counts[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        if right_file_sizes[i] > 0:
            plt.annotate(seq_name, (right_file_sizes[i], right_event_counts[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    plt.xlabel('File Size (MB)')
    plt.ylabel('Event Count')
    plt.title('Event Count vs File Size Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(FIG_DIR, "figure_size_vs_count_correlation.png")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"📊 Saved Size vs Count Correlation to: {fname}")

    print("✅ Metadata visualizations completed!")


def _create_camera_visualizations(durations: Sequence[float], event_counts: Sequence[int],
                                event_densities: Sequence[float], camera_label: str) -> None:
    """Create the three main visualization figures for a specific camera."""

    if not durations:
        return

    # --- Figure 1: Durations ---
    _print_header(f"Duration Stats ({camera_label}) - ms")
    _compute_duration_stats(durations)

    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, edgecolor='black', alpha=0.7)
    plt.title(f"Event Duration Distribution ({camera_label})")
    plt.xlabel("Duration (ms)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    camera_suffix = camera_label.lower().replace(" ", "_")
    fname = os.path.join(FIG_DIR, f"figure1_duration_{camera_suffix}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"📊 Saved Figure 1 (Duration - {camera_label}) to: {fname}")

    # --- Figure 2: Event Numbers (Counts) ---
    arr_counts = np.asarray(event_counts)
    _print_header(f"Event Counts Stats ({camera_label})")
    print(f"Mean: {arr_counts.mean():.2f}, Max: {arr_counts.max()}, Min: {arr_counts.min()}")

    plt.figure(figsize=(10, 6))
    plt.hist(event_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.title(f"Event Counts Distribution ({camera_label})")
    plt.xlabel("Number of Events")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    fname = os.path.join(FIG_DIR, f"figure2_counts_{camera_suffix}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"📊 Saved Figure 2 (Counts - {camera_label}) to: {fname}")

    # --- Figure 3: Event Density (events/ms) ---
    arr_dens = np.asarray(event_densities)
    _print_header(f"Event Density Stats ({camera_label}) - ev/ms")
    print(f"Mean: {arr_dens.mean():.2f}, Max: {arr_dens.max():.2f}, Min: {arr_dens.min():.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(event_densities, bins=50, edgecolor='black', alpha=0.7)
    plt.title(f"Event Density Distribution ({camera_label})")
    plt.xlabel("Density (events/ms)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    fname = os.path.join(FIG_DIR, f"figure3_density_{camera_suffix}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"📊 Saved Figure 3 (Density - {camera_label}) to: {fname}")


def _save_summary_stats(all_metrics: Dict[str, Any],
                       left_durations: Sequence[float], left_event_counts: Sequence[int], left_event_densities: Sequence[float],
                       right_durations: Sequence[float], right_event_counts: Sequence[int], right_event_densities: Sequence[float],
                       combined_durations: Sequence[float], combined_event_counts: Sequence[int], combined_event_densities: Sequence[float]) -> None:
    """Save summary statistics to text file."""

    summary_file = os.path.join(FIG_DIR, "dsce_summary.txt")

    with open(summary_file, 'w') as f:
        f.write("DSEC Dataset Analysis Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Dataset path: {DATA_DIR}\n")
        f.write(f"Number of sequences analyzed: {len(all_metrics)}\n")
        f.write(f"Figures directory: {FIG_DIR}\n\n")

        # Overall statistics for each camera (based on metadata)
        f.write("Overall Statistics:\n\n")

        # Collect metadata-based statistics
        left_total_events = 0
        left_total_size = 0
        left_sequences = 0
        right_total_events = 0
        right_total_size = 0
        right_sequences = 0

        for seq_metrics in all_metrics.values():
            left_events = seq_metrics.get("left_events", {})
            right_events = seq_metrics.get("right_events", {})
            file_info = seq_metrics.get("file_info", {})

            if "event_count" in left_events:
                left_total_events += left_events["event_count"]
                left_sequences += 1
            if "event_count" in right_events:
                right_total_events += right_events["event_count"]
                right_sequences += 1

            left_total_size += file_info.get("left_events_file_size_mb", 0)
            right_total_size += file_info.get("right_events_file_size_mb", 0)

        f.write("Dataset Overview:\n")
        f.write(f"  Total sequences: {len(all_metrics)}\n")
        f.write(f"  Sequences with left camera data: {left_sequences}\n")
        f.write(f"  Sequences with right camera data: {right_sequences}\n\n")

        if left_sequences > 0:
            f.write("Left Camera Statistics:\n")
            f.write(f"  Total events (estimated): {left_total_events:,}\n")
            f.write(f"  Average events per sequence: {left_total_events/left_sequences:,.0f}\n")
            f.write(f"  Total file size: {left_total_size:.1f} MB\n")
            f.write(f"  Average file size per sequence: {left_total_size/left_sequences:.1f} MB\n\n")

        if right_sequences > 0:
            f.write("Right Camera Statistics:\n")
            f.write(f"  Total events (estimated): {right_total_events:,}\n")
            f.write(f"  Average events per sequence: {right_total_events/right_sequences:,.0f}\n")
            f.write(f"  Total file size: {right_total_size:.1f} MB\n")
            f.write(f"  Average file size per sequence: {right_total_size/right_sequences:.1f} MB\n\n")

        total_events = left_total_events + right_total_events
        total_size = left_total_size + right_total_size
        f.write("Combined Statistics:\n")
        f.write(f"  Total events (estimated): {total_events:,}\n")
        f.write(f"  Total file size: {total_size:.1f} MB\n\n")

        # Note about data access limitations
        f.write("Note: Duration is calculated from image timestamps, enabling full temporal analysis.\n")
        f.write("Event counts are estimated from HDF5 metadata. Density = events/duration.\n")
        f.write("Full temporal statistics are now available despite HDF5 compression limitations.\n\n")

        # Per-sequence summary
        f.write("Per-Sequence Summary:\n")
        f.write("-" * 120 + "\n")
        f.write("<25")
        f.write("-" * 120 + "\n")

        for seq_name, metrics in all_metrics.items():
            left_events = metrics.get("left_events", {})
            right_events = metrics.get("right_events", {})
            file_info = metrics.get("file_info", {})

            left_count = left_events.get("event_count", 0)
            right_count = right_events.get("event_count", 0)
            total_count = left_count + right_count

            left_size = file_info.get("left_events_file_size_mb", 0)
            right_size = file_info.get("right_events_file_size_mb", 0)
            total_size = left_size + right_size

            f.write("<25")

        f.write("\n")
        f.write("Analysis completed successfully!\n")

    print(f"📝 Saved summary to: {summary_file}")


def main() -> None:
    """Analyze the DSEC dataset."""
    import time
    start_time = time.time()

    # Redirect stdout to a file + console
    log_file = os.path.join(FIG_DIR, "analysis_summary_dsce.txt")
    class Tee(object):
        def __init__(self, name, mode):
            self.file = open(name, mode)
            self.stdout = sys.stdout
            sys.stdout = self

        def __del__(self):
            sys.stdout = self.stdout
            self.file.close()

        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    # Only tee if we haven't already (to avoid double wrapping in interactive sessions)
    if not isinstance(sys.stdout, Tee):
        sys.stdout = Tee(log_file, "w")

    _print_header("🎯 DSEC Dataset Overview")
    print(f"Analysis results saved to: {log_file}")
    print(f"Dataset path: {DATA_DIR}")
    print(f"Figures directory: {FIG_DIR}")

    # Analyze all sequences
    (all_metrics,
     left_durations, left_event_counts, left_event_densities,
     right_durations, right_event_counts, right_event_densities,
     combined_durations, combined_event_counts, combined_event_densities) = _analyze_all_sequences()

    # Create visualizations
    _create_visualizations(all_metrics,
                          left_durations, left_event_counts, left_event_densities,
                          right_durations, right_event_counts, right_event_densities,
                          combined_durations, combined_event_counts, combined_event_densities)

    # Save summary
    _save_summary_stats(all_metrics,
                       left_durations, left_event_counts, left_event_densities,
                       right_durations, right_event_counts, right_event_densities,
                       combined_durations, combined_event_counts, combined_event_densities)

    elapsed_time = time.time() - start_time
    print(f"🎉 Analysis completed in {elapsed_time:.1f} seconds!")
if __name__ == "__main__":
    main()
