#!/usr/bin/env python3
"""
Script to create empty EventGPT DSEC dataset JSON structure.

This script creates a HuggingFace-compatible dataset skeleton by:
1. Scanning event .npy files from dataset_dir/event_npy/
2. Including corresponding video data paths from dataset_dir/video/ OR dataset_dir/mp4/
3. Creating JSON entries with empty question and gpt fields

Dataset Structure:
    dataset_dir/
    ├── event_npy/          # Event data (.npy files)
    │   └── interlaken_00_a/
    │       └── 000000.npy
    ├── video/              # Video data (folder of images)
    │   └── interlaken_00_a/
    │       └── 000000/     # Folder containing image frames
    │           ├── 000000.png
    │           ├── 000001.png
    │           └── ...
    └── mp4/                # MP4 video files (for Video-LLaVA)
        └── interlaken_00_a/
            └── 000000.mp4   # MP4 video file

Output:
    JSON saved to dataset_dir/EventGPT_Instruction_Subset.json

Dataset Format:
    {
      "id": "uuid",
      "split": "my_egpt_dsec_seq_5s",
      "event_data": "interlaken_00_a/000000.npy",
      "video_data": "interlaken_00_a/000000",
      "video_type": "images"  # or "mp4"
      "conversations": [
        {"from": "human", "value": ""},
        {"from": "gpt", "value": ""}
      ]
    }

Usage:
    # Use video folders (default)
    python generate_json.py --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

    # Use MP4 files (for Video-LLaVA)
    python generate_json.py --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s --use_mp4

    # Process specific sequence
    python generate_json.py --sequence interlaken_00_a --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

    # Limit samples for testing
    python generate_json.py --max_samples 100 --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s
"""

import os
import json
import argparse
import uuid
from tqdm import tqdm


def scan_event_files(event_npy_dir, sequence_filter=None):
    """Scan event_npy directory and return list of event file paths."""
    event_files = []

    if not os.path.exists(event_npy_dir):
        print(f"Event directory not found: {event_npy_dir}")
        return event_files

    # Walk through all subdirectories (sequences)
    for root, dirs, files in os.walk(event_npy_dir):
        for file in files:
            if file.endswith('.npy'):
                rel_path = os.path.relpath(os.path.join(root, file), event_npy_dir)
                seq_name = os.path.dirname(rel_path)

                # Filter by sequence if specified
                if sequence_filter and seq_name != sequence_filter:
                    continue

                event_files.append(rel_path)

    return sorted(event_files)


def get_video_path(event_file, use_mp4=False):
    """Get the corresponding video path for an event file."""
    # event_file is like "interlaken_00_a/000000.npy"
    # video is at "interlaken_00_a/000000" (no extension) for images
    # or "interlaken_00_a/000000.mp4" for MP4 files
    base_path = event_file.replace('.npy', '')
    if use_mp4:
        return base_path + ".mp4"
    return base_path


def create_dataset_structure(event_files, video_dir, mp4_dir, split_name):
    """Create the dataset structure with empty question and gpt fields."""
    dataset = []
    missing_videos = 0
    missing_mp4s = 0

    for event_file in tqdm(event_files, desc="🏗️  Creating entries", unit="entry"):
        video_path = get_video_path(event_file, use_mp4=False)
        mp4_path = get_video_path(event_file, use_mp4=True)

        # Verify video folder exists
        full_video_path = os.path.join(video_dir, video_path)
        if not os.path.isdir(full_video_path):
            missing_videos += 1
            if missing_videos <= 5:  # Only show first 5 warnings
                print(f"⚠️  Warning: Video folder not found: {full_video_path}")

        # Verify MP4 file exists
        full_mp4_path = os.path.join(mp4_dir, mp4_path)
        if not os.path.isfile(full_mp4_path):
            missing_mp4s += 1
            if missing_mp4s <= 5:  # Only show first 5 warnings
                print(f"⚠️  Warning: MP4 file not found: {full_mp4_path}")

        dataset.append({
            "id": str(uuid.uuid4()),
            "split": split_name,
            "event_data": event_file,
            "video_data": video_path,
            "mp4_data": mp4_path,
            "conversations": [
                {
                    "from": "human",
                    "value": ""  # Empty question
                },
                {
                    "from": "gpt",
                    "value": ""  # Empty answer
                }
            ]
        })

    if missing_videos > 5:
        print(f"⚠️  ... and {missing_videos - 5} more missing video folders")
    if missing_mp4s > 5:
        print(f"⚠️  ... and {missing_mp4s - 5} more missing MP4 files")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate empty JSON dataset structure for EventGPT")
    parser.add_argument("--dataset_dir", type=str, default="/mnt/hdd/data/my_egpt_dsec_seq_5s",
                        help="Path to the dataset directory")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Process only this sequence (e.g., 'thun_01_a')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for testing")
    args = parser.parse_args()

    event_npy_dir = os.path.join(args.dataset_dir, "event_npy")
    video_dir = os.path.join(args.dataset_dir, "video")
    mp4_dir = os.path.join(args.dataset_dir, "mp4")
    split_name = os.path.basename(args.dataset_dir.rstrip('/'))  # e.g., "my_egpt_dsec_seq_5s"
    
    # Determine output filename based on sequence
    if args.sequence:
        output_filename = f"EventGPT_Instruction_Subset_{args.sequence}.json"
    else:
        output_filename = "EventGPT_Instruction_Subset.json"
        
    output_json_path = os.path.join(args.dataset_dir, output_filename)

    # Clear output file from previous run if it exists
    if os.path.exists(output_json_path):
        print(f"🗑️  Removing previous output file: {output_json_path}")
        os.remove(output_json_path)

    # Scan event files
    print(f"🔍 Scanning event files in {event_npy_dir}")
    event_files = scan_event_files(event_npy_dir, args.sequence)
    print(f"📁 Found {len(event_files)} event files")

    if args.max_samples:
        event_files = event_files[:args.max_samples]
        print(f"Limited to {len(event_files)} event files for testing")

    # Create dataset structure with empty fields
    print(f"🎬 Looking for video folders in {video_dir}")
    print(f"🎬 Looking for MP4 files in {mp4_dir}")
    print(f"📛 Split name: {split_name}")
    print("🏗️  Creating dataset structure...")
    dataset = create_dataset_structure(event_files, video_dir, mp4_dir, split_name)
    print(f"📊 Created dataset with {len(dataset)} entries")

    # Save dataset
    print(f"💾 Saving dataset to {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("🎉 Done!")


if __name__ == "__main__":
    main()

