#!/usr/bin/env python3
"""
================================================================================
                    IMU-VIDEO PAIRED DATASET DOWNLOADER
================================================================================

Available Datasets:
1. Ego4D - Requires license (ego4d-data.org) - 3,025 hours
2. MuJo - Fitness dataset with synthetic IMU
3. VIDIMU - 54 subjects, daily activities
4. SensorCaps - 35,960 IMU-caption pairs (LLaSA training data)
5. OpenSQA - 179,727 IMU Q&A pairs (LLaSA training data)

Usage:
  python download_datasets.py --dataset sensorcaps --output_dir /mnt/hdd/data
  python download_datasets.py --dataset opensqa --output_dir /mnt/hdd/data
  python download_datasets.py --dataset llasa_model --output_dir /mnt/hdd/data

================================================================================
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def download_sensorcaps(output_dir: str):
    """Download SensorCaps dataset (35,960 IMU-caption pairs)."""
    print("\n" + "=" * 60)
    print("Downloading SensorCaps Dataset")
    print("=" * 60)
    print("Source: https://huggingface.co/datasets/BASH-Lab/SensorCaps")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "huggingface-cli", "download",
        "BASH-Lab/SensorCaps",
        "--repo-type", "dataset",
        "--local-dir", os.path.join(output_dir, "SensorCaps")
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def download_opensqa(output_dir: str):
    """Download OpenSQA dataset (179,727 IMU Q&A pairs)."""
    print("\n" + "=" * 60)
    print("Downloading OpenSQA Dataset")
    print("=" * 60)
    print("Source: https://huggingface.co/datasets/BASH-Lab/OpenSQA")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "huggingface-cli", "download",
        "BASH-Lab/OpenSQA",
        "--repo-type", "dataset",
        "--local-dir", os.path.join(output_dir, "OpenSQA")
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def download_llasa_model(output_dir: str):
    """Download LLaSA-7B model (~13.5GB)."""
    print("\n" + "=" * 60)
    print("Downloading LLaSA-7B Model")
    print("=" * 60)
    print("Source: https://huggingface.co/BASH-Lab/LLaSA-7B")
    print("Size: ~13.5GB")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "huggingface-cli", "download",
        "BASH-Lab/LLaSA-7B",
        "--local-dir", os.path.join(output_dir, "LLaSA-7B")
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def download_ego4d_info():
    """Print information about downloading Ego4D dataset."""
    print("\n" + "=" * 60)
    print("Ego4D Dataset Information")
    print("=" * 60)
    print("""
Ego4D is the largest egocentric video dataset with IMU data.

To download:
1. Register at https://ego4d-data.org/
2. Accept the license agreement
3. Use the Ego4D CLI:

   pip install ego4d
   ego4d --output_directory=/mnt/hdd/data/ego4d \\
         --datasets full_scale \\
         --modalities imu video

Components with IMU:
- MMG-Ego4D subset (Multimodal Generalization)
- Contains video, audio, and IMU modalities

Size: ~5TB for full dataset (can download subsets)
    """)


def download_mujo_info():
    """Print information about downloading MuJo dataset."""
    print("\n" + "=" * 60)
    print("MuJo Dataset Information")
    print("=" * 60)
    print("""
MuJo: Multimodal Joint Feature Space Learning

Contains:
- Video
- Video-derived poses
- Synthetic IMU (via IMUTube)
- Textual descriptions

Paper: https://arxiv.org/abs/2406.03857

Note: Dataset availability varies by paper release.
Check the paper's GitHub for download instructions.
    """)


def download_vidimu_info():
    """Print information about downloading VIDIMU dataset."""
    print("\n" + "=" * 60)
    print("VIDIMU Dataset Information")
    print("=" * 60)
    print("""
VIDIMU: Multimodal video and IMU kinematic dataset

Contains:
- 13 daily activities
- 54 subjects
- Synchronized video and IMU

Paper: https://www.nature.com/articles/s41597-023-02554-9

Download: The dataset should be available through Nature's data repository.
Check the paper supplementary materials for download links.
    """)


def main():
    parser = argparse.ArgumentParser(description="Download IMU-Video paired datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["sensorcaps", "opensqa", "llasa_model",
                                "ego4d", "mujo", "vidimu", "all_info"],
                        help="Dataset to download")
    parser.add_argument("--output_dir", type=str, default="/mnt/hdd/data",
                        help="Output directory for downloads")

    args = parser.parse_args()

    if args.dataset == "sensorcaps":
        download_sensorcaps(args.output_dir)
    elif args.dataset == "opensqa":
        download_opensqa(args.output_dir)
    elif args.dataset == "llasa_model":
        download_llasa_model(args.output_dir)
    elif args.dataset == "ego4d":
        download_ego4d_info()
    elif args.dataset == "mujo":
        download_mujo_info()
    elif args.dataset == "vidimu":
        download_vidimu_info()
    elif args.dataset == "all_info":
        download_ego4d_info()
        download_mujo_info()
        download_vidimu_info()


if __name__ == "__main__":
    main()
