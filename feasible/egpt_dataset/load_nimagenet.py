# README
# ======
#
# N-ImageNet Dataset Loader
# -------------------------
#
# This script downloads the N-ImageNet dataset from HuggingFace.
# The dataset is downloaded to /mnt/hdd/data/N-ImageNet directory.
#
# Author: Alice Zhang
#
# Requirements:
# - huggingface_hub
# - datasets (optional, for alternative loading method)
#
# Usage:
# Run this script directly: python load_nimagenet.py
# Or import and use the download function in your code.
#
# Note: You may need to login to HuggingFace first using:
# huggingface-cli login

# from huggingface_hub import login
# login()

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("82magnolia/N-ImageNet", cache_dir=".")

from huggingface_hub import snapshot_download

print("Downloading dataset...")
snapshot_download(
    repo_id="82magnolia/N-ImageNet",
    repo_type="dataset",
    local_dir="/mnt/hdd/data/N-ImageNet",
    # ignore_patterns=["*.arrow"],
    max_workers=1,
)

print("Dataset downloaded successfully.")

