#!/bin/bash
#
# Generate EventGPT DSEC Dataset for a Specific Duration
# =========================================================
#
# This script runs the complete pipeline to generate a dataset:
#   1. Build dataset (event_npy + video frames)
#   2. Generate JSON instruction file
#   3. Preprocess event images
#   4. Convert videos to MP4
#
# Usage:
#   ./generate_duration_dataset.sh <duration>
#
# Examples:
#   ./generate_duration_dataset.sh 5s
#   ./generate_duration_dataset.sh 200ms
#   ./generate_duration_dataset.sh 8s
#
# Supported duration formats: 200ms, 500ms, 1s, 2s, 4s, 5s, 8s, 10s, 16s, 20s
#

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <duration>"
    echo "Example: $0 5s"
    echo ""
    echo "Supported formats: 200ms, 500ms, 1s, 2s, 4s, 5s, 8s, 10s, 16s, 20s"
    exit 1
fi

DURATION=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="/mnt/hdd/data"
OUTPUT_DIR="${OUTPUT_BASE}/my_egpt_dsec_seq_${DURATION}"

echo "========================================"
echo "EventGPT DSEC Dataset Generation"
echo "========================================"
echo "Duration: ${DURATION}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"
echo ""

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate egpt

# Step 1: Build dataset
echo "========================================"
echo "Step 1/4: Building dataset..."
echo "========================================"
cd "${SCRIPT_DIR}"
python build_my_egpt_dsec_seq.py --durations "${DURATION}"

# Check if build succeeded
if [ ! -d "${OUTPUT_DIR}/event_npy" ]; then
    echo "ERROR: Dataset build failed - no event_npy directory created"
    exit 1
fi

# Step 2: Generate JSON
echo ""
echo "========================================"
echo "Step 2/4: Generating JSON..."
echo "========================================"
python generate_json.py --dataset_dir "${OUTPUT_DIR}"

# Check if JSON was created
if [ ! -f "${OUTPUT_DIR}/EventGPT_Instruction_Subset.json" ]; then
    echo "ERROR: JSON generation failed"
    exit 1
fi

# Step 3: Preprocess event images
echo ""
echo "========================================"
echo "Step 3/4: Preprocessing event images..."
echo "========================================"
python preprocess_event_images.py --dataset_dir "${OUTPUT_DIR}"

# Check if event_image directory was created
if [ ! -d "${OUTPUT_DIR}/event_image" ]; then
    echo "ERROR: Event image preprocessing failed"
    exit 1
fi

# Step 4: Convert videos to MP4
echo ""
echo "========================================"
echo "Step 4/4: Converting videos to MP4..."
echo "========================================"
python preprocess_videos_to_mp4.py --dataset_dir "${OUTPUT_DIR}"

# Check if mp4 directory was created
if [ ! -d "${OUTPUT_DIR}/mp4" ]; then
    echo "ERROR: MP4 conversion failed"
    exit 1
fi

# Summary
echo ""
echo "========================================"
echo "Dataset Generation Complete!"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Contents:"
ls -la "${OUTPUT_DIR}"
echo ""
echo "Clip counts:"
echo "  event_npy: $(find ${OUTPUT_DIR}/event_npy -name '*.npy' 2>/dev/null | wc -l) files"
echo "  event_image: $(find ${OUTPUT_DIR}/event_image -name '*.png' 2>/dev/null | wc -l) files"
echo "  video: $(find ${OUTPUT_DIR}/video -type d -mindepth 2 2>/dev/null | wc -l) clips"
echo "  mp4: $(find ${OUTPUT_DIR}/mp4 -name '*.mp4' 2>/dev/null | wc -l) files"
echo ""
echo "Done!"
