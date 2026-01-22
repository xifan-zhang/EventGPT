# My EventGPT-DSEC Dataset - Roadmap

This directory contains scripts for preparing and managing the custom EventGPT-DSEC dataset.

## Dataset Structure

```
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
├── mp4/                # MP4 video files (for Video-LLaVA)
│   └── interlaken_00_a/
│       └── 000000.mp4   # MP4 video file
└── EventGPT_Instruction_Subset.json  # Dataset annotation file
```

## Scripts

### 1. `build_my_egpt_dsec_train.py`
**Purpose**: Main script to build the custom EventGPT-DSEC training dataset.

**Status**: Existing script (from git status)

### 2. `generate_json.py`
**Purpose**: Generate HuggingFace-compatible dataset JSON structure.

**Features**:
- Scans event .npy files from `event_npy/`
- Includes both video folder paths AND MP4 file paths
- Creates JSON entries with empty question and gpt fields for manual annotation
- Verifies both video folders and MP4 files exist

**Output Format**:
```json
{
  "id": "uuid",
  "split": "my_egpt_dsec_seq_5s",
  "event_data": "interlaken_00_a/000000.npy",
  "video_data": "interlaken_00_a/000000",
  "mp4_data": "interlaken_00_a/000000.mp4",
  "conversations": [
    {"from": "human", "value": ""},
    {"from": "gpt", "value": ""}
  ]
}
```

**Usage**:
```bash
# Generate JSON for all sequences
python generate_json.py --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

# Generate JSON for specific sequence
python generate_json.py --sequence interlaken_00_a --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

# Limit samples for testing
python generate_json.py --max_samples 100 --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s
```

**Status**: ✅ Completed - Includes both video and mp4 fields

### 3. `preprocess_videos_to_mp4.py`
**Purpose**: Convert image sequences to MP4 video files for Video-LLaVA compatibility.

**Details**:
- Video-LLaVA requires actual video files decoded with PyAV, not pre-extracted image sequences
- Converts image folders to MP4 using imageio with libx264 codec
- Pixel format: yuv420p (compatible with PyAV)
- FPS: 20 (interval=50ms)

**Usage**:
```bash
# Preprocess all video frames to MP4
python preprocess_videos_to_mp4.py --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

# Process specific sequence
python preprocess_videos_to_mp4.py --sequence interlaken_00_a --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

# Limit samples for testing
python preprocess_videos_to_mp4.py --max_samples 10 --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s

# Specify FPS (default 20, interval=50ms)
python preprocess_videos_to_mp4.py --fps 20 --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s
```

**Requirements**:
- imageio
- imageio-ffmpeg
- PIL/Pillow

**Status**: 🔄 In Progress - Currently running

### 4. `preprocess_event_images.py`
**Purpose**: Preprocess event camera data into image format.

**Status**: Existing script (from git status)

### 5. `generate_json.py` (for annotations)
**Purpose**: Script to generate JSON from annotated data.

**Status**: Existing script (from git status)

## Workflow

### Complete Dataset Preparation Pipeline

1. **Generate initial JSON** (with empty conversations)
   ```bash
   python generate_json.py --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s
   ```

2. **Preprocess images to MP4** (for Video-LLaVA)
   ```bash
   python preprocess_videos_to_mp4.py --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s --fps 20
   ```

3. **Regenerate JSON** (includes both video and mp4 paths)
   ```bash
   python generate_json.py --dataset_dir /mnt/hdd/data/my_egpt_dsec_seq_5s
   ```

4. **Manual Annotation**: Fill in the `conversations` fields with questions and answers

5. **Training**: Use the annotated dataset for EventGPT training

## Integration with Benchmark Inference

The generated JSON is compatible with `feasible/benchmark_inference/benchmark_inference.py`:

- **EventGPT**: Uses `event_data` field (event .npy files)
- **LLaVA 1.5**: Uses `video_data` field (image folders)
- **Video-LLaVA**: Uses `mp4_data` field (MP4 files decoded with PyAV)

## Technical Details

### Video Format Specifications

**Image Folders** (for LLaVA 1.5):
- Format: Individual PNG files
- Naming: Sequential numbers (000000.png, 000001.png, ...)
- Location: `dataset_dir/video/sequence/clip/`

**MP4 Files** (for Video-LLaVA):
- Codec: libx264 (H.264)
- Pixel Format: yuv420p
- FPS: 20 (50ms interval)
- Location: `dataset_dir/mp4/sequence/clip.mp4`
- Decoder: PyAV (required by Video-LLaVA)

### Event Data Format

- Format: NumPy arrays (.npy)
- Shape: (T, H, W, C) where T=timesteps, H=height, W=width, C=channels
- Content: Event camera data (DVS format)
- Location: `dataset_dir/event_npy/sequence/clip.npy`

## Notes

- **Video-LLaVA Compatibility**: Video-LLaVA is trained on actual video files and requires PyAV decoding. Pre-extracted image sequences do not work, hence the MP4 conversion step.
- **Dual Format Support**: The JSON includes both `video_data` (for LLaVA 1.5) and `mp4_data` (for Video-LLaVA), allowing benchmark comparison between approaches.
- **FPS Setting**: The correct FPS is 20 (50ms interval = 1000ms/50ms = 20 fps). This preserves the original temporal characteristics of the data.
