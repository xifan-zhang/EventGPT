# EventGPT-DSEC Dataset Roadmap Summary

## Overview
Preparation pipeline for a custom EventGPT-DSEC training dataset combining event camera data, video frames, and MP4 files.

## Dataset Structure
```
dataset_dir/
├── event_npy/          # Event data (.npy) - shape (T, H, W, C)
├── video/              # Image frames (PNG) - for LLaVA 1.5
├── mp4/                # MP4 videos (H.264) - for Video-LLaVA
└── *.json              # HuggingFace-compatible annotations
```

## Core Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `build_my_egpt_dsec_train.py` | Main dataset builder | Existing |
| `generate_json.py` | Create JSON structure with paths | ✅ Complete |
| `preprocess_videos_to_mp4.py` | Convert image sequences to MP4 | 🔄 In Progress |
| `preprocess_event_images.py` | Event data preprocessing | Existing |

## Pipeline

1. **Generate JSON skeleton**
   ```bash
   python generate_json.py --dataset_dir /path/to/data
   ```

2. **Convert images to MP4** (Video-LLaVA requirement)
   ```bash
   python preprocess_videos_to_mp4.py --dataset_dir /path/to/data --fps 20
   ```

3. **Regenerate JSON** (include all paths)
   ```bash
   python generate_json.py --dataset_dir /path/to/data
   ```

4. **Annotate** - Fill in `conversations` fields manually

5. **Train** - Use annotated dataset for EventGPT

## Model Compatibility

| Model | Data Field | Format |
|-------|------------|--------|
| EventGPT | `event_data` | .npy files |
| LLaVA 1.5 | `video_data` | Image folders |
| Video-LLaVA | `mp4_data` | MP4 files (PyAV decoded) |

## Key Technical Specs

- **MP4**: libx264 codec, yuv420p, 20 FPS (50ms interval)
- **Event data**: NumPy arrays, DVS format
- **Images**: Sequential PNG files (000000.png, 000001.png, ...)
