# EventGPT-DSEC Dataset Building - Current Status

**Created**: January 23, 2025, 17:35:42 UTC
**Status**: PAUSED
**Last Activity**: Task pause command executed at 17:30 UTC

---

## Quick Reference

| Metric | Value |
|--------|-------|
| **Training Completeness** | ~25% (1/8 durations complete) |
| **Test Completeness** | ~97% (7.5/8 durations complete) |
| **Total Samples (Train)** | 205/328 sequences (62%) |
| **Total Samples (Test)** | 95/96 sequences (99%) |
| **Bottleneck** | Event image rendering (slow CPU operation) |
| **Estimated Resume Time** | 2-3 weeks for full completion |

---

## Executive Summary

Multi-scale EventGPT dataset construction from DSEC (DVS and events dataset) with 8 duration variants (500ms-16s). Building both **training** (41 sequences × 8 durations) and **test** (12 sequences × 8 durations) sets with complete preprocessing pipeline.

**Status**: PAUSED (Jan 23, 2025) - Tasks paused for manual intervention

---

## Dataset Architecture

```
/data/
├── my_egpt_dsec_train/              # Training set
│   ├── my_egpt_dsec_train_500ms/
│   ├── my_egpt_dsec_train_1s/       ✓ COMPLETE
│   ├── my_egpt_dsec_train_2s/
│   ├── my_egpt_dsec_train_4s/
│   ├── my_egpt_dsec_train_5s/
│   ├── my_egpt_dsec_train_8s/
│   ├── my_egpt_dsec_train_10s/
│   └── my_egpt_dsec_train_16s/      ⏳ In Progress
│
└── my_egpt_dsec_test/               # Test set
    ├── my_egpt_dsec_seq_500ms/      ✓ COMPLETE
    ├── my_egpt_dsec_seq_1s/         ✓ COMPLETE
    ├── my_egpt_dsec_seq_2s/         ✓ COMPLETE
    ├── my_egpt_dsec_seq_4s/         ✓ COMPLETE
    ├── my_egpt_dsec_seq_5s/         ✓ COMPLETE
    ├── my_egpt_dsec_seq_8s/         ✓ COMPLETE
    ├── my_egpt_dsec_seq_10s/        ✓ COMPLETE
    └── my_egpt_dsec_seq_16s/        95% (needs 2 event_images)

For each duration:
├── event_npy/           # Raw event data (NumPy arrays)
├── event_image/         # Rendered event visualizations (PNG)
├── video/              # Video frames extracted from sequences
├── mp4/                # Video files encoded as MP4 (Video-LLaVA format)
└── *.json              # HuggingFace-compatible annotations
```

---

## Core Processing Scripts

| Script | Purpose | Status | Notes |
|--------|---------|--------|-------|
| `build_my_egpt_dsec_train.py` | Extract clips from DSEC HDF5, create event_npy + video | ✓ Working | Uses streaming mode for 16s (2.3B events/seq) |
| `preprocess_event_images.py` | Render event data to PNG images | ⏳ In Progress | Slowest component (~1-2s per image) |
| `preprocess_videos_to_mp4.py` | Convert image sequences to H.264 MP4 | ⏳ In Progress | 8 parallel workers, ~180 MP4s/min |
| `generate_json.py` | Create annotation JSON with file paths | ✓ Complete | Generates HuggingFace-compatible format |

---

## Current Dataset Completion Status

### Training Set Progress

| Duration | event_npy | event_image | video | mp4 | json | Status |
|----------|-----------|-------------|-------|-----|------|--------|
| **500ms** | 41 ✓ | 3/41 | 41 ✓ | 628 | 1 | 27% |
| **1s** | 41 ✓ | 41 ✓ | 41 ✓ | **5208 ✓** | 1 | **✓ COMPLETE** |
| **2s** | 41 ✓ | 1/41 | 41 ✓ | 88 | 1 | 2% |
| **4s** | 41 ✓ | 1/41 | 41 ✓ | 40 | 1 | 2% |
| **5s** | 41 ✓ | 0/41 | 41 ✓ | 0 | 1 | 0% |
| **8s** | 41 ✓ | 0/41 | 18/41 | 0 | 1 | 0% |
| **10s** | 41 ✓ | 41 ✓ | 41 ✓ | 143 | 1 | 3% |
| **16s** | 3/41 | 0/41 | 3/41 | 0 | 0 | **Building** |

**Train Summary**: 1 duration complete, 6 partial, 1 in-progress

### Test Set Progress

| Duration | event_npy | event_image | video | mp4 | json | Status |
|----------|-----------|-------------|-------|-----|------|--------|
| **500ms** | 12 ✓ | 12 ✓ | 12 ✓ | 2220 ✓ | 3 | ✓ |
| **1s** | 12 ✓ | 12 ✓ | 12 ✓ | 1100 ✓ | 3 | ✓ |
| **2s** | 12 ✓ | 12 ✓ | 12 ✓ | 540 ✓ | 3 | ✓ |
| **4s** | 12 ✓ | 12 ✓ | 12 ✓ | 260 ✓ | 3 | ✓ |
| **5s** | 12 ✓ | 12 ✓ | 12 ✓ | 205 ✓ | 4 | ✓ |
| **8s** | 12 ✓ | 12 ✓ | 12 ✓ | 122 ✓ | 1 | ✓ |
| **10s** | 11 ✓ | 11 ✓ | 11 ✓ | 93 ✓ | 3 | ✓ |
| **16s** | 10 ✓ | 8/10 | 10 ✓ | 52 ✓ | 1 | 95% |

**Test Summary**: 7/8 durations complete, 1 needs final 2 event_images

---

## How the Pipeline Works

### Phase 1: Extract Event Data (HDF5 → NumPy)
- **Script**: `build_my_egpt_dsec_train.py` / `build_my_egpt_dsec_seq.py`
- **Input**: DSEC HDF5 files (from `/mnt/hdd/data/DSEC/train` or `test`)
- **Output**: `event_npy/sequence_name/clip_id.npy`
- **Process**:
  1. Open HDF5 file with timestamp offset
  2. Split into clips with 50% overlap (sliding window)
  3. Extract events (p, t, x, y) for each time window
  4. Save as NumPy dictionary
- **Status**: ✓ Complete for all durations (except 16s partial)
- **Challenge**: 16s clips contain 100-400M events each = slow I/O

### Phase 2: Extract & Copy Video Frames
- **Script**: Same build script extracts images alongside event data
- **Input**: DSEC PNG frames from `/images/left/rectified/`
- **Output**: `video/sequence_name/clip_id/` (original PNG names)
- **Process**: Copy frames that fall within each clip's time window
- **Status**: ✓ Complete (except 16s)
- **Result**: Time-aligned event + video data for each clip

### Phase 3: Render Event Visualizations
- **Script**: `preprocess_event_images.py`
- **Input**: event_npy files
- **Output**: `event_image/sequence_name/` (32-bit RGBA PNG)
- **Process**:
  1. Load events for each clip
  2. Render accumulation image (grayscale or color)
  3. Save as PNG
- **Status**: ⏳ In Progress (slow, ~1-2s per image)
- **Current**:
  - Train 1s: ✓ Complete (5,208 images)
  - Train 10s: ✓ Complete (1,025 images)
  - Others: 0-3% complete

### Phase 4: Encode Video Files
- **Script**: `preprocess_videos_to_mp4.py`
- **Input**: `video/sequence_name/clip_id/` PNG frames
- **Output**: `mp4/sequence_name/clip_id.mp4`
- **Process**:
  1. Load frame sequence
  2. Encode with libx264 (H.264)
  3. Set FPS (default 20 = 50ms interval)
  4. Use 8 parallel workers
- **Status**: ⏳ In Progress
- **Current**:
  - Train 1s: ✓ Complete (5,208 MP4s)
  - Others: 0-628 per duration
- **Speed**: ~180 MP4s/min with 8 workers

### Phase 5: Generate Annotations
- **Script**: `generate_json.py`
- **Input**: Dataset directories
- **Output**: `dataset_dir/dataset.json`
- **Format**: HuggingFace-compatible instruction-following format
- **Status**: ✓ Complete
- **Example**:
```json
{
  "id": "clip_000000",
  "conversations": [
    {"from": "human", "value": "<event>\nDescribe the motion."},
    {"from": "gpt", "value": "..."}
  ],
  "video_data": "mp4/sequence_name/000000.mp4",
  "event_data": "event_npy/sequence_name/000000.npy"
}
```

---

## Data Specifications

### Event Data Format (NumPy)
Each `.npy` file contains a dictionary:
```python
{
    'p': np.array([0, 1, 0, ...]),           # Polarity (0=off, 1=on)
    't': np.array([ts1, ts2, ts3, ...]),     # Timestamp (microseconds)
    'x': np.array([x1, x2, x3, ...]),        # X coordinate (0-639)
    'y': np.array([y1, y2, y3, ...]),        # Y coordinate (0-479)
    'time_start': 57267307977,               # Clip start time (µs)
    'time_end': 57277307977,                 # Clip end time (µs)
    'duration_ms': 10000.0,                  # Duration (milliseconds)
    'num_events': 1234567                    # Event count
}
```

### Event Image Format
- **Type**: 32-bit RGBA PNG
- **Size**: 640×480 (DSEC resolution)
- **Content**: Accumulated event polarity rendered as grayscale
- **Purpose**: Visualization for training + debugging

### MP4 Specifications
- **Codec**: H.264 (libx264)
- **Pixel Format**: YUV420p
- **FPS**: 20 (one frame per 50ms)
- **Duration**: Varies by clip duration
- **Purpose**: Video-LLaVA model input

### JSON Structure
HuggingFace-compatible dataset format with optional fields for:
- Event data path (`event_data`)
- Video/MP4 path (`video_data`, `mp4_data`)
- Conversations (instruction-following format)

---

## Known Challenges & Solutions

### Challenge 1: 16s Clips Size
**Problem**: 16-second clips contain 100-400M events each (~1.6-6.4GB raw)
- Single HDF5 file: 2.3B events per sequence
- Reading/processing all events causes OOM

**Solution Applied**:
- ✓ Streaming mode: Load clip-by-clip instead of full sequence
- ✓ Binary search for time windows: O(log N) instead of full scan
- ✓ Memory cleanup: Explicit `gc.collect()` after each clip

**Current Status**: 3/41 sequences processed, streaming works but slow

### Challenge 2: Event Image Rendering
**Problem**: Rendering 10,000+ images takes ~5-8 hours per duration
- NumPy accumulation + PIL rendering is CPU-bound
- No GPU acceleration currently

**Solution**:
- ✓ Multi-threaded processing (parallel workers)
- ✓ Batch caching (load multiple events before rendering)
- ⏳ Could parallelize across sequences

### Challenge 3: MP4 Encoding Latency
**Problem**: H.264 encoding adds significant latency
- Each MP4 requires loading images + FFmpeg encoding
- I/O bound on HDD access

**Solution**:
- ✓ 8 parallel workers reduce overall time
- ✓ Local PIL caching
- Speed: ~180 MP4s/min (≈ 1 duration in 30-40 min)

### Challenge 4: Temporal Alignment
**Problem**: Events and video frames have different time origins
- HDF5 timestamps offset from frame timestamps
- Solution: Apply `t_offset` correction

**Solution Applied**:
- ✓ Load `t_offset` from HDF5
- ✓ Apply offset during event loading
- ✓ Verify clip boundaries match frame timestamps

---

## Recommended Next Steps (When Resuming)

### Priority 1: Complete Test 16s (Last 2 images)
- Needs: 2 event_image files
- Time: ~10 minutes
- Command:
```bash
python preprocess_event_images.py --data_dir /path/to/my_egpt_dsec_seq_16s
```

### Priority 2: Finish Train Event Images
- Needs: ~43K images across 5 durations (500ms, 2s, 4s, 5s, 8s)
- Duration: Slow (~1-2s per image)
- Parallel: Can run 5 durations in parallel
- Command:
```bash
for d in 500ms 2s 4s 5s 8s; do
    python preprocess_event_images.py --data_dir my_egpt_dsec_train_$d &
done
```

### Priority 3: Generate Train MP4s
- Needs: ~18K MP4s across 5 durations (2s, 4s, 5s, 8s, 10s)
- Duration: Moderate (~30-60 min per duration with 8 workers)
- Command:
```bash
for d in 2s 4s 5s 8s 10s; do
    python preprocess_videos_to_mp4.py --dataset_dir my_egpt_dsec_train_$d &
done
```

### Priority 4: Complete Train 16s Build
- Needs: 38 more sequences (3/41 done)
- Slowest: Requires HDF5 parsing + event loading
- Time: Highly variable (12-60s per clip × 13 clips × 38 sequences)
- Command:
```bash
python build_my_egpt_dsec_train.py --durations 16s
```

---

## System Requirements

- **Storage**: ~200GB total (raw HDF5) → ~100GB (processed)
- **RAM**: 62GB recommended (streaming reduces peak to ~5-6GB)
- **CPU**: 8+ cores (parallel processing)
- **Dependencies**:
  - Python 3.10+
  - NumPy, h5py (HDF5 handling)
  - PIL/Pillow (image rendering)
  - imageio + imageio-ffmpeg (MP4 encoding)
  - tqdm (progress tracking)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| HDF5 plugin not found | Set `HDF5_PLUGIN_PATH` environment variable |
| OOM on 16s clips | Already using streaming mode; may need to reduce batch size |
| Slow event image rendering | Normal (CPU-bound); consider GPU rendering in future |
| MP4 encoding fails | Verify imageio-ffmpeg installed: `pip install imageio-ffmpeg` |
| Missing frames in output | Check DSEC image directory exists: `/path/to/sequence/images/left/rectified/` |

---

## Performance Metrics

### Speed (as of Jan 23, 2025)

| Operation | Speed | Notes |
|-----------|-------|-------|
| Event extraction | Variable | 12-60s per clip (depends on event density) |
| Event image rendering | ~1-2s per image | CPU-bound, could be parallelized |
| MP4 encoding | ~180 files/min | 8 parallel workers |
| Build 16s clip | ~20-60s | Includes HDF5 I/O, streaming mode |

### Dataset Sizes (Observed)

| Component | Size per Duration | Notes |
|-----------|-------------------|-------|
| event_npy | 10-50 MB | Varies with event density |
| event_image | 20-50 GB | 640×480 PNG × ~5,000 clips |
| video frames | 30-80 GB | PNG images |
| mp4 files | 5-15 GB | H.264 encoded |

---

## References

- **DSEC**: Gehrig et al., "DSEC: A Stereo Event Camera Dataset"
- **EventGPT**: Internal training framework for event-based vision models
- **Video-LLaVA**: LLaVA with video understanding
- **H.264/MP4**: Standard video codec for compatibility

---

## Status Log

- **Jan 23, 2025, 17:30**: All preprocessing tasks paused. Test 16s at 80%, Train 1s complete, others in progress.
- **Jan 23, 2025, 16:08**: Resumed all tasks after initial pause. 45 parallel scripts running.
- **Jan 23, 2025, 15:29**: Initial 16s build crashed after 1 sequence. Restarted with streaming mode.
- **Jan 23, 2025, 14:47**: Test set complete except Test 16s (needs 2 event_images).
- **Jan 22, 2025**: Dataset architecture finalized, scripts created.

---

**Last Updated**: Jan 23, 2025 17:35 UTC
**Current Status**: PAUSED - Ready to resume processing
**Document Created**: January 23, 2025, 17:35:42 UTC
**Next Review**: Check status before resuming tasks
