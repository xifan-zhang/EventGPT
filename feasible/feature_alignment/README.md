# Feature Alignment Module for EventGPT

A comprehensive module for aligning Event camera features to RGB/CLIP feature spaces, enabling effective cross-modal learning and speculative decoding.

## Overview

This module provides multiple alignment strategies to bridge the gap between event camera representations and standard RGB vision features (e.g., CLIP). The alignment is crucial for:
- Enabling speculative decoding with event-based draft models
- Cross-modal retrieval and understanding
- Leveraging pre-trained RGB vision-language models with event data

## Available Alignment Strategies

| Strategy | Data Requirement | Performance | Use Case |
|----------|------------------|-------------|----------|
| **Contrastive (CEIA)** | 5-10K paired samples | High | Best for paired Event-RGB data |
| **Triple-Modal** | >10K paired samples | Highest | Maximum performance with large datasets |
| **Reconstruction** | Pre-trained E2VID | Medium | No paired data available |
| **Lightweight** | <5K paired samples | Medium | Quick validation, limited resources |

## Quick Start

### 1. Strategy Selection

```python
from feature_alignment import auto_select_alignment_strategy, print_strategy_recommendation

recommendation = auto_select_alignment_strategy(
    has_paired_data=True,
    data_size=5000,
    has_e2vid=False,
    target_performance='good'
)
print_strategy_recommendation(recommendation)
```

### 2. Run Alignment on DSEC Dataset

```bash
# Run on all durations
python run_egpt_dsec_alignment.py \
    --durations 200ms 500ms 1s 2s 4s 5s 10s 20s \
    --strategy contrastive \
    --num_epochs 20 \
    --batch_size 32

# Run on specific duration
python run_egpt_dsec_alignment.py \
    --durations 200ms \
    --strategy lightweight \
    --num_epochs 10
```

### 3. Using Pre-extracted Features

```python
from feature_alignment import (
    ContrastiveAlignmentConfig,
    ContrastiveAlignmentModule,
    ContrastiveAlignmentTrainer,
    AlignmentDataset
)

# Load pre-extracted features
train_dataset = AlignmentDataset.from_directory('./features/train')
val_dataset = AlignmentDataset.from_directory('./features/val')

# Configure and train
config = ContrastiveAlignmentConfig(
    event_dim=768,
    target_dim=1024,
    temperature=0.07
)
alignment = ContrastiveAlignmentModule(config)
trainer = ContrastiveAlignmentTrainer(alignment, train_dataset, val_dataset)
trainer.train(num_epochs=20, save_path='alignment.pt')
```

## Module Structure

```
feature_alignment/
├── __init__.py              # Main exports
├── base.py                  # Base classes (AlignmentConfig, BaseAlignmentModule)
├── contrastive.py           # Contrastive alignment (CEIA-style)
├── lightweight.py           # Lightweight adapters
├── reconstruction.py        # E2VID-based reconstruction bridging
├── triple_modal.py          # Triple-modal alignment (E-CLIP style)
├── diagnosis.py             # Alignment diagnostics
├── utils.py                 # Utilities and evaluation
├── run_egpt_dsec_alignment.py  # DSEC dataset training script
└── checkpoints/             # Saved models
    ├── alignment_200ms/
    ├── alignment_500ms/
    └── ...
```

## DSEC Dataset Durations

| Duration | Samples | Description |
|----------|---------|-------------|
| 200ms | 5,575 | Fine-grained temporal resolution |
| 500ms | 2,220 | Short-term dynamics |
| 1s | 1,100 | Standard temporal window |
| 2s | 540 | Medium temporal context |
| 4s | 260 | Extended temporal context |
| 5s | 205 | Long-term patterns |
| 10s | 93 | Very long sequences |
| 20s | 38 | Maximum temporal context |

## Evaluation Metrics

- **MSE**: Mean squared error between aligned and target features
- **Cosine Similarity**: Feature space alignment quality
- **R@1, R@5, R@10**: Retrieval recall at different K values
- **Acceptance Rate**: For speculative decoding integration

## Training Results (200ms Duration)

```
Strategy: Contrastive
Train samples: 4,460
Val samples: 1,115
Feature dim: 150,528 (224x224x3 flattened)

Training Progress:
- Epoch 1: Loss ~2.9, Acc ~10%
- Epoch 3: Loss ~0.2, Acc ~95%
- Epoch 5: Loss ~0.1, Acc ~98%
```

## Roadmap

### Phase 1: Feature Extraction (Completed)
- [x] Create dataset loaders for DSEC event images
- [x] Implement paired Event-RGB data loading
- [x] Support all duration configurations

### Phase 2: Alignment Training (In Progress)
- [x] Contrastive alignment on 200ms
- [ ] Run alignment on all durations (500ms, 1s, 2s, 4s, 5s, 10s, 20s)
- [ ] Compare contrastive vs lightweight strategies
- [ ] Hyperparameter optimization

### Phase 3: Evaluation & Comparison
- [ ] LLaVA baseline with single RGB image
- [ ] Cross-duration generalization test
- [ ] Feature space visualization (t-SNE/UMAP)
- [ ] Speculative decoding acceptance rate measurement

### Phase 4: Integration
- [ ] Integrate best alignment into EventGPT pipeline
- [ ] Benchmark inference speed improvements
- [ ] Document optimal configurations per use case

## Usage with EventGPT

```python
# Load trained alignment
from feature_alignment import ContrastiveAlignmentModule

alignment = ContrastiveAlignmentModule.load('checkpoints/alignment_200ms/contrastive_alignment.pt')

# Use in inference
event_features = event_encoder(event_data)
aligned_features = alignment(event_features)

# Now aligned_features can be used with RGB-trained decoders
```

## References

- CEIA: Contrastive Event-Image Alignment for Scene Understanding
- E-CLIP: Event-based CLIP for Cross-Modal Learning
- DSEC: A Stereo Event Camera Dataset for Driving Scenarios
