# Feasibility Study TODO

## Feature Alignment

- [ ] Run feature alignment on all EGPT DSEC train durations (200ms, 500ms, 1s, 2s, 4s, 5s, 10s, 20s)
- [ ] Evaluate alignment quality metrics (MSE, Cosine Similarity, R@1, R@5, R@10)
- [ ] Compare contrastive vs lightweight alignment strategies
- [ ] Analyze how alignment quality varies with event duration

## LLaVA Baseline Comparison

- [ ] Set up LLaVA with single RGB image as baseline
- [ ] Compare LLaVA (RGB) vs EventGPT (Event) performance on same scenes
- [ ] Measure inference speed differences
- [ ] Evaluate quality metrics on DSEC test set
- [ ] Document modality-specific strengths/weaknesses

## Experiments

- [ ] Feature space visualization (t-SNE/UMAP) before and after alignment
- [ ] Ablation study on alignment hyperparameters
- [ ] Cross-duration generalization test
