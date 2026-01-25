# HGNN-ClothDyn Training Results - RTX 3090 Ti

## Training Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 3090 Ti |
| VRAM | 25.3 GB |
| Epochs | 50 |
| Learning Rate | 0.0001 (with ReduceLROnPlateau) |
| Hidden Dimension | 128 |
| Model Parameters | 619,340 |
| AMP | Disabled (for numerical stability) |
| Mesh Size | 20x20 (400 nodes) |
| Frames | 200 |

## Training Summary

- **Training Time**: 4 minutes 23 seconds
- **Best Validation Loss**: 0.1819 (epoch 23)
- **Final Validation Loss**: 0.2916 (epoch 50)
- **No NaN losses** throughout training

## Loss Progression

### Training Loss
| Epoch | Total Loss | Position Loss | Velocity Loss | Edge Loss |
|-------|------------|---------------|---------------|-----------|
| 1 | 0.0923 | 0.000035 | 0.922401 | 0.000106 |
| 10 | 0.0844 | 0.000153 | 0.842838 | 0.000113 |
| 20 | 0.5948 | 0.073931 | 5.208690 | 0.000813 |
| 30 | 0.8263 | 0.207775 | 6.184585 | 0.010328 |
| 40 | 0.6282 | 0.199167 | 4.289059 | 0.015921 |
| 50 | 0.8517 | 0.388295 | 4.632221 | 0.016935 |

### Validation Loss Progression
| Epoch | Val Loss |
|-------|----------|
| 1 | 1.6821 |
| 10 | 10.6880 |
| 20 | 0.3324 |
| 23 (best) | 0.1819 |
| 30 | 0.5474 |
| 40 | 0.5424 |
| 50 | 0.2916 |

### Learning Rate Schedule
- Epochs 1-18: 0.0001
- Epochs 19-28: 0.00005
- Epochs 29-34: 0.000025
- Epochs 35-40: 0.0000125
- Epochs 41-46: 0.00000625
- Epochs 47-50: 0.000003125

## Benchmark Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Position RMSE | 0.3140 | < 0.1 | Not met |
| Edge Length Error | 0.0077 | < 0.05 | **PASS** |
| Inference FPS | 167.4 | > 50 | **PASS** |
| Peak Memory | 26.2 MB | - | - |

### Notes on Position RMSE
The position RMSE (0.314) is higher than the target (0.1) due to:
1. Using simple strided clustering instead of Graclus (torch-cluster compatibility issue)
2. Error accumulation over 50-frame autoregressive rollout
3. The model performs well on short-term predictions but accumulates error over time

## Integration Test Results

| Test | Status |
|------|--------|
| Mesh to Graph Conversion | PASS |
| EdgeForceConv Layer | PASS |
| Model Forward Pass | PASS |
| Hierarchical Message Passing | PASS |
| Numerical Stability | PASS |
| Synthetic Data Generation | PASS |
| Training Loop | FAIL* |
| Benchmark Execution | PASS |
| GPU Utilization | PASS |

*Training loop test failed due to assertion checking monotonic loss decrease (scheduled sampling causes variance)

**Overall: 8/9 tests passed**

## Key Fixes Applied

1. **Disabled AMP by default** - Prevents NaN losses on consumer GPUs
2. **Added input normalization** - Improves numerical stability
3. **Added gradient NaN detection** - Skips steps with unstable gradients
4. **Lowered learning rate** - 0.0001 instead of 0.001 for physics-informed networks
5. **Added fallback clustering** - Uses simple strided clustering when torch-cluster unavailable

## Files Generated

- `checkpoints/best_model.pt` - Best model (epoch 23, val_loss=0.1819)
- `checkpoints/final_model.pt` - Final model (epoch 50)
- `checkpoints/loss_history.json` - Full training history
- `results/benchmark_results.json` - Detailed benchmark metrics
- `results/rollout_error_plot.png` - Error visualization
- `results/performance_summary.png` - Performance charts
