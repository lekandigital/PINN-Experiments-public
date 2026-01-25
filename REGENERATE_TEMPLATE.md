# Regeneration Instructions

This file documents how to recreate artifacts that are deterministically reproducible from code + config.

## Quick Reference

| Artifact | Command | Output | Time |
|----------|---------|--------|------|
| Synthetic data | `python generate_data.py --seed 42` | `data/*.h5` | ~5 min |
| Training | `python train.py --config config.yaml` | `checkpoints/` | ~2 hrs |
| Benchmarks | `python benchmark.py` | `results/` | ~10 min |

## Detailed Instructions

### 1. Synthetic Dataset

```bash
python src/generate_data.py --seed 42 --n_samples 10000
```

**Output**: `data/synthetic_train.h5` (~50MB)
**Time**: ~5 minutes
**Deterministic**: Yes (seeded)

### 2. Model Training

```bash
python src/train.py \
    --data data/synthetic_train.h5 \
    --epochs 50 \
    --seed 42 \
    --checkpoint-dir checkpoints
```

**Output**: 
- `checkpoints/best_model.pt`
- `checkpoints/final_model.pt`

**Time**: ~2 hours on RTX 4090
**Deterministic**: Approximately (GPU non-determinism)

### 3. Benchmark Plots

```bash
python src/benchmark.py \
    --checkpoint checkpoints/best_model.pt \
    --output results/
```

**Output**: `results/plots/*.png`, `results/metrics.json`
**Time**: ~10 minutes
**Deterministic**: Yes

## Non-Regenerable Artifacts

These MUST be preserved in backups:

- `checkpoints/best_model.pt` — Hours of GPU time
- `results/final_metrics.json` — Experiment results
- Any manually curated data
