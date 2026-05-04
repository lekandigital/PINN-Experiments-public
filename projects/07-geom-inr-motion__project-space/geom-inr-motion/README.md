# Geom-INR-Motion

**Implicit Neural Representations with Differential Geometry Regularization for Human Motion Prediction**

An experimental research implementation of continuous-time motion prediction using SIREN-based INRs (Implicit Neural Representations) with curvature and torsion losses inspired by differential geometry.

## Overview

This project implements a novel approach to human motion prediction:

1. **SIREN Architecture**: Uses sine-activated neural networks following Sitzmann et al. (2020) for smooth, continuous motion representation
2. **Differential Geometry Losses**: Regularizes motion trajectories using curvature (κ) and torsion (τ) penalties
3. **Continuous-Time Queries**: Model can be queried at any time t ∈ [0, 1], not just discrete frames
4. **Actor Embeddings**: Supports multi-actor training with per-actor embeddings

## Project Structure

```
geom-inr-motion/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── minimal_test.py        # Quick validation test (run first!)
├── data_pipeline.py       # AMASS/H3.6M data loading
├── model.py               # GeomINR architecture
├── geometry_losses.py     # Curvature & torsion losses
├── train.py               # Training pipeline
├── export_utils.py        # BVH/USD/NPZ export
└── evaluation.ipynb       # Evaluation notebook
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Minimal Test (Priority!)

This validates the core INR training loop works:

```bash
python minimal_test.py
```

**Expected Output:**
- Loss decreases from ~1.0 to <0.01 in 100 steps
- Test prediction shows low MPJPE

### 3. Train on Synthetic Data

```bash
python train.py \
    --epochs 100 \
    --batch_size 2048 \
    --lr 1e-4 \
    --curvature_weight 0.001 \
    --torsion_weight 0.0001
```

### 4. Evaluate Model

Open `evaluation.ipynb` in Jupyter:

```bash
jupyter notebook evaluation.ipynb
```

## Training on Real Data

### AMASS Dataset

1. Download AMASS from https://amass.is.tue.mpg.de/
2. Set `--data_dir /path/to/amass`
3. Run training:

```bash
python train.py \
    --data_dir /path/to/amass \
    --dataset amass \
    --epochs 500 \
    --batch_size 4096 \
    --curvature_weight 0.001 \
    --torsion_weight 0.0001 \
    --output_dir checkpoints/
```

### Human3.6M Dataset

1. Download H3.6M from http://vision.imar.ro/human3.6m/
2. Set `--data_dir /path/to/h36m` and `--dataset h36m`

## Model Architecture

```
GeomINR(
  Input: (t, actor_id) → (1+E, ) where E=32 actor embedding dim
  
  FourierPositionalEncoding(L=10) → (2*L+1, )
  
  SIREN Layers:
    Linear(input_dim, 256) + Sine(ω=30)
    Linear(256, 256) + Sine(ω=30)  [× 3]
    Linear(256, J*3)  # J=24 joints
  
  SkeletonGraphModule (optional):
    Graph smoothing over skeleton topology
  
  Output: (J*3, ) joint positions
)
```

### SIREN Initialization

Following Sitzmann et al. 2020:
- First layer: U(-1/input_dim, 1/input_dim)
- Hidden layers: U(-√(6/n)/ω, √(6/n)/ω)

## Loss Functions

### Primary Loss
- **MPJPE**: Mean Per-Joint Position Error (L2 distance)

### Geometry Regularization

**Curvature (κ)**:
$$\kappa = \frac{|r' \times r''|}{|r'|^3}$$

**Torsion (τ)**:
$$\tau = \frac{(r' \times r'') \cdot r'''}{|r' \times r''|^2}$$

Where r(t) is the joint trajectory, and derivatives are computed via finite differences.

### Combined Loss
```
L = L_mpjpe + λ_κ * L_curvature + λ_τ * L_torsion + λ_jerk * L_jerk
```

Default weights: λ_κ=0.001, λ_τ=0.0001, λ_jerk=0.0

## VastAI Deployment

For GPU training on VastAI L40S (48GB VRAM):

1. Launch instance with PyTorch 2.1+ image
2. Clone this repo
3. Install deps: `pip install -r requirements.txt`
4. Train with large batch:

```bash
python train.py \
    --batch_size 8192 \
    --epochs 1000 \
    --model large \
    --output_dir /workspace/checkpoints
```

## Export Formats

After training, export motion predictions:

```python
from export_utils import MotionExporter

exporter = MotionExporter(model, device='cuda')

# Export to BVH (Blender, Maya, etc.)
exporter.export_sequence(times, actor_id=0, format='bvh', path='motion.bvh')

# Export to USD (Omniverse, Unreal)
exporter.export_sequence(times, actor_id=0, format='usd', path='motion.usd')

# Export to NPZ (Python/NumPy)
exporter.export_sequence(times, actor_id=0, format='npz', path='motion.npz')
```

## Configuration

### TrainingConfig Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | SIREN hidden dimension |
| `num_layers` | 4 | Number of SIREN layers |
| `omega_0` | 30.0 | SIREN frequency |
| `num_actors` | 100 | Actor embedding count |
| `actor_embed_dim` | 32 | Actor embedding dimension |
| `batch_size` | 2048 | Training batch size |
| `lr` | 1e-4 | Learning rate |
| `curvature_weight` | 0.001 | κ loss weight |
| `torsion_weight` | 0.0001 | τ loss weight |

## Evaluation Metrics

- **MPJPE** (mm): Primary accuracy metric
- **PA-MPJPE** (mm): Procrustes-aligned MPJPE
- **Jerk**: Motion smoothness (lower is better)
- **Curvature/Torsion**: Geometry regularization effectiveness

## Known Limitations

1. **Data Dependency**: Requires AMASS or H3.6M for realistic training
2. **Single-Sequence Model**: Currently trains separate model per sequence
3. **No Temporal Conditioning**: INR sees only time, not past frames
4. **USD Export**: Requires `pxr` library (falls back to USDA text)

## Future Work

- [ ] Add conditional INR with past-frame encoding
- [ ] Multi-sequence training with action embeddings  
- [ ] Real-time inference optimization
- [ ] Integration with physics simulation
- [ ] Transformer-based time encoder

## References

- [SIREN](https://arxiv.org/abs/2006.09661): Implicit Neural Representations with Periodic Activation Functions
- [AMASS](https://amass.is.tue.mpg.de/): Archive of Motion Capture as Surface Shapes
- [Human3.6M](http://vision.imar.ro/human3.6m/): Large Scale Dataset for 3D Human Pose Estimation

## License

MIT License - See LICENSE file for details.

## Citation

```bibtex
@misc{geom-inr-motion,
  title={Geom-INR-Motion: Implicit Neural Representations with Differential Geometry Regularization},
  author={Research Implementation},
  year={2024},
  note={Experimental research code}
}
```
