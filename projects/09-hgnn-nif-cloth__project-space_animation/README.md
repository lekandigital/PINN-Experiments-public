# hgnn-nif-cloth

### Demo

| Videos (MP4/GIF) | Images |
| --- | --- |
| [comparison.mp4](hgnn-nif-cloth/artifacts/taichi_final/comparison.mp4)<br>[hgnn_nif_demo.mp4](hgnn-nif-cloth/artifacts/taichi_final/hgnn_nif_demo.mp4)<br>[hgnn_trust_window.mp4](hgnn-nif-cloth/artifacts/taichi_final/hgnn_trust_window.mp4)<br>[physics_demo.mp4](hgnn-nif-cloth/artifacts/taichi_final/physics_demo.mp4)<br>[physics_trust_window.mp4](hgnn-nif-cloth/artifacts/taichi_final/physics_trust_window.mp4)<br>[trust_comparison.mp4](hgnn-nif-cloth/artifacts/taichi_final/trust_comparison.mp4) | ![](hgnn-nif-cloth/artifacts/taichi_final/comparison.gif)<br>![](hgnn-nif-cloth/artifacts/taichi_final/poster.png) |


## Update (April 17, 2026)

- This project-space is now the fuller demo-oriented variant, with rollout metrics, outputs, demo-site assets, and Taichi media.
- Training, physics-data generation, and rollout/metrics scripts were added around the hybrid model workflow.
- Use this variant when you need the end-to-end animation and demo path rather than the lighter base tree.

## Overview
This project-space contains the hgnn-nif-cloth implementation and related resources.

## Structure

```
hgnn-nif-cloth__project-space/
├── README.md                 # This file
├── hgnn-nif-cloth/          # Source code
│   ├── src/
│   ├── notebooks/
│   ├── configs/
│   ├── checkpoints/          # Model weights (from vast.ai backups)
│   └── results/              # Benchmark outputs
├── docker/                   # Docker definitions
│   ├── Dockerfile
│   └── scripts/
├── docker-backups/           # Local backups (not in git)
│   ├── files/
│   └── archived/
└── docs/                     # Research documentation
```

## Quick Start

```bash
cd hgnn-nif-cloth
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
