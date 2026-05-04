# PINN-Experiments

## Demos

### 01-GeoPINN-Manifold
<video src="projects/01-GeoPINN-Manifold__project-space/GeoPINN-Manifold/artifacts/taichi_final/comparison.mp4" controls="controls" style="max-width: 100%;"></video>

### 02-WavePINN-NIF-ComplexMedia
<video src="projects/02-WavePINN-NIF-ComplexMedia__project-space/WavePINN-NIF-ComplexMedia/artifacts/taichi_final/comparison.mp4" controls="controls" style="max-width: 100%;"></video>

### 05-clothgnn
<video src="projects/05-clothgnn__project-space/clothgnn/artifacts/taichi_final/comparison.mp4" controls="controls" style="max-width: 100%;"></video>

### 08-hgnn-clothdyn
<video src="projects/08-hgnn-clothdyn__project-space/hgnn-clothdyn/artifacts/taichi_final/comparison.mp4" controls="controls" style="max-width: 100%;"></video>

### 09-hgnn-nif-cloth
<video src="projects/09-hgnn-nif-cloth__project-space_animation/hgnn-nif-cloth/artifacts/taichi_final/comparison.mp4" controls="controls" style="max-width: 100%;"></video>

### 12-nif-cloth4d-temporal
<video src="projects/12-nif-cloth4d-temporal__project-space/nif-cloth4d-temporal/artifacts/taichi_final/comparison.mp4" controls="controls" style="max-width: 100%;"></video>

### 13-nif-cloth4d
<video src="projects/13-nif-cloth4d__project-space/nif-cloth4d/artifacts/taichi_final/comparison.mp4" controls="controls" style="max-width: 100%;"></video>

### 17-wavepinn-nif
<video src="projects/17-wavepinn-nif__project-space/wavepinn-nif/artifacts/taichi_final/wave_demo.mp4" controls="controls" style="max-width: 100%;"></video>



A monorepo containing Physics-Informed Neural Network (PINN), Graph Neural Network (GNN), and Neural Implicit Field (NIF) experiments for scientific computing and physics simulation.

## Update (April 17, 2026)

- The repo now includes shared infrastructure under `shared/`, `shared_training/`, `benchmark_harness/`, `tools/distillation/`, `implicit_fields/`, and `data/coastal/`.
- Demo and export surfaces expanded across the monorepo, especially in projects 01, 05, 08, 09-animation, 13, and 17.
- Workflow docs for key management and private-to-public sync are now part of normal repo operations.

## Repository Structure

```
PINN-Experiments/
├── .gitignore
├── README.md
├── WORKFLOW.md                           # Vast.ai workflow documentation
├── vastai-manage.sh                      # Instance management script
│
├── projects/
│   │
│   │ ─── PINN Projects ────────────────────────────────────────────────
│   ├── pinn-lite-foil__project-space/    # Airfoil flow prediction
│   ├── maxwell-pinn-nif__project-space/  # Maxwell equations (electromagnetics)
│   ├── GeoPINN-Manifold__project-space/  # Geometric learning on manifolds
│   ├── cell-path-pinns__project-space/   # Biological pathway PINNs
│   ├── surfpinn__project-space/          # Surface physics
│   ├── wavepinn-nif__project-space/      # Acoustic wave propagation
│   ├── WavePINN-NIF-ComplexMedia__project-space/
│   │
│   │ ─── GNN Projects ─────────────────────────────────────────────────
│   ├── coastflow-gnn__project-space/     # Coastal flow GNN
│   ├── clothgnn__project-space/          # Cloth simulation GNN
│   ├── hgnn-clothdyn__project-space/     # Hierarchical GNN cloth dynamics
│   ├── pegnn-deform__project-space/      # Physics-Enhanced GNN deformation
│   │
│   │ ─── NIF Projects ─────────────────────────────────────────────────
│   ├── geom-inr-motion__project-space/   # Geometry INR motion
│   ├── clothgeom-nif__project-space/     # Cloth geometry NIF
│   ├── nif-cloth4d__project-space/       # 4D cloth NIF
│   ├── nif-cloth4d-temporal__project-space/
│   ├── nif-cloth3d__project-space/
│   │
│   │ ─── Hybrid Projects ──────────────────────────────────────────────
│   └── hgnn-nif-cloth__project-space/    # Combined HGNN + NIF for cloth
│
└── research-docs/                        # All ChatGPT research documentation
    ├── parts1-originals/                 # Original text exports
    └── *.md, *.txt                       # Research notes and plans
```

## Project Spaces

Each project follows a standardized structure:

```
<project-name>__project-space/
├── README.md                             # Project documentation
├── <project-name>/                       # Source code
│   ├── src/
│   ├── notebooks/
│   ├── configs/
│   ├── checkpoints/                      # Model weights (from vast.ai)
│   └── results/                          # Benchmark outputs
├── docker/                               # Docker definitions
│   ├── Dockerfile
│   └── scripts/
├── docker-backups/                       # Local backups (NOT in git)
│   ├── files/                            # Workspace snapshots
│   └── archived/                         # Milestone images
└── docs/                                 # Research documentation
```

## Project Categories

### PINN Projects (Physics-Informed Neural Networks)
| Project | Domain | Description |
|---------|--------|-------------|
| pinn-lite-foil | Aerodynamics | Lightweight PINN for airfoil flow |
| maxwell-pinn-nif | Electromagnetics | Maxwell equations solver |
| GeoPINN-Manifold | Geometry | PDEs on manifolds |
| cell-path-pinns | Biology | Cell pathway modeling |
| surfpinn | Surface Physics | Free-surface water simulation |
| wavepinn-nif | Acoustics | Acoustic wave propagation |

### GNN Projects (Graph Neural Networks)
| Project | Domain | Description |
|---------|--------|-------------|
| coastflow-gnn | Coastal Engineering | Coastal flow prediction |
| clothgnn | Cloth Simulation | Basic cloth GNN |
| hgnn-clothdyn | Cloth Simulation | Hierarchical cloth dynamics |
| pegnn-deform | Soft-body | Physics-encoded deformation |

### NIF Projects (Neural Implicit Fields)
| Project | Domain | Description |
|---------|--------|-------------|
| geom-inr-motion | Motion | Geometry-aware motion |
| clothgeom-nif | Cloth | Cloth geometry representation |
| nif-cloth4d | Cloth | 4D temporal cloth |

## Vast.ai Workflow

See `WORKFLOW.md` for complete documentation on:
- Instance lifecycle management
- Docker image backups
- File backup procedures
- Artifact preservation

## Quick Start

```bash
# Navigate to a project
cd projects/pinn-lite-foil__project-space/pinn-lite-foil

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
```

## License

See individual project directories for licensing information.
