# coastflow-gnn

## Update (April 17, 2026)

- Geometry support expanded under `src/geometry`, with more explicit differential-geometry reuse from the shared repo layers.
- The project now aligns more closely with `data/coastal/` for bathymetry, forcing, mesh, and validation inputs.
- This README update reflects the shift toward shared coastal and export infrastructure.

## Overview
This project-space contains the coastflow-gnn implementation and related resources.

## Structure

```
coastflow-gnn__project-space/
├── README.md                 # This file
├── coastflow-gnn/          # Source code
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
cd coastflow-gnn
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
