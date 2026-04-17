# pegnn-deform

## Update (April 17, 2026)

- Body-SDF export and physics-model integration work were added or refreshed in the current project tree.
- Project 14 now sits closer to the shared addon, collision, and export-pipeline infrastructure.
- This README update keeps the deform workflow aligned with the repo's current shared-tooling direction.

## Overview
This project-space contains the pegnn-deform implementation and related resources.

## Structure

```
pegnn-deform__project-space/
├── README.md                 # This file
├── pegnn-deform/          # Source code
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
cd pegnn-deform
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
