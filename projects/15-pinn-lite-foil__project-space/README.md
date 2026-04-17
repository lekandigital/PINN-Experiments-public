# pinn-lite-foil

## Update (April 17, 2026)

- This pass mainly syncs the docs with the current monorepo structure, workflow docs, and shared utilities.
- The project itself did not receive the same level of recent reshaping as projects 13 and 17.
- Shared benchmarking and distillation infrastructure is now available at the repo level when needed.

## Overview
This project-space contains the pinn-lite-foil implementation and related resources.

## Structure

```
pinn-lite-foil__project-space/
├── README.md                 # This file
├── pinn-lite-foil/          # Source code
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
cd pinn-lite-foil
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
