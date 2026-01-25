# maxwell-pinn-nif

## Overview
This project-space contains the maxwell-pinn-nif implementation and related resources.

## Structure

```
maxwell-pinn-nif__project-space/
├── README.md                 # This file
├── maxwell-pinn-nif/          # Source code
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
cd maxwell-pinn-nif
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
