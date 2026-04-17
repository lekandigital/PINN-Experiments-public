# maxwell-pinn-nif

## Update (April 17, 2026)

- This refresh aligns the docs with the current monorepo workflow and shared utility layers.
- The electromagnetics project did not undergo a major private-head layout shift compared with projects 13 and 17.
- Shared benchmarking and export patterns can now be referenced from repo-level tooling where useful.

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
