# nif-cloth3d

## Update (April 17, 2026)

- The model path was refreshed and shared collision hooks were added around the interactive cloth workflow.
- Project 11 now fits more cleanly into the shared addon and cloth-tooling story in the repo.
- This README update is mainly about keeping docs aligned with the current shared infrastructure.

## Overview
This project-space contains the nif-cloth3d implementation and related resources.

## Structure

```
nif-cloth3d__project-space/
├── README.md                 # This file
├── nif-cloth3d/          # Source code
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
cd nif-cloth3d
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
