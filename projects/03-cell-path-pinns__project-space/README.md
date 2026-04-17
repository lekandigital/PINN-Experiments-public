# cell-path-pinns

## Update (April 17, 2026)

- The implementation now centers on `src/core`, `src/domains`, and `src/export` modules instead of a looser single-file layout.
- New demos cover game-AI and robotics-style trajectory-planning flows.
- Test coverage was expanded around the core, domain, and trajectory modules.

## Overview
This project-space contains the cell-path-pinns implementation and related resources.

## Structure

```
cell-path-pinns__project-space/
├── README.md                 # This file
├── cell-path-pinns/          # Source code
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
cd cell-path-pinns
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
