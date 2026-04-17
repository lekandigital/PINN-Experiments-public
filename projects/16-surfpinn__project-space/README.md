# surfpinn

## Update (April 17, 2026)

- Geometry support was expanded in the current project implementation.
- SurfPINN now sits closer to the repo's coastal-data and shared-geometry infrastructure than earlier README versions implied.
- This doc refresh keeps the project aligned with the current monorepo context.

## Overview
This project-space contains the surfpinn implementation and related resources.

## Structure

```
surfpinn__project-space/
├── README.md                 # This file
├── surfpinn/          # Source code
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
cd surfpinn
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
