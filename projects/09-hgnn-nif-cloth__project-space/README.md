# hgnn-nif-cloth

## Update (April 17, 2026)

- The core hybrid project received code-level refreshes in model exports, SIREN support, and training losses.
- Docker requirements were updated to match the current training and runtime expectations.
- For the richer rollout and demo pipeline, see the `_animation` project-space variant tracked beside this one.

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
