# nif-cloth4d-temporal

### Demo

<video src="nif-cloth4d-temporal/artifacts/taichi_final/comparison.mp4" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>


## Update (April 17, 2026)

- The temporal cloth stack received updates in both the Fourier MLP and SIREN model implementations.
- Shared collision integration is now part of the tracked project surface.
- This README refresh keeps the temporal training and export story aligned with the current repo layout.

## Overview
This project-space contains the nif-cloth4d-temporal implementation and related resources.

## Structure

```
nif-cloth4d-temporal__project-space/
├── README.md                 # This file
├── nif-cloth4d-temporal/          # Source code
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
cd nif-cloth4d-temporal
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
