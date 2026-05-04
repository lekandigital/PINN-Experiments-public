# nif-cloth4d

### Demo

<video src="nif-cloth4d/artifacts/taichi_final/comparison.mp4" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>


## Update (April 17, 2026)

- Project 13 is one of the most actively updated areas in the current private `main`.
- The project-space now includes an explicit `demo/` surface plus expanded pipeline helpers such as `scripts/run_full_pipeline.py`.
- Model and training code plus shared collision hookups were updated to support the current cloth workflow.

## Overview
This project-space contains the nif-cloth4d implementation and related resources.

## Structure

```
nif-cloth4d__project-space/
├── README.md                 # This file
├── nif-cloth4d/          # Source code
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
cd nif-cloth4d
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
