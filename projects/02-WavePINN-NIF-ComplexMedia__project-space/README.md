# WavePINN-NIF-ComplexMedia

### Demo

<video src="https://raw.githubusercontent.com/lekandigital/PINN-Experiments-public/main/projects/02-WavePINN-NIF-ComplexMedia__project-space/WavePINN-NIF-ComplexMedia/artifacts/taichi_final/comparison.mp4" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>


## Update (April 17, 2026)

- This project's docs were synced to the current monorepo shape and shared coastal-data context.
- The repo now expects real-world ocean and acoustics integrations to come through `data/coastal/` and the shared tooling layers.
- No large private-head layout rewrite landed here recently, so this refresh is intentionally documentation-focused.

## Overview
This project-space contains the WavePINN-NIF-ComplexMedia implementation and related resources.

## Structure

```
WavePINN-NIF-ComplexMedia__project-space/
├── README.md                 # This file
├── WavePINN-NIF-ComplexMedia/          # Source code
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
cd WavePINN-NIF-ComplexMedia
pip install -r requirements.txt  # if exists
python -m pytest tests/ -v       # run tests
```

## Related Documentation
See `docs/` directory for research notes and design documents.
