# ClothGeom-NIF

> ⚠️ **ARCHIVED**: This project has been absorbed into Project 09 (HGNN-NIF-Cloth).
> 
> The core contributions of this project have been extracted into shared modules:
> - Mesh extraction (`MeshExtractor`, `MeshExtractionConfig`) → `implicit_fields/mesh_extraction.py`
> - SIREN architecture → `implicit_fields/siren.py` (already shared)
> - NIFDecoder with variance output → `projects/09-hgnn-nif-cloth/src/models/nif_decoder.py`
> - Inverse design optimization → `projects/09-hgnn-nif-cloth/src/inverse/pose_matching.py`
>
> **Do not develop further.** Use Project 09 for cloth simulation work.

---

## Update (April 17, 2026)

- Recent private-head work refreshed the project docs and SIREN-centered model implementation.
- This project continues to anchor the cloth-geometry implicit-field path within the repo's broader `implicit_fields` work.
- The README now calls out the current monorepo context without changing the project's basic usage model.

## Original Description

SIREN-based SDF decoder for static cloth geometry.

**Performance:**
- ~593K parameters
- Trains in 4.2 minutes on RTX 3090
- 32/32 tests passing
- Produces watertight meshes via marching cubes
- Inverse design optimization (optimize latent → target shape)

## What Was Extracted

### To `implicit_fields/mesh_extraction.py`
- `MeshExtractor` class - Stateful extractor for latent-conditioned models
- `MeshExtractionConfig` - Configuration dataclass
- Progressive multi-resolution extraction
- Batched SDF evaluation for memory efficiency

### To Project 09 (HGNN-NIF-Cloth)
- `NIFDecoder` - Full decoder with SDF + variance dual heads
- `InverseDesignOptimizer` - SDF volume matching
- `PoseMatchingOptimizer` - Point cloud matching for captured data
- `interpolate_latents` - Latent space interpolation for animation

## Migration Guide

If you were using this project, update your imports:

```python
# Old (Project 04)
from models import NIFDecoder
from inference.mesh_extractor import MeshExtractor, MeshExtractionConfig
from inverse_design import InverseDesignOptimizer

# New (shared modules)
from implicit_fields import MeshExtractor, MeshExtractionConfig
from implicit_fields import extract_mesh, laplacian_smooth, compute_mesh_quality

# New (Project 09)
from src.models import NIFDecoder, create_nif_decoder
from src.inverse import InverseDesignOptimizer, PoseMatchingOptimizer
```

## Why This Project Was Absorbed

Project 04 (ClothGeom-NIF) focused on **static cloth geometry** - learning SDFs without dynamics. Project 09 (HGNN-NIF-Cloth) is a superset that includes:

1. **All P04 capabilities** - SDF learning, mesh extraction, inverse design
2. **Temporal dynamics** - Time-conditioned models for cloth animation
3. **Hierarchical GNN** - Multi-resolution cloth simulation
4. **Physics constraints** - Energy-based dynamics

Since P04's code was production-ready but didn't add unique value beyond what P09 needs, it made sense to consolidate.

---

## Structure (Preserved for Reference)

```
clothgeom-nif__project-space/
├── README.md                 # This file
├── clothgeom-nif/          # Source code
│   ├── models/               # NIFDecoder, SIREN layers
│   ├── inference/            # MeshExtractor
│   ├── inverse_design.py     # Inverse optimization
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation
│   ├── tests/                # Test suite (32 tests)
│   └── configs/              # Training configurations
├── docker/                   # Docker definitions
└── docs/                     # Research documentation
```

## Running Tests (Verification)

The original tests can still be run to verify the codebase works:

```bash
cd clothgeom-nif
pip install -r requirements.txt
python -m pytest tests/ -v
```

These tests verify the original implementation. For ongoing development, use Project 09's test suite.
