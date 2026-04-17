# Neural Simulation Addon

A professional-quality Blender addon that runs neural cloth, body, and motion simulations in real-time. Supports multiple model backends through a pluggable architecture.

## Features

- **Multiple Backend Support**: SIREN, GNN, HGNN, PEGNN models
- **Real-time Viewport Preview**: Interactive simulation with force controls
- **SDF-based Cloth Extraction**: Marching cubes for smooth surfaces
- **Vertex Displacement**: Direct mesh deformation
- **Skeleton Animation**: Continuous-time motion prediction
- **Bake to Keyframes**: Export for rendering/animation

## Supported Backends

| Backend | Project | Input | Output | Sequential |
|---------|---------|-------|--------|------------|
| NIF-Cloth3D (Legacy) | 11 | 8D coords | Displacements | No |
| NIF-Cloth4D | 13 | 4D coords | SDF Grid | No |
| HGNN-NIF-Cloth | 09 | Graph + coords | SDF Grid | Yes (GRU) |
| ClothGNN | 05 | Graph | Displacements | Yes (GRU) |
| Motion SIREN | 07 | Time | Joint Positions | No |
| PEGNN-Deform | 14 | Graph | Displacements | Yes |

## Requirements

- Blender 3.6 or higher
- Python 3.10+
- PyTorch 2.0+
- NumPy
- scikit-image (for SDF mesh extraction)
- torch-geometric (optional, for GNN backends)

## Installation

### 1. Install Python Dependencies

```bash
# From Blender's Python
/path/to/blender/python/bin/python -m pip install torch numpy scikit-image

# For GNN backends (optional)
/path/to/blender/python/bin/python -m pip install torch-geometric
```

### 2. Install Addon in Blender

1. Go to **Edit > Preferences > Add-ons > Install**
2. Select the `addon` folder (or zip it first)
3. Enable "Neural Simulation" in the addon list

## Usage

### Basic Workflow

1. Select a backend from the **Neural Sim** sidebar panel
2. Browse to a checkpoint file (`.pt` or `.pth`)
3. Click **Load Model**
4. Select your target mesh or armature
5. Configure forces and materials (if supported)
6. Press **Play** or step through frames

### Backend Selection

Each backend has different capabilities:

- **Cloth simulation**: Use NIF-Cloth variants
- **High quality**: HGNN-NIF-Cloth produces smoother results
- **Fast preview**: NIF-Cloth4D is ultra-fast but lower resolution
- **Motion capture**: Motion SIREN for skeleton animation
- **Soft body**: PEGNN-Deform for physics-based deformation

### Material Presets

The addon includes material presets for common fabrics:

- Silk (lightweight, flowing)
- Cotton (medium weight)
- Denim (heavy, stiff)
- Leather (thick, minimal drape)
- Chiffon (ultra-light, sheer)
- Wool (warm, moderate drape)
- Velvet (luxurious, soft pile)
- Lycra (stretchy, athletic)

### Baking Animation

To bake neural simulation to keyframes:

1. Set the frame range in playback panel
2. Click **Bake to Keyframes**
3. The addon creates shape keys (mesh) or bone keyframes (skeleton)

## Project Structure

```
addon/
├── __init__.py           # Addon entry point
├── backend/
│   ├── interface.py      # Abstract interface & data classes
│   ├── manager.py        # Backend discovery & lifecycle
│   └── adapters/         # Model-specific adapters
│       ├── adapter_05_cloth_gnn.py
│       ├── adapter_07_motion.py
│       ├── adapter_09_hgnn_cloth.py
│       ├── adapter_11_legacy.py
│       ├── adapter_13_nif_cloth.py
│       └── adapter_14_deform.py
├── core/
│   ├── mesh_bridge.py    # Blender ↔ numpy conversion
│   ├── frame_handler.py  # Animation system integration
│   ├── request_builder.py
│   ├── result_applier.py
│   └── performance.py    # FPS tracking
├── ui/
│   ├── properties.py     # Blender property groups
│   ├── panels.py         # Capability-driven UI panels
│   └── operators.py      # Blender operators
├── presets/
│   └── materials/        # Material preset JSON files
└── tests/
    ├── test_interface.py
    ├── test_manager.py
    ├── test_mesh_bridge.py
    └── test_blender_integration.py
```

## Adding a New Backend

1. Create a new adapter in `backend/adapters/`:

```python
from ..interface import (
    ModelBackend,
    BackendCapabilities,
    PredictionRequest,
    PredictionResult,
    ModelCategory,
    OutputFormat,
    InputRequirement,
)

class MyNewBackend(ModelBackend):
    BACKEND_ID = "my_backend"
    
    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="My Backend",
            version="1.0.0",
            category=ModelCategory.CLOTH,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            required_inputs={InputRequirement.MESH_VERTICES},
            # ... more capabilities
        )
    
    def load(self, checkpoint_path: str) -> bool:
        # Load your model
        return True
    
    def predict(self, request: PredictionRequest) -> PredictionResult:
        # Run inference
        pass
    
    def reset(self) -> None:
        # Reset state
        pass
```

2. The manager will auto-discover it on next addon load.

## Running Tests

### Unit Tests (no Blender required)

```bash
cd shared/addon
python -m pytest tests/test_interface.py tests/test_manager.py -v
```

### Blender Integration Tests

```bash
blender --background --python tests/test_blender_integration.py
```

## Troubleshooting

### "Import torch failed"

PyTorch needs to be installed in Blender's Python:
```bash
/path/to/blender/python/bin/python -m pip install torch
```

### "No backends found"

Check that adapters are in `backend/adapters/` and have `BACKEND_ID` class attribute.

### "Model shape mismatch"

The checkpoint may be incompatible. Check:
- Expected input dimensions match your mesh
- Model architecture matches the adapter

### Performance issues

- Reduce SDF resolution for faster previews
- Use stateless backends (NIF-Cloth4D) for real-time
- Bake to keyframes for final rendering

## License

MIT License - See LICENSE file for details.

## Credits

Developed as part of the PINN-Experiments research project.
