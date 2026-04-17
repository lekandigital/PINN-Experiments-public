"""
Neural Simulation - Model-Agnostic Blender Addon

A professional-quality Blender addon that runs neural cloth, body, and motion
simulations in real-time. Supports multiple model backends through a pluggable
architecture.

Features:
- Multiple backend support: SIREN, GNN, HGNN, PEGNN models
- Real-time viewport preview with force controls
- SDF-based cloth extraction via marching cubes
- Vertex displacement for body deformation
- Skeleton animation for motion prediction
- Bake to keyframes for rendering/export

Supported Backends:
- NIF-Cloth3D (Legacy): 8D SIREN, vertex displacements
- NIF-Cloth4D (Preview): 4D SIREN, SDF output, ultra-fast
- HGNN-NIF-Cloth (High Quality): Hierarchical GNN + SIREN, SDF output
- ClothGNN (Lightweight): GRU-based GNN, vertex displacements
- Motion SIREN: Continuous-time skeleton animation
- PEGNN-Deform: Physics-encoded GNN for soft body

Installation:
1. Install dependencies: pip install torch numpy scikit-image
2. Install addon in Blender: Edit > Preferences > Add-ons > Install
3. Enable "Neural Simulation" addon
4. Find panel in 3D View > Sidebar > Neural Sim tab

Usage:
1. Select a backend from the dropdown
2. Choose or browse to a checkpoint file
3. Select target mesh/armature
4. Click "Load Model"
5. Configure forces/materials
6. Press Play or step through frames

Author: PINN-Experiments Team
Version: 1.0.0
Blender: 3.6+
"""

bl_info = {
    "name": "Neural Simulation",
    "author": "PINN-Experiments Team",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Neural Sim",
    "description": "Real-time neural cloth, body, and motion simulation with pluggable backends",
    "category": "Physics",
    "doc_url": "https://github.com/PINN-Experiments/neural-sim-addon",
    "tracker_url": "https://github.com/PINN-Experiments/neural-sim-addon/issues",
}

import bpy
from pathlib import Path


# Module-level variables for lazy loading
_registered = False


def register():
    """Register the addon with Blender."""
    global _registered
    
    if _registered:
        return
    
    # Setup Python path for internal imports
    addon_dir = Path(__file__).parent
    import sys
    if str(addon_dir.parent) not in sys.path:
        sys.path.insert(0, str(addon_dir.parent))
    
    # Import and register UI classes
    from .ui import classes as ui_classes
    from .ui.properties import register_properties
    
    # Register properties first
    register_properties()
    
    # Register all UI classes
    for cls in ui_classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            # Already registered
            pass
    
    # Initialize backend manager
    try:
        from .backend import get_manager
        manager = get_manager()
        
        # Set adapters path
        adapters_path = addon_dir / "backend" / "adapters"
        manager.set_adapters_path(adapters_path)
        
        # Discover backends
        backends = manager.discover_backends()
        print(f"[NeuralSim] Found {len(backends)} backends: {backends}")
    except Exception as e:
        print(f"[NeuralSim] Backend initialization error: {e}")
    
    _registered = True
    print("[NeuralSim] Addon registered")


def unregister():
    """Unregister the addon from Blender."""
    global _registered
    
    if not _registered:
        return
    
    # Unregister frame handler
    try:
        from .core import unregister_frame_handler
        unregister_frame_handler()
    except:
        pass
    
    # Cleanup backend manager
    try:
        from .backend import reset_manager
        reset_manager()
    except:
        pass
    
    # Unregister UI classes
    try:
        from .ui import classes as ui_classes
        from .ui.properties import unregister_properties
        
        for cls in reversed(ui_classes):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
        
        unregister_properties()
    except:
        pass
    
    _registered = False
    print("[NeuralSim] Addon unregistered")


# Allow running as script for testing
if __name__ == "__main__":
    register()
