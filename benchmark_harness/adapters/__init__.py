"""
Project adapters for the benchmark harness.

Each adapter implements the ProjectAdapter interface for a specific project.
"""

from pathlib import Path
from typing import Optional, Dict, Type, Any

# Import all adapters
from .p01_geopinn_manifold import P01Adapter
from .p02_wavepinn_nif_complexmedia import P02WavePINNNIFAdapter
from .p03_cell_path_pinns import P03Adapter
from .p04_clothgeom_nif import P04ClothGeomNIFAdapter
from .p05_clothgnn import P05ClothGNNAdapter
from .p06_coastflow_gnn import P06CoastFlowGNNAdapter
from .p07_geom_inr_motion import P07GeomINRMotionAdapter
from .p08_hgnn_clothdyn import P08HGNNClothDynAdapter
from .p09_hgnn_nif_cloth import P09Adapter
from .p10_maxwell_pinn_nif import P10MaxwellPINNNIFAdapter
from .p11_nif_cloth3d import P11NIFCloth3DAdapter
from .p12_nif_cloth4d_temporal import P12NIFCloth4DTemporalAdapter
from .p13_nif_cloth4d import P13Adapter
from .p14_pegnn_deform import P14PEGNNDeformAdapter
from .p15_pinn_lite_foil import P15PINNLiteFoilAdapter
from .p16_surfpinn import P16SurfPINNAdapter
from .p17_wavepinn_scalar import P17WavePINNScalarAdapter

# Aliases for consistent naming
P01GeoPINNManifoldAdapter = P01Adapter
P03CellPathPINNsAdapter = P03Adapter
P09HGNNNIFClothAdapter = P09Adapter
P13NIFCloth4DAdapter = P13Adapter

# Adapter registry - maps project IDs to adapter classes
ADAPTERS: Dict[str, Type] = {
    "P01": P01Adapter,
    "P02": P02WavePINNNIFAdapter,
    "P03": P03Adapter,
    "P04": P04ClothGeomNIFAdapter,
    "P05": P05ClothGNNAdapter,
    "P06": P06CoastFlowGNNAdapter,
    "P07": P07GeomINRMotionAdapter,
    "P08": P08HGNNClothDynAdapter,
    "P09": P09Adapter,
    "P10": P10MaxwellPINNNIFAdapter,
    "P11": P11NIFCloth3DAdapter,
    "P12": P12NIFCloth4DTemporalAdapter,
    "P13": P13Adapter,
    "P14": P14PEGNNDeformAdapter,
    "P15": P15PINNLiteFoilAdapter,
    "P16": P16SurfPINNAdapter,
    "P17": P17WavePINNScalarAdapter,
}

# Project paths mapping
PROJECT_PATHS = {
    "P01": "01-GeoPINN-Manifold__project-space",
    "P02": "02-WavePINN-NIF-ComplexMedia__project-space",
    "P03": "03-cell-path-pinns__project-space",
    "P04": "04-clothgeom-nif__project-space",
    "P05": "05-clothgnn__project-space",
    "P06": "06-coastflow-gnn__project-space",
    "P07": "07-geom-inr-motion__project-space",
    "P08": "08-hgnn-clothdyn__project-space",
    "P09": "09-hgnn-nif-cloth__project-space",
    "P10": "10-maxwell-pinn-nif__project-space",
    "P11": "11-nif-cloth3d__project-space",
    "P12": "12-nif-cloth4d-temporal__project-space",
    "P13": "13-nif-cloth4d__project-space",
    "P14": "14-pegnn-deform__project-space",
    "P15": "15-pinn-lite-foil__project-space",
    "P16": "16-surfpinn__project-space",
    "P17": "17-wavepinn-nif__project-space",
}


def get_adapter(project_id: str, projects_root: Optional[Path] = None, config: Any = None):
    """
    Get adapter instance for a project.
    
    Args:
        project_id: Project identifier (e.g., "P01", "P09")
        projects_root: Root path containing all project directories
        config: Optional BenchmarkConfig
    
    Returns:
        Adapter instance
    """
    # Normalize project ID
    pid = project_id.upper()
    if pid.startswith("P") and pid[1:].isdigit():
        pid = f"P{int(pid[1:]):02d}"  # Normalize P1 -> P01
    
    # Handle variants (e.g., P09-anim, P15-onnx)
    base_pid = pid.split("-")[0] if "-" in pid else pid
    
    adapter_class = ADAPTERS.get(base_pid)
    if adapter_class is None:
        raise ValueError(f"No adapter found for project: {project_id}. "
                        f"Available: {list(ADAPTERS.keys())}")
    
    # Determine project path
    if projects_root is None:
        projects_root = Path(__file__).parent.parent.parent / "projects"
    
    project_path = projects_root / PROJECT_PATHS.get(base_pid, "")
    
    return adapter_class(project_path)


def list_available_adapters() -> Dict[str, str]:
    """List all available adapters with their project names."""
    adapters_info = {}
    for pid, adapter_class in ADAPTERS.items():
        try:
            # Get project name from adapter
            name = getattr(adapter_class, 'PROJECT_NAME', pid)
            adapters_info[pid] = name
        except Exception:
            adapters_info[pid] = pid
    return adapters_info


__all__ = [
    "ADAPTERS",
    "PROJECT_PATHS",
    "get_adapter",
    "list_available_adapters",
    # Individual adapters
    "P01GeoPINNManifoldAdapter",
    "P02WavePINNNIFAdapter",
    "P03Adapter",
    "P04ClothGeomNIFAdapter",
    "P05ClothGNNAdapter",
    "P06CoastFlowGNNAdapter",
    "P07GeomINRMotionAdapter",
    "P08HGNNClothDynAdapter",
    "P09Adapter",
    "P10MaxwellPINNNIFAdapter",
    "P11NIFCloth3DAdapter",
    "P12NIFCloth4DTemporalAdapter",
    "P13Adapter",
    "P14PEGNNDeformAdapter",
    "P15PINNLiteFoilAdapter",
    "P16SurfPINNAdapter",
    "P17WavePINNScalarAdapter",
]
