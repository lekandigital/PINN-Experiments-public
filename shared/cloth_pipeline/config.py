"""
Configuration system for the cloth simulation pipeline.

Uses Pydantic for validation and YAML for config files.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal, Union
from pathlib import Path
import yaml
import hashlib
import json


@dataclass
class IngestConfig:
    """Configuration for data ingestion stage."""
    format: Literal["abc", "obj_sequence", "blender"] = "obj_sequence"
    source_path: str = ""
    file_pattern: str = "*.obj"  # or "*.abc" or regex for OBJ sequences
    fps: float = 24.0
    rest_frame: int = 0  # Which frame is the rest pose
    
    # OBJ sequence specific
    obj_naming_pattern: str = r"frame_(\d+)\.obj"  # Regex for frame number extraction
    
    # Alembic specific
    abc_mesh_path: str = ""  # Path within the Alembic hierarchy, empty = auto-detect
    
    # Blender specific
    blender_object_name: str = ""  # Name of cloth object in Blender scene
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None


@dataclass
class GraphConfig:
    """Configuration for graph construction."""
    compute_multihop: bool = True
    max_hops: int = 3
    coarsening_method: Literal["graclus", "metis", "greedy"] = "greedy"
    num_coarsening_levels: int = 2
    coarsening_ratio: float = 0.5  # Target ratio of vertices at each level


@dataclass
class SDFConfig:
    """Configuration for SDF computation."""
    method: Literal["thickened", "unsigned", "both"] = "thickened"
    thickness: float = 0.002  # meters, for thickened method
    grid_resolution: int = 64
    num_point_samples: int = 50000
    near_surface_ratio: float = 0.7  # 70% samples near surface, 30% in volume
    near_surface_distance: float = 0.01  # How close is "near surface" in meters
    
    # Bounding box expansion (relative to mesh bbox)
    bbox_padding: float = 0.1  # 10% padding on each side


@dataclass
class TemporalConfig:
    """Configuration for temporal processing."""
    window_length: int = 30  # frames per training window
    window_stride: int = 10  # frames between window starts
    compute_velocities: bool = True
    compute_accelerations: bool = True
    velocity_method: Literal["forward", "backward", "central"] = "central"


@dataclass
class FeatureConfig:
    """Configuration for vertex feature computation."""
    compute_curvature: bool = True
    compute_normals: bool = True
    curvature_method: Literal["mean", "gaussian", "both"] = "mean"


@dataclass
class NormalizationConfig:
    """Configuration for normalization."""
    center_rest_pose: bool = True
    compute_global_stats: bool = True
    normalize_positions: bool = False  # If True, divide by std
    normalize_velocities: bool = False


@dataclass
class TransformConfig:
    """Combined transform configuration."""
    graph: GraphConfig = field(default_factory=GraphConfig)
    sdf: SDFConfig = field(default_factory=SDFConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


@dataclass
class Project04ExportConfig:
    """Export config for Project 04 (ClothGeom-NIF) - Static SDF."""
    enabled: bool = False
    sdf_resolution: int = 64
    frames: str = "keyframes"  # "all", "keyframes", or "every_nth:N"
    latent_dim: int = 128  # Latent code dimension
    include_edge_strains: bool = True


@dataclass 
class Project05ExportConfig:
    """Export config for Project 05 (ClothGNN) - Graph Sequence."""
    enabled: bool = False
    feature_set: List[str] = field(default_factory=lambda: ["position", "velocity", "acceleration"])
    node_feature_dim: int = 16  # Pad to this dimension


@dataclass
class Project08ExportConfig:
    """Export config for Project 08 (HGNN-ClothDyn) - Hierarchical Graph."""
    enabled: bool = False
    feature_set: List[str] = field(default_factory=lambda: ["position", "velocity", "acceleration"])
    include_hierarchy: bool = True
    include_rest_lengths: bool = True
    include_collision_mask: bool = False


@dataclass
class Project09ExportConfig:
    """Export config for Project 09 (HGNN-NIF-Cloth) - Hybrid Graph+SDF."""
    enabled: bool = False
    sdf_samples_per_frame: int = 50000
    include_body_collision: bool = True
    fine_mesh_resolution: int = 40  # Grid dimensions (40x40 = 1600 vertices)
    coarse_mesh_resolution: int = 20  # Coarse level (20x20 = 400 vertices)


@dataclass
class Project11ExportConfig:
    """Export config for Project 11 (NIF-Cloth3D-Interactive) - Blender Integration."""
    enabled: bool = False
    include_forces: bool = True
    include_material_id: bool = True
    default_material_id: int = 0


@dataclass
class Project12ExportConfig:
    """Export config for Project 12 (NIF-Cloth4D-Temporal) - Spacetime SDF."""
    enabled: bool = False
    spacetime_samples: int = 100000
    temporal_interpolation: bool = True
    interpolation_substeps: int = 2  # Interpolated frames between real frames


@dataclass
class Project13ExportConfig:
    """Export config for Project 13 (NIF-Cloth4D) - Compact 4D SDF."""
    enabled: bool = False
    spacetime_samples: int = 10000  # Fewer samples for this compact model
    sdf_resolution: int = 64


@dataclass
class ExportTargetsConfig:
    """Configuration for all export targets."""
    project_04: Project04ExportConfig = field(default_factory=Project04ExportConfig)
    project_05: Project05ExportConfig = field(default_factory=Project05ExportConfig)
    project_08: Project08ExportConfig = field(default_factory=Project08ExportConfig)
    project_09: Project09ExportConfig = field(default_factory=Project09ExportConfig)
    project_11: Project11ExportConfig = field(default_factory=Project11ExportConfig)
    project_12: Project12ExportConfig = field(default_factory=Project12ExportConfig)
    project_13: Project13ExportConfig = field(default_factory=Project13ExportConfig)


@dataclass
class ExportConfig:
    """Configuration for export stage."""
    output_root: str = "./training_data"
    targets: ExportTargetsConfig = field(default_factory=ExportTargetsConfig)
    
    # Common export options
    compress: bool = True  # Use compression for HDF5/NPZ
    overwrite: bool = False  # Overwrite existing exports
    export_metadata: bool = True  # Write metadata JSON alongside exports


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.
    
    This is the top-level config loaded from YAML.
    """
    ingest: IngestConfig = field(default_factory=IngestConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Pipeline-level options
    verbose: bool = True
    num_workers: int = 4  # For parallel processing
    cache_intermediates: bool = True  # Cache transform results
    cache_dir: str = ".cloth_pipeline_cache"
    
    def get_hash(self) -> str:
        """Get a hash of this configuration for caching."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary."""
        return _dataclass_to_dict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create from nested dictionary."""
        return _dict_to_dataclass(cls, data)


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass to dict."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def _dict_to_dataclass(cls: type, data: Dict[str, Any]) -> Any:
    """Recursively convert dict to dataclass."""
    if not hasattr(cls, '__dataclass_fields__'):
        return data
    
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    
    for field_name, field_type in field_types.items():
        if field_name not in data:
            continue
        
        value = data[field_name]
        
        # Handle nested dataclasses
        if hasattr(field_type, '__dataclass_fields__'):
            kwargs[field_name] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[field_name] = value
    
    return cls(**kwargs)


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        PipelineConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        data = {}
    
    return PipelineConfig.from_dict(data)


def save_config(config: PipelineConfig, config_path: Union[str, Path]) -> None:
    """
    Save pipeline configuration to a YAML file.
    
    Args:
        config: PipelineConfig instance to save
        config_path: Path to write the YAML file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def create_default_config() -> PipelineConfig:
    """Create a default pipeline configuration."""
    return PipelineConfig()


def create_example_config_yaml() -> str:
    """
    Generate example YAML configuration string.
    
    Returns:
        YAML string with example configuration and comments
    """
    return """# Cloth Pipeline Configuration
# =============================
# This configuration controls the cloth simulation data pipeline.
# It processes professional simulation caches (Houdini Vellum, Marvelous Designer, Blender)
# into training data for physics-informed ML models.

# Input configuration
ingest:
  format: "obj_sequence"  # Options: "abc", "obj_sequence", "blender"
  source_path: "/path/to/simulation_caches/"
  file_pattern: "frame_*.obj"
  fps: 24.0
  rest_frame: 0  # Which frame represents the rest pose
  
  # For OBJ sequences:
  obj_naming_pattern: "frame_(\\d+)\\.obj"  # Regex to extract frame numbers
  
  # For Alembic:
  # abc_mesh_path: "/cloth/mesh"  # Path in Alembic hierarchy
  
  # For Blender:
  # blender_object_name: "Cloth"
  # frame_start: 1
  # frame_end: 250

# Transform configuration
transform:
  graph:
    compute_multihop: true
    max_hops: 3
    coarsening_method: "greedy"  # Options: "graclus", "metis", "greedy"
    num_coarsening_levels: 2
    coarsening_ratio: 0.5
  
  sdf:
    method: "thickened"  # Options: "thickened", "unsigned", "both"
    thickness: 0.002  # meters
    grid_resolution: 64
    num_point_samples: 50000
    near_surface_ratio: 0.7
    bbox_padding: 0.1
  
  temporal:
    window_length: 30
    window_stride: 10
    compute_velocities: true
    compute_accelerations: true
    velocity_method: "central"
  
  features:
    compute_curvature: true
    compute_normals: true
    curvature_method: "mean"
  
  normalization:
    center_rest_pose: true
    compute_global_stats: true

# Export configuration
export:
  output_root: "/path/to/training_data/"
  compress: true
  overwrite: false
  export_metadata: true
  
  targets:
    # Project 04: ClothGeom-NIF (Static SDF)
    project_04:
      enabled: false
      sdf_resolution: 64
      frames: "keyframes"  # Options: "all", "keyframes", "every_nth:5"
      latent_dim: 128
    
    # Project 05: ClothGNN (Graph Sequence)
    project_05:
      enabled: false
      feature_set: ["position", "velocity", "acceleration"]
      node_feature_dim: 16
    
    # Project 08: HGNN-ClothDyn (Hierarchical Graph)
    project_08:
      enabled: false
      feature_set: ["position", "velocity", "acceleration"]
      include_hierarchy: true
      include_rest_lengths: true
    
    # Project 09: HGNN-NIF-Cloth (Hybrid Graph+SDF) - FLAGSHIP
    project_09:
      enabled: true
      sdf_samples_per_frame: 50000
      include_body_collision: true
      fine_mesh_resolution: 40
      coarse_mesh_resolution: 20
    
    # Project 11: NIF-Cloth3D-Interactive (Blender)
    project_11:
      enabled: false
      include_forces: true
      include_material_id: true
    
    # Project 12: NIF-Cloth4D-Temporal (Spacetime SDF)
    project_12:
      enabled: false
      spacetime_samples: 100000
      temporal_interpolation: true
    
    # Project 13: NIF-Cloth4D (Compact 4D SDF)
    project_13:
      enabled: false
      spacetime_samples: 10000

# Pipeline options
verbose: true
num_workers: 4
cache_intermediates: true
cache_dir: ".cloth_pipeline_cache"
"""
