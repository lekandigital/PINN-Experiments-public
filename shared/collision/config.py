"""Configuration dataclasses for the collision system."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch


@dataclass
class SDFConfig:
    """Configuration for SDF field computation and queries."""
    
    resolution: int = 128
    """Grid resolution for SDF computation (64, 128, or 256)."""
    
    padding: float = 0.1
    """Padding around mesh bounding box as fraction of bbox size."""
    
    sign_method: str = 'pseudonormal'
    """Method for computing sign: 'pseudonormal' (robust) or 'raycast' (fast)."""
    
    use_cached_bvh: bool = True
    """Whether to cache BVH structure for repeated queries."""
    
    device: str = 'cuda'
    """Device for SDF computation ('cuda' or 'cpu')."""
    
    dtype: torch.dtype = torch.float32
    """Data type for SDF grid."""
    
    @property
    def grid_size(self) -> int:
        """Total number of grid cells."""
        return self.resolution ** 3
    
    @property
    def memory_mb(self) -> float:
        """Estimated memory usage in MB."""
        bytes_per_float = 4 if self.dtype == torch.float32 else 2
        return (self.grid_size * bytes_per_float) / (1024 * 1024)


@dataclass
class CollisionConfig:
    """Configuration for collision detection and response."""
    
    # Detection parameters
    proximity_threshold: float = 0.005
    """Distance threshold for proximity detection (normalized coordinates)."""
    
    enable_proximity_detection: bool = True
    """Whether to flag vertices close to but not penetrating the surface."""
    
    enable_edge_edge_detection: bool = False
    """Whether to perform continuous collision detection for edge-edge intersections."""
    
    enable_self_collision: bool = False
    """Whether to detect cloth self-intersections."""
    
    self_collision_min_distance: float = 0.005
    """Minimum allowed distance for self-collision."""
    
    self_collision_sample_ratio: float = 0.1
    """Fraction of vertex-face pairs to sample for self-collision."""
    
    # Response parameters
    stiffness: float = 1000.0
    """Collision response stiffness for force-based correction."""
    
    friction: float = 0.3
    """Coulomb friction coefficient (0 = frictionless, 1 = sticky)."""
    
    damping: float = 0.1
    """Velocity damping near contact surfaces."""
    
    max_correction: float = 0.1
    """Maximum position correction per step (prevents explosions)."""
    
    # Loss weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'penetration': 10.0,
        'proximity': 1.0,
        'contact': 0.5,
        'eikonal': 0.1,
    })
    """Weights for different collision loss components."""
    
    proximity_margin: float = 0.01
    """Soft buffer zone distance for proximity loss."""
    
    # SDF configuration
    sdf: SDFConfig = field(default_factory=SDFConfig)
    """SDF field configuration."""
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.proximity_threshold > 0, "proximity_threshold must be positive"
        assert 0 <= self.friction <= 1, "friction must be in [0, 1]"
        assert 0 <= self.damping <= 1, "damping must be in [0, 1]"
        assert self.sdf.resolution in [32, 64, 128, 256, 512], \
            "resolution must be 32, 64, 128, 256, or 512"


@dataclass
class BodyConfig:
    """Configuration for deformable body interface."""
    
    sdf_resolution: int = 128
    """Resolution for body SDF computation."""
    
    update_every_frame: bool = True
    """Whether to recompute SDF every frame or use interpolation."""
    
    velocity_smoothing: float = 0.5
    """Temporal smoothing for velocity field estimation (0 = no smoothing)."""
    
    use_skinning_fallback: bool = True
    """Whether to support LBS skinning when PEGNN is not available."""
    
    cache_previous_frames: int = 2
    """Number of previous frames to cache for velocity estimation."""
