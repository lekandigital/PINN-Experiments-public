"""
Pipeline configuration dataclasses and YAML loading.

Defines how the three stages communicate and the data formats used.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# SMPL 24-joint skeleton definition
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand"
]

SMPL_PARENTS = [
    -1, 0, 0, 0,
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 9, 9, 12,
    13, 14, 16, 17,
    18, 19, 20, 21
]


class MotionOutputFormat(Enum):
    """Format of motion model output."""
    JOINT_POSITIONS_GLOBAL = auto()
    JOINT_ROTATIONS_LOCAL_EULER = auto()
    JOINT_ROTATIONS_LOCAL_QUAT = auto()
    JOINT_TRANSFORMS_GLOBAL = auto()


@dataclass
class SkeletonSpec:
    """Skeleton specification for the pipeline."""
    num_joints: int = 24
    joint_names: List[str] = field(default_factory=lambda: SMPL_JOINT_NAMES.copy())
    parent_indices: List[int] = field(default_factory=lambda: SMPL_PARENTS.copy())
    rest_positions: List[List[float]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.rest_positions:
            self.rest_positions = self._default_rest_positions()
    
    def _default_rest_positions(self) -> List[List[float]]:
        """Default T-pose rest positions for SMPL skeleton."""
        return [
            [0.0, 0.9, 0.0],      # pelvis
            [-0.1, 0.85, 0.0],    # left_hip
            [0.1, 0.85, 0.0],     # right_hip
            [0.0, 1.0, 0.0],      # spine1
            [-0.1, 0.45, 0.0],    # left_knee
            [0.1, 0.45, 0.0],     # right_knee
            [0.0, 1.15, 0.0],     # spine2
            [-0.1, 0.05, 0.0],    # left_ankle
            [0.1, 0.05, 0.0],     # right_ankle
            [0.0, 1.3, 0.0],      # spine3
            [-0.1, 0.0, 0.1],     # left_foot
            [0.1, 0.0, 0.1],      # right_foot
            [0.0, 1.45, 0.0],     # neck
            [-0.15, 1.4, 0.0],    # left_collar
            [0.15, 1.4, 0.0],     # right_collar
            [0.0, 1.6, 0.0],      # head
            [-0.25, 1.35, 0.0],   # left_shoulder
            [0.25, 1.35, 0.0],    # right_shoulder
            [-0.55, 1.35, 0.0],   # left_elbow
            [0.55, 1.35, 0.0],    # right_elbow
            [-0.85, 1.35, 0.0],   # left_wrist
            [0.85, 1.35, 0.0],    # right_wrist
            [-0.95, 1.35, 0.0],   # left_hand
            [0.95, 1.35, 0.0],    # right_hand
        ]
    
    @classmethod
    def smpl_24(cls) -> 'SkeletonSpec':
        """Create standard SMPL 24-joint skeleton spec."""
        return cls(
            num_joints=24,
            joint_names=SMPL_JOINT_NAMES.copy(),
            parent_indices=SMPL_PARENTS.copy()
        )


@dataclass
class SkinningSpec:
    """Skinning configuration for body deformation."""
    method: str = "lbs"  # "lbs" or "dqs"
    max_influences: int = 4
    weight_threshold: float = 0.001


@dataclass
class ClothSpec:
    """Cloth simulation configuration."""
    resolution: int = 32
    width: float = 0.6
    height: float = 0.8
    stiffness: float = 100.0
    damping: float = 0.99
    collision_margin: float = 0.005
    coarse_graph_size: int = 256


@dataclass
class PipelineConfig:
    """
    Main pipeline configuration.
    
    Contains all settings for the animation pipeline including
    skeleton spec, skinning parameters, and cloth simulation config.
    """
    name: str = "animation"
    fps: float = 30.0
    coordinate_system: Optional[str] = "Y_UP_RIGHT_HANDED"
    
    skeleton: SkeletonSpec = field(default_factory=SkeletonSpec)
    skinning: SkinningSpec = field(default_factory=SkinningSpec)
    cloth: ClothSpec = field(default_factory=ClothSpec)
    
    checkpoints: Dict[str, Optional[str]] = field(default_factory=dict)
    meshes: Dict[str, Optional[str]] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary."""
        skeleton_data = data.get('skeleton', {})
        skinning_data = data.get('skinning', {})
        cloth_data = data.get('cloth', {})
        
        skeleton = SkeletonSpec(
            num_joints=skeleton_data.get('num_joints', 24),
            joint_names=skeleton_data.get('joint_names', SMPL_JOINT_NAMES.copy()),
            parent_indices=skeleton_data.get('parent_indices', SMPL_PARENTS.copy()),
            rest_positions=skeleton_data.get('rest_positions', [])
        )
        
        skinning = SkinningSpec(
            method=skinning_data.get('method', 'lbs'),
            max_influences=skinning_data.get('max_influences', 4),
            weight_threshold=skinning_data.get('weight_threshold', 0.001)
        )
        
        cloth = ClothSpec(
            resolution=cloth_data.get('resolution', 32),
            width=cloth_data.get('width', 0.6),
            height=cloth_data.get('height', 0.8),
            stiffness=cloth_data.get('stiffness', 100.0),
            damping=cloth_data.get('damping', 0.99),
            collision_margin=cloth_data.get('collision_margin', 0.005),
            coarse_graph_size=cloth_data.get('coarse_graph_size', 256)
        )
        
        return cls(
            name=data.get('name', 'animation'),
            fps=data.get('fps', 30.0),
            coordinate_system=data.get('coordinate_system', 'Y_UP_RIGHT_HANDED'),
            skeleton=skeleton,
            skinning=skinning,
            cloth=cloth,
            checkpoints=data.get('checkpoints', {}),
            meshes=data.get('meshes', {}),
            output=data.get('output', {})
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML is required to save YAML configs.")
        
        data = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'fps': self.fps,
            'coordinate_system': self.coordinate_system,
            'skeleton': {
                'num_joints': self.skeleton.num_joints,
                'joint_names': self.skeleton.joint_names,
                'parent_indices': self.skeleton.parent_indices,
                'rest_positions': self.skeleton.rest_positions,
            },
            'skinning': {
                'method': self.skinning.method,
                'max_influences': self.skinning.max_influences,
                'weight_threshold': self.skinning.weight_threshold,
            },
            'cloth': {
                'resolution': self.cloth.resolution,
                'width': self.cloth.width,
                'height': self.cloth.height,
                'stiffness': self.cloth.stiffness,
                'damping': self.cloth.damping,
                'collision_margin': self.cloth.collision_margin,
                'coarse_graph_size': self.cloth.coarse_graph_size,
            },
            'checkpoints': self.checkpoints,
            'meshes': self.meshes,
            'output': self.output,
        }
