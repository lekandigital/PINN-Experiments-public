"""
Animation Pipeline - Main orchestrator

Chains three PINN-Experiments projects into a single forward pass:
1. Motion Stage (Project 07): Continuous-time skeleton motion
2. Body Stage (Project 14): LBS + soft tissue deformation  
3. Cloth Stage (Project 09): Physics-based cloth simulation

Usage:
    from shared.pipeline import AnimationPipeline, PipelineConfig
    
    config = PipelineConfig.from_yaml("config.yaml")
    pipeline = AnimationPipeline(config)
    pipeline.load(
        motion_checkpoint="path/to/motion.pt",
        body_mesh="path/to/body.obj",
        cloth_mesh="path/to/cloth.obj"
    )
    
    # Single frame
    frame = pipeline.step(t=0.5)
    
    # Full sequence
    frames = pipeline.run_sequence(t_start=0, t_end=2.0, fps=30)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Iterator
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time

from .config import PipelineConfig, SkeletonSpec, SkinningSpec, ClothSpec
from .stages import MotionStage, MotionOutput, BodyStage, BodyOutput, ClothStage, ClothOutput
from .transforms import convert_coordinates, CoordinateSystem


@dataclass
class FrameResult:
    """
    Complete animation frame output.
    
    Contains all data needed to render or export a single frame:
    - Skeleton pose (joint positions and rotations)
    - Deformed body mesh with normals
    - Simulated cloth mesh with normals
    - Timing and diagnostic information
    """
    # Time
    time: float
    frame_index: int
    
    # Skeleton
    joint_positions: torch.Tensor   # (J, 3) global positions
    joint_rotations: torch.Tensor   # (J, 4) local quaternions (wxyz)
    root_position: torch.Tensor     # (3,) root translation
    root_rotation: torch.Tensor     # (4,) root quaternion
    
    # Body mesh
    body_vertices: torch.Tensor     # (V_body, 3) deformed vertices
    body_normals: torch.Tensor      # (V_body, 3) vertex normals
    body_faces: torch.Tensor        # (F_body, 3) face indices
    
    # Cloth mesh
    cloth_vertices: torch.Tensor    # (V_cloth, 3) simulated vertices
    cloth_normals: torch.Tensor     # (V_cloth, 3) vertex normals
    cloth_faces: torch.Tensor       # (F_cloth, 3) face indices
    
    # Diagnostics
    compute_time_ms: float = 0.0
    motion_time_ms: float = 0.0
    body_time_ms: float = 0.0
    cloth_time_ms: float = 0.0
    cloth_strain_energy: Optional[float] = None
    cloth_penetrations: int = 0
    
    def to_numpy(self) -> 'FrameResult':
        """Convert all tensors to numpy arrays."""
        return FrameResult(
            time=self.time,
            frame_index=self.frame_index,
            joint_positions=self.joint_positions.cpu().numpy(),
            joint_rotations=self.joint_rotations.cpu().numpy(),
            root_position=self.root_position.cpu().numpy(),
            root_rotation=self.root_rotation.cpu().numpy(),
            body_vertices=self.body_vertices.cpu().numpy(),
            body_normals=self.body_normals.cpu().numpy(),
            body_faces=self.body_faces.cpu().numpy(),
            cloth_vertices=self.cloth_vertices.cpu().numpy(),
            cloth_normals=self.cloth_normals.cpu().numpy(),
            cloth_faces=self.cloth_faces.cpu().numpy(),
            compute_time_ms=self.compute_time_ms,
            motion_time_ms=self.motion_time_ms,
            body_time_ms=self.body_time_ms,
            cloth_time_ms=self.cloth_time_ms,
            cloth_strain_energy=self.cloth_strain_energy,
            cloth_penetrations=self.cloth_penetrations,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'time': self.time,
            'frame_index': self.frame_index,
            'joint_positions': self.joint_positions.cpu().numpy() if torch.is_tensor(self.joint_positions) else self.joint_positions,
            'joint_rotations': self.joint_rotations.cpu().numpy() if torch.is_tensor(self.joint_rotations) else self.joint_rotations,
            'root_position': self.root_position.cpu().numpy() if torch.is_tensor(self.root_position) else self.root_position,
            'root_rotation': self.root_rotation.cpu().numpy() if torch.is_tensor(self.root_rotation) else self.root_rotation,
            'body_vertices': self.body_vertices.cpu().numpy() if torch.is_tensor(self.body_vertices) else self.body_vertices,
            'body_normals': self.body_normals.cpu().numpy() if torch.is_tensor(self.body_normals) else self.body_normals,
            'body_faces': self.body_faces.cpu().numpy() if torch.is_tensor(self.body_faces) else self.body_faces,
            'cloth_vertices': self.cloth_vertices.cpu().numpy() if torch.is_tensor(self.cloth_vertices) else self.cloth_vertices,
            'cloth_normals': self.cloth_normals.cpu().numpy() if torch.is_tensor(self.cloth_normals) else self.cloth_normals,
            'cloth_faces': self.cloth_faces.cpu().numpy() if torch.is_tensor(self.cloth_faces) else self.cloth_faces,
            'compute_time_ms': self.compute_time_ms,
            'cloth_strain_energy': self.cloth_strain_energy,
            'cloth_penetrations': self.cloth_penetrations,
        }


class AnimationPipeline:
    """
    Main animation pipeline orchestrating motion → body → cloth.
    
    Chains three neural network models into a single forward pass:
    1. Motion: Query skeleton pose at continuous time t
    2. Body: Apply LBS + soft tissue deformation
    3. Cloth: Simulate cloth with body collision
    
    Features:
    - Continuous time queries (not locked to frame rate)
    - Automatic coordinate system conversion between stages
    - GPU acceleration with optional CPU fallback
    - Streaming sequence generation
    - Frame result caching
    """
    
    def __init__(self, config: PipelineConfig, device: torch.device = None):
        """
        Initialize the animation pipeline.
        
        Args:
            config: Pipeline configuration
            device: Compute device (defaults to CUDA if available)
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize stages
        self.motion_stage = MotionStage(config, self.device)
        self.body_stage = BodyStage(config, self.device)
        self.cloth_stage = ClothStage(config, self.device)
        
        # State
        self.is_loaded = False
        self.frame_index = 0
        self.current_time = 0.0
        
        # Coordinate system config
        self.motion_coords = CoordinateSystem.Y_UP_RIGHT_HANDED  # BVH convention
        self.body_coords = CoordinateSystem.Y_UP_RIGHT_HANDED    # Standard
        self.output_coords = config.coordinate_system or CoordinateSystem.Y_UP_RIGHT_HANDED
        
        # Performance tracking
        self.last_frame_times = {
            'motion': 0.0,
            'body': 0.0,
            'cloth': 0.0,
            'total': 0.0
        }
    
    def load(self,
             motion_checkpoint: Optional[str] = None,
             body_checkpoint: Optional[str] = None,
             cloth_checkpoint: Optional[str] = None,
             body_mesh: Optional[str] = None,
             skinning_weights: Optional[str] = None,
             cloth_mesh: Optional[str] = None,
             pinned_vertices: Optional[List[int]] = None,
             source_joint_names: Optional[List[str]] = None):
        """
        Load all models and mesh data.
        
        Args:
            motion_checkpoint: Path to Project 07 model checkpoint
            body_checkpoint: Path to Project 14 PEGNN checkpoint
            cloth_checkpoint: Path to Project 09 HGNN-NIF checkpoint
            body_mesh: Path to rest pose body mesh (OBJ)
            skinning_weights: Path to skinning weights (NPZ)
            cloth_mesh: Path to rest pose cloth mesh (OBJ)
            pinned_vertices: Cloth vertex indices to pin to body
            source_joint_names: Joint names in motion model output (for remapping)
        """
        print(f"[AnimationPipeline] Loading on device: {self.device}")
        
        # Load motion stage
        print("[AnimationPipeline] Loading motion stage...")
        self.motion_stage.load(
            checkpoint_path=motion_checkpoint,
            source_joint_names=source_joint_names
        )
        
        # Load body stage
        print("[AnimationPipeline] Loading body stage...")
        self.body_stage.load(
            checkpoint_path=body_checkpoint,
            body_mesh_path=body_mesh,
            skinning_weights_path=skinning_weights
        )
        
        # Load cloth stage
        print("[AnimationPipeline] Loading cloth stage...")
        self.cloth_stage.load(
            checkpoint_path=cloth_checkpoint,
            cloth_mesh_path=cloth_mesh,
            pinned_vertex_ids=pinned_vertices
        )
        
        self.is_loaded = True
        print("[AnimationPipeline] All stages loaded successfully")
    
    def step(self, t: float, profile: bool = False) -> FrameResult:
        """
        Compute a single animation frame at time t.
        
        This is the core pipeline forward pass:
        1. Query motion model for skeleton pose
        2. Apply LBS + soft tissue to body mesh
        3. Simulate cloth with body collision
        
        Args:
            t: Time in seconds
            profile: Whether to record per-stage timing
            
        Returns:
            FrameResult with complete frame data
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        
        total_start = time.perf_counter()
        
        # Stage 1: Motion
        if profile:
            motion_start = time.perf_counter()
        
        motion_output = self.motion_stage.step(t)
        
        if profile:
            motion_time = (time.perf_counter() - motion_start) * 1000
        else:
            motion_time = 0.0
        
        # Convert motion coordinates if needed
        if self.motion_coords != self.body_coords:
            motion_output = self._convert_motion_coordinates(
                motion_output, self.motion_coords, self.body_coords
            )
        
        # Stage 2: Body deformation
        if profile:
            body_start = time.perf_counter()
        
        body_output = self.body_stage.step(motion_output, t)
        
        if profile:
            body_time = (time.perf_counter() - body_start) * 1000
        else:
            body_time = 0.0
        
        # Stage 3: Cloth simulation
        if profile:
            cloth_start = time.perf_counter()
        
        cloth_output = self.cloth_stage.step(body_output, motion_output, t)
        
        if profile:
            cloth_time = (time.perf_counter() - cloth_start) * 1000
        else:
            cloth_time = 0.0
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        # Update timing
        self.last_frame_times = {
            'motion': motion_time,
            'body': body_time,
            'cloth': cloth_time,
            'total': total_time
        }
        
        # Convert output coordinates if needed
        joint_positions = motion_output.joint_positions
        body_vertices = body_output.deformed_vertices
        cloth_vertices = cloth_output.vertices
        
        if self.body_coords != self.output_coords:
            joint_positions = convert_coordinates(
                joint_positions.cpu().numpy(),
                self.body_coords, self.output_coords
            )
            joint_positions = torch.tensor(joint_positions, device=self.device)
            
            body_vertices = convert_coordinates(
                body_vertices.cpu().numpy(),
                self.body_coords, self.output_coords
            )
            body_vertices = torch.tensor(body_vertices, device=self.device)
            
            cloth_vertices = convert_coordinates(
                cloth_vertices.cpu().numpy(),
                self.body_coords, self.output_coords
            )
            cloth_vertices = torch.tensor(cloth_vertices, device=self.device)
        
        # Compute penetration count
        penetration_count = 0
        if cloth_output.penetration_mask is not None:
            penetration_count = cloth_output.penetration_mask.sum().item()
        
        # Build result
        result = FrameResult(
            time=t,
            frame_index=self.frame_index,
            joint_positions=joint_positions,
            joint_rotations=motion_output.joint_rotations,
            root_position=motion_output.root_position,
            root_rotation=motion_output.root_rotation,
            body_vertices=body_vertices,
            body_normals=body_output.normals,
            body_faces=body_output.faces,
            cloth_vertices=cloth_vertices,
            cloth_normals=cloth_output.normals,
            cloth_faces=cloth_output.faces,
            compute_time_ms=total_time,
            motion_time_ms=motion_time,
            body_time_ms=body_time,
            cloth_time_ms=cloth_time,
            cloth_strain_energy=cloth_output.strain_energy,
            cloth_penetrations=penetration_count,
        )
        
        self.frame_index += 1
        self.current_time = t
        
        return result
    
    def _convert_motion_coordinates(self, motion_output: MotionOutput,
                                     source: CoordinateSystem,
                                     target: CoordinateSystem) -> MotionOutput:
        """Convert motion output between coordinate systems."""
        joint_positions = convert_coordinates(
            motion_output.joint_positions.cpu().numpy(),
            source, target
        )
        root_position = convert_coordinates(
            motion_output.root_position.cpu().numpy().reshape(1, 3),
            source, target
        ).squeeze()
        
        # Note: quaternion conversion is more complex, keeping as-is for now
        return MotionOutput(
            joint_positions=torch.tensor(joint_positions, device=self.device),
            joint_rotations=motion_output.joint_rotations,
            root_position=torch.tensor(root_position, device=self.device),
            root_rotation=motion_output.root_rotation,
            velocity=motion_output.velocity,
            acceleration=motion_output.acceleration,
        )
    
    def run_sequence(self, t_start: float = 0.0, t_end: float = 1.0,
                     fps: float = 30.0, profile: bool = False
                     ) -> List[FrameResult]:
        """
        Run the full animation sequence.
        
        Args:
            t_start: Start time in seconds
            t_end: End time in seconds
            fps: Frames per second
            profile: Whether to record timing
            
        Returns:
            List of FrameResult for each frame
        """
        self.reset()
        
        num_frames = int((t_end - t_start) * fps) + 1
        times = np.linspace(t_start, t_end, num_frames)
        
        results = []
        for t in times:
            result = self.step(float(t), profile=profile)
            results.append(result)
        
        return results
    
    def stream_sequence(self, t_start: float = 0.0, t_end: float = 1.0,
                        fps: float = 30.0, profile: bool = False
                        ) -> Iterator[FrameResult]:
        """
        Stream animation frames as a generator.
        
        More memory efficient than run_sequence for long animations.
        
        Args:
            t_start: Start time in seconds
            t_end: End time in seconds
            fps: Frames per second
            profile: Whether to record timing
            
        Yields:
            FrameResult for each frame
        """
        self.reset()
        
        num_frames = int((t_end - t_start) * fps) + 1
        times = np.linspace(t_start, t_end, num_frames)
        
        for t in times:
            yield self.step(float(t), profile=profile)
    
    def reset(self):
        """Reset all stages to initial state."""
        self.motion_stage.reset()
        self.body_stage.reset()
        self.cloth_stage.reset()
        self.frame_index = 0
        self.current_time = 0.0
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics from last frame."""
        return self.last_frame_times.copy()
    
    def get_rest_meshes(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get rest pose meshes for body and cloth.
        
        Returns:
            Dict with 'body' and 'cloth' keys, each containing (vertices, faces)
        """
        return {
            'body': self.body_stage.get_rest_mesh(),
            'cloth': self.cloth_stage.get_rest_mesh(),
        }
    
    def set_cloth_pinned_vertices(self, vertex_ids: List[int]):
        """Update which cloth vertices are pinned to the body."""
        self.cloth_stage.set_pinned_vertices(vertex_ids)
    
    def to(self, device: torch.device):
        """Move pipeline to a different device."""
        self.device = device
        # Would need to move all stage models and data
        # For now, just update device reference
        return self
    
    @property
    def num_joints(self) -> int:
        """Number of skeleton joints."""
        return self.config.skeleton.num_joints
    
    @property
    def joint_names(self) -> List[str]:
        """Skeleton joint names."""
        return self.config.skeleton.joint_names
    
    def __repr__(self) -> str:
        return (f"AnimationPipeline("
                f"joints={self.num_joints}, "
                f"device={self.device}, "
                f"loaded={self.is_loaded})")


def create_pipeline(config_path: Optional[str] = None,
                    device: Optional[torch.device] = None,
                    **kwargs) -> AnimationPipeline:
    """
    Factory function to create a configured AnimationPipeline.
    
    Args:
        config_path: Path to YAML config file
        device: Compute device
        **kwargs: Override config values
        
    Returns:
        Configured AnimationPipeline instance
    """
    if config_path:
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return AnimationPipeline(config, device)
