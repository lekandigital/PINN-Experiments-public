"""
Animation Pipeline - Shared module for chaining PINN-Experiments projects.

This module provides a unified pipeline for character animation that chains:
- Project 07 (Geom-INR-Motion): Continuous-time skeleton motion
- Project 14 (PEGNN-Deform): LBS + soft tissue deformation
- Project 09 (HGNN-NIF-Cloth): Physics-based cloth simulation

Quick Start:
    from shared.pipeline import AnimationPipeline, PipelineConfig
    
    # Create with default config
    pipeline = AnimationPipeline(PipelineConfig())
    pipeline.load()
    
    # Generate a frame
    frame = pipeline.step(t=0.5)
    
    # Access results
    body_mesh = frame.body_vertices, frame.body_faces
    cloth_mesh = frame.cloth_vertices, frame.cloth_faces
    skeleton = frame.joint_positions, frame.joint_rotations

Full Example:
    from shared.pipeline import (
        AnimationPipeline, 
        PipelineConfig,
        create_pipeline
    )
    from shared.pipeline.exporters import export_npz, export_obj
    
    # Load from config file
    config = PipelineConfig.from_yaml("config.yaml")
    pipeline = AnimationPipeline(config)
    
    # Load with checkpoints
    pipeline.load(
        motion_checkpoint="models/motion.pt",
        body_mesh="assets/body.obj",
        cloth_mesh="assets/cloth.obj"
    )
    
    # Run sequence
    frames = pipeline.run_sequence(t_start=0, t_end=2.0, fps=30)
    
    # Export
    export_npz([f.to_dict() for f in frames], "animation.npz")
"""

from .animation_pipeline import AnimationPipeline, FrameResult, create_pipeline
from .config import (
    PipelineConfig,
    SkeletonSpec,
    SkinningSpec,
    ClothSpec,
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
)
from .transforms import (
    CoordinateSystem,
    convert_coordinates,
    forward_kinematics,
    linear_blend_skinning,
    linear_blend_skinning_torch,
    dual_quaternion_skinning,
)
from .stages import (
    MotionStage,
    MotionOutput,
    BodyStage,
    BodyOutput,
    ClothStage,
    ClothOutput,
)

__version__ = "0.1.0"

__all__ = [
    # Main pipeline
    "AnimationPipeline",
    "FrameResult",
    "create_pipeline",
    
    # Configuration
    "PipelineConfig",
    "SkeletonSpec",
    "SkinningSpec", 
    "ClothSpec",
    "SMPL_JOINT_NAMES",
    "SMPL_PARENTS",
    
    # Transforms
    "CoordinateSystem",
    "convert_coordinates",
    "forward_kinematics",
    "linear_blend_skinning",
    "linear_blend_skinning_torch",
    "dual_quaternion_skinning",
    
    # Stages
    "MotionStage",
    "MotionOutput",
    "BodyStage",
    "BodyOutput",
    "ClothStage",
    "ClothOutput",
]
