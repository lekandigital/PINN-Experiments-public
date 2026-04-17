"""
Knowledge Distillation & Edge Deployment Pipeline

A model-agnostic pipeline for:
1. Distilling large teacher models into compact students
2. Exporting to ONNX format
3. Quantizing to INT8/FP16
4. Benchmarking across hardware targets
5. Packaging for web/mobile/edge deployment

Usage:
    from tools.distillation import DistillationPipeline, registry
    
    # Register your models
    registry.register("my-teacher", TeacherModel, default_config={...})
    registry.register("my-student", StudentModel, default_config={...})
    
    # Run the pipeline
    pipeline = DistillationPipeline("config.yaml")
    results = pipeline.run()

CLI:
    python -m tools.distillation.pipeline --config config.yaml
    python -m tools.distillation.pipeline --config config.yaml --stages export quantize benchmark
"""

from .registry import registry, ModelRegistry
from .config import DistillConfig, load_config
from .pipeline import DistillationPipeline

__all__ = [
    "registry",
    "ModelRegistry", 
    "DistillConfig",
    "load_config",
    "DistillationPipeline",
]

__version__ = "0.1.0"
