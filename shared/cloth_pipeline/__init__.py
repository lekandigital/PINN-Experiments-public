"""
Cloth Pipeline - Professional cloth simulation data pipeline for ML training.

A unified data pipeline that ingests simulation caches from Houdini Vellum,
Marvelous Designer, or Blender, converts them to training formats for
physics-informed cloth simulation models.

Serves Projects: 04, 05, 08, 09, 11, 12, 13

Usage:
    python -m cloth_pipeline run --config cloth_pipeline_config.yaml
    python -m cloth_pipeline ingest --source /path/to/sim.abc --info
    python -m cloth_pipeline export --config config.yaml --target project_09
    python -m cloth_pipeline validate --data /path/to/training_data/
"""

__version__ = "0.1.0"

from .types import ClothSequence, CollisionBody
from .config import PipelineConfig, load_config

__all__ = [
    "ClothSequence",
    "CollisionBody", 
    "PipelineConfig",
    "load_config",
    "__version__",
]
