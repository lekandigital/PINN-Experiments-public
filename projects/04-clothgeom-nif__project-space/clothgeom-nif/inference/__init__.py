"""
ClothGeom-NIF Inference Package

Mesh extraction and inference utilities.
"""

from .mesh_extractor import (
    MeshExtractionConfig,
    MeshExtractor,
    load_model_for_extraction,
    batch_extract_meshes
)

__all__ = [
    'MeshExtractionConfig',
    'MeshExtractor',
    'load_model_for_extraction',
    'batch_extract_meshes'
]
