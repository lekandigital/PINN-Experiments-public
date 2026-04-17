"""
Data ingestion module for the cloth simulation pipeline.

Supports reading from:
- Alembic (.abc) files
- OBJ sequences
- Blender cloth simulations
"""

from .obj_sequence_reader import read_obj_sequence, OBJSequenceReader
from .abc_reader import read_alembic, AlembicReader
from .utils import validate_topology, interpolate_frames

from pathlib import Path
from typing import Union, Optional
from ..types import ClothSequence


def ingest_source(
    source_path: str,
    format_type: str,
    fps: float = 24.0,
    rest_frame: int = 0,
    **kwargs
) -> ClothSequence:
    """
    Ingest cloth simulation data from various sources.
    
    Args:
        source_path: Path to the simulation data
        format_type: One of 'abc', 'obj_sequence', 'blender'
        fps: Source framerate
        rest_frame: Which frame represents the rest pose
        **kwargs: Format-specific options
        
    Returns:
        ClothSequence ready for transformation
        
    Raises:
        ValueError: If format_type is not recognized
        FileNotFoundError: If source doesn't exist
    """
    source_path = Path(source_path)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")
    
    if format_type == 'abc':
        return read_alembic(source_path, fps=fps, **kwargs)
    elif format_type == 'obj_sequence':
        return read_obj_sequence(source_path, fps=fps, rest_frame=rest_frame, **kwargs)
    elif format_type == 'blender':
        raise NotImplementedError(
            "Blender ingestion must be run from within Blender. "
            "Use cloth_pipeline.ingest.blender_recorder.record_simulation() instead."
        )
    else:
        raise ValueError(f"Unknown format: {format_type}. Supported: abc, obj_sequence, blender")


__all__ = [
    'ingest_source',
    'read_obj_sequence',
    'read_alembic',
    'OBJSequenceReader',
    'AlembicReader',
    'validate_topology',
    'interpolate_frames',
]
