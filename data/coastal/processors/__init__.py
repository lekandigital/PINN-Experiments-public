"""
Processor package for coastal data infrastructure.

Transforms raw downloaded data into formats usable by ML models.
"""

from .mesh_generator import CoastalMeshGenerator, MeshParameters, generate_mesh
from .forcing_builder import ForcingBuilder, ForcingParameters
from .validation_builder import ValidationBuilder, ValidationParameters, build_validation_dataset
from .coordinate_utils import (
    latlon_to_utm,
    utm_to_latlon,
    get_utm_zone,
    get_utm_epsg,
    great_circle_distance,
    great_circle_distance_array,
    compute_distance_to_shore,
    point_in_polygon,
    bearing,
    destination_point,
)

__all__ = [
    # Mesh generation
    "CoastalMeshGenerator",
    "MeshParameters",
    "generate_mesh",
    # Forcing building
    "ForcingBuilder",
    "ForcingParameters",
    # Validation building
    "ValidationBuilder",
    "ValidationParameters",
    "build_validation_dataset",
    # Coordinate utilities
    "latlon_to_utm",
    "utm_to_latlon",
    "get_utm_zone",
    "get_utm_epsg",
    "great_circle_distance",
    "great_circle_distance_array",
    "compute_distance_to_shore",
    "point_in_polygon",
    "bearing",
    "destination_point",
]
