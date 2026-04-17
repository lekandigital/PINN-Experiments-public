"""
Transform module for the cloth simulation pipeline.

Transforms raw mesh sequences into ML-ready features:
- Graph construction (edges, features, coarsening)
- SDF computation (thickened mesh, point sampling)
- Temporal features (velocities, accelerations, windows)
- Vertex features (curvature, normals)
- Normalization
"""

from .graph import (
    build_graph,
    compute_edge_features,
    coarsen_graph,
    GraphData,
)
from .sdf import (
    compute_sdf_volume,
    compute_sdf_samples,
    SDFData,
)
from .temporal import (
    compute_temporal_features,
    create_temporal_windows,
    TemporalData,
)
from .features import (
    compute_vertex_features,
    compute_curvature,
    compute_normals,
)
from .normalization import (
    compute_normalization_stats,
    normalize_sequence,
    NormalizationStats,
)

__all__ = [
    # Graph
    'build_graph',
    'compute_edge_features',
    'coarsen_graph',
    'GraphData',
    # SDF
    'compute_sdf_volume',
    'compute_sdf_samples',
    'SDFData',
    # Temporal
    'compute_temporal_features',
    'create_temporal_windows',
    'TemporalData',
    # Features
    'compute_vertex_features',
    'compute_curvature',
    'compute_normals',
    # Normalization
    'compute_normalization_stats',
    'normalize_sequence',
    'NormalizationStats',
]
