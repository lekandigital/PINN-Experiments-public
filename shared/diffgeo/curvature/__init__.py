"""
Curvature subpackage - Discrete curvature computations.

Provides:
- Gaussian curvature (via angle defect)
- Mean curvature (via Laplace-Beltrami of position)
- Principal curvatures and directions (via shape operator)
- First and second fundamental forms
- Vertex and face normal computations
"""

from .gaussian import gaussian_curvature, total_gaussian_curvature
from .mean import mean_curvature, mean_curvature_vector
from .principal import principal_curvatures, shape_operator_eigenvalues
from .fundamental_forms import first_fundamental_form, second_fundamental_form
from .normals import vertex_normals, face_normals, area_weighted_normals

__all__ = [
    "gaussian_curvature",
    "total_gaussian_curvature",
    "mean_curvature",
    "mean_curvature_vector",
    "principal_curvatures",
    "shape_operator_eigenvalues",
    "first_fundamental_form",
    "second_fundamental_form",
    "vertex_normals",
    "face_normals",
    "area_weighted_normals",
]
