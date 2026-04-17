"""
Operators subpackage - Discrete Exterior Calculus operators.

Provides:
- Discrete gradient (d0): vertex scalars → edge 1-forms
- Discrete divergence: edge 1-forms → vertex scalars
- Discrete curl (d1): edge 1-forms → face 2-forms
- Laplace-Beltrami operator (cotangent weighted)
- Hodge star operators
"""

from .gradient import discrete_gradient, build_gradient_matrix
from .divergence import discrete_divergence, build_divergence_matrix
from .curl import discrete_curl, build_curl_matrix
from .laplace_beltrami import (
    laplace_beltrami,
    laplace_beltrami_weak,
    build_cotangent_laplacian,
    compute_cotangent_weights,
)
from .hodge import hodge_star_0, hodge_star_1, hodge_star_2
from .exterior_derivative import build_d0, build_d1

__all__ = [
    "discrete_gradient",
    "build_gradient_matrix",
    "discrete_divergence",
    "build_divergence_matrix",
    "discrete_curl",
    "build_curl_matrix",
    "laplace_beltrami",
    "laplace_beltrami_weak",
    "build_cotangent_laplacian",
    "compute_cotangent_weights",
    "hodge_star_0",
    "hodge_star_1",
    "hodge_star_2",
    "build_d0",
    "build_d1",
]
