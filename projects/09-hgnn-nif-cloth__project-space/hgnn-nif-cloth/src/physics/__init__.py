"""
Physics Subpackage for HGNN-NIF-Cloth

Intrinsic differential geometry-based physics computations using the
shared diffgeo module. Provides accurate curvature-based bending energy
to replace the placeholder implementation.
"""

from .intrinsic_energy import (
    IntrinsicBendingEnergy,
    intrinsic_bending_loss,
    MembraneEnergy,
    membrane_loss,
)

__all__ = [
    "IntrinsicBendingEnergy",
    "intrinsic_bending_loss",
    "MembraneEnergy",
    "membrane_loss",
]
