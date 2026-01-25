"""
Maxwell-PINN-NIF: Physics-Informed Neural Network for Maxwell's Equations

A comprehensive research kit for neural implicit field solvers that tackle
the full-vector Maxwell equations in complex, anisotropic media.

Modules:
    pinn_model: Core neural network architecture and physics losses
    pml_loss: Perfectly Matched Layer absorbing boundary conditions
    dataset_generator: Synthetic training data generation
    train: Training loop with mixed-precision and logging
"""

from .pinn_model import (
    MaxwellPINN,
    FourierFeatureEncoding,
    divergence_free_loss,
    maxwell_curl_residual,
    total_physics_loss,
)

from .pml_loss import (
    PMLLoss,
    PMLRegion,
    HardBoundaryLoss,
    CombinedBoundaryLoss,
)

from .dataset_generator import (
    generate_uniform_material,
    generate_layered_material,
    generate_random_inclusions,
    generate_sinusoidal_material,
    generate_training_dataset,
    save_dataset_hdf5,
    load_dataset_hdf5,
)

__version__ = "1.0.0"
__author__ = "Maxwell-PINN-NIF Team"

__all__ = [
    # Model
    "MaxwellPINN",
    "FourierFeatureEncoding",
    "divergence_free_loss",
    "maxwell_curl_residual",
    "total_physics_loss",
    # Boundary conditions
    "PMLLoss",
    "PMLRegion", 
    "HardBoundaryLoss",
    "CombinedBoundaryLoss",
    # Dataset
    "generate_uniform_material",
    "generate_layered_material",
    "generate_random_inclusions",
    "generate_sinusoidal_material",
    "generate_training_dataset",
    "save_dataset_hdf5",
    "load_dataset_hdf5",
]
