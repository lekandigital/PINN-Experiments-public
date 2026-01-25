"""
NIF-Cloth4D: Package Initialization
"""

from .nif_cloth4d import (
    SineActivation,
    FourierFeatureSIREN,
    NIFCloth4DLoss,
    create_model,
    compute_sdf_loss,
    compute_eikonal_loss,
    compute_stretch_loss,
    compute_bend_loss
)

from .synthetic_data import (
    create_wavy_cloth_sdf,
    create_sphere_sdf,
    create_falling_cloth_sdf,
    save_sdf_to_hdf5,
    load_sdf_from_hdf5,
    generate_dataset,
    sample_points_from_sdf
)

__version__ = "0.1.0"
__author__ = "NIF-Cloth4D Research Team"

__all__ = [
    # Network components
    'SineActivation',
    'FourierFeatureSIREN',
    'NIFCloth4DLoss',
    'create_model',
    
    # Loss functions
    'compute_sdf_loss',
    'compute_eikonal_loss',
    'compute_stretch_loss',
    'compute_bend_loss',
    
    # Data generation
    'create_wavy_cloth_sdf',
    'create_sphere_sdf',
    'create_falling_cloth_sdf',
    'save_sdf_to_hdf5',
    'load_sdf_from_hdf5',
    'generate_dataset',
    'sample_points_from_sdf',
]
