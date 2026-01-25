"""
ClothGeom-NIF Data Package

Synthetic cloth data generation and dataset loading utilities.
"""

from .synthetic_cloth_generator import (
    ClothConfig,
    DeformationConfig,
    SyntheticClothGenerator,
    generate_dataset
)

from .cloth_dataset import (
    ClothSDFDataset,
    MultiResolutionClothDataset,
    create_dataloader,
    collate_flatten
)

__all__ = [
    # Generator
    'ClothConfig',
    'DeformationConfig', 
    'SyntheticClothGenerator',
    'generate_dataset',
    # Dataset
    'ClothSDFDataset',
    'MultiResolutionClothDataset',
    'create_dataloader',
    'collate_flatten'
]
