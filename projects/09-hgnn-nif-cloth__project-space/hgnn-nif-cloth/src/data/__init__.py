# Data utilities for HGNN-NIF-Cloth
"""
Data generation and loading utilities.

Components:
- generate_synthetic_cloth_data: Create test data without Blender
- H5ClothDataset: PyTorch Dataset for HDF5 cloth data
"""

from .synthetic_data import generate_synthetic_cloth_data
from .dataset import H5ClothDataset

__all__ = [
    "generate_synthetic_cloth_data",
    "H5ClothDataset",
]
