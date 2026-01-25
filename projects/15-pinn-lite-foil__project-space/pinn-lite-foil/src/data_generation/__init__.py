"""
Data Generation Package
NACA airfoil generation, OpenFOAM CFD simulation, and HDF5 export.
"""

from .naca_generator import generate_naca4, generate_parametric_sweep
from .hdf5_exporter import load_from_hdf5, create_synthetic_dataset

__all__ = [
    'generate_naca4',
    'generate_parametric_sweep', 
    'load_from_hdf5',
    'create_synthetic_dataset'
]
