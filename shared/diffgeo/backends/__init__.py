"""
Backends subpackage - Framework-specific tensor implementations.

Provides thin wrappers for applying DEC operators to PyTorch/JAX tensors:
- NumPy backend (default, for operator construction)
- PyTorch backend (for Projects 01, 06, 09)
- JAX backend (for Project 16)
"""

from .base import DiffGeoBackend, get_backend

# Lazy imports to avoid requiring all backends
_backends = {}

def get_numpy_backend():
    """Get the NumPy backend (always available)."""
    if 'numpy' not in _backends:
        from .numpy_backend import NumPyBackend
        _backends['numpy'] = NumPyBackend()
    return _backends['numpy']

def get_torch_backend(device='cpu'):
    """Get PyTorch backend. Requires torch."""
    key = f'torch_{device}'
    if key not in _backends:
        from .torch_backend import TorchBackend
        _backends[key] = TorchBackend(device=device)
    return _backends[key]

def get_jax_backend():
    """Get JAX backend. Requires jax."""
    if 'jax' not in _backends:
        from .jax_backend import JAXBackend
        _backends['jax'] = JAXBackend()
    return _backends['jax']

__all__ = [
    "DiffGeoBackend",
    "get_backend",
    "get_numpy_backend",
    "get_torch_backend",
    "get_jax_backend",
]
