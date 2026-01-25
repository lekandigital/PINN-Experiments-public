"""
Reproducibility utilities for NIF-Cloth4D-Temporal.

Provides seed locking and device selection for deterministic experiments.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: If True, enables PyTorch deterministic mode
                      (may reduce performance)
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # MPS (Apple Silicon, if available)
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have separate seed function yet
        pass
    
    # Deterministic operations
    if deterministic:
        # These settings can reduce performance but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for deterministic algorithms
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # PyTorch 1.8+ deterministic flag
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                torch.use_deterministic_algorithms(False)
    else:
        # Enable cuDNN autotuning for faster training
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(
    device_name: Optional[str] = None,
    gpu_id: int = 0,
) -> torch.device:
    """
    Get the best available device for computation.
    
    Priority: CUDA > MPS > CPU (unless specified)
    
    Args:
        device_name: Force specific device ('cuda', 'mps', 'cpu')
        gpu_id: GPU index to use (if multiple GPUs available)
        
    Returns:
        torch.device object
    """
    if device_name is not None:
        if device_name == 'cuda':
            if torch.cuda.is_available():
                return torch.device(f'cuda:{gpu_id}')
            else:
                print("Warning: CUDA requested but not available. Using CPU.")
                return torch.device('cpu')
        elif device_name == 'mps':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                print("Warning: MPS requested but not available. Using CPU.")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    # Auto-detect best device
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cpu': True,
        'cuda': torch.cuda.is_available(),
        'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cuda_devices': [],
    }
    
    if info['cuda']:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['cuda_devices'].append({
                'index': i,
                'name': props.name,
                'memory_gb': props.total_memory / 1024**3,
                'compute_capability': f"{props.major}.{props.minor}",
            })
    
    return info


def print_device_info() -> None:
    """Print device information to console."""
    info = get_device_info()
    
    print("Available Devices:")
    print(f"  CPU: Yes")
    print(f"  CUDA: {'Yes' if info['cuda'] else 'No'}")
    print(f"  MPS: {'Yes' if info['mps'] else 'No'}")
    
    if info['cuda_devices']:
        print("\nCUDA Devices:")
        for gpu in info['cuda_devices']:
            print(f"  [{gpu['index']}] {gpu['name']}")
            print(f"      Memory: {gpu['memory_gb']:.1f} GB")
            print(f"      Compute: {gpu['compute_capability']}")
