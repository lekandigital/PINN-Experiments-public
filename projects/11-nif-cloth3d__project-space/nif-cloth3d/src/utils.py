"""
Utility functions for NIF-Cloth3D-Interactive.
"""

import os
import yaml
import json
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def normalize_vertices(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize vertices to unit cube centered at origin.
    
    Returns:
        normalized vertices, center, scale
    """
    center = vertices.mean(axis=0)
    vertices_centered = vertices - center
    scale = np.abs(vertices_centered).max()
    vertices_normalized = vertices_centered / scale
    return vertices_normalized, center, scale


def denormalize_vertices(vertices: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Denormalize vertices from unit cube."""
    return vertices * scale + center


def compute_mesh_edges(faces: np.ndarray) -> np.ndarray:
    """
    Compute unique edges from triangle faces.
    
    Args:
        faces: (F, 3) array of triangle indices
        
    Returns:
        (E, 2) array of edge indices
    """
    edges = set()
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edges.add(edge)
    return np.array(list(edges))


def compute_vertex_neighbors(vertices: np.ndarray, edges: np.ndarray) -> Dict[int, List[int]]:
    """
    Compute neighbor list for each vertex.
    
    Args:
        vertices: (N, 3) vertex positions
        edges: (E, 2) edge indices
        
    Returns:
        Dictionary mapping vertex index to list of neighbor indices
    """
    neighbors = {i: [] for i in range(len(vertices))}
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    return neighbors


def compute_rest_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Compute rest lengths for all edges.
    
    Args:
        vertices: (N, 3) vertex positions
        edges: (E, 2) edge indices
        
    Returns:
        (E,) array of rest lengths
    """
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    return np.linalg.norm(v1 - v0, axis=1)


def grid_mesh(resolution: int, size: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a regular grid mesh.
    
    Args:
        resolution: Number of vertices per side
        size: Total size of the mesh
        
    Returns:
        vertices: (N, 3) vertex positions
        faces: (F, 3) triangle indices
    """
    x = np.linspace(-size/2, size/2, resolution)
    y = np.linspace(-size/2, size/2, resolution)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    
    vertices = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).astype(np.float32)
    
    # Generate triangle faces
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            # Two triangles per quad
            faces.append([idx, idx + resolution, idx + 1])
            faces.append([idx + 1, idx + resolution, idx + resolution + 1])
    faces = np.array(faces)
    
    return vertices, faces


def apply_pinned_constraints(
    vertices: np.ndarray,
    rest_vertices: np.ndarray,
    pinned_indices: np.ndarray
) -> np.ndarray:
    """
    Apply pinned vertex constraints.
    
    Args:
        vertices: Current vertex positions
        rest_vertices: Rest vertex positions
        pinned_indices: Indices of pinned vertices
        
    Returns:
        Vertices with pinned constraints applied
    """
    result = vertices.copy()
    result[pinned_indices] = rest_vertices[pinned_indices]
    return result


class MovingAverage:
    """Compute running moving average."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = []
    
    def update(self, value: float) -> float:
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return np.mean(self.values)
    
    @property
    def average(self) -> float:
        return np.mean(self.values) if self.values else 0.0


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def compute_memory_usage(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """
    Estimate memory usage for model inference.
    
    Returns:
        Dictionary with memory estimates in MB
    """
    # Parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Activation memory (rough estimate)
    batch_size, input_dim = input_shape
    
    # For SIREN: input -> hidden -> ... -> output
    # Estimate based on largest hidden layer
    hidden_dims = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            hidden_dims.append(module.weight.shape[0])
    
    max_hidden = max(hidden_dims) if hidden_dims else 256
    activation_memory = batch_size * max_hidden * 4 * 8  # float32, rough estimate
    
    return {
        'parameters_mb': param_memory / (1024 ** 2),
        'activations_mb': activation_memory / (1024 ** 2),
        'total_mb': (param_memory + activation_memory) / (1024 ** 2)
    }


def export_to_onnx(
    model: torch.nn.Module,
    save_path: str,
    input_dim: int = 8,
    batch_size: int = 1000,
    opset_version: int = 17
) -> None:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        save_path: Path to save ONNX file
        input_dim: Input dimension
        batch_size: Batch size for export
        opset_version: ONNX opset version
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(batch_size, input_dim, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=['input'],
        output_names=['displacement'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'displacement': {0: 'batch_size'}
        },
        opset_version=opset_version
    )
    print(f"Model exported to {save_path}")


def export_to_torchscript(
    model: torch.nn.Module,
    save_path: str,
    input_dim: int = 8,
    batch_size: int = 1000,
    use_trace: bool = True
) -> None:
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        save_path: Path to save .pt file
        input_dim: Input dimension
        batch_size: Batch size for tracing
        use_trace: Use tracing vs scripting
    """
    model.eval()
    device = next(model.parameters()).device
    
    if use_trace:
        dummy_input = torch.randn(batch_size, input_dim, device=device)
        traced = torch.jit.trace(model, dummy_input)
    else:
        traced = torch.jit.script(model)
    
    traced.save(save_path)
    print(f"Model exported to {save_path}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test grid mesh generation
    vertices, faces = grid_mesh(10, 2.0)
    print(f"Grid mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Test edge computation
    edges = compute_mesh_edges(faces)
    print(f"Computed {len(edges)} edges")
    
    # Test normalization
    normalized, center, scale = normalize_vertices(vertices)
    print(f"Normalization: center={center}, scale={scale}")
    
    # Test moving average
    ma = MovingAverage(10)
    for i in range(20):
        avg = ma.update(i)
    print(f"Moving average: {avg:.2f}")
    
    print("All utility tests passed!")
