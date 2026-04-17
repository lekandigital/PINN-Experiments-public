"""
Project 06: CoastFlow-GNN Integration Example

This example demonstrates how to use the coastal data infrastructure
with the CoastFlow-GNN project for graph-based coastal flow prediction.

Project 06 uses PyTorch Geometric for GNN-based surrogate modeling
of coastal hydrodynamics including:
- Storm surge prediction
- Tidal circulation
- Wave-driven currents

The coastal data package provides:
- Real bathymetry from NOAA CUDEM
- Tide gauge observations for validation
- Wave buoy data for boundary conditions
- Unstructured mesh generation compatible with PyG
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Optional imports for PyTorch/PyG
try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch/PyG not available - some features disabled")

# Import coastal data infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.coastal import (
    CoastalDataManager,
    RegionConfig,
    BoundingBox,
    REGIONS,
    CoastalMesh,
    MeshParameters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_coastflow_dataset(
    region: str = "chesapeake_bay",
    start_date: str = "2023-01-01",
    end_date: str = "2023-01-07",
    cache_dir: Optional[str] = None,
) -> dict:
    """
    Create a complete dataset for CoastFlow-GNN training.
    
    This function downloads real coastal data and packages it
    in a format suitable for GNN training.
    
    Args:
        region: Region name (see REGIONS dict for options)
        start_date: Start date for observations (YYYY-MM-DD)
        end_date: End date for observations (YYYY-MM-DD)
        cache_dir: Directory for caching downloaded data
        
    Returns:
        Dict containing:
            - mesh: CoastalMesh object
            - pyg_data: PyG Data object (if PyTorch available)
            - bathymetry: BathymetryGrid
            - tide_stations: List of TideStation
            - water_levels: Dict of WaterLevelTimeSeries
            - validation: ValidationDataset
    """
    # Initialize manager
    manager = CoastalDataManager(cache_dir=cache_dir or "./coastflow_cache")
    
    # Configure mesh parameters for CoastFlow
    # - Finer resolution near coast for resolving wetting/drying
    # - Coarser offshore to reduce computational cost
    mesh_params = MeshParameters(
        min_edge_length=200.0,   # 200m minimum near shore
        max_edge_length=3000.0,  # 3km maximum offshore
        depth_grading_factor=0.15,  # Strong depth-based grading
        shore_grading_factor=0.1,   # Strong shore-based grading
        shore_grading_distance=15000.0,  # 15km grading zone
    )
    
    logger.info(f"Creating CoastFlow dataset for {region}")
    logger.info(f"  Period: {start_date} to {end_date}")
    
    # Get complete data bundle
    bundle = manager.get_bundle(
        region=region,
        start_date=start_date,
        end_date=end_date,
        include_waves=True,
        mesh_params=mesh_params,
    )
    
    # Build result dict
    result = {
        'mesh': bundle.mesh,
        'bathymetry': bundle.bathymetry,
        'shoreline': bundle.shoreline,
        'tide_stations': bundle.tide_stations,
        'water_levels': bundle.water_levels,
        'tidal_forcing': bundle.tidal_forcing,
        'wave_buoys': bundle.wave_buoys,
        'validation': bundle.validation,
    }
    
    # Create PyG Data object if PyTorch available
    if HAS_TORCH and bundle.mesh is not None:
        result['pyg_data'] = mesh_to_pyg_data(
            bundle.mesh,
            include_forcing=True,
            tidal_forcing=bundle.tidal_forcing,
        )
    
    return result


def mesh_to_pyg_data(
    mesh: CoastalMesh,
    include_forcing: bool = True,
    tidal_forcing = None,
) -> "Data":
    """
    Convert CoastalMesh to PyTorch Geometric Data object.
    
    Creates a graph representation suitable for GNN training:
    - Node features: depth, distance to shore, boundary type, coordinates
    - Edge indices: mesh connectivity
    - Edge features: edge length, direction
    
    Args:
        mesh: CoastalMesh object
        include_forcing: Whether to include tidal forcing as features
        tidal_forcing: Optional TidalForcing object
        
    Returns:
        PyG Data object
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch and PyG required for this function")
    
    # Node features
    # Shape: (num_nodes, num_features)
    node_features = []
    
    # 1. Depth (normalized)
    depths = mesh.depths
    depth_normalized = depths / np.maximum(depths.max(), 1.0)
    node_features.append(depth_normalized[:, np.newaxis])
    
    # 2. Distance to shore (normalized, in km)
    if mesh.distance_to_shore is not None:
        dist_shore = mesh.distance_to_shore / 1000.0  # Convert to km
        dist_normalized = dist_shore / np.maximum(dist_shore.max(), 1.0)
        node_features.append(dist_normalized[:, np.newaxis])
    
    # 3. Boundary type (one-hot encoded)
    n_boundary_types = 3  # interior, shoreline, ocean
    boundary_onehot = np.zeros((mesh.n_nodes, n_boundary_types))
    for i, bt in enumerate(mesh.boundary_types):
        if bt < n_boundary_types:
            boundary_onehot[i, bt] = 1.0
    node_features.append(boundary_onehot)
    
    # 4. Normalized coordinates
    vertices = mesh.vertices  # (N, 2) [lon, lat]
    lon_normalized = (vertices[:, 0] - vertices[:, 0].min()) / \
                     (vertices[:, 0].max() - vertices[:, 0].min() + 1e-6)
    lat_normalized = (vertices[:, 1] - vertices[:, 1].min()) / \
                     (vertices[:, 1].max() - vertices[:, 1].min() + 1e-6)
    node_features.append(lon_normalized[:, np.newaxis])
    node_features.append(lat_normalized[:, np.newaxis])
    
    # 5. Tidal forcing (M2, S2 amplitudes and phases)
    if include_forcing and tidal_forcing is not None:
        # Normalize amplitudes (typically 0-2m)
        amp_normalized = tidal_forcing.amplitudes / 2.0
        node_features.append(amp_normalized)
        
        # Normalize phases to [0, 1]
        phase_normalized = tidal_forcing.phases / 360.0
        node_features.append(phase_normalized)
    
    # Concatenate all features
    x = np.concatenate(node_features, axis=1).astype(np.float32)
    
    # Edge index (COO format for PyG)
    # Convert from (E, 2) to (2, E)
    edge_index = mesh.edges.T.astype(np.int64)
    
    # Make edges bidirectional
    edge_index = np.concatenate([edge_index, edge_index[[1, 0]]], axis=1)
    
    # Edge features
    edge_features = []
    
    # Edge lengths (normalized)
    v0 = mesh.vertices[mesh.edges[:, 0]]
    v1 = mesh.vertices[mesh.edges[:, 1]]
    
    # Approximate edge length in km (at this scale)
    edge_lengths = np.sqrt(
        ((v1[:, 0] - v0[:, 0]) * 111 * np.cos(np.radians(v0[:, 1])))**2 +
        ((v1[:, 1] - v0[:, 1]) * 111)**2
    )
    edge_lengths_normalized = edge_lengths / edge_lengths.max()
    
    # Duplicate for bidirectional edges
    edge_lengths_normalized = np.tile(edge_lengths_normalized, 2)
    edge_features.append(edge_lengths_normalized[:, np.newaxis])
    
    # Edge direction (angle from node 0 to node 1)
    dx = (v1[:, 0] - v0[:, 0]) * np.cos(np.radians(v0[:, 1]))
    dy = v1[:, 1] - v0[:, 1]
    angles = np.arctan2(dy, dx) / np.pi  # Normalize to [-1, 1]
    
    # For bidirectional: second half has opposite direction
    angles_bidir = np.concatenate([angles, -angles])
    edge_features.append(angles_bidir[:, np.newaxis])
    
    edge_attr = np.concatenate(edge_features, axis=1).astype(np.float32)
    
    # Create PyG Data object
    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        pos=torch.from_numpy(mesh.vertices.astype(np.float32)),
    )
    
    # Add mesh-specific attributes
    data.num_nodes = mesh.n_nodes
    data.depths = torch.from_numpy(depths.astype(np.float32))
    data.boundary_mask = torch.from_numpy(
        (mesh.boundary_types > 0).astype(np.float32)
    )
    
    return data


def prepare_training_data(
    mesh: CoastalMesh,
    water_levels: dict,
    tide_stations: list,
    n_timesteps: int = 24,
    dt_hours: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare time-series training data for CoastFlow-GNN.
    
    Creates input/output pairs for training:
    - Input: Previous state + forcing
    - Output: Next state
    
    Args:
        mesh: CoastalMesh
        water_levels: Dict of WaterLevelTimeSeries
        tide_stations: List of TideStation
        n_timesteps: Number of time steps to generate
        dt_hours: Time step in hours
        
    Returns:
        Tuple of (inputs, outputs, times)
    """
    # For demonstration, create synthetic evolution based on tidal harmonics
    # In production, this would use ADCIRC or other model output
    
    t = np.arange(n_timesteps) * dt_hours  # Time in hours
    
    # Simple M2 tidal signal for demonstration
    omega_m2 = 2 * np.pi / 12.42  # M2 period ~12.42 hours
    
    # Generate water elevation at each node
    n_nodes = mesh.n_nodes
    
    # Amplitude varies with depth (smaller in shallow water due to dissipation)
    amplitude = 0.5 * np.clip(mesh.depths / 20, 0.1, 1.0)  # 0.5m max
    
    # Phase varies spatially (wave propagation)
    phase = mesh.vertices[:, 0] * 0.1  # Simple eastward propagation
    
    # Generate time series: shape (n_nodes, n_timesteps)
    elevation = np.zeros((n_nodes, n_timesteps))
    for i in range(n_nodes):
        elevation[i] = amplitude[i] * np.cos(omega_m2 * t - phase[i])
    
    # Create input/output pairs for autoregressive training
    # Input: state at time t
    # Output: state at time t+1
    inputs = elevation[:, :-1]  # (n_nodes, n_timesteps-1)
    outputs = elevation[:, 1:]  # (n_nodes, n_timesteps-1)
    
    return inputs.T, outputs.T, t  # Transpose to (n_timesteps, n_nodes)


def validate_predictions(
    predictions: np.ndarray,
    validation_data: dict,
    mesh: CoastalMesh,
) -> dict:
    """
    Validate model predictions against observations.
    
    Args:
        predictions: (n_nodes, n_timesteps) predicted water levels
        validation_data: ValidationDataset
        mesh: CoastalMesh
        
    Returns:
        Dict of validation metrics
    """
    if 'water_level' not in validation_data:
        return {'error': 'No water level validation data'}
    
    val_dataset = validation_data['water_level']
    
    metrics = {
        'stations': [],
        'rmse': [],
        'correlation': [],
    }
    
    for point in val_dataset.points:
        node_idx = point.mesh_node_idx
        
        # Get predictions at this node
        pred = predictions[node_idx]
        obs = point.observations
        
        # Time alignment would be needed here in production
        # For now, compute simple statistics
        
        # Compute RMSE (on overlapping data)
        n_compare = min(len(pred), len(obs))
        if n_compare > 0:
            rmse = np.sqrt(np.mean((pred[:n_compare] - obs[:n_compare])**2))
            
            # Correlation
            if np.std(pred[:n_compare]) > 0 and np.std(obs[:n_compare]) > 0:
                corr = np.corrcoef(pred[:n_compare], obs[:n_compare])[0, 1]
            else:
                corr = np.nan
            
            metrics['stations'].append(point.station_id)
            metrics['rmse'].append(rmse)
            metrics['correlation'].append(corr)
    
    # Aggregate metrics
    if metrics['rmse']:
        metrics['mean_rmse'] = np.nanmean(metrics['rmse'])
        metrics['mean_correlation'] = np.nanmean(metrics['correlation'])
    
    return metrics


def example_usage():
    """
    Demonstrate the complete workflow for CoastFlow-GNN.
    """
    print("=" * 60)
    print("CoastFlow-GNN Integration Example")
    print("=" * 60)
    print()
    
    # 1. Create dataset
    print("1. Creating dataset...")
    print("   Region: Chesapeake Mouth (smaller for demo)")
    
    dataset = create_coastflow_dataset(
        region="chesapeake_mouth",  # Smaller region for faster demo
        start_date="2023-06-01",
        end_date="2023-06-03",
        cache_dir="./demo_cache",
    )
    
    mesh = dataset['mesh']
    print(f"   Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    print(f"   Tide stations: {len(dataset['tide_stations'])}")
    print(f"   Wave buoys: {len(dataset['wave_buoys'])}")
    print()
    
    # 2. Show PyG Data structure
    if 'pyg_data' in dataset:
        pyg_data = dataset['pyg_data']
        print("2. PyG Data structure:")
        print(f"   Node features: {pyg_data.x.shape}")
        print(f"   Edge index: {pyg_data.edge_index.shape}")
        print(f"   Edge features: {pyg_data.edge_attr.shape}")
        print()
    
    # 3. Prepare training data
    print("3. Preparing training data...")
    inputs, outputs, times = prepare_training_data(
        mesh,
        dataset['water_levels'],
        dataset['tide_stations'],
        n_timesteps=25,  # 24-hour simulation
        dt_hours=1.0,
    )
    print(f"   Input shape: {inputs.shape}")
    print(f"   Output shape: {outputs.shape}")
    print()
    
    # 4. Show validation capabilities
    print("4. Validation data available:")
    if dataset['validation']:
        for var_name, val_data in dataset['validation'].items():
            print(f"   {var_name}: {val_data.n_observations} observations "
                  f"at {val_data.spatial_coverage} stations")
    print()
    
    print("=" * 60)
    print("Example complete!")
    print()
    print("Next steps for your GNN training:")
    print("  1. Define your GNN architecture (e.g., GraphSAGE, GAT)")
    print("  2. Use pyg_data as input graph structure")
    print("  3. Train on (inputs, outputs) pairs")
    print("  4. Validate using validate_predictions()")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
