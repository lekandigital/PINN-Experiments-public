"""
Synthetic Coastal Dataset for CoastFlow-GNN

Generates synthetic coastal mesh data with:
- Procedural DEM-like elevation surfaces
- Random wind forcing conditions
- Ground truth from simplified shallow water equations

This enables rapid prototyping without requiring real NOAA DEM data
or OpenFOAM simulations.
"""

import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from typing import Optional, Tuple, List
from scipy.spatial import Delaunay
import os


class SyntheticCoastalDataset(InMemoryDataset):
    """
    Synthetic coastal flow dataset for training CoastFlow-GNN.
    
    Each sample contains:
        - x: Node features [N, 6] = [x, y, z, elevation, wind_u, wind_v]
        - y: Labels [N, 4] = [u_x, u_y, u_z, wave_height]
        - edge_index: Graph connectivity from Delaunay triangulation
        - pos: Node positions [N, 3]
        - boundary_mask: Boolean mask for boundary nodes
        - boundary_type: Boundary condition type (0=interior, 1=wall, 2=free_surface)
    
    Args:
        root: Root directory for dataset storage
        num_samples: Total number of samples to generate
        num_nodes: Number of nodes per mesh (approximate)
        grid_size: Physical domain size in meters
        transform: Data transformation
        pre_transform: Pre-transformation
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        root: Optional[str] = None,
        num_samples: int = 140,
        num_nodes: int = 200,
        grid_size: float = 1000.0,
        transform=None,
        pre_transform=None,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.grid_size = grid_size
        self.seed = seed
        
        if root is None:
            root = './data/synthetic_coastal'
        
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) -> List[str]:
        return []  # No raw files needed
    
    @property
    def processed_file_names(self) -> List[str]:
        return [f'coastal_data_{self.num_samples}_{self.num_nodes}.pt']
    
    def download(self):
        pass  # No download needed
    
    def process(self):
        """Generate synthetic coastal data."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        data_list = []
        
        for i in range(self.num_samples):
            data = self._generate_sample(i)
            data_list.append(data)
        
        self.save(data_list, self.processed_paths[0])
    
    def _generate_sample(self, sample_idx: int) -> Data:
        """Generate a single coastal mesh sample."""
        
        # Generate irregular mesh points
        pos = self._generate_mesh_points()
        N = pos.shape[0]
        
        # Generate elevation (DEM-like surface)
        elevation = self._generate_elevation(pos)
        
        # Generate wind forcing
        wind_u, wind_v = self._generate_wind_forcing(pos)
        
        # Create node features: [x, y, z, elevation, wind_u, wind_v]
        z_coords = elevation  # Use elevation as z for 3D position
        node_features = torch.cat([
            pos,
            elevation.unsqueeze(-1),
            wind_u.unsqueeze(-1),
            wind_v.unsqueeze(-1),
        ], dim=-1)  # [N, 6]
        
        # Generate graph connectivity via Delaunay triangulation
        edge_index = self._create_graph_edges(pos[:, :2])
        
        # Identify boundary nodes
        boundary_mask, boundary_type = self._identify_boundaries(pos)
        
        # Generate ground truth using simplified shallow water equations
        u, wave_height = self._compute_ground_truth(
            pos, elevation, wind_u, wind_v, edge_index, boundary_mask
        )
        
        # Labels: [u_x, u_y, u_z, wave_height]
        labels = torch.cat([u, wave_height.unsqueeze(-1)], dim=-1)  # [N, 4]
        
        # Create 3D positions
        pos_3d = torch.cat([pos[:, :2], elevation.unsqueeze(-1)], dim=-1)
        
        data = Data(
            x=node_features.float(),
            y=labels.float(),
            edge_index=edge_index,
            pos=pos_3d.float(),
            boundary_mask=boundary_mask,
            boundary_type=boundary_type,
        )
        
        return data
    
    def _generate_mesh_points(self) -> torch.Tensor:
        """Generate irregular mesh points in 2D domain."""
        # Base regular grid with some jitter
        n_side = int(np.sqrt(self.num_nodes))
        x = np.linspace(0, self.grid_size, n_side)
        y = np.linspace(0, self.grid_size, n_side)
        xx, yy = np.meshgrid(x, y)
        
        # Add jitter for irregularity
        jitter_scale = self.grid_size / n_side * 0.2
        xx = xx + np.random.randn(*xx.shape) * jitter_scale
        yy = yy + np.random.randn(*yy.shape) * jitter_scale
        
        # Flatten
        points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        
        # Add some random points for more irregularity
        n_extra = int(self.num_nodes * 0.1)
        extra_points = np.random.rand(n_extra, 2) * self.grid_size
        points = np.vstack([points, extra_points])
        
        # Clip to domain
        points = np.clip(points, 0, self.grid_size)
        
        # Add placeholder z coordinate (will be replaced by elevation)
        z = np.zeros((points.shape[0], 1))
        points_3d = np.hstack([points, z])
        
        return torch.tensor(points_3d, dtype=torch.float32)
    
    def _generate_elevation(self, pos: torch.Tensor) -> torch.Tensor:
        """Generate DEM-like elevation surface."""
        x, y = pos[:, 0], pos[:, 1]
        
        # Normalize coordinates
        x_norm = x / self.grid_size
        y_norm = y / self.grid_size
        
        # Create coastal profile: descending from land to sea
        # Use Perlin-like noise with multiple octaves
        elevation = torch.zeros(x.shape[0])
        
        # Base coastal gradient (land to sea)
        coastal_gradient = 20 * (1 - x_norm) - 5  # Land on left, sea on right
        
        # Add terrain features
        freq1 = 2 * np.pi * 3  # Low frequency terrain
        freq2 = 2 * np.pi * 7  # Higher frequency features
        
        terrain = (
            3 * torch.sin(freq1 * x_norm) * torch.cos(freq1 * 0.5 * y_norm)
            + 1.5 * torch.sin(freq2 * x_norm + 1.3) * torch.sin(freq2 * y_norm)
            + 0.5 * torch.sin(freq2 * 2 * x_norm) * torch.cos(freq2 * 1.5 * y_norm)
        )
        
        # Random bumps
        n_bumps = np.random.randint(3, 8)
        bumps = torch.zeros_like(x)
        for _ in range(n_bumps):
            cx = np.random.rand() * self.grid_size
            cy = np.random.rand() * self.grid_size
            height = np.random.rand() * 5 + 1
            width = np.random.rand() * 100 + 50
            dist_sq = (x - cx)**2 + (y - cy)**2
            bumps += height * torch.exp(-dist_sq / (2 * width**2))
        
        elevation = coastal_gradient + terrain + bumps
        
        return elevation
    
    def _generate_wind_forcing(
        self, pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate wind field."""
        N = pos.shape[0]
        
        # Random base wind direction and speed
        wind_speed = np.random.rand() * 15 + 5  # 5-20 m/s
        wind_dir = np.random.rand() * 2 * np.pi  # Random direction
        
        wind_u_base = wind_speed * np.cos(wind_dir)
        wind_v_base = wind_speed * np.sin(wind_dir)
        
        # Add spatial variation (atmospheric boundary layer effects)
        x_norm = pos[:, 0] / self.grid_size
        y_norm = pos[:, 1] / self.grid_size
        
        # Wind variation
        wind_u = torch.full((N,), wind_u_base)
        wind_v = torch.full((N,), wind_v_base)
        
        # Add some turbulent fluctuations
        wind_u += torch.randn(N) * 1.0
        wind_v += torch.randn(N) * 1.0
        
        # Reduce wind over land (sheltering effect)
        land_mask = x_norm < 0.3
        wind_u[land_mask] *= 0.5
        wind_v[land_mask] *= 0.5
        
        return wind_u.float(), wind_v.float()
    
    def _create_graph_edges(self, pos_2d: torch.Tensor) -> torch.Tensor:
        """Create graph edges using Delaunay triangulation."""
        points = pos_2d.numpy()
        
        try:
            tri = Delaunay(points)
            
            # Extract edges from triangulation
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        edge = tuple(sorted([simplex[i], simplex[j]]))
                        edges.add(edge)
            
            edges = list(edges)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Make undirected (add reverse edges)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            
        except Exception as e:
            # Fallback: k-nearest neighbors
            print(f"Delaunay failed, using KNN: {e}")
            from torch_geometric.nn import knn_graph
            edge_index = knn_graph(pos_2d, k=8, loop=False)
        
        return edge_index
    
    def _identify_boundaries(
        self, pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identify boundary nodes and their types."""
        N = pos.shape[0]
        x, y = pos[:, 0], pos[:, 1]
        
        boundary_mask = torch.zeros(N, dtype=torch.bool)
        boundary_type = torch.zeros(N, dtype=torch.long)  # 0 = interior
        
        # Define boundary tolerance
        tol = self.grid_size * 0.02
        
        # Left boundary (inlet/land): type 1 (wall/no-slip)
        left_mask = x < tol
        boundary_mask[left_mask] = True
        boundary_type[left_mask] = 1
        
        # Right boundary (outlet/sea): type 4 (outlet)
        right_mask = x > self.grid_size - tol
        boundary_mask[right_mask] = True
        boundary_type[right_mask] = 4
        
        # Top/bottom boundaries: type 1 (walls)
        top_mask = y > self.grid_size - tol
        bottom_mask = y < tol
        boundary_mask[top_mask | bottom_mask] = True
        boundary_type[top_mask] = 1
        boundary_type[bottom_mask] = 1
        
        # Free surface: nodes near water surface (elevation around 0)
        # This is simplified - in reality would depend on water level
        
        return boundary_mask, boundary_type
    
    def _compute_ground_truth(
        self,
        pos: torch.Tensor,
        elevation: torch.Tensor,
        wind_u: torch.Tensor,
        wind_v: torch.Tensor,
        edge_index: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ground truth using simplified shallow water equations.
        
        This is a simplified analytical solution for testing purposes.
        Real training would use OpenFOAM simulation data.
        """
        N = pos.shape[0]
        x, y = pos[:, 0], pos[:, 1]
        x_norm = x / self.grid_size
        
        # Simplified velocity field
        # Flow driven by wind and pressure gradient (from elevation)
        
        # Base flow from wind forcing
        u_x = wind_u * 0.03  # Surface current is ~3% of wind speed
        u_y = wind_v * 0.03
        
        # Add pressure-driven flow from elevation gradient
        # Flow tends downhill
        u_x -= 0.5 * torch.sin(2 * np.pi * x_norm)
        
        # Vertical velocity (simplified)
        u_z = torch.zeros(N)
        # Small vertical motion from wave oscillation
        wave_freq = 2 * np.pi * 2 / self.grid_size
        u_z = 0.1 * torch.sin(wave_freq * x) * torch.cos(wave_freq * y)
        
        # Apply boundary conditions
        u_x[boundary_mask & (pos[:, 0] < self.grid_size * 0.1)] = 0  # Left wall
        u_y[boundary_mask & (pos[:, 0] < self.grid_size * 0.1)] = 0
        u_z[boundary_mask & (pos[:, 0] < self.grid_size * 0.1)] = 0
        
        # Combine velocity
        u = torch.stack([u_x, u_y, u_z], dim=-1)
        
        # Wave height
        # Wind-wave generation: H ≈ C * U² * F / g (simplified Sverdrup-Munk)
        # Where F is fetch length
        g = 9.81
        fetch = x  # Distance from shore
        wind_speed = torch.sqrt(wind_u**2 + wind_v**2)
        
        # Simplified wave height formula
        wave_height = 0.01 * wind_speed**2 * torch.sqrt(fetch.clamp(min=1)) / g
        
        # Add wave variability
        wave_height += 0.1 * torch.sin(wave_freq * x) * torch.cos(wave_freq * 0.5 * y)
        
        # Ensure positive wave height
        wave_height = wave_height.clamp(min=0)
        
        # Reduce wave height in shallow water (near land)
        shallow_mask = elevation > -2
        wave_height[shallow_mask] *= 0.5
        
        return u.float(), wave_height.float()


def create_data_loaders(
    dataset: SyntheticCoastalDataset,
    train_ratio: float = 0.714,  # 100/140
    val_ratio: float = 0.143,    # 20/140
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders.
    
    Default split: 100 train, 20 val, 20 test (from 140 samples)
    
    Args:
        dataset: SyntheticCoastalDataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        batch_size: Batch size
        shuffle: Shuffle training data
        num_workers: DataLoader workers
        
    Returns:
        train_loader, val_loader, test_loader
    """
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    # Deterministic split
    torch.manual_seed(42)
    indices = torch.randperm(n)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing SyntheticCoastalDataset...")
    
    # Create dataset
    dataset = SyntheticCoastalDataset(
        root='./data/test_synthetic',
        num_samples=20,
        num_nodes=100,
        seed=42,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Inspect first sample
    data = dataset[0]
    print(f"\nSample 0:")
    print(f"  Node features (x): {data.x.shape}")
    print(f"  Labels (y): {data.y.shape}")
    print(f"  Edge index: {data.edge_index.shape}")
    print(f"  Position: {data.pos.shape}")
    print(f"  Boundary mask: {data.boundary_mask.sum().item()} boundary nodes")
    
    # Test data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, batch_size=4
    )
    
    print(f"\nData loaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch:")
    print(f"  x: {batch.x.shape}")
    print(f"  y: {batch.y.shape}")
    print(f"  batch: {batch.batch.shape}")
    
    print("\n✓ Dataset tests passed!")
