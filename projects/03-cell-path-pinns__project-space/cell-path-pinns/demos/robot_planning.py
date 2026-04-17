"""
Demo: Robot Path Planning on Terrain with Obstacles

This demo shows how to use the geodesic trajectory framework for robot
path planning, finding energy-efficient paths that avoid obstacles.

Usage:
    python -m demos.robot_planning

Features demonstrated:
    - TerrainField with obstacles and goals
    - BoundaryConditionedTrajectory for exact start/end
    - RobotPathLoss with curvature constraints
    - Visualization with shared/trajectory utilities
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import TrajectoryPINN
from src.core.trajectory_pinn import BoundaryConditionedTrajectory
from src.domains.robotics import TerrainField, RobotPathLoss, RoboticsAdapter


def create_terrain_with_obstacles():
    """Create a terrain with hills and obstacles for demo."""
    
    class DemoTerrain(TerrainField):
        """Custom terrain with analytic hills and obstacles."""
        
        def __init__(self):
            super().__init__(spatial_dim=2, mode="learned", hidden_dim=32)
            
            # Override forward for demo
            self._use_custom = True
        
        def forward(self, x, t=None):
            if not self._use_custom:
                return super().forward(x, t)
            
            # Base: gentle hills
            elevation = 0.3 * torch.sin(2 * np.pi * x[:, 0:1]) * torch.cos(2 * np.pi * x[:, 1:2])
            
            # Obstacle 1: wall segment
            obstacle1_dist = torch.sqrt((x[:, 0:1] - 0.3)**2 + (x[:, 1:1] - 0.5)**2)
            obstacle1 = 50.0 * torch.exp(-20 * obstacle1_dist**2)
            
            # Obstacle 2: circular barrier
            obstacle2_dist = torch.sqrt((x[:, 0:1] + 0.2)**2 + (x[:, 1:2] - 0.3)**2)
            obstacle2 = 50.0 * torch.exp(-20 * obstacle2_dist**2)
            
            # Obstacle 3: elongated barrier
            obstacle3_x = torch.abs(x[:, 0:1] - 0.0)
            obstacle3_y = torch.abs(x[:, 1:2] + 0.3)
            obstacle3 = 30.0 * torch.exp(-10 * (obstacle3_x**2 + 0.2 * obstacle3_y**2))
            
            return elevation + obstacle1 + obstacle2 + obstacle3
    
    return DemoTerrain()


def train_robot_planner(terrain, n_trajectories=50, epochs=500, verbose=True):
    """
    Train a geodesic path planner for robot navigation.
    
    Args:
        terrain: TerrainField instance
        n_trajectories: Number of training trajectories
        epochs: Training epochs
        verbose: Print progress
        
    Returns:
        Trained trajectory network
    """
    # Create boundary-conditioned network (guarantees exact endpoints)
    trajectory_net = BoundaryConditionedTrajectory(
        spatial_dim=2,
        hidden_dim=64,
        num_layers=3,
    )
    
    # Loss function
    loss_fn = RobotPathLoss(
        terrain_field=terrain,
        lambda_speed=0.1,
        lambda_boundary=10.0,  # Not needed for BC network, but included
        lambda_terrain=1.0,
        lambda_curvature=0.5,
        target_speed=1.0,
        max_curvature=2.0,
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(trajectory_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training data: random start/end pairs
    torch.manual_seed(42)
    starts = torch.rand(n_trajectories, 2) * 2 - 1  # [-1, 1]²
    ends = torch.rand(n_trajectories, 2) * 2 - 1
    
    # Time points for trajectory
    t = torch.linspace(0, 1, 50).unsqueeze(-1)  # [50, 1]
    times = torch.linspace(0, 1, 50).unsqueeze(0).expand(1, -1)  # [1, 50]
    
    # Training loop
    history = {'loss': [], 'terrain': [], 'curvature': [], 'speed': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        total_loss = 0.0
        losses_epoch = {'terrain': 0, 'curvature': 0, 'speed': 0}
        
        for i in range(n_trajectories):
            start = starts[i]
            end = ends[i]
            
            # Predict trajectory
            path = trajectory_net(t, start=start, end=end)  # [50, 2]
            path_batch = path.unsqueeze(0)  # [1, 50, 2]
            
            # Compute loss
            loss, loss_dict = loss_fn(
                positions=path_batch,
                times=times,
                start=start.unsqueeze(0),
                end=end.unsqueeze(0),
            )
            
            total_loss += loss
            for key in losses_epoch:
                if key in loss_dict:
                    losses_epoch[key] += loss_dict[key].item()
        
        total_loss /= n_trajectories
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trajectory_net.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Record history
        history['loss'].append(total_loss.item())
        for key in losses_epoch:
            history[key].append(losses_epoch[key] / n_trajectories)
        
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:4d}: Loss = {total_loss.item():.4f}, "
                  f"Terrain = {losses_epoch['terrain']/n_trajectories:.4f}, "
                  f"Curvature = {losses_epoch['curvature']/n_trajectories:.4f}")
    
    return trajectory_net, history


def visualize_paths(trajectory_net, terrain, n_paths=6, save_path=None):
    """
    Visualize planned paths on terrain.
    
    Creates a grid of path visualizations showing different start/end pairs.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Fixed random seed for reproducibility
    torch.manual_seed(123)
    
    for i, ax in enumerate(axes):
        # Random start/end
        start = torch.rand(2) * 1.6 - 0.8  # [-0.8, 0.8]
        end = torch.rand(2) * 1.6 - 0.8
        
        # Generate path
        t = torch.linspace(0, 1, 100).unsqueeze(-1)
        with torch.no_grad():
            path = trajectory_net(t, start=start, end=end)
        
        path_np = path.numpy()
        
        # Plot terrain
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32)
        
        with torch.no_grad():
            Z = terrain(points).numpy().reshape(100, 100)
        
        contour = ax.contourf(X, Y, Z, levels=30, cmap='terrain', alpha=0.7)
        ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Plot path
        ax.plot(path_np[:, 0], path_np[:, 1], 'b-', linewidth=2.5, label='Path')
        ax.scatter(start[0], start[1], c='green', s=150, marker='o', 
                   edgecolors='white', linewidths=2, zorder=5, label='Start')
        ax.scatter(end[0], end[1], c='red', s=200, marker='*', 
                   edgecolors='white', linewidths=2, zorder=5, label='End')
        
        # Path metrics
        path_length = np.sum(np.sqrt(np.sum(np.diff(path_np, axis=0)**2, axis=1)))
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Path {i+1} (Length: {path_length:.2f})')
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Robot Path Planning with Geodesic Trajectories', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def benchmark_inference(trajectory_net, n_runs=100):
    """Benchmark inference time."""
    t = torch.linspace(0, 1, 50).unsqueeze(-1)
    start = torch.rand(2)
    end = torch.rand(2)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = trajectory_net(t, start=start, end=end)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = trajectory_net(t, start=start, end=end)
        times.append((time.perf_counter() - start_time) * 1000)
    
    print(f"\n=== Inference Benchmark ===")
    print(f"Mean: {np.mean(times):.3f} ms")
    print(f"Std:  {np.std(times):.3f} ms")
    print(f"Min:  {np.min(times):.3f} ms")
    print(f"Max:  {np.max(times):.3f} ms")
    print(f"Target (<10ms): {'✓ PASS' if np.mean(times) < 10 else '✗ FAIL'}")


def main():
    """Run robot path planning demo."""
    print("=" * 60)
    print("Robot Path Planning Demo - Geodesic Trajectories")
    print("=" * 60)
    
    # Create terrain
    print("\n1. Creating terrain with obstacles...")
    terrain = create_terrain_with_obstacles()
    
    # Train planner
    print("\n2. Training path planner...")
    trajectory_net, history = train_robot_planner(
        terrain, 
        n_trajectories=50, 
        epochs=300,
        verbose=True
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in trajectory_net.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Benchmark
    print("\n3. Benchmarking inference...")
    benchmark_inference(trajectory_net)
    
    # Visualize
    print("\n4. Visualizing paths...")
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    visualize_paths(
        trajectory_net, 
        terrain, 
        save_path=str(output_dir / "robot_planning_demo.png")
    )
    
    # Export to ONNX (if available)
    print("\n5. Exporting to ONNX...")
    try:
        from src.export import quick_export_onnx, benchmark_onnx_inference
        
        onnx_path = str(output_dir / "robot_planner.onnx")
        quick_export_onnx(
            trajectory_net,
            onnx_path,
            spatial_dim=2,
            n_time_steps=50,
        )
        print(f"Exported to: {onnx_path}")
        
        # Benchmark ONNX
        onnx_stats = benchmark_onnx_inference(onnx_path, n_time_steps=50)
        print(f"ONNX inference: {onnx_stats['mean_ms']:.3f} ± {onnx_stats['std_ms']:.3f} ms")
        
    except ImportError as e:
        print(f"ONNX export skipped: {e}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
