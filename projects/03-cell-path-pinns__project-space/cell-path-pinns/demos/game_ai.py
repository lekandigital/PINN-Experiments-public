"""
Demo: Game AI NPC Pathfinding with Geodesic Trajectories

This demo shows how to use the geodesic trajectory framework for
real-time game AI pathfinding with sub-millisecond inference.

Usage:
    python -m demos.game_ai

Features demonstrated:
    - GameWorldField with obstacles, hazards, and goals
    - Lightweight network optimized for speed
    - Real-time inference benchmarking (<1ms target)
    - Interactive visualization

Performance targets:
    - Inference: <1ms for 60 FPS games
    - Model size: <100KB for embedding
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.domains.game_ai import GameWorldField, GameAILoss, GameAIAdapter
from src.core.trajectory_pinn import BoundaryConditionedTrajectory


def create_game_world():
    """Create a game world with obstacles, hazards, and goals."""
    adapter = GameAIAdapter(world_size=100, hidden_dim=32)
    
    # Add obstacles (walls, buildings) - impassable
    adapter.add_obstacle((30, 50), radius=12)  # Central pillar
    adapter.add_obstacle((70, 30), radius=8)   # Side building
    adapter.add_obstacle((50, 75), radius=10)  # Upper obstacle
    
    # Add hazards (lava, poison) - passable but costly
    adapter.add_hazard((50, 50), radius=15, cost=5.0)  # Central hazard
    adapter.add_hazard((20, 20), radius=8, cost=3.0)   # Corner hazard
    
    # Add goal (treasure, exit)
    adapter.add_goal((85, 85), attraction=15.0)
    
    return adapter


def train_game_pathfinder(adapter, epochs=200, verbose=True):
    """
    Train a lightweight pathfinder for real-time game AI.
    
    Args:
        adapter: GameAIAdapter with world configuration
        epochs: Training epochs
        verbose: Print progress
        
    Returns:
        Trained model and loss function
    """
    model, loss_fn = adapter.create_model(
        lambda_speed=0.1,
        lambda_boundary=10.0,
        lambda_navigation=1.0,
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training data: random start positions, fixed goal
    torch.manual_seed(42)
    n_trajectories = 30
    starts = torch.rand(n_trajectories, 2) * 60 + 10  # [10, 70]²
    goal = torch.tensor([85.0, 85.0])  # Fixed goal
    
    # Time points
    t = torch.linspace(0, 1, 50).unsqueeze(-1)
    times = torch.linspace(0, 1, 50).unsqueeze(0)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        total_loss = 0.0
        
        for i in range(n_trajectories):
            start = starts[i]
            
            # Predict path
            path = model(t, start=start, end=goal)
            path_batch = path.unsqueeze(0)
            
            # Compute loss
            loss, _ = loss_fn(
                positions=path_batch,
                times=times,
                start=start.unsqueeze(0),
                end=goal.unsqueeze(0),
            )
            
            total_loss += loss
        
        total_loss /= n_trajectories
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:4d}: Loss = {total_loss.item():.4f}")
    
    return model


def visualize_game_world(model, adapter, n_paths=5, save_path=None):
    """
    Visualize the game world with paths.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    world = model.game_world
    world_size = world.world_size
    
    # Plot potential field background
    x = np.linspace(0, world_size, 100)
    y = np.linspace(0, world_size, 100)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32)
    
    with torch.no_grad():
        Z = world(points).numpy().reshape(100, 100)
    
    # Clip for visualization
    Z = np.clip(Z, -20, 50)
    
    contour = ax.contourf(X, Y, Z, levels=30, cmap='RdYlGn_r', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Cost (lower is better)')
    
    # Draw obstacles
    for center, radius in world._obstacles:
        circle = Circle(center.numpy(), radius, fill=True, 
                       color='black', alpha=0.8, label='Obstacle')
        ax.add_patch(circle)
    
    # Draw hazards
    for center, radius, cost in world._hazards:
        circle = Circle(center.numpy(), radius, fill=True,
                       color='orange', alpha=0.5, label='Hazard')
        ax.add_patch(circle)
    
    # Draw goals
    for position, attraction in world._goals:
        ax.scatter(*position.numpy(), c='gold', s=300, marker='*',
                   edgecolors='black', linewidths=2, zorder=10, label='Goal')
    
    # Plot paths from different starting positions
    torch.manual_seed(999)
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_paths))
    goal = torch.tensor([85.0, 85.0])
    
    for i in range(n_paths):
        # Random start avoiding obstacles
        while True:
            start = torch.rand(2) * 60 + 10
            # Check not inside obstacle
            ok = True
            for center, radius in world._obstacles:
                if torch.norm(start - center) < radius + 5:
                    ok = False
                    break
            if ok:
                break
        
        # Generate path
        path = model.find_path(tuple(start.numpy()), tuple(goal.numpy()), n_points=100)
        path_np = path.numpy()
        
        ax.plot(path_np[:, 0], path_np[:, 1], color=colors[i], 
                linewidth=2.5, alpha=0.9, label=f'NPC {i+1}')
        ax.scatter(start[0], start[1], c='green', s=100, marker='o',
                   edgecolors='white', zorder=5)
    
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Game AI Pathfinding with Geodesic Trajectories\n'
                 '(Black = Obstacles, Orange = Hazards, Star = Goal)', fontsize=12)
    
    # Legend (avoiding duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def benchmark_realtime(model, n_runs=1000):
    """
    Benchmark inference for real-time requirements.
    
    Target: <1ms mean inference for 60 FPS games.
    """
    world_size = model.game_world.world_size
    
    # Warmup
    for _ in range(50):
        start = tuple(np.random.uniform(10, 90, 2))
        goal = tuple(np.random.uniform(10, 90, 2))
        _ = model.find_path(start, goal, n_points=50)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = tuple(np.random.uniform(10, 90, 2))
        goal = tuple(np.random.uniform(10, 90, 2))
        _ = model.find_path(start, goal, n_points=50)
        times.append(model.last_inference_ms)
    
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    p95_ms = np.percentile(times, 95)
    p99_ms = np.percentile(times, 99)
    
    print(f"\n{'='*50}")
    print("Real-Time Benchmark Results")
    print(f"{'='*50}")
    print(f"Runs:     {n_runs}")
    print(f"Mean:     {mean_ms:.3f} ms")
    print(f"Std:      {std_ms:.3f} ms")
    print(f"P95:      {p95_ms:.3f} ms")
    print(f"P99:      {p99_ms:.3f} ms")
    print(f"Min:      {np.min(times):.3f} ms")
    print(f"Max:      {np.max(times):.3f} ms")
    print(f"{'='*50}")
    
    # Check requirements
    target_1ms = mean_ms < 1.0
    target_p99 = p99_ms < 2.0
    
    print(f"\n<1ms Target:  {'✓ PASS' if target_1ms else '✗ FAIL'}")
    print(f"P99 <2ms:     {'✓ PASS' if target_p99 else '✗ FAIL'}")
    
    # FPS estimation
    if mean_ms > 0:
        max_fps = 1000 / mean_ms
        print(f"\nMax theoretical FPS: {max_fps:.0f}")
        print(f"60 FPS budget usage: {(mean_ms / 16.67) * 100:.1f}%")
    
    return {
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'p95_ms': p95_ms,
        'p99_ms': p99_ms,
        'meets_target': target_1ms,
    }


def estimate_model_size(model):
    """Estimate model size for game embedding."""
    n_params = sum(p.numel() for p in model.parameters())
    size_bytes = n_params * 4  # float32
    size_kb = size_bytes / 1024
    
    print(f"\n{'='*50}")
    print("Model Size Analysis")
    print(f"{'='*50}")
    print(f"Parameters:  {n_params:,}")
    print(f"Size:        {size_kb:.2f} KB")
    print(f"<100KB:      {'✓ PASS' if size_kb < 100 else '✗ FAIL'}")
    
    return size_kb


def main():
    """Run game AI pathfinding demo."""
    print("=" * 60)
    print("Game AI Pathfinding Demo - Geodesic Trajectories")
    print("Target: <1ms inference for 60 FPS games")
    print("=" * 60)
    
    # Create game world
    print("\n1. Creating game world...")
    adapter = create_game_world()
    print("   - 3 obstacles (impassable)")
    print("   - 2 hazards (costly)")
    print("   - 1 goal (attractive)")
    
    # Train pathfinder
    print("\n2. Training lightweight pathfinder...")
    model = train_game_pathfinder(adapter, epochs=200, verbose=True)
    
    # Model size
    size_kb = estimate_model_size(model)
    
    # Benchmark
    print("\n3. Real-time benchmark...")
    stats = benchmark_realtime(model, n_runs=500)
    
    # Visualize
    print("\n4. Visualizing paths...")
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    visualize_game_world(
        model, 
        adapter,
        n_paths=5,
        save_path=str(output_dir / "game_ai_demo.png")
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Inference:  {stats['mean_ms']:.3f} ms ({'✓' if stats['meets_target'] else '✗'} <1ms)")
    print(f"Size:       {size_kb:.1f} KB ({'✓' if size_kb < 100 else '✗'} <100KB)")
    print(f"Ready for real-time game AI: {'YES' if stats['meets_target'] and size_kb < 100 else 'NEEDS OPTIMIZATION'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
