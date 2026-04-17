"""
Game AI Domain Adapter: NPC Pathfinding on Cost Landscapes.

Models NPC navigation as geodesics on game world cost maps.
Optimized for real-time inference (<1ms) for 60 FPS games.

Key Features:
- Obstacle avoidance (high potential barriers)
- Goal attraction (negative potential at objectives)
- Hazard costs (lava, poison = moderate potential)
- Patrol paths, stealth routes, etc.

Performance Target:
- Inference time: <1ms for 50-point trajectories
- Model size: <100KB for embedding in games
"""

from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import time

from ..core.potential_field import PotentialFieldBase
from ..core.geodesic_loss import GeodesicLoss


class GameWorldField(PotentialFieldBase):
    """
    Game world potential field for NPC navigation.
    
    Combines:
    - Obstacles (impassable, very high potential)
    - Hazards (passable but costly)
    - Goals (attractive, negative potential)
    - Terrain costs (optional base traversability)
    """
    
    def __init__(
        self,
        spatial_dim: int = 2,
        world_size: float = 100.0,
        hidden_dim: int = 32,  # Smaller for fast inference
    ):
        """
        Initialize game world field.
        
        Args:
            spatial_dim: Usually 2 for top-down games
            world_size: World coordinate range (for normalization)
            hidden_dim: Network hidden dim (keep small for speed)
        """
        super().__init__(spatial_dim=spatial_dim, time_dependent=False)
        
        self.world_size = world_size
        self.hidden_dim = hidden_dim
        
        # Lightweight base terrain network
        self.base_net = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Component lists
        self._obstacles = []  # (center, radius)
        self._hazards = []    # (center, radius, cost)
        self._goals = []      # (position, attraction)
    
    def add_obstacle(self, center: Tuple[float, ...], radius: float):
        """Add impassable obstacle."""
        self._obstacles.append((
            torch.tensor(center, dtype=torch.float32),
            radius
        ))
    
    def add_hazard(self, center: Tuple[float, ...], radius: float, cost: float = 5.0):
        """Add passable hazard with cost."""
        self._hazards.append((
            torch.tensor(center, dtype=torch.float32),
            radius,
            cost
        ))
    
    def add_goal(self, position: Tuple[float, ...], attraction: float = 10.0):
        """Add goal with attractive potential."""
        self._goals.append((
            torch.tensor(position, dtype=torch.float32),
            attraction
        ))
    
    def clear_dynamic(self):
        """Clear all dynamic elements (obstacles, hazards, goals)."""
        self._obstacles = []
        self._hazards = []
        self._goals = []
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute game world potential.
        
        Optimized for fast inference.
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Normalize coordinates
        x_norm = x / self.world_size
        
        # Base terrain potential (small contribution)
        potential = self.base_net(x_norm) * 0.1
        
        # Obstacles: very high potential barrier
        for center, radius in self._obstacles:
            center = center.to(device)
            dist = torch.norm(x - center, dim=-1, keepdim=True)
            # Sharp barrier at obstacle boundary
            obstacle_pot = 100.0 * torch.exp(-5.0 * torch.relu(dist - radius))
            # Extra high inside
            inside = (dist < radius).float()
            obstacle_pot = obstacle_pot + inside * 1000.0
            potential = potential + obstacle_pot
        
        # Hazards: moderate cost
        for center, radius, cost in self._hazards:
            center = center.to(device)
            dist = torch.norm(x - center, dim=-1, keepdim=True)
            # Gaussian cost centered at hazard
            hazard_pot = cost * torch.exp(-0.5 * (dist / radius) ** 2)
            potential = potential + hazard_pot
        
        # Goals: negative potential (attractive)
        for position, attraction in self._goals:
            position = position.to(device)
            dist = torch.norm(x - position, dim=-1, keepdim=True)
            # Attractive well near goal
            goal_pot = -attraction * torch.exp(-0.5 * (dist / 5.0) ** 2)
            potential = potential + goal_pot
        
        return potential


class GameAILoss(GeodesicLoss):
    """
    Game AI pathfinding loss.
    
    Tuned for:
    - Fast training (few epochs)
    - Obstacle avoidance
    - Smooth, natural-looking paths
    """
    
    def __init__(
        self,
        game_world: GameWorldField,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_navigation: float = 1.0,
        lambda_smoothness: float = 0.1,
        target_speed: float = 1.0,
    ):
        super().__init__(
            potential_field=game_world,
            lambda_speed=lambda_speed,
            lambda_boundary=lambda_boundary,
            lambda_gradient=lambda_navigation,
            lambda_curvature=lambda_smoothness,
            lambda_data=0.0,
            target_speed=target_speed,
            max_curvature=0.5,  # Allow sharp turns for game characters
            gradient_direction="descent",
        )


class GameAIAdapter:
    """
    High-level adapter for game AI pathfinding.
    
    Optimized for real-time inference in games.
    
    Example:
        >>> adapter = GameAIAdapter(world_size=100)
        >>> adapter.add_obstacle((50, 50), radius=10)
        >>> adapter.add_goal((90, 90))
        >>> model, loss = adapter.create_model()
        >>> 
        >>> # Real-time path query
        >>> path = model.find_path(start=(10, 10), goal=(90, 90), n_points=50)
        >>> print(f"Inference time: {model.last_inference_ms:.2f} ms")
    """
    
    def __init__(
        self,
        spatial_dim: int = 2,
        world_size: float = 100.0,
        hidden_dim: int = 32,
    ):
        self.spatial_dim = spatial_dim
        self.world_size = world_size
        self.hidden_dim = hidden_dim
        
        self._obstacles = []
        self._hazards = []
        self._goals = []
    
    def add_obstacle(self, center: Tuple[float, ...], radius: float):
        """Add obstacle."""
        self._obstacles.append((center, radius))
    
    def add_hazard(self, center: Tuple[float, ...], radius: float, cost: float = 5.0):
        """Add hazard."""
        self._hazards.append((center, radius, cost))
    
    def add_goal(self, position: Tuple[float, ...], attraction: float = 10.0):
        """Add goal."""
        self._goals.append((position, attraction))
    
    def create_model(
        self,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_navigation: float = 1.0,
    ) -> Tuple[nn.Module, GameAILoss]:
        """Create lightweight pathfinding model."""
        from ..core.trajectory_pinn import BoundaryConditionedTrajectory
        
        # Create game world
        game_world = GameWorldField(
            spatial_dim=self.spatial_dim,
            world_size=self.world_size,
            hidden_dim=self.hidden_dim,
        )
        
        # Add elements
        for center, radius in self._obstacles:
            game_world.add_obstacle(center, radius)
        for center, radius, cost in self._hazards:
            game_world.add_hazard(center, radius, cost)
        for position, attraction in self._goals:
            game_world.add_goal(position, attraction)
        
        # Lightweight trajectory network
        trajectory_net = BoundaryConditionedTrajectory(
            spatial_dim=self.spatial_dim,
            hidden_dim=32,  # Small for fast inference
            num_layers=2,   # Shallow for speed
        )
        
        # Loss
        loss_fn = GameAILoss(
            game_world=game_world,
            lambda_speed=lambda_speed,
            lambda_boundary=lambda_boundary,
            lambda_navigation=lambda_navigation,
        )
        
        # Fast inference model
        class GamePathfinder(nn.Module):
            def __init__(self, traj_net, world):
                super().__init__()
                self.trajectory_net = traj_net
                self.game_world = world
                self.last_inference_ms = 0.0
            
            def forward(self, t, start, end):
                return self.trajectory_net(t, start=start, end=end)
            
            @torch.no_grad()
            def find_path(
                self,
                start: Tuple[float, ...],
                goal: Tuple[float, ...],
                n_points: int = 50,
            ) -> torch.Tensor:
                """
                Find path from start to goal.
                
                Designed for real-time use - tracks inference time.
                """
                start_time = time.perf_counter()
                
                start_t = torch.tensor(start, dtype=torch.float32)
                goal_t = torch.tensor(goal, dtype=torch.float32)
                t = torch.linspace(0, 1, n_points).unsqueeze(-1)
                
                path = self.trajectory_net(t, start=start_t, end=goal_t)
                
                self.last_inference_ms = (time.perf_counter() - start_time) * 1000
                
                return path
            
            def benchmark(self, n_runs: int = 100) -> Dict[str, float]:
                """Benchmark inference speed."""
                import numpy as np
                
                # Random start/goal pairs
                times = []
                for _ in range(n_runs):
                    start = tuple(np.random.uniform(0, self.game_world.world_size, self.game_world.spatial_dim))
                    goal = tuple(np.random.uniform(0, self.game_world.world_size, self.game_world.spatial_dim))
                    
                    _ = self.find_path(start, goal)
                    times.append(self.last_inference_ms)
                
                return {
                    'mean_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times),
                    'meets_1ms_target': np.mean(times) < 1.0,
                }
        
        model = GamePathfinder(trajectory_net, game_world)
        
        return model, loss_fn
    
    def count_parameters(self, model: nn.Module) -> int:
        """Count model parameters for size estimation."""
        return sum(p.numel() for p in model.parameters())
    
    def estimate_size_kb(self, model: nn.Module) -> float:
        """Estimate model size in KB."""
        n_params = self.count_parameters(model)
        # float32 = 4 bytes
        return (n_params * 4) / 1024
