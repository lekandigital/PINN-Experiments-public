"""
Robotics Domain Adapter: Path Planning on Terrain.

Models robot trajectories as geodesics on terrain/traversability maps.
Suitable for:
- Mobile robot navigation
- Autonomous vehicle path planning
- Drone trajectory optimization

Key Physics:
- Robots move to minimize energy expenditure
- Terrain elevation = traversal cost
- Obstacles create high-potential barriers
- Kinematic constraints (curvature, speed limits)

Potential Convention:
- High elevation / obstacles → high potential → avoid
- Low potential = easy terrain = preferred path
"""

from typing import Optional, Tuple, Dict, Union
import torch
import torch.nn as nn
import numpy as np

from ..core.potential_field import PotentialFieldBase, LearnedPotentialField
from ..core.geodesic_loss import GeodesicLoss, CurvaturePenalty


class TerrainField(PotentialFieldBase):
    """
    Terrain/traversability field for robot path planning.
    
    Can be initialized from:
    - Learned from trajectory data
    - Interpolated from elevation grid (DEM)
    - Analytic cost functions
    
    Components that can be combined:
    - Base elevation (from terrain data)
    - Obstacle costs (very high potential)
    - Goal attraction (negative potential at goal)
    """
    
    def __init__(
        self,
        spatial_dim: int = 2,
        mode: str = "learned",
        hidden_dim: int = 64,
        terrain_grid: Optional[torch.Tensor] = None,
        grid_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ):
        """
        Initialize terrain field.
        
        Args:
            spatial_dim: Spatial dimension (2 for ground robots, 3 for drones)
            mode: "learned" (from data) or "grid" (interpolated from DEM)
            hidden_dim: Hidden dimension for learned mode
            terrain_grid: [H, W] elevation grid for grid mode
            grid_bounds: ((x_min, x_max), (y_min, y_max)) for grid mode
        """
        super().__init__(spatial_dim=spatial_dim, time_dependent=False)
        
        self.mode = mode
        self.hidden_dim = hidden_dim
        
        if mode == "learned":
            self.net = nn.Sequential(
                nn.Linear(spatial_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self._init_weights()
        
        elif mode == "grid":
            if terrain_grid is None:
                raise ValueError("terrain_grid required for grid mode")
            self.register_buffer('terrain_grid', terrain_grid.float())
            
            if grid_bounds is None:
                grid_bounds = ((0, 1), (0, 1))
            self.x_min, self.x_max = grid_bounds[0]
            self.y_min, self.y_max = grid_bounds[1]
        
        # Optional obstacle and goal components
        self.obstacles = nn.ParameterList()
        self.obstacle_radii = []
        self.obstacle_costs = []
        
        self.goals = nn.ParameterList()
        self.goal_attractions = []
    
    def _init_weights(self):
        """Initialize learned network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def add_obstacle(
        self,
        center: Union[torch.Tensor, Tuple[float, ...]],
        radius: float,
        cost: float = 100.0,
    ):
        """
        Add circular/spherical obstacle.
        
        Args:
            center: Obstacle center position
            radius: Obstacle radius
            cost: Potential cost at obstacle center
        """
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32)
        
        self.obstacles.append(nn.Parameter(center, requires_grad=False))
        self.obstacle_radii.append(radius)
        self.obstacle_costs.append(cost)
    
    def add_goal(
        self,
        position: Union[torch.Tensor, Tuple[float, ...]],
        attraction: float = 10.0,
    ):
        """
        Add goal point with attractive potential.
        
        Args:
            position: Goal position
            attraction: Attraction strength (negative potential)
        """
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=torch.float32)
        
        self.goals.append(nn.Parameter(position, requires_grad=False))
        self.goal_attractions.append(attraction)
    
    def _interpolate_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of terrain grid."""
        # Normalize to [0, 1] range
        x_norm = (x[:, 0] - self.x_min) / (self.x_max - self.x_min + 1e-8)
        y_norm = (x[:, 1] - self.y_min) / (self.y_max - self.y_min + 1e-8)
        
        # Convert to grid indices
        H, W = self.terrain_grid.shape
        x_idx = x_norm * (W - 1)
        y_idx = y_norm * (H - 1)
        
        # Clamp to valid range
        x_idx = x_idx.clamp(0, W - 1 - 1e-4)
        y_idx = y_idx.clamp(0, H - 1 - 1e-4)
        
        # Bilinear interpolation
        x0 = x_idx.long()
        y0 = y_idx.long()
        x1 = (x0 + 1).clamp(max=W - 1)
        y1 = (y0 + 1).clamp(max=H - 1)
        
        # Interpolation weights
        wx = x_idx - x0.float()
        wy = y_idx - y0.float()
        
        # Get corner values
        v00 = self.terrain_grid[y0, x0]
        v01 = self.terrain_grid[y0, x1]
        v10 = self.terrain_grid[y1, x0]
        v11 = self.terrain_grid[y1, x1]
        
        # Bilinear interpolation
        elevation = (
            v00 * (1 - wx) * (1 - wy) +
            v01 * wx * (1 - wy) +
            v10 * (1 - wx) * wy +
            v11 * wx * wy
        )
        
        return elevation.unsqueeze(-1)
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute terrain potential at positions.
        
        Combines:
        - Base terrain (learned or interpolated)
        - Obstacle costs
        - Goal attractions
        """
        # Base terrain
        if self.mode == "learned":
            potential = self.net(x)
        else:
            potential = self._interpolate_grid(x)
        
        # Add obstacles
        for center, radius, cost in zip(self.obstacles, self.obstacle_radii, self.obstacle_costs):
            dist = torch.norm(x - center.to(x.device), dim=-1, keepdim=True)
            # Gaussian barrier: high cost near obstacle, falls off with distance
            obstacle_potential = cost * torch.exp(-0.5 * (dist / radius) ** 2)
            # Extra penalty inside obstacle
            inside_mask = (dist < radius).float()
            obstacle_potential = obstacle_potential + inside_mask * cost * 10
            potential = potential + obstacle_potential
        
        # Add goal attractions
        for position, attraction in zip(self.goals, self.goal_attractions):
            dist = torch.norm(x - position.to(x.device), dim=-1, keepdim=True)
            # Negative potential (attractive) near goal
            goal_potential = -attraction * torch.exp(-0.5 * (dist / 0.5) ** 2)
            potential = potential + goal_potential
        
        return potential


class RobotPathLoss(GeodesicLoss):
    """
    Robot path planning loss function.
    
    Extends GeodesicLoss with:
    - Curvature constraints (turning radius)
    - Kinematic feasibility
    - Optional: velocity limits, acceleration limits
    """
    
    def __init__(
        self,
        terrain_field: TerrainField,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_terrain: float = 1.0,
        lambda_curvature: float = 1.0,
        lambda_data: float = 0.0,
        target_speed: float = 1.0,
        max_curvature: float = 1.0,
    ):
        """
        Initialize robot path loss.
        
        Args:
            terrain_field: Terrain potential field
            lambda_speed: Weight for constant speed
            lambda_boundary: Weight for start/end constraints
            lambda_terrain: Weight for terrain-following (gradient descent)
            lambda_curvature: Weight for curvature penalty
            lambda_data: Weight for data fitting (usually 0 for planning)
            target_speed: Target robot speed
            max_curvature: Maximum allowed curvature (1/turning_radius)
        """
        super().__init__(
            potential_field=terrain_field,
            lambda_speed=lambda_speed,
            lambda_boundary=lambda_boundary,
            lambda_gradient=lambda_terrain,
            lambda_curvature=lambda_curvature,
            lambda_data=lambda_data,
            target_speed=target_speed,
            max_curvature=max_curvature,
            gradient_direction="descent",  # Avoid high-cost terrain
        )
        
        self.terrain_field = terrain_field


class RoboticsAdapter:
    """
    High-level adapter for robotics path planning.
    
    Provides:
    - Pre-configured models for robot navigation
    - Terrain/obstacle setup utilities
    - Path optimization interface
    
    Example:
        >>> adapter = RoboticsAdapter()
        >>> adapter.add_obstacle((0.5, 0.5), radius=0.2)
        >>> adapter.set_goal((0.9, 0.9))
        >>> model, loss_fn = adapter.create_model()
        >>> path = adapter.plan_path(model, start=(0.1, 0.1), end=(0.9, 0.9))
    """
    
    def __init__(
        self,
        spatial_dim: int = 2,
        hidden_dim: int = 64,
        terrain_mode: str = "learned",
    ):
        """
        Initialize robotics adapter.
        
        Args:
            spatial_dim: Spatial dimension (2 for ground, 3 for aerial)
            hidden_dim: Network hidden dimension
            terrain_mode: "learned" or "grid"
        """
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.terrain_mode = terrain_mode
        
        # Store obstacles and goals for terrain setup
        self._obstacles = []
        self._goals = []
    
    def add_obstacle(
        self,
        center: Tuple[float, ...],
        radius: float,
        cost: float = 100.0,
    ):
        """Add obstacle to environment."""
        self._obstacles.append((center, radius, cost))
    
    def set_goal(self, position: Tuple[float, ...], attraction: float = 10.0):
        """Set goal position."""
        self._goals.append((position, attraction))
    
    def create_model(
        self,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_terrain: float = 1.0,
        lambda_curvature: float = 1.0,
        target_speed: float = 1.0,
        max_curvature: float = 1.0,
        terrain_grid: Optional[torch.Tensor] = None,
    ) -> Tuple[nn.Module, RobotPathLoss]:
        """
        Create path planning model and loss.
        
        Returns:
            Tuple of (model, loss_function)
        """
        from ..core.trajectory_pinn import BoundaryConditionedTrajectory
        
        # Create terrain field
        terrain_field = TerrainField(
            spatial_dim=self.spatial_dim,
            mode=self.terrain_mode,
            hidden_dim=self.hidden_dim,
            terrain_grid=terrain_grid,
        )
        
        # Add obstacles and goals
        for center, radius, cost in self._obstacles:
            terrain_field.add_obstacle(center, radius, cost)
        for position, attraction in self._goals:
            terrain_field.add_goal(position, attraction)
        
        # Create trajectory network with hard boundary conditions
        trajectory_net = BoundaryConditionedTrajectory(
            spatial_dim=self.spatial_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3,
        )
        
        # Loss function
        loss_fn = RobotPathLoss(
            terrain_field=terrain_field,
            lambda_speed=lambda_speed,
            lambda_boundary=lambda_boundary,
            lambda_terrain=lambda_terrain,
            lambda_curvature=lambda_curvature,
            target_speed=target_speed,
            max_curvature=max_curvature,
        )
        
        # Combined model
        class RobotPathModel(nn.Module):
            def __init__(self, traj_net, terr_field):
                super().__init__()
                self.trajectory_net = traj_net
                self.terrain_field = terr_field
            
            def forward(self, t, start, end):
                return self.trajectory_net(t, start=start, end=end)
            
            def plan_path(self, start, end, n_steps=100):
                """Plan path from start to end."""
                if not isinstance(start, torch.Tensor):
                    start = torch.tensor(start, dtype=torch.float32)
                if not isinstance(end, torch.Tensor):
                    end = torch.tensor(end, dtype=torch.float32)
                
                t = torch.linspace(0, 1, n_steps).unsqueeze(-1)
                return self.trajectory_net(t, start=start, end=end)
        
        model = RobotPathModel(trajectory_net, terrain_field)
        
        return model, loss_fn
    
    def create_terrain_with_obstacles(
        self,
        n_obstacles: int = 5,
        bounds: Tuple[float, float] = (0, 1),
        seed: Optional[int] = None,
    ) -> TerrainField:
        """
        Create terrain with random obstacles.
        
        Args:
            n_obstacles: Number of random obstacles
            bounds: Terrain bounds
            seed: Random seed
            
        Returns:
            TerrainField with obstacles
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        terrain = TerrainField(
            spatial_dim=self.spatial_dim,
            mode="learned",
            hidden_dim=self.hidden_dim,
        )
        
        # Add random obstacles
        for _ in range(n_obstacles):
            center = tuple(
                np.random.uniform(bounds[0] + 0.15, bounds[1] - 0.15)
                for _ in range(self.spatial_dim)
            )
            radius = np.random.uniform(0.05, 0.15)
            terrain.add_obstacle(center, radius, cost=100.0)
        
        return terrain
