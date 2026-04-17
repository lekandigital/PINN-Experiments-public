"""
Domain-Specific Adapters for Geodesic Trajectory Prediction.

Each adapter specializes the core framework for a specific domain:
- Biology: Microbe chemotaxis on nutrient fields
- Robotics: Path planning on terrain/obstacle maps
- Migration: Animal movement on environmental gradients
- Finance: Agent trajectories in profit/risk space
- Game AI: NPC navigation on cost landscapes
"""

from .biology import NutrientField, ChemotaxisLoss, BiologyAdapter
from .robotics import TerrainField, RobotPathLoss, RoboticsAdapter
from .migration import EnvironmentalField, MigrationLoss, MigrationAdapter
from .finance import ProfitLandscape, FinanceLoss, FinanceAdapter
from .game_ai import GameWorldField, GameAILoss, GameAIAdapter

__all__ = [
    # Biology
    "NutrientField",
    "ChemotaxisLoss",
    "BiologyAdapter",
    # Robotics  
    "TerrainField",
    "RobotPathLoss",
    "RoboticsAdapter",
    # Migration
    "EnvironmentalField",
    "MigrationLoss",
    "MigrationAdapter",
    # Finance
    "ProfitLandscape",
    "FinanceLoss",
    "FinanceAdapter",
    # Game AI
    "GameWorldField",
    "GameAILoss",
    "GameAIAdapter",
]
