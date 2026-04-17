"""
Demos for Geodesic Trajectory Library.

Available demos:
- robot_planning: Path planning on terrain with obstacles
- game_ai: NPC pathfinding for real-time games (<1ms inference)
- chemotaxis: Original biology demo (microbe movement)
"""

from . import robot_planning
from . import game_ai
# from . import chemotaxis  # Optional: refactored biology demo

__all__ = ['robot_planning', 'game_ai']
