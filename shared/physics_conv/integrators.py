"""
Time Integration Schemes for Physics Simulation.

These integrators convert forces/accelerations to position and velocity updates.
All integrators are stateless and handle batched inputs.

Classes:
    ExplicitEuler: Simple first-order forward Euler
    SemiImplicitEuler: Symplectic Euler with better energy conservation
    VelocityVerlet: Second-order symplectic integrator (best for cloth/springs)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import Tensor


class Integrator(ABC):
    """Abstract base class for time integrators."""
    
    @abstractmethod
    def step(
        self,
        pos: Tensor,
        vel: Tensor,
        acc: Tensor,
        dt: float,
        mass: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Perform one integration step.
        
        Args:
            pos: Positions [batch, num_nodes, dim] or [num_nodes, dim]
            vel: Velocities [batch, num_nodes, dim] or [num_nodes, dim]
            acc: Accelerations [batch, num_nodes, dim] or [num_nodes, dim]
            dt: Time step size
            mass: Optional per-node masses [batch, num_nodes, 1] or [num_nodes, 1]
        
        Returns:
            new_pos: Updated positions
            new_vel: Updated velocities
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return integrator name."""
        raise NotImplementedError
    
    @property
    def order(self) -> int:
        """Return order of accuracy."""
        return 1


class ExplicitEuler(Integrator):
    """Explicit (Forward) Euler integration.
    
    x_new = x + dt * v
    v_new = v + dt * a
    
    Simple but can be unstable for stiff systems.
    Energy tends to increase over time (unstable for oscillators).
    
    Use for: Quick prototyping, non-stiff systems
    Avoid for: Cloth simulation, spring systems (use Verlet instead)
    """
    
    def __init__(self, damping: float = 1.0):
        """
        Args:
            damping: Velocity damping factor (1.0 = no damping, 0.99 = light damping)
        """
        self.damping = damping
    
    def step(
        self,
        pos: Tensor,
        vel: Tensor,
        acc: Tensor,
        dt: float,
        mass: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward Euler step."""
        # v' = v + dt * a
        new_vel = vel + dt * acc
        
        # Apply damping
        new_vel = new_vel * self.damping
        
        # x' = x + dt * v
        new_pos = pos + dt * vel
        
        return new_pos, new_vel
    
    @property
    def name(self) -> str:
        return "ExplicitEuler"


class SemiImplicitEuler(Integrator):
    """Semi-Implicit (Symplectic) Euler integration.
    
    v_new = v + dt * a
    x_new = x + dt * v_new  (uses NEW velocity)
    
    Symplectic: conserves energy on average (bounded oscillation).
    Better than explicit Euler for oscillatory systems.
    
    Use for: Spring systems, cloth simulation, general physics
    """
    
    def __init__(self, damping: float = 1.0):
        """
        Args:
            damping: Velocity damping factor applied after velocity update
        """
        self.damping = damping
    
    def step(
        self,
        pos: Tensor,
        vel: Tensor,
        acc: Tensor,
        dt: float,
        mass: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Semi-implicit Euler step."""
        # v' = v + dt * a
        new_vel = vel + dt * acc
        
        # Apply damping
        new_vel = new_vel * self.damping
        
        # x' = x + dt * v' (uses NEW velocity - key difference from explicit)
        new_pos = pos + dt * new_vel
        
        return new_pos, new_vel
    
    @property
    def name(self) -> str:
        return "SemiImplicitEuler"


class VelocityVerlet(Integrator):
    """Velocity Verlet integration.
    
    x_new = x + v*dt + 0.5*a*dt²
    a_new = F(x_new) / m  (caller must provide this)
    v_new = v + 0.5*(a + a_new)*dt
    
    Second-order accurate, symplectic, time-reversible.
    Excellent energy conservation - the gold standard for molecular dynamics.
    
    Note: Full Verlet requires computing acceleration at the new position.
    This implementation provides a single-step version where you pass
    the new acceleration or use the same acceleration (leapfrog style).
    
    Use for: Cloth simulation, molecular dynamics, any energy-conserving system
    """
    
    def __init__(self, damping: float = 1.0):
        """
        Args:
            damping: Velocity damping factor
        """
        self.damping = damping
    
    def step(
        self,
        pos: Tensor,
        vel: Tensor,
        acc: Tensor,
        dt: float,
        mass: Optional[Tensor] = None,
        acc_new: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Velocity Verlet step.
        
        Args:
            pos: Current positions
            vel: Current velocities
            acc: Current accelerations (F/m at current position)
            dt: Time step
            mass: Not used (acceleration already includes mass)
            acc_new: Optional acceleration at new position. If not provided,
                    uses current acceleration (equivalent to leapfrog).
        
        Returns:
            new_pos: Updated positions
            new_vel: Updated velocities
        """
        # x' = x + v*dt + 0.5*a*dt²
        new_pos = pos + vel * dt + 0.5 * acc * (dt ** 2)
        
        # If no new acceleration provided, use current (leapfrog approximation)
        if acc_new is None:
            acc_new = acc
        
        # v' = v + 0.5*(a + a_new)*dt
        new_vel = vel + 0.5 * (acc + acc_new) * dt
        
        # Apply damping
        new_vel = new_vel * self.damping
        
        return new_pos, new_vel
    
    def half_step_velocity(
        self,
        vel: Tensor,
        acc: Tensor,
        dt: float,
    ) -> Tensor:
        """Compute half-step velocity update.
        
        Used for proper Verlet integration where you need:
        1. v_half = v + 0.5*a*dt
        2. x_new = x + v_half*dt
        3. a_new = F(x_new)/m
        4. v_new = v_half + 0.5*a_new*dt
        
        Args:
            vel: Current velocity
            acc: Current acceleration
            dt: Time step
        
        Returns:
            v_half: Half-step velocity
        """
        return vel + 0.5 * acc * dt
    
    @property
    def name(self) -> str:
        return "VelocityVerlet"
    
    @property
    def order(self) -> int:
        return 2


class IntegratorFactory:
    """Factory for creating integrators by name."""
    
    _integrators = {
        "euler": ExplicitEuler,
        "explicit_euler": ExplicitEuler,
        "semi_implicit": SemiImplicitEuler,
        "semi_implicit_euler": SemiImplicitEuler,
        "symplectic": SemiImplicitEuler,
        "verlet": VelocityVerlet,
        "velocity_verlet": VelocityVerlet,
    }
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Integrator:
        """Create an integrator by name.
        
        Args:
            name: Integrator name (case-insensitive)
            **kwargs: Arguments passed to integrator constructor
        
        Returns:
            integrator: Integrator instance
        
        Raises:
            ValueError: If integrator name is unknown
        """
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        
        if name_lower not in cls._integrators:
            available = ", ".join(cls._integrators.keys())
            raise ValueError(
                f"Unknown integrator: {name}. Available: {available}"
            )
        
        return cls._integrators[name_lower](**kwargs)
    
    @classmethod
    def available(cls) -> list:
        """Return list of available integrator names."""
        return list(cls._integrators.keys())
