"""
Model Registry for Knowledge Distillation

Projects register their models here so the distillation pipeline can
instantiate teachers and students by name without hardcoded imports.

Usage in project's register_models.py:
    from tools.distillation.registry import registry
    from .models import MyTeacherModel, MyStudentModel
    
    registry.register(
        name="my-teacher",
        model_class=MyTeacherModel,
        default_config={"hidden_dim": 128},
        input_spec={"type": "tensor", "shape": [-1, 3]},
        output_spec={"type": "tensor", "shape": [-1, 3]},
        physics_loss_class=MyPhysicsLoss,  # Optional
    )

Usage in pipeline:
    model_class, config, input_spec, output_spec = registry.get("my-teacher")
    model = model_class(**config)
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Type, runtime_checkable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class InputSpec:
    """Specification for model input."""
    type: str  # "tensor", "graph", "point_cloud"
    shape: list[int] | None = None  # Shape with -1 for dynamic dims
    dtype: str = "float32"
    description: str = ""
    
    # For graph inputs
    node_features: int | None = None
    edge_features: int | None = None
    
    # Extra metadata
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputSpec:
    """Specification for model output."""
    type: str  # "tensor", "field", "graph"
    shape: list[int] | None = None
    dtype: str = "float32"
    description: str = ""
    
    # For multi-output models
    channels: dict[str, int] | None = None  # {"height": 1, "velocity": 2, ...}
    
    # Extra metadata
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PhysicsLoss(Protocol):
    """Protocol for domain-specific physics losses."""
    
    def __call__(
        self, 
        model_output: torch.Tensor, 
        inputs: dict[str, torch.Tensor],
        model: nn.Module | None = None
    ) -> torch.Tensor:
        """Compute physics-informed loss term."""
        ...


@runtime_checkable
class SamplingStrategy(Protocol):
    """Protocol for domain-specific input sampling."""
    
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Generate a batch of inputs for distillation training."""
        ...
    
    def sample_epoch(self, batch_size: int, num_samples: int) -> list[dict[str, torch.Tensor]]:
        """Generate batches for a full epoch."""
        ...


@dataclass
class ModelRegistration:
    """Complete registration information for a model."""
    name: str
    model_class: Type[nn.Module]
    default_config: dict[str, Any]
    input_spec: InputSpec
    output_spec: OutputSpec
    physics_loss_class: Type[PhysicsLoss] | None = None
    sampling_strategy_class: Type[SamplingStrategy] | None = None
    
    # Model metadata
    description: str = ""
    project: str = ""  # Source project name
    param_count: int | None = None
    framework: str = "pytorch"  # "pytorch", "jax", "tensorflow"


class ModelRegistry:
    """
    Central registry for models that can be used in distillation.
    
    This is a singleton - use the global `registry` instance.
    """
    
    def __init__(self):
        self._models: dict[str, ModelRegistration] = {}
        self._discovery_paths: list[Path] = []
    
    def register(
        self,
        name: str,
        model_class: Type[nn.Module],
        default_config: dict[str, Any] | None = None,
        input_spec: InputSpec | dict[str, Any] | None = None,
        output_spec: OutputSpec | dict[str, Any] | None = None,
        physics_loss_class: Type[PhysicsLoss] | None = None,
        sampling_strategy_class: Type[SamplingStrategy] | None = None,
        description: str = "",
        project: str = "",
        param_count: int | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a model for use in the distillation pipeline.
        
        Args:
            name: Unique identifier for this model (e.g., "hgnn-nif-cloth")
            model_class: The PyTorch model class
            default_config: Default kwargs for model instantiation
            input_spec: Specification of model inputs
            output_spec: Specification of model outputs
            physics_loss_class: Optional class that implements PhysicsLoss protocol
            sampling_strategy_class: Optional class that implements SamplingStrategy protocol
            description: Human-readable description
            project: Source project (e.g., "09-HGNN-NIF-Cloth")
            param_count: Approximate parameter count
            overwrite: If True, allow overwriting existing registration
        
        Raises:
            ValueError: If name already registered and overwrite=False
        """
        if name in self._models and not overwrite:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Use overwrite=True to replace it."
            )
        
        # Convert dict specs to dataclasses
        if isinstance(input_spec, dict):
            input_spec = InputSpec(**input_spec)
        elif input_spec is None:
            input_spec = InputSpec(type="tensor")
        
        if isinstance(output_spec, dict):
            output_spec = OutputSpec(**output_spec)
        elif output_spec is None:
            output_spec = OutputSpec(type="tensor")
        
        registration = ModelRegistration(
            name=name,
            model_class=model_class,
            default_config=default_config or {},
            input_spec=input_spec,
            output_spec=output_spec,
            physics_loss_class=physics_loss_class,
            sampling_strategy_class=sampling_strategy_class,
            description=description,
            project=project,
            param_count=param_count,
        )
        
        self._models[name] = registration
        logger.info(f"Registered model: {name} from {project or 'unknown project'}")
    
    def get(self, name: str) -> ModelRegistration:
        """
        Retrieve a model registration by name.
        
        Args:
            name: The registered model name
            
        Returns:
            ModelRegistration with all model information
            
        Raises:
            KeyError: If model not found
        """
        if name not in self._models:
            available = ", ".join(self._models.keys()) or "(none)"
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}. "
                f"Make sure the project's register_models.py has been imported."
            )
        return self._models[name]
    
    def get_model_class(self, name: str) -> Type[nn.Module]:
        """Get just the model class."""
        return self.get(name).model_class
    
    def get_default_config(self, name: str) -> dict[str, Any]:
        """Get the default configuration for a model."""
        return self.get(name).default_config.copy()
    
    def list(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def list_detailed(self) -> list[ModelRegistration]:
        """List all registrations with full details."""
        return list(self._models.values())
    
    def create_model(
        self, 
        name: str, 
        config_overrides: dict[str, Any] | None = None
    ) -> nn.Module:
        """
        Instantiate a registered model.
        
        Args:
            name: Registered model name
            config_overrides: Override default config values
            
        Returns:
            Instantiated model
        """
        reg = self.get(name)
        config = reg.default_config.copy()
        if config_overrides:
            config.update(config_overrides)
        
        return reg.model_class(**config)
    
    def load_model(
        self,
        name: str,
        checkpoint_path: str,
        config_overrides: dict[str, Any] | None = None,
        device: str = "cpu",
        strict: bool = True,
    ) -> nn.Module:
        """
        Instantiate and load weights for a registered model.
        
        Args:
            name: Registered model name
            checkpoint_path: Path to saved weights
            config_overrides: Override default config values
            device: Device to load model on
            strict: Whether to require exact state dict match
            
        Returns:
            Model with loaded weights
        """
        model = self.create_model(name, config_overrides)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the dict is the state dict itself
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        
        return model
    
    def clear(self) -> None:
        """Clear all registrations. Mainly for testing."""
        self._models.clear()
    
    def add_discovery_path(self, path: str | Path) -> None:
        """Add a path to search for register_models.py files."""
        self._discovery_paths.append(Path(path))
    
    def discover_models(self, workspace_root: str | Path) -> int:
        """
        Auto-discover models by importing register_models.py from project directories.
        
        Args:
            workspace_root: Root of the PINN-Experiments workspace
            
        Returns:
            Number of register_models.py files found and imported
        """
        workspace_root = Path(workspace_root)
        projects_dir = workspace_root / "projects"
        
        if not projects_dir.exists():
            logger.warning(f"Projects directory not found: {projects_dir}")
            return 0
        
        count = 0
        
        # Look for project directories
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            
            # Look for register_models.py in various locations
            register_paths = [
                project_dir / "register_models.py",
                project_dir / "src" / "register_models.py",
            ]
            
            # Also check for nested project structure (e.g., pinn-lite-foil/pinn-lite-foil/)
            nested_dirs = [d for d in project_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            for nested in nested_dirs:
                register_paths.append(nested / "register_models.py")
                register_paths.append(nested / "src" / "register_models.py")
            
            for register_path in register_paths:
                if register_path.exists():
                    try:
                        self._import_register_file(register_path)
                        count += 1
                        logger.info(f"Discovered models from: {register_path}")
                    except Exception as e:
                        logger.warning(f"Failed to import {register_path}: {e}")
        
        return count
    
    def _import_register_file(self, path: Path) -> None:
        """Import a register_models.py file."""
        # Add parent to sys.path temporarily
        parent = str(path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        
        try:
            # Create a unique module name to avoid conflicts
            module_name = f"_register_models_{path.parent.name}_{id(path)}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        finally:
            if parent in sys.path:
                sys.path.remove(parent)


# Global singleton registry
registry = ModelRegistry()


# Convenience decorator for registration
def register_model(
    name: str,
    default_config: dict[str, Any] | None = None,
    input_spec: InputSpec | dict[str, Any] | None = None,
    output_spec: OutputSpec | dict[str, Any] | None = None,
    **kwargs
) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """
    Decorator to register a model class.
    
    Usage:
        @register_model("my-model", default_config={"hidden_dim": 64})
        class MyModel(nn.Module):
            ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        registry.register(
            name=name,
            model_class=cls,
            default_config=default_config,
            input_spec=input_spec,
            output_spec=output_spec,
            **kwargs
        )
        return cls
    return decorator
