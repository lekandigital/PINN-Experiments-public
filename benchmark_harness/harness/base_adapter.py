"""
Abstract base class for project adapters.

Each project must implement this interface to be benchmarkable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .core import BenchmarkResult, BenchmarkConfig, ModelInfo, TrainingInfo


@dataclass
class ProjectInfo:
    """Basic project identification."""
    project_id: str           # e.g., "P01", "P09-anim"
    project_name: str         # e.g., "GeoPINN-Manifold"
    framework: str            # "pytorch", "jax", "onnx", "tensorflow"
    project_path: Path        # Path to project directory
    description: str = ""     # Brief description
    domain: str = ""          # Physics domain


class ProjectAdapter(ABC):
    """
    Abstract base class for project-specific benchmark adapters.
    
    Each of the 17 projects needs an adapter that implements these methods
    to enable uniform benchmarking.
    
    The adapter is responsible for:
    1. Loading the model (from checkpoint or config)
    2. Preparing representative test inputs
    3. Running inference
    4. Computing domain-specific accuracy metrics
    5. Providing training metadata
    
    Example usage:
        adapter = P09Adapter(project_path="/path/to/project")
        model = adapter.load_model(device="cuda")
        input_data = adapter.prepare_test_input(device="cuda")
        output = adapter.run_inference(model, input_data)
        metrics = adapter.compute_domain_metrics(output, adapter.get_reference_data())
    """
    
    def __init__(self, project_path: Optional[Path] = None):
        """
        Initialize the adapter.
        
        Args:
            project_path: Path to the project directory. If None, uses default.
        """
        self.project_path = Path(project_path) if project_path else self._default_project_path()
    
    @abstractmethod
    def _default_project_path(self) -> Path:
        """Return the default path to the project directory."""
        pass
    
    @abstractmethod
    def get_project_info(self) -> ProjectInfo:
        """
        Return project identification information.
        
        Returns:
            ProjectInfo with id, name, framework, and path
        """
        pass
    
    @abstractmethod
    def load_model(self, device: str = "cuda") -> Any:
        """
        Load the trained model.
        
        Args:
            device: Target device ("cuda", "cpu", or specific like "cuda:0")
        
        Returns:
            The loaded model object (framework-specific type)
        
        Raises:
            FileNotFoundError: If checkpoint is not found
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def prepare_test_input(
        self, 
        device: str = "cuda",
        batch_size: int = 1,
    ) -> Any:
        """
        Prepare a representative test input batch.
        
        This should return input data that exercises the model's
        typical use case. For models without real test data,
        generate synthetic data matching the expected format.
        
        Args:
            device: Target device
            batch_size: Number of samples in batch
        
        Returns:
            Input data in the format expected by run_inference
            (tensor, tuple of tensors, dict, etc.)
        """
        pass
    
    @abstractmethod
    def run_inference(self, model: Any, input_data: Any) -> Any:
        """
        Run a single forward pass through the model.
        
        This method should:
        1. Put model in eval mode (if applicable)
        2. Disable gradient computation (if applicable)
        3. Run the forward pass
        4. Return raw output
        
        Args:
            model: The loaded model
            input_data: Input from prepare_test_input
        
        Returns:
            Raw model output (tensor, dict, tuple, etc.)
        """
        pass
    
    @abstractmethod
    def compute_domain_metrics(
        self, 
        predictions: Any, 
        references: Any,
    ) -> dict[str, Any]:
        """
        Compute domain-specific accuracy metrics.
        
        This goes beyond standard RMSE/L2 error to capture
        physics-meaningful quantities like:
        - Eikonal violation for SDF models
        - Edge preservation for cloth meshes
        - Divergence residual for EM fields
        - PDE residuals for physics-informed models
        
        Args:
            predictions: Model output from run_inference
            references: Ground truth from get_reference_data
        
        Returns:
            Dict of metric name -> value
        """
        pass
    
    @abstractmethod
    def get_reference_data(self) -> Any:
        """
        Load reference/ground truth data for accuracy computation.
        
        Returns:
            Reference data matching the format of model output
            
        Raises:
            FileNotFoundError: If reference data is not available
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model: Any) -> ModelInfo:
        """
        Get model size and parameter information.
        
        Args:
            model: The loaded model
        
        Returns:
            ModelInfo with parameter counts and sizes
        """
        pass
    
    @abstractmethod
    def get_training_info(self) -> TrainingInfo:
        """
        Extract training metadata from logs or documentation.
        
        Returns:
            TrainingInfo with training time, epochs, hardware
        """
        pass
    
    @abstractmethod
    def get_test_dataset_description(self) -> str:
        """
        Describe the test data used for benchmarking.
        
        Returns:
            Human-readable description including:
            - Synthetic vs real data
            - Dataset size
            - Data source/generation method
        """
        pass
    
    def has_checkpoint(self) -> bool:
        """
        Check if a trained checkpoint exists.
        
        Override in subclass if checkpoint detection is non-trivial.
        
        Returns:
            True if checkpoint exists and is loadable
        """
        checkpoint_patterns = [
            "*.pt", "*.pth", "*.ckpt", "*.pkl",
            "checkpoints/*.pt", "checkpoints/*.pth",
            "models/*.pt", "models/*.onnx",
        ]
        
        for pattern in checkpoint_patterns:
            matches = list(self.project_path.glob(pattern))
            if matches:
                return True
        return False
    
    def get_checkpoint_path(self) -> Optional[Path]:
        """
        Get the path to the best available checkpoint.
        
        Override in subclass for project-specific logic.
        
        Returns:
            Path to checkpoint file, or None if not found
        """
        # Priority order for checkpoint discovery
        search_patterns = [
            "checkpoints/best*.pt",
            "checkpoints/final*.pt",
            "checkpoints/*.pt",
            "models/best*.pt",
            "models/*.pt",
            "*.pt",
            "*.pth",
            "*.ckpt",
        ]
        
        for pattern in search_patterns:
            matches = sorted(self.project_path.glob(pattern))
            if matches:
                return matches[-1]  # Return most recent/highest
        
        return None
    
    def supports_batch_inference(self) -> bool:
        """
        Whether the model supports batched inference.
        
        Override to return False for models that only support batch_size=1.
        
        Returns:
            True if batched inference is supported
        """
        return True
    
    def get_recommended_batch_sizes(self) -> list[int]:
        """
        Get recommended batch sizes for throughput testing.
        
        Override for models with specific batch size requirements.
        
        Returns:
            List of batch sizes to test
        """
        return [1, 8, 32] if self.supports_batch_inference() else [1]
    
    def cleanup(self, model: Any) -> None:
        """
        Clean up resources after benchmarking.
        
        Override if the model needs explicit cleanup (e.g., ONNX sessions).
        
        Args:
            model: The model to clean up
        """
        pass
    
    def is_synthetic_fallback(self) -> bool:
        """
        Whether the adapter is using synthetic test data.
        
        Returns:
            True if no real test data is available
        """
        try:
            self.get_reference_data()
            return False
        except (FileNotFoundError, NotImplementedError):
            return True


class PyTorchAdapter(ProjectAdapter):
    """
    Base class for PyTorch-based project adapters.
    
    Provides common PyTorch-specific functionality.
    """
    
    def load_model(self, device: str = "cuda") -> Any:
        """Load PyTorch model with standard pattern."""
        import torch
        
        model = self._create_model()
        
        checkpoint_path = self.get_checkpoint_path()
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        return model
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the model instance (without loading weights)."""
        pass
    
    def run_inference(self, model: Any, input_data: Any) -> Any:
        """Run PyTorch inference with no_grad."""
        import torch
        
        model.eval()
        with torch.no_grad():
            if isinstance(input_data, dict):
                return model(**input_data)
            elif isinstance(input_data, (tuple, list)):
                return model(*input_data)
            else:
                return model(input_data)
    
    def get_model_info(self, model: Any) -> ModelInfo:
        """Get model info for PyTorch model."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_bytes + buffer_bytes) / (1024**2)
        
        return ModelInfo(
            parameter_count=trainable,
            parameter_count_non_trainable=non_trainable,
            model_size_mb=size_mb,
        )


class JAXAdapter(ProjectAdapter):
    """
    Base class for JAX/Haiku-based project adapters.
    
    Provides common JAX-specific functionality.
    """
    
    def __init__(self, project_path: Optional[Path] = None):
        super().__init__(project_path)
        self._params = None
        self._state = None
        self._apply_fn = None
    
    @abstractmethod
    def _create_model(self) -> tuple[Any, Any, Any]:
        """
        Create the Haiku model.
        
        Returns:
            Tuple of (apply_fn, params, state)
        """
        pass
    
    def load_model(self, device: str = "cuda") -> Any:
        """Load JAX/Haiku model."""
        apply_fn, params, state = self._create_model()
        self._apply_fn = apply_fn
        self._params = params
        self._state = state
        
        # Return a callable wrapper
        return (apply_fn, params, state)
    
    def run_inference(self, model: Any, input_data: Any) -> Any:
        """Run JAX inference."""
        import jax
        
        apply_fn, params, state = model
        rng = jax.random.PRNGKey(42)
        
        if isinstance(input_data, dict):
            output, _ = apply_fn(params, state, rng, **input_data, is_training=False)
        else:
            output, _ = apply_fn(params, state, rng, input_data, is_training=False)
        
        return output
    
    def get_model_info(self, model: Any) -> ModelInfo:
        """Get model info for JAX model."""
        import jax
        
        _, params, _ = model
        
        leaves = jax.tree_util.tree_leaves(params)
        total_params = sum(x.size for x in leaves if hasattr(x, 'size'))
        total_bytes = sum(x.size * x.dtype.itemsize for x in leaves if hasattr(x, 'size'))
        
        return ModelInfo(
            parameter_count=total_params,
            parameter_count_non_trainable=0,
            model_size_mb=total_bytes / (1024**2),
        )


class ONNXAdapter(ProjectAdapter):
    """
    Base class for ONNX Runtime-based project adapters.
    """
    
    def __init__(self, project_path: Optional[Path] = None):
        super().__init__(project_path)
        self._session = None
    
    @abstractmethod
    def get_onnx_model_path(self) -> Path:
        """Return path to the ONNX model file."""
        pass
    
    def load_model(self, device: str = "cuda") -> Any:
        """Load ONNX model."""
        import onnxruntime as ort
        
        model_path = self.get_onnx_model_path()
        
        if device.startswith("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        return self._session
    
    def run_inference(self, model: Any, input_data: Any) -> Any:
        """Run ONNX inference."""
        session = model
        
        if isinstance(input_data, dict):
            return session.run(None, input_data)
        else:
            # Assume single input
            input_name = session.get_inputs()[0].name
            return session.run(None, {input_name: input_data})
    
    def get_model_info(self, model: Any) -> ModelInfo:
        """Get model info for ONNX model."""
        import os
        
        model_path = self.get_onnx_model_path()
        file_size_mb = os.path.getsize(model_path) / (1024**2)
        
        # Try to count parameters
        total_params = 0
        try:
            import onnx
            onnx_model = onnx.load(str(model_path))
            import numpy as np
            for initializer in onnx_model.graph.initializer:
                total_params += int(np.prod(initializer.dims))
        except Exception:
            pass
        
        return ModelInfo(
            parameter_count=total_params,
            parameter_count_non_trainable=0,
            model_size_mb=file_size_mb,
            onnx_size_mb=file_size_mb,
        )
    
    def cleanup(self, model: Any) -> None:
        """Clean up ONNX session."""
        self._session = None
