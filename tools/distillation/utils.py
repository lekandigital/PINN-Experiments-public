"""
Shared utilities for the distillation pipeline.

Includes logging, device management, checkpointing, and reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    name: str = "distillation"
) -> logging.Logger:
    """
    Set up logging for the distillation pipeline.
    
    Args:
        level: Logging level (e.g., logging.INFO)
        log_file: Optional file to write logs to
        name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Device Management
# =============================================================================

def get_device(preference: str = "auto") -> torch.device:
    """
    Get the appropriate compute device.
    
    Args:
        preference: "auto", "cuda", "mps", or "cpu"
        
    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    elif preference == "mps":
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict[str, Any]:
    """Get information about the current compute environment."""
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    
    return info


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Note: These settings can impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.debug(f"Set random seed to {seed}")


@contextmanager
def reproducible_context(seed: int) -> Iterator[None]:
    """Context manager for reproducible operations."""
    # Save current state
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    
    # Set seed
    set_seed(seed)
    
    try:
        yield
    finally:
        # Restore state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


# =============================================================================
# Checkpointing
# =============================================================================

@dataclass
class CheckpointMetadata:
    """Metadata stored with checkpoints."""
    epoch: int
    step: int
    loss: float
    timestamp: str
    config_hash: str
    metrics: dict[str, float]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    step: int,
    loss: float,
    path: str | Path,
    config: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch
        step: Current training step
        loss: Current loss value
        path: Where to save the checkpoint
        config: Optional config to hash for verification
        metrics: Optional additional metrics
        extra: Optional extra data to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    config_hash = ""
    if config:
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    metadata = CheckpointMetadata(
        epoch=epoch,
        step=step,
        loss=loss,
        timestamp=datetime.now().isoformat(),
        config_hash=config_hash,
        metrics=metrics or {},
    )
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": asdict(metadata),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if extra:
        checkpoint["extra"] = extra
    
    torch.save(checkpoint, path)
    logger.debug(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to the checkpoint
        model: Optional model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        strict: Whether to require exact state dict match
        
    Returns:
        The full checkpoint dict including metadata
    """
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Get the size of a model's parameters in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1e6


def freeze_model(model: nn.Module) -> None:
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True
    model.train()


# =============================================================================
# Timing Utilities
# =============================================================================

class Timer:
    """Simple timer for measuring code execution time."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float = 0.0
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            if self.name:
                logger.debug(f"{self.name}: {self.elapsed:.4f}s")
    
    def reset(self) -> None:
        self.start_time = None
        self.elapsed = 0.0


@contextmanager
def cuda_sync_timer(name: str = "") -> Iterator[Timer]:
    """Timer that synchronizes CUDA operations before measuring."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    timer = Timer(name)
    with timer:
        yield timer
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =============================================================================
# Progress and Metrics
# =============================================================================

class MetricsTracker:
    """Track and aggregate training metrics."""
    
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}
        self.step_count = 0
    
    def update(self, metrics: dict[str, float]) -> None:
        """Add a new set of metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        self.step_count += 1
    
    def average(self, key: str, last_n: int | None = None) -> float:
        """Get the average of a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        values = self.metrics[key]
        if last_n is not None:
            values = values[-last_n:]
        return sum(values) / len(values)
    
    def latest(self, key: str) -> float:
        """Get the latest value of a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]
    
    def summary(self, last_n: int | None = None) -> dict[str, float]:
        """Get average of all metrics."""
        return {key: self.average(key, last_n) for key in self.metrics}
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.step_count = 0


# =============================================================================
# File Utilities
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_unique_path(base_path: str | Path, suffix: str = "") -> Path:
    """Get a unique path by appending a number if the path exists."""
    base_path = Path(base_path)
    if not base_path.exists():
        return base_path
    
    stem = base_path.stem
    ext = base_path.suffix
    parent = base_path.parent
    
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}{ext}"
        if not new_path.exists():
            return new_path
        counter += 1


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load data from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
