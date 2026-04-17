"""
Model adapters for the shared training library.

Each adapter wraps a specific model architecture to provide a uniform interface
for the ScheduledSamplingTrainer.
"""

from .base_adapter import RolloutModelAdapter, AdapterState
from .gru_adapter import GRUAdapter
from .inr_adapter import INRAdapter
from .gnn_adapter import GNNAdapter

__all__ = [
    "RolloutModelAdapter",
    "AdapterState",
    "GRUAdapter",
    "INRAdapter",
    "GNNAdapter",
]
