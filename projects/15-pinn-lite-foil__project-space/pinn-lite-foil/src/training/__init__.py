"""
Training Package
Baseline PINN training and knowledge distillation.
"""

from .baseline_pinn import NavierStokesPINN, train_pinn, load_dataset
from .distillation import DistillationTrainer, train_distillation

__all__ = [
    'NavierStokesPINN',
    'train_pinn',
    'load_dataset',
    'DistillationTrainer',
    'train_distillation'
]
