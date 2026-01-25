#!/usr/bin/env python
"""
Main entry point for WavePINN-NIF training.

Usage:
    python main.py --config config/default_config.yaml
    python main.py --quick-test
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import jax


def main():
    parser = argparse.ArgumentParser(description='WavePINN-NIF Training')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with minimal epochs')
    parser.add_argument('--test', action='store_true',
                        help='Run test suite')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Print JAX info
    print("=" * 60)
    print("WavePINN-NIF-ComplexMedia")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print("=" * 60)
    
    if args.test:
        # Run test suite
        from tests.test_forward_sim import run_all_tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
    
    if args.quick_test:
        # Quick test
        from src.trainer import quick_train
        print("\nRunning quick test...")
        trainer, metrics = quick_train(n_epochs=50, batch_size=128)
        print(f"\nFinal metrics: {metrics}")
        return
    
    # Full training
    print(f"\nLoading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs if specified
    if args.epochs is not None:
        config['training']['n_epochs'] = args.epochs
    
    # Disable W&B if requested
    if args.no_wandb:
        config['logging']['use_wandb'] = False
    
    # Import trainer
    from src.trainer import WavePINNTrainer
    from src.data_generator import SyntheticWaveData
    
    # Generate data
    print("\nGenerating synthetic data...")
    domain = config.get('domain', {})
    data_config = config.get('data', {})
    
    data_gen = SyntheticWaveData(
        domain_size=tuple(domain.get('domain_size', [1.0, 1.0])),
        n_collocation=data_config.get('n_collocation', 10000),
        n_boundary=data_config.get('n_boundary', 1000),
        n_initial=data_config.get('n_initial', 1000),
        t_max=domain.get('t_max', 1.0),
        seed=config.get('seed', 42)
    )
    
    data = data_gen.sample_collocation_points()
    
    # Optional: generate velocity field
    medium_type = data_config.get('medium_type', 'layered')
    if medium_type == 'layered':
        c_field = data_gen.generate_layered_medium(n_layers=data_config.get('n_layers', 5))
    elif medium_type == 'random':
        c_field = data_gen.generate_random_medium(
            correlation_length=data_config.get('correlation_length', 0.1)
        )
    else:
        c_field = data_gen.generate_homogeneous_medium(c_value=3.0)
    
    print(f"  Interior points: {data['interior'].shape}")
    print(f"  Velocity field: {c_field.shape}")
    print(f"  Velocity range: [{float(c_field.min()):.2f}, {float(c_field.max()):.2f}]")
    
    # Create trainer
    print("\nInitializing trainer...")
    use_wandb = config.get('logging', {}).get('use_wandb', True) and not args.no_wandb
    trainer = WavePINNTrainer(config, use_wandb=use_wandb)
    
    # Train
    params = trainer.train(data)
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_metrics = trainer.evaluate({
        'interior': data['interior'][:500],
        'boundary': {k: v[:100] for k, v in data['boundary'].items()},
        'initial': data['initial'][:200],
    })
    
    print("\nFinal metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
