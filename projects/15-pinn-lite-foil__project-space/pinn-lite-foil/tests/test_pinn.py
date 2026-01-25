"""
Unit Tests for PINN Model
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

# TensorFlow config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.baseline_pinn import NavierStokesPINN


class TestNavierStokesPINN:
    """Test suite for PINN model."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return NavierStokesPINN(
            hidden_layers=4,
            hidden_units=32,
            activation='tanh',
            nu=1e-3
        )
    
    def test_model_creation(self, model):
        """Test basic model creation."""
        assert model is not None
        assert len(model.dense_layers) == 4
    
    def test_forward_pass(self, model):
        """Test forward pass with batch input."""
        # Build model
        batch_size = 16
        inputs = tf.random.normal((batch_size, 3))
        
        outputs = model(inputs)
        
        assert outputs.shape == (batch_size, 3)
        assert outputs.dtype == tf.float32
    
    def test_output_range(self, model):
        """Test that outputs are finite."""
        inputs = tf.random.normal((100, 3))
        outputs = model(inputs)
        
        assert tf.reduce_all(tf.math.is_finite(outputs))
    
    def test_pde_residuals(self, model):
        """Test PDE residual computation."""
        # Build model first
        model(tf.zeros((1, 3)))
        
        x = tf.constant([0.5, 0.6, 0.7])
        y = tf.constant([0.1, 0.2, 0.3])
        aoa = tf.constant([5.0, 5.0, 5.0])
        
        cont, mom_x, mom_y = model.compute_pde_residuals(x, y, aoa)
        
        # Residuals should be finite
        assert tf.reduce_all(tf.math.is_finite(cont))
        assert tf.reduce_all(tf.math.is_finite(mom_x))
        assert tf.reduce_all(tf.math.is_finite(mom_y))
    
    def test_model_config(self, model):
        """Test model configuration retrieval."""
        config = model.get_config()
        
        assert config['hidden_layers'] == 4
        assert config['hidden_units'] == 32
        assert config['activation'] == 'tanh'
        assert config['nu'] == 1e-3
    
    def test_model_save_load(self, model):
        """Test model save and load."""
        # Build model
        inputs = tf.random.normal((10, 3))
        outputs1 = model(inputs)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save weights
            weights_path = Path(tmpdir) / 'model.weights.h5'
            model.save_weights(str(weights_path))
            
            # Create new model and load weights
            model2 = NavierStokesPINN(
                hidden_layers=4,
                hidden_units=32,
                activation='tanh',
                nu=1e-3
            )
            model2(tf.zeros((1, 3)))  # Build
            model2.load_weights(str(weights_path))
            
            # Compare outputs
            outputs2 = model2(inputs)
            
            assert tf.reduce_all(tf.abs(outputs1 - outputs2) < 1e-5)
    
    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        inputs = tf.Variable(tf.random.normal((10, 3)))
        
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = tf.reduce_mean(outputs)
        
        grads = tape.gradient(loss, model.trainable_variables)
        
        # All gradients should be non-None and finite
        for grad in grads:
            assert grad is not None
            assert tf.reduce_all(tf.math.is_finite(grad))
    
    def test_different_architectures(self):
        """Test various architecture configurations."""
        configs = [
            {'hidden_layers': 2, 'hidden_units': 16},
            {'hidden_layers': 4, 'hidden_units': 32},
            {'hidden_layers': 8, 'hidden_units': 128},
        ]
        
        for config in configs:
            model = NavierStokesPINN(**config)
            inputs = tf.random.normal((5, 3))
            outputs = model(inputs)
            
            assert outputs.shape == (5, 3)
            assert len(model.dense_layers) == config['hidden_layers']
    
    def test_batch_size_invariance(self, model):
        """Model should work with different batch sizes."""
        model(tf.zeros((1, 3)))  # Build
        
        for batch_size in [1, 10, 100, 1000]:
            inputs = tf.random.normal((batch_size, 3))
            outputs = model(inputs)
            
            assert outputs.shape == (batch_size, 3)


class TestPINNTraining:
    """Test PINN training functionality."""
    
    @pytest.fixture
    def model_and_data(self):
        """Create model and synthetic training data."""
        model = NavierStokesPINN(
            hidden_layers=2,
            hidden_units=16,
            lambda_data=1.0,
            lambda_pde=1.0
        )
        
        # Synthetic data
        n_samples = 100
        inputs = np.random.randn(n_samples, 3).astype(np.float32)
        targets = np.random.randn(n_samples, 3).astype(np.float32)
        
        return model, inputs, targets
    
    def test_compile_and_fit(self, model_and_data):
        """Test model compilation and basic training."""
        model, inputs, targets = model_and_data
        
        model.compile(optimizer='adam')
        
        # Train for a few steps
        history = model.fit(
            inputs, targets,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        assert 'loss' in history.history
        assert len(history.history['loss']) == 5
    
    def test_loss_decreases(self, model_and_data):
        """Loss should generally decrease during training."""
        model, inputs, targets = model_and_data
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
        
        history = model.fit(
            inputs, targets,
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        losses = history.history['loss']
        
        # Loss should decrease overall
        assert losses[-1] < losses[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
