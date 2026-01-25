"""
Unit Tests for ONNX Inference
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Check if onnxruntime is available
pytest.importorskip("onnxruntime")
pytest.importorskip("tf2onnx")

import onnxruntime as ort
from training.baseline_pinn import NavierStokesPINN


class TestONNXConversion:
    """Test ONNX model conversion and inference."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and minimally train a model."""
        model = NavierStokesPINN(
            hidden_layers=2,
            hidden_units=16,
            nu=1e-3
        )
        
        # Build model
        dummy_input = tf.random.normal((10, 3))
        model(dummy_input)
        
        return model
    
    @pytest.fixture
    def onnx_model_path(self, trained_model):
        """Convert model to ONNX and return path."""
        import tf2onnx
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            
            # Convert
            input_signature = [
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name='input')
            ]
            
            tf2onnx.convert.from_keras(
                trained_model,
                input_signature=input_signature,
                opset=13,
                output_path=str(onnx_path)
            )
            
            yield str(onnx_path)
    
    def test_onnx_conversion(self, trained_model):
        """Test basic ONNX conversion."""
        import tf2onnx
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'test_model.onnx'
            
            input_signature = [
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name='input')
            ]
            
            model_proto, _ = tf2onnx.convert.from_keras(
                trained_model,
                input_signature=input_signature,
                opset=13,
                output_path=str(onnx_path)
            )
            
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0
    
    def test_onnx_inference(self, trained_model):
        """Test ONNX model inference."""
        import tf2onnx
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / 'model.onnx')
            
            input_signature = [
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name='input')
            ]
            
            tf2onnx.convert.from_keras(
                trained_model,
                input_signature=input_signature,
                opset=13,
                output_path=onnx_path
            )
            
            # Load and run inference
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            
            test_input = np.random.randn(5, 3).astype(np.float32)
            output = sess.run([output_name], {input_name: test_input})[0]
            
            assert output.shape == (5, 3)
            assert np.all(np.isfinite(output))
    
    def test_keras_onnx_consistency(self, trained_model):
        """ONNX output should match Keras output."""
        import tf2onnx
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / 'model.onnx')
            
            input_signature = [
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name='input')
            ]
            
            tf2onnx.convert.from_keras(
                trained_model,
                input_signature=input_signature,
                opset=13,
                output_path=onnx_path
            )
            
            # Test input
            test_input = np.random.randn(10, 3).astype(np.float32)
            
            # Keras prediction
            keras_output = trained_model(test_input).numpy()
            
            # ONNX prediction
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            onnx_output = sess.run([output_name], {input_name: test_input})[0]
            
            # Should be very close
            max_diff = np.max(np.abs(keras_output - onnx_output))
            assert max_diff < 1e-4
    
    def test_onnx_batch_sizes(self, trained_model):
        """ONNX model should work with various batch sizes."""
        import tf2onnx
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / 'model.onnx')
            
            input_signature = [
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name='input')
            ]
            
            tf2onnx.convert.from_keras(
                trained_model,
                input_signature=input_signature,
                opset=13,
                output_path=onnx_path
            )
            
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            
            for batch_size in [1, 5, 10, 100]:
                test_input = np.random.randn(batch_size, 3).astype(np.float32)
                output = sess.run([output_name], {input_name: test_input})[0]
                
                assert output.shape == (batch_size, 3)


class TestONNXPerformance:
    """Test ONNX inference performance."""
    
    @pytest.fixture
    def onnx_session(self):
        """Create ONNX session with test model."""
        import tf2onnx
        
        model = NavierStokesPINN(hidden_layers=4, hidden_units=32)
        model(tf.zeros((1, 3)))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / 'model.onnx')
            
            input_signature = [
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name='input')
            ]
            
            tf2onnx.convert.from_keras(
                model,
                input_signature=input_signature,
                opset=13,
                output_path=onnx_path
            )
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            sess_options.intra_op_num_threads = 1
            
            session = ort.InferenceSession(onnx_path, sess_options=sess_options)
            
            yield session
    
    def test_inference_latency(self, onnx_session):
        """Test single inference latency."""
        import time
        
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        
        test_input = np.random.randn(1, 3).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            onnx_session.run([output_name], {input_name: test_input})
        
        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            onnx_session.run([output_name], {input_name: test_input})
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        mean_ms = np.mean(times)
        
        # Should be reasonably fast (< 10ms on most hardware)
        assert mean_ms < 10.0
        
        print(f"\nInference latency: {mean_ms:.3f} ms (target: <1ms)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
