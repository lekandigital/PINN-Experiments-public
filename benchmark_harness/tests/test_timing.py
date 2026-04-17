"""
Unit tests for timing utilities.
"""

import time
import numpy as np
import pytest

from harness.timing import (
    TimingResult,
    time_generic,
)


class TestTimingResult:
    """Test TimingResult dataclass."""
    
    def test_statistics(self):
        """Test statistical properties of TimingResult."""
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = TimingResult(
            times_ms=times,
            num_warmup=10,
            num_runs=5,
            device="cpu",
            framework="test",
        )
        
        assert result.mean_ms == pytest.approx(3.0)
        assert result.median_ms == pytest.approx(3.0)
        assert result.min_ms == pytest.approx(1.0)
        assert result.max_ms == pytest.approx(5.0)
        assert result.num_runs == 5
    
    def test_percentiles(self):
        """Test percentile calculations."""
        # 100 values from 1 to 100
        times = list(range(1, 101))
        result = TimingResult(
            times_ms=times,
            num_warmup=0,
            num_runs=100,
            device="cpu",
            framework="test",
        )
        
        assert result.p95_ms == pytest.approx(95.05, rel=0.01)
        assert result.p99_ms == pytest.approx(99.01, rel=0.01)
    
    def test_to_timing_stats(self):
        """Test conversion to TimingStats."""
        times = [1.0, 2.0, 3.0]
        result = TimingResult(
            times_ms=times,
            num_warmup=5,
            num_runs=3,
            device="cuda",
            framework="pytorch",
        )
        
        stats = result.to_timing_stats()
        assert stats.mean_ms == pytest.approx(2.0)
        assert stats.num_runs == 3


class TestGenericTiming:
    """Test generic timing function."""
    
    def test_time_simple_function(self):
        """Test timing a simple function."""
        def simple_fn():
            return sum(range(1000))
        
        result = time_generic(
            simple_fn,
            num_warmup=5,
            num_runs=10,
            device="cpu",
            framework="test",
        )
        
        assert result.num_warmup == 5
        assert result.num_runs == 10
        assert len(result.times_ms) == 10
        assert result.mean_ms > 0
        assert result.framework == "test"
    
    def test_time_with_sleep(self):
        """Test that timing captures actual duration."""
        sleep_ms = 10
        
        def sleep_fn():
            time.sleep(sleep_ms / 1000)
        
        result = time_generic(
            sleep_fn,
            num_warmup=2,
            num_runs=5,
            device="cpu",
            framework="test",
        )
        
        # Should be at least sleep_ms (with some tolerance for overhead)
        assert result.mean_ms >= sleep_ms * 0.9
        # Should not be wildly longer
        assert result.mean_ms < sleep_ms * 2
    
    def test_warmup_runs(self):
        """Test that warmup runs are executed but not counted."""
        call_count = [0]
        
        def counting_fn():
            call_count[0] += 1
            return call_count[0]
        
        result = time_generic(
            counting_fn,
            num_warmup=10,
            num_runs=5,
            device="cpu",
            framework="test",
        )
        
        # Should have called function warmup + runs times
        assert call_count[0] == 15
        # But only recorded runs
        assert len(result.times_ms) == 5


class TestPyTorchTiming:
    """Test PyTorch-specific timing."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"),
        reason="PyTorch not available"
    )
    def test_pytorch_cpu(self):
        """Test PyTorch timing on CPU."""
        import torch
        from harness.timing import time_inference_pytorch
        
        model = torch.nn.Linear(10, 5)
        model.eval()
        
        input_data = torch.randn(32, 10)
        
        result = time_inference_pytorch(
            model, input_data,
            num_warmup=5,
            num_runs=10,
            device="cpu",
        )
        
        assert result.framework == "pytorch"
        assert result.device == "cpu"
        assert result.num_runs == 10
        assert result.mean_ms > 0
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"),
        reason="PyTorch not available"
    )
    def test_pytorch_cuda(self):
        """Test PyTorch timing on CUDA."""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from harness.timing import time_inference_pytorch
        
        model = torch.nn.Linear(10, 5).cuda()
        model.eval()
        
        input_data = torch.randn(32, 10).cuda()
        
        result = time_inference_pytorch(
            model, input_data,
            num_warmup=10,
            num_runs=20,
            device="cuda",
            use_cuda_events=True,
        )
        
        assert result.framework == "pytorch"
        assert result.device == "cuda"
        assert result.num_runs == 20


class TestONNXTiming:
    """Test ONNX Runtime timing."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not available"),
        reason="ONNX Runtime not available"
    )
    def test_onnx_timing(self):
        """Test ONNX Runtime timing."""
        # This would require an actual ONNX model file
        # For now, we just test that the import works
        from harness.timing import time_inference_onnx
        assert callable(time_inference_onnx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
