#!/usr/bin/env python3
"""
Inference speed test for NIF-Cloth4D-Temporal.

Validates that the model achieves ≥3 fps on target GPU.
Target: < 333ms per frame (3 fps) on NVIDIA L40S.

Usage:
    python scripts/test_inference.py
    python scripts/test_inference.py --checkpoint checkpoints/best.pt
    python scripts/test_inference.py --hidden_dim 512 --num_layers 7
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from src.models import FourierFeatureMLP
from src.utils import set_seed, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test inference speed')
    
    # Model
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_freqs', type=int, default=16)
    parser.add_argument('--use_gru', action='store_true')
    parser.add_argument('--gru_hidden', type=int, default=128)
    
    # Test settings
    parser.add_argument('--num_frames', type=int, default=120,
                       help='Number of frames in sequence (4s @ 30fps = 120)')
    parser.add_argument('--num_points', type=int, default=8192,
                       help='Points to sample per frame')
    parser.add_argument('--warmup_frames', type=int, default=10,
                       help='Warmup frames before timing')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use mixed precision')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    
    # Device
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'])
    
    return parser.parse_args()


def sample_cloth_surface(num_points: int, device: torch.device) -> torch.Tensor:
    """
    Sample points from the cloth surface/volume.
    
    In production, this would sample from the actual cloth mesh.
    For testing, we sample from a box approximating cloth extent.
    """
    # Cloth approximate bounds: [-0.5, 0.5] x [0.5, 1.5] x [-0.5, 0.5]
    xyz = torch.zeros(num_points, 3, device=device)
    xyz[:, 0] = torch.rand(num_points, device=device) - 0.5  # x: [-0.5, 0.5]
    xyz[:, 1] = torch.rand(num_points, device=device) + 0.5  # y: [0.5, 1.5]
    xyz[:, 2] = torch.rand(num_points, device=device) - 0.5  # z: [-0.5, 0.5]
    return xyz


def test_inference_speed(
    model: FourierFeatureMLP,
    device: torch.device,
    num_frames: int = 120,
    num_points: int = 8192,
    warmup_frames: int = 10,
    use_amp: bool = True,
    fps: float = 30.0,
) -> dict:
    """
    Test inference speed of the model.
    
    Args:
        model: Trained or random model
        device: Target device
        num_frames: Number of frames to test
        num_points: Points to sample per frame
        warmup_frames: Frames to run before timing
        use_amp: Use mixed precision
        fps: Target framerate
        
    Returns:
        Dictionary with timing results
    """
    model = model.to(device)
    model.eval()
    
    dt = 1.0 / fps
    target_ms = 1000.0 / 3.0  # 333ms for 3 fps
    
    print(f"\n{'=' * 60}")
    print("NIF-Cloth4D-Temporal Inference Speed Test")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Points per frame: {num_points:,}")
    print(f"Sequence length: {num_frames} frames ({num_frames / fps:.1f}s)")
    print(f"Mixed precision: {use_amp}")
    print(f"Target: < {target_ms:.1f}ms per frame (≥3 fps)")
    print(f"{'=' * 60}\n")
    
    # Initialize hidden state (batch size = num_points for point-based inference)
    hidden = model.init_hidden(num_points, device) if model.use_gru else None

    # Warmup
    print(f"Warming up ({warmup_frames} frames)...")
    with torch.no_grad():
        amp_context = torch.cuda.amp.autocast(enabled=use_amp) if device.type == 'cuda' else torch.cpu.amp.autocast(enabled=False)
        with amp_context:
            for frame in range(warmup_frames):
                xyz = sample_cloth_surface(num_points, device)
                t = torch.full((num_points, 1), frame * dt, device=device)
                sdf, hidden = model(xyz, t, hidden)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
    
    # Actual timing
    print("\nRunning speed test...")
    frame_times = []
    
    # Reset hidden state
    hidden = model.init_hidden(num_points, device) if model.use_gru else None
    
    with torch.no_grad():
        amp_context = torch.cuda.amp.autocast(enabled=use_amp) if device.type == 'cuda' else torch.cpu.amp.autocast(enabled=False)
        with amp_context:
            for frame in range(num_frames):
                # Sample points
                xyz = sample_cloth_surface(num_points, device)
                t = torch.full((num_points, 1), frame * dt, device=device)
                
                # Time inference
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t_start = time.perf_counter()
                
                sdf, hidden = model(xyz, t, hidden)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t_end = time.perf_counter()
                
                elapsed_ms = (t_end - t_start) * 1000
                frame_times.append(elapsed_ms)
                
                # Print progress every 20 frames
                if (frame + 1) % 20 == 0:
                    avg_ms = sum(frame_times[-20:]) / 20
                    fps_achieved = 1000.0 / avg_ms if avg_ms > 0 else float('inf')
                    status = "✓" if avg_ms < target_ms else "✗"
                    print(f"  Frame {frame + 1:3d}/{num_frames}: {avg_ms:6.2f}ms ({fps_achieved:5.1f} fps) {status}")
    
    # Results
    frame_times = torch.tensor(frame_times)
    results = {
        'mean_ms': frame_times.mean().item(),
        'std_ms': frame_times.std().item(),
        'min_ms': frame_times.min().item(),
        'max_ms': frame_times.max().item(),
        'median_ms': frame_times.median().item(),
        'p95_ms': frame_times.quantile(0.95).item(),
        'p99_ms': frame_times.quantile(0.99).item(),
        'fps_mean': 1000.0 / frame_times.mean().item(),
        'fps_median': 1000.0 / frame_times.median().item(),
        'passes_target': frame_times.mean().item() < target_ms,
    }
    
    # Print results
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"Mean:   {results['mean_ms']:7.2f}ms ({results['fps_mean']:5.1f} fps)")
    print(f"Median: {results['median_ms']:7.2f}ms ({results['fps_median']:5.1f} fps)")
    print(f"Std:    {results['std_ms']:7.2f}ms")
    print(f"Min:    {results['min_ms']:7.2f}ms")
    print(f"Max:    {results['max_ms']:7.2f}ms")
    print(f"P95:    {results['p95_ms']:7.2f}ms")
    print(f"P99:    {results['p99_ms']:7.2f}ms")
    print(f"{'=' * 60}")
    
    if results['passes_target']:
        print(f"✅ PASSED: Mean {results['fps_mean']:.1f} fps exceeds target 3 fps")
    else:
        print(f"❌ FAILED: Mean {results['fps_mean']:.1f} fps below target 3 fps")
        print("   Consider: reducing hidden_dim, num_layers, or enabling AMP")
    
    print(f"{'=' * 60}\n")
    
    return results


def test_memory_usage(
    model: FourierFeatureMLP,
    device: torch.device,
    num_points: int = 8192,
    use_amp: bool = True,
) -> dict:
    """Test GPU memory usage during inference."""
    if device.type != 'cuda':
        print("Memory test only available on CUDA devices")
        return {}
    
    model = model.to(device)
    model.eval()
    
    torch.cuda.reset_peak_memory_stats()
    
    # Run inference
    hidden = model.init_hidden(num_points, device) if model.use_gru else None
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            xyz = sample_cloth_surface(num_points, device)
            t = torch.full((num_points, 1), 0.5, device=device)
            sdf, _ = model(xyz, t, hidden)
    
    torch.cuda.synchronize()
    
    results = {
        'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
        'peak_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
        'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
    }
    
    print(f"\nGPU Memory Usage:")
    print(f"  Allocated: {results['allocated_mb']:.1f} MB")
    print(f"  Peak:      {results['peak_mb']:.1f} MB")
    print(f"  Reserved:  {results['reserved_mb']:.1f} MB")
    
    return results


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    # Create or load model
    if args.checkpoint:
        print(f"Loading model from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Get config from checkpoint
        config = checkpoint.get('config', None)
        if config:
            model = FourierFeatureMLP(
                hidden_dim=config.model.hidden_dim,
                num_layers=config.model.num_layers,
                num_freqs=config.model.num_freqs,
                use_gru=config.model.use_gru,
                gru_hidden=config.model.gru_hidden,
            )
        else:
            model = FourierFeatureMLP(
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_freqs=args.num_freqs,
                use_gru=args.use_gru,
                gru_hidden=args.gru_hidden,
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Creating random model for speed testing...")
        model = FourierFeatureMLP(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_freqs=args.num_freqs,
            use_gru=args.use_gru,
            gru_hidden=args.gru_hidden,
        )
    
    # Run tests
    results = test_inference_speed(
        model=model,
        device=device,
        num_frames=args.num_frames,
        num_points=args.num_points,
        warmup_frames=args.warmup_frames,
        use_amp=args.use_amp,
    )
    
    if device.type == 'cuda':
        test_memory_usage(model, device, args.num_points, args.use_amp)
    
    # Return exit code based on whether target was met
    sys.exit(0 if results['passes_target'] else 1)


if __name__ == '__main__':
    main()
