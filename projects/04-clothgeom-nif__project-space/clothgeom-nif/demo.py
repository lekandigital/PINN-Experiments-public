#!/usr/bin/env python3
"""
ClothGeom-NIF Demo Script

End-to-end demonstration of the Neural Implicit Field decoder:
1. Loads trained model (or uses random weights for testing)
2. Generates latent codes (random or from dataset)
3. Extracts meshes at multiple resolutions
4. Visualizes results
5. Computes evaluation metrics (if ground truth available)

Usage:
    python demo.py --checkpoint outputs/checkpoints/best.pt
    python demo.py --test  # Run without trained model (random weights)
    python demo.py --checkpoint outputs/checkpoints/best.pt --data data/generated/cloth_dataset.h5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_nif_decoder
from inference import MeshExtractor, MeshExtractionConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description='ClothGeom-NIF demonstration script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to dataset (for using real latent codes)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/demo',
        help='Output directory'
    )
    
    parser.add_argument(
        '--num', '-n',
        type=int,
        default=5,
        help='Number of meshes to generate'
    )
    
    parser.add_argument(
        '--resolution', '-r',
        type=int,
        default=64,
        help='Mesh extraction resolution'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run demo with random weights (for testing)'
    )
    
    parser.add_argument(
        '--interpolate',
        action='store_true',
        help='Generate interpolation sequence between two latents'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show 3D visualizations'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"📦 Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {
        'latent_dim': 128,
        'hidden_dim': 256,
        'num_layers': 8
    })
    
    model = create_nif_decoder(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('best_val_loss', 'unknown')
    
    print(f"   Loaded from epoch {epoch}")
    print(f"   Validation loss: {val_loss}")
    print(f"   Latent dim: {model_config.get('latent_dim', 128)}")
    
    return model, model_config


def create_demo_model(latent_dim=128, device='cpu'):
    """Create model with random weights for testing."""
    print("🔧 Creating demo model with random weights...")
    
    model_config = {
        'latent_dim': latent_dim,
        'hidden_dim': 128,
        'num_layers': 4
    }
    
    model = create_nif_decoder(model_config)
    model = model.to(device)
    model.eval()
    
    return model, model_config


def generate_latents(num, latent_dim, data_path=None, device='cpu'):
    """Generate or load latent codes."""
    if data_path and os.path.exists(data_path):
        import h5py
        print(f"📊 Loading latents from dataset...")
        with h5py.File(data_path, 'r') as f:
            latents = torch.from_numpy(f['latent'][:num]).float()
        print(f"   Loaded {len(latents)} latent codes")
    else:
        print(f"🎲 Generating {num} random latent codes...")
        latents = torch.randn(num, latent_dim)
        latents = latents / latents.norm(dim=1, keepdim=True)
    
    return latents.to(device)


def generate_interpolation(latent1, latent2, num_steps=10):
    """Generate interpolation sequence between two latents."""
    print(f"🔀 Generating {num_steps}-step interpolation...")
    
    latents = []
    for t in np.linspace(0, 1, num_steps):
        interp = (1 - t) * latent1 + t * latent2
        interp = interp / interp.norm()  # Stay on unit sphere
        latents.append(interp)
    
    return torch.stack(latents)


def extract_and_save_meshes(model, latents, output_dir, resolution=64):
    """Extract meshes from latent codes and save."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = MeshExtractor(
        model,
        config=MeshExtractionConfig(resolution=resolution)
    )
    
    mesh_info = []
    
    for i, latent in enumerate(latents):
        print(f"\n--- Mesh {i+1}/{len(latents)} ---")
        start_time = time.time()
        
        vertices, faces, normals = extractor.extract_mesh(latent)
        
        elapsed = time.time() - start_time
        
        if len(vertices) > 0:
            output_path = output_dir / f'cloth_{i:04d}.obj'
            MeshExtractor.save_mesh(vertices, faces, str(output_path), normals)
            
            mesh_info.append({
                'index': i,
                'vertices': len(vertices),
                'faces': len(faces),
                'path': str(output_path),
                'time': elapsed
            })
            
            print(f"   ✅ Saved: {output_path}")
            print(f"   Vertices: {len(vertices)}, Faces: {len(faces)}")
            print(f"   Time: {elapsed:.2f}s")
        else:
            print(f"   ⚠️ Empty mesh (no surface found)")
            mesh_info.append({
                'index': i,
                'vertices': 0,
                'faces': 0,
                'path': None,
                'time': elapsed
            })
    
    return mesh_info


def visualize_meshes(output_dir, max_show=5):
    """Visualize extracted meshes."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import trimesh
    except ImportError:
        print("⚠️ Visualization requires matplotlib and trimesh")
        return
    
    output_dir = Path(output_dir)
    mesh_files = sorted(output_dir.glob('*.obj'))[:max_show]
    
    if not mesh_files:
        print("No meshes to visualize")
        return
    
    n_meshes = len(mesh_files)
    cols = min(3, n_meshes)
    rows = (n_meshes + cols - 1) // cols
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    
    for i, mesh_file in enumerate(mesh_files):
        mesh = trimesh.load(str(mesh_file))
        
        if len(mesh.vertices) == 0:
            continue
        
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Subsample faces if too many
        faces_to_plot = mesh.faces
        if len(faces_to_plot) > 5000:
            indices = np.random.choice(len(faces_to_plot), 5000, replace=False)
            faces_to_plot = faces_to_plot[indices]
        
        # Create polygon collection
        mesh_polys = mesh.vertices[faces_to_plot]
        collection = Poly3DCollection(mesh_polys, alpha=0.8, linewidth=0.1, edgecolor='gray')
        collection.set_facecolor('steelblue')
        ax.add_collection3d(collection)
        
        # Set limits
        v_min = mesh.vertices.min(axis=0)
        v_max = mesh.vertices.max(axis=0)
        center = (v_min + v_max) / 2
        scale = (v_max - v_min).max() / 2 * 1.2
        
        ax.set_xlim(center[0] - scale, center[0] + scale)
        ax.set_ylim(center[1] - scale, center[1] + scale)
        ax.set_zlim(center[2] - scale, center[2] + scale)
        
        ax.set_title(f'{mesh_file.stem}\n{len(mesh.vertices)}V, {len(mesh.faces)}F')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / 'visualization.png'
    plt.savefig(fig_path, dpi=150)
    print(f"\n📊 Saved visualization to: {fig_path}")
    plt.show()


def compute_metrics(mesh_info):
    """Compute summary metrics."""
    valid_meshes = [m for m in mesh_info if m['vertices'] > 0]
    
    if not valid_meshes:
        return None
    
    metrics = {
        'total_meshes': len(mesh_info),
        'valid_meshes': len(valid_meshes),
        'avg_vertices': np.mean([m['vertices'] for m in valid_meshes]),
        'avg_faces': np.mean([m['faces'] for m in valid_meshes]),
        'avg_time': np.mean([m['time'] for m in valid_meshes]),
        'total_time': sum(m['time'] for m in mesh_info)
    }
    
    return metrics


def print_summary(mesh_info, metrics, output_dir):
    """Print summary of demo results."""
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    
    if metrics:
        print(f"✅ Valid meshes: {metrics['valid_meshes']}/{metrics['total_meshes']}")
        print(f"📊 Average vertices: {metrics['avg_vertices']:.0f}")
        print(f"📊 Average faces: {metrics['avg_faces']:.0f}")
        print(f"⏱️  Average extraction time: {metrics['avg_time']:.2f}s")
        print(f"⏱️  Total time: {metrics['total_time']:.2f}s")
    else:
        print("❌ No valid meshes generated")
    
    print(f"\n📁 Output directory: {output_dir}")
    print("\n💡 Next steps:")
    print("   - View meshes: open outputs/demo/*.obj in Blender/MeshLab")
    print("   - Higher resolution: python demo.py --resolution 128")
    print("   - More meshes: python demo.py --num 20")
    print("=" * 60)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("ClothGeom-NIF Demo")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load or create model
    if args.test or args.checkpoint is None:
        if not args.test and args.checkpoint is None:
            print("⚠️ No checkpoint specified, using random weights")
        model, model_config = create_demo_model(device=device)
    else:
        if not os.path.exists(args.checkpoint):
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            print("   Run training first: python train.py")
            print("   Or use --test flag for demo with random weights")
            sys.exit(1)
        model, model_config = load_model(args.checkpoint, device)
    
    latent_dim = model_config.get('latent_dim', 128)
    
    # Generate latents
    if args.interpolate:
        # Generate interpolation between two random latents
        base_latents = generate_latents(2, latent_dim, args.data, device)
        latents = generate_interpolation(base_latents[0], base_latents[1], args.num)
    else:
        latents = generate_latents(args.num, latent_dim, args.data, device)
    
    # Extract meshes
    print(f"\n🔧 Extracting meshes at resolution {args.resolution}...")
    mesh_info = extract_and_save_meshes(
        model, latents, args.output, args.resolution
    )
    
    # Save latents
    latent_path = Path(args.output) / 'latents.pt'
    torch.save(latents.cpu(), latent_path)
    print(f"\n💾 Saved latent codes to: {latent_path}")
    
    # Compute metrics
    metrics = compute_metrics(mesh_info)
    
    # Print summary
    print_summary(mesh_info, metrics, args.output)
    
    # Visualize if requested
    if args.visualize:
        visualize_meshes(args.output)


if __name__ == '__main__':
    main()
