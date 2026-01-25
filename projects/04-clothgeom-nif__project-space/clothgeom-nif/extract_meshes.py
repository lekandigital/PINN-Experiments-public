#!/usr/bin/env python3
"""
Mesh Extraction CLI for ClothGeom-NIF

Extract watertight meshes from trained NIF decoder model.

Usage:
    python extract_meshes.py --checkpoint outputs/checkpoints/best.pt --output meshes/
    python extract_meshes.py --checkpoint outputs/checkpoints/best.pt --resolution 256 --num 10
    python extract_meshes.py --checkpoint outputs/checkpoints/best.pt --latent latent.npy
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.mesh_extractor import (
    MeshExtractor,
    MeshExtractionConfig,
    load_model_for_extraction,
    batch_extract_meshes
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract meshes from trained ClothGeom-NIF model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/meshes',
        help='Output directory for meshes'
    )
    
    parser.add_argument(
        '--resolution', '-r',
        type=int,
        default=128,
        help='Marching cubes resolution'
    )
    
    parser.add_argument(
        '--num', '-n',
        type=int,
        default=5,
        help='Number of random meshes to generate'
    )
    
    parser.add_argument(
        '--latent',
        type=str,
        default=None,
        help='Path to latent code(s) file (.npy or .pt)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to dataset (extract meshes for first N samples)'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['obj', 'ply', 'stl'],
        default='obj',
        help='Output mesh format'
    )
    
    parser.add_argument(
        '--progressive',
        action='store_true',
        help='Extract at multiple resolutions (64, 128, 256)'
    )
    
    parser.add_argument(
        '--smooth',
        type=int,
        default=0,
        help='Laplacian smoothing iterations'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show 3D visualization of meshes'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def load_latents(
    args,
    model_config: dict,
    device: torch.device
) -> torch.Tensor:
    """Load or generate latent codes."""
    latent_dim = model_config.get('latent_dim', 128)
    
    if args.latent:
        # Load from file
        if args.latent.endswith('.npy'):
            latents = torch.from_numpy(np.load(args.latent)).float()
        elif args.latent.endswith('.pt'):
            latents = torch.load(args.latent)
        else:
            raise ValueError(f"Unknown latent file format: {args.latent}")
        
        if latents.dim() == 1:
            latents = latents.unsqueeze(0)
        
        print(f"Loaded {len(latents)} latent codes from {args.latent}")
    
    elif args.data:
        # Load from dataset
        import h5py
        with h5py.File(args.data, 'r') as f:
            latents = torch.from_numpy(f['latent'][:args.num]).float()
        
        print(f"Loaded {len(latents)} latent codes from dataset")
    
    else:
        # Generate random latents
        torch.manual_seed(args.seed)
        latents = torch.randn(args.num, latent_dim)
        
        # Normalize (as expected by model)
        latents = latents / (torch.norm(latents, dim=1, keepdim=True) + 1e-8)
        
        print(f"Generated {len(latents)} random latent codes")
    
    return latents.to(device)


def visualize_mesh(vertices, faces, title="Mesh"):
    """Visualize mesh using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    if len(vertices) == 0:
        print(f"Empty mesh: {title}")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create polygon collection
    mesh_faces = vertices[faces]
    
    # Subsample if too many faces
    if len(mesh_faces) > 10000:
        indices = np.random.choice(len(mesh_faces), 10000, replace=False)
        mesh_faces = mesh_faces[indices]
    
    collection = Poly3DCollection(mesh_faces, alpha=0.8, linewidth=0.1, edgecolor='k')
    collection.set_facecolor('steelblue')
    ax.add_collection3d(collection)
    
    # Set axis limits
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    margin = (v_max - v_min).max() * 0.1
    
    ax.set_xlim(v_min[0] - margin, v_max[0] + margin)
    ax.set_ylim(v_min[1] - margin, v_max[1] + margin)
    ax.set_zlim(v_min[2] - margin, v_max[2] + margin)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\nVertices: {len(vertices)}, Faces: {len(faces)}")
    
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("ClothGeom-NIF Mesh Extraction")
    print("=" * 60)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\n📦 Loading model from {args.checkpoint}...")
    model, model_config = load_model_for_extraction(args.checkpoint, device)
    print(f"   Latent dim: {model_config.get('latent_dim', 128)}")
    print(f"   Hidden dim: {model_config.get('hidden_dim', 256)}")
    
    # Load/generate latents
    print(f"\n🎲 Preparing latent codes...")
    latents = load_latents(args, model_config, device)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create extractor
    config = MeshExtractionConfig(
        resolution=args.resolution,
        smooth_iterations=args.smooth
    )
    extractor = MeshExtractor(model, device, config)
    
    # Extract meshes
    print(f"\n🔧 Extracting meshes at resolution {args.resolution}...")
    
    for i, latent in enumerate(latents):
        print(f"\n--- Mesh {i+1}/{len(latents)} ---")
        
        if args.progressive:
            # Progressive extraction
            meshes = extractor.progressive_extract(latent, resolutions=[64, 128, 256])
            
            for mesh_data in meshes:
                res = mesh_data['resolution']
                output_path = output_dir / f'mesh_{i:04d}_res{res}.{args.format}'
                
                MeshExtractor.save_mesh(
                    mesh_data['vertices'],
                    mesh_data['faces'],
                    str(output_path),
                    mesh_data['normals']
                )
                print(f"   Saved: {output_path} ({mesh_data['num_vertices']} verts, {mesh_data['num_faces']} faces)")
        else:
            # Single resolution
            vertices, faces, normals = extractor.extract_mesh(latent)
            
            output_path = output_dir / f'mesh_{i:04d}.{args.format}'
            MeshExtractor.save_mesh(vertices, faces, str(output_path), normals)
            print(f"   Saved: {output_path} ({len(vertices)} verts, {len(faces)} faces)")
            
            if args.visualize:
                visualize_mesh(vertices, faces, f"Mesh {i+1}")
    
    # Save latents for reference
    latent_path = output_dir / 'latents.pt'
    torch.save(latents.cpu(), latent_path)
    print(f"\n💾 Saved latent codes to: {latent_path}")
    
    print(f"\n✅ Extraction complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Total meshes: {len(latents)}")
    
    if not args.visualize:
        print(f"\n💡 Tip: View meshes with:")
        print(f"   - Blender: File > Import > Wavefront (.obj)")
        print(f"   - MeshLab: File > Import Mesh")
        print(f"   - Online: https://3dviewer.net/")


if __name__ == '__main__':
    main()
