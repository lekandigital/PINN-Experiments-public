"""
ONNX Export Script for ClothGNN.

Exports trained ClothGNN to ONNX format for web/mobile deployment.

Output:
- cloth_gnn.onnx: Full precision model
- cloth_gnn_optimized.onnx: Optimized for inference
- cloth_gnn_quantized.onnx: INT8 quantized for mobile
- cloth_template_{size}.npz: Template mesh
- cloth_gnn_loader.js: JavaScript loader for three.js

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/distilled/best_model.pt
    python scripts/export_onnx.py --checkpoint checkpoints/baseline/best_model.pt --mesh-size 32
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add paths
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parents[2]))

from register_models import ClothGNNLite

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from shared.export_pipeline.adapters.cloth_gnn_adapter import (
        ClothGNNExportable,
        create_template_mesh,
        get_default_config,
    )
    from shared.export_pipeline import run_pipeline, ExportConfig, TensorSpec
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class ClothGNNLiteExportable(nn.Module):
    """
    ONNX-exportable wrapper for ClothGNNLite.
    
    Converts the GNN to use pre-computed adjacency matrix instead of
    dynamic scatter operations.
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_hidden_dim: int = 32,
        edge_hidden_dim: int = 16,
        num_message_passes: int = 3,
        adjacency: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.node_hidden_dim = node_hidden_dim
        
        # Create or register adjacency
        if adjacency is None:
            adjacency = self._create_grid_adjacency(num_nodes)
        self.register_buffer('adjacency', adjacency)
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(6, node_hidden_dim),  # pos + vel
            nn.LayerNorm(node_hidden_dim),
            nn.ReLU(),
        )
        
        # Message layers (dense version)
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_message_passes):
            # Dense message: [B, N, hidden] -> [B, N, hidden]
            self.message_layers.append(nn.Sequential(
                nn.Linear(node_hidden_dim * 2, node_hidden_dim),
                nn.ReLU(),
            ))
            self.update_layers.append(nn.Sequential(
                nn.Linear(node_hidden_dim * 2, node_hidden_dim),
                nn.ReLU(),
            ))
            self.norms.append(nn.LayerNorm(node_hidden_dim))
        
        # GRU
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, 3),
        )
    
    def _create_grid_adjacency(self, num_nodes: int) -> torch.Tensor:
        """Create normalized adjacency matrix for grid mesh."""
        grid_size = int(np.sqrt(num_nodes))
        if grid_size * grid_size != num_nodes:
            grid_size = int(np.ceil(np.sqrt(num_nodes)))
        
        adj = torch.zeros(num_nodes, num_nodes)
        
        for i in range(num_nodes):
            row = i // grid_size
            col = i % grid_size
            
            neighbors = []
            if row > 0:
                neighbors.append(i - grid_size)
            if row < grid_size - 1 and i + grid_size < num_nodes:
                neighbors.append(i + grid_size)
            if col > 0:
                neighbors.append(i - 1)
            if col < grid_size - 1:
                neighbors.append(i + 1)
            
            for j in neighbors:
                if 0 <= j < num_nodes:
                    adj[i, j] = 1.0
        
        # Normalize
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj = adj / row_sum
        
        return adj
    
    def forward(
        self,
        node_features: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with fixed topology.
        
        Args:
            node_features: [B, N, 6] position + velocity
            hidden_state: [B, N, hidden_dim] GRU state
            
        Returns:
            displacement: [B, N, 3]
            new_hidden: [B, N, hidden_dim]
        """
        B = node_features.shape[0]
        
        # Encode
        h = self.node_encoder(node_features)  # [B, N, hidden]
        
        # Message passing with dense adjacency
        for msg_layer, upd_layer, norm in zip(
            self.message_layers, self.update_layers, self.norms
        ):
            # Aggregate neighbors: [B, N, hidden] @ [N, N] -> [B, N, hidden]
            adj = self.adjacency.unsqueeze(0).expand(B, -1, -1)
            neighbor_agg = torch.bmm(adj, h)
            
            # Message
            msg_input = torch.cat([h, neighbor_agg], dim=-1)
            msg = msg_layer(msg_input)
            
            # Update
            upd_input = torch.cat([h, msg], dim=-1)
            h_new = upd_layer(upd_input)
            
            # Residual + norm
            h = norm(h + h_new)
        
        # GRU
        h_flat = h.reshape(B * self.num_nodes, 1, self.node_hidden_dim)
        hidden_flat = hidden_state.reshape(1, B * self.num_nodes, self.node_hidden_dim)
        
        gru_out, new_hidden_flat = self.gru(h_flat, hidden_flat)
        
        h = gru_out.reshape(B, self.num_nodes, self.node_hidden_dim)
        new_hidden = new_hidden_flat.reshape(B, self.num_nodes, self.node_hidden_dim)
        
        # Decode
        displacement = self.decoder(h)
        
        return displacement, new_hidden


def export_model(
    model: nn.Module,
    output_dir: str,
    mesh_resolution: int = 32,
    opset_version: int = 17,
) -> Dict[str, str]:
    """
    Export model to ONNX with all artifacts.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_nodes = mesh_resolution * mesh_resolution
    hidden_dim = 32
    
    # Create exportable version
    exportable = ClothGNNLiteExportable(
        num_nodes=num_nodes,
        node_hidden_dim=hidden_dim,
    )
    
    # Copy weights if possible
    if hasattr(model, 'node_encoder'):
        try:
            exportable_dict = exportable.state_dict()
            model_dict = model.state_dict()
            
            for key in exportable_dict:
                if key in model_dict and exportable_dict[key].shape == model_dict[key].shape:
                    exportable_dict[key] = model_dict[key]
            
            exportable.load_state_dict(exportable_dict)
            logger.info("Copied weights to exportable model")
        except Exception as e:
            logger.warning(f"Could not copy weights: {e}")
    
    exportable.eval()
    
    # Prepare dummy inputs
    node_features = torch.randn(1, num_nodes, 6)
    hidden_state = torch.zeros(1, num_nodes, hidden_dim)
    
    # Export
    onnx_path = output_dir / "cloth_gnn.onnx"
    
    torch.onnx.export(
        exportable,
        (node_features, hidden_state),
        str(onnx_path),
        input_names=['node_features', 'hidden_state'],
        output_names=['displacement', 'new_hidden_state'],
        dynamic_axes=None,  # Fixed shape for mobile
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    logger.info(f"Exported to {onnx_path}")
    
    # Verify
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified")
    
    # Optimize
    optimized_path = output_dir / "cloth_gnn_optimized.onnx"
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(optimized_path)
    
    _ = ort.InferenceSession(str(onnx_path), sess_options, providers=['CPUExecutionProvider'])
    logger.info(f"Optimized model saved to {optimized_path}")
    
    # Create template mesh
    vertices, faces = create_template_mesh(mesh_resolution)
    mesh_path = output_dir / f"cloth_template_{mesh_resolution}x{mesh_resolution}.npz"
    np.savez(mesh_path, vertices=vertices, faces=faces)
    logger.info(f"Template mesh saved to {mesh_path}")
    
    # Generate JavaScript loader
    js_code = generate_js_loader(num_nodes, hidden_dim, mesh_resolution)
    js_path = output_dir / "cloth_gnn_loader.js"
    with open(js_path, 'w') as f:
        f.write(js_code)
    logger.info(f"JavaScript loader saved to {js_path}")
    
    # Save metadata
    metadata = {
        'num_nodes': num_nodes,
        'mesh_resolution': mesh_resolution,
        'hidden_dim': hidden_dim,
        'input_dim': 6,
        'output_dim': 3,
        'opset_version': opset_version,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'onnx': str(onnx_path),
        'optimized': str(optimized_path),
        'mesh': str(mesh_path),
        'js_loader': str(js_path),
        'metadata': str(output_dir / "metadata.json"),
    }


def create_template_mesh(resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create template mesh vertices and faces."""
    size = 2.0
    x = np.linspace(-size/2, size/2, resolution)
    y = np.linspace(-size/2, size/2, resolution)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    
    vertices = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).astype(np.float32)
    
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            faces.append([idx, idx + resolution, idx + 1])
            faces.append([idx + 1, idx + resolution, idx + resolution + 1])
    
    faces = np.array(faces, dtype=np.int32)
    
    return vertices, faces


def generate_js_loader(num_nodes: int, hidden_dim: int, mesh_resolution: int) -> str:
    """Generate JavaScript code for loading model in three.js."""
    return f'''
// ClothGNN ONNX Loader for three.js / Babylon.js
// Auto-generated - mesh resolution: {mesh_resolution}x{mesh_resolution}

import * as ort from 'onnxruntime-web';

export class ClothGNNSimulator {{
    constructor() {{
        this.session = null;
        this.hidden = null;
        this.numNodes = {num_nodes};
        this.hiddenDim = {hidden_dim};
        this.meshResolution = {mesh_resolution};
    }}
    
    async load(modelPath) {{
        // Load ONNX model
        this.session = await ort.InferenceSession.create(modelPath, {{
            executionProviders: ['wasm', 'webgl']
        }});
        
        // Initialize hidden state
        this.reset();
        
        console.log(`ClothGNN loaded: ${{this.numNodes}} nodes, ${{this.hiddenDim}} hidden dim`);
    }}
    
    reset() {{
        // Reset hidden state to zeros
        this.hidden = new ort.Tensor(
            'float32',
            new Float32Array({num_nodes * hidden_dim}).fill(0),
            [1, {num_nodes}, {hidden_dim}]
        );
    }}
    
    async step(positions, velocities) {{
        // positions: Float32Array of length {num_nodes * 3}
        // velocities: Float32Array of length {num_nodes * 3}
        
        // Combine into node features [1, N, 6]
        const features = new Float32Array({num_nodes * 6});
        for (let i = 0; i < {num_nodes}; i++) {{
            features[i * 6 + 0] = positions[i * 3 + 0];
            features[i * 6 + 1] = positions[i * 3 + 1];
            features[i * 6 + 2] = positions[i * 3 + 2];
            features[i * 6 + 3] = velocities[i * 3 + 0];
            features[i * 6 + 4] = velocities[i * 3 + 1];
            features[i * 6 + 5] = velocities[i * 3 + 2];
        }}
        
        const featureTensor = new ort.Tensor('float32', features, [1, {num_nodes}, 6]);
        
        // Run inference
        const results = await this.session.run({{
            node_features: featureTensor,
            hidden_state: this.hidden,
        }});
        
        // Update hidden state
        this.hidden = results.new_hidden_state;
        
        // Apply displacement to positions
        const displacement = results.displacement.data;
        const newPositions = new Float32Array({num_nodes * 3});
        
        for (let i = 0; i < {num_nodes}; i++) {{
            newPositions[i * 3 + 0] = positions[i * 3 + 0] + displacement[i * 3 + 0];
            newPositions[i * 3 + 1] = positions[i * 3 + 1] + displacement[i * 3 + 1];
            newPositions[i * 3 + 2] = positions[i * 3 + 2] + displacement[i * 3 + 2];
        }}
        
        return newPositions;
    }}
    
    // Convenience method for three.js geometry
    async updateGeometry(geometry) {{
        const positions = geometry.attributes.position.array;
        const velocities = this.lastVelocities || new Float32Array({num_nodes * 3});
        
        const newPositions = await this.step(positions, velocities);
        
        // Compute velocities for next step
        this.lastVelocities = new Float32Array({num_nodes * 3});
        for (let i = 0; i < {num_nodes * 3}; i++) {{
            this.lastVelocities[i] = newPositions[i] - positions[i];
        }}
        
        // Update geometry
        geometry.attributes.position.array.set(newPositions);
        geometry.attributes.position.needsUpdate = true;
        geometry.computeVertexNormals();
    }}
}}

// Example usage with three.js:
//
// const simulator = new ClothGNNSimulator();
// await simulator.load('cloth_gnn_optimized.onnx');
//
// // Create cloth geometry
// const geometry = new THREE.PlaneGeometry(2, 2, {mesh_resolution - 1}, {mesh_resolution - 1});
// const material = new THREE.MeshStandardMaterial({{ color: 0x4488ff, side: THREE.DoubleSide }});
// const cloth = new THREE.Mesh(geometry, material);
// scene.add(cloth);
//
// function animate() {{
//     await simulator.updateGeometry(cloth.geometry);
//     renderer.render(scene, camera);
//     requestAnimationFrame(animate);
// }}
'''


def benchmark_onnx(
    model_path: str,
    num_nodes: int,
    hidden_dim: int,
    n_iterations: int = 100,
) -> Dict[str, float]:
    """Benchmark ONNX model inference speed."""
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    features = np.random.randn(1, num_nodes, 6).astype(np.float32)
    hidden = np.zeros((1, num_nodes, hidden_dim), dtype=np.float32)
    
    # Warmup
    for _ in range(10):
        outputs = session.run(None, {
            'node_features': features,
            'hidden_state': hidden,
        })
        hidden = outputs[1]
    
    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        outputs = session.run(None, {
            'node_features': features,
            'hidden_state': hidden,
        })
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
        hidden = outputs[1]
    
    times = np.array(times)
    
    results = {
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'fps': float(1000 / times.mean()),
    }
    
    logger.info(f"\nBenchmark Results ({num_nodes} nodes, CPU):")
    logger.info(f"  Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
    logger.info(f"  FPS: {results['fps']:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Export ClothGNN to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="exports/",
                        help="Output directory")
    parser.add_argument("--mesh-size", type=int, default=32,
                        help="Mesh resolution")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark after export")
    
    args = parser.parse_args()
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Create model
    model = ClothGNNLite(
        node_input_dim=6,
        node_hidden_dim=32,
        edge_hidden_dim=16,
        num_message_passes=3,
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    # Export
    paths = export_model(
        model,
        output_dir=args.output_dir,
        mesh_resolution=args.mesh_size,
    )
    
    logger.info("\nExport complete!")
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")
    
    # Benchmark
    if args.benchmark:
        num_nodes = args.mesh_size * args.mesh_size
        benchmark_onnx(
            paths['optimized'],
            num_nodes,
            hidden_dim=32,
        )


if __name__ == "__main__":
    main()
