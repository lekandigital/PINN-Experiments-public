# ClothGNN Testing & Improvement Prompt for RTX 3090

## Mission
Run the ClothGNN test suite on a local NVIDIA RTX 3090 server, verify model functionality, benchmark performance, and implement targeted improvements.

## Target Hardware
- **Server**: `REDACTED_SERVER` (SSH)
- **Password**: `REDACTED_PASSWORD`
- **GPU**: NVIDIA GeForce RTX 3090 Ti (24GB VRAM)
- **CUDA**: 13.0
- **Driver**: 580.82.09
- **OS**: Ubuntu 22.04.5 LTS

---

## Phase 1: Environment Setup on RTX 3090 Server

### Step 1.1: Connect and Verify GPU
```bash
ssh REDACTED_SERVER
# Password: REDACTED_PASSWORD

# Verify GPU is accessible
nvidia-smi
python3 --version
```

### Step 1.2: Create Project Directory and Clone/Copy Files
**Agent Task**: Either clone from git or copy the project files to the server.

```bash
# Create workspace
mkdir -p ~/clothgnn_test
cd ~/clothgnn_test

# Option A: If syncing from local machine (run from local):
# scp -r /Users/lekanadeyeri/Dev/PINN-Experiments/projects/05-clothgnn__project-space/clothgnn REDACTED_SERVER:~/clothgnn_test/

# Option B: Create files directly on server (if needed)
```

### Step 1.3: Install Python Dependencies
**Agent Task**: Set up Python environment with PyTorch Geometric.

```bash
# Create virtual environment (recommended)
python3 -m venv ~/clothgnn_venv
source ~/clothgnn_venv/bin/activate

# Install PyTorch with CUDA 12.1+ support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric

# Install additional dependencies
pip install h5py numpy scipy matplotlib tqdm
```

---

## Phase 2: Run Test Suite

### Step 2.1: Generate Test Data
```bash
cd ~/clothgnn_test/clothgnn/tests
python3 generate_test_data.py
```

**Expected Output**:
- Creates `gravity_10x10.h5` (100 vertices)
- Creates `wind_10x10.h5` (100 vertices)
- Creates `gravity_32x32.h5` (1024 vertices)

### Step 2.2: Run Full Test Suite
```bash
cd ~/clothgnn_test/clothgnn/tests
python3 test_model.py 2>&1 | tee ~/clothgnn_test/test_results_rtx3090.txt
```

**Expected Tests**:
1. `test_encoder()` - Encoder forward pass
2. `test_decoder()` - Decoder forward pass
3. `test_forward_pass()` - Complete model forward pass
4. `test_collision_module()` - Collision detection with sphere SDF
5. `test_loss_computation()` - All 4 loss functions + backprop
6. `test_training_step()` - Single training iteration
7. `test_multi_step_rollout()` - 10-step rollout stability
8. `test_gpu_performance()` - Benchmark FPS (target: >500 FPS for 1K vertices on RTX 3090)
9. `test_larger_mesh_performance()` - Scalability test (100-5000 vertices)

### Step 2.3: Record GPU Usage During Tests
```bash
# In a separate terminal:
ssh REDACTED_SERVER
watch -n 1 nvidia-smi
```

---

## Phase 3: Verify Success Criteria

**All tests should pass with:**
- ✓ No NaN or Inf values in forward pass
- ✓ Gradients computed correctly
- ✓ Training loss decreasing
- ✓ Multi-step rollout stable (no divergence)
- ✓ FPS benchmark: **>500 FPS** for 1K vertices on RTX 3090 (L40S achieved 1,901 FPS)
- ✓ Memory usage: **<2GB VRAM** for 1K vertices

---

## Phase 4: Improvements to Implement

After verifying tests pass, implement these targeted improvements:

### 4.1: Add requirements.txt
**Agent Task**: Create dependency file in `clothgnn/` directory.

```
# clothgnn/requirements.txt
torch>=2.1.0
torch-geometric>=2.4.0
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
h5py>=3.0.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### 4.2: Add Model Checkpointing
**Agent Task**: Add save/load utilities to `clothgnn/models/clothgnn.py`.

```python
def save_checkpoint(self, path, optimizer=None, epoch=None, loss=None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': self.state_dict(),
        'hidden_dim': self.hidden_dim,
        'config': {
            'node_feat_dim': self.encoder.lin_node.in_features,
            'hidden_dim': self.hidden_dim,
        }
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    torch.save(checkpoint, path)

@classmethod
def load_checkpoint(cls, path, device='cuda'):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = cls(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint
```

### 4.3: Add Simple Training Script
**Agent Task**: Create `clothgnn/train.py`.

```python
"""
Simple training script for ClothGNN.
Usage: python train.py --data_path <path> --epochs 100 --batch_size 1
"""
import argparse
import torch
from torch_geometric.data import Data
import h5py
from tqdm import tqdm

from models.clothgnn import ClothGNNModel
from utils.losses import position_loss, total_loss, compute_rest_lengths

def load_trajectory_data(data_path):
    """Load HDF5 trajectory dataset."""
    with h5py.File(data_path, 'r') as f:
        trajectory = torch.tensor(f['trajectory'][:], dtype=torch.float32)
        edge_index = torch.tensor(f['edge_index'][:], dtype=torch.long)
        rest_positions = torch.tensor(f['rest_positions'][:], dtype=torch.float32)
    return trajectory, edge_index, rest_positions

def create_node_features(positions, velocities=None, dim=16):
    """Create node features from positions and velocities."""
    if velocities is None:
        velocities = torch.zeros_like(positions)
    # Concatenate position (3) + velocity (3) + padding (10) = 16
    features = torch.cat([positions, velocities], dim=1)
    padding = torch.zeros(len(positions), dim - 6)
    return torch.cat([features, padding], dim=1)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Load data
    trajectory, edge_index, rest_positions = load_trajectory_data(args.data_path)
    rest_lengths = compute_rest_lengths(rest_positions, edge_index)

    num_frames = len(trajectory)
    num_nodes = len(rest_positions)

    # Move to device
    trajectory = trajectory.to(device)
    edge_index = edge_index.to(device)
    rest_lengths = rest_lengths.to(device)

    # Initialize model
    model = ClothGNNModel(node_feat_dim=16, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        # Reset hidden state for each sequence
        h = model.init_hidden(num_nodes, device)

        # Iterate through frames
        for t in range(num_frames - 1):
            current_pos = trajectory[t]
            next_pos = trajectory[t + 1]
            target_disp = next_pos - current_pos

            # Create features
            if t > 0:
                velocity = current_pos - trajectory[t - 1]
            else:
                velocity = torch.zeros_like(current_pos)

            node_features = create_node_features(current_pos, velocity)
            data = Data(x=node_features, edge_index=edge_index, pos=current_pos)

            # Forward pass
            pred_disp, h = model(data, h)

            # Compute loss
            pred_pos = current_pos + pred_disp
            loss = total_loss(
                pred_disp, target_disp, pred_pos,
                edge_index, rest_lengths,
                lambda_p=1.0, lambda_e=args.lambda_e
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Detach hidden state for truncated BPTT
            h = h.detach()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (num_frames - 1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            model.save_checkpoint(
                f"checkpoint_epoch{epoch+1}.pt",
                optimizer=optimizer,
                epoch=epoch+1,
                loss=avg_loss
            )

    # Save final model
    model.save_checkpoint("model_final.pt", optimizer=optimizer, epoch=args.epochs, loss=avg_loss)
    print(f"Training complete. Final loss: {avg_loss:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ClothGNN model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--lambda_e', type=float, default=0.1, help='Edge length loss weight')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N epochs')

    args = parser.parse_args()
    train(args)
```

### 4.4: Add Configuration File
**Agent Task**: Create `clothgnn/config/default.yaml`.

```yaml
# ClothGNN Configuration
model:
  node_feat_dim: 16
  hidden_dim: 64
  encoder_type: "gnn"
  decoder_layers: 3

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 1
  gradient_clip: 1.0

loss:
  lambda_position: 1.0
  lambda_edge: 0.1
  lambda_shear: 0.05
  lambda_seam: 0.02

collision:
  grid_dim: 32
  bounds:
    x: [-1.0, 1.0]
    y: [-1.0, 2.0]
    z: [-1.0, 1.0]

data:
  mesh_resolution: 10
  num_frames: 60
  dt: 0.01
```

### 4.5: Add Inference Pipeline
**Agent Task**: Create `clothgnn/inference.py`.

```python
"""
High-level inference wrapper for ClothGNN.
Usage: python inference.py --model checkpoint.pt --input mesh.h5 --output predictions.h5
"""
import argparse
import torch
import h5py
import numpy as np
from models.clothgnn import ClothGNNModel
from torch_geometric.data import Data

class ClothGNNInference:
    """Inference wrapper with preprocessing and batching."""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model, checkpoint = ClothGNNModel.load_checkpoint(model_path, self.device)
        self.model.eval()
        self.hidden_state = None

    def reset_hidden(self, num_nodes):
        """Reset hidden state for new sequence."""
        self.hidden_state = self.model.init_hidden(num_nodes, self.device)

    def predict_step(self, positions, edge_index, velocity=None):
        """Predict one frame displacement."""
        num_nodes = len(positions)

        if self.hidden_state is None:
            self.reset_hidden(num_nodes)

        # Create features
        if velocity is None:
            velocity = torch.zeros_like(positions)

        features = torch.cat([positions, velocity], dim=1)
        padding = torch.zeros(num_nodes, 10, device=self.device)
        node_features = torch.cat([features, padding], dim=1)

        data = Data(
            x=node_features,
            edge_index=edge_index.to(self.device),
            pos=positions.to(self.device)
        )

        with torch.no_grad():
            pred_disp, self.hidden_state = self.model(data, self.hidden_state)

        return pred_disp.cpu()

    def rollout(self, initial_positions, edge_index, num_steps):
        """Multi-step prediction rollout."""
        self.reset_hidden(len(initial_positions))

        positions = initial_positions.clone()
        trajectory = [positions.clone()]

        for step in range(num_steps):
            velocity = trajectory[-1] - trajectory[-2] if len(trajectory) > 1 else torch.zeros_like(positions)
            displacement = self.predict_step(positions, edge_index, velocity)
            positions = positions + displacement
            trajectory.append(positions.clone())

        return torch.stack(trajectory)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model checkpoint path')
    parser.add_argument('--input', required=True, help='Input mesh HDF5')
    parser.add_argument('--output', required=True, help='Output predictions HDF5')
    parser.add_argument('--steps', type=int, default=60, help='Prediction steps')
    args = parser.parse_args()

    # Load input
    with h5py.File(args.input, 'r') as f:
        positions = torch.tensor(f['rest_positions'][:], dtype=torch.float32)
        edge_index = torch.tensor(f['edge_index'][:], dtype=torch.long)

    # Run inference
    inferencer = ClothGNNInference(args.model)
    trajectory = inferencer.rollout(positions, edge_index, args.steps)

    # Save output
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('trajectory', data=trajectory.numpy())
        f.create_dataset('edge_index', data=edge_index.numpy())

    print(f"Saved {args.steps} frames to {args.output}")

if __name__ == '__main__':
    main()
```

---

## Phase 5: Verify Improvements

### Step 5.1: Test Training Script
```bash
cd ~/clothgnn_test/clothgnn
python3 train.py --data_path tests/gravity_10x10.h5 --epochs 50 --save_every 25
```

**Expected**:
- Loss decreases over epochs
- Checkpoints saved at epoch 25 and 50
- Final model saved as `model_final.pt`

### Step 5.2: Test Inference Pipeline
```bash
python3 inference.py --model model_final.pt --input tests/gravity_10x10.h5 --output predictions.h5 --steps 30
```

### Step 5.3: Run Tests Again
```bash
cd tests
python3 test_model.py
```

---

## Summary Checklist

Execute in order, confirming each:

- [ ] Connect to RTX 3090 server via SSH
- [ ] Verify GPU is accessible with `nvidia-smi`
- [ ] Create virtual environment and install PyTorch + PyG
- [ ] Copy/sync project files to server
- [ ] Generate test data with `generate_test_data.py`
- [ ] Run full test suite with `test_model.py`
- [ ] Verify all tests pass with no NaN/Inf
- [ ] Record FPS benchmark (should be >500 FPS for 1K vertices)
- [ ] Record GPU memory usage
- [ ] Create `requirements.txt`
- [ ] Add model checkpointing to `clothgnn.py`
- [ ] Create `train.py` training script
- [ ] Create `config/default.yaml`
- [ ] Create `inference.py` inference pipeline
- [ ] Run training on synthetic data
- [ ] Test inference pipeline
- [ ] Verify all improvements work correctly

**End of prompt. Begin execution and report progress at each phase.**
