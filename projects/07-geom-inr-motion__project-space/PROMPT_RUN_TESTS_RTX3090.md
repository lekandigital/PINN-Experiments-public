# Geom-INR-Motion: Validation & Improvement on RTX 3090

## OBJECTIVE
Validate and improve the Geom-INR-Motion implementation using an NVIDIA RTX 3090 Ti (24GB VRAM) on remote machine `REDACTED_SERVER`. The core implementation is complete - focus on validation, real data testing, and performance optimization.

---

## REMOTE MACHINE DETAILS

```
Host: REDACTED_SERVER
Password: REDACTED_PASSWORD
GPU: NVIDIA GeForce RTX 3090 Ti
VRAM: 24GB
CUDA: 13.0
Driver: 580.82.09
OS: Ubuntu 22.04.5 LTS
```

**SSH Command:**
```bash
ssh REDACTED_SERVER
# Password: REDACTED_PASSWORD
```

---

## PHASE 1: Environment Setup on RTX 3090 Machine

### Task 1.1: Copy Project Files to Remote

```bash
# From local Mac, copy entire project to remote
scp -r /Users/lekanadeyeri/Dev/PINN-Experiments/projects/07-geom-inr-motion__project-space \
    REDACTED_SERVER:/home/o/

# SSH into remote
ssh REDACTED_SERVER
cd /home/o/07-geom-inr-motion__project-space/geom-inr-motion
```

### Task 1.2: Create Python Environment

```bash
# Create conda environment (recommended)
conda create -n geom-inr python=3.10 -y
conda activate geom-inr

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Verify GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, Device: NVIDIA GeForce RTX 3090 Ti
```

---

## PHASE 2: Validation Tests

### Task 2.1: Run Minimal Test (PRIORITY - Do First)

```bash
cd /home/o/07-geom-inr-motion__project-space/geom-inr-motion
python minimal_test.py
```

**Expected Output:**
```
Device: cuda
PyTorch version: 2.x.x
CUDA version: 12.x
GPU: NVIDIA GeForce RTX 3090 Ti
GPU Memory: 24.0 GB

[1/4] Creating synthetic motion dataset...
[2/4] Building cubic spline interpolators...
[3/4] Building TinyINR network...
[4/4] Training for 100 steps...

✅ MINIMAL TEST PASSED! (Loss < 0.01)
```

**If Test Fails:** Debug issues before proceeding. Common fixes:
- NumPy version conflict: `pip install numpy<2`
- CUDA mismatch: Reinstall PyTorch for correct CUDA version

### Task 2.2: Full Training Validation (500 epochs)

```bash
# Train with same config as VastAI run for comparison
python train.py \
    --synthetic \
    --epochs 500 \
    --batch-size 4096 \
    --checkpoint-dir ./checkpoints_rtx3090 \
    --model base \
    --lr 1e-4 \
    --curvature-weight 0.001 \
    --torsion-weight 0.0001

# Note: RTX 3090 has 24GB vs L40S's 48GB, so use smaller batch size (4096 vs 8192)
```

**Expected Results (compare to VastAI L40S run):**
- VastAI L40S: Final MPJPE = 6.47mm, Best Val Loss = 18.32, Time = 10.1 min
- RTX 3090: Should achieve similar metrics, time may vary (~15-20 min estimate)

### Task 2.3: Load and Evaluate Pre-trained Checkpoints

If you have checkpoints from the VastAI run:

```bash
# Copy checkpoints to remote machine (from local Mac)
scp -r /path/to/checkpoints/* REDACTED_SERVER:/home/o/07-geom-inr-motion__project-space/geom-inr-motion/checkpoints/

# On remote, evaluate the model
python -c "
import torch
from model import create_model

# Load checkpoint
checkpoint = torch.load('checkpoints/model_best.pt', map_location='cuda')
model = create_model('base', num_joints=24, num_actors=100).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
print(f'Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters')

# Test forward pass
actor_ids = torch.zeros(100, dtype=torch.long).cuda()
joint_idxs = torch.arange(24).repeat(100 // 24 + 1)[:100].cuda()
times = torch.rand(100).cuda()
output = model(actor_ids, joint_idxs, times)
print(f'Output shape: {output.shape}')  # Expected: (100, 3)
"
```

---

## PHASE 3: Real Data Validation

### Task 3.1: Download AMASS Sample Data

```bash
# AMASS requires registration at https://amass.is.tue.mpg.de/
# For testing, use a small subset

# Create data directory
mkdir -p /home/o/data/amass

# Download CMU subset (smallest, ~500MB)
# After registration, download manually or use wget with your credentials

# Alternative: Use Human3.6M if you have access
# Or continue with synthetic data for initial validation
```

### Task 3.2: Test Data Pipeline with Synthetic Data

```python
# test_data_pipeline.py
from data_pipeline import create_synthetic_dataset, MotionDataset
from torch.utils.data import DataLoader

# Create synthetic dataset
train_seqs, val_seqs = create_synthetic_dataset(
    num_sequences=50,
    num_frames=200,
    num_joints=24
)

print(f"Train sequences: {len(train_seqs)}")
print(f"Val sequences: {len(val_seqs)}")
print(f"Sample shape: {train_seqs[0].positions.shape}")

# Create DataLoader
train_dataset = MotionDataset(train_seqs, trajectory_length=50, samples_per_epoch=10000)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# Test batch
batch = next(iter(train_loader))
print(f"Batch keys: {batch.keys()}")
print(f"Actor IDs shape: {batch['actor_ids'].shape}")
print(f"Times shape: {batch['times'].shape}")
print(f"Targets shape: {batch['targets'].shape}")
```

---

## PHASE 4: Model Improvements

### Task 4.1: Hyperparameter Tuning

Test different configurations to improve performance:

```bash
# Experiment 1: Larger batch size (monitor VRAM)
python train.py --synthetic --epochs 200 --batch-size 8192 --checkpoint-dir ./exp_large_batch

# Experiment 2: Higher curvature weight for smoother motion
python train.py --synthetic --epochs 200 --batch-size 4096 \
    --curvature-weight 0.01 --torsion-weight 0.001 \
    --checkpoint-dir ./exp_high_geom

# Experiment 3: Cosine annealing scheduler (if not default)
# Modify train.py to use torch.optim.lr_scheduler.CosineAnnealingLR

# Experiment 4: Model size comparison
python train.py --synthetic --epochs 200 --model small --checkpoint-dir ./exp_small_model
python train.py --synthetic --epochs 200 --model large --checkpoint-dir ./exp_large_model
```

### Task 4.2: Add Jerk Loss for Smoother Motion

Edit `train.py` to enable jerk loss:

```python
# In training config or command line
--jerk-weight 0.0001
```

The jerk loss is already implemented in `geometry_losses.py` but currently weighted at 0.0.

### Task 4.3: Implement Motion Interpolation Test

Create a test to verify continuous-time interpolation quality:

```python
# test_interpolation.py
import torch
import numpy as np
from model import create_model
from scipy.interpolate import CubicSpline

# Load trained model
model = create_model('base', num_joints=24, num_actors=100).cuda()
checkpoint = torch.load('checkpoints/model_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate predictions at fractional times
times = torch.linspace(0, 1, 100).cuda()
actor_ids = torch.zeros(100, dtype=torch.long).cuda()
joint_idx = torch.zeros(100, dtype=torch.long).cuda()  # Root joint

with torch.no_grad():
    predictions = model(actor_ids, joint_idx, times)

# Compute smoothness (jerk)
positions = predictions.cpu().numpy()
velocity = np.diff(positions, axis=0)
acceleration = np.diff(velocity, axis=0)
jerk = np.diff(acceleration, axis=0)
smoothness = np.mean(np.abs(jerk))

print(f"Interpolation smoothness (lower is better): {smoothness:.6f}")

# Verify cubic spline-like smoothness
# Compare to linear interpolation baseline
linear_interp = np.linspace(positions[0], positions[-1], len(positions))
linear_jerk = np.diff(np.diff(np.diff(linear_interp, axis=0), axis=0), axis=0)
linear_smoothness = np.mean(np.abs(linear_jerk))

print(f"Linear interpolation smoothness: {linear_smoothness:.6f}")
print(f"Improvement: {(linear_smoothness - smoothness) / linear_smoothness * 100:.1f}%")
```

---

## PHASE 5: Export and Visualization

### Task 5.1: Export Motion to BVH

```python
# export_motion.py
from export_utils import export_bvh, SMPL_JOINT_NAMES, SMPL_PARENTS
from model import create_model
import torch
import numpy as np

# Load model
model = create_model('base', num_joints=24, num_actors=100).cuda()
checkpoint = torch.load('checkpoints/model_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate full skeleton motion
num_frames = 100
positions = np.zeros((num_frames, 24, 3))

for frame_idx in range(num_frames):
    t = torch.tensor([frame_idx / num_frames]).cuda()
    for joint_idx in range(24):
        actor_id = torch.tensor([0], dtype=torch.long).cuda()
        j_idx = torch.tensor([joint_idx], dtype=torch.long).cuda()
        with torch.no_grad():
            pos = model(actor_id, j_idx, t)
        positions[frame_idx, joint_idx] = pos.cpu().numpy()

# Export to BVH
export_bvh(
    filepath='generated_motion.bvh',
    positions=positions,
    joint_names=SMPL_JOINT_NAMES,
    parents=SMPL_PARENTS,
    fps=30,
    scale=100  # Convert to cm
)

print("Exported to generated_motion.bvh")
```

### Task 5.2: Visualize with Matplotlib

```python
# visualize_trajectory.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load generated positions (from export script above)
positions = np.load('generated_positions.npz')['positions']  # Save this in export script

fig = plt.figure(figsize=(15, 5))

# Plot 1: 3D trajectory of root joint
ax1 = fig.add_subplot(131, projection='3d')
root_pos = positions[:, 0, :]  # Root joint over time
ax1.plot(root_pos[:, 0], root_pos[:, 1], root_pos[:, 2], 'b-', linewidth=2)
ax1.set_title('Root Joint Trajectory')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Plot 2: Skeleton at keyframes
ax2 = fig.add_subplot(132, projection='3d')
for frame_idx in [0, 25, 50, 75, 99]:
    skeleton = positions[frame_idx]
    ax2.scatter(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], alpha=0.5, label=f'Frame {frame_idx}')
ax2.set_title('Skeleton at Keyframes')
ax2.legend()

# Plot 3: Per-joint position variance
ax3 = fig.add_subplot(133)
joint_variance = positions.var(axis=0).sum(axis=1)  # Variance per joint
ax3.bar(range(24), joint_variance)
ax3.set_title('Joint Movement Variance')
ax3.set_xlabel('Joint Index')
ax3.set_ylabel('Position Variance')

plt.tight_layout()
plt.savefig('motion_visualization.png', dpi=150)
plt.show()
print("Saved visualization to motion_visualization.png")
```

---

## PHASE 6: Benchmarking

### Task 6.1: Inference Speed Benchmark

```python
# benchmark_inference.py
import torch
import time
from model import create_model

model = create_model('base', num_joints=24, num_actors=100).cuda()
model.eval()

# Warmup
for _ in range(10):
    actor_ids = torch.randint(0, 100, (1000,)).cuda()
    joint_idxs = torch.randint(0, 24, (1000,)).cuda()
    times = torch.rand(1000).cuda()
    with torch.no_grad():
        _ = model(actor_ids, joint_idxs, times)

# Benchmark
torch.cuda.synchronize()
start = time.time()
num_iterations = 100
batch_size = 10000

for _ in range(num_iterations):
    actor_ids = torch.randint(0, 100, (batch_size,)).cuda()
    joint_idxs = torch.randint(0, 24, (batch_size,)).cuda()
    times = torch.rand(batch_size).cuda()
    with torch.no_grad():
        output = model(actor_ids, joint_idxs, times)

torch.cuda.synchronize()
elapsed = time.time() - start

samples_per_second = (num_iterations * batch_size) / elapsed
print(f"Inference speed: {samples_per_second:,.0f} samples/second")
print(f"Time per full skeleton (24 joints): {24 / samples_per_second * 1000:.3f} ms")
print(f"Achievable FPS for single actor: {samples_per_second / 24:.0f} FPS")
```

### Task 6.2: Memory Usage Analysis

```python
# memory_analysis.py
import torch
from model import create_model

def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1024**3  # GB

print(f"Initial GPU memory: {get_gpu_memory():.3f} GB")

model = create_model('base', num_joints=24, num_actors=100).cuda()
print(f"After model load: {get_gpu_memory():.3f} GB")

# Test different batch sizes
for batch_size in [1024, 2048, 4096, 8192, 16384, 32768]:
    try:
        torch.cuda.empty_cache()
        actor_ids = torch.randint(0, 100, (batch_size,)).cuda()
        joint_idxs = torch.randint(0, 24, (batch_size,)).cuda()
        times = torch.rand(batch_size).cuda()

        output = model(actor_ids, joint_idxs, times)
        loss = output.mean()
        loss.backward()

        print(f"Batch size {batch_size}: {get_gpu_memory():.3f} GB - ✓")
        del actor_ids, joint_idxs, times, output, loss
    except RuntimeError as e:
        print(f"Batch size {batch_size}: OOM - ✗")
        break

# RTX 3090 (24GB) should handle batch_size up to ~16384 for training
```

---

## PHASE 7: Sync Results Back to Local

After completing experiments:

```bash
# On remote machine, compress results
cd /home/o/07-geom-inr-motion__project-space/geom-inr-motion
tar -czvf rtx3090_results.tar.gz \
    checkpoints_rtx3090/ \
    *.png \
    *.bvh \
    training_rtx3090.log

# From local Mac, download results
scp REDACTED_SERVER:/home/o/07-geom-inr-motion__project-space/geom-inr-motion/rtx3090_results.tar.gz \
    /Users/lekanadeyeri/Dev/PINN-Experiments/projects/07-geom-inr-motion__project-space/geom-inr-motion/

# Extract
cd /Users/lekanadeyeri/Dev/PINN-Experiments/projects/07-geom-inr-motion__project-space/geom-inr-motion
tar -xzvf rtx3090_results.tar.gz
```

---

## SUCCESS CRITERIA

1. **Minimal test passes**: Loss < 0.01 after 100 steps
2. **Full training completes**: 500 epochs without errors
3. **MPJPE comparable to VastAI run**: Within 10% of 6.47mm
4. **Export works**: BVH file generated and valid
5. **Inference speed**: >100k samples/second on RTX 3090
6. **Memory efficient**: Training with batch_size >= 4096

---

## TROUBLESHOOTING

**CUDA out of memory:**
- Reduce batch_size by 50%
- Use `--model small` variant
- Enable gradient checkpointing (modify model.py)

**NumPy compatibility:**
- `pip install numpy<2.0`

**ModuleNotFoundError:**
- Ensure correct conda environment activated
- `pip install -r requirements.txt`

**SSH connection issues:**
- Verify IP address: `ping 192.168.86.152`
- Check if machine is awake

---

## EXECUTION CHECKLIST

- [ ] SSH into RTX 3090 machine
- [ ] Create conda environment
- [ ] Install PyTorch + dependencies
- [ ] Run minimal_test.py successfully
- [ ] Train model (500 epochs)
- [ ] Compare metrics to VastAI baseline
- [ ] Run interpolation quality test
- [ ] Export BVH animation
- [ ] Generate visualization plots
- [ ] Benchmark inference speed
- [ ] Sync results to local machine

---

**Note:** This project is 80-85% complete. The RTX 3090 validation will confirm reproducibility and establish a local development baseline. Future work includes real AMASS data validation and Blender plugin completion.
