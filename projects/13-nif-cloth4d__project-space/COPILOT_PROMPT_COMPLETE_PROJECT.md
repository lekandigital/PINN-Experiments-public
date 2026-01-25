# Comprehensive Prompt: Complete NIF-Cloth4D on RTX 3090

## Mission: Execute and Validate the NIF-Cloth4D Neural Implicit Cloth Simulator

You are tasked with **completing** the NIF-Cloth4D project by actually running the training pipeline on an RTX 3090 Ti. The code is fully implemented but has never been executed. Your job is to run it, validate it works, and produce real results.

## Remote Machine Details

```
Host: 192.168.86.152
User: o
Password: REDACTED_PASSWORD
GPU: NVIDIA GeForce RTX 3090 Ti (24GB VRAM)
CUDA: 13.0
Driver: 580.82.09
OS: Ubuntu 22.04.5 LTS
```

## Project Location (Local Mac)

```
/Users/lekanadeyeri/Dev/PINN-Experiments/projects/13-nif-cloth4d__project-space/nif-cloth4d/
```

## Your Tasks

### Phase 1: Deploy to Remote Machine

1. **Create project directory on remote**
   ```bash
   ssh REDACTED_SERVER
   mkdir -p ~/projects/nif-cloth4d
   exit
   ```

2. **Copy all files from local to remote**
   ```bash
   scp -r /Users/lekanadeyeri/Dev/PINN-Experiments/projects/13-nif-cloth4d__project-space/nif-cloth4d/* REDACTED_SERVER:~/projects/nif-cloth4d/
   ```

3. **SSH into remote and install dependencies**
   ```bash
   ssh REDACTED_SERVER
   cd ~/projects/nif-cloth4d

   # Check Python and CUDA
   python3 --version
   nvidia-smi

   # Install dependencies
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip3 install numpy h5py matplotlib PyMCubes trimesh
   pip3 install tensorboard

   # Optional USD support (may fail, that's OK)
   pip3 install usd-core || echo "USD not available, will export OBJ only"
   ```

### Phase 2: Verify Code Works

1. **Test imports**
   ```bash
   cd ~/projects/nif-cloth4d
   python3 -c "
   import torch
   print(f'PyTorch: {torch.__version__}')
   print(f'CUDA available: {torch.cuda.is_available()}')
   print(f'GPU: {torch.cuda.get_device_name(0)}')

   from nif_cloth4d import FourierFeatureSIREN, create_model
   config = {'in_dim': 4, 'cond_dim': 0, 'hidden_dim': 128, 'hidden_layers': 4, 'w0_initial': 30.0}
   model = create_model(config)
   print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

   # Test forward pass on GPU
   model = model.cuda()
   x = torch.randn(1024, 4).cuda()
   y = model(x)
   print(f'Forward pass: {x.shape} -> {y.shape}')
   print('All tests passed!')
   "
   ```

2. **Test data generation**
   ```bash
   python3 synthetic_data.py --output_dir /tmp/cloth_test_data --num_frames 5 --grid_size 64 --data_type falling
   ls -la /tmp/cloth_test_data/sdf/
   ```

### Phase 3: Run Full Training Pipeline

1. **Option A: Use the automated script**
   ```bash
   chmod +x run_pipeline.sh
   ./run_pipeline.sh /tmp/cloth_test_data ./output
   ```

2. **Option B: Run step by step (recommended for debugging)**

   ```bash
   # Step 1: Generate data
   python3 synthetic_data.py \
       --output_dir /tmp/cloth_test_data \
       --num_frames 10 \
       --grid_size 64 \
       --data_type falling

   # Step 2: Train (watch GPU memory with nvidia-smi in another terminal)
   python3 train_nif_cloth4d.py \
       --config config_test.yaml \
       --data_dir /tmp/cloth_test_data \
       --output_dir ./checkpoints

   # Step 3: Evaluate
   python3 test_evaluation.py \
       --checkpoint ./checkpoints/model_best.pt \
       --data_dir /tmp/cloth_test_data \
       --output_dir ./evaluation \
       --resolution 64 \
       --times "0.0,0.5,1.0"

   # Step 4: Export meshes
   python3 export_mesh.py \
       --checkpoint ./checkpoints/model_best.pt \
       --output ./exports \
       --times "0.0,0.5,1.0" \
       --resolution 64 \
       --formats "obj"
   ```

3. **Monitor training** (in a second SSH terminal)
   ```bash
   watch -n 2 nvidia-smi
   ```

### Phase 4: Validate Results

1. **Check outputs exist**
   ```bash
   echo "=== Checkpoints ==="
   ls -lh ./checkpoints/*.pt

   echo "=== Evaluation ==="
   ls -lh ./evaluation/

   echo "=== Exports ==="
   ls -lh ./exports/
   ```

2. **Extract key metrics**
   ```bash
   # Print final training loss
   cat ./checkpoints/training_history.txt | tail -5

   # Print evaluation metrics
   cat ./evaluation/metrics.txt
   ```

3. **Verify mesh files are valid**
   ```bash
   # Check OBJ files have content
   for f in ./exports/*.obj; do
       echo "$f: $(wc -l < $f) lines, $(head -1 $f)"
   done
   ```

### Phase 5: Copy Results Back to Local Mac

```bash
# From the local Mac terminal:
mkdir -p /Users/lekanadeyeri/Dev/PINN-Experiments/projects/13-nif-cloth4d__project-space/nif-cloth4d/checkpoints
mkdir -p /Users/lekanadeyeri/Dev/PINN-Experiments/projects/13-nif-cloth4d__project-space/nif-cloth4d/evaluation
mkdir -p /Users/lekanadeyeri/Dev/PINN-Experiments/projects/13-nif-cloth4d__project-space/nif-cloth4d/exports

scp -r REDACTED_SERVER:~/projects/nif-cloth4d/checkpoints/* \
    /Users/lekanadeyeri/Dev/PINN-Experiments/projects/13-nif-cloth4d__project-space/nif-cloth4d/checkpoints/

scp -r REDACTED_SERVER:~/projects/nif-cloth4d/evaluation/* \
    /Users/lekanadeyeri/Dev/PINN-Experiments/projects/13-nif-cloth4d__project-space/nif-cloth4d/evaluation/

scp -r REDACTED_SERVER:~/projects/nif-cloth4d/exports/* \
    /Users/lekanadeyeri/Dev/PINN-Experiments/projects/13-nif-cloth4d__project-space/nif-cloth4d/exports/
```

### Phase 6: Generate Test Report

Create `TEST_REPORT.md` in the project directory with:

```markdown
# NIF-Cloth4D Test Report

## Date: [YYYY-MM-DD]

## 1. Environment Setup

- **GPU**: NVIDIA RTX 3090 Ti (24GB VRAM)
- **CUDA**: 13.0
- **PyTorch**: [version]
- **Python**: [version]
- **Host**: 192.168.86.152

## 2. Data Generation

- **Type**: Falling cloth (synthetic SDF)
- **Frames**: 10
- **Resolution**: 64x64x64
- **Files generated**: [list HDF5 files]
- **Total size**: [X MB]

## 3. Training Results

- **Epochs**: 50
- **Final training loss**: [value]
- **Final validation loss**: [value]
- **Training time**: [X minutes]
- **Peak GPU memory**: [X GB]

### Loss Curve
[Include loss curve plot or describe trend]

## 4. Evaluation Metrics

| Metric | t=0.0 | t=0.5 | t=1.0 |
|--------|-------|-------|-------|
| Chamfer Distance | | | |
| Hausdorff Distance | | | |

**Target**: Chamfer < 0.05

## 5. Export Verification

| File | Format | Size | Vertices | Valid |
|------|--------|------|----------|-------|
| mesh_t0.0.obj | OBJ | | | |
| mesh_t0.5.obj | OBJ | | | |
| mesh_t1.0.obj | OBJ | | | |

## 6. Conclusion

- [ ] Training completed without errors
- [ ] Loss decreased to < 0.01
- [ ] Chamfer distance < 0.05
- [ ] Meshes exported successfully
- [ ] All results copied to local machine

## 7. Issues Encountered

[Document any errors and how they were resolved]
```

## Success Criteria

- [ ] SSH connection to 192.168.86.152 works
- [ ] PyTorch detects CUDA and RTX 3090 Ti
- [ ] Data generation produces 10 HDF5 files
- [ ] Training runs for 50 epochs without OOM
- [ ] Final loss < 0.01
- [ ] Chamfer distance < 0.05
- [ ] At least 3 mesh files exported
- [ ] All results copied back to local Mac
- [ ] TEST_REPORT.md completed with actual values

## Troubleshooting

### "CUDA out of memory"
Reduce batch_size in config_test.yaml from 8192 to 4096 or 2048:
```yaml
training:
  batch_size: 4096  # Reduced for 24GB GPU
```

### "No module named X"
```bash
pip3 install X
```

### Training is slow
The RTX 3090 Ti should train 50 epochs in ~10-15 minutes. If slower:
- Check no other processes using GPU: `nvidia-smi`
- Ensure CUDA is being used: model should show "cuda:0"

### Can't SSH
```bash
# Test connection
ping 192.168.86.152

# If firewall issue, from remote:
sudo ufw allow 22
```

## Expected Timeline

| Step | Duration |
|------|----------|
| Deploy & install | 5-10 min |
| Data generation | 1-2 min |
| Training (50 epochs) | 10-15 min |
| Evaluation | 2-3 min |
| Export | 1 min |
| Copy back | 1-2 min |
| **Total** | **~25-35 min** |

## Output Deliverables

After completion, the project should contain:

```
nif-cloth4d/
├── checkpoints/
│   ├── model_best.pt          # Best trained model
│   ├── model_epoch_50.pt      # Final checkpoint
│   └── training_history.npz   # Loss history
├── evaluation/
│   ├── metrics.txt            # Chamfer/Hausdorff scores
│   ├── loss_curve.png         # Training visualization
│   └── mesh_comparison.png    # GT vs predicted
├── exports/
│   ├── mesh_t0.0.obj          # Mesh at t=0
│   ├── mesh_t0.5.obj          # Mesh at t=0.5
│   └── mesh_t1.0.obj          # Mesh at t=1.0
└── TEST_REPORT.md             # Filled report
```

## Important Notes

1. **No Vast.ai needed** - Your RTX 3090 Ti (24GB) is sufficient
2. **Local files are source of truth** - Always copy FROM Mac TO remote, then results back
3. **Don't modify remote code** - Make changes locally and re-sync
4. **Check GPU memory** - Use `nvidia-smi` to monitor during training
5. **Save before destroying** - Always copy results back before cleaning up

Begin by SSH'ing into the remote machine and checking GPU availability.
