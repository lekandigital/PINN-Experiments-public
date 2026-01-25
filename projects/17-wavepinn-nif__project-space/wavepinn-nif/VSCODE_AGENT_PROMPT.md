# VS Code Agent Prompt: WavePINN Training & Blender Export

## Context

I have the WavePINN project at GitHub: `https://github.com/lekandigital/wavepinn_nif_scalar`

I have SSH access to a Vast.ai GPU server:
- **SSH**: `ssh -i ~/.ssh/key1 -p 30409 root@118.163.199.123`
- **GPU**: NVIDIA L40S (48GB VRAM)
- **Pre-installed**: PyTorch with CUDA

The training was already completed successfully with these results:
- Initial loss: 4.09e+02
- Final loss: 4.05e-05
- Reduction: 100%
- Training time: 134 seconds

## Task: Export Wavefield for Blender

Please do the following:

### Step 1: Pull latest code to server

```bash
ssh -i ~/.ssh/key1 -p 30409 root@118.163.199.123 "cd /root/wavepinn_nif_scalar && git pull"
```

If the repo doesn't exist, clone it:
```bash
ssh -i ~/.ssh/key1 -p 30409 root@118.163.199.123 "cd /root && git clone https://github.com/lekandigital/wavepinn_nif_scalar.git"
```

### Step 2: Run Blender export on GPU

```bash
ssh -i ~/.ssh/key1 -p 30409 root@118.163.199.123 "
cd /root/wavepinn_nif_scalar
pip3 install pillow -q
python3 src/export_blender.py --output blender_export --frames 60 --resolution 128
"
```

### Step 3: Download export to local machine

```bash
scp -i ~/.ssh/key1 -P 30409 -r root@118.163.199.123:/root/wavepinn_nif_scalar/blender_export ~/Downloads/
```

### Step 4: Update Blender import script

1. Open `blender/import_animation.py`
2. Change `EXPORT_PATH` to: `"/Users/YOUR_USERNAME/Downloads/blender_export"`
3. Copy the script to clipboard

### Step 5: Import into Blender

1. Open Blender
2. Go to Scripting workspace
3. Click New → paste the script
4. Press Alt+P to run
5. Press Space to play animation

## Alternative: Train from scratch

If you want to re-train the model:

```bash
ssh -i ~/.ssh/key1 -p 30409 root@118.163.199.123 "
cd /root/wavepinn_nif_scalar
python3 src/train_v2.py --epochs 3000 --output wavepinn.pt
python3 src/export_blender.py --model wavepinn.pt --output blender_export
"
```

Then download as in Step 3.

## Expected Output

```
~/Downloads/blender_export/
├── images/
│   ├── wave_0000.png
│   ├── wave_0001.png
│   └── ... (60 frames)
└── metadata.json
```

The animation shows acoustic wave propagation through a heterogeneous velocity field, learned by a physics-informed neural network.
