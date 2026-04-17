# HGNN-NIF Cloth Demo — run commands

All paths are relative to `hgnn-nif-cloth/`. Python needs `torch`, `taichi>=1.7`, `scipy`, `numpy`, `imageio`, `pillow` (only `python3.12` has torch installed locally).

## 1. Data + model (already done, checkpoint present)

```bash
# 40x40 mass-spring physics sequence (writes outputs/physics_baseline/)
python3.12 scripts/gen_physics_data.py --frames 600 --resolution 40

# Train TemporalHGNN_NIF on the physics rollout (run on the 3090 Ti box).
# Already produced checkpoints/hgnn_nif_cloth_trained.pt (best val_loss).
python3.12 scripts/train_dynamics.py \
  --data outputs/physics_baseline --history 3 --rollout_k 4 \
  --noise_std 0.003 --epochs 400 --batch 16 --lr 1e-3

# Autoregressive rollout + metrics + mesh_info.json
python3.12 scripts/rollout_and_metrics.py --rmse_threshold 0.05
```

Outputs used by the viewer:
- `outputs/physics_baseline/cloth_sequence.npy` (600, 1600, 3)
- `outputs/hgnn_nif_prediction/cloth_sequence.npy`
- `outputs/mesh_info.json` — mesh topology + trustworthy rollout window

Trustworthy window (RMSE < 0.05): **9 frames (0.30 s)** — the autoregressive model tracks physics cleanly for the first third of a second, then drifts.

## 2. Viewer — interactive

```bash
python3.12 src/taichi_cloth_demo.py                       # pitch preset (default)
python3.12 src/taichi_cloth_demo.py --preset research     # honest 40x40 wireframe
python3.12 src/taichi_cloth_demo.py --preset dramatic     # cinematic close 3/4
python3.12 src/taichi_cloth_demo.py --trust-only          # clamp to trust window
```

Keybinds: `space` play/pause · `r` reset · `m` toggle physics/HGNN · `w` wireframe · `[/]` cycle presets · RMB drag to orbit.

## 3. Smoke + camera QA (outputs to `artifacts/`)

```bash
# Single frame, default preset=pitch, mode=hgnn
python3.12 src/taichi_cloth_demo.py --smoke --frame 60

# Render as physics instead
python3.12 src/taichi_cloth_demo.py --smoke --frame 60 --mode physics \
  --out artifacts/smoke_physics.png

# 3 candidate camera angles for the pitch preset
python3.12 src/taichi_cloth_demo.py --camera-test --frame 80
```

## 4. Preset stills (6 PNGs)

```bash
for p in research pitch dramatic; do
  for m in hgnn physics; do
    python3.12 src/taichi_cloth_demo.py --smoke --preset $p \
      --frame 60 --mode $m --out artifacts/${p}_${m}.png
  done
done
```

## 5. Animation export

```bash
# Pitch preset, 120 frames each (physics + HGNN)
python3.12 src/taichi_cloth_demo.py --export-preset pitch --anim-frames 120

# All presets, 120 frames each
python3.12 src/taichi_cloth_demo.py --export --anim-frames 120
```

Writes one PNG per frame under `artifacts/<preset>_hgnn/` and `artifacts/<preset>_physics/`.

## 6. Encode MP4s

```bash
cd artifacts
ffmpeg -y -framerate 30 -i pitch_physics/f_%04d.png -c:v libx264 -pix_fmt yuv420p pitch_physics.mp4
ffmpeg -y -framerate 30 -i pitch_hgnn/f_%04d.png     -c:v libx264 -pix_fmt yuv420p pitch_hgnn.mp4
# Side-by-side compare
ffmpeg -y -framerate 30 -i pitch_physics/f_%04d.png -i pitch_hgnn/f_%04d.png \
  -filter_complex "[0:v][1:v]hstack=inputs=2[out]" -map "[out]" \
  -c:v libx264 -pix_fmt yuv420p pitch_compare.mp4
```

## Preset summary

| preset   | upsample | wireframe | camera                          | use case                 |
|----------|----------|-----------|---------------------------------|--------------------------|
| research | 1 (raw)  | yes       | hero high, fov 44               | honest 40x40 inspection  |
| pitch    | 4 (cubic→blur) | no | hero high, fov 44               | presentation-quality     |
| dramatic | 5 (cubic→blur) | no | 3/4 low, fov 44                 | cinematic close-up       |

All presets share a dark studio background (RGB ~0.03), warm key + cool fill lighting, and a height-based diverging colormap (warm = sag, cool = high).
