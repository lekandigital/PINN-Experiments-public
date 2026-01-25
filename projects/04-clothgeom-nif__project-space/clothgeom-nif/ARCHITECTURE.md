# ClothGeom-NIF Architecture

Technical documentation for the Neural Implicit Field decoder architecture,
training procedure, and design decisions.

## Overview

ClothGeom-NIF maps latent cloth states to signed distance fields (SDFs) using
a SIREN-based neural network. The architecture is designed for:

1. **High-frequency detail capture**: Wrinkles, folds, fine cloth features
2. **Smooth latent interpolation**: Continuous transitions between cloth states
3. **Uncertainty estimation**: Variance field for LOD blending
4. **Real-time inference**: Efficient mesh extraction

## Model Architecture

### Input Representation

```
Input = [coordinates, latent_code]
      = [x, y, z, l₁, l₂, ..., l₁₂₈]
      
Total input dimension: 3 + 128 = 131
```

**Coordinates (3D)**:
- Normalized to [-1, 1]³
- Represent query points in 3D space

**Latent Code (128D)**:
- Encodes cloth state (node positions + edge strains)
- L2-normalized to lie on unit hypersphere
- Created by random projection of cloth features

### Network Layers

```
Layer 0 (First):   131 → 256, ω₀ = 30.0, special init
Layer 1-3:         256 → 256, ω₀ = 30.0, standard init
Layer 4 (Skip):    256+131 → 256, skip connection from input
Layer 5-7:         256 → 256, ω₀ = 30.0, standard init

SDF Head:          256 → 128 → 1 (linear output)
Variance Head:     256 → 128 → 1 (softplus output)
```

### SIREN Activation Details

**Sine activation**:
```python
f(x) = sin(ω₀ · x)
```

**Why ω₀ = 30?**
- Higher ω₀ = higher frequency capacity
- ω₀ = 30 allows learning of features at cloth wrinkle scale
- Lower ω₀ would blur fine details
- Higher ω₀ could cause training instability

### Weight Initialization

Critical for SIREN networks to work properly.

**First layer** (coordinate input):
```python
w ~ U(-1/fan_in, 1/fan_in)
# For fan_in = 131: w ~ U(-0.0076, 0.0076)
```

This ensures the input to the sine function spans [-π, π].

**Hidden layers**:
```python
w ~ U(-√(6/fan_in)/ω₀, √(6/fan_in)/ω₀)
# For fan_in = 256, ω₀ = 30: w ~ U(-0.0051, 0.0051)
```

This maintains the variance of activations through the network.

**Why this matters**:
- Standard Xavier/He init would cause exploding activations
- SIREN init keeps sin() arguments in reasonable range
- Wrong init = network outputs random noise, never converges

### Skip Connection

**Location**: After layer 4 (middle of network)

**Purpose**:
- Helps gradient flow to early layers
- Allows combining low-level and high-level features
- Input is concatenated, not added

**Implementation**:
```python
if layer_idx == 4:
    x = torch.cat([x, input], dim=-1)  # 256 + 131 = 387
    # Layer 4 has 387 → 256 mapping
```

### Dual Output Heads

**SDF Head**:
- Predicts signed distance to cloth surface
- Negative inside, positive outside
- Zero at the surface
- Linear output (unbounded)

**Variance Head**:
- Predicts uncertainty/confidence
- Used for LOD blending (high variance = coarser mesh OK)
- Softplus activation (ensures positive output)

## Latent Code Construction

### From Cloth Simulation

Given a cloth state with N nodes and E edges:

```
nodes: [N, 3]     # 3D positions
strains: [E, 1]   # Edge strain values
```

**Latent construction**:
```python
# 1. Flatten
nodes_flat = nodes.flatten()        # [N*3]
strains_flat = strains.flatten()    # [E]

# 2. Concatenate
features = concat([nodes_flat, strains_flat])  # [N*3 + E]

# 3. Random projection (fixed matrix)
projection = random_matrix(len(features), latent_dim)
latent = features @ projection

# 4. Normalize
latent = latent / norm(latent)
```

**Why random projection?**
- Dimensionality reduction from ~4000D to 128D
- Preserves relative distances (Johnson-Lindenstrauss)
- Fixed projection = consistent encoding

### Latent Interpolation

For smooth transitions between cloth states:

```python
# Linear interpolation (with renormalization)
latent_interp = (1 - t) * latent1 + t * latent2
latent_interp = latent_interp / norm(latent_interp)
```

This traces a geodesic on the unit hypersphere.

## Training

### Loss Function

**Combined loss**:
```
L = λ_sdf · L_sdf + λ_eik · L_eikonal + λ_var · L_variance
```

**SDF Loss** (λ = 1.0):
```python
L_sdf = mean((pred_sdf - gt_sdf)²)
```

Surface-weighted variant (higher weight near surface):
```python
weight = 1 + (surface_weight - 1) * exp(-10 * |gt_sdf|)
L_sdf = mean(weight * (pred_sdf - gt_sdf)²)
```

**Eikonal Loss** (λ = 0.1):
```python
grad = gradient(pred_sdf, coords)
L_eikonal = mean((|grad| - 1)²)
```

Why Eikonal?
- Valid SDFs have unit gradient magnitude
- Regularizes the SDF to be smooth
- Improves mesh quality

**Variance Loss** (λ = 0.01):
```python
L_variance = mean(pred_variance)
```

Why variance regularization?
- Prevents variance from exploding
- Encourages confident predictions
- Low weight to allow natural uncertainty

### Point Sampling

**Uniform sampling** (50%):
- Random points in [-1.5, 1.5]³
- Covers entire volume

**Surface sampling** (50%):
- Importance sampling near SDF = 0
- Higher density where geometry exists
- Critical for learning surface details

### Training Procedure

```python
for epoch in range(100):
    for batch in dataloader:
        # Flatten batch
        coords = batch['coords'].reshape(-1, 3)  # [B*N, 3]
        latent = expand(batch['latent'], N)      # [B*N, 128]
        gt_sdf = batch['sdf'].reshape(-1, 1)     # [B*N, 1]
        
        # Forward with gradient tracking
        coords.requires_grad_(True)
        pred_sdf, pred_var = model(coords, latent)
        
        # Compute gradient for Eikonal
        gradient = autograd.grad(pred_sdf, coords, ...)
        
        # Compute loss
        loss = criterion(pred_sdf, gt_sdf, pred_var, gradient)
        
        # Backward + optimize
        loss.backward()
        optimizer.step()
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-4 | Adam optimizer |
| Batch size | 8 | Cloth states per batch |
| Points/state | 8192 | Sampled points |
| Epochs | 100 | With early stopping |
| LR schedule | Cosine | Anneals to 1e-6 |
| Gradient clip | 1.0 | Prevents explosions |

### Mixed Precision

FP16 training reduces memory by ~40%:

```python
scaler = GradScaler()

with autocast():
    pred_sdf, pred_var = model(coords, latent)
    loss = criterion(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Mesh Extraction

### Marching Cubes Algorithm

**Input**: SDF volume [R, R, R]
**Output**: Triangle mesh (vertices, faces)

**Steps**:
1. Evaluate SDF at regular grid points
2. For each cube in grid:
   - Determine which corners are inside (SDF < 0)
   - Look up triangle configuration in table
   - Interpolate vertex positions on edges
3. Collect all triangles into mesh

**Resolution vs Quality**:

| Resolution | Vertices | Faces | Time |
|------------|----------|-------|------|
| 64³ | ~5k | ~10k | ~1s |
| 128³ | ~20k | ~40k | ~3s |
| 256³ | ~80k | ~160k | ~15s |

### Post-Processing

**Laplacian Smoothing**:
- Removes marching cubes staircase artifacts
- Preserves overall shape
- 1-3 iterations recommended

**Mesh Simplification**:
- Quadric decimation to reduce polygon count
- For real-time rendering (target: 10k faces)

## Design Decisions

### Why SIREN over ReLU?

| Aspect | SIREN | ReLU |
|--------|-------|------|
| High-frequency | Excellent | Poor |
| Initialization | Critical | Standard |
| Training stability | Good | Better |
| Gradient quality | Smooth | Non-smooth |

For cloth with fine wrinkles, SIREN's frequency capacity is essential.

### Why SDF over Occupancy?

| Aspect | SDF | Occupancy |
|--------|-----|-----------|
| Mesh quality | Smooth | Blocky |
| Gradient info | Surface normal | None |
| Training signal | Dense | Sparse |
| Eikonal constraint | Yes | N/A |

SDF provides richer geometric information.

### Why Concatenation over FiLM?

**FiLM (Feature-wise Linear Modulation)**:
```python
h = gamma(latent) * h + beta(latent)
```

**Concatenation**:
```python
h = layer(concat([coords, latent]))
```

We use concatenation because:
- Simpler implementation
- Works well for moderate latent dimensions
- FiLM adds complexity without clear benefit for 128D latent

### Why Skip Connection?

Without skip: gradient must flow through 8 layers
With skip: gradient has direct path to input

For deep SIREN networks, skip connections:
- Improve training stability
- Help early layers learn faster
- Enable richer feature combinations

## Performance Considerations

### Memory Usage

| Component | Size (FP32) | Size (FP16) |
|-----------|-------------|-------------|
| Model params | ~2.5 MB | ~1.25 MB |
| Batch (8192 pts) | ~300 MB | ~150 MB |
| SDF grid (128³) | ~8 MB | ~4 MB |

### Inference Speed

Optimizations applied:
- Batched evaluation (65536 points/batch)
- Mixed precision inference
- No gradient tracking (torch.no_grad)

Potential future optimizations:
- tinycudann hash encoding
- Voxel caching
- CUDA kernels for marching cubes

## Future Directions

1. **Hash Encoding**: Instant-NGP style for 10x speedup
2. **Adaptive Resolution**: Variable SDF grid density
3. **Temporal Coherence**: Animation-aware training
4. **Inverse Design**: Optimize latent for target silhouette
5. **Real-time Unity/Omniverse**: Native plugin with ONNX export
