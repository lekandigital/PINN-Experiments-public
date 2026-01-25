"""Final verification of Laplace-Beltrami on sphere"""
import torch
torch.set_printoptions(precision=6)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sample sphere points (Fibonacci)
n_test = 100
indices = torch.arange(n_test, dtype=torch.float32, device=device) + 0.5
phi = torch.acos(1 - 2 * indices / n_test)
theta = 3.14159 * (1 + 5**0.5) * indices

x = torch.sin(phi) * torch.cos(theta)
y = torch.sin(phi) * torch.sin(theta)
z = torch.cos(phi)

points = torch.stack([x, y, z], dim=-1).requires_grad_(True)
n = points  # normal = position on unit sphere

# u = xy
u = (points[:, 0] * points[:, 1]).unsqueeze(-1)

# Gradient
grad_u = torch.autograd.grad(u, points, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

# Euclidean Laplacian
laplacian_euclidean = torch.zeros_like(u)
for i in range(3):
    grad_u_i = grad_u[:, i:i+1]
    grad2_u_i = torch.autograd.grad(grad_u_i, points, grad_outputs=torch.ones_like(grad_u_i), create_graph=True, retain_graph=True)[0]
    laplacian_euclidean += grad2_u_i[:, i:i+1]

# n·∇u (radial derivative)
n_dot_grad_u = torch.sum(n * grad_u, dim=1, keepdim=True)

# ∇(n·∇u)
grad_n_dot_grad_u = torch.autograd.grad(n_dot_grad_u, points, grad_outputs=torch.ones_like(n_dot_grad_u), create_graph=True, retain_graph=True)[0]

# n·∇(n·∇u) 
n_dot_grad_n_dot_grad_u = torch.sum(n * grad_n_dot_grad_u, dim=1, keepdim=True)

# For u = xy:
# grad(u) = (y, x, 0)
# n·grad(u) = xy + yx + 0 = 2xy
# This means the radial derivative ∂u/∂r at r=1 is 2xy

# For a homogeneous polynomial of degree k: u(r*x) = r^k u(x)
# So ∂u/∂r = k*u/r, i.e., n·∇u = k*u at r=1
# For xy, k=2, so n·∇u = 2*xy ✓

# Also ∂²u/∂r² = k(k-1)*u/r² = 2*xy at r=1
# So the radial second derivative should be 2xy

# From the debug: n·∇(n·∇u) - n·∇u = -0.128 for point with u = -0.064
# That's -2*u = -2*(-0.064) = 0.128, but we got -0.128
# Hmm, sign issue!

# Actually: for homogeneous polynomial of degree k, the extension from sphere to R³ is:
# U(x) = |x|^k * u(x/|x|) where u is on the sphere
# The spherical Laplacian relates to the Euclidean by:
# Δ_S u = Δ U|_{r=1} - ∂²U/∂r²|_{r=1} - 2∂U/∂r|_{r=1}
# For U = r^k f(θ,φ), ∂U/∂r = kr^{k-1}f, ∂²U/∂r² = k(k-1)r^{k-2}f
# At r=1: ∂U/∂r = kf = ku, ∂²U/∂r² = k(k-1)f = k(k-1)u

# BUT! xy is not r^2 * f(θ,φ). It's just xy directly.
# xy = r² sin²θ sinφ cosφ (in spherical coords r,θ,φ)
# So xy = r² * (some angular function)
# At r=1, xy = angular function

# The key is: the Euclidean Laplacian of xy is 0.
# But the radial derivatives are: ∂(xy)/∂r = ? 

# Let's think differently. The standard formula for the sphere is:
# Δ_S = Δ - (n·∇)² - 2(n·∇)
# where n·∇ is the radial derivative operator

# Let's compute (n·∇)²u directly:
# (n·∇)u = n·grad(u) = 2xy for u=xy
# (n·∇)²u = n·grad(n·grad(u)) = n·grad(2xy) = 2*n·grad(xy) = 2*2xy = 4xy

# So: Δ_S(xy) = 0 - 4xy - 2(2xy) = -4xy - 4xy = -8xy

# But the spherical harmonic eigenvalue says -6xy. Something's off.

# Wait, the formula Δ_S = Δ - (n·∇)² - 2(n·∇) is for radius 1.
# Let me double-check with explicit computation.

# For xy on sphere using spherical coords (θ, φ):
# x = sinθ cosφ, y = sinθ sinφ, z = cosθ
# xy = sin²θ cosφ sinφ = sin²θ sin(2φ)/2

# Δ_S = (1/sinθ) ∂/∂θ(sinθ ∂/∂θ) + (1/sin²θ) ∂²/∂φ²

# Let f(θ,φ) = sin²θ sin(2φ)/2
# ∂f/∂φ = sin²θ cos(2φ)
# ∂²f/∂φ² = -2 sin²θ sin(2φ) = -4 * sin²θ sin(2φ)/2 = -4f

# ∂f/∂θ = sinθ cosθ sin(2φ) = sin(2θ)/2 * sin(2φ)
# sinθ ∂f/∂θ = sin²θ cosθ sin(2φ)
# ∂/∂θ(sinθ ∂f/∂θ) = 2sinθ cosθ cosθ sin(2φ) - sin³θ sin(2φ)
#                   = sinθ sin(2φ)(2cos²θ - sin²θ)
#                   = sinθ sin(2φ)(2cos²θ - (1-cos²θ))
#                   = sinθ sin(2φ)(3cos²θ - 1)

# (1/sinθ) * above = sin(2φ)(3cos²θ - 1)

# Hmm this is getting complicated. Let me just trust the numerical result.

# The numerical result shows:
# n·∇(n·∇u) = 2*(n·∇u) exactly (since n·∇(n·∇u) - n·∇u = -u and n·∇u = 2u)
# So n·∇(n·∇u) = n·∇u + (-u) = 2u - u = u... no wait

# From output: n·∇(n·∇u) = -0.256684 for u = -0.064171, n·∇u = -0.128342
# -0.256684 = 2 * (-0.128342) ✓
# So n·∇(n·∇u) = 2 * n·∇u = 4u

# Therefore Δ_S = 0 - 4u - 2(2u) = -4u - 4u = -8u? But eigenvalue is -6u.

# THE ISSUE: n·∇(n·∇u) ≠ (n·∇)²u when n varies with position!
# n·∇(n·∇u) = n·∇(scalar) where scalar = n·∇u
# = n_i ∂/∂x_i (n_j ∂u/∂x_j)
# = n_i (∂n_j/∂x_i * ∂u/∂x_j + n_j ∂²u/∂x_i∂x_j)
# = n_i ∂n_j/∂x_i * ∂u/∂x_j + n_i n_j ∂²u/∂x_i∂x_j

# On sphere, n = x, so ∂n_j/∂x_i = δ_ij
# First term: n_i δ_ij ∂u/∂x_j = n_j ∂u/∂x_j = n·∇u
# Second term: n_i n_j ∂²u/∂x_i∂x_j = n^T Hessian(u) n

# So: n·∇(n·∇u) = n·∇u + n^T H n

# For u = xy: Hessian = [[0,1,0],[1,0,0],[0,0,0]]
# n^T H n = 2 n_1 n_2 = 2xy = 2u

# Therefore: n·∇(n·∇u) = 2u + 2u = 4u ✓ (matches numerical)

# Now the TRUE (n·∇)² operator:
# (n·∇)²u = n·∇(n·∇u) where we treat n as a FIXED vector at that point
# = n_i n_j ∂²u/∂x_i∂x_j = n^T H n = 2xy = 2u

# So the CORRECT formula is:
# Δ_S u = Δu - n^T H n - 2(n·∇u)

# Compute Hessian numerically:
def compute_hessian_term(u, points):
    """Compute n^T @ Hessian(u) @ n"""
    grad_u = torch.autograd.grad(u, points, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    n = points
    hessian_n = torch.zeros_like(u)
    for i in range(3):
        grad_u_i = grad_u[:, i:i+1]
        grad2_u = torch.autograd.grad(grad_u_i, points, grad_outputs=torch.ones_like(grad_u_i), create_graph=True, retain_graph=True)[0]
        for j in range(3):
            hessian_n += n[:, i:i+1] * n[:, j:j+1] * grad2_u[:, j:j+1]
    return hessian_n

# Recompute
points2 = torch.stack([x, y, z], dim=-1).requires_grad_(True)
u2 = (points2[:, 0] * points2[:, 1]).unsqueeze(-1)

# Method: compute n^T H n directly
grad_u2 = torch.autograd.grad(u2, points2, grad_outputs=torch.ones_like(u2), create_graph=True, retain_graph=True)[0]

# Compute each Hessian element and accumulate n^T H n
n2 = points2
nTHn = torch.zeros_like(u2)
for i in range(3):
    for j in range(3):
        # H_ij = ∂²u/∂x_i∂x_j
        if i == 0 and j == 0:
            grad_xi = grad_u2[:, 0:1]  # ∂u/∂x
            H_ij = torch.autograd.grad(grad_xi, points2, grad_outputs=torch.ones_like(grad_xi), create_graph=True, retain_graph=True)[0][:, 0:1]
        elif i == 0 and j == 1:
            grad_xi = grad_u2[:, 0:1]
            H_ij = torch.autograd.grad(grad_xi, points2, grad_outputs=torch.ones_like(grad_xi), create_graph=True, retain_graph=True)[0][:, 1:2]
        elif i == 0 and j == 2:
            grad_xi = grad_u2[:, 0:1]
            H_ij = torch.autograd.grad(grad_xi, points2, grad_outputs=torch.ones_like(grad_xi), create_graph=True, retain_graph=True)[0][:, 2:3]
        elif i == 1 and j == 0:
            grad_xi = grad_u2[:, 1:2]
            H_ij = torch.autograd.grad(grad_xi, points2, grad_outputs=torch.ones_like(grad_xi), create_graph=True, retain_graph=True)[0][:, 0:1]
        elif i == 1 and j == 1:
            grad_xi = grad_u2[:, 1:2]
            H_ij = torch.autograd.grad(grad_xi, points2, grad_outputs=torch.ones_like(grad_xi), create_graph=True, retain_graph=True)[0][:, 1:2]
        elif i == 1 and j == 2:
            grad_xi = grad_u2[:, 1:2]
            H_ij = torch.autograd.grad(grad_xi, points2, grad_outputs=torch.ones_like(grad_xi), create_graph=True, retain_graph=True)[0][:, 2:3]
        elif i == 2 and j == 0:
            grad_xi = grad_u2[:, 2:3]
            H_ij = torch.autograd.grad(grad_xi, points2, grad_outputs=torch.ones_like(grad_xi), create_graph=True, retain_graph=True)[0][:, 0:1]
        elif i == 2 and j == 1:
            grad_xi = grad_u2[:, 2:3]
            H_ij = torch.autograd.grad(grad_xi, points2, grad_outputs=torch.ones_like(grad_xi), create_graph=True, retain_graph=True)[0][:, 1:2]
        elif i == 2 and j == 2:
            grad_xi = grad_u2[:, 2:3]
            H_ij = torch.autograd.grad(grad_xi, points2, grad_outputs=torch.ones_like(grad_xi), create_graph=True, retain_graph=True)[0][:, 2:3]
        
        nTHn += n2[:, i:i+1] * n2[:, j:j+1] * H_ij

# n·∇u
n_dot_grad_u2 = torch.sum(n2 * grad_u2, dim=1, keepdim=True)

# Euclidean Laplacian
lap_euc2 = torch.zeros_like(u2)
for i in range(3):
    grad_u_i = grad_u2[:, i:i+1]
    grad2_u = torch.autograd.grad(grad_u_i, points2, grad_outputs=torch.ones_like(grad_u_i), create_graph=True, retain_graph=True)[0]
    lap_euc2 += grad2_u[:, i:i+1]

# CORRECT formula: Δ_S u = Δu - n^T H n - 2(n·∇u)
lap_beltrami = lap_euc2 - nTHn - 2 * n_dot_grad_u2

# Expected: -6u for l=2 spherical harmonic
expected6 = -6 * u2

# Check errors
error = torch.abs(lap_beltrami - expected6)
print("Testing CORRECT formula: Δ_S u = Δu - n^T H n - 2(n·∇u)")
print(f"n^T H n at point 0: {nTHn[0].item():.6f}, expected 2u: {(2*u2[0]).item():.6f}")
print(f"n·∇u at point 0: {n_dot_grad_u2[0].item():.6f}, expected 2u: {(2*u2[0]).item():.6f}")
print()
print(f"Δ_S u at point 0: {lap_beltrami[0].item():.6f}")
print(f"Expected -6u at point 0: {expected6[0].item():.6f}")
print()
print(f"Max error: {error.max().item():.6f}")
print(f"Mean error: {error.mean().item():.6f}")
print()

# Alternative: maybe the development guide meant something different?
# Let's also check -2u
expected2 = -2 * u2
error2 = torch.abs(lap_beltrami - expected2)
print(f"If expected is -2u:")
print(f"Max error: {error2.max().item():.6f}")
print(f"Mean error: {error2.mean().item():.6f}")
