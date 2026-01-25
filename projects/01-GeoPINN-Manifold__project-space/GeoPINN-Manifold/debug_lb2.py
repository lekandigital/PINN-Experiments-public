"""Debug Laplace-Beltrami computation - Version 2"""
import torch
torch.set_printoptions(precision=6)

n_test = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sample sphere points (Fibonacci)
indices = torch.arange(n_test, dtype=torch.float32, device=device) + 0.5
phi = torch.acos(1 - 2 * indices / n_test)
theta = 3.14159 * (1 + 5**0.5) * indices

x = torch.sin(phi) * torch.cos(theta)
y = torch.sin(phi) * torch.sin(theta)
z = torch.cos(phi)

points = torch.stack([x, y, z], dim=-1).requires_grad_(True)

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

# n·∇u
n = points
n_dot_grad_u = torch.sum(n * grad_u, dim=1, keepdim=True)

# ∇(n·∇u)
grad_n_dot_grad_u = torch.autograd.grad(n_dot_grad_u, points, grad_outputs=torch.ones_like(n_dot_grad_u), create_graph=True, retain_graph=True)[0]

# n·∇(n·∇u)
n_dot_grad_n_dot_grad_u = torch.sum(n * grad_n_dot_grad_u, dim=1, keepdim=True)

# Expected: -2xy
expected = (-2 * points[:, 0] * points[:, 1]).unsqueeze(-1)

# Test formulas
print("Testing various formulas:")
print()

# Formula A: Δu - n·∇(n·∇u)
v1 = laplacian_euclidean - n_dot_grad_n_dot_grad_u
print(f"A: Δu - n·∇(n·∇u) = {v1[0].item():.6f}, expected = {expected[0].item():.6f}, ratio = {v1[0].item()/expected[0].item():.3f}")

# Formula B: Δu - n·∇(n·∇u) + 2(n·∇u) 
v2 = laplacian_euclidean - n_dot_grad_n_dot_grad_u + 2 * n_dot_grad_u
print(f"B: Δu - n·∇(n·∇u) + 2(n·∇u) = {v2[0].item():.6f}, expected = {expected[0].item():.6f}, ratio = {v2[0].item()/expected[0].item():.3f}")

# Formula C: Δu - 2(n·∇u) for spherical harmonics on unit sphere (simplified)
# For Y_lm, we have Δ_S Y = -l(l+1) Y. For xy (l=2 mode), Δ_S(xy) = -6xy/3 = -2xy
# The eigenvalue formula is Δ_S u = -(l(l+1)) u for eigenfunctions

# Let's try: Δ_S u = Δu + |∇u|² - (n·∇u)² / |x|² - 2(n·∇u) for r=1
# No, simpler approach: the standard formula is:
# Δ_S u = Δu - ∂²u/∂r² - (d-1)/r * ∂u/∂r
# On unit sphere (d=3, r=1): Δ_S u = Δu - ∂²u/∂r² - 2 ∂u/∂r

# ∂u/∂r = n·∇u (radial derivative)
# For ∂²u/∂r² we need to be careful. The radial Hessian term.

# Actually: n·∇(n·∇u) = ∂²u/∂r² + |∇_S u|²/r = ∂²u/∂r² (at r=1 if we project properly)
# But there's an issue with computing this correctly.

# Direct approach: on unit sphere, the spherical Laplacian for u=xy is:
# Δ_S(xy) = -l(l+1)xy where l=2, so Δ_S(xy) = -6xy/3 = -2xy ✓ (but xy is only part of Y_2^m)

# Let's verify differently: 
# |∇u|² = y² + x² (the tangent gradient squared should be this minus radial component)
grad_u_squared = torch.sum(grad_u * grad_u, dim=1, keepdim=True)
n_dot_grad_u_squared = n_dot_grad_u ** 2
tangent_grad_squared = grad_u_squared - n_dot_grad_u_squared

print(f"\n|∇u|² = {grad_u_squared[0].item():.6f}")
print(f"(n·∇u)² = {n_dot_grad_u_squared[0].item():.6f}")
print(f"|∇_S u|² = {tangent_grad_squared[0].item():.6f}")

# The key insight: when we compute n·∇(n·∇u), we're computing ∂/∂r(∂u/∂r)
# But n·∇(n·∇u) includes the derivative of n as well (since n = x varies with position)

# Let's decompose: ∇(n·∇u) = ∇(Σ n_i ∂u/∂x_i) 
# = Σ [∂n_i/∂x_j * ∂u/∂x_i + n_i * ∂²u/∂x_i∂x_j]
# On unit sphere, ∂n_i/∂x_j = δ_ij (since n = x)
# So ∇(n·∇u) = ∇u + n·Hessian(u)

# Therefore: n·∇(n·∇u) = n·∇u + n·H·n = (n·∇u) + ∂²u/∂r²

# So: ∂²u/∂r² = n·∇(n·∇u) - n·∇u
radial_second = n_dot_grad_n_dot_grad_u - n_dot_grad_u
print(f"\n∂²u/∂r² = n·∇(n·∇u) - n·∇u = {radial_second[0].item():.6f}")

# Standard formula: Δ_S u = Δu - ∂²u/∂r² - 2/r * ∂u/∂r (at r=1)
v3 = laplacian_euclidean - radial_second - 2 * n_dot_grad_u
print(f"C: Δu - ∂²u/∂r² - 2(∂u/∂r) = {v3[0].item():.6f}, expected = {expected[0].item():.6f}, ratio = {v3[0].item()/expected[0].item():.3f}")

# Another form: substituting back
# Δ_S = Δu - (n·∇(n·∇u) - n·∇u) - 2*n·∇u = Δu - n·∇(n·∇u) + n·∇u - 2*n·∇u
#     = Δu - n·∇(n·∇u) - n·∇u
v4 = laplacian_euclidean - n_dot_grad_n_dot_grad_u - n_dot_grad_u
print(f"D: Δu - n·∇(n·∇u) - n·∇u = {v4[0].item():.6f}, expected = {expected[0].item():.6f}")

# Hmm, v3 still gives 2x. Let me try direct formula for u = x*y:
# u = r² sin²φ cosθ sinθ = r² sin²φ * sin(2θ)/2
# For r=1: u = sin²φ * sin(2θ)/2

# The spherical Laplacian in (θ, φ):
# Δ_S = 1/(sinφ) ∂/∂φ(sinφ ∂/∂φ) + 1/sin²φ ∂²/∂θ²

# Let f = sin²φ sin(2θ)/2
# ∂f/∂θ = sin²φ cos(2θ)
# ∂²f/∂θ² = -2 sin²φ sin(2θ) = -4u
# 1/sin²φ * ∂²f/∂θ² = -4u/sin²φ... this is getting complicated

# For eigenfunctions, we can use: xy is a linear combo of Y_2^{±2}
# Y_2^2 ~ sin²θ e^{2iφ} and xy = (Y_2^{-2} - Y_2^2)/(2i) * normalization
# Δ_S Y_l^m = -l(l+1) Y_l^m, so Δ_S(xy) should = -2(2+1) xy = -6xy 

print(f"\n\nActually for spherical harmonics l=2: Δ_S(xy) = -l(l+1) xy = -6xy")
print(f"But xy on sphere is normalized: xy = c * Y_2^{2}, the eigenvalue is -6")
print(f"\nOur expected -2xy seems wrong? Let's check:")
print(f"-6*xy at point 0: {(-6 * points[0,0] * points[0,1]).item():.6f}")
print(f"But wait - xy on sphere is NOT a spherical harmonic directly!")

# xy in R³ restricted to sphere: u(x,y,z) = xy where x²+y²+z² = 1
# This IS proportional to Re(Y_2^2) ~ sin²θ sin(2φ)/2 ~ xy/(x²+y²+z²) = xy (on sphere)
# So Δ_S(xy) = -6 * xy

v6 = -6 * u
print(f"\nIf Δ_S(xy) = -6xy:")
print(f"Computed A: {v1[0].item():.6f}")
print(f"Expected -6xy: {v6[0].item():.6f}")
print(f"Ratio: {v1[0].item()/v6[0].item():.3f}")
