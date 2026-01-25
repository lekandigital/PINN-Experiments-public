"""Debug Laplace-Beltrami computation"""
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

print('Points (first 3):')
print(points[:3])
print()
print('u = xy:', u[:3].squeeze())
print()

# Gradient of u = xy: ∇u = (y, x, 0)
grad_u = torch.autograd.grad(u, points, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
print('grad_u (should be [y, x, 0]):')
print(grad_u[:3])
print()

# Euclidean Laplacian: Δu = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z² = 0 for u=xy
laplacian_euclidean = torch.zeros_like(u)
for i in range(3):
    grad_u_i = grad_u[:, i:i+1]
    grad2_u_i = torch.autograd.grad(grad_u_i, points, grad_outputs=torch.ones_like(grad_u_i), create_graph=True, retain_graph=True)[0]
    laplacian_euclidean += grad2_u_i[:, i:i+1]

print('Euclidean Laplacian Δu (should be 0):', laplacian_euclidean[:3].squeeze())
print()

# n·∇u where n = (x, y, z) on unit sphere
n = points
n_dot_grad_u = torch.sum(n * grad_u, dim=1, keepdim=True)  # x*y + y*x + z*0 = 2xy
print('n·∇u (should be 2xy):', n_dot_grad_u[:3].squeeze())
print('Expected 2xy:', (2 * points[:3, 0] * points[:3, 1]))
print()

# ∇(n·∇u)
grad_n_dot_grad_u = torch.autograd.grad(n_dot_grad_u, points, grad_outputs=torch.ones_like(n_dot_grad_u), create_graph=True, retain_graph=True)[0]
print('grad(n·∇u):')
print(grad_n_dot_grad_u[:3])
print()

# n·∇(n·∇u)
n_dot_grad_n_dot_grad_u = torch.sum(n * grad_n_dot_grad_u, dim=1, keepdim=True)
print('n·grad(n·grad_u):', n_dot_grad_n_dot_grad_u[:3].squeeze())
print()

# Simple formula: Δ_S u = Δu - n·∇(n·∇u)
lap_beltrami_v1 = laplacian_euclidean - n_dot_grad_n_dot_grad_u
print('Version 1 (Δu - n·grad(n·grad_u)):', lap_beltrami_v1[:3].squeeze())

# With curvature: Δ_S u = Δu - 2H(n·∇u) - n·∇(n·∇u), H=1
H = 1.0
lap_beltrami_v2 = laplacian_euclidean - 2 * H * n_dot_grad_u - n_dot_grad_n_dot_grad_u
print('Version 2 (with -2H(n·grad_u)):', lap_beltrami_v2[:3].squeeze())

# Expected: -2xy
expected = -2 * points[:3, 0] * points[:3, 1]
print()
print('Expected Δ_S(xy) = -2xy:', expected)
