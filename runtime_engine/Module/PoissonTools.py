# coding = utf-8
"""Shared helpers for the Poisson experiment.

The project uses the convention

    u_xx + u_yy = f(x, y)

with a zero Dirichlet boundary on [0, 1]^2.  The exact solution below is
consistent with that sign convention.
"""
import math
from typing import Optional

import torch


def as_bool(value, default: bool = False) -> bool:
    """Parse bool-ish config values coming from CSV files."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return default


def fourier_feature_dim(coord_dim: int = 2, modes: int = 4,
                        include_input: bool = True) -> int:
    base = coord_dim if include_input else 0
    return base + 2 * coord_dim * int(modes)


def encode_fourier(x: torch.Tensor, modes: int = 4,
                   include_input: bool = True) -> torch.Tensor:
    """Return [x, y, sin/cos(2*pi*k*x), sin/cos(2*pi*k*y)]."""
    if modes <= 0:
        return x

    feats = [x] if include_input else []
    for k in range(1, int(modes) + 1):
        omega = 2.0 * math.pi * k
        for d in range(x.shape[1]):
            xd = x[:, d:d + 1]
            feats.append(torch.sin(omega * xd))
            feats.append(torch.cos(omega * xd))
    return torch.cat(feats, dim=1)


def poisson_envelope(x: torch.Tensor) -> torch.Tensor:
    """Zero on the boundary of [0, 1]^2 and positive in the interior."""
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    return x1 * (1.0 - x1) * x2 * (1.0 - x2)


def apply_zero_dirichlet_hard_bc(raw_x: torch.Tensor,
                                 network_output: torch.Tensor) -> torch.Tensor:
    """Hard zero Dirichlet constraint for the Poisson unit square."""
    return poisson_envelope(raw_x) * network_output


def poisson_exact(x: torch.Tensor) -> torch.Tensor:
    """Exact solution for the current 4-mode Poisson benchmark."""
    out = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    for k in range(1, 5):
        coeff = 0.5 * ((-1) ** k) / (2.0 * math.pi ** 2)
        out = out + coeff * torch.sin(k * math.pi * x1) * torch.sin(k * math.pi * x2)
    return out


def poisson_source(x: torch.Tensor) -> torch.Tensor:
    """Right-hand side f for u_xx + u_yy = f."""
    out = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    for k in range(1, 5):
        coeff = 0.5 * ((-1) ** (k + 1)) * (k ** 2)
        out = out + coeff * torch.sin(k * math.pi * x1) * torch.sin(k * math.pi * x2)
    return out


def poisson_residual_pointwise(model, x: torch.Tensor,
                               create_graph: bool = True) -> torch.Tensor:
    """Pointwise residual r = u_xx + u_yy - f(x, y)."""
    x = x.clone().detach().requires_grad_(True)
    u = model(x)
    if isinstance(u, tuple):
        u = u[0]

    grad_u = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=create_graph, retain_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y, x, grad_outputs=torch.ones_like(u_y),
        create_graph=create_graph, retain_graph=True
    )[0][:, 1:2]
    return u_xx + u_yy - poisson_source(x)


def poisson_residual_loss(model, x: torch.Tensor) -> torch.Tensor:
    residual = poisson_residual_pointwise(model, x, create_graph=True)
    return torch.mean(residual ** 2)


def sample_interior_points(n: int, device, dtype=torch.float32,
                           eps: float = 1e-5,
                           generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Uniform random interior points in (0, 1)^2."""
    pts = torch.rand((int(n), 2), device=device, dtype=dtype, generator=generator)
    if eps > 0:
        pts = eps + (1.0 - 2.0 * eps) * pts
    return pts


def boundary_points(n_per_side: int, device, dtype=torch.float32) -> torch.Tensor:
    """Four unit-square boundary segments, including corners."""
    n = int(n_per_side)
    t = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(-1, 1)
    zeros = torch.zeros_like(t)
    ones = torch.ones_like(t)
    return torch.cat([
        torch.cat([zeros, t], dim=1),
        torch.cat([ones, t], dim=1),
        torch.cat([t, zeros], dim=1),
        torch.cat([t, ones], dim=1),
    ], dim=0)
