"""
Generate a standard Burgers benchmark reference and sparse inverse data.

Benchmark:
    u_t + u u_x - nu u_xx = 0
    x in [-1, 1], t in [0, 1]
    u(x, 0) = -sin(pi*x)
    u(-1, t) = u(1, t) = 0
    nu = 0.01 / pi

The solver is a conservative finite-volume method with Rusanov flux,
central diffusion, and SSP-RK3 time stepping.  The generated dense grid is
used as the Burgers "Exact/Reference" solution for plotting and metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def rhs_burgers(u: np.ndarray, dx: float, nu: float) -> np.ndarray:
    """Semi-discrete conservative Burgers RHS with zero Dirichlet endpoints."""
    dudt = np.zeros_like(u)
    f = 0.5 * u * u
    a = np.maximum(np.abs(u[:-1]), np.abs(u[1:]))
    flux = 0.5 * (f[:-1] + f[1:]) - 0.5 * a * (u[1:] - u[:-1])
    dudt[1:-1] = (
        -(flux[1:] - flux[:-1]) / dx
        + nu * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    )
    return dudt


def rk3_step(u: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """SSP-RK3 step with hard zero boundary conditions."""
    u0 = u
    u1 = u0 + dt * rhs_burgers(u0, dx, nu)
    u1[0] = 0.0
    u1[-1] = 0.0

    u2 = 0.75 * u0 + 0.25 * (u1 + dt * rhs_burgers(u1, dx, nu))
    u2[0] = 0.0
    u2[-1] = 0.0

    un = (1.0 / 3.0) * u0 + (2.0 / 3.0) * (u2 + dt * rhs_burgers(u2, dx, nu))
    un[0] = 0.0
    un[-1] = 0.0
    return un


def solve_reference(nx: int, nt: int, nu: float, ic_sign: float,
                    cfl: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    dx = x[1] - x[0]

    u = ic_sign * np.sin(np.pi * x)
    u[0] = 0.0
    u[-1] = 0.0

    out = np.zeros((nt, nx), dtype=np.float64)
    out[0] = u
    current_t = 0.0
    out_idx = 1
    steps = 0

    while out_idx < nt:
        target_t = t[out_idx]
        umax = max(float(np.max(np.abs(u))), 1e-8)
        dt_conv = dx / umax
        dt_diff = dx * dx / max(2.0 * nu, 1e-12)
        dt = cfl * min(dt_conv, dt_diff)
        dt = min(dt, target_t - current_t)
        u = rk3_step(u, dt, dx, nu)
        current_t += dt
        steps += 1
        if current_t >= target_t - 1e-14:
            out[out_idx] = u
            out_idx += 1

    print(f"Solved Burgers reference: nx={nx}, nt={nt}, steps={steps}, nu={nu:.10f}")
    return x, t, out


def sample_points(rng: np.random.Generator, n_uniform: int,
                  n_shock: int = 0) -> np.ndarray:
    xu = rng.uniform([-0.95, 0.03], [0.95, 0.97], size=(n_uniform, 2))
    if n_shock <= 0:
        return xu
    xs = rng.normal(loc=0.0, scale=0.18, size=(n_shock, 1))
    xs = np.clip(xs, -0.95, 0.95)
    ts = rng.uniform(0.15, 0.97, size=(n_shock, 1))
    return np.vstack([xu, np.hstack([xs, ts])])


def write_sparse_data(db_dir: Path, x: np.ndarray, t: np.ndarray,
                      u_tx: np.ndarray, seed: int) -> None:
    interp = RegularGridInterpolator(
        (t, x), u_tx, bounds_error=False, fill_value=None)
    rng = np.random.default_rng(seed)

    specs = {
        "1": sample_points(rng, n_uniform=100, n_shock=0),
        "2": sample_points(rng, n_uniform=200, n_shock=0),
        "3": sample_points(rng, n_uniform=60, n_shock=80),
    }

    for sid, xy in specs.items():
        values = interp(np.column_stack([xy[:, 1], xy[:, 0]])).reshape(-1, 1)
        arr = np.hstack([xy, values])
        path = db_dir / f"Burgers_inv_data_{sid}.csv"
        np.savetxt(path, arr, delimiter=",", fmt="%.9f")
        print(f"Wrote {path} ({arr.shape[0]} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=801)
    parser.add_argument("--nt", type=int, default=201)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ic_sign", type=float, default=-1.0)
    parser.add_argument("--out_dir", type=str, default="./Database")
    args = parser.parse_args()

    nu = 0.01 / np.pi
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, t, u_tx = solve_reference(
        nx=args.nx, nt=args.nt, nu=nu,
        ic_sign=args.ic_sign, cfl=args.cfl)

    ref_path = out_dir / "Burgers_inv_reference.npz"
    np.savez_compressed(
        ref_path,
        x=x, t=t, u=u_tx, nu=np.array([nu], dtype=np.float64),
        ic_sign=np.array([args.ic_sign], dtype=np.float64),
        equation=np.array(["u_t + u*u_x - nu*u_xx = 0"]),
    )
    print(f"Wrote {ref_path}")

    write_sparse_data(out_dir, x, t, u_tx, seed=args.seed)


if __name__ == "__main__":
    main()
