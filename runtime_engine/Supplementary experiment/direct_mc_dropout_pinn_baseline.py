"""Direct MC-Dropout PINN baseline (no teacher, no distillation, no reconstruction).

JCP review item V: a true single-stage MC-Dropout PINN that the proposed
GBSD method is compared against.

What this is
============
* Train ONE MC-Dropout network from scratch on `lambda_pde * L_pde +
  lambda_bc * L_bc + lambda_data * L_d`, where the loss terms reuse the
  same standalone residual/boundary functions the teacher uses.
* No teacher PINN. No knowledge distillation. No structure discovery.
  No structured reconstruction.
* Same network width and depth as the proposed Dense Bayesian Student.
* Same dropout rate, same MC-sample count at inference.
* Same evaluation grid and the same per-(case, seed) blind split file
  produced by `utils/blind_split.py`.

What this is NOT
================
* This is NOT a Deep Ensemble PINN. To produce that, run this script
  for several seeds (3 or 5) and ensemble the outputs.
* This is NOT a heteroscedastic likelihood. The variance is purely
  epistemic from MC Dropout.

Usage
-----
    python "Supplementary experiment/direct_mc_dropout_pinn_baseline.py" \
        --case Poisson --seed 0 --preset full

Outputs
-------
    Results/supplementary/baselines/direct_mc_dropout_pinn/<case>_s<seed>/
        model.pth
        predictions.npz
        history.csv
        guard_decision_baseline.csv   # for cross-comparison with GBSD
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Make project root importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Module.Student_MCDropout import Net as MCNet                       # noqa: E402
from Module.UncertaintyEstimation import UncertaintyEstimator           # noqa: E402
from Module import PoissonTools as PT                                   # noqa: E402
from Module.Training import burgers_residual, laplace_residual          # noqa: E402
from utils.blind_split import ensure_split, masked_rel_l2               # noqa: E402
from utils.posterior_predict import (                                   # noqa: E402
    _load_config, _build_layer, make_eval_grid, get_exact_solution,
    _poisson_kwargs,
)


BUDGETS = {
    'quick_check': 500,
    'preview': 1000,
    'medium': 10000,
    'full': 20000,
}


def _set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pde_residual_loss(case: str, model, x: torch.Tensor) -> torch.Tensor:
    if case == 'Laplace':
        return laplace_residual(model, x)
    if case == 'Burgers_inv':
        return burgers_residual(model, x)  # uses nu = 0.01/pi
    if case == 'Poisson':
        return PT.poisson_residual_loss(model, x)
    raise ValueError(f'Unknown case: {case}')


def _boundary_loss(case: str, model, n_b: int, config, device):
    if case == 'Poisson':
        pts = PT.boundary_points(n_b, device=device)
        pred = model(pts)
        if isinstance(pred, tuple):
            pred = pred[0]
        exact = PT.poisson_exact(pts)
        return torch.mean((pred - exact) ** 2)

    x_min = float(config.get('x_min', -1.0))
    x_max = float(config.get('x_max', 1.0))
    y_min = float(config.get('y_min', -1.0))
    y_max = float(config.get('y_max', 1.0))

    # Four edges sampled uniformly.
    y = torch.linspace(y_min, y_max, n_b, device=device).reshape(-1, 1)
    x_left = torch.full_like(y, x_min)
    x_right = torch.full_like(y, x_max)
    x_h = torch.linspace(x_min, x_max, n_b, device=device).reshape(-1, 1)
    y_bot = torch.full_like(x_h, y_min)
    y_top = torch.full_like(x_h, y_max)

    pts = torch.cat([
        torch.cat([x_left, y], dim=1),
        torch.cat([x_right, y], dim=1),
        torch.cat([x_h, y_bot], dim=1),
        torch.cat([x_h, y_top], dim=1),
    ], dim=0)

    # Use the analytical reference at the boundary if available.
    np_pts = pts.detach().cpu().numpy()
    exact = get_exact_solution(case, np_pts)
    if exact is None:
        # Burgers_inv with no analytic boundary: enforce IC u(x,0)=-sin(pi x).
        if case == 'Burgers_inv':
            x_ic = torch.linspace(x_min, x_max, n_b, device=device).reshape(-1, 1)
            t_ic = torch.zeros_like(x_ic)
            ic_pts = torch.cat([x_ic, t_ic], dim=1)
            pred_ic = model(ic_pts)
            if isinstance(pred_ic, tuple):
                pred_ic = pred_ic[0]
            u_ic = -torch.sin(np.pi * x_ic)
            return torch.mean((pred_ic - u_ic) ** 2)
        return torch.tensor(0.0, device=device)

    exact_t = torch.tensor(exact, device=device, dtype=torch.float32)
    pred = model(pts)
    if isinstance(pred, tuple):
        pred = pred[0]
    return torch.mean((pred - exact_t) ** 2)


def _data_loss(case: str, model, config, device):
    """Sparse-observation supervision (Burgers inverse only)."""
    if case != 'Burgers_inv':
        return torch.tensor(0.0, device=device)
    data_serial = str(config.get('data_serial', '1,3')).split(',')
    arrays = []
    for s in data_serial:
        path = f'./Database/{case}_data_{s.strip()}.csv'
        if os.path.isfile(path):
            arrays.append(pd.read_csv(path, header=None).values)
    if not arrays:
        return torch.tensor(0.0, device=device)
    db = np.vstack(arrays)
    xs = torch.tensor(db[:, 0:2], dtype=torch.float32, device=device)
    us = torch.tensor(db[:, 2:3], dtype=torch.float32, device=device)
    pred = model(xs)
    if isinstance(pred, tuple):
        pred = pred[0]
    return torch.mean((pred - us) ** 2)


def _sample_interior(case: str, n_f: int, config, device):
    if case == 'Poisson':
        return PT.sample_interior_points(n_f, device=device)
    x_min = float(config.get('x_min', -1.0))
    x_max = float(config.get('x_max', 1.0))
    y_min = float(config.get('y_min', -1.0))
    y_max = float(config.get('y_max', 1.0))
    x = torch.rand(n_f, 1, device=device) * (x_max - x_min) + x_min
    y = torch.rand(n_f, 1, device=device) * (y_max - y_min) + y_min
    return torch.cat([x, y], dim=1)


def train(case: str, seed: int, preset: str,
          n_mc_samples: int, device: torch.device,
          out_dir: Path):
    _set_seed(seed)
    config = _load_config(case)
    layer = _build_layer(config, student=True)
    model_kwargs = _poisson_kwargs(case, config)

    dropout_rate = float(config.get('dropout_rate', 0.05))
    train_dropout = float(config.get('train_dropout_rate', max(dropout_rate * 0.1, 0.005)))

    # Same width/depth as the proposed dense student.
    model = MCNet(layer, dropout_rate=train_dropout, **model_kwargs).to(device)

    lr = float(config.get('learning_rate', 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps = BUDGETS.get(preset, BUDGETS['full'])

    lambda_pde = float(config.get('lambda_pde_student',
                                  config.get('lambda_pde_teacher', 1.0)))
    lambda_bc = float(config.get('lambda_bc_student',
                                 config.get('lambda_bc_teacher', 1.0)))
    lambda_data = float(config.get('lambda_data_student',
                                   config.get('lambda_data_teacher', 0.0)))

    n_f = int(config.get('grid_node_num', 80)) ** 2
    n_b = int(config.get('bun_node_num', 100))

    history = []
    start = time.perf_counter()

    for step in range(steps):
        model.train()  # dropout ON during training
        x = _sample_interior(case, min(n_f, 4000), config, device)
        loss_pde = _pde_residual_loss(case, model, x)
        loss_bc = _boundary_loss(case, model, n_b, config, device)
        loss_d = _data_loss(case, model, config, device)
        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_data * loss_d

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            history.append({
                'step': step + 1,
                'loss': float(loss.detach().cpu()),
                'loss_pde': float(loss_pde.detach().cpu()),
                'loss_bc': float(loss_bc.detach().cpu()),
                'loss_data': float(loss_d.detach().cpu()),
            })
            print(f'  [direct-mc {case} s{seed}] step={step+1}/{steps} '
                  f'loss={loss.item():.4e} pde={loss_pde.item():.4e} '
                  f'bc={loss_bc.item():.4e} data={loss_d.item():.4e}')

    elapsed_train = time.perf_counter() - start

    # Restore the MC inference dropout rate.
    model.set_dropout_rate(dropout_rate)

    # Save model.
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / 'model.pth')
    pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)

    # Predict on eval grid.
    x_eval, X1, X2 = make_eval_grid(case, 100, device, config)
    estimator = UncertaintyEstimator(model, n_samples=n_mc_samples)
    t0 = time.perf_counter()
    preds = estimator.predict(x_eval)
    elapsed_mc = time.perf_counter() - t0

    mean = preds['mean'].cpu().numpy().reshape(-1)
    mc_mean = preds['mc_mean'].cpu().numpy().reshape(-1)
    std = preds['std'].cpu().numpy().reshape(-1)
    samples = preds['samples'].cpu().numpy()
    exact = get_exact_solution(case, x_eval.cpu().numpy()).reshape(-1)

    # Pre-declared blind split shared with GBSD runs.
    split = ensure_split(case, seed, n_total=exact.size)
    blind_mask = split['blind_test_mask']
    guard_mask = split['guard_mask']

    rel_blind = masked_rel_l2(mean, exact, blind_mask)
    rel_guard = masked_rel_l2(mean, exact, guard_mask)
    print(f'  Direct MC-Dropout PINN [{case} s{seed}]: '
          f'rL2_guard={rel_guard:.4e}, rL2_blind={rel_blind:.4e}, '
          f'avg_std={std.mean():.4e}, train_time={elapsed_train:.1f}s, '
          f'mc_time={elapsed_mc:.3f}s')

    np.savez(out_dir / 'predictions.npz',
             x=x_eval.cpu().numpy(), X1=X1, X2=X2,
             direct_mc_mean=mean, direct_mc_mc_mean=mc_mean,
             direct_mc_std=std, direct_mc_samples=samples,
             exact=exact,
             guard_mask=guard_mask, blind_test_mask=blind_mask,
             split_seed=np.array([seed]),
             rel_L2_blind=np.array([rel_blind]),
             rel_L2_guard=np.array([rel_guard]),
             train_time_s=np.array([elapsed_train]),
             mc_inference_time_s=np.array([elapsed_mc]),
             n_mc_samples=np.array([n_mc_samples]))

    # One-row baseline summary CSV (joinable with GBSD's blind_test_metrics.csv).
    out_csv = out_dir.parent / 'direct_mc_dropout_pinn_summary.csv'
    write_header = not out_csv.is_file()
    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['case', 'seed', 'preset', 'rel_L2_blind',
                        'rel_L2_guard', 'avg_std', 'train_time_s',
                        'mc_inference_time_s', 'n_mc'])
        w.writerow([case, seed, preset, rel_blind, rel_guard,
                    float(std.mean()), elapsed_train, elapsed_mc,
                    n_mc_samples])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True,
                        choices=['Laplace', 'Poisson', 'Burgers_inv'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preset', default='full',
                        choices=list(BUDGETS.keys()))
    parser.add_argument('--n_mc', type=int, default=200)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    out_dir = (PROJECT_ROOT / 'Results' / 'supplementary' / 'baselines' /
               'direct_mc_dropout_pinn' / f'{args.case}_s{args.seed}_{args.preset}')
    train(args.case, args.seed, args.preset, args.n_mc, device, out_dir)


if __name__ == '__main__':
    main()
