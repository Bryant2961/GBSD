"""Multi-seed Direct MC-Dropout PINN UQ baseline runner.

Wraps the existing `direct_mc_dropout_pinn_baseline.py` so we can run
multiple seeds per case in one command, archive each seed separately
under the new full_baselines/uq layout, and emit a single
`uq_baselines_raw.csv` row per (case, seed) tagged
method='direct_mc_dropout_pinn'.

What "Direct MC-Dropout PINN" means here (unchanged from v3.37):
  * No teacher distillation.
  * No structured reconstruction.
  * No final-source guard.
  * Train an MC-Dropout net directly on lambda_pde*L_pde + lambda_bc*L_bc
    + lambda_data*L_d, then evaluate on the SAME blind split as GBSD.

Output layout:
    Results/supplementary/full_baselines/uq/direct_mc_dropout_pinn/
        <preset>/<case>/seed_<seed>/
            model.pth, predictions.npz, history.csv,
            metrics.json, environment.json
    Results/supplementary/tables/uq_baselines_raw.csv     # appended
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


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


SUPP_ROOT = PROJECT_ROOT / "Results" / "supplementary"
OUT_ROOT = SUPP_ROOT / "full_baselines" / "uq" / "direct_mc_dropout_pinn"
TABLE_DIR = SUPP_ROOT / "tables"

CASES = ("Laplace", "Poisson", "Burgers_inv")
BUDGETS = {
    "smoke": 20,
    "quick_check": 500,
    "preview": 1000,
    "medium": 10000,
    "full": 20000,
}
BURGERS_NU_TRUE = 0.01 / math.pi


def _set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _training_steps(config: Dict, preset: str) -> int:
    """Use the case-specific dense-student budget for full runs."""
    if preset == "full":
        return int(config.get("student_train_steps",
                              config.get("train_steps", BUDGETS["full"])))
    return int(BUDGETS.get(preset, BUDGETS["full"]))


# Reuse loss helpers from the v3.37 baseline.
def _pde_loss(case, model, x):
    if case == "Laplace":
        return laplace_residual(model, x)
    if case == "Burgers_inv":
        return burgers_residual(model, x)  # uses analytic nu = 0.01/pi
    if case == "Poisson":
        return PT.poisson_residual_loss(model, x)
    raise ValueError(f"Unknown case: {case}")


def _boundary_loss(case, model, n_b, config, device):
    if case == "Poisson":
        pts = PT.boundary_points(n_b, device=device)
        pred = model(pts)
        if isinstance(pred, tuple):
            pred = pred[0]
        exact = PT.poisson_exact(pts)
        return torch.mean((pred - exact) ** 2)

    x_min = float(config.get("x_min", -1.0))
    x_max = float(config.get("x_max", 1.0))
    y_min = float(config.get("y_min", -1.0))
    y_max = float(config.get("y_max", 1.0))

    y = torch.linspace(y_min, y_max, n_b, device=device).reshape(-1, 1)
    x_left = torch.full_like(y, x_min); x_right = torch.full_like(y, x_max)
    x_h = torch.linspace(x_min, x_max, n_b, device=device).reshape(-1, 1)
    y_bot = torch.full_like(x_h, y_min); y_top = torch.full_like(x_h, y_max)
    pts = torch.cat([
        torch.cat([x_left, y], dim=1), torch.cat([x_right, y], dim=1),
        torch.cat([x_h, y_bot], dim=1), torch.cat([x_h, y_top], dim=1),
    ], dim=0)
    exact = get_exact_solution(case, pts.detach().cpu().numpy())
    if exact is None and case == "Burgers_inv":
        x_ic = torch.linspace(x_min, x_max, n_b, device=device).reshape(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        ic_pts = torch.cat([x_ic, t_ic], dim=1)
        pred_ic = model(ic_pts)
        if isinstance(pred_ic, tuple):
            pred_ic = pred_ic[0]
        u_ic = -torch.sin(math.pi * x_ic)
        return torch.mean((pred_ic - u_ic) ** 2)
    if exact is None:
        return torch.tensor(0.0, device=device)
    target = torch.tensor(exact, dtype=torch.float32, device=device)
    pred = model(pts)
    if isinstance(pred, tuple):
        pred = pred[0]
    return torch.mean((pred - target) ** 2)


def _data_loss(case, model, config, device):
    if case != "Burgers_inv":
        return torch.tensor(0.0, device=device)
    serials = str(config.get("data_serial", "1,3")).split(",")
    arrays = []
    for s in serials:
        path = PROJECT_ROOT / "Database" / f"{case}_data_{s.strip()}.csv"
        if path.is_file():
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


def _sample_interior(case, n, config, device):
    if case == "Poisson":
        return PT.sample_interior_points(n, device=device)
    x_min = float(config.get("x_min", -1.0))
    x_max = float(config.get("x_max", 1.0))
    y_min = float(config.get("y_min", -1.0))
    y_max = float(config.get("y_max", 1.0))
    x = torch.rand(n, 1, device=device) * (x_max - x_min) + x_min
    y = torch.rand(n, 1, device=device) * (y_max - y_min) + y_min
    return torch.cat([x, y], dim=1)


def _coverage95(m, s, e):
    m = np.asarray(m).reshape(-1)
    s = np.maximum(np.asarray(s).reshape(-1), 1e-12)
    e = np.asarray(e).reshape(-1)
    return float(np.mean((e >= m - 1.96 * s) & (e <= m + 1.96 * s)))


def _nll_gaussian(m, s, e):
    m = np.asarray(m).reshape(-1)
    s = np.maximum(np.asarray(s).reshape(-1), 1e-12)
    e = np.asarray(e).reshape(-1)
    var = s ** 2
    return float(np.mean(0.5 * np.log(2.0 * np.pi * var) + (e - m) ** 2 / (2.0 * var)))


def _corr(m, s, e):
    err = np.abs(np.asarray(m).reshape(-1) - np.asarray(e).reshape(-1))
    s = np.asarray(s).reshape(-1)
    if float(np.std(err)) <= 1e-12 or float(np.std(s)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(err, s)[0, 1])


def _json_safe(obj):
    """Convert NaN/Inf floats to None so the JSON is strict-valid."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def train_single(case: str, seed: int, preset: str,
                 n_mc_samples: int, device: torch.device,
                 force: bool = False) -> Dict:
    out_dir = OUT_ROOT / preset / case / f"seed_{int(seed)}"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        print(f"  [skip] direct-mc {case} seed={seed} already done.")
        with open(metrics_path) as f:
            return json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)

    config = _load_config(case)
    layer = _build_layer(config, student=True)
    model_kwargs = _poisson_kwargs(case, config)
    dropout_rate = float(config.get("dropout_rate", 0.05))
    train_dropout = float(config.get("train_dropout_rate",
                                     max(dropout_rate * 0.1, 0.005)))
    model = MCNet(layer, dropout_rate=train_dropout, **model_kwargs).to(device)

    lr = float(config.get("learning_rate", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps = _training_steps(config, preset)
    lambda_pde = float(config.get("lambda_pde_student",
                                   config.get("lambda_pde_teacher", 1.0)))
    lambda_bc = float(config.get("lambda_bc_student",
                                  config.get("lambda_bc_teacher", 1.0)))
    lambda_data = float(config.get("lambda_data_student",
                                    config.get("lambda_data_teacher", 0.0)))
    n_f = int(config.get("grid_node_num", 80)) ** 2
    n_b = int(config.get("bun_node_num", 100))

    history = []
    t0 = time.perf_counter()
    for step in range(steps):
        model.train()
        x = _sample_interior(case, min(n_f, 4000), config, device)
        loss_pde = _pde_loss(case, model, x)
        loss_bc = _boundary_loss(case, model, n_b, config, device)
        loss_d = _data_loss(case, model, config, device)
        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_data * loss_d
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            history.append({
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "loss_pde": float(loss_pde.detach().cpu()),
                "loss_bc": float(loss_bc.detach().cpu()),
                "loss_data": float(loss_d.detach().cpu()),
            })
            print(f"  [direct-mc {case} s{seed}] step={step+1}/{steps} "
                  f"loss={loss.item():.4e}")
    train_time = time.perf_counter() - t0

    # Restore the MC inference dropout rate.
    model.set_dropout_rate(dropout_rate)
    torch.save(model.state_dict(), out_dir / "model.pth")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Predict.
    x_eval, X1, X2 = make_eval_grid(case, 100, device, config)
    estimator = UncertaintyEstimator(model, n_samples=n_mc_samples)
    t0 = time.perf_counter()
    preds = estimator.predict(x_eval)
    mc_inference_time = time.perf_counter() - t0
    mean = preds["mean"].cpu().numpy().reshape(-1)
    mc_mean = preds["mc_mean"].cpu().numpy().reshape(-1)
    std = preds["std"].cpu().numpy().reshape(-1)
    exact = get_exact_solution(case, x_eval.cpu().numpy())
    exact = exact.reshape(-1) if exact is not None \
        else np.full(mean.shape, math.nan)

    split = ensure_split(case, seed, n_total=exact.size)
    guard = split["guard_mask"]; blind = split["blind_test_mask"]
    valid = ~np.isnan(exact)
    if int((blind & valid).sum()) == 0 or int((guard & valid).sum()) == 0:
        raise RuntimeError(
            f"No valid guard/blind evaluation points for {case} seed={seed}. "
            "Check that the reference solution file exists, especially "
            "Database/Burgers_inv_reference.npz for Burgers_inv.")

    rel_blind = float(masked_rel_l2(mean, exact, blind & valid))
    rel_guard = float(masked_rel_l2(mean, exact, guard & valid))
    m_b = mean[blind & valid]; s_b = std[blind & valid]; e_b = exact[blind & valid]

    nu_pred = math.nan
    nu_rel = math.nan
    # Direct MC-Dropout PINN cannot estimate nu (no learnable nu in the model);
    # we mark as NaN for consistency with the table schema.

    row = {
        "case": case,
        "seed_or_member": int(seed),
        "method": "direct_mc_dropout_pinn",
        "rel_l2_blind": rel_blind,
        "rel_l2_guard": rel_guard,
        "coverage95_blind": _coverage95(m_b, s_b, e_b),
        "corr_abs_error_std_blind": _corr(m_b, s_b, e_b),
        "nll_blind": _nll_gaussian(m_b, s_b, e_b),
        "mean_interval_width_blind": float(np.mean(2 * 1.96 * np.maximum(s_b, 1e-12))),
        "avg_std_blind": float(np.mean(s_b)),
        "runtime_train_s": float(train_time),
        "runtime_mc_or_ensemble_inference_s": float(mc_inference_time),
        "n_mc_samples_or_ensemble_size": int(n_mc_samples),
        "nu_pred": nu_pred,
        "nu_relative_error": nu_rel,
        "n_blind": int((blind & valid).sum()),
        "n_guard": int((guard & valid).sum()),
        "output_dir": str(out_dir),
        "preset": preset,
    }
    np.savez(out_dir / "predictions.npz",
             x=x_eval.cpu().numpy(), X1=X1, X2=X2,
             mean=mean, mc_mean=mc_mean, std=std, exact=exact,
             guard_mask=guard, blind_test_mask=blind)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(_json_safe(row), f, indent=2)
    with open(out_dir / "environment.json", "w") as f:
        json.dump({
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python_version": sys.version.split(" ")[0],
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_name": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else None),
            "platform": platform.platform(),
            "seed": int(seed),
            "argv": sys.argv,
        }, f, indent=2)
    print(f"  [done] direct-mc {case} seed={seed}: "
          f"rL2_blind={rel_blind:.4e}, cov95={row['coverage95_blind']:.3f}, "
          f"NLL={row['nll_blind']:.4f}")
    return row


def append_to_uq_table(rows: List[Dict]):
    if not rows:
        return
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "uq_baselines_raw.csv"
    df_new = pd.DataFrame(rows)
    if out.is_file():
        try:
            df_old = pd.read_csv(out)
            key_cols = ["case", "seed_or_member", "method", "preset"]
            key_new = df_new[key_cols].astype(str).agg("|".join, axis=1)
            key_old = df_old[key_cols].astype(str).agg("|".join, axis=1)
            df_old = df_old[~key_old.isin(key_new.tolist())]
            df = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df = df.sort_values(["preset", "case", "method", "seed_or_member"])
    df.to_csv(out, index=False)
    print(f"\nAppended to: {out} ({len(df)} rows total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=list(CASES) + ["all"], default="all")
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="Repeatable. Default: --seed 0 --seed 1 --seed 2")
    parser.add_argument("--preset", default="full", choices=list(BUDGETS.keys()))
    parser.add_argument("--n_mc", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    cases = list(CASES) if args.case == "all" else [args.case]
    seeds = args.seed if args.seed else [0, 1, 2]
    rows = []
    for case in cases:
        for seed in seeds:
            try:
                rows.append(train_single(case, int(seed), args.preset,
                                         args.n_mc, device, args.force))
            except Exception as e:
                import traceback
                print(f"  [fail] direct-mc {case} seed={seed}: {e}")
                traceback.print_exc()
    append_to_uq_table(rows)


if __name__ == "__main__":
    main()
