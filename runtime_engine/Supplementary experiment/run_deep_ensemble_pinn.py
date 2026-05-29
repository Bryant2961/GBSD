"""Deep Ensemble PINN UQ baseline runner.

What this is
============
K = 5 (default) independently trained deterministic PINNs per case. Each
ensemble member:
  * Uses a different seed.
  * Trains a plain MLP (no dropout) from scratch on
    lambda_pde*L_pde + lambda_bc*L_bc + lambda_data*L_d.
  * Has no teacher distillation, no structured reconstruction.
The ensemble prediction is the mean across members; the ensemble std
across members is used as the uncertainty estimate.

Why this is a true Deep Ensemble PINN
=====================================
Unlike `seed_variance_of_dense_student.py`, which stacks distilled
MC-Dropout students that all consumed the same teacher, here each
member is its own from-scratch PINN. The teacher channel is NOT shared
across members, so the variance is genuinely epistemic over
independently trained PINNs.

Output layout:
    Results/supplementary/full_baselines/uq/deep_ensemble_pinn/
        <preset>/<case>/ensemble_<id>/
            members/seed_<seed>/model.pth + history.csv + metrics.json
            ensemble_predictions.npz       # ensemble mean + ensemble std
            ensemble_metrics.json          # the row for uq_baselines_raw.csv
            environment.json
    Results/supplementary/tables/uq_baselines_raw.csv     # appended

A single ensemble_id groups several seed members together so a future
analysis can pool multiple ensembles (if desired) for higher-K studies.
"""
from __future__ import annotations

import argparse
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
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Module.PINN import Net as PINNNet                                  # noqa: E402
from Module import PoissonTools as PT                                   # noqa: E402
from Module.Training import burgers_residual, laplace_residual          # noqa: E402
from utils.blind_split import ensure_split, masked_rel_l2               # noqa: E402
from utils.posterior_predict import (                                   # noqa: E402
    _load_config, _build_layer, make_eval_grid, get_exact_solution,
    _poisson_kwargs,
)


SUPP_ROOT = PROJECT_ROOT / "Results" / "supplementary"
OUT_ROOT = SUPP_ROOT / "full_baselines" / "uq" / "deep_ensemble_pinn"
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


# Loss helpers (identical to the direct MC-Dropout PINN runner so the
# two baselines are matched on training procedure).
def _pde_loss(case, model, x):
    if case == "Laplace":
        return laplace_residual(model, x)
    if case == "Burgers_inv":
        # For Burgers inverse with a deterministic PINN baseline, we use the
        # analytic nu. The "inverse" aspect (learning nu) is a feature of the
        # proposed method, not of this UQ baseline; we make this explicit in
        # the manuscript by marking nu_pred = NaN for ensemble members.
        return burgers_residual(model, x, nu=BURGERS_NU_TRUE)
    if case == "Poisson":
        return PT.poisson_residual_loss(model, x)
    raise ValueError(f"Unknown case: {case}")


def _boundary_loss(case, model, n_b, config, device):
    if case == "Poisson":
        pts = PT.boundary_points(n_b, device=device)
        pred = model(pts)
        if isinstance(pred, tuple):
            pred = pred[0]
        return torch.mean((pred - PT.poisson_exact(pts)) ** 2)
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
        return torch.mean((pred_ic + torch.sin(math.pi * x_ic)) ** 2)
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


def _nll(m, s, e):
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


def train_member(case, member_seed, preset, device, config, model_kwargs,
                 member_dir: Path) -> Dict:
    """Train ONE ensemble member with the given member_seed."""
    _set_seed(member_seed)
    layer = _build_layer(config, student=True)
    model = PINNNet(layer, **model_kwargs).to(device)

    lr = float(config.get("learning_rate", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps = _training_steps(config, preset)
    lambda_pde = float(config.get("lambda_pde_teacher", 1.0))
    lambda_bc = float(config.get("lambda_bc_teacher", 1.0))
    lambda_data = float(config.get("lambda_data_teacher", 0.0))
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
    train_time = time.perf_counter() - t0

    member_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), member_dir / "model.pth")
    pd.DataFrame(history).to_csv(member_dir / "history.csv", index=False)
    with open(member_dir / "metrics.json", "w") as f:
        json.dump({"member_seed": int(member_seed),
                   "train_time_s": float(train_time),
                   "final_loss": history[-1]["loss"] if history else math.nan},
                  f, indent=2)
    return {"member_seed": int(member_seed),
            "train_time_s": float(train_time),
            "model": model}


def evaluate_ensemble(case, members, ensemble_id, ensemble_seed, preset,
                      device, ensemble_dir: Path, config) -> Dict:
    """Build the ensemble mean/std prediction and report blind-test UQ."""
    x_eval, X1, X2 = make_eval_grid(case, 100, device, config)
    exact_np = get_exact_solution(case, x_eval.cpu().numpy())
    exact = (exact_np.reshape(-1) if exact_np is not None
             else np.full(x_eval.shape[0], math.nan))

    member_means = []
    inference_t0 = time.perf_counter()
    for m in members:
        m["model"].eval()
        with torch.no_grad():
            pred = m["model"](x_eval)
            if isinstance(pred, tuple):
                pred = pred[0]
            member_means.append(pred.cpu().numpy().reshape(-1))
    ensemble_inference_time = time.perf_counter() - inference_t0

    stack = np.vstack(member_means)
    ens_mean = np.mean(stack, axis=0)
    ens_std = np.std(stack, axis=0, ddof=1)
    ens_std = np.maximum(ens_std, 1e-12)

    # Use ensemble_seed for the blind split so the report is on the same grid
    # split GBSD uses for that nominal seed (the reviewer asked for shared
    # train/guard/blind split across the main and baseline experiments).
    split = ensure_split(case, ensemble_seed, n_total=exact.size)
    blind = split["blind_test_mask"]; guard = split["guard_mask"]
    valid = ~np.isnan(exact)
    if int((blind & valid).sum()) == 0 or int((guard & valid).sum()) == 0:
        raise RuntimeError(
            f"No valid guard/blind evaluation points for {case} "
            f"ensemble_seed={ensemble_seed}. Check that the reference "
            "solution file exists, especially "
            "Database/Burgers_inv_reference.npz for Burgers_inv.")

    rel_blind = float(masked_rel_l2(ens_mean, exact, blind & valid))
    rel_guard = float(masked_rel_l2(ens_mean, exact, guard & valid))
    m_b = ens_mean[blind & valid]; s_b = ens_std[blind & valid]
    e_b = exact[blind & valid]

    row = {
        "case": case,
        "seed_or_member": f"ens{int(ensemble_id)}",
        "method": "deep_ensemble_pinn",
        "rel_l2_blind": rel_blind,
        "rel_l2_guard": rel_guard,
        "coverage95_blind": _coverage95(m_b, s_b, e_b),
        "corr_abs_error_std_blind": _corr(m_b, s_b, e_b),
        "nll_blind": _nll(m_b, s_b, e_b),
        "mean_interval_width_blind": float(np.mean(2 * 1.96 * np.maximum(s_b, 1e-12))),
        "avg_std_blind": float(np.mean(s_b)),
        "runtime_train_s": float(sum(m["train_time_s"] for m in members)),
        "runtime_mc_or_ensemble_inference_s": float(ensemble_inference_time),
        "n_mc_samples_or_ensemble_size": int(len(members)),
        "nu_pred": math.nan,
        "nu_relative_error": math.nan,
        "n_blind": int((blind & valid).sum()),
        "n_guard": int((guard & valid).sum()),
        "ensemble_id": int(ensemble_id),
        "member_seeds": ";".join(str(int(m["member_seed"])) for m in members),
        "preset": preset,
        "output_dir": str(ensemble_dir),
    }
    np.savez(ensemble_dir / "ensemble_predictions.npz",
             x=x_eval.cpu().numpy(), X1=X1, X2=X2,
             ensemble_mean=ens_mean, ensemble_std=ens_std,
             exact=exact, members=stack,
             guard_mask=guard, blind_test_mask=blind)
    with open(ensemble_dir / "ensemble_metrics.json", "w") as f:
        json.dump(_json_safe(row), f, indent=2)
    with open(ensemble_dir / "environment.json", "w") as f:
        json.dump({
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python_version": sys.version.split(" ")[0],
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_name": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else None),
            "platform": platform.platform(),
            "ensemble_id": int(ensemble_id),
            "ensemble_seed": int(ensemble_seed),
            "ensemble_size": int(len(members)),
            "argv": sys.argv,
        }, f, indent=2)
    print(f"  [done] deep-ensemble {case} ens{ensemble_id} (K={len(members)}): "
          f"rL2_blind={rel_blind:.4e}, cov95={row['coverage95_blind']:.3f}, "
          f"NLL={row['nll_blind']:.4f}")
    return row


def train_ensemble(case: str, ensemble_id: int, ensemble_seed: int,
                   K: int, preset: str, device: torch.device,
                   force: bool = False) -> Dict:
    ensemble_dir = OUT_ROOT / preset / case / f"ensemble_{int(ensemble_id)}"
    metrics_path = ensemble_dir / "ensemble_metrics.json"
    if metrics_path.is_file() and not force:
        print(f"  [skip] deep-ensemble {case} ens{ensemble_id} already done.")
        with open(metrics_path) as f:
            return json.load(f)
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(case)
    model_kwargs = _poisson_kwargs(case, config)
    # Member seeds: deterministic from (case, ensemble_seed, k).
    rng = np.random.default_rng(ensemble_seed + 9999)
    member_seeds = rng.integers(0, 10_000_000, size=K).tolist()

    members = []
    for k, seed in enumerate(member_seeds):
        member_dir = ensemble_dir / "members" / f"seed_{int(seed)}"
        if (member_dir / "model.pth").is_file() and not force:
            # Reload trained member.
            print(f"  [reload] {case} ens{ensemble_id} member {k+1}/{K} seed={seed}")
            layer = _build_layer(config, student=True)
            model = PINNNet(layer, **model_kwargs).to(device)
            model.load_state_dict(torch.load(member_dir / "model.pth",
                                              map_location=device,
                                              weights_only=False))
            # Read train_time from metrics.json if available.
            tt = 0.0
            mp = member_dir / "metrics.json"
            if mp.is_file():
                with open(mp) as f:
                    tt = float(json.load(f).get("train_time_s", 0.0))
            members.append({"member_seed": int(seed),
                            "train_time_s": tt, "model": model})
        else:
            print(f"  [train] {case} ens{ensemble_id} member {k+1}/{K} seed={seed}")
            members.append(train_member(case, int(seed), preset, device,
                                        config, model_kwargs, member_dir))

    return evaluate_ensemble(case, members, ensemble_id, ensemble_seed,
                             preset, device, ensemble_dir, config)


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
    parser.add_argument("--ensemble_seeds", type=int, nargs="+", default=[0],
                        help="One ensemble per seed value here (default: [0])")
    parser.add_argument("--K", type=int, default=5,
                        help="Ensemble size (number of independent PINNs).")
    parser.add_argument("--preset", default="full", choices=list(BUDGETS.keys()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    cases = list(CASES) if args.case == "all" else [args.case]
    rows = []
    for case in cases:
        for ens_id, seed in enumerate(args.ensemble_seeds):
            try:
                rows.append(train_ensemble(case, ens_id, int(seed), args.K,
                                           args.preset, device, args.force))
            except Exception as e:
                import traceback
                print(f"  [fail] deep-ensemble {case} ens{ens_id}: {e}")
                traceback.print_exc()
    append_to_uq_table(rows)


if __name__ == "__main__":
    main()
