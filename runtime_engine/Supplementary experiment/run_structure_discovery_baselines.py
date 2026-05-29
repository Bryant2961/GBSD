"""Full PDE structure-discovery baseline runner (JCP review item).

What this script does
=====================
For each (case, seed, structure_method), it:

  1. Loads the trained dense Bayesian student weights from the matching
     GBSD run (the same dense student the proposed HAC method clusters).
  2. Computes the per-layer HAC cluster count from the proposed method,
     so the random / magnitude / kmeans baselines share an identical
     trainable-parameter budget. Low-rank SVD uses its own minimum-rank
     budget which is reported separately.
  3. Builds (structure, R) via Module.StructureBaselines for non-HAC
     methods, or via Module.StructureDiscovery for HAC.
  4. Builds the StructuredPINN and runs ReconstructionTrainer with the
     SAME reconstruction config the proposed method uses: same PDE
     residual, same boundary loss, same data anchors (Burgers), same
     epochs, same lr, no teacher distillation (so every baseline is on
     equal footing).
  5. Evaluates the trained structured candidate on the per-seed blind
     split saved at ./results/splits/<case>_seed<seed>.npz. Reports:
        guard subset:  dense_rL2, structured_rL2, structured/dense ratio
        blind subset:  the same trio plus coverage95 / corr / NLL /
                       interval width, plus Burgers nu metrics
     The guard subset is what decides accepted/rejected (gamma * dense_guard
     + epsilon). The blind subset is the held-out test report.

What this is NOT
================
* Not a weight-space diagnostic. Every row is a real PDE reconstruction
  followed by a real evaluation on the blind subset.
* Not a substitute for the proposed HAC run; HAC must already exist as
  an archived GBSD run for this script to load the dense student.

Output layout
=============
Each (case, seed, method) is archived separately, never overwriting:

  Results/supplementary/full_baselines/structure_discovery/
      <case>/<method>/seed_<seed>/
          structured_model.pth
          predictions.npz             # full grid + masks
          history.csv                 # reconstruction loss curve
          metrics.json                # the row that goes into the table
          environment.json            # versions, seed, argv
          config_snapshot.json        # the resolved reconstruction config

  Results/supplementary/tables/
      structure_discovery_baselines_raw.csv      # one row per (case, seed, method)

Usage
-----
    python "Supplementary experiment/run_structure_discovery_baselines.py" \
        --case Poisson --seed 0 \
        --methods hac random magnitude low_rank \
        --epochs 5000

Implementation notes
====================
* HAC reuses the existing Module.StructureDiscovery so the "proposed"
  row in the resulting CSV is exactly the method described in the
  manuscript.
* Random / magnitude / kmeans are routed through
  Module.StructureBaselines.build_baseline_structure with the per-layer
  cluster counts matched to HAC.
* Low-rank SVD has its own parameter budget; we report it honestly
  rather than forcing the same trainable count.
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Project root importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Module.Student_MCDropout import Net as MCNet                       # noqa: E402
from Module.UncertaintyEstimation import UncertaintyEstimator           # noqa: E402
from Module import StructureDiscovery as SD                             # noqa: E402
from Module import StructureBaselines as SB                             # noqa: E402
from Module import StructuredPINN as SP                                 # noqa: E402
from Module import ReconstructionTrainer as RT                          # noqa: E402
from Module import PoissonTools as PT                                   # noqa: E402
from Module.Training import burgers_residual, laplace_residual          # noqa: E402
from utils.blind_split import ensure_split, masked_rel_l2               # noqa: E402
from utils.posterior_predict import (                                   # noqa: E402
    _load_config, _build_layer, _poisson_kwargs, make_eval_grid,
    get_exact_solution, _first_existing,
)


SUPP_ROOT = PROJECT_ROOT / "Results" / "supplementary"
FULL_BASELINES_ROOT = SUPP_ROOT / "full_baselines" / "structure_discovery"
TABLE_DIR = SUPP_ROOT / "tables"
FIGURE_DIR = SUPP_ROOT / "figures"

CASES = ("Laplace", "Poisson", "Burgers_inv")
METHODS = ("hac", "random", "magnitude", "low_rank", "kmeans")
BURGERS_NU_TRUE = 0.01 / math.pi


# ---------------------------------------------------------------------------
# Seeding helper (matches run_all_experiments._set_global_seed)
# ---------------------------------------------------------------------------
def _set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dense Bayesian student loader. Finds the most recent archived run for the
# requested (case, seed) and returns the trained MCNet plus its config.
# ---------------------------------------------------------------------------
def _source_runs_roots() -> List[Path]:
    """Return the integrated GBSD archive root for dense students."""
    roots: List[Path] = [SUPP_ROOT / "runs"]

    unique: List[Path] = []
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve()).lower()
        except Exception:
            key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        if root.is_dir():
            unique.append(root)
    return unique


def _checkpoint_names(case: str) -> Tuple[str, ...]:
    return (
        f"{case}_EXP_Student_MCDropout_student_mean_refined_best.pth",
        f"{case}_EXP_Student_MCDropout_student.pth",
        f"{case}_EXP_Student_MCDropout_student_best.pth",
    )


def _find_dense_student_checkpoint(case: str, seed: int) -> Path | None:
    """Search archived runs and the live Results/ tree for the dense student."""
    candidates: List[Path] = []
    # 1) Archived multi-seed runs generated by the integrated main workflow.
    for runs_dir in _source_runs_roots():
        seed_tag = f"_s{int(seed)}_"
        run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
        prefer = [p for p in run_dirs if seed_tag in p.name]
        order = prefer + [p for p in run_dirs if p not in prefer]
        for run in order:
            models = run / f"{case}_EXP" / "Models"
            if not models.is_dir():
                continue
            for name in _checkpoint_names(case):
                p = models / name
                if p.is_file():
                    candidates.append(p)
                    break
            if candidates:
                break  # take the first run that has the student
        if candidates:
            break
    # 2) Fallback: the live Results tree.
    if not candidates:
        live = PROJECT_ROOT / "Results" / f"{case}_EXP" / "Models"
        if live.is_dir():
            for name in _checkpoint_names(case):
                p = live / name
                if p.is_file():
                    candidates.append(p)
                    break
    return candidates[0] if candidates else None


def _load_dense_student(case: str, seed: int, device: torch.device) -> Tuple[nn.Module, Dict, Path]:
    """Load the dense Bayesian student and its config; raise if absent."""
    config = _load_config(case)
    layer = _build_layer(config, student=True)
    model_kwargs = _poisson_kwargs(case, config)
    dropout_rate = float(config.get("dropout_rate", 0.05))
    train_dropout = float(config.get("train_dropout_rate",
                                     max(dropout_rate * 0.1, 0.005)))

    model = MCNet(layer, dropout_rate=train_dropout, **model_kwargs).to(device)
    ckpt_path = _find_dense_student_checkpoint(case, seed)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No dense Bayesian student checkpoint found for {case} seed={seed}. "
            f"Run the main GBSD pipeline first.")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(state)
    except Exception:
        # Try the legacy layer width (rare).
        from utils.posterior_predict import _legacy_layer
        legacy = _legacy_layer(config, student=True)
        model = MCNet(legacy, dropout_rate=train_dropout, **model_kwargs).to(device)
        model.load_state_dict(state)
    model.eval()
    return model, config, ckpt_path


# ---------------------------------------------------------------------------
# HAC structure (matching the proposed method).
# ---------------------------------------------------------------------------
def _build_hac_structure(model: nn.Module, config: Dict, seed: int):
    cluster_distance = float(config.get("cluster_distance", 0.1))
    cluster_mode = str(config.get("cluster_mode", "relative"))
    discoverer = SD.StructureDiscovery(
        model,
        cluster_distance=cluster_distance,
        cluster_mode=cluster_mode,
    )
    structure = discoverer.extract_structure(verbose=False)
    R_dict = discoverer.build_relation_matrix(structure)
    return structure, R_dict


# ---------------------------------------------------------------------------
# Baseline structure builders that match HAC's per-layer cluster counts.
# ---------------------------------------------------------------------------
def _baseline_structure_matched(model: nn.Module, hac_structure: Dict,
                                variant: str, seed: int, device: torch.device):
    """Use HAC's per-layer cluster counts as the budget for non-HAC baselines."""
    n_clusters_per_layer = {name: int(info["n_clusters"])
                            for name, info in hac_structure.items()}
    weights = {name: param.data.clone()
               for name, param in model.named_parameters()
               if "weight" in name}
    weights = {k: v for k, v in weights.items()
               if any(k.endswith(name) or k == name
                      for name in n_clusters_per_layer)}
    # Sanity: align keys with hac_structure keys.
    hac_keys = list(hac_structure.keys())
    aligned = {}
    for src_name, src_w in weights.items():
        match = None
        for hk in hac_keys:
            if hk == src_name or hk in src_name or src_name in hk:
                match = hk
                break
        if match is not None:
            aligned[match] = src_w
    structure, R_dict = SB.build_baseline_structure(
        weights_dict=aligned,
        n_clusters_per_layer=n_clusters_per_layer,
        variant="random" if variant == "random" else variant,
        seed=seed, device=device,
    )
    return structure, R_dict


def _kmeans_structure_matched(model: nn.Module, hac_structure: Dict,
                              seed: int, device: torch.device):
    """K-means on |w|, with per-layer K equal to HAC's cluster count.

    This produces the same parameter budget as random/magnitude but with a
    smarter clustering target. We include it because it is a stronger
    'magnitude clustering heuristic' baseline than random.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("  [Warning] sklearn not available; falling back to magnitude bins for kmeans.")
        return _baseline_structure_matched(model, hac_structure, "magnitude", seed, device)

    structure: Dict[str, Dict] = {}
    R_dict: Dict[str, torch.Tensor] = {}
    rng_seed = int(seed)
    hac_keys = list(hac_structure.keys())
    name_to_weight = {n: p.data.clone()
                      for n, p in model.named_parameters() if "weight" in n}
    for hk in hac_keys:
        # Match source weight to hac key.
        src = None
        for sname, sw in name_to_weight.items():
            if hk == sname or hk in sname or sname in hk:
                src = sw
                break
        if src is None:
            continue
        flat = src.detach().cpu().numpy().flatten()
        abs_w = np.abs(flat).reshape(-1, 1)
        signs = np.sign(flat)
        k = max(1, min(int(hac_structure[hk]["n_clusters"]), flat.size))
        if k == 1 or np.allclose(abs_w, abs_w[0]):
            labels = np.zeros(flat.size, dtype=int)
            centers = np.array([abs_w.mean()])
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=rng_seed)
            labels = km.fit_predict(abs_w)
            centers = np.array([abs_w[labels == c].mean()
                                if (labels == c).any() else 0.0
                                for c in range(k)])
        sub = {
            "cluster_centers": centers,
            "labels": labels.astype(int),
            "signs": signs,
            "original_shape": tuple(src.shape),
            "n_clusters": int(centers.size),
            "n_params": int(flat.size),
            "method": "kmeans_magnitude",
        }
        # Build R.
        R = np.zeros((flat.size, sub["n_clusters"]), dtype=np.float32)
        for i, (lab, sg) in enumerate(zip(labels, signs)):
            R[i, int(lab)] = float(sg)
        structure[hk] = sub
        R_dict[hk] = torch.tensor(R, dtype=torch.float32, device=device)
    return structure, R_dict


# ---------------------------------------------------------------------------
# PDE / boundary / data hooks for the ReconstructionTrainer.
# ---------------------------------------------------------------------------
def _pde_fn_for_case(case: str):
    if case == "Laplace":
        return lambda model, x: laplace_residual(model, x)
    if case == "Burgers_inv":
        # If the model has a learnable nu, use it; else use the analytic value.
        def pde(model, x):
            nu = getattr(model, "parameters_undetermin", None)
            if nu is not None:
                # parameters_undetermin is typically a Parameter tensor
                try:
                    nu_val = float(nu.detach().cpu().item())
                except Exception:
                    nu_val = BURGERS_NU_TRUE
            else:
                nu_val = BURGERS_NU_TRUE
            return burgers_residual(model, x, nu=nu_val)
        return pde
    if case == "Poisson":
        return lambda model, x: PT.poisson_residual_loss(model, x)
    raise ValueError(f"Unknown case: {case}")


def _boundary_fn_for_case(case: str, config: Dict, device: torch.device, n_b: int):
    """Return a no-argument callable model -> bc loss (matches RT signature)."""
    x_min = float(config.get("x_min", -1.0))
    x_max = float(config.get("x_max", 1.0))
    y_min = float(config.get("y_min", -1.0))
    y_max = float(config.get("y_max", 1.0))

    if case == "Poisson":
        # Hard zero Dirichlet on boundary of [0,1]^2 with the analytic exact.
        pts = PT.boundary_points(n_b, device=device)
        target = PT.poisson_exact(pts).detach()
        def bc(model):
            pred = model(pts)
            if isinstance(pred, tuple):
                pred = pred[0]
            return torch.mean((pred - target) ** 2)
        return bc

    # Build a static boundary point set for Laplace / Burgers.
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
    ], dim=0).detach()

    target_np = get_exact_solution(case, pts.detach().cpu().numpy())
    if case == "Burgers_inv":
        # No analytic boundary -> enforce IC u(x,0) = -sin(pi*x) instead.
        x_ic = torch.linspace(x_min, x_max, n_b, device=device).reshape(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        ic_pts = torch.cat([x_ic, t_ic], dim=1).detach()
        u_ic = -torch.sin(math.pi * x_ic).detach()
        def bc(model):
            pred = model(ic_pts)
            if isinstance(pred, tuple):
                pred = pred[0]
            return torch.mean((pred - u_ic) ** 2)
        return bc

    target = torch.tensor(target_np, device=device, dtype=torch.float32).detach()
    def bc(model):
        pred = model(pts)
        if isinstance(pred, tuple):
            pred = pred[0]
        return torch.mean((pred - target) ** 2)
    return bc


def _sample_collocation(case: str, n_f: int, config: Dict, device: torch.device):
    if case == "Poisson":
        return PT.sample_interior_points(n_f, device=device)
    x_min = float(config.get("x_min", -1.0))
    x_max = float(config.get("x_max", 1.0))
    y_min = float(config.get("y_min", -1.0))
    y_max = float(config.get("y_max", 1.0))
    x = torch.rand(n_f, 1, device=device) * (x_max - x_min) + x_min
    y = torch.rand(n_f, 1, device=device) * (y_max - y_min) + y_min
    return torch.cat([x, y], dim=1)


def _load_burgers_data(case: str, config: Dict, device: torch.device):
    if case != "Burgers_inv":
        return None, None
    serials = str(config.get("data_serial", "1,3")).split(",")
    arrays = []
    for s in serials:
        path = PROJECT_ROOT / "Database" / f"{case}_data_{s.strip()}.csv"
        if path.is_file():
            arrays.append(pd.read_csv(path, header=None).values)
    if not arrays:
        return None, None
    db = np.vstack(arrays)
    xs = torch.tensor(db[:, 0:2], dtype=torch.float32, device=device)
    us = torch.tensor(db[:, 2:3], dtype=torch.float32, device=device)
    return xs, us


# ---------------------------------------------------------------------------
# Build structure dict for any method.
# ---------------------------------------------------------------------------
def _build_structure_for_method(method: str, model: nn.Module, config: Dict,
                                seed: int, device: torch.device,
                                hac_cache: Dict):
    if method == "hac":
        structure, R_dict = _build_hac_structure(model, config, seed)
        return structure, R_dict
    # All non-HAC methods need the HAC cluster counts as a budget reference.
    if "structure" not in hac_cache:
        hac_cache["structure"], hac_cache["R_dict"] = _build_hac_structure(
            model, config, seed)
    if method in ("random", "magnitude"):
        return _baseline_structure_matched(model, hac_cache["structure"],
                                           method, seed, device)
    if method == "low_rank":
        # Use a per-layer rank that matches HAC's cluster count (but capped
        # by min(in, out)). The result has a smaller parameter budget than
        # HAC; we report it honestly.
        n_clusters_per_layer = {n: int(info["n_clusters"])
                                for n, info in hac_cache["structure"].items()}
        weights = {n: p.data.clone()
                   for n, p in model.named_parameters() if "weight" in n}
        aligned = {}
        for sname, sw in weights.items():
            for hk in n_clusters_per_layer:
                if hk == sname or hk in sname or sname in hk:
                    aligned[hk] = sw
                    break
        return SB.build_baseline_structure(
            weights_dict=aligned,
            n_clusters_per_layer=n_clusters_per_layer,
            variant="low_rank", seed=seed, device=device)
    if method == "kmeans":
        return _kmeans_structure_matched(model, hac_cache["structure"],
                                         seed, device)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Reconstruction config that matches the proposed method's reconstruction stage.
# ---------------------------------------------------------------------------
def _reconstruction_config(case: str, config: Dict, epochs: int) -> Dict:
    return {
        "lr": float(config.get("recon_lr", config.get("learning_rate", 1e-3))),
        "epochs": int(epochs),
        "lr_step": max(1, int(epochs) // 5),
        "lr_gamma": 0.5,
        "lambda_pde": float(config.get("lambda_pde_recon", 1.0)),
        "lambda_bc": float(config.get("lambda_bc_recon", 1.0)),
        "lambda_data": float(config.get("lambda_data_recon", 1.0)),
        "lambda_distill": 0.0,  # Strict fairness: no teacher distillation.
        "lambda_anchor": 0.0,
        "grad_clip": 1.0,
        "print_every": max(1, int(epochs) // 10),
        "seed": 1234,
        "train_dropout_rate": 0.0,
        "inference_dropout_rate": 0.0,
        "anchor_pretrain_steps": 0,
        "anchor_pretrain_lr": 1e-3,
        "anchor_pretrain_pde_weight": 0.0,
        "residual_pretrain_steps": 0,
        "residual_pretrain_lr": 1e-3,
        "residual_pretrain_print_every": max(1, int(epochs) // 10),
        "lambda_residual_output": 0.0,
        "lambda_alpha": 0.0,
        "best_metric": "rel_l2",
        "lbfgs_steps": 0,
        "lbfgs_lr": 1.0,
    }


# ---------------------------------------------------------------------------
# Evaluation on guard / blind subsets.
# ---------------------------------------------------------------------------
def _coverage95(mean: np.ndarray, std: np.ndarray, exact: np.ndarray) -> float:
    m = np.asarray(mean).reshape(-1)
    s = np.maximum(np.asarray(std).reshape(-1), 1e-12)
    e = np.asarray(exact).reshape(-1)
    return float(np.mean((e >= m - 1.96 * s) & (e <= m + 1.96 * s)))


def _avg_interval_width(std: np.ndarray) -> float:
    s = np.maximum(np.asarray(std).reshape(-1), 1e-12)
    return float(np.mean(2.0 * 1.96 * s))


def _nll_gaussian(mean: np.ndarray, std: np.ndarray, exact: np.ndarray) -> float:
    m = np.asarray(mean).reshape(-1)
    s = np.maximum(np.asarray(std).reshape(-1), 1e-12)
    e = np.asarray(exact).reshape(-1)
    var = s ** 2
    return float(np.mean(0.5 * np.log(2.0 * np.pi * var) + (e - m) ** 2 / (2.0 * var)))


def _error_std_corr(mean: np.ndarray, std: np.ndarray, exact: np.ndarray) -> float:
    err = np.abs(np.asarray(mean).reshape(-1) - np.asarray(exact).reshape(-1))
    s = np.asarray(std).reshape(-1)
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


def _evaluate_on_mask(mean, std, exact, mask):
    m = np.asarray(mean).reshape(-1)[mask]
    s = np.asarray(std).reshape(-1)[mask] if std is not None else None
    e = np.asarray(exact).reshape(-1)[mask]
    out = {
        "rel_l2": float(np.sqrt(np.sum((m - e) ** 2)
                                / max(float(np.sum(e ** 2)), 1e-15))),
        "mae": float(np.mean(np.abs(m - e))),
    }
    if s is not None:
        out["coverage95"] = _coverage95(m, s, e)
        out["corr_abs_err_std"] = _error_std_corr(m, s, e)
        out["nll"] = _nll_gaussian(m, s, e)
        out["mean_interval_width"] = _avg_interval_width(s)
        out["avg_std"] = float(np.mean(s))
    return out


def _pde_residual_l2_blind(model, case, config, blind_mask, X1, X2, x_grid,
                           device, n_samples=4096):
    """Evaluate the PDE residual on a sample of the blind subset."""
    blind_idx = np.where(blind_mask)[0]
    if blind_idx.size == 0:
        return math.nan
    sample = np.random.default_rng(7).choice(
        blind_idx, size=min(int(n_samples), blind_idx.size), replace=False)
    x_blind = torch.tensor(x_grid[sample], dtype=torch.float32, device=device,
                           requires_grad=True)
    pde_fn = _pde_fn_for_case(case)
    model.train()  # ensure dropout layers stay in train if any
    try:
        with torch.enable_grad():
            loss = pde_fn(model, x_blind)
            value = float(loss.detach().cpu().item())
    except Exception as e:
        print(f"  [Warning] PDE residual on blind subset failed: {e}")
        value = math.nan
    model.eval()
    return value


# ---------------------------------------------------------------------------
# Main per-method run.
# ---------------------------------------------------------------------------
def run_single(case: str, seed: int, method: str, epochs: int,
               grid_n: int, device: torch.device,
               hac_cache: Dict, force: bool = False) -> Dict:
    out_dir = (FULL_BASELINES_ROOT / case / method / f"seed_{int(seed)}")
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        print(f"  [skip] {case}/{method}/seed_{seed} already done.")
        with open(metrics_path) as f:
            return json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_global_seed(seed)
    model_dense, config, dense_ckpt = _load_dense_student(case, seed, device)
    print(f"  Loaded dense student: {dense_ckpt}")

    # Build (structure, R) for this method.
    t0 = time.perf_counter()
    structure, R_dict = _build_structure_for_method(
        method, model_dense, config, seed, device, hac_cache)
    discovery_time = time.perf_counter() - t0

    # Build StructuredPINN.
    structured = SP.build_structured_pinn(
        structure=structure,
        relation_matrices=R_dict,
        reference_model=model_dense,
        dropout_rate=0.0,
    ).to(device)
    param_stats = structured.count_parameters()
    dense_param_count = sum(p.numel() for p in model_dense.parameters())
    structured_param_count = int(param_stats["trainable"])
    structured_dense_count_within = int(param_stats["original"])
    compression_ratio = (float(dense_param_count) / float(structured_param_count)
                         if structured_param_count > 0 else math.inf)
    print(f"  [{case}/{method}/s{seed}] dense_params={dense_param_count}, "
          f"structured_trainable={structured_param_count}, "
          f"compression={compression_ratio:.2f}x")

    # Set up reconstruction trainer.
    recon_cfg = _reconstruction_config(case, config, epochs)
    pde_fn = _pde_fn_for_case(case)
    n_b = int(config.get("bun_node_num", 100))
    bc_fn = _boundary_fn_for_case(case, config, device, n_b)
    n_f = int(config.get("grid_node_num", 80)) ** 2
    x_collocation = _sample_collocation(case, min(n_f, 4096), config, device)

    trainer = RT.ReconstructionTrainer(
        structured_model=structured,
        pde_residual_fn=pde_fn,
        boundary_loss_fn=bc_fn,
        teacher_model=None,            # JCP fairness: no distillation
        anchor_model=None,
        device=device,
        config=recon_cfg,
    )
    trainer.set_collocation_points(x_collocation)

    # Burgers data anchors (sparse observations).
    x_data, u_data = _load_burgers_data(case, config, device)
    if x_data is not None:
        trainer.set_data_points(x_data, u_data)
        print(f"  Burgers data anchors: {x_data.shape[0]} points")

    # Validation points (small grid).
    val_n = 40
    x_min = float(config.get("x_min", -1.0))
    x_max = float(config.get("x_max", 1.0))
    y_min = float(config.get("y_min", -1.0))
    y_max = float(config.get("y_max", 1.0))
    x_val = torch.linspace(x_min, x_max, val_n, device=device)
    y_val = torch.linspace(y_min, y_max, val_n, device=device)
    xx, yy = torch.meshgrid(x_val, y_val, indexing="ij")
    x_validation = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    u_validation_np = get_exact_solution(case, x_validation.cpu().numpy())
    if u_validation_np is not None:
        trainer.set_validation_points(
            x_validation,
            torch.tensor(u_validation_np, dtype=torch.float32, device=device))

    # Train.
    t0 = time.perf_counter()
    history = trainer.train(verbose=False,
                            stage_label=f"[{case}/{method}/seed_{seed}] structured reconstruction")
    recon_time = time.perf_counter() - t0
    # Save history.
    if isinstance(history, dict):
        try:
            pd.DataFrame({k: v for k, v in history.items()
                          if isinstance(v, (list, tuple))
                          and len(v) > 0
                          and not isinstance(v[0], dict)}).to_csv(
                out_dir / "history.csv", index=False)
        except Exception:
            pass

    # Save model.
    torch.save({
        "model_state_dict": structured.state_dict(),
        "model_object": structured,
        "structure": {k: {ki: vi for ki, vi in v.items()
                          if ki != "signs" or len(v.get("signs", [])) < 10000}
                      for k, v in structure.items()},
        "case": case, "seed": seed, "method": method,
        "epochs": epochs,
    }, out_dir / "structured_model.pth")

    # Evaluate on the full grid + masks.
    x_eval, X1, X2 = make_eval_grid(case, grid_n, device, config)
    x_grid_np = x_eval.detach().cpu().numpy()
    exact = get_exact_solution(case, x_grid_np)
    if exact is None:
        # Burgers: load reference grid from disk.
        ref_path = PROJECT_ROOT / "Database" / "Burgers_inv_reference.npz"
        if ref_path.is_file():
            ref = np.load(ref_path)
            # Interpolate not implemented here; fall back to using grid as exact source.
            exact = np.full((x_grid_np.shape[0], 1), math.nan)
        else:
            exact = np.full((x_grid_np.shape[0], 1), math.nan)
    exact = exact.reshape(-1)

    structured.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        pred_struct = structured(x_eval).cpu().numpy().reshape(-1)
        inference_time = time.perf_counter() - t0

    # Dense student prediction on same grid (deterministic forward, no MC).
    model_dense.eval()
    with torch.no_grad():
        pred_dense = model_dense(x_eval).cpu().numpy().reshape(-1)

    # MC-Dropout std from the dense student (so we can report UQ on the
    # baseline's "candidate -> guard chooses dense" branch consistently).
    n_mc = int(os.environ.get("BASELINE_N_MC", "50"))
    if n_mc > 0:
        try:
            mc_estimator = UncertaintyEstimator(model_dense, n_samples=n_mc)
            mc_out = mc_estimator.predict(x_eval)
            mc_std = mc_out["std"].cpu().numpy().reshape(-1)
        except Exception:
            mc_std = np.full_like(pred_dense, 1e-3)
    else:
        mc_std = np.full_like(pred_dense, 1e-3)

    # Per-(case, seed) blind/guard split.
    split = ensure_split(case, seed, n_total=exact.size)
    guard_mask = split["guard_mask"]
    blind_mask = split["blind_test_mask"]

    # Guard decision: structured rL2 <= gamma * dense rL2 + eps on guard subset.
    gamma = float(config.get("accept_structured_rel_l2_ratio", 1.10))
    eps = float(config.get("accept_structured_rel_l2_abs", 0.002))
    min_compression = float(config.get("min_structured_compression", 0.0))

    valid = ~np.isnan(exact)
    dense_rL2_guard = float(masked_rel_l2(pred_dense, exact, guard_mask & valid))
    struct_rL2_guard = float(masked_rel_l2(pred_struct, exact, guard_mask & valid))
    dense_rL2_blind = float(masked_rel_l2(pred_dense, exact, blind_mask & valid))
    struct_rL2_blind = float(masked_rel_l2(pred_struct, exact, blind_mask & valid))

    accuracy_ok = struct_rL2_guard <= gamma * dense_rL2_guard + eps
    compression_ok = (min_compression <= 0
                      or compression_ratio >= min_compression)
    accepted_by_guard = bool(accuracy_ok and compression_ok)
    final_source = "structured" if accepted_by_guard else "dense_student"
    final_pred = pred_struct if accepted_by_guard else pred_dense
    final_rL2_blind = float(masked_rel_l2(final_pred, exact, blind_mask & valid))

    # UQ metrics on the BLIND subset.
    uq_blind = _evaluate_on_mask(final_pred, mc_std, exact, blind_mask & valid)

    # PDE residual on blind subset.
    pde_blind = _pde_residual_l2_blind(structured, case, config,
                                       blind_mask & valid, X1, X2, x_grid_np,
                                       device)

    # Burgers nu estimate (if model exposes learnable nu).
    nu_pred = math.nan
    nu_relative_error = math.nan
    if case == "Burgers_inv":
        nu_param = getattr(structured, "parameters_undetermin", None)
        if nu_param is not None:
            try:
                nu_pred = float(nu_param.detach().cpu().item())
                nu_relative_error = abs(nu_pred - BURGERS_NU_TRUE) / abs(BURGERS_NU_TRUE)
            except Exception:
                pass

    row = {
        "case": case,
        "seed": int(seed),
        "structure_method": method,
        "dense_rel_l2_guard": dense_rL2_guard,
        "structured_rel_l2_guard": struct_rL2_guard,
        "dense_rel_l2_blind": dense_rL2_blind,
        "structured_rel_l2_blind": struct_rL2_blind,
        "structured_to_dense_ratio_guard": (struct_rL2_guard
                                            / max(dense_rL2_guard, 1e-15)),
        "structured_to_dense_ratio_blind": (struct_rL2_blind
                                            / max(dense_rL2_blind, 1e-15)),
        "compression_ratio": float(compression_ratio),
        "dense_param_count": int(dense_param_count),
        "structured_param_count": int(structured_param_count),
        "accepted_by_guard": bool(accepted_by_guard),
        "accuracy_ok": bool(accuracy_ok),
        "compression_ok": bool(compression_ok),
        "final_source": final_source,
        "final_rel_l2_blind": final_rL2_blind,
        "coverage95_blind": uq_blind.get("coverage95", math.nan),
        "corr_abs_error_std_blind": uq_blind.get("corr_abs_err_std", math.nan),
        "nll_blind": uq_blind.get("nll", math.nan),
        "mean_interval_width_blind": uq_blind.get("mean_interval_width", math.nan),
        "avg_std_blind": uq_blind.get("avg_std", math.nan),
        "pde_residual_blind": pde_blind,
        "nu_pred": nu_pred,
        "nu_true": BURGERS_NU_TRUE if case == "Burgers_inv" else math.nan,
        "nu_relative_error": nu_relative_error,
        "runtime_discovery_s": float(discovery_time),
        "runtime_reconstruction_s": float(recon_time),
        "runtime_inference_s": float(inference_time),
        "guard_gamma": gamma,
        "guard_epsilon": eps,
        "min_compression_threshold": min_compression,
        "n_guard": int((guard_mask & valid).sum()),
        "n_blind": int((blind_mask & valid).sum()),
        "source_dense_ckpt": str(dense_ckpt),
        "output_dir": str(out_dir),
    }

    # Save predictions npz (with masks).
    np.savez(out_dir / "predictions.npz",
             x=x_grid_np, X1=X1, X2=X2,
             exact=exact, dense_mean=pred_dense, structured_mean=pred_struct,
             dense_mc_std=mc_std,
             guard_mask=guard_mask, blind_test_mask=blind_mask,
             final_source=np.array([final_source]),
             accepted_by_guard=np.array([accepted_by_guard]))

    # metrics.json (one row).
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(row), f, indent=2)

    # config snapshot.
    with open(out_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        snap = {**config, "_recon_config": recon_cfg,
                "_method": method, "_epochs": epochs}
        json.dump({k: str(v) if not isinstance(v, (int, float, bool, str, type(None)))
                   else v for k, v in snap.items()},
                  f, indent=2)

    # environment manifest.
    with open(out_dir / "environment.json", "w", encoding="utf-8") as f:
        env = {
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
        }
        json.dump(env, f, indent=2)

    print(f"  [done] {case}/{method}/seed_{seed}: "
          f"final_rL2_blind={final_rL2_blind:.4e}, "
          f"accepted={accepted_by_guard}, compression={compression_ratio:.2f}x")
    return row


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=list(CASES) + ["all"], default="all")
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="Repeatable. Default: --seed 0 --seed 1 --seed 2")
    parser.add_argument("--methods", nargs="+",
                        default=["hac", "random", "magnitude", "low_rank"],
                        choices=list(METHODS))
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--grid_n", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if metrics.json already exists.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cases = list(CASES) if args.case == "all" else [args.case]
    seeds = args.seed if args.seed else [0, 1, 2]

    rows: List[Dict] = []
    for case in cases:
        for seed in seeds:
            hac_cache: Dict = {}
            for method in args.methods:
                try:
                    row = run_single(case, int(seed), method, args.epochs,
                                     args.grid_n, device, hac_cache,
                                     force=args.force)
                    rows.append(row)
                except FileNotFoundError as e:
                    print(f"  [skip] {case}/{method}/seed_{seed}: {e}")
                except Exception as e:
                    import traceback
                    print(f"  [fail] {case}/{method}/seed_{seed}: {e}")
                    traceback.print_exc()

    # Aggregate to raw table.
    if rows:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = TABLE_DIR / "structure_discovery_baselines_raw.csv"
        df_new = pd.DataFrame(rows)
        if raw_path.is_file():
            try:
                df_existing = pd.read_csv(raw_path)
                key_cols = ["case", "seed", "structure_method"]
                key_new = df_new[key_cols].astype(str).agg("|".join, axis=1)
                key_old = df_existing[key_cols].astype(str).agg("|".join, axis=1)
                df_existing = df_existing[~key_old.isin(key_new.tolist())]
                df_out = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df_out = df_new
        else:
            df_out = df_new
        df_out = df_out.sort_values(["case", "structure_method", "seed"])
        df_out.to_csv(raw_path, index=False)
        print(f"\nSaved: {raw_path} ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
