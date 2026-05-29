"""Aggregate the UQ baseline runs and (optionally) GBSD reference into
mean/std tables + comparison figure.

Inputs (any/all of):
    Results/supplementary/tables/uq_baselines_raw.csv          # baselines
    Results/supplementary/tables/multiseed_guarded_raw.csv     # GBSD ref
    Results/supplementary/runs/<run_id>/raw/<case>_predictions.npz
        (used as a fallback to derive GBSD blind-test UQ rows if the
         raw multiseed table is missing the required UQ columns)

Outputs:
    Results/supplementary/tables/uq_baselines_mean_std.csv
    Results/supplementary/figures/fig_uq_baseline_comparison.{png,pdf,svg}

The figure has six panels arranged as 2 x 3:
  (a) Relative L2 on blind subset
  (b) Coverage95 on blind subset
  (c) Error-std correlation on blind subset
  (d) NLL on blind subset
  (e) Mean interval width on blind subset
  (f) Burgers nu relative error (only methods that estimate nu)

Caveats:
  * Methods that have only seed=0 are shown with a "seed-0 only" annotation
    in the legend.
  * Deep Ensemble is K=5 per ensemble; member count is shown as
    "(K=5, n_ens=N)" where N = number of ensembles aggregated.
  * GBSD's row, when present, is shown alongside as the reference; we never
    rename it as a baseline.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (TABLE_DIR, FIGURE_DIR, ensure_output_dirs,
                    archived_runs, calibration_eval_mask, load_npz,
                    coverage95, error_std_corr, nll_gaussian,
                    avg_interval_width, rel_l2)


RAW_PATH = TABLE_DIR / "uq_baselines_raw.csv"
GBSD_RAW_PATH = TABLE_DIR / "multiseed_guarded_raw.csv"
MEAN_PATH = TABLE_DIR / "uq_baselines_mean_std.csv"
FIG_NAME = "fig_uq_baseline_comparison"

METHOD_DISPLAY = {
    "direct_mc_dropout_pinn": "Direct MC-Dropout PINN",
    "deep_ensemble_pinn":     "Deep Ensemble PINN",
    "gbsd_dense_bayesian_student": "GBSD (dense Bayesian student)",
    "gbsd_guarded_final":     "GBSD (guarded final source)",
}
METHOD_COLORS = {
    "direct_mc_dropout_pinn": "#e377c2",
    "deep_ensemble_pinn":     "#1f77b4",
    "gbsd_dense_bayesian_student": "#d62728",
    "gbsd_guarded_final":     "#9467bd",
}
METHOD_ORDER = ["direct_mc_dropout_pinn", "deep_ensemble_pinn",
                "gbsd_dense_bayesian_student", "gbsd_guarded_final"]

CASES = ("Laplace", "Poisson", "Burgers_inv")


def _derive_gbsd_uq_rows() -> List[Dict]:
    """Walk archived GBSD runs and build per-seed UQ rows on the blind subset.

    This lets us include GBSD as a method='gbsd_guarded_final' row in the
    same comparison table without re-running the main pipeline.
    """
    rows: List[Dict] = []
    records = archived_runs()
    for rec in records:
        if rec.seed is None:
            continue
        for case in CASES:
            data = load_npz(rec, case)
            if data is None:
                continue
            if "exact" not in data or "bayesian_mean" not in data \
                    or "bayesian_std" not in data:
                continue
            exact = np.asarray(data["exact"]).reshape(-1)
            mean = np.asarray(data["bayesian_mean"]).reshape(-1)
            std = np.asarray(data["bayesian_std"]).reshape(-1)
            # Prefer the saved blind_test_mask; fall back to ~calibration mask.
            if "blind_test_mask" in data:
                blind = np.asarray(data["blind_test_mask"]).astype(bool).reshape(-1)
            else:
                em = calibration_eval_mask(data)
                if em is None:
                    continue
                blind = em
            valid = ~np.isnan(exact)
            mask = blind & valid
            if mask.sum() < 10:
                continue
            m = mean[mask]; s = std[mask]; e = exact[mask]
            row = {
                "case": case,
                "seed_or_member": int(rec.seed),
                "method": "gbsd_guarded_final",
                "rel_l2_blind": rel_l2(m, e),
                "coverage95_blind": coverage95(m, s, e),
                "corr_abs_error_std_blind": error_std_corr(m, s, e),
                "nll_blind": nll_gaussian(m, s, e),
                "mean_interval_width_blind": avg_interval_width(s),
                "avg_std_blind": float(np.mean(s)),
                "n_blind": int(mask.sum()),
                "runtime_train_s": math.nan,
                "runtime_mc_or_ensemble_inference_s": math.nan,
                "n_mc_samples_or_ensemble_size": math.nan,
                "nu_pred": math.nan,
                "nu_relative_error": math.nan,
                "output_dir": str(rec.root),
                "preset": "from_archived_run",
            }
            # Also emit a dense-student row if the dense mean+std are saved.
            if ("bayesian_dense_mean" in data
                    and "bayesian_dense_std" in data):
                dm = np.asarray(data["bayesian_dense_mean"]).reshape(-1)[mask]
                ds = np.asarray(data["bayesian_dense_std"]).reshape(-1)[mask]
                rows.append({**row,
                             "method": "gbsd_dense_bayesian_student",
                             "rel_l2_blind": rel_l2(dm, e),
                             "coverage95_blind": coverage95(dm, ds, e),
                             "corr_abs_error_std_blind": error_std_corr(dm, ds, e),
                             "nll_blind": nll_gaussian(dm, ds, e),
                             "mean_interval_width_blind": avg_interval_width(ds),
                             "avg_std_blind": float(np.mean(ds))})
            rows.append(row)
    return rows


def _select_preset(df: pd.DataFrame, requested: str | None) -> tuple[pd.DataFrame, str | None]:
    """Keep quick_check and full baseline rows from being averaged together."""
    if "preset" not in df.columns:
        return df, None
    presets = [str(p) for p in df["preset"].dropna().unique()]
    baseline_presets = [p for p in presets
                        if p not in ("from_archived_run", "nan", "")]
    if requested:
        selected = requested
    elif "full" in baseline_presets:
        selected = "full"
    elif "quick_check" in baseline_presets:
        selected = "quick_check"
    elif baseline_presets:
        selected = sorted(baseline_presets)[0]
    else:
        selected = None
    if selected is None:
        return df, None
    preset_values = df["preset"].astype(str)
    keep = (preset_values == selected) | (preset_values == "from_archived_run")
    return df[keep].copy(), selected


def aggregate(include_gbsd: bool = True, preset: str | None = None):
    frames = []
    if RAW_PATH.is_file():
        frames.append(pd.read_csv(RAW_PATH))
    if include_gbsd:
        gbsd_rows = _derive_gbsd_uq_rows()
        if gbsd_rows:
            frames.append(pd.DataFrame(gbsd_rows))
    if not frames:
        raise SystemExit(f"No UQ baseline or GBSD data available.")
    df = pd.concat(frames, ignore_index=True)
    df, selected_preset = _select_preset(df, preset)
    if selected_preset:
        print(f"Aggregating UQ rows for preset='{selected_preset}' "
              "(GBSD archived rows are kept when available).")

    # Coerce numeric columns.
    numeric = ["rel_l2_blind", "coverage95_blind", "corr_abs_error_std_blind",
               "nll_blind", "mean_interval_width_blind", "avg_std_blind",
               "runtime_train_s", "runtime_mc_or_ensemble_inference_s",
               "n_mc_samples_or_ensemble_size", "nu_pred", "nu_relative_error"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    grouped = df.groupby(["case", "method"], dropna=False)
    rows = []
    for (case, method), sub in grouped:
        row = {"case": case, "method": method, "n_seeds_or_ens": int(len(sub))}
        for c in numeric:
            if c in sub.columns:
                v = sub[c]
                row[f"{c}_mean"] = float(v.mean()) if v.notna().any() else math.nan
                row[f"{c}_std"] = float(v.std(ddof=1)) if v.notna().sum() >= 2 else math.nan
        rows.append(row)
    df_mean = pd.DataFrame(rows).sort_values(["case", "method"])
    df_mean.to_csv(MEAN_PATH, index=False)
    print(f"Saved: {MEAN_PATH} ({len(df_mean)} rows)")
    return df, df_mean


def _annotated_label(method: str, df_raw: pd.DataFrame, case: str) -> str:
    base = METHOD_DISPLAY.get(method, method)
    sub = df_raw[(df_raw["case"] == case) & (df_raw["method"] == method)]
    n = len(sub)
    if n == 0:
        return base
    if method == "deep_ensemble_pinn":
        K_vals = pd.to_numeric(sub["n_mc_samples_or_ensemble_size"],
                               errors="coerce").dropna()
        K = int(K_vals.iloc[0]) if len(K_vals) else 5
        return f"{base}\n(K={K}, n_ens={n})"
    if n == 1:
        return f"{base}\n(seed-0 only)"
    return f"{base}\n(n_seed={n})"


def plot(df_raw: pd.DataFrame, df_mean: pd.DataFrame):
    ensure_output_dirs()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    methods = [m for m in METHOD_ORDER if m in df_mean["method"].unique()]
    methods += [m for m in df_mean["method"].unique() if m not in methods]
    cases = [c for c in CASES if c in df_mean["case"].unique()]
    has_burgers = "Burgers_inv" in cases

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("rel_l2_blind", "Relative L2 (blind)", True, axes[0, 0]),
        ("coverage95_blind", "Coverage95 (blind)", False, axes[0, 1]),
        ("corr_abs_error_std_blind", "Corr(|err|, std) (blind)", False, axes[0, 2]),
        ("nll_blind", "NLL (blind)", False, axes[1, 0]),
        ("mean_interval_width_blind", "Mean interval width (blind)", True, axes[1, 1]),
    ]

    width = 0.8 / max(1, len(methods))
    x_cases = np.arange(len(cases))

    for panel_idx, (col, title, log, ax) in enumerate(panels):
        for j, method in enumerate(methods):
            means = []; stds = []
            for case in cases:
                sub = df_mean[(df_mean["case"] == case)
                              & (df_mean["method"] == method)]
                if sub.empty or f"{col}_mean" not in sub.columns:
                    means.append(np.nan); stds.append(0.0)
                else:
                    means.append(float(sub[f"{col}_mean"].iloc[0]))
                    s = sub[f"{col}_std"].iloc[0]
                    stds.append(float(s) if pd.notna(s) else 0.0)
            label_for_legend = METHOD_DISPLAY.get(method, method)
            ax.bar(x_cases + (j - (len(methods) - 1) / 2) * width,
                   means, width, yerr=stds, capsize=3,
                   color=METHOD_COLORS.get(method, "#444"),
                   label=label_for_legend if panel_idx == 0 else None)
        ax.set_xticks(x_cases); ax.set_xticklabels(cases, rotation=10)
        if log:
            ax.set_yscale("log")
        if col == "coverage95_blind":
            ax.axhline(0.95, color="black", ls="--", lw=1, alpha=0.5)
            ax.set_ylim(0, 1.05)
        ax.set_title(f"({chr(ord('a') + panel_idx)}) {title}")
        ax.grid(True, axis="y", alpha=0.3)

    # Panel (f): Burgers nu relative error (only methods that report nu).
    ax = axes[1, 2]
    if has_burgers:
        burgers = df_raw[(df_raw["case"] == "Burgers_inv")]
        nu_methods = []
        nu_means = []
        nu_stds = []
        for method in methods:
            sub = burgers[burgers["method"] == method]
            vals = pd.to_numeric(sub.get("nu_relative_error", pd.Series([])),
                                 errors="coerce").dropna()
            if len(vals) >= 1:
                nu_methods.append(method)
                nu_means.append(float(vals.mean()))
                nu_stds.append(float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0)
        if nu_methods:
            xs = np.arange(len(nu_methods))
            ax.bar(xs, nu_means, 0.6, yerr=nu_stds, capsize=3,
                   color=[METHOD_COLORS.get(m, "#444") for m in nu_methods])
            ax.set_xticks(xs)
            ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in nu_methods],
                               rotation=20, ha="right", fontsize=8)
            ax.set_yscale("log")
            ax.set_ylabel("Burgers ν relative error")
            ax.set_title("(f) Burgers ν relative error")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5,
                    "No method estimated ν in this experiment.\n"
                    "(Direct MC-Dropout and Deep Ensemble PINN\n"
                    "use the analytic ν during PDE residual.)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9)
            ax.set_title("(f) Burgers ν relative error")
            ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Burgers case not in dataset.",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(4, len(labels)), fontsize=9,
                   bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("UQ baseline comparison on the blind-test subset",
                 fontsize=13, y=1.06)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        out = FIGURE_DIR / f"{FIG_NAME}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-gbsd", action="store_true",
                        help="Do not include GBSD-derived rows in the table/figure.")
    parser.add_argument("--preset", default=None,
                        choices=["quick_check", "preview", "medium", "full"],
                        help="Aggregate only this baseline preset. Default: full if present, otherwise quick_check.")
    args = parser.parse_args()
    ensure_output_dirs()
    df_raw, df_mean = aggregate(include_gbsd=not args.no_gbsd,
                                preset=args.preset)
    if not args.no_figure:
        plot(df_raw, df_mean)


if __name__ == "__main__":
    main()
