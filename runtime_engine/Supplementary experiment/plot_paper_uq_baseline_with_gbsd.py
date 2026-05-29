"""Create paper-ready UQ baseline figures with the GBSD final source included.

This script reads integrated runtime artifacts:
  * full UQ baselines:
      Results/supplementary/tables/uq_baselines_raw.csv
  * GBSD multi-seed guarded runs:
      Results/supplementary/tables/multiseed_guarded_raw.csv

It recomputes Gaussian NLL from prediction files using a small standard
deviation floor. The floor is used ONLY for NLL stability; coverage, interval
width, rL2, and correlation are computed from the original predictions.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UQ_RAW = PROJECT_ROOT / "Results" / "supplementary" / "tables" / "uq_baselines_raw.csv"
GBSD_RAW = PROJECT_ROOT / "Results" / "supplementary" / "tables" / "multiseed_guarded_raw.csv"
PAPER_FIG_DIR = PROJECT_ROOT / "Results" / "supplementary" / "figures"

DEFAULT_FIG_STEM = "fig_uq_baseline_comparison_with_gbsd"

CASES = ["Laplace", "Poisson", "Burgers_inv"]
CASE_LABELS = {
    "Laplace": "Laplace",
    "Poisson": "Poisson",
    "Burgers_inv": "Burgers inverse",
}
METHODS = [
    "direct_mc_dropout_pinn",
    "deep_ensemble_pinn",
    "gbsd_final_source",
]
METHOD_LABELS = {
    "direct_mc_dropout_pinn": "Direct MC-Dropout PINN",
    "deep_ensemble_pinn": "Deep Ensemble PINN",
    "gbsd_final_source": "GBSD final source",
}
METHOD_COLORS = {
    "direct_mc_dropout_pinn": "#CC79A7",
    "deep_ensemble_pinn": "#0072B2",
    "gbsd_final_source": "#009E73",
}
METHOD_HATCHES = {
    "direct_mc_dropout_pinn": "",
    "deep_ensemble_pinn": "//",
    "gbsd_final_source": "xx",
}

NLL_STD_FLOOR = 1e-6


def rel_l2(mean: np.ndarray, exact: np.ndarray) -> float:
    denom = float(np.linalg.norm(exact))
    if denom <= 0:
        return math.nan
    return float(np.linalg.norm(mean - exact) / denom)


def coverage95(mean: np.ndarray, std: np.ndarray, exact: np.ndarray) -> float:
    std = np.maximum(std, 1e-12)
    return float(np.mean((exact >= mean - 1.96 * std)
                         & (exact <= mean + 1.96 * std)))


def corr_abs_error_std(mean: np.ndarray, std: np.ndarray, exact: np.ndarray) -> float:
    err = np.abs(mean - exact)
    if float(np.std(err)) <= 1e-12 or float(np.std(std)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(err, std)[0, 1])


def gaussian_nll(mean: np.ndarray, std: np.ndarray, exact: np.ndarray,
                 std_floor: float = NLL_STD_FLOOR) -> float:
    std = np.maximum(std, std_floor)
    var = std ** 2
    return float(np.mean(0.5 * np.log(2.0 * np.pi * var)
                         + (exact - mean) ** 2 / (2.0 * var)))


def mean_interval_width(std: np.ndarray) -> float:
    return float(np.mean(2.0 * 1.96 * np.maximum(std, 1e-12)))


def load_prediction_row(row: pd.Series) -> Dict:
    out_dir = Path(str(row["output_dir"]))
    if row["method"] == "direct_mc_dropout_pinn":
        data = np.load(out_dir / "predictions.npz")
        mean = np.asarray(data["mean"]).reshape(-1)
        std = np.asarray(data["std"]).reshape(-1)
    elif row["method"] == "deep_ensemble_pinn":
        data = np.load(out_dir / "ensemble_predictions.npz")
        mean = np.asarray(data["ensemble_mean"]).reshape(-1)
        std = np.asarray(data["ensemble_std"]).reshape(-1)
    else:
        raise ValueError(f"Unknown baseline method: {row['method']}")
    exact = np.asarray(data["exact"]).reshape(-1)
    mask = np.asarray(data["blind_test_mask"]).astype(bool).reshape(-1) & ~np.isnan(exact)
    return metric_row(
        case=str(row["case"]),
        method=str(row["method"]),
        replicate=str(row["seed_or_member"]),
        mean=mean[mask],
        std=std[mask],
        exact=exact[mask],
        runtime_train_s=float(row.get("runtime_train_s", math.nan)),
        n_blind=int(mask.sum()),
        extra={"source_path": str(out_dir)},
    )


def load_gbsd_row(row: pd.Series) -> Dict:
    data = np.load(str(row["source_path"]))
    mean = np.asarray(data["bayesian_mean"]).reshape(-1)
    std = np.asarray(data["bayesian_std"]).reshape(-1)
    exact = np.asarray(data["exact"]).reshape(-1)
    mask = np.asarray(data["blind_test_mask"]).astype(bool).reshape(-1) & ~np.isnan(exact)
    extra = {
        "source_path": str(row["source_path"]),
        "final_source": str(row.get("final_source", "")),
        "compression": to_float(row.get("compression", math.nan)),
        "nu_relative_error": to_float(row.get("nu_relative_error", math.nan)),
    }
    return metric_row(
        case=str(row["case"]),
        method="gbsd_final_source",
        replicate=str(row.get("seed", "")),
        mean=mean[mask],
        std=std[mask],
        exact=exact[mask],
        runtime_train_s=math.nan,
        n_blind=int(mask.sum()),
        extra=extra,
    )


def metric_row(case: str, method: str, replicate: str,
               mean: np.ndarray, std: np.ndarray, exact: np.ndarray,
               runtime_train_s: float, n_blind: int,
               extra: Dict | None = None) -> Dict:
    row = {
        "case": case,
        "method": method,
        "replicate": replicate,
        "rel_l2": rel_l2(mean, exact),
        "coverage95": coverage95(mean, std, exact),
        "corr_abs_error_std": corr_abs_error_std(mean, std, exact),
        "nll_std_floor_1e-6": gaussian_nll(mean, std, exact),
        "mean_interval_width": mean_interval_width(std),
        "avg_std": float(np.mean(std)),
        "runtime_train_s": runtime_train_s,
        "n_blind": n_blind,
    }
    if extra:
        row.update(extra)
    return row


def to_float(value) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def summarize(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric = [
        "rel_l2", "coverage95", "corr_abs_error_std",
        "nll_std_floor_1e-6", "mean_interval_width", "avg_std",
        "runtime_train_s", "nu_relative_error",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summary_rows = []
    for (case, method), sub in df.groupby(["case", "method"], sort=False):
        out = {
            "case": case,
            "method": method,
            "n": int(len(sub)),
        }
        for col in numeric:
            if col in sub:
                values = sub[col].dropna()
                out[f"{col}_mean"] = float(values.mean()) if len(values) else math.nan
                out[f"{col}_std"] = float(values.std(ddof=1)) if len(values) >= 2 else math.nan
        summary_rows.append(out)
    summary = pd.DataFrame(summary_rows)
    summary["case"] = pd.Categorical(summary["case"], CATEGORIES_ORDER, ordered=True)
    summary["method"] = pd.Categorical(summary["method"], METHODS, ordered=True)
    return summary.sort_values(["case", "method"]).reset_index(drop=True)


CATEGORIES_ORDER = CASES


def _bar_panel(ax, summary: pd.DataFrame, metric: str, ylabel: str,
               letter: str, log: bool = False, target: float | None = None,
               lower_better: bool = False):
    x = np.arange(len(CASES))
    width = 0.23
    offsets = np.linspace(-width, width, len(METHODS))
    for offset, method in zip(offsets, METHODS):
        means = []
        errs = []
        for case in CASES:
            sub = summary[(summary["case"] == case) & (summary["method"] == method)]
            if sub.empty:
                means.append(np.nan)
                errs.append(0.0)
            else:
                means.append(float(sub[f"{metric}_mean"].iloc[0]))
                err = sub[f"{metric}_std"].iloc[0]
                errs.append(float(err) if pd.notna(err) else 0.0)
        ax.bar(x + offset, means, width=width,
               yerr=errs, capsize=2.5,
               color=METHOD_COLORS[method],
               edgecolor="black", linewidth=0.6,
               hatch=METHOD_HATCHES[method],
               alpha=0.92)
    if log:
        ax.set_yscale("log")
    if target is not None:
        ax.axhline(target, color="0.25", linestyle=(0, (4, 2)),
                   linewidth=1.2, zorder=0)
        ax.text(0.98, target + 0.006 if target > 0 else target,
                f"target {target:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="0.2")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[c] for c in CASES], rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    if lower_better:
        ax.text(0.98, 0.05, "lower is better", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="0.25")
    ax.grid(axis="y", color="0.86", linewidth=0.7)
    ax.set_axisbelow(True)


def plot(summary: pd.DataFrame, fig_stem: str):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.7), constrained_layout=True)
    _bar_panel(axes[0, 0], summary, "rel_l2",
               r"Relative $L_2$ error", "a", log=True)
    _bar_panel(axes[0, 1], summary, "coverage95",
               "Coverage at 95% interval", "b", target=0.95)
    axes[0, 1].set_ylim(0.82, 1.03)
    _bar_panel(axes[0, 2], summary, "corr_abs_error_std",
               r"Corr$(|e|,\sigma)$", "c")
    axes[0, 2].set_ylim(-0.3, 0.9)
    _bar_panel(axes[1, 0], summary, "nll_std_floor_1e-6",
               r"Gaussian NLL, $\sigma_{\min}=10^{-6}$", "d",
               lower_better=True)
    _bar_panel(axes[1, 1], summary, "mean_interval_width",
               "Mean interval width", "e", log=True)

    ax = axes[1, 2]
    burgers = summary[(summary["case"] == "Burgers_inv")
                      & (summary["method"] == "gbsd_final_source")]
    if not burgers.empty and pd.notna(burgers["nu_relative_error_mean"].iloc[0]):
        mean = float(burgers["nu_relative_error_mean"].iloc[0])
        err = burgers["nu_relative_error_std"].iloc[0]
        err = float(err) if pd.notna(err) else 0.0
        ax.bar([0], [mean], yerr=[err], capsize=3,
               color=METHOD_COLORS["gbsd_final_source"],
               edgecolor="black", linewidth=0.6,
               hatch=METHOD_HATCHES["gbsd_final_source"],
               width=0.42)
        ax.set_yscale("log")
        ax.set_xlim(-0.7, 0.7)
        ax.set_xticks([0])
        ax.set_xticklabels(["GBSD final source"])
        ax.set_ylabel(r"Burgers $\nu$ relative error")
        ax.grid(axis="y", color="0.86", linewidth=0.7)
        ax.text(0.50, 0.95,
                "UQ baselines use fixed analytic $\\nu$",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="0.25")
    else:
        ax.text(0.5, 0.5,
                "Burgers parameter inversion is\nreported only for GBSD.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9)
        ax.axis("off")
    ax.text(-0.13, 1.04, "f", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")

    handles = [
        Patch(facecolor=METHOD_COLORS[m], edgecolor="black",
              hatch=METHOD_HATCHES[m], label=METHOD_LABELS[m])
        for m in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.035))

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out = PAPER_FIG_DIR / f"{fig_stem}.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="",
                        help="Optional suffix appended to the output stem, e.g. _v2.")
    args = parser.parse_args()

    if not UQ_RAW.is_file():
        raise FileNotFoundError(UQ_RAW)
    if not GBSD_RAW.is_file():
        raise FileNotFoundError(GBSD_RAW)
    fig_stem = f"{DEFAULT_FIG_STEM}{args.suffix}"
    out_raw = PAPER_FIG_DIR / f"{fig_stem}_raw.csv"
    out_summary = PAPER_FIG_DIR / f"{fig_stem}_summary.csv"

    rows: List[Dict] = []
    uq = pd.read_csv(UQ_RAW)
    uq = uq[uq["preset"].astype(str) == "full"].copy()
    for _, row in uq.iterrows():
        if row["method"] in ("direct_mc_dropout_pinn", "deep_ensemble_pinn"):
            rows.append(load_prediction_row(row))

    gbsd = pd.read_csv(GBSD_RAW)
    for _, row in gbsd.iterrows():
        rows.append(load_gbsd_row(row))

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out_raw, index=False)
    summary = summarize(rows)
    summary.to_csv(out_summary, index=False)
    print(f"Saved: {out_raw}")
    print(f"Saved: {out_summary}")
    plot(summary, fig_stem)


if __name__ == "__main__":
    main()
