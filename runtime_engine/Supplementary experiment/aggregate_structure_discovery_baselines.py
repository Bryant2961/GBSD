"""Aggregate structure-discovery baseline runs into mean/std tables and figure.

Reads from:
    Results/supplementary/tables/structure_discovery_baselines_raw.csv

Writes:
    Results/supplementary/tables/structure_discovery_baselines_mean_std.csv
    Results/supplementary/figures/fig_structure_discovery_pde_baselines.{png,pdf,svg}

The figure has up to four panels:
    (a) Blind-test structured rL2 per method (log y).
    (b) Compression ratio per method.
    (c) Guard-accepted fraction across seeds per method.
    (d) Burgers nu relative error per method (Burgers only).

Usage:
    python "Supplementary experiment/aggregate_structure_discovery_baselines.py"
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import TABLE_DIR, FIGURE_DIR, ensure_output_dirs

RAW_PATH = TABLE_DIR / "structure_discovery_baselines_raw.csv"
MEAN_PATH = TABLE_DIR / "structure_discovery_baselines_mean_std.csv"
FIG_NAME = "fig_structure_discovery_pde_baselines"

METHOD_LABELS = {
    "hac": "Proposed HAC",
    "random": "Random clusters",
    "magnitude": "Magnitude bins",
    "kmeans": "K-means (|w|)",
    "low_rank": "Low-rank SVD",
}
METHOD_COLORS = {
    "hac": "#2ca02c",
    "random": "#7f7f7f",
    "magnitude": "#ff7f0e",
    "kmeans": "#9467bd",
    "low_rank": "#1f77b4",
}
METHOD_ORDER = ["hac", "random", "magnitude", "kmeans", "low_rank"]


def aggregate():
    if not RAW_PATH.is_file():
        raise SystemExit(f"Raw table missing: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    if df.empty:
        raise SystemExit("Raw table is empty.")

    metrics_for_stats = [
        "dense_rel_l2_guard", "structured_rel_l2_guard",
        "dense_rel_l2_blind", "structured_rel_l2_blind",
        "structured_to_dense_ratio_guard", "structured_to_dense_ratio_blind",
        "compression_ratio", "dense_param_count", "structured_param_count",
        "final_rel_l2_blind", "coverage95_blind", "corr_abs_error_std_blind",
        "nll_blind", "mean_interval_width_blind", "avg_std_blind",
        "pde_residual_blind",
        "nu_pred", "nu_relative_error",
        "runtime_discovery_s", "runtime_reconstruction_s", "runtime_inference_s",
    ]
    metrics_for_stats = [m for m in metrics_for_stats if m in df.columns]

    grouped = df.groupby(["case", "structure_method"], dropna=False)
    rows = []
    for (case, method), sub in grouped:
        row = {"case": case, "structure_method": method, "n_seeds": int(len(sub))}
        for m in metrics_for_stats:
            v = pd.to_numeric(sub[m], errors="coerce")
            row[f"{m}_mean"] = float(v.mean()) if v.notna().any() else math.nan
            row[f"{m}_std"] = float(v.std(ddof=1)) if v.notna().sum() >= 2 else math.nan
        if "accepted_by_guard" in sub.columns:
            row["accepted_fraction"] = float(
                pd.to_numeric(sub["accepted_by_guard"].astype(bool),
                              errors="coerce").mean())
        rows.append(row)
    df_out = pd.DataFrame(rows).sort_values(["case", "structure_method"])
    df_out.to_csv(MEAN_PATH, index=False)
    print(f"Saved: {MEAN_PATH} ({len(df_out)} rows)")
    return df, df_out


def _present_methods(df_mean: pd.DataFrame):
    found = list(df_mean["structure_method"].unique())
    return [m for m in METHOD_ORDER if m in found] + [m for m in found
                                                       if m not in METHOD_ORDER]


def plot(df_raw: pd.DataFrame, df_mean: pd.DataFrame):
    ensure_output_dirs()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    methods = _present_methods(df_mean)
    cases = list(dict.fromkeys(df_mean["case"]))
    n_methods = len(methods)
    has_burgers = "Burgers_inv" in cases

    fig, axes = plt.subplots(
        1, 4 if has_burgers else 3,
        figsize=(15 if has_burgers else 12, 4))

    width = 0.8 / max(1, n_methods)
    x_cases = np.arange(len(cases))

    # Panel (a): blind structured rL2 per method, grouped by case
    ax = axes[0]
    for j, method in enumerate(methods):
        means = []
        stds = []
        for case in cases:
            sub = df_mean[(df_mean["case"] == case)
                          & (df_mean["structure_method"] == method)]
            if sub.empty:
                means.append(np.nan); stds.append(0.0)
            else:
                means.append(float(sub["structured_rel_l2_blind_mean"].iloc[0]))
                stds.append(float(sub["structured_rel_l2_blind_std"].iloc[0])
                            if pd.notna(sub["structured_rel_l2_blind_std"].iloc[0])
                            else 0.0)
        ax.bar(x_cases + (j - (n_methods - 1) / 2) * width, means, width,
               yerr=stds, capsize=3,
               color=METHOD_COLORS.get(method, "#444"),
               label=METHOD_LABELS.get(method, method))
    ax.set_xticks(x_cases); ax.set_xticklabels(cases, rotation=10)
    ax.set_yscale("log")
    ax.set_ylabel("Blind-test rL2 (structured)")
    ax.set_title("(a) Structured rL2 on blind subset")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # Panel (b): compression ratio per method, grouped by case
    ax = axes[1]
    for j, method in enumerate(methods):
        means = []; stds = []
        for case in cases:
            sub = df_mean[(df_mean["case"] == case)
                          & (df_mean["structure_method"] == method)]
            if sub.empty:
                means.append(np.nan); stds.append(0.0)
            else:
                means.append(float(sub["compression_ratio_mean"].iloc[0]))
                stds.append(float(sub["compression_ratio_std"].iloc[0])
                            if pd.notna(sub["compression_ratio_std"].iloc[0])
                            else 0.0)
        ax.bar(x_cases + (j - (n_methods - 1) / 2) * width, means, width,
               yerr=stds, capsize=3,
               color=METHOD_COLORS.get(method, "#444"))
    ax.set_xticks(x_cases); ax.set_xticklabels(cases, rotation=10)
    ax.set_ylabel("Compression ratio (dense / structured)")
    ax.set_title("(b) Compression ratio")
    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel (c): guard-accepted fraction per method, grouped by case
    ax = axes[2]
    for j, method in enumerate(methods):
        vals = []
        for case in cases:
            sub = df_mean[(df_mean["case"] == case)
                          & (df_mean["structure_method"] == method)]
            if sub.empty:
                vals.append(np.nan)
            else:
                vals.append(float(sub["accepted_fraction"].iloc[0])
                            if "accepted_fraction" in sub else np.nan)
        ax.bar(x_cases + (j - (n_methods - 1) / 2) * width, vals, width,
               color=METHOD_COLORS.get(method, "#444"))
    ax.set_xticks(x_cases); ax.set_xticklabels(cases, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Guard-accepted fraction")
    ax.set_title("(c) Guard accepted / rejected")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel (d): Burgers nu relative error per method
    if has_burgers:
        ax = axes[3]
        burgers = df_raw[df_raw["case"] == "Burgers_inv"]
        for j, method in enumerate(methods):
            sub = burgers[burgers["structure_method"] == method]
            vals = pd.to_numeric(sub["nu_relative_error"], errors="coerce")
            vals = vals.dropna().values
            if vals.size > 0:
                ax.bar(j, float(vals.mean()), 0.6, yerr=float(vals.std())
                       if vals.size >= 2 else 0,
                       capsize=3,
                       color=METHOD_COLORS.get(method, "#444"),
                       label=METHOD_LABELS.get(method, method))
            else:
                ax.bar(j, 0, 0.6,
                       color=METHOD_COLORS.get(method, "#444"))
        ax.set_xticks(np.arange(n_methods))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=20, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("Burgers ν relative error")
        ax.set_title("(d) Burgers ν inversion")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Structure-discovery baselines: full PDE reconstruction "
                 "under matched compression budget", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        out = FIGURE_DIR / f"{FIG_NAME}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    df_raw, df_mean = aggregate()
    if not args.no_figure:
        plot(df_raw, df_mean)


if __name__ == "__main__":
    main()
