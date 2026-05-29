"""Paper-ready structure-discovery baseline figure.

This figure replaces the earlier empty Burgers-nu panel with a physics
consistency panel (blind-subset PDE residual). It uses existing v3.38
structure-discovery baseline CSV files only; no experiments are rerun.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "Results" / "supplementary" / "tables" / "structure_discovery_baselines_raw.csv"
DEFAULT_OUT = PROJECT_ROOT / "Results" / "supplementary" / "figures"

CASE_ORDER = ["Laplace", "Poisson", "Burgers_inv"]
CASE_LABELS = {
    "Laplace": "Laplace",
    "Poisson": "Poisson",
    "Burgers_inv": "Burgers inverse",
}
METHOD_ORDER = ["hac", "magnitude", "low_rank", "random"]
METHOD_LABELS = {
    "hac": "HAC",
    "magnitude": "Magnitude",
    "low_rank": "Low-rank",
    "random": "Random",
}
COLORS = {
    "hac": "#0072B2",
    "magnitude": "#E69F00",
    "low_rank": "#009E73",
    "random": "#D55E00",
}


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for col in [
        "structured_rel_l2_blind",
        "structured_to_dense_ratio_blind",
        "accepted_by_guard",
        "pde_residual_blind",
        "compression_ratio",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["accepted_by_guard"].dtype == object:
        df["accepted_by_guard"] = df["accepted_by_guard"].astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    group = df.groupby(["case", "structure_method"], sort=False)
    rows = []
    for (case, method), g in group:
        row = {"case": case, "structure_method": method, "n": len(g)}
        for col in [
            "structured_rel_l2_blind",
            "structured_to_dense_ratio_blind",
            "accepted_by_guard",
            "pde_residual_blind",
            "compression_ratio",
        ]:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["case"] = pd.Categorical(summary["case"], CASE_ORDER, ordered=True)
    summary["structure_method"] = pd.Categorical(summary["structure_method"], METHOD_ORDER, ordered=True)
    return summary.sort_values(["case", "structure_method"]).reset_index(drop=True)


def asymmetric_yerr(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    std = np.nan_to_num(np.asarray(std, dtype=float), nan=0.0)
    lower = np.minimum(std, np.maximum(mean * 0.8, 1e-16))
    upper = std
    return np.vstack([lower, upper])


def grouped_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    mean_col: str,
    std_col: str,
    ylabel: str,
    *,
    log: bool = False,
    ylim: tuple[float, float] | None = None,
    show_error: bool = True,
) -> None:
    x = np.arange(len(CASE_ORDER))
    width = 0.18
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2.0) * width
    for idx, method in enumerate(METHOD_ORDER):
        vals = []
        errs = []
        for case in CASE_ORDER:
            row = summary[(summary["case"] == case) & (summary["structure_method"] == method)]
            vals.append(float(row[mean_col].iloc[0]) if len(row) else np.nan)
            errs.append(float(row[std_col].iloc[0]) if len(row) else 0.0)
        vals_arr = np.asarray(vals, dtype=float)
        errs_arr = np.asarray(errs, dtype=float)
        if log:
            yerr = asymmetric_yerr(vals_arr, errs_arr)
        else:
            yerr = errs_arr
        ax.bar(
            x + offsets[idx],
            vals_arr,
            width=width * 0.92,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.6,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        if show_error:
            ax.errorbar(
                x + offsets[idx],
                vals_arr,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=2.0,
                capthick=0.8,
                zorder=4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[c] for c in CASE_ORDER])
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.75, zorder=0)
    ax.tick_params(axis="both", length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default="fig_structure_discovery_pde_baselines_paper")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    summary = summarize(raw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{args.stem}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.1), constrained_layout=False)
    ax = axes.ravel()

    grouped_bars(
        ax[0],
        summary,
        "structured_rel_l2_blind_mean",
        "structured_rel_l2_blind_std",
        r"Structured candidate $rL_2$",
        log=True,
    )
    ax[0].set_title("Blind-set structured accuracy")
    add_panel_label(ax[0], "a")

    grouped_bars(
        ax[1],
        summary,
        "structured_to_dense_ratio_blind_mean",
        "structured_to_dense_ratio_blind_std",
        r"Structured / dense $rL_2$",
        log=True,
    )
    ax[1].axhline(1.0, color="#4d4d4d", linestyle=":", linewidth=1.0, zorder=2)
    ax[1].set_title("Accuracy degradation ratio")
    add_panel_label(ax[1], "b")

    grouped_bars(
        ax[2],
        summary,
        "accepted_by_guard_mean",
        "accepted_by_guard_std",
        "Accepted fraction",
        log=False,
        ylim=(0.0, 1.08),
        show_error=False,
    )
    ax[2].set_yticks([0.0, 0.5, 1.0])
    ax[2].set_title("Guard decision stability")
    add_panel_label(ax[2], "c")

    grouped_bars(
        ax[3],
        summary,
        "pde_residual_blind_mean",
        "pde_residual_blind_std",
        "Blind-set PDE residual",
        log=True,
    )
    ax[3].set_title("Physics consistency")
    add_panel_label(ax[3], "d")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.52, 1.01),
        columnspacing=1.5,
        handlelength=1.5,
    )
    fig.subplots_adjust(top=0.88, left=0.085, right=0.985, bottom=0.10, wspace=0.28, hspace=0.42)

    for ext in ["png", "pdf", "svg"]:
        out = args.out_dir / f"{args.stem}.{ext}"
        fig.savefig(out, dpi=450 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {out}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
