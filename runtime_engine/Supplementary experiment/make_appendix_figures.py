"""Generate appendix figures from supplementary CSV tables."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGURE_DIR, TABLE_DIR, ensure_output_dirs


def save(fig, name: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURE_DIR / (name + '.png')}")


def fig_multiseed():
    path = TABLE_DIR / "multiseed_guarded_mean_std.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    ax.bar(x, df["final_rel_l2_mean"], yerr=df["final_rel_l2_std"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["case"] + "\n" + df["final_source"], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Final relative L2")
    ax.set_title("Appendix D.1: Multi-seed guarded stability")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig_D1_multiseed_guarded_stability")


def fig_guard_ablation():
    path = TABLE_DIR / "guard_ablation_mean_std.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    keep = df[df["variant"].isin(["always_structured", "full_guard"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    cases = list(dict.fromkeys(keep["case"]))
    variants = ["always_structured", "full_guard"]
    width = 0.35
    x = np.arange(len(cases))
    for j, variant in enumerate(variants):
        vals = []
        errs = []
        for case in cases:
            row = keep[(keep["case"] == case) & (keep["variant"] == variant)]
            vals.append(float(row["selected_rL2_mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["selected_rL2_std"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + (j - 0.5) * width, vals, width, yerr=errs, capsize=3, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_yscale("log")
    ax.set_ylabel("Selected relative L2")
    ax.set_title("Appendix D.2: Final-source guard ablation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig_D2_final_source_guard_ablation")


def fig_uq_ablation():
    path = TABLE_DIR / "uq_ablation_Poisson_mean_std.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    labels = df["variant"].str.replace("_", "\n")
    for ax, col, title in [
        (axes[0], "coverage95_mean", "Coverage95"),
        (axes[1], "corr_mean", "Error-std corr"),
        (axes[2], "avg_interval_width_mean", "Average interval width"),
    ]:
        ax.bar(np.arange(len(df)), df[col])
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Appendix D.3: Poisson UQ calibration ablation")
    fig.tight_layout()
    save(fig, "fig_D3_poisson_uq_calibration_ablation")


def fig_reconstruction():
    path = TABLE_DIR / "reconstruction_ablation_summary.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = df["case"] + "\n" + df["variant"]
    ax.bar(np.arange(len(df)), df["structured_rel_l2"])
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Structured relative L2")
    ax.set_title("Appendix D.4: Structured reconstruction ablation")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_D4_structured_reconstruction_ablation")


def fig_clustering():
    path = TABLE_DIR / "clustering_sweep_summary.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    df = df[df["kind"] == "cluster_sweep"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for case, sub in df.groupby("case"):
        ax.plot(sub["compression"], sub["structured_to_dense_ratio"],
                marker="o", label=case)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Structured / dense relative L2")
    ax.set_title("Appendix D.5: Compression-accuracy sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "fig_D5_clustering_threshold_sweep")


def fig_runtime():
    path = TABLE_DIR / "runtime_and_params_mean_std.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    x = np.arange(len(df))
    axes[0].bar(x, df["compression_mean"], yerr=df["compression_std"], capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(df["case"], rotation=20)
    axes[0].set_ylabel("Compression ratio")
    axes[0].set_title("Compression")
    axes[1].bar(x - 0.18, df["teacher_time_s_mean"], width=0.36, label="Teacher")
    axes[1].bar(x + 0.18, df["student_time_s_mean"], width=0.36, label="Student")
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["case"], rotation=20)
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Training time")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Appendix D.6: Cost and compression summary")
    fig.tight_layout()
    save(fig, "fig_D6_runtime_parameter_compression")


def fig_nll():
    path = TABLE_DIR / "nll_sanity_check_mean_std.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = df["case"] + "\n" + df["variant"].str.replace("_", "\n")
    ax.bar(np.arange(len(df)), df["nll_mean"])
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Gaussian NLL")
    ax.set_title("Appendix D.7: NLL and uncertainty-scale sanity check")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_D7_nll_sanity_check")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_output_dirs()
    fig_multiseed()
    fig_guard_ablation()
    fig_uq_ablation()
    fig_reconstruction()
    fig_clustering()
    fig_runtime()
    fig_nll()


if __name__ == "__main__":
    main()

