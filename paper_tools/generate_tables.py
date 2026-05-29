"""Generate paper-ready table CSVs from official GBSD summary files.

The files in ``results/paper_tables/summary_*.csv`` are audit-oriented and
seed-level.  This script turns them into manuscript tables with mean/std
statistics and explicit GBSD rows where a comparison table needs them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PROBLEM_ORDER = ["laplace", "poisson", "burgers_inverse"]
PROBLEM_LABELS = {
    "laplace": "Laplace",
    "poisson": "Poisson",
    "burgers_inverse": "Burgers inverse",
}

TABLE_SOURCES = {
    "main": "summary_main_blind.csv",
    "runtime": "summary_runtime_params.csv",
    "guard": "summary_guard_ablation.csv",
    "uq": "summary_uq_ablation.csv",
    "baselines": "summary_baselines.csv",
    "reconstruction": "summary_reconstruction_ablation.csv",
    "threshold": "summary_threshold_sensitivity.csv",
    "structure": "summary_structure_stability.csv",
    "zero_distillation": "summary_zero_distillation_ablation.csv",
}


def _read(summary_dir: Path, filename: str, *, required: bool = True) -> pd.DataFrame:
    path = summary_dir / filename
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _mean(df: pd.DataFrame, column: str) -> float | None:
    values = _numeric(df, column).dropna()
    if values.empty:
        return None
    return float(values.mean())


def _std(df: pd.DataFrame, column: str) -> float | None:
    values = _numeric(df, column).dropna()
    if values.empty:
        return None
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _fmt(value: float | None, *, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return ""
    if percent:
        return f"{100.0 * value:.3f}%"
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value < 1e-2 or abs_value >= 1e3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _fmt_mean_std(mean: float | None, std: float | None, *, percent: bool = False) -> str:
    if mean is None or pd.isna(mean):
        return ""
    if std is None or pd.isna(std):
        return _fmt(mean, percent=percent)
    return f"{_fmt(mean, percent=percent)} +/- {_fmt(std, percent=percent)}"


def _source_counts(df: pd.DataFrame) -> str:
    if "final_source" not in df.columns:
        return ""
    counts = df["final_source"].dropna().astype(str).value_counts().sort_index()
    return "; ".join(f"{name}:{int(count)}" for name, count in counts.items())


def _accept_rate(df: pd.DataFrame) -> float | None:
    if "final_source" in df.columns:
        values = df["final_source"].dropna().astype(str)
        if not values.empty:
            return float((values == "structured").mean())
    for column in ["accepted_by_guard", "accepted_by_variant", "guard_accepted_by_variant"]:
        if column in df.columns:
            values = _numeric(df, column).dropna()
            if not values.empty:
                return float(values.mean())
    return None


def _grouped(
    df: pd.DataFrame,
    by: Iterable[str],
    metrics: Iterable[str],
    *,
    percent_metrics: set[str] | None = None,
) -> pd.DataFrame:
    percent_metrics = percent_metrics or set()
    rows: list[dict[str, object]] = []
    if df.empty:
        return pd.DataFrame()
    group_cols = list(by)
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(group_cols, keys))
        if "problem" in row:
            row["problem_label"] = PROBLEM_LABELS.get(str(row["problem"]), str(row["problem"]))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else int(len(group))
        if "protocol_id" in group.columns:
            protocol = group["protocol_id"].dropna().astype(str).unique()
            row["protocol_id"] = protocol[0] if len(protocol) == 1 else ";".join(sorted(protocol))
        if "final_source" in group.columns:
            row["final_source_counts"] = _source_counts(group)
            row["structured_accept_rate_mean"] = _accept_rate(group)
            row["structured_accept_rate"] = _fmt(_accept_rate(group))
        for metric in metrics:
            mean = _mean(group, metric)
            std = _std(group, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[metric] = _fmt_mean_std(mean, std, percent=metric in percent_metrics)
        rows.append(row)
    out = pd.DataFrame(rows)
    if "problem" in out.columns:
        out["_problem_order"] = out["problem"].map({p: i for i, p in enumerate(PROBLEM_ORDER)}).fillna(999)
        sort_cols = ["_problem_order"] + [c for c in group_cols if c != "problem"]
        out = out.sort_values(sort_cols).drop(columns=["_problem_order"]).reset_index(drop=True)
    return out


def _write(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / name, index=False)


def _table_main(main_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "dense_rL2",
        "structured_rL2",
        "final_rL2",
        "compression_ratio",
        "coverage95",
        "nll",
        "aiw",
        "error_std_corr",
        "nu_rel_error",
    ]
    return _grouped(main_df, ["problem"], metrics, percent_metrics={"nu_rel_error"})


def _table_runtime(runtime_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "teacher_params",
        "dense_params",
        "structured_trainable_params",
        "structured_effective_params",
        "compression_ratio",
        "teacher_training_time_s",
        "student_training_time_s",
        "structured_training_time_s",
        "inference_time_s",
        "total_wall_clock_time_s",
    ]
    out = _grouped(runtime_df, ["problem"], metrics)
    for label in ["device_name", "cuda_version", "pytorch_version", "ram_gb"]:
        if label in runtime_df.columns and "problem" in runtime_df.columns:
            values = (
                runtime_df.groupby("problem")[label]
                .apply(lambda s: ";".join(sorted(set(s.dropna().astype(str)))))
                .rename(label)
                .reset_index()
            )
            out = out.merge(values, on="problem", how="left")
    return out


def _table_guard(guard_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["final_rL2", "dense_rL2", "structured_rL2", "compression_ratio"]
    out = _grouped(guard_df, ["problem", "variant"], metrics)
    keep = [
        "problem",
        "problem_label",
        "variant",
        "n_seeds",
        "protocol_id",
        "final_source_counts",
        "structured_accept_rate_mean",
        "structured_accept_rate",
    ]
    metric_cols = [c for c in out.columns if c.endswith("_mean") or c.endswith("_std") or c in metrics]
    columns = list(dict.fromkeys(c for c in keep + metric_cols if c in out.columns))
    return out[columns]


def _table_uq(uq_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["coverage95", "nll", "aiw", "error_std_corr", "final_rL2"]
    return _grouped(uq_df, ["problem", "variant"], metrics)


def _with_gbsd_rows(main_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    gbsd = main_df.copy()
    gbsd["variant"] = "GBSD"
    gbsd["experiment"] = "strong_baselines"
    gbsd["baseline_source"] = "main_blind_full_gbsd"
    if baseline_df.empty:
        return gbsd
    baseline = baseline_df.copy()
    baseline["baseline_source"] = "summary_baselines"
    return pd.concat([gbsd, baseline], ignore_index=True, sort=False)


def _table_baselines(main_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    combined = _with_gbsd_rows(main_df, baseline_df)
    metrics = [
        "final_rL2",
        "coverage95",
        "nll",
        "aiw",
        "error_std_corr",
        "compression_ratio",
        "nu_rel_error",
    ]
    return _grouped(combined, ["problem", "variant"], metrics, percent_metrics={"nu_rel_error"})


def _table_reconstruction(recon_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "structured_rL2",
        "final_rL2",
        "compression_ratio",
        "coverage95",
        "error_std_corr",
        "nu_rel_error",
    ]
    return _grouped(recon_df, ["problem", "variant"], metrics, percent_metrics={"nu_rel_error"})


def _table_threshold(threshold_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_rL2",
        "structured_rL2",
        "compression_ratio",
        "coverage95",
        "error_std_corr",
    ]
    out = _grouped(threshold_df, ["problem", "variant"], metrics)
    for label in ["sensitivity_parameter", "sensitivity_value"]:
        if label in threshold_df.columns:
            values = (
                threshold_df.groupby(["problem", "variant"], dropna=False)[label]
                .apply(lambda s: ";".join(sorted(set(s.dropna().astype(str)))))
                .rename(label)
                .reset_index()
            )
            out = out.merge(values, on=["problem", "variant"], how="left")
    return out


def _table_structure(structure_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["final_rL2", "structured_rL2", "compression_ratio", "coverage95", "error_std_corr"]
    out = _grouped(structure_df, ["problem", "variant"], metrics)
    if "stability_scope" in structure_df.columns:
        values = (
            structure_df.groupby(["problem", "variant"], dropna=False)["stability_scope"]
            .apply(lambda s: ";".join(sorted(set(s.dropna().astype(str)))))
            .rename("stability_scope")
            .reset_index()
        )
        out = out.merge(values, on=["problem", "variant"], how="left")
    if "stability_scope" not in out.columns:
        out["stability_scope"] = "outcome_level"
    out["paper_scope_note"] = "Outcome-level stability; not selected-term frequency."
    return out


def _table_zero_distillation(zero_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["dense_rL2", "final_rL2", "coverage95", "nll", "aiw", "error_std_corr"]
    return _grouped(zero_df, ["problem", "variant"], metrics)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default="results/paper_tables")
    parser.add_argument("--output-dir", default="results/paper_tables/generated")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    summary_dir = ROOT / args.summary_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    data: dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_SOURCES.items():
        try:
            data[key] = _read(summary_dir, filename, required=not args.allow_missing)
        except FileNotFoundError:
            missing.append(filename)
            data[key] = pd.DataFrame()
    if missing and not args.allow_missing:
        print("Missing required summary sources:")
        for item in missing:
            print(f"  - {item}")
        return 2

    main_df = data["main"]
    tables = {
        "table_4_2_main_results.csv": _table_main(main_df),
        "table_4_4_runtime_params.csv": _table_runtime(data["runtime"]),
        "table_5_1_guard_ablation.csv": _table_guard(data["guard"]),
        "table_5_3_uq_ablation.csv": _table_uq(data["uq"]),
        "table_5_4_strong_baselines.csv": _table_baselines(main_df, data["baselines"]),
        "appendix_reconstruction_ablation.csv": _table_reconstruction(data["reconstruction"]),
        "appendix_threshold_sensitivity.csv": _table_threshold(data["threshold"]),
        "appendix_structure_stability.csv": _table_structure(data["structure"]),
        "appendix_zero_distillation_ablation.csv": _table_zero_distillation(data["zero_distillation"]),
    }
    for name, table in tables.items():
        _write(table, output_dir, name)

    print(f"Generated paper-ready table CSVs in {output_dir}")
    for name, table in tables.items():
        print(f"  {name}: {len(table)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
