"""Verify that paper assets use official GBSD summary files consistently."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbsd.reporting.schema import (  # noqa: E402
    FORBIDDEN_OFFICIAL_LABEL_FRAGMENTS,
    REQUIRED_METRIC_FIELDS,
    SUMMARY_FILES,
)

OFFICIAL_PROBLEMS = {"laplace", "poisson", "burgers_inverse"}
MAIN_CONSISTENCY_METRICS = [
    "final_rL2",
    "coverage95",
    "nll",
    "aiw",
    "error_std_corr",
    "compression_ratio",
    "nu_rel_error",
]
UQ_CONSISTENCY_METRICS = ["coverage95", "nll", "aiw", "error_std_corr", "final_rL2"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _contains_forbidden_text(df: pd.DataFrame) -> list[str]:
    offenders: list[str] = []
    text_cols = [c for c in df.columns if df[c].dtype == object]
    for col in text_cols:
        values = df[col].dropna().astype(str)
        for fragment in FORBIDDEN_OFFICIAL_LABEL_FRAGMENTS:
            if values.str.contains(fragment, regex=False).any():
                offenders.append(f"{col} contains {fragment}")
    return offenders


def _as_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _mean_by_problem(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame(columns=["problem", metric])
    out = (
        df.assign(**{metric: _as_numeric(df, metric)})
        .groupby("problem", as_index=False)[metric]
        .mean()
    )
    return out


def _check_mean_table_matches(
    errors: list[str],
    *,
    source: pd.DataFrame,
    target: pd.DataFrame,
    metrics: list[str],
    target_name: str,
) -> None:
    if source.empty or target.empty or "problem" not in source.columns or "problem" not in target.columns:
        return
    for metric in metrics:
        target_col = f"{metric}_mean"
        if metric not in source.columns or target_col not in target.columns:
            continue
        source_means = _mean_by_problem(source, metric).rename(columns={metric: "source_mean"})
        generated = target[["problem", target_col]].rename(columns={target_col: "target_mean"}).copy()
        merged = source_means.merge(generated, on="problem", how="inner")
        for _, row in merged.iterrows():
            if not _close(row["source_mean"], row["target_mean"]):
                errors.append(f"{target_name} {metric} mismatch for {row['problem']}")


def _close(a: float, b: float) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def _check_generated_tables(summary_dir: Path, loaded: dict[str, pd.DataFrame]) -> list[str]:
    errors: list[str] = []
    generated = summary_dir / "generated"
    if not generated.is_dir():
        errors.append("Missing generated paper table directory; run paper_tools/generate_tables.py")
        return errors

    main_table_path = generated / "table_4_2_main_results.csv"
    baseline_table_path = generated / "table_5_4_strong_baselines.csv"
    guard_table_path = generated / "table_5_1_guard_ablation.csv"
    uq_table_path = generated / "table_5_3_uq_ablation.csv"

    try:
        table_42 = _read_csv(main_table_path)
    except FileNotFoundError:
        errors.append("Missing generated table_4_2_main_results.csv")
        table_42 = pd.DataFrame()
    try:
        table_54 = _read_csv(baseline_table_path)
    except FileNotFoundError:
        errors.append("Missing generated table_5_4_strong_baselines.csv")
        table_54 = pd.DataFrame()
    try:
        table_51 = _read_csv(guard_table_path)
    except FileNotFoundError:
        errors.append("Missing generated table_5_1_guard_ablation.csv")
        table_51 = pd.DataFrame()
    try:
        table_53 = _read_csv(uq_table_path)
    except FileNotFoundError:
        errors.append("Missing generated table_5_3_uq_ablation.csv")
        table_53 = pd.DataFrame()

    if not table_42.empty:
        problems = set(table_42["problem"].astype(str)) if "problem" in table_42.columns else set()
        if problems != OFFICIAL_PROBLEMS:
            errors.append(f"table_4_2_main_results.csv must contain exactly {sorted(OFFICIAL_PROBLEMS)}, got {sorted(problems)}")
        if len(table_42) != 3:
            errors.append("table_4_2_main_results.csv must be problem-level mean/std, not seed-level")
        for metric in ["final_rL2_mean", "coverage95_mean", "compression_ratio_mean"]:
            if metric not in table_42.columns:
                errors.append(f"table_4_2_main_results.csv missing {metric}")

    if not table_54.empty:
        if "variant" not in table_54.columns:
            errors.append("table_5_4_strong_baselines.csv missing variant column")
        else:
            gbsd = table_54[table_54["variant"].astype(str) == "GBSD"]
            if len(gbsd) != 3:
                errors.append("table_5_4_strong_baselines.csv must include one GBSD row per problem")
            elif set(gbsd["problem"].astype(str)) != OFFICIAL_PROBLEMS:
                errors.append("table_5_4_strong_baselines.csv GBSD rows do not cover all official problems")

    main_df = loaded.get("main_blind")
    if main_df is not None and not main_df.empty and not table_42.empty and not table_54.empty:
        _check_mean_table_matches(
            errors,
            source=main_df,
            target=table_42,
            metrics=MAIN_CONSISTENCY_METRICS,
            target_name="table_4_2",
        )
        gbsd = table_54[table_54["variant"].astype(str) == "GBSD"]
        _check_mean_table_matches(
            errors,
            source=main_df,
            target=gbsd,
            metrics=MAIN_CONSISTENCY_METRICS,
            target_name="table_5_4 GBSD",
        )
        if not table_53.empty and "variant" in table_53.columns:
            full_calibration = table_53[table_53["variant"].astype(str) == "full_calibration"]
            _check_mean_table_matches(
                errors,
                source=main_df,
                target=full_calibration,
                metrics=UQ_CONSISTENCY_METRICS,
                target_name="table_5_3 full_calibration",
            )

    if not table_51.empty:
        forbidden_uq_cols = {"coverage95", "coverage95_mean", "nll", "nll_mean", "aiw", "aiw_mean", "error_std_corr", "error_std_corr_mean"}
        present = sorted(forbidden_uq_cols.intersection(table_51.columns))
        if present:
            errors.append(f"table_5_1_guard_ablation.csv should not report UQ columns: {present}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default="results/paper_tables")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    summary_dir = ROOT / args.summary_dir
    errors: list[str] = []
    loaded: dict[str, pd.DataFrame] = {}

    for key, filename in SUMMARY_FILES.items():
        path = summary_dir / filename
        if not path.is_file():
            errors.append(f"Missing summary file: {filename}")
            continue
        try:
            df = _read_csv(path)
            loaded[key] = df
            for offender in _contains_forbidden_text(df):
                errors.append(f"{filename}: {offender}")
        except Exception as exc:
            errors.append(f"Cannot read {filename}: {exc}")

    raw = loaded.get("all_raw")
    if raw is not None and not raw.empty:
        for field in REQUIRED_METRIC_FIELDS:
            if field not in raw.columns:
                errors.append(f"summary_all_raw.csv missing field: {field}")
        if "protocol_id" in raw.columns and raw["protocol_id"].isna().any():
            errors.append("summary_all_raw.csv has rows without protocol_id")
        if "source_run_dir" in raw.columns:
            if raw["source_run_dir"].astype(str).str.contains(r"^[A-Za-z]:", regex=True).any():
                errors.append("summary_all_raw.csv contains absolute Windows source paths")
    elif raw is not None:
        errors.append("summary_all_raw.csv has no official result rows")

    main_df = loaded.get("main_blind")
    baseline_df = loaded.get("baselines")
    if main_df is not None and baseline_df is not None:
        if "variant" in baseline_df.columns and "GBSD" in set(baseline_df["variant"].astype(str)):
            errors.append("summary_baselines.csv should not carry copied GBSD rows; generated table 5.4 adds GBSD from main_blind")

    errors.extend(_check_generated_tables(summary_dir, loaded))

    if errors and not args.allow_incomplete:
        print("Consistency check failed:")
        for error in errors:
            print(f"  - {error}")
        return 2
    if errors:
        print("Consistency diagnostics (allowed incomplete):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
