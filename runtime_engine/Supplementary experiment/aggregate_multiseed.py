"""Aggregate guarded Bayesian multi-seed stability metrics."""
from __future__ import annotations

import argparse
from datetime import datetime
import math

import pandas as pd

from common import (
    CASES,
    TABLE_DIR,
    collect_runs,
    ensure_output_dirs,
    evaluate_prediction,
    load_npz,
    read_parameter_error,
    scalar_float,
    scalar_string,
    summarize_mean_std,
    write_csv,
)


def build_rows(seed0_root: str | None) -> list[dict]:
    rows: list[dict] = []
    for record in collect_runs(seed0_root):
        for case in CASES:
            data = load_npz(record, case)
            if data is None or "exact" not in data:
                continue
            exact = data["exact"]
            row = {
                "run_id": record.name,
                "seed": record.seed,
                "case": case,
                "source_path": str(record.raw_dir / f"{case}_predictions.npz"),
                "final_source": scalar_string(data, "bayesian_source", "unknown"),
                "std_source": scalar_string(data, "bayesian_std_source", ""),
                "compression": scalar_float(
                    data, "bayesian_structured_compression", math.nan),
                "nu_relative_error": read_parameter_error(record, case),
            }

            if "bayesian_dense_mean" in data:
                dense = evaluate_prediction(
                    data["bayesian_dense_mean"],
                    data["bayesian_dense_std"] if "bayesian_dense_std" in data else None,
                    exact,
                )
                row["dense_rL2"] = dense["rel_l2"]
            if "bayesian_structured_mean" in data:
                structured = evaluate_prediction(
                    data["bayesian_structured_mean"],
                    (data["bayesian_structured_std"]
                     if "bayesian_structured_std" in data else None),
                    exact,
                )
                row["structured_rL2"] = structured["rel_l2"]
            if "bayesian_mean" in data:
                final = evaluate_prediction(
                    data["bayesian_mean"],
                    data["bayesian_std"] if "bayesian_std" in data else None,
                    exact,
                )
                for key, value in final.items():
                    row[f"final_{key}"] = value
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-root", default=None,
                        help="Optional v3.34 root containing seed=0 results.")
    args = parser.parse_args()

    ensure_output_dirs()
    rows = build_rows(args.seed0_root)
    if not rows:
        raise SystemExit("No archived runs found. Run seed sweeps first.")

    df = pd.DataFrame(rows).sort_values(["case", "seed", "run_id"])
    df["generated_at"] = datetime.now().isoformat(timespec="seconds")
    write_csv(df, TABLE_DIR / "multiseed_guarded_raw.csv")

    value_cols = [
        "dense_rL2",
        "structured_rL2",
        "final_rel_l2",
        "final_coverage95",
        "final_corr",
        "final_nll",
        "final_avg_interval_width",
        "compression",
        "nu_relative_error",
    ]
    summary = summarize_mean_std(df, ["case", "final_source"], value_cols)
    write_csv(summary.sort_values(["case", "final_source"]),
              TABLE_DIR / "multiseed_guarded_mean_std.csv")

    print("\nMulti-seed summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
