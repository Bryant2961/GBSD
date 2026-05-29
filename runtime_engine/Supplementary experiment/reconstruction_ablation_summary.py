"""Summarize structured reconstruction ablations against the frozen baseline."""
from __future__ import annotations

import argparse
import math

import pandas as pd

from common import (
    ABLATION_DIR,
    TABLE_DIR,
    RunRecord,
    ensure_output_dirs,
    evaluate_prediction,
    load_npz,
    read_parameter_error,
    scalar_float,
    scalar_string,
    seed0_record,
    write_csv,
)


CASES = ("Poisson", "Burgers_inv")
VARIANTS = ("no_dense_anchor", "no_pde_reconstruction")


def ablation_records() -> list[tuple[str, str, RunRecord]]:
    records: list[tuple[str, str, RunRecord]] = []
    if not ABLATION_DIR.is_dir():
        return records
    for path in sorted(p for p in ABLATION_DIR.iterdir() if p.is_dir()):
        name = path.name
        case = next((c for c in CASES if c in name), None)
        variant = next((v for v in VARIANTS if name.startswith(v)), None)
        if case is None or variant is None:
            continue
        records.append((
            variant,
            case,
            RunRecord(
                name=name,
                seed=0,
                root=path,
                raw_dir=path / "raw",
                metrics_dir=path / "metrics",
                case_root_pattern="{case}_EXP",
            ),
        ))
    return records


def summarize_record(variant: str, case: str, record: RunRecord) -> dict | None:
    data = load_npz(record, case)
    if data is None or "exact" not in data:
        return None
    exact = data["exact"]
    row = {
        "case": case,
        "variant": variant,
        "run_id": record.name,
        "final_source": scalar_string(data, "bayesian_source", "unknown"),
        "compression": scalar_float(data, "bayesian_structured_compression", math.nan),
        "nu_relative_error": read_parameter_error(record, case),
        "source_path": str(record.raw_dir / f"{case}_predictions.npz"),
    }
    if "bayesian_dense_mean" in data:
        row["dense_rL2"] = evaluate_prediction(
            data["bayesian_dense_mean"],
            data["bayesian_dense_std"] if "bayesian_dense_std" in data else None,
            exact,
        )["rel_l2"]
    if "bayesian_structured_mean" in data:
        structured = evaluate_prediction(
            data["bayesian_structured_mean"],
            data["bayesian_structured_std"] if "bayesian_structured_std" in data else None,
            exact,
        )
        for key, value in structured.items():
            row[f"structured_{key}"] = value
    if "bayesian_mean" in data:
        final = evaluate_prediction(
            data["bayesian_mean"],
            data["bayesian_std"] if "bayesian_std" in data else None,
            exact,
        )
        for key, value in final.items():
            row[f"final_{key}"] = value
    dense = row.get("dense_rL2", math.nan)
    structured = row.get("structured_rel_l2", math.nan)
    if not math.isnan(dense) and dense > 0 and not math.isnan(structured):
        row["structured_to_dense_ratio"] = structured / dense
    else:
        row["structured_to_dense_ratio"] = math.nan
    return row


def build_rows(seed0_root: str | None) -> list[dict]:
    rows: list[dict] = []
    baseline = seed0_record(seed0_root)
    if baseline is not None:
        for case in CASES:
            row = summarize_record("full_reconstruction_baseline", case, baseline)
            if row is not None:
                rows.append(row)
    for variant, case, record in ablation_records():
        row = summarize_record(variant, case, record)
        if row is not None:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-root", default=None)
    args = parser.parse_args()

    ensure_output_dirs()
    rows = build_rows(args.seed0_root)
    if not rows:
        raise SystemExit("No reconstruction ablation outputs found.")
    df = pd.DataFrame(rows).sort_values(["case", "variant"])
    out = TABLE_DIR / "reconstruction_ablation_summary.csv"
    write_csv(df, out)

    display_cols = [
        "case", "variant", "final_source", "dense_rL2", "structured_rel_l2",
        "final_rel_l2", "structured_to_dense_ratio", "compression",
        "final_coverage95", "final_corr", "nu_relative_error",
    ]
    print("\nReconstruction ablation summary:")
    print(df[[c for c in display_cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
