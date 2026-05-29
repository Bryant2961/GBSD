"""Summarize clustering-threshold sweep outputs."""
from __future__ import annotations

import argparse
import math
import re

import pandas as pd

from common import (
    ABLATION_DIR,
    TABLE_DIR,
    RunRecord,
    cfg_float,
    ensure_output_dirs,
    evaluate_prediction,
    load_npz,
    read_config,
    read_parameter_error,
    scalar_float,
    scalar_string,
    seed0_record,
    write_csv,
)


CASES = ("Poisson", "Burgers_inv")


def threshold_from_name(name: str) -> float:
    match = re.search(r"_0p(\d+)", name)
    if not match:
        return math.nan
    return float("0." + match.group(1))


def cluster_records() -> list[tuple[float, str, RunRecord]]:
    records: list[tuple[float, str, RunRecord]] = []
    if not ABLATION_DIR.is_dir():
        return records
    for path in sorted(p for p in ABLATION_DIR.iterdir() if p.is_dir()):
        name = path.name
        if not name.startswith("cluster_"):
            continue
        case = next((c for c in CASES if c in name), None)
        if case is None:
            continue
        records.append((
            threshold_from_name(name),
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


def summarize_record(kind: str, threshold: float, case: str,
                     record: RunRecord) -> dict | None:
    data = load_npz(record, case)
    if data is None or "exact" not in data:
        return None
    exact = data["exact"]
    cfg = read_config(case)
    gamma = cfg_float(cfg, "accept_structured_rel_l2_ratio", 1.10)
    eps = cfg_float(cfg, "accept_structured_rel_l2_abs", 0.0)
    cmin = cfg_float(cfg, "min_structured_compression", 0.0)

    row = {
        "case": case,
        "kind": kind,
        "cluster_distance": threshold,
        "run_id": record.name,
        "final_source": scalar_string(data, "bayesian_source", "unknown"),
        "compression": scalar_float(data, "bayesian_structured_compression", math.nan),
        "nu_relative_error": read_parameter_error(record, case),
        "guard_gamma": gamma,
        "guard_epsilon": eps,
        "guard_cmin": cmin,
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
    compression = row.get("compression", math.nan)
    row["structured_to_dense_ratio"] = (
        structured / dense
        if not math.isnan(dense) and dense > 0 and not math.isnan(structured)
        else math.nan
    )
    row["accuracy_ok"] = (
        structured <= gamma * dense + eps
        if not math.isnan(dense) and not math.isnan(structured)
        else False
    )
    row["compression_ok"] = (
        compression >= cmin
        if not math.isnan(compression)
        else False
    )
    row["guard_accept"] = bool(row["accuracy_ok"] and row["compression_ok"])
    return row


def build_rows(seed0_root: str | None) -> list[dict]:
    rows: list[dict] = []
    baseline = seed0_record(seed0_root)
    if baseline is not None:
        for case in CASES:
            row = summarize_record("baseline_v3.34", math.nan, case, baseline)
            if row is not None:
                rows.append(row)
    for threshold, case, record in cluster_records():
        row = summarize_record("cluster_sweep", threshold, case, record)
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
        raise SystemExit("No clustering sweep outputs found.")
    df = pd.DataFrame(rows).sort_values(["case", "kind", "cluster_distance"],
                                        na_position="first")
    out = TABLE_DIR / "clustering_sweep_summary.csv"
    write_csv(df, out)

    display_cols = [
        "case", "kind", "cluster_distance", "final_source", "dense_rL2",
        "structured_rel_l2", "structured_to_dense_ratio", "compression",
        "guard_accept", "final_coverage95", "final_corr", "nu_relative_error",
    ]
    print("\nClustering sweep summary:")
    print(df[[c for c in display_cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
