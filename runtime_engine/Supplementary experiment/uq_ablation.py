"""UQ calibration ablations from saved posterior predictions."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from common import (
    CASES,
    TABLE_DIR,
    apply_mask,
    calibration_eval_mask,
    collect_runs,
    ensure_output_dirs,
    error_std_corr,
    coverage95,
    avg_interval_width,
    load_npz,
    nll_gaussian,
    preferred_raw_std,
    scalar_float,
    write_csv,
)


def fit_temperature(mean, std, exact, mask=None, target: float = 0.95) -> float:
    m = apply_mask(mean, mask)
    s = np.maximum(apply_mask(std, mask), 1e-12)
    e = apply_mask(exact, mask)
    ratios = np.abs(e - m) / (1.96 * s)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return 1.0
    return float(max(np.quantile(ratios, target), 1e-8))


def score_variant(run_id: str, seed: int | None, case: str, variant: str,
                  mean, std, exact, eval_mask) -> dict:
    m = apply_mask(mean, eval_mask)
    s = np.maximum(apply_mask(std, eval_mask), 1e-12)
    e = apply_mask(exact, eval_mask)
    return {
        "run_id": run_id,
        "seed": seed,
        "case": case,
        "variant": variant,
        "coverage95": coverage95(m, s, e),
        "corr": error_std_corr(m, s, e),
        "nll": nll_gaussian(m, s, e),
        "avg_interval_width": avg_interval_width(s),
        "eval_points": int(np.asarray(e).size),
    }


def build_rows(seed0_root: str | None, cases: list[str]) -> list[dict]:
    rows: list[dict] = []
    for record in collect_runs(seed0_root):
        for case in cases:
            data = load_npz(record, case)
            if data is None or "exact" not in data or "bayesian_mean" not in data:
                continue

            exact = data["exact"]
            mean = data["bayesian_mean"]
            raw_std, raw_key = preferred_raw_std(data, case)
            if raw_std is None:
                continue

            eval_mask = calibration_eval_mask(data)
            fit_mask = None
            if "std_calibration_mask" in data:
                fit_mask = np.asarray(data["std_calibration_mask"]).astype(bool).reshape(-1)

            rows.append(score_variant(record.name, record.seed, case,
                                      f"raw_mc_std_only:{raw_key}",
                                      mean, raw_std, exact, eval_mask))

            temp = fit_temperature(mean, raw_std, exact, fit_mask)
            rows.append(score_variant(record.name, record.seed, case,
                                      "raw_mc_std_plus_temperature",
                                      mean, raw_std * temp, exact, eval_mask))

            if ("bayesian_dense_mean" in data
                    and "bayesian_structured_mean" in data):
                gap = np.abs(data["bayesian_dense_mean"]
                             - data["bayesian_structured_mean"])
                gap_floor = max(float(np.mean(gap)) * 1e-3, 1e-12)
                gap_std = np.maximum(gap, gap_floor)
                gap_temp = fit_temperature(mean, gap_std, exact, fit_mask)
                rows.append(score_variant(record.name, record.seed, case,
                                          "disagreement_only_temperature",
                                          mean, gap_std * gap_temp, exact,
                                          eval_mask))

                mc_gap = np.sqrt(np.maximum(raw_std, 1e-12) ** 2
                                 + gap_std ** 2)
                mc_gap_temp = fit_temperature(mean, mc_gap, exact, fit_mask)
                rows.append(score_variant(record.name, record.seed, case,
                                          "raw_mc_plus_unsharpened_disagreement",
                                          mean, mc_gap * mc_gap_temp, exact,
                                          eval_mask))

            if "bayesian_std" in data:
                row = score_variant(record.name, record.seed, case,
                                    "final_3.34_calibrated_std",
                                    mean, data["bayesian_std"], exact, eval_mask)
                row["stored_temperature_factor"] = scalar_float(
                    data, "std_temperature_factor", math.nan)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="Poisson",
                        help="Case name or all. Default: Poisson.")
    parser.add_argument("--seed0-root", default=None)
    args = parser.parse_args()

    ensure_output_dirs()
    cases = list(CASES) if args.case == "all" else [args.case]
    rows = build_rows(args.seed0_root, cases)
    if not rows:
        raise SystemExit("No UQ data found.")

    df = pd.DataFrame(rows).sort_values(["case", "seed", "variant"])
    suffix = "all" if args.case == "all" else args.case
    write_csv(df, TABLE_DIR / f"uq_ablation_{suffix}_raw.csv")

    summary = df.groupby(["case", "variant"], dropna=False).agg(
        n=("coverage95", "size"),
        coverage95_mean=("coverage95", "mean"),
        coverage95_std=("coverage95", "std"),
        corr_mean=("corr", "mean"),
        corr_std=("corr", "std"),
        nll_mean=("nll", "mean"),
        nll_std=("nll", "std"),
        avg_interval_width_mean=("avg_interval_width", "mean"),
        avg_interval_width_std=("avg_interval_width", "std"),
    ).reset_index()
    write_csv(summary.sort_values(["case", "variant"]),
              TABLE_DIR / f"uq_ablation_{suffix}_mean_std.csv")

    print("\nUQ ablation summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
