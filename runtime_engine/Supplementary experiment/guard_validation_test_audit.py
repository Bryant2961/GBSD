"""Audit final-source guard decisions with separated validation/test grids.

This is a post-processing audit for reviewer concerns about using reference
errors in source selection. It does not retrain models. For each archived run,
it splits the saved evaluation grid into a guard-validation subset and a blind
test subset, applies the guard only on the validation subset, then reports the
selected source on the held-out test subset.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from common import (
    CASES,
    TABLE_DIR,
    cfg_float,
    collect_runs,
    ensure_output_dirs,
    load_npz,
    read_config,
    rel_l2,
    scalar_float,
    write_csv,
)


def split_masks(n: int, seed: int | None, case: str, guard_fraction: float):
    case_offset = {"Laplace": 101, "Poisson": 211, "Burgers_inv": 307}[case]
    rng = np.random.default_rng(case_offset + 1009 * int(seed or 0))
    idx = rng.permutation(n)
    n_guard = max(10, int(round(n * guard_fraction)))
    guard_idx = idx[:n_guard]
    test_idx = idx[n_guard:]
    if len(test_idx) < 10:
        raise ValueError("test split too small")
    guard = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    guard[guard_idx] = True
    test[test_idx] = True
    return guard, test


def masked_rel_l2(pred, exact, mask) -> float:
    return rel_l2(np.asarray(pred).reshape(-1)[mask],
                  np.asarray(exact).reshape(-1)[mask])


def build_rows(seed0_root: str | None, guard_fraction: float):
    rows = []
    for record in collect_runs(seed0_root):
        for case in CASES:
            data = load_npz(record, case)
            if data is None or "exact" not in data:
                continue
            if "bayesian_dense_mean" not in data or "bayesian_structured_mean" not in data:
                continue

            exact = np.asarray(data["exact"]).reshape(-1)
            dense = np.asarray(data["bayesian_dense_mean"]).reshape(-1)
            structured = np.asarray(data["bayesian_structured_mean"]).reshape(-1)
            guard_mask, test_mask = split_masks(len(exact), record.seed, case, guard_fraction)

            cfg = read_config(case)
            gamma = cfg_float(cfg, "accept_structured_rel_l2_ratio", 1.10)
            eps = cfg_float(cfg, "accept_structured_rel_l2_abs", 0.0)
            cmin = cfg_float(cfg, "min_structured_compression", 0.0)
            compression = scalar_float(data, "bayesian_structured_compression", math.nan)

            dense_guard = masked_rel_l2(dense, exact, guard_mask)
            struct_guard = masked_rel_l2(structured, exact, guard_mask)
            dense_test = masked_rel_l2(dense, exact, test_mask)
            struct_test = masked_rel_l2(structured, exact, test_mask)

            accuracy_ok = struct_guard <= gamma * dense_guard + eps
            compression_ok = (cmin <= 0.0) or (compression >= cmin)
            accept = bool(accuracy_ok and compression_ok)
            selected = structured if accept else dense
            selected_source = "structured" if accept else "dense_student"

            rows.append({
                "run_id": record.name,
                "seed": record.seed,
                "case": case,
                "guard_fraction": guard_fraction,
                "n_guard": int(guard_mask.sum()),
                "n_test": int(test_mask.sum()),
                "guard_dense_rL2": dense_guard,
                "guard_structured_rL2": struct_guard,
                "test_dense_rL2": dense_test,
                "test_structured_rL2": struct_test,
                "test_selected_rL2": masked_rel_l2(selected, exact, test_mask),
                "selected_source_by_guard_split": selected_source,
                "stored_final_source": str(np.asarray(data.get("bayesian_source", [""])).reshape(-1)[0]),
                "structured_to_dense_ratio_guard": struct_guard / max(dense_guard, 1e-15),
                "structured_to_dense_ratio_test": struct_test / max(dense_test, 1e-15),
                "compression": compression,
                "guard_gamma": gamma,
                "guard_epsilon": eps,
                "guard_cmin": cmin,
                "accuracy_ok_on_guard": bool(accuracy_ok),
                "compression_ok": bool(compression_ok),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-root", default=None)
    parser.add_argument("--guard-fraction", type=float, default=0.30)
    args = parser.parse_args()

    ensure_output_dirs()
    rows = build_rows(args.seed0_root, args.guard_fraction)
    if not rows:
        raise SystemExit("No dense/structured prediction pairs found.")

    df = pd.DataFrame(rows).sort_values(["case", "seed", "run_id"])
    write_csv(df, TABLE_DIR / "guard_validation_test_audit_raw.csv")

    summary = df.groupby(["case", "selected_source_by_guard_split"], dropna=False).agg(
        n=("test_selected_rL2", "size"),
        test_selected_rL2_mean=("test_selected_rL2", "mean"),
        test_selected_rL2_std=("test_selected_rL2", "std"),
        test_dense_rL2_mean=("test_dense_rL2", "mean"),
        test_structured_rL2_mean=("test_structured_rL2", "mean"),
        compression_mean=("compression", "mean"),
    ).reset_index()
    write_csv(summary, TABLE_DIR / "guard_validation_test_audit_mean_std.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

