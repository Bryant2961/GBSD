"""Final source guard ablation from saved dense/structured predictions."""
from __future__ import annotations

import argparse
import math

import pandas as pd

from common import (
    CASES,
    TABLE_DIR,
    cfg_float,
    collect_runs,
    ensure_output_dirs,
    evaluate_prediction,
    load_npz,
    read_config,
    scalar_float,
    summarize_mean_std,
    write_csv,
)


VARIANTS = (
    "always_dense",
    "always_structured",
    "accuracy_only_guard",
    "compression_only_guard",
    "full_guard",
)


def choose_source(variant: str, dense_r: float, struct_r: float,
                  compression: float, cfg: dict[str, str]) -> str:
    ratio = cfg_float(cfg, "accept_structured_rel_l2_ratio", 1.10)
    eps = cfg_float(cfg, "accept_structured_rel_l2_abs", 0.002)
    cmin = cfg_float(cfg, "min_structured_compression", 0.0)
    accuracy_ok = struct_r <= dense_r * ratio + eps
    compression_ok = cmin <= 0 or (math.isfinite(compression)
                                   and compression >= cmin)

    if variant == "always_dense":
        return "dense"
    if variant == "always_structured":
        return "structured"
    if variant == "accuracy_only_guard":
        return "structured" if accuracy_ok else "dense"
    if variant == "compression_only_guard":
        return "structured" if compression_ok else "dense"
    if variant == "full_guard":
        return "structured" if (accuracy_ok and compression_ok) else "dense"
    raise ValueError(f"Unknown guard variant: {variant}")


def build_rows(seed0_root: str | None) -> list[dict]:
    rows: list[dict] = []
    for record in collect_runs(seed0_root):
        for case in CASES:
            data = load_npz(record, case)
            if data is None or "exact" not in data:
                continue
            if "bayesian_dense_mean" not in data or "bayesian_structured_mean" not in data:
                continue

            cfg = read_config(case)
            exact = data["exact"]
            dense_r = evaluate_prediction(
                data["bayesian_dense_mean"], None, exact)["rel_l2"]
            struct_r = evaluate_prediction(
                data["bayesian_structured_mean"], None, exact)["rel_l2"]
            compression = scalar_float(data, "bayesian_structured_compression",
                                       math.nan)
            ratio = cfg_float(cfg, "accept_structured_rel_l2_ratio", 1.10)
            eps = cfg_float(cfg, "accept_structured_rel_l2_abs", 0.002)
            cmin = cfg_float(cfg, "min_structured_compression", 0.0)

            for variant in VARIANTS:
                source = choose_source(variant, dense_r, struct_r, compression, cfg)
                selected_r = struct_r if source == "structured" else dense_r
                rows.append({
                    "run_id": record.name,
                    "seed": record.seed,
                    "case": case,
                    "variant": variant,
                    "selected_source": source,
                    "selected_rL2": selected_r,
                    "dense_rL2": dense_r,
                    "structured_rL2": struct_r,
                    "structured_to_dense_ratio": struct_r / max(dense_r, 1e-15),
                    "compression": compression,
                    "accept_ratio": ratio,
                    "accept_abs": eps,
                    "min_compression": cmin,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-root", default=None)
    args = parser.parse_args()

    ensure_output_dirs()
    rows = build_rows(args.seed0_root)
    if not rows:
        raise SystemExit("No dense/structured prediction pairs found.")

    df = pd.DataFrame(rows).sort_values(["case", "seed", "variant"])
    write_csv(df, TABLE_DIR / "guard_ablation_raw.csv")

    summary = summarize_mean_std(
        df,
        ["case", "variant", "selected_source"],
        ["selected_rL2", "dense_rL2", "structured_rL2",
         "structured_to_dense_ratio", "compression"],
    )
    write_csv(summary.sort_values(["case", "variant", "selected_source"]),
              TABLE_DIR / "guard_ablation_mean_std.csv")

    print("\nGuard ablation summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
