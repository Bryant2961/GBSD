"""Collect official per-seed GBSD outputs into summary tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gbsd.reporting.schema import REQUIRED_METRIC_FIELDS, SUMMARY_FILES


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_seed_records(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        record = _load_json(metrics_path)
        for optional_name, prefix in [
            ("timing.json", "timing"),
            ("model_size.json", "model_size"),
            ("guard_decision.json", "guard"),
            ("run_manifest.json", "manifest"),
        ]:
            path = run_dir / optional_name
            if path.is_file():
                for key, value in _load_json(path).items():
                    record.setdefault(key, value)
                    record.setdefault(f"{prefix}_{key}", value)
        # Keep summary artifacts portable: this path is relative to the input
        # results tree (official protocol or isolated smoke tree).
        record.setdefault("source_run_dir", run_dir.relative_to(root).as_posix())
        records.append(record)
    if not records:
        return pd.DataFrame(columns=REQUIRED_METRIC_FIELDS)
    return pd.DataFrame(records)


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    return [field for field in REQUIRED_METRIC_FIELDS if field not in df.columns]


def write_summaries(raw: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw.empty:
        raw = pd.DataFrame(columns=REQUIRED_METRIC_FIELDS)
    raw.to_csv(output_dir / SUMMARY_FILES["all_raw"], index=False)
    if raw.empty:
        empty = pd.DataFrame(columns=REQUIRED_METRIC_FIELDS)
        for filename in SUMMARY_FILES.values():
            empty.to_csv(output_dir / filename, index=False)
        return

    numeric_cols = raw.select_dtypes(include="number").columns.tolist()
    group_cols = [c for c in ["experiment", "problem", "variant"] if c in raw.columns]
    if not group_cols:
        raw.to_csv(output_dir / SUMMARY_FILES["all_mean_std"], index=False)
    else:
        summary = raw.groupby(group_cols)[numeric_cols].agg(["mean", "std", "count"])
        summary.columns = ["_".join(col).strip("_") for col in summary.columns]
        summary.reset_index().to_csv(output_dir / SUMMARY_FILES["all_mean_std"], index=False)

    if "experiment" in raw.columns:
        mapping = {
            "main_blind": "main_blind",
            "strong_baselines": "baselines",
            "guard_ablation": "guard_ablation",
            "uq_calibration_ablation": "uq_ablation",
            "reconstruction_ablation": "reconstruction_ablation",
            "zero_distillation_ablation": "zero_distillation_ablation",
            "runtime_params": "runtime_params",
            "threshold_sensitivity": "threshold_sensitivity",
            "structure_stability": "structure_stability",
        }
        for experiment, key in mapping.items():
            subset = raw[raw["experiment"] == experiment]
            subset.to_csv(output_dir / SUMMARY_FILES[key], index=False)
